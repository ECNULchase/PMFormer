from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn.functional as F
import torch_scatter

import spconv.pytorch as spconv
import torch
import torch.nn as nn
from ..builder import DETECTORS
from .base import Base3DDetector
from mmcv.ops import Voxelization
from ..builder import ALGORITHMS, build_backbone
from .voxel_utils import quantitize, VFELayerMinusPP, PcPreprocessor3DSlim, gather_feature, VFELayerMinus

from mmcv.runner import BaseModule, force_fp32
import torch.nn.functional as F
from collections import OrderedDict
import torch.distributed as dist
import mmselfsup.utils.lovasz_loss as Lovasz_loss


import copy
from torch.nn.modules.batchnorm import _BatchNorm
import math
from torch import distributed as dist
from torch.autograd.function import Function

#for maskformer
from .mask2former.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from .mask2former.matcher import HungarianMatcher
from .mask2former.criterion import SetCriterion
from .mask2former.post_process import semantic_inference
from .mask2former.basic_block import ResNetFCN

class AllReduce(Function):

    @staticmethod
    def forward(ctx, input):
        input_list = [
            torch.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce in-place operations is unreliable
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class NaiveSparseSyncBatchNorm1d(nn.BatchNorm1d):
    """Syncronized Batch Normalization for 3D Tensors.
    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/
        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        In 3D detection, different workers has points of different shapes,
        whish also cause instability.
        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    def forward(self, input):
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not self.training or dist.get_world_size() == 1:
            return super().forward(input)
        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0])
        meansqr = torch.mean(input * input, dim=[0])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1)
        bias = bias.reshape(1, -1)
        return input * scale + bias


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, output, proj_labels_copy, ignore_index=255):
        return F.cross_entropy(output, proj_labels_copy, ignore_index=ignore_index)


def voxel_sem_target(point_voxel_coors, sem_label):
    """make sparse voxel tensor of semantic labels
    Args:
        point_voxel_coors(N, bxyz): point-wise voxel coors
        sem_label(N, ): point-wise semantic label
    Return:
        unq_sem(M, ): voxel-wise semantic label
        unq_voxel(M, bxyz): voxel-wise voxel coors
    """
    voxel_sem = torch.cat([point_voxel_coors, sem_label.reshape(-1, 1)], dim=-1)
    unq_voxel_sem, unq_sem_count = torch.unique(voxel_sem, return_counts=True, dim=0)
    unq_voxel, unq_ind = torch.unique(unq_voxel_sem[:, :4], return_inverse=True, dim=0)
    label_max_ind = torch_scatter.scatter_max(unq_sem_count, unq_ind)[1]
    unq_sem = unq_voxel_sem[:, -1][label_max_ind]
    return unq_sem, unq_voxel


class BasicBlock(spconv.SparseModule):
    def __init__(self, C_in, C_out, indice_key):
        super(BasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 1, indice_key=indice_key, bias=False),
            NaiveSparseSyncBatchNorm1d(C_out, ),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 3, indice_key=indice_key, bias=False),
            NaiveSparseSyncBatchNorm1d(C_out, ),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(C_out, C_out, 3, indice_key=indice_key, bias=False),
            NaiveSparseSyncBatchNorm1d(C_out, ),
            # nn.LeakyReLU(0.1)
        )
        self.relu2 = spconv.SparseSequential(
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        identity = self.layers_in(x)
        out = self.layers(x)
        output = spconv.SparseConvTensor(sum([i.features for i in [identity, out]]),
                                         out.indices, out.spatial_shape, out.batch_size)
        output.indice_dict = out.indice_dict
        output.grid = out.grid
        return self.relu2(output)


def make_layers_sp(C_in, C_out, blocks, indice_key):
    layers = []
    layers.append(BasicBlock(C_in, C_out, indice_key))
    for _ in range(1, blocks):
        layers.append(BasicBlock(C_out, C_out, indice_key))
    return spconv.SparseSequential(*layers)


def scatter(x, idx, method, dim=0):
    if method == "max":
        return torch_scatter.scatter_max(x, idx, dim=dim)[0]
    elif method == "mean":
        return torch_scatter.scatter_mean(x, idx, dim=dim)
    elif method == "sum":
        return torch_scatter.scatter_add(x, idx, dim=dim)
    else:
        print("unknown method")
        exit(-1)


def gather(x, idx):
    """
    :param x: voxelwise features
    :param idx:
    :return: pointwise features
    """
    return x[idx]


class SFE(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, layer_name, layer_num=2):
        super(SFE, self).__init__()
        self.spconv_layers = make_layers_sp(in_channels, out_channels, layer_num, layer_name)

    def forward(self, inputs):
        conv_features = self.spconv_layers(inputs)
        return conv_features


class VFELayerMinusSlim(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize=None,
                 last_layer=False,
                 attention=False,
                 name=''):

        super(VFELayerMinusSlim, self).__init__()
        self.name = 'VFELayerMinusSlim' + name
        self.last_vfe = last_layer
        self.normalize = normalize
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        # input batch pointnum feature_num
        self.linear = nn.Linear(in_channels, self.units, bias=True)
        self.weight_linear = nn.Linear(6, self.units, bias=True)

        if self.normalize:
            self.norm = nn.BatchNorm1d(self.units)  # , **self.normalize

    def forward(self, inputs, idx_used, sizes, mean=None, activate=False, gs=None):
        x = self.linear(inputs)
        if activate:
            x = F.relu(x)
        if gs is not None:
            x = x * gs
        if mean is not None:
            x_weight = self.weight_linear(mean)
            if activate:
                x_weight = F.relu(x_weight)
            x = x * x_weight
        index, value = torch.unique(idx_used, return_inverse=True, dim=0)
        max_feature, fk = torch_scatter.scatter_max(x, value, dim=0)
        gather_max_feature = max_feature[value, :]
        x_concated = torch.cat((x, gather_max_feature), dim=1)
        # return x_concated, max_feature
        return x_concated


class SGFE(nn.Module):
    def __init__(self, input_channels, output_channels, reduce_channels, name, p_scale=[2, 4, 6, 8]):
        super(SGFE, self).__init__()
        self.inplanes = input_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name

        self.feature_reduce = nn.Linear(input_channels, reduce_channels)
        self.pooling_scale = p_scale
        self.fc_list = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for i, scale in enumerate(self.pooling_scale):
            self.fc_list.append(nn.Sequential(
            nn.Linear(reduce_channels, reduce_channels//2),
            nn.ReLU(),
            ))
            self.fcs.append(nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2)))
        self.scale_selection = nn.Sequential(
            nn.Linear(len(self.pooling_scale) * reduce_channels//2,
                                       reduce_channels),nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2, bias=False),
                                nn.ReLU(inplace=False))
        self.out_fc = nn.Linear(reduce_channels//2, reduce_channels, bias=False)
        self.linear_output = nn.Sequential(
            nn.Linear(2 * reduce_channels, reduce_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduce_channels, output_channels),
        )
        

    def forward(self, coords_info, top_mean_ms,
                input_data, output_scale, method="max", 
                with_fm=True, input_coords=None, input_coords_inv=None):

        topoutput_feature_ms = []
        output_feature_pw = []
        reduced_feature = F.relu(self.feature_reduce(input_data))
        # output = fusion_list
        output_list = [reduced_feature]
        for j, ps in enumerate(self.pooling_scale):
            # index = torch.cat([coords_info[ps]['bxyz_indx'][:, 0].unsqueeze(-1),
            #                    torch.flip(coords_info[ps]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
            index = torch.cat([input_coords[:, 0].unsqueeze(-1),
                              (input_coords[:, 1:] // ps).int()], dim=1)
            unq, unq_inv = torch.unique(index, return_inverse=True, dim=0)
            # unq = unq.type(torch.int64)
            fkm = scatter(reduced_feature, unq_inv, method="mean", dim=0)# + torch_scatter.scatter_max(reduced_feature, unq_inv, dim=0)[0]
            att = self.fc_list[j](fkm)[unq_inv]
            out = ( att)
            output_list.append(out)
        scale_features = torch.stack(output_list[1:], dim=1)#.view(-1, len(self.pooling_scale), 64)
        feat_S = scale_features.sum(1)
        feat_Z = self.fc(feat_S)
        attention_vectors = [fc(feat_Z) for fc in self.fcs]
        attention_vectors = torch.sigmoid(torch.stack(attention_vectors, dim=1))
        scale_features = self.out_fc(torch.sum(scale_features * attention_vectors, dim=1))

        output_f = torch.cat([reduced_feature, scale_features], dim=1)
        proj = self.linear_output(output_f)
        proj = proj[input_coords_inv]
        if with_fm:
            index = torch.cat([coords_info[output_scale]['bxyz_indx'][:, 0].unsqueeze(-1),
                               torch.flip(coords_info[output_scale]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
            
            unq, unq_inv = torch.unique(index, return_inverse=True, dim=0)
            tv_fmap = scatter(proj, unq_inv, method="max", dim=0)
            return proj, tv_fmap, unq, unq_inv,index
        else:
            return proj, None, None, None


@DETECTORS.register_module()
class pmformer(nn.Module):
    def __init__(self, train_cfg, **kwargs):
        super(pmformer, self).__init__()
        params = train_cfg
        self.scales = params['scales']
        self.multi_frame = False
        self.ce_loss = CELoss()
        self.lovasz_loss = Lovasz_loss.Lovasz_loss(ignore=255)
        class_weight = params.class_weight
        dice_weight = params.dice_weight
        mask_weight = params.mask_weight
        matcher = HungarianMatcher(
            cost_class=params.match_class_weight, #2.0,
            cost_dice=params.match_dice_weight, #2.0,
            cost_mask=params.match_mask_weight,#5.0,
            num_points=112 * 112 ,
            )
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        dec_layers = 6
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        losses = ["labels", "masks"]
        self.criterion = SetCriterion(
                params.n_class,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=0.1,
                losses=losses,
                num_points= 50000,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
        )
        
        self.multi_scale_top_layers = nn.ModuleDict()
        self.feature_list = {
            "0.5": [10, 64],
            "1": [10, 64],
        }
        self.target_scale = 1
        for scale in self.scales:
            top_layer = VFELayerMinusSlim(self.feature_list[str(scale)][0],
                                          self.feature_list[str(scale)][1],
                                          "top_layer_" + str(scale))
            if scale == 0.5:
                rescale = int(0.5 * 10)
            else:
                rescale = scale
            self.multi_scale_top_layers[str(rescale)] = top_layer

        self.aggtopmeanproj = nn.Linear(6, 64, bias=True)
        self.aggtopproj = nn.Linear(128, 64, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, ),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.tv_agglayer = VFELayerMinus(64,
                                         128,
                                         "tvagg",
                                         weight_dims=8)

        self.conv1_block = SFE(64, 64, "svpfe_0")
        self.conv2_block = SFE(64, 64, "svpfe_1")
        self.conv3_block = SFE(64, 64, "svpfe_2")
        self.conv4_block = SFE(64, 64, "svpfe_3")

        self.proj1_block = SGFE(input_channels=64, output_channels=64,\
                                reduce_channels=64, name="proj1")
        
        self.proj2_block = SGFE(input_channels=64, output_channels=64,\
                                reduce_channels=64, name="proj2")
        self.proj3_block = SGFE(input_channels=64, output_channels=64,\
                                reduce_channels=64, name="proj3")
        self.proj4_block = SGFE(input_channels=64, output_channels=64,\
                                reduce_channels=64, name="proj4")

 
        self.out_linears = nn.Sequential(
            NaiveSparseSyncBatchNorm1d(64 + 64 + 64  + 64 + 64, ),
            nn.Linear(64 + 64 + 64 + 64 + 64, 128, bias=False),
            NaiveSparseSyncBatchNorm1d(128, ),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128, bias=False),
            NaiveSparseSyncBatchNorm1d(128, ),
            nn.LeakyReLU(0.1),
        )

        
        self.num_class = params['n_class']


        self.transformer_predictor = MultiScaleMaskedTransformerDecoder(
                in_channels=64,#64
                mask_classification=True,
                num_classes=self.num_class,
                hidden_dim=256, 
                num_queries=params['num_quries'], #50
                nheads=8,
                dim_feedforward=2048,
                dec_layers=params['dec_layers'],
                pre_norm=False,
                mask_dim=128,
                enforce_input_project=False,
            )
        
        self.reset_params(params)
    
    def init_weights(self, pretrained=None):
        # print(pretrained)
        # exit(0)
        pass
    def reset_params(self, params):
        self.x_lims = params['lims'][0]
        self.y_lims = params['lims'][1]
        self.z_lims = params['lims'][2]
        self.offset = params['offset']
        self.target_scale = params['target_scale']
  
        self.grid_meters = params['grid_meters']
        self.sizes = [int(round((self.x_lims[1] - self.x_lims[0]) / self.grid_meters[0])),
                      int(round((self.y_lims[1] - self.y_lims[0]) / self.grid_meters[1])),
                      (int(round((self.z_lims[1] - self.z_lims[0]) / self.grid_meters[2])))]
        self.lims = [self.x_lims, self.y_lims, self.z_lims]

        self.pooling_scale = params['pooling_scale']

        self.preprocess = PcPreprocessor3DSlim(self.lims, self.grid_meters, scales=self.pooling_scale)

    def add_pcmean_and_gridmeanv2(self, pc, idx, idx_used,
                                  xyz_indx, size_x, size_y, lims, m_pergrid, return_mean=False):

        index, value = torch.unique(idx_used, return_inverse=True, dim=0)
        pc_mean = torch_scatter.scatter_mean(pc[:, :3], value, dim=0)[value]

        pc_subtract_mean = pc[:, :3] - pc_mean
        m_pergird = torch.tensor([m_pergrid[0], m_pergrid[1], m_pergrid[2]], dtype=torch.float, device=pc.device)
        xmin_ymin_zmin = torch.tensor([lims[0], lims[1], lims[2]], dtype=torch.float, device=pc.device)

        pc_gridmean = (xyz_indx.type(torch.cuda.FloatTensor) + self.offset) * m_pergird + xmin_ymin_zmin
        grid_center_minus_mean = pc[:, :3] - pc_gridmean
        pc_feature = torch.cat((pc, pc_subtract_mean, grid_center_minus_mean), dim=1)  # same input
        mean = torch.cat((pc_subtract_mean, grid_center_minus_mean), dim=1)  # different input_mean
        # print(pc_feature.size(), mean.size())
        if return_mean:
            return pc_feature, mean
        else:
            return pc_feature

    def extract_geometry_feature(self, pc, out):

        multi_scales_feature = {}
        multi_scales_point_feature = {}
        topoutput_feature_ms = {}
        aggtopoutput_feature_ms = {}
        topoutput_feature_pwms = {}
        topoutput_mean_ms = {}
        for scale in self.scales:
            multi_scales_point_feature[str(scale)] = []
            multi_scales_feature[str(scale)] = []
            topoutput_feature_ms[str(scale)] = []
            aggtopoutput_feature_ms[str(scale)] = []
            topoutput_feature_pwms[str(scale)] = []
            topoutput_mean_ms[str(scale)] = []
        # first stage feature extractor

        for j, scale in enumerate(self.scales):
            size_x = int(round(self.sizes[0] / scale))
            size_y = int(round(self.sizes[1] / scale))
            size_z = int(round(self.sizes[2] / scale))

            idx_i = out[scale]['bxyz_indx']
            idx_l = idx_i.long()
            if scale == 0.5:
                rescale = int(0.5 * 10)
            else:
                rescale = scale
            pc_top, topview_mean = self.add_pcmean_and_gridmeanv2(pc, idx_l,
                                                                  idx_l,
                                                                  idx_l[:, 1:], size_x, size_y,
                                                                  [self.lims[0][0], self.lims[1][0], self.lims[2][0]],
                                                                  [self.grid_meters[0] * scale,
                                                                   self.grid_meters[1] * scale,
                                                                   self.grid_meters[2] * scale],
                                                                  return_mean=True)
            # print(torch.max(topview_mean), torch.min(topview_mean))
            topoutput_mean_ms[str(scale)] = topview_mean
            feat = self.multi_scale_top_layers[str(rescale)](pc_top, idx_l,
                                                             size_x * size_y,
                                                             mean=topview_mean)
            topoutput_feature_pwms[str(scale)] = feat

        # feature projection and aggregation
        aggfv_list = []

        tvms_feature = []
        for scale in self.scales:
            tvms_feature.append(topoutput_feature_pwms[str(scale)])
        tvms_feature = torch.cat(tvms_feature, dim=1)
        # size_x = int(self.sizes[0] // self.target_scale)
        # size_y = int(self.sizes[1] // self.target_scale)

        agg_tpfeature = F.relu(self.aggtopmeanproj(topoutput_mean_ms[str(self.target_scale)])) \
                        * F.relu(self.aggtopproj(tvms_feature))

        agg_fusionfeature = agg_tpfeature

        pidx_i = out[self.target_scale]['bxyz_indx']
        pidx_l = pidx_i.long()
        # pidx_in_used = pidx_i.view(-1, 1)
        index, value = torch.unique(pidx_l, return_inverse=True, dim=0)
        v = self.tv_agglayer.linear(agg_fusionfeature)
        maxf = torch_scatter.scatter_max(v, value, dim=0)[0]

        aggfv_list.append(self.mlp(tvms_feature))

        return maxf, topoutput_mean_ms, aggfv_list[0], index, value, pidx_l

    
    def forward_train(self, data, pw_label=None, grid_label=None):
        pc_tmp = data['points']
        batch_size = len(pc_tmp)
        if pw_label is not None:
            pw_label = torch.cat(pw_label, dim=0)
        with torch.no_grad():
            indicator = [0]
            pc_ibatch = []
            for i in range(batch_size):
                pc_i = pc_tmp[i]
                pc_ibatch.append(pc_i)
                indicator.append(pc_i.size(0) + indicator[-1])
            pc = torch.cat(pc_ibatch, dim=0)
            filter_pc, info = self.preprocess(pc, indicator)

        feature, topoutput_mean_ms, agg_fv1, coord_ind, full_coord, full_coord_index = self.extract_geometry_feature(filter_pc, info)
        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)

        input_tensor = spconv.SparseConvTensor(
            feature, coord.int(), np.int32(self.sizes)[::-1].tolist(), batch_size
        )
        conv1_output = self.conv1_block(input_tensor)
        proj1_pw, proj1_vw, vw1_coord, pw1_coord,pw1_coord_index  = \
            self.proj1_block(info, None, conv1_output.features, output_scale=2, input_coords=coord.int(),
            input_coords_inv=full_coord)

        conv2_input_tensor = spconv.SparseConvTensor(
            proj1_vw, vw1_coord.int(), (np.array(self.sizes, np.int32) // 2)[::-1], batch_size
        )
        conv2_output = self.conv2_block(conv2_input_tensor)
        proj2_pw, proj2_vw, vw2_coord, pw2_coord,pw2_coord_index = \
            self.proj2_block(info, None, conv2_output.features, output_scale=4, input_coords=vw1_coord.int(),
            input_coords_inv=pw1_coord)
        
        conv3_input_tensor = spconv.SparseConvTensor(
            proj2_vw, vw2_coord.int(), (np.array(self.sizes, np.int32) // 4)[::-1], batch_size
        )
        conv3_output = self.conv3_block(conv3_input_tensor)
        proj3_pw, proj3_vw, vw3_coord, pw3_coord,pw3_coord_index = \
            self.proj3_block(info, None,conv3_output.features, output_scale=4, input_coords=vw2_coord.int(),
            input_coords_inv=pw2_coord)

        conv4_input_tensor = spconv.SparseConvTensor(
            proj3_vw, vw3_coord.int(), (np.array(self.sizes, np.int32) // 4)[::-1], batch_size
        )
        conv4_output = self.conv4_block(conv4_input_tensor)
        proj4_pw, _, _, _ = self.proj4_block(info, None, conv4_output.features, output_scale=4, with_fm=False,
                                             input_coords=vw3_coord.int(),input_coords_inv=pw3_coord)


        
        pw_feature = torch.cat([proj1_pw, proj2_pw, proj3_pw, proj4_pw, agg_fv1], dim=1).contiguous()

        score = self.out_linears(pw_feature)

        batch_index = full_coord_index[:,0]
        scatter_index = torch.cat([pw3_coord.unsqueeze(0),pw1_coord.unsqueeze(0),full_coord.unsqueeze(0)],dim=0)    #conv4 conv2 conv1
        scatter_batch_index = torch.cat([pw3_coord_index[:,0].unsqueeze(0),pw1_coord_index[:,0].unsqueeze(0),full_coord_index[:,0].unsqueeze(0)],dim=0)
        for i in range(batch_size):
            batch_conv4_output = conv4_output.features[conv4_output.indices[:,0]==i]
            batch_score = score[batch_index==i]
            batch_scatter_index = scatter_index[scatter_batch_index==i].reshape(3,-1).contiguous()
            batch_conv4_output_pos = conv4_output.indices[conv4_output.indices[:,0]==i][:,1:]
            if i==0:
                outputs_class,outputs_mask,aux_outputs = self.transformer_predictor([batch_conv4_output],
                                                                        batch_score.unsqueeze(0).transpose(2,1).contiguous(),
                                                                        [batch_conv4_output_pos],batch_scatter_index)
            else:
                temp_outputs_class,temp_outputs_mask,temp_aux_outputs = self.transformer_predictor([batch_conv4_output],
                                                                        batch_score.unsqueeze(0).transpose(2,1).contiguous(),
                                                                        [batch_conv4_output_pos],batch_scatter_index)

                outputs_class = torch.cat((outputs_class,temp_outputs_class),dim=0)
                outputs_mask = torch.cat((outputs_mask,temp_outputs_mask),dim=1)
                for i in range(len(temp_aux_outputs)):
                    temp_layer_outputs_class = temp_aux_outputs[i]['pred_logits']
                    temp_layer_outputs_mask = temp_aux_outputs[i]['pred_masks']
                    aux_outputs[i]['pred_logits'] = torch.cat((aux_outputs[i]['pred_logits'],temp_layer_outputs_class),dim=0)
                    aux_outputs[i]['pred_masks'] = torch.cat((aux_outputs[i]['pred_masks'],temp_layer_outputs_mask),dim=1)
        out = {
            'pred_logits': outputs_class,
            'pred_masks': outputs_mask,
            'batch_index': batch_index,
            'aux_outputs': aux_outputs
        }

        return out

    def forward(self, return_loss=True, **data):
        if return_loss:
            points_label = data["points_label"]
            output_teacher = self.forward_train(data, pw_label=points_label)
            targets = [data['target_labels'],data['target_masks']]
            loss_dict = self.criterion(output_teacher, targets)
            for k in list(loss_dict.keys()):
                if k in self.criterion.weight_dict:
                    loss_dict[k] *= self.criterion.weight_dict[k]
            return loss_dict

        else:
            out = self.forward_train(data, get_ori=False)
            batch_index = out['batch_index']
            pred_labels = out['pred_logits']
            pred_masks = out['pred_masks']
            
            out_t = []
            for i in range(pred_labels.shape[0]):
                batch_pred_labels = pred_labels[i]
                batch_pred_masks = pred_masks[:,batch_index==i]
                out_t.append(semantic_inference(batch_pred_labels,batch_pred_masks))
            outputs = torch.cat(out_t, 0)
            return dict(x=outputs)
    def aug_test(self):
        pass
    
    def simple_test(self):
        pass

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.
        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.
                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['points']))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.
        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['points']))

        return outputs