# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

# from network.lovasz_losses import lovasz_softmax
# from mask2former.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

###########detectron2 #################
def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1).to(torch.int32)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # temp_loss = F.binary_cross_entropy(inputs, targets)
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

batch_sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule



def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1).cuda()
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        #loss for semantic
        self.CE_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets[0], indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.cuda())
        losses = {"loss_ce": loss_ce*self.weight_dict["loss_ce"]}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        batch_index = outputs['batch_index']
        tgt_masks = [t for t in targets[1]]
        for i in range(len(targets[0])):
            batch_src_masks = src_masks[:,batch_index==i]
            query_index = src_idx[1][src_idx[0]==i]
            batch_src_masks = batch_src_masks[query_index].flatten(0)
            
            tgt_index = tgt_idx[1][tgt_idx[0]==i]
            batch_tgt_masks = tgt_masks[i][tgt_index].float().flatten(0)
            
            # if batch_src_masks.shape[-1]<max_points:
            #     num_pad = max_points - batch_src_masks.shape[-1]
            #     src_const_pad = nn.ConstantPad1d(padding=(1,num_pad),value=-1e4)
            #     tgt_const_pad = nn.ConstantPad1d(padding=(1,num_pad),value= 0)
            #     batch_src_masks = src_const_pad(batch_src_masks)[:,:max_points]
            #     batch_tgt_masks = tgt_const_pad(batch_tgt_masks)[:,:max_points]
            if i==0:
                result_src_masks = batch_src_masks
                result_tgt_masks = batch_tgt_masks
            else:
                result_src_masks = torch.cat((result_src_masks,batch_src_masks),dim=0)
                result_tgt_masks = torch.cat((result_tgt_masks,batch_tgt_masks),dim=0)
        
        result_src_masks = result_src_masks.unsqueeze(0).float()
        result_tgt_masks = result_tgt_masks.unsqueeze(0)
        losses = {
            "loss_mask": batch_sigmoid_focal_loss_jit(result_src_masks, result_tgt_masks, num_masks),
            # "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(result_src_masks, result_tgt_masks, num_masks),
        }
        # for k in list(losses.keys()):
        #     if k in self.weight_dict:
        #         losses[k] *= self.weight_dict[k]

        del result_src_masks
        del result_tgt_masks
        return losses
        # masks = [t for t in targets[1]]
        # index = tgt_idx[1][tgt_idx[0]==0]
        # target_masks = masks[0][index]
        # for i in range(1,len(targets[1])):
        #     index = tgt_idx[1][tgt_idx[0]==i]
        #     target_masks = torch.cat((target_masks,masks[i][index]),dim=0)
        
        # TODO use valid to mask invalid areas due to padding in loss
        # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # target_masks = target_masks
        # target_masks = target_masks[tgt_idx]

        # losses = {
        #     "loss_mask": sigmoid_ce_loss_jit(src_masks, target_masks, num_masks),
        #     "loss_dice": dice_loss_jit(src_masks, target_masks, num_masks),
        # }

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        # src_masks = src_masks[:, None]
        # target_masks = target_masks[:, None]

        # with torch.no_grad():
            # sample point_coords
            # point_coords = get_uncertain_point_coords_with_randomness(
            #     src_masks,
            #     lambda logits: calculate_uncertainty(logits),
            #     self.num_points,
            #     self.oversample_ratio,
            #     self.importance_sample_ratio,
            # )
            
            # # point_coords = torch.rand(1, 240,180,16, 3, device=target_masks.device)
            # # get gt labels
            # point_labels = point_sample(
            #     target_masks,
            #     point_coords,
            #     align_corners=False,
            # ).squeeze(1)
            # point_labels = target_masks.flatten(1)

        # point_logits = point_sample(
        #     src_masks,
        #     point_coords,
        #     align_corners=False,
        # ).squeeze(1)
        #for debug 
        # pred_label = outputs["pred_logits"]
        # pred_label = pred_label[src_idx].argmax(-1)
        # target_label = [t for t in targets[0]]
        # # target_label = target_label[0][index]
        # for i in range(len(target_label)):
        #     temp_index = tgt_idx[0]==i
        #     target_label[i] = target_label[i][tgt_idx[1][temp_index]]
        # target_label = torch.cat(target_label)
        # for i in range(len(target_masks)):
        #     temp_src_index = torch.where(src_masks[i]>0)
        #     temp_src_pred = src_masks[i][temp_src_index]

        #     temp_target_index = torch.where(target_masks[i]>0)
        #     temp_target_pred = src_masks[i][temp_target_index]
            
        #     temp_target_label = target_label[i]
        #     temp_pred_label = pred_label[i]
        #     a=1
            
            
        point_labels = target_masks.flatten(1)
        point_logits = src_masks.flatten(1)
        valid = (point_labels!=-1)
        point_labels = point_labels[valid].unsqueeze(0)
        point_logits = point_logits[valid].unsqueeze(0)
        # temp = sigmoid_focal_loss(point_logits, point_labels, num_masks)
        # temp1 = dice_loss(point_logits, point_labels, num_masks)
        # temp2 = sigmoid_ce_loss(point_logits, point_labels, num_masks)
        
        losses = {
            "loss_mask": batch_sigmoid_focal_loss_jit(point_logits, point_labels, num_masks),
            # "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }
        for k in list(losses.keys()):
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets,semantic_pred=None,semantic_gt=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t) for t in targets[0])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # num_masks = torch.as_tensor(
        #     [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        # )
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_masks)
        # num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs['batch_index'] = outputs['batch_index']
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        #for semantic loss
        # semantic_loss = lovasz_softmax(F.softmax(semantic_pred), semantic_gt,ignore=255) + self.CE_loss(semantic_pred,semantic_gt)
        # losses.update({'loss_semantic':semantic_loss})

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
