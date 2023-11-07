import imp


import torch
from torch.nn import functional as F

def semantic_inference(mask_cls, mask_pred):
        #for debug 
        # scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qh->ch", mask_cls, mask_pred)
        result = semseg.transpose(1,0).contiguous()
        # result = torch.add(result,1)
        # result [result==19]=0
        return result
def semantic_inference_argmax(mask_cls, mask_pred):
        num_classes= 19
        object_mask_threshold = 0.7
        overlap_threshold = 0.8
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(num_classes) & (scores > object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1) * cur_masks

        h, w= cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((w), dtype=torch.int64, device=cur_masks.device)


        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)

            
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                # isthing = pred_class<10 and pred_class>0
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] > 0.5).sum().item()

                mask = (cur_mask_ids == k) & (cur_masks[k] >0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue


                    
                    panoptic_seg[mask] = pred_class
                  
  

            return panoptic_seg
def panoptic_inference(mask_cls, mask_pred,thing_seg_2d):
        num_classes= 16
        object_mask_threshold = 0.7
        overlap_threshold = 0.8
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(num_classes) & (scores > object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w= cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int64, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()+1
                # isthing = pred_class<10 and pred_class>0
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] > 0.5).sum().item()

                mask = (cur_mask_ids == k) & (cur_masks[k] >0.5) & thing_seg_2d

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    # # merge stuff regions
                    # if not isthing:
                    #     if int(pred_class) in stuff_memory_list.keys():
                    #         panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                    #         continue
                    #     else:
                    #         stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    if pred_class>10:
                        panoptic_seg[mask] = pred_class
                    else:
                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id * 65536+pred_class
                    # panoptic_seg[mask] = pred_class * 65536+current_segment_id
                    # segments_info.append(
                    #     {
                    #         "id": current_segment_id,
                    #         "isthing": bool(isthing),
                    #         "category_id": int(pred_class),
                    #     }
                    # )

            return panoptic_seg

def merge_semantic_and_instance(sem_seg, ins_seg ,label_divisor, thing_list, void_label,thing_seg):
    """
    Post-processing for panoptic segmentation, by merging semantic segmentation label and class agnostic
        instance segmentation label.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W, Z], predicted semantic label.
        sem: A Tensor of shape [1, C, H, W, Z], predicted semantic logit.
        ins_seg: A Tensor of shape [1, H, W], predicted instance label.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        thing_list: A List of thing class id.
        void_label: An Integer, indicates the region has no confident prediction.
        thing_seg: A Tensor of shape [1, H, W, Z], predicted foreground mask.
    Returns:
        A Tensor of shape [1, H, W, Z] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.
    """
    # In case thing mask does not align with semantic prediction
    # semantic_thing_seg = torch.zeros_like(sem_seg,dtype=torch.bool)
    # for thing_class in thing_list:
    #     semantic_thing_seg[sem_seg == thing_class] = True
    
    # try to avoid the for loop
    # semantic_thing_seg = sem_seg<=max(thing_list)
    semantic_thing_seg = torch.logical_and(sem_seg <= max(thing_list), sem_seg != void_label)

    ins_seg = torch.unsqueeze(ins_seg,3).expand_as(sem_seg)
    thing_mask = (ins_seg > 0) & semantic_thing_seg & thing_seg
    if not torch.nonzero(thing_mask).size(0) == 0:
        # sem_sum = torch_scatter.scatter_add(sem.permute(0,2,3,4,1)[thing_mask],ins_seg[thing_mask],dim=0)
        
        sem_seg[thing_mask] = ins_seg[thing_mask]
    else:
        sem_seg[semantic_thing_seg & thing_seg] = void_label
    return sem_seg


def get_panoptic_segmentation(mask_cls, mask_pred,thing_list, label_divisor=2**16, void_label=0,
                              threshold=0.1, nms_kernel=5, top_k=100, foreground_mask=None, polar=False,
                              pseudo_threshold=None, uq_grid=None, uq_alpha=None, uq_box_labels=None):
    """
    Post-processing for panoptic segmentation.
    Arguments:
        sem: A Tensor of shape [N, C, H, W, Z] of raw semantic output, where N is the batch size, for consistent,
            we only support N=1.
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        label_divisor: An Integer, used to convert panoptic id = instance_id * label_divisor + semantic_id.
        void_label: An Integer, indicates the region has no confident prediction.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        foreground_mask: A processed Tensor of shape [N, H, W, Z], we only support N=1.
    Returns:
        A Tensor of shape [1, H, W, Z] (to be gathered by distributed data parallel), int64.
    Raises:
        ValueError, if batch size is not 1.
    """
    # if sem.dim() != 5 and sem.dim() != 4:
    #     raise ValueError('Semantic prediction with un-supported dimension: {}.'.format(sem.dim()))
    # if sem.dim() == 5 and sem.size(0) != 1:
    #     raise ValueError('Only supports inference for batch size = 1')

    # if foreground_mask is not None:
    #     if foreground_mask.dim() != 4:
    #         raise ValueError('Foreground prediction with un-supported dimension: {}.'.format(sem.dim()))

    # if sem.dim() == 5:
    #     sem = F.softmax(sem)
    #     if uq_grid is not None:
    #         thing_list_tensor = torch.tensor(thing_list).cuda() - 1
    #         uq_alpha = torch.from_numpy(uq_alpha).cuda()
    #         uq_grid = torch.from_numpy(uq_grid).cuda()
    #         uq_box_labels = torch.from_numpy(uq_box_labels).cuda()

    #         predict = torch.argmax(sem, dim=1)
    #         uq_grid_predict = predict[0, uq_grid[:, 0], uq_grid[:, 1], uq_grid[:, 2]]
    #         uq_grid_predict_thing = torch.isin(uq_grid_predict, thing_list_tensor)
    #         uq_grid = uq_grid[uq_grid_predict_thing]
    #         uq_alpha = uq_alpha[uq_grid_predict_thing]
    #         uq_box_labels = uq_box_labels - 1
    #         uq_box_labels = uq_box_labels[uq_grid_predict_thing]

    #         augment_alpha = uq_alpha + (1 - uq_alpha) * sem[0, uq_box_labels, uq_grid[:, 0], uq_grid[:, 1], uq_grid[:, 2]]
    #         decay_alpha = (1 - uq_alpha) * sem[0, :, uq_grid[:, 0], uq_grid[:, 1], uq_grid[:, 2]]
    #         sem[0, :, uq_grid[:, 0], uq_grid[:, 1], uq_grid[:, 2]] = decay_alpha
    #         sem[0, uq_box_labels, uq_grid[:, 0], uq_grid[:, 1], uq_grid[:, 2]] = augment_alpha

    #     semantic = torch.argmax(sem, dim=1)
    #     # shift back to original label idx 
    #     semantic = torch.add(semantic, 1)

    #     if pseudo_threshold is not None:
    #         uncertain_mask = torch.max(sem, dim=1)[0] < pseudo_threshold
    #         # thing类单独卡阈值
    #         # if uq_grid is not None:
    #         #     thing_list_tensor = torch.tensor(thing_list).cuda()
    #         #     predict_thing_mask = torch.isin(semantic, thing_list_tensor)
    #         #     uncertain_stuff_mask = torch.logical_and(uncertain_mask, ~predict_thing_mask)
    #         #
    #         #     uncertain_thing_mask = torch.logical_and(torch.max(sem, dim=1)[0] < 0.6, predict_thing_mask)
    #         #     uncertain_mask = torch.logical_or(uncertain_stuff_mask, uncertain_thing_mask)
    #         # thing类不卡阈值
    #         # if uq_grid is not None:
    #         #     thing_list_tensor = torch.tensor(thing_list).cuda()
    #         #     predict_thing_mask = torch.isin(semantic, thing_list_tensor)
    #         #     uncertain_mask = torch.logical_and(uncertain_mask, ~predict_thing_mask)
    #         # 没有热力值的thing voxel在2D框外,置为0
    #         # if uq_grid is not None:
    #         #     thing_list_tensor = torch.tensor(thing_list).cuda()
    #         #     predict_thing_mask = torch.isin(semantic, thing_list_tensor)
    #         #     uncertain_mask = torch.logical_and(uncertain_mask, ~predict_thing_mask)
    #         # 将所有stuff类别设置为ignore,thing类不卡阈值
    #         # if uq_grid is not None:
    #         #     thing_list_tensor = torch.tensor(thing_list).cuda()
    #         #     predict_thing_mask = torch.isin(semantic, thing_list_tensor)
    #         #     uncertain_mask = ~predict_thing_mask
    #         semantic[uncertain_mask] = void_label
    # else:
    #     semantic = sem.type(torch.ByteTensor).cuda()
    #     # shift back to original label idx 
    #     semantic = torch.add(semantic, 1).type(torch.LongTensor).cuda()
    #     one_hot = torch.zeros((sem.size(0),torch.max(semantic).item()+1,sem.size(1),sem.size(2),sem.size(3))).cuda()
    #     sem = one_hot.scatter_(1,torch.unsqueeze(semantic,1),1.)
    #     sem = sem[:,1:,:,:,:]


    if foreground_mask is not None:
        thing_seg = foreground_mask
    else:
        thing_seg = None

    thing_seg_2d = thing_seg.squeeze().max(-1)[0]
    panoptic= panoptic_inference(mask_cls, mask_pred,thing_seg_2d)
    # panoptic = merge_semantic_and_instance(semantic, instance.unsqueeze(0), label_divisor, thing_list, void_label, thing_seg)
    panoptic = panoptic.unsqueeze(0)
    panoptic = torch.unsqueeze(panoptic ,3).expand_as(thing_seg)
    # panoptic[thing_seg] = 1
    return panoptic