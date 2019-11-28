# Instances

## class `Instances`
相当于一个各种对象 (boxes, masks, labels, scores) 的 list  
**all fields must have the same size / len**

Some basic usage:  
1. Set/Get a field:  
    instances.gt_boxes = Boxes(...)  
    instances.XX -- 添加新field  
    print(instances.pred_masks)  
    print('gt_masks' in instances)  
2. `len(instances)` returns the number of instances  
3. Indexing: `instances[indices]` will apply the indexing on all the fields and returns a new `Instances`.  
Typically, `indices` is a binary vector of length num_instances, or a vector of integer indices.

## Review
在 rpn.py / rpn_outputs.py 中返回的 `proposals` 是一个 `List[Instances]` 这篇笔记探讨 `proposals` 中 `Instances` 的数据格式

## Data format
### relevant code:
```
res = Instances(image_size)
res.proposal_boxes = boxes[keep]
res.objectness_logits = scores_per_img[keep]
results.append(res)
```
`res` 即为 `proposals`  
可以推断得出有 proposal_boxes, objectness_logits 这两个 field 
### format of proposal_boxes
    boxes = Boxes(topk_proposals[n])
#### class `Boxes` stores a N * 4 Tensor in self.tensor   
    __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "`Boxes`"
    pairwise_iou(boxes1, boxes2) -> (N, M) Tensor
    matched_boxlist_iou(boxes1, boxes2) -> (N, ) Tensor # compute iou between corresponding pairs 
详细见 detectron2 docs
### format of objectness_logits
float Tensor


