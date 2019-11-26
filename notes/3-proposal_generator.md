# proposal_generator
cfg.PROPOSAL_GENERATOR.NAME = "RPN"  
definition in "modeling/proposal_generator/rpn.py"

## class RPN(nn.Module)
### forward(self, images, features， gt_instances=None) 此函数被调用

Returns:   
proposals: list[Instances] or None  
这里 instance 是 class RPNOutputs  
loss: dict[Tensor]

>proposals = find_top_rpn_proposals(RPNOutputs, ...)
proposals 为执行 Non Max Suppression 并去除过小的 proposal 的剩余部分  
proposals 根据置信度排序 (不必要)  
proposals 是一个包含 class `Instances` 的 list

**所以最后要把RPN.forward()的结果抠出来**

### class `Instances`
相当于一个各种对象 (boxes, masks, labels, scores) 的 list

Some basic usage:  
1. Set/Get a field:  
    instances.gt_boxes = Boxes(...)  
    instances.XX -- 添加新field  
    print(instances.pred_masks)  
    print('gt_masks' in instances)  
2. `len(instances)` returns the number of instances  
3. Indexing: `instances[indices]` will apply the indexing on all the fields and returns a new `Instances`.  
Typically, `indices` is a binary vector of length num_instances, or a vector of integer indices.

详细内容见下篇笔记 **4-Instances**

### find_top_rpn_proposals(...) -> List[`Instances`]

-----------------------------
## class `RPNOutputs` (RPN的raw输出, 中间变量, 并不重要)

### Shape shorthand in this module:
`N`: number of images in the minibatch  
`L`: number of feature maps per image on which RPN is run  
`A`: number of cell anchors (must be the same for all feature maps)  
`Hi`, `Wi`: height and width of the i-th feature map  
`4`: size of the box parameterization  
### Naming convention:  
`objectness`: refers to the binary classification of an anchor as object vs. not object.  
`deltas`: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box transform (see :class:`box_regression.Box2BoxTransform`). 
`pred_objectness_logits`: predicted objectness scores in [-inf, +inf]; use sigmoid(pred_objectness_logits) to estimate P(object).  
`gt_objectness_logits`: ground-truth binary classification labels for objectness  
`pred_anchor_deltas`: predicted box2box transform deltas  
`gt_anchor_deltas`: ground-truth box2box transform deltas

### Properties:  

**box2box_transform (Box2BoxTransform)**: :class:`Box2BoxTransform` instance for anchor-proposal transformations.  

**anchor_matcher (Matcher)**: :class:`Matcher` instance for matching anchors to
ground-truth boxes; used to determine training labels.  

**batch_size_per_image (int)**: number of proposals to sample when training

**positive_fraction (float)**: target fraction of sampled proposals that should be positive

**images (ImageList)**: :class:`ImageList` instance representing N input images

**pred_objectness_logits (list[Tensor])**: A list of L elements.  
Element i is a tensor of shape (N, A, Hi, Wi) representing
the predicted objectness logits for anchors.  

**pred_anchor_deltas (list[Tensor])**: A list of L elements. Element i is a tensor of shape
(N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
to proposals.    

**anchors (list[list[Boxes]])**: A list of N elements. Each element is a list of L
Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
n (i.e. the cell anchors repeated over all locations in feature map (n, l)).  

**boundary_threshold (int)**: if >= 0, then anchors that extend beyond the image
boundary by more than boundary_thresh are not used in training. Set to a very large
number or < 0 to disable this behavior. Only needed in training.  

**gt_boxes (list[Boxes], optional)**: A list of N elements. Element i a Boxes storing
the ground-truth ("gt") boxes for image i.  

**smooth_l1_beta (float)**: The transition point between L1 and L2 loss in
the smooth L1 loss function. When set to 0, the loss becomes L1. When
set to +inf, the loss becomes constant 0.  

#### class `Boxes`: A n*4 Tensor storing a list of boxes    

