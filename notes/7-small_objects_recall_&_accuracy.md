# Compute Recall of Small Objects & General Accuracy

## Recall of small objects:
1. Filter the boxes with area < `area_threshold`
2. Calculate recall

##  Accuracy
Read proposal boxes from pickle files

## Prepare Model Inputs
1. modify detectron2/data/datasets/cityscapes.py, make it sorted
2. use detectron2.data.DatasetMapper(cfg, is_train=false) to map the `detectron2std` format to `modelInput` format 
3. add precomputed proposals to the dataset: `List[dict]`

Forward the precomputed proposals to model, getting the `output`

------------
## TODOs
https://github.com/facebookresearch/detectron2/blob/2d36275c6b048f6477468cc25601f1abc17f4e4f/detectron2/modeling/roi_heads/fast_rcnn.py#L41  

Find predicted classes in it.  

    line 117: result.pred_classes = filter_inds[:, 1]

this `result` directly returns to detectron2/modeling/roi_heads/roi_heads.py, line 624L outputs = FastRCNNOutputs(...)

**Remember to disable the topk selection and score_threshold and NMS!!!!**