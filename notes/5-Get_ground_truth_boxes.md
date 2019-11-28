
# Get ground truth boxes
There is no gt_boxes in inference time, so we need to find the tool functions and process the ***corresponding*** ground truth data manually  

There are 2 possible solutions:  
1. Extract image sequence in inference time as well, then manually load the data (requires cityscapes script)
2. dig into the `evaluator` and extract the ground-truth boxes according to its order

~~Augment the `evaluator`~~

## Get Ground Truth Bounding Boxes
TODOTODOTODOTODOTODO

## class CityscapesEvaluator
see [official docs](https://detectron2.readthedocs.io/modules/evaluation.html)  

in CityscapesEvaluator.evaluate(), the function calls cityscapes script  
```
results = cityscapes_eval.evaluateImgLists(
    predictionImgList, groundTruthImgList, cityscapes_eval.args
)["averages"]
```
### ImgList Format
    line 98: groundTruthImgList = glob.glob(cityscapes_eval.args.groundTruthSearch)

glob.glob(pathname) -> List[`string`]  
find file(s)

-----------------------------------------
### detectron2.data.datasets.load_cityscapes_instances(image_dir, gt_dir, from_json=True, to_polygons=True)
Parameters:  
* **image_dir** (`str`) – path to the raw dataset. e.g., “~/cityscapes/leftImg8bit/train”.
* **gt_dir** (`str`) – path to the raw annotations. e.g., “~/cityscapes/gtFine/train”.
* **from_json** (`bool`) – whether to read annotations from the raw json file or the png files.
* **to_polygons** (`bool`) – whether to represent the segmentation as polygons (COCO’s format) instead of masks (cityscapes’s format).

Returns:
* list[dict] - a list of dicts in Detectron2 standard format. (See [Using Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html))
