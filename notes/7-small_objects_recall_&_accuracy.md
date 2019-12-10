# Compute Recall of Small Objects & General Accuracy

## Recall of small objects:
1. Filter the boxes with area < `area_threshold`
2. Calculate recall

##  Accuracy  
use DefaultPredictor to input one image at a time, pickle middle data out from the function, then rename it to the corresponding imageName

Accuracy 定义: 对于每个proposal, 定义他的ground truth为与他iou最大且大于一定阈值((0.4, 0.5)之类, 放宽一点)的gtbox, 并以分的类别标记正负样本

## 找到result.pred_classes 和 label 的对应关系
result 是一个 `instance` with field `pred_classes`  
试着从evaluator中找到对应关系  
参考 https://github.com/facebookresearch/detectron2/blob/2d36275c6b048f6477468cc25601f1abc17f4e4f/detectron2/evaluation/evaluator.py  
Line 83, evaluator 为 cityscapes evaluator  

then call to evaluator.process() in  
https://github.com/facebookresearch/detectron2/blob/2d36275c6b048f6477468cc25601f1abc17f4e4f/detectron2/evaluation/cityscapes_evaluation.py

    line 60: 
    pred_class = output.pred_classes[i]
    classes = self._metadata.thing_classes[pred_class]
    class_id = name2label[classes].id

To find `thing_classes`, we goes to  
https://github.com/facebookresearch/detectron2/blob/2d36275c6b048f6477468cc25601f1abc17f4e4f/detectron2/data/datasets/cityscapes.py

    line 300:
    thing_classes = [k.name for k in labels if k.hasInstances and not k.ignoreInEval]
    meta = Metadata().set(thing_classes=thing_classes)

To find out what's labels, hasInstances, ignoreInEval   
reference from **cityscapes scripts**:
```python
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

```
Finally, we found the 8 labels

------------
## TODOs
遍历json, 提取出小物体json, inference所有数据之后手动算他的mAp(在object detection 任务上)  
改变iou, recall-precision 作为xy 绘图, 计算在x=1, y=1(理想状态)下的面积 作为mAp


**Remember to disable the topk selection and score_threshold and NMS!!!!**  
disable them by extracting data before selection  
https://github.com/facebookresearch/detectron2/blob/2d36275c6b048f6477468cc25601f1abc17f4e4f/detectron2/modeling/roi_heads/fast_rcnn.py#L66  
Done

## Next


写 paper 的 introduction, abstract 之类

-------