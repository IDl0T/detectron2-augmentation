# Generalized RCNN

## class GeneralizedRCNN – in modeling/meta_arch/rcnn.py 
Generalized R-CNN. Any models that contains the following three components: 
1. Per-image feature extraction (aka backbone)
2. Region proposal generation 
3. Per-region feature extraction and prediction  

### 重要member: 
```
self.backbone = build_backbone(cfg)  

self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())  

self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape()) 
```

## forward(self, batched_inputs) 
if not training, call inference()

## inference(self, batched_inputs, detected_instances=None, do_postprocess=True) 
```images = self.preprocess_image(batched_inputs)  
features = self.backbone(images.tensor)  
proposals, _ = self.proposal_generator(images, features, None) 
```
试对 proposal_generator 进行研究, 并提取proposal  