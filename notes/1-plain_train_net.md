# plain_train_net.py

## 入口: main(args)
顺序调用调用文件内定义的  
cfg = setup(args)  
model = build_model(cfg) -- definition in detectron2/modeling/meta_arch/build.py  
model is a torch.nn.Module  
do_train(cfg, model) -- 若eval_only则不执行, 并额外执行加载checkpoint步骤  
do_test(cfg, model) 

### Sampler 概念
sampler: 索引生成器, 是 dataloader 的一个组件  
sampler 生成的索引 在 data 中得到数据  
有 sequential sampler, random sampler, weighted sampler 等 

## do_test(cfg, model)
data_loader = build_detection_test_loader(cfg, dataset_name)  
>得到一个torch.utils.data.dataloader, batch_size = 1  

evaluator = get_evaluator(...)  
>Create evaluator(s) for a given dataset.  
This uses the special metadata "evaluator_type" associated with each builtin dataset.  
函数定义在 plain_train_net.py 内, 以一堆 if else 判断字符串, 尝试使用预先做好的对著名数据集的专用evaluator  
我们使用的是 CityscapesEvaluator

result[one_of_the_val_sets] = inference_on_dataset(model, data_loader, evaluator)  
>见下文详细分析

return results  

## inference_on_dataset(model, data_loader, evaluator) 
对于每一组数据 (eval_only 中 batch_size = 1) 核心代码如下:  
outputs = model(inputs) -- 调用 GeneralizedRCNN.forward() 见下文详细分析  
evaluator.process(inputs, outputs)
>这里是调用 cityscapes scripts 来计算各指标, 我们不需要这一部分  

## model config:
见Base-RCNN-FPN.yaml  
META_ARCHITECTURE: GeneralizedRCNN --- modeling/meta_arch/rcnn.py 
