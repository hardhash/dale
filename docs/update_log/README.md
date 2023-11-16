
### v2023.11a

1. NLP板块布局重构，模块更改为《2022年以前的NLP》和《大模型时代的NLP》并新增阅读指引，v2023.11版本之前的内容均放在《2022年以前的NLP》，在《大模型时代的NLP》更新如下内容：
   
   - ModelScope - 无需魔法实现模型快捷下载
   - Transformers基础组件 - 包括pipeline，tokenizer，model，datasets，evaluate，trainer，让你几十行代码即可轻松训练NLP模型
   - 基于Transformer的各类NLP下游任务解决方案
     - 命名实体识别(NER)任务
     - 机器阅读理解(MRC)任务
     - 多项选择(Multi-choices)任务
     - 文本相似度匹配(Text-Match)任务
     - 文本摘要(Text-Summarization)任务
   - Transformers改变NLP生态 - 不仅提高代码复用率，教你如何从源头调参实现小显卡(4Gb)撬动大模型(baseline 15Gb开销)
   - BitFit大模型调参微调方案，简单粗暴冻结参数，实现局部训练
   - 基于PEFT的大模型调参微调方案，教你如何使用PEFT几行代码高效调参实现小显卡(8Gb)撬动大模型(1.4B)
     - Prompt-Tuning
     - P-Tuning
     - Prefix-Tuning
     - LoRa
     - IA3
     - PEFT常用模块
   - 16bit(FP16)的半精度训练，教你如何让机器轻松微调驾驭7B+大模型
     - LLaMA2-7B半精度训练
     - ChatGLM3-6B-Base半精度训练
   - 8bit(INT8)和QLoRA(4bit, NF4)的低精度训练/推理，教你如何用量化把FP16模型转化为低位宽模型，让你在8Gb显卡上训练7B模型。
     - LLaMA2-7B 8bit/16bit混合精度训练
     - ChatGLM3-6B-Base 8bit/16bit混合精度训练
     - LLaMA2-7B 4bit/16bit混合精度训练
     - ChatGLM3-6B-Base 4bit/16bit混合精度训练

2. CV板块新增：
    
   - MMPretrain的图像分类(Image Classification)任务解决方案

3. GNN板块更名为图模型，把图机器学习 & 图神经网络基础原理拆分成图机器学习和图神经网络两部分，图机器学习保留之前除GCN的内容，GCN归类到图神经网络中，并在图神经网络中更新如下内容：

   - 万物皆可GNN - GNN发展的必要性和应用方向
   
4. 上方导航栏NLP部分新增:
   - Transformers实现BERT文本分类 - 教你从头实现分类模型，Transformers框架简洁实现以及Optuna自动调参
   - Transformers实现相似文本召回精排 - 经典双塔模型带你入门推荐系统
   - Prompt生成式对话机器人 - 还有什么比自己动手训练一个ChatGPT更有趣的呢

5. 论文精读板块

   **计算机视觉-ILSVRC**，新增AlexNet，VGG，GoogLeNet_v1-v4，ResNet，ResNeXt，SENet，DenseNet

[//]: # (   **计算机视觉-目标分割**，新增FCN，UNet，FusionNet，SegNet，DeconvNet，DeepLab_v1-v3plus)

[//]: # ()
[//]: # (   **计算机视觉-目标检测**，新增YOLOv3，FPN，FasterRCNN，Mask RCNN，Cascade RCNN，Cascade Mask RCNN)

[//]: # ()
[//]: # (   **计算机视觉-transformer**，新增Vit，PVT，Swin transformer)

[//]: # ()
[//]: # (   **计算机视觉-新时代卷积backbones**，新增EfficientNet_v1-v2，ConvNeXt_v1-v2，MAE，LVT)

6. 竞赛总结板块
    
    - 新增【(2023 数据挖掘)Seed2023新能源(**TOP 10%**)】

7. 博客JavaScript功能模块和板块
    
   - 新增返回顶部按钮（下翻距离2500后自动出现）
   - 上线**【更新日志】**板块

****

### v2023.10

博客上线，撒花🌹🌷🌺🌸🌼💐🏵️🪷🍀🪻（PS：悄悄准备点emoji素材以后优化排版模板）