## 从开始扩散到DALLE2

> 2015

<b>*Deep unsupervised learning using nonequilibrium thermodynamics*</b>
为了消除对训练图像连续应用的高斯噪声，它用了一种被称为“潜在扩散模型”(latent diffusion model, LDM)变体。

> 2020

DDPM将扩散模型的思想用于图像生成。

>> 关于生成模型

江湖存在几个主流门派：

- GAN
- VAE
- Flow-based models
- Diffusion models

由于主要介绍扩散模型，因此其他的模型不做故事线展开。不过直到今天（2023年10月），扩散模型无论是可解释性还是效果，都已经超越了其他模型，成为领头羊。

> 2021年2月

<b>*Improved Denoising Diffusion Probabilistic Models*</b>一文改进了DDPM，主要是把添加噪声的schedule从线性改成余弦，使得DDPM对低分辨率图像也有较好学习效果。

> 2021年5月

<b>*Diffusion Models Beat GANs on Image Synthesis*</b>一文从GAN的实验中得到启发，对扩散模型进行了大量的消融实验，找到了更好的架构更深更宽的模型，提出Classifier Guidance，训练一个分类器能更好的告诉UNet的模型在反向过程生成新图片的时候，当前图片有多像需要生成的物体。

> 2022年6月

<b>*Classifier-Free Diffusion Guidance*</b>改进了Classifier Guidance，该文作者认为前文的方法需要用预训练的模型或者额外训练一个模型，不仅成本比较高而且训练的过程是不可控的。提出Classifier-Free Guidance。实际上，但方法本身仍然是"昂贵的"，因为训练的时候需要生成两个输出。在扩散模型本身就很慢的情况下，会进一步增加耗时。

> 2021年12月

<b>*GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models*</b>一文标示着OpenAI开始探索文本条件下的图像生成，该方法沿袭了OpenAI一贯的做法，什么模块效果好就用什么，然后进一步增加模型的参数量和数据量。

> 2022年4月

让大名鼎鼎的DALLE2成为划时代的经典之作主要是由于他惊艳世人的效果——<b>*Hierarchical Text-Conditional Image Generation with CLIP Latents*</b>。如果说前面所提到的方法将扩散模型优化到比同期GAN模型指标还要好，让研究人员看到了扩散模型在生成领域的前景，那么Dalle2则将扩散模型引入了公众视野。

从方法上说，DALLE2使用了扩散模型classifier-free guidance，为了实现文本生成图，模型实现上使用的是transformer的decoder，模型的输入非常多，包含文本、CLIP的text embedding、扩散模型中常见的time step的embedding，还有加过噪声之后CLIP的image embedding；输出则预测没有噪声的CLIP的image embedding。和之前扩散模型不一样的地方在于没有使用DDPM中预测噪声的方式，而是直接还原每一步的图像。