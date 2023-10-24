扩散模型是一类生成模型，运用物理热力学中的扩散思想，主要包括前向扩散和反向扩散。

> 生成模型

根据给定的样本（训练数据）生成新样本。首先给定一批训练数据$X$，假设其分布服从某种复杂的真实分布$p(x)$，则给定的训练样本数据可视为从该分布中采样的观测样本$x$。
如果能够从这些观测样本中估计出训练数据的真实分布，就可以从该分布中源源不断的采出新样本。这个源源不断采集新样本的过程就是生成模型正在做的，它的作用就是估计训练数据的真实分布并将其假定喂$q(x)$。这也就是拟合网络。

>> 存在的问题

如何衡量估计分布$q(x)$和真实分布$p(x)$的差距。最简单的办法是要求所有的训练样本采集自$q(x)$的概率最大，这也就是极大似然估计思想，这也是生成模型最基本的思想之一，生成模型的学习目标就是对训练数据的分布进行建模。

> 扩散模型的思想

扩散模型思想来非平衡热力学。简单说就是一个混沌体系熵增的演化过程。

## 通俗理解

在水中滴入一滴墨水，这滴墨水刚开始在水中会形成一个非常浓的斑点，我们认为这是它的初始状态。
我们想描述这个墨水的初始状态分布其实是很困难的。然后墨水开始向着四面八方扩散，如果拍摄记录整个过程，用帧的概念，我们可以认为每一帧（每一个时间步）墨水都在向水中进行扩散，直到水的颜色逐渐变成墨水的颜色。
当墨水颜色和水的颜色一致后，我们就很容易用数学公式描述此时（终止状态）的概率分布。

在例子的情况下，非平衡热力学可以用来描述每一个时间步状态的概率分布。简单来说，就是把连续过程离散化，然后做估计。实际上，如果可以想到办法把这个过程反过来，那就可以从最简单的终止时刻的概率分布（例子是一个均匀分布），去逐步还原上N个时间步的复杂分布。

## 扩散模型底层逻辑：DDPM(Denoising Diffusion Probabilistic Model)

DDPM认为扩散过程是马尔可夫过程，即每一个时间步的分布是由它上一个时间步状态加上当前空间步高斯噪声得到的，即它假设扩散的逆过程是高斯分布。

### 前向过程（加噪）

前向过程就是给数据添加噪声。每一个时间步$t$给上一个时间步$t-1$的数据$x_{t-1}$添加高斯噪声，从而生成带有噪声的数据$x_t$，同时数据$x_t$也会被送入下一个时间步$t+1$继续添加噪声。

**注意：在传统的图像处理时，往往是通过一个高斯滤波器来去除噪声干扰，使得图像尽可能纯粹，在这里我们拿到的图像被认为是最纯粹的图像，因此需要反向增加高斯噪声**

****
> 【拓展】高斯噪声 

给定均值为$\mu$，方差为$\sigma^2$的单一变量高斯分布$\mathcal N (\mu,\sigma^2)$其概率密度函数为：
$q(z) = \frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(z-\mu)^2}{2\sigma^2})$

很多时候为了方便就把前面的常系数去掉，即$q(z) \propto exp(-\frac{(z-\mu)^2}{2\sigma^2})$，此外，给定两个高斯分布$X \sim \mathcal N(\mu_1, \sigma_1^2),Y \sim \mathcal N(\mu_2, \sigma_2^2)$，他们叠加后的$aX+bY$满足：

$aX+bY \sim \mathcal N (a\mu_1+b\mu_2, a^2\sigma_1^2+b^2\sigma_2^2)$
****

然后，我们用公式去描述前向传播过程，噪声的方差$\sigma$由位于区间(0,1)的固定值$\beta_t$确定，均值$\mu$则由固定值$\beta_t$和当前时刻数据分布决定，不妨让均值为0方便计算。从$t-1$时间步到$t$时间步的单步扩散加噪：

$$x_t=\sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}z_t，z_t\sim\mathcal N(0,I)$$

根据定义，加噪过程是确定的，并不是可学习的过程，将其写成概率分布的形式，则有：

$$q(x_t | x_{t-1}) = \mathcal N (x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI ))$$

此外，加噪过程是一个马尔科夫链过程，所以联合概率分布（最终噪声的分布）数学表达式如下：
$$q(x_{1:T}|x_0)=\prod \limits_{t=1}^T q(x_t|x_{t-1})$$

理论上，只要这个T足够大，这个混沌系统将再次趋于稳定，根据马尔可夫链的性值最终就可以得到纯随机噪声分布的数据，类似墨水稳定系统。

为了不造成阅读障碍，再证明一下这个“理论上”成立的事：

定义$\alpha_t = 1 - \beta_t$，即$\alpha_t + \beta_t = 1$，带入$x_t$表达式并推导，可以得到$x_0$到$x_t$的公式：

$x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt\beta_t z_t$

$= \sqrt\alpha_t x_{t-1}+\sqrt\beta_tz_t$

$= \sqrt\alpha_t \textcolor{red}{(\sqrt\alpha_{t-1} x_{t-2}+\sqrt\beta_{t-1}z_{t-1})} + \sqrt\beta_tz_t$

$= \sqrt{\alpha_t\alpha_{t-1}} \textcolor{red}{(\sqrt\alpha_{t-2} x_{t-3}+\sqrt\beta_{t-2}z_{t-2})}+ \sqrt{\alpha_t\beta_{t-1}}z_{t-1}+ \sqrt\beta_tz_t$

$...$

$= \sqrt{\alpha_t\alpha_{t-1}\dots\alpha_1}x_0 + \sqrt{\alpha_t\alpha_{t-1}\dots\alpha_2\beta_1}z_1 + \sqrt{\alpha_t\alpha_{t-1}\dots\alpha_2\beta_2}z_2 + \dots + \sqrt{\alpha_t\alpha_{t-1}\beta_{t-2}}z_{t-2} + \sqrt{\alpha_t\alpha_t\beta_{t-1}}z_{t-1} + \sqrt\beta_tz_t$

****
>> 在上面的式子中，第二项到最后一项都是独立高斯噪声，均值都是之前设置好的0，方差为各自系数平方。多个高斯分布叠加的结果仍然服从整体均值0，方差为各方差之和的高斯分布，此外，还可以证明上式每一项系数的平方和都是1 证明如下：

$\alpha+t\alpha_{t-1}\dots\alpha_1+\alpha_t\alpha_{t-1}\dots\alpha_2\beta_1+\dots+\alpha_t\alpha_t\beta_{t-1}+\beta_t$

$=\alpha+t\alpha_{t-1}\dots\alpha_2(\alpha_1+\beta_1)+\alpha_t\alpha_{t-1}\dots\alpha_3\beta_2+\dots+\alpha_t\alpha_t\beta_{t-1}+\beta_t$

$=\alpha+t\alpha_{t-1}\dots\alpha_2\textcolor{red}{1}+\alpha_t\alpha_{t-1}\dots\alpha_3\beta_2+\dots+\alpha_t\alpha_t\beta_{t-1}+\beta_t$

$=\alpha+t\alpha_{t-1}\dots\alpha_3(\alpha_2+\beta_2)+\alpha_t\alpha_{t-1}\dots\alpha_4\beta_3+\dots+\alpha_t\alpha_t\beta_{t-1}+\beta_t$

$=\alpha+t\alpha_{t-1}\dots\alpha_3\textcolor{red}{1}+\alpha_t\alpha_{t-1}\dots\alpha_4\beta_3+\dots+\alpha_t\alpha_t\beta_{t-1}+\beta_t$

$\dots$

$=\alpha_t+\beta_t = 1$
****
把$\alpha_t\alpha_{t-1}\dots\alpha_1$记作$\overline\alpha_t$，则正态噪声的方差之和为$1-\overline\alpha_t$，此时：

$$\overline x_t = \sqrt{\overline \alpha_t}x_0+\sqrt{1-\overline \alpha_t}\overline z_t，\overline z_t \sim \mathcal N (0,I)$$

因此，$x_t$实际上是原始图像$x_0$和随机噪声\overline \z_t的线性组合，只要给定初始值和每一步的方差\beta_t，就可以得到任何时刻的$x_t$，把整个过程写成概率分布的形式：

$$q(x_t|x_0)=\mathcal N(x_t;\sqrt{\overline \alpha_t}x_{t-1},(1-\overline \alpha_t)I)=\mathcal N(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$$

并且当$T$足够大时，$\overline \alpha_t \to 0， 1-\overline \alpha_t \to 1，x_T \sim \mathcal N$

### 反向过程（去噪）

前向的逆操作，把前向加噪的数据再一步一步去噪，恢复出清晰干净的数据。即前向过程把$x_0$加噪变成$x_T$，反向过程把$x_T$逐步恢复到x_0。前向过程用$q(x_t|x_{t-1})$表示，反向过程则是求$q(x_{t-1}|x_t)$。假如说有办法可以实现这样操作，那就可以从一个随机高斯噪声分布$\mathcal N (0,I)$重建出一个真实的原始样本，也就是图像生成。因为这个是还原去噪，所以在细节上并不等同于原始图像。

> 思考：加噪过程是不可学习的，固定的模式，但是降噪过程是可学习的。用神经网络逐步去噪，得到$x_{T-1} \dots$最终得到没有噪声的$x_0$，那么加噪的过程实际上就是在给神经网络创造label。让网络学习什么应该降，什么不应该降。

接下来开始研究这个逆运算的计算可行性。在前向过程中每一步施加的噪声都来自特定已知的高斯分布，但是采样却有无数种可能性，所以想通过理论计算求得$q(x_{t-1}|x_t)$存在很大难度，除非无穷采样来获得极大似然估计。如果是这种思想，那么，完全可以通过一个深度网络（参数$\theta$）来拟合。

反向求解仍然是一个马尔可夫链求解，网络以当前时刻$t$和当前时刻状态$x_t$作为输入，构建反向过程条件概率，其中均值和方差都是含参估计，因此：

$$p_{\theta}(x_{t-1}|x_t) = \mathcal N(x_{t-1};\mu_{\theta}(x_t,t), \sum_{\theta}(x_t, t))$$

$$p_{\theta}(x_{0:T})=p_{\theta}(x_T)p_{\theta}(x_{T-1}|x_T)\dots p_{\theta}(x_0|x_1)=p_{\theta}(x_T)\prod \limits_{t=1}^Tp_{\theta}(x_{t-1}|x_t)$$

这个公式很好理解，在[NLP中文分词隐马尔可夫链](nlp/n2l/)部分有一个类似的例子。在这里就不展开说这个公式的意义了。

****
> 【拓展】贝叶斯

>> 贝叶斯先验概率：$P(\theta)$：根据历史经验假设事件$\theta$发生概率

>> 贝叶斯似然概率：$P(x | \theta)$：假设已经发生事件$\theta$后，发生事件x的概率；

>> 贝叶斯后验概率：$P(\theta | x)$：假设已经发生事件x后，发生事件$\theta$的概率；

>> 贝叶斯标准常量$P(x)$：不根据经验的，事件x发生的本身的概率

他们之间存在计算关系：$P(\theta | x) = \frac{P(x | \theta)P(\theta)}{P(x)}$

针对如上情景，贝叶斯计算公式如下：

$Function 1: P(A,B)=P(B|A)P(A)=P(A|B)P(B)$

$Function 2: P(A,B,C)=P(C|B,A)P(B,A)=P(C|B,A)P(B|A)P(A)$

$Function 3: P(B,C|A)=P(B|A)P(C|A,B)$

若满足马尔可夫链关系$A \to B \to C$，即当前时刻概率分布仅和上一步相关，那么可以简化：

$Function 4: P(A,B,C)=P(C|B)P(B|A)P(A)$

$Function 5: P(B,C|A)=P(B|A)P(C|B)$
****

有了这个贝叶斯的前置知识，根据公式1，可以把反向过程优化：

$$q(x_{t-1}|x_t) = q(x_t|x_{t-1})\frac{q(x_{t-1})}{q(x_t)}$$

在这里，$q(x_{t-1})$显然不知道，但是如果知道$x_0$，根据公式2，那么后验概率就是：

$$q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1}. x_0)\times q(x_{t-1}|x_0)}{q(x_t|x_0)} = \mathcal N(x_{t-1}, \tilde{\mu}(x_t,x_0),\tilde{\beta_t}I)$$

根据前向过程的结论：

$Result 1:q(x_{t-1}|x_0)=\sqrt{\overline \alpha_{t-1}}x_0+\sqrt{1- \overline \alpha_{t-1}} \overline z_{t-1}) \sim \mathcal N(x_{t-1};\sqrt{\overline \alpha_{t-1}}x_0,(1-\overline \alpha_{t-1})I)$

$Result 2:q(x_t|x_0)=\sqrt{\overline \alpha_{t}}x_0 + \sqrt{1- \overline \alpha_{t}} \overline z_{t}) \sim \mathcal N(x_{t};\sqrt{\overline \alpha_{t}}x_0,(1-\overline \alpha_{t})I)$

$Result 3:q(x_t|x_{t-1},x_0)=q(x_t|x_{t-1})=\sqrt \alpha_t x_{t-1}+\sqrt \beta_t z_t \sim \mathcal N(x_{t};\sqrt{\overline \alpha_{t}}x_{t-1},\beta_tI)$

因此，把三个结果代入反向过程$q(x_{t-1}|x_t, x_0)$计算得到：

$q(x_{t-1}|x_t, x_0)=\frac{\mathcal N(x_{t};\sqrt{\overline \alpha_{t}}x_{t-1},\beta_tI) \times \mathcal N(x_{t-1};\sqrt{\overline \alpha_{t-1}}x_0,(1-\overline \alpha_{t-1})I)}{\mathcal N(x_{t};\sqrt{\overline \alpha_{t}}x_0,(1-\overline \alpha_{t})I)}$

$\propto exp(-\frac{1}{2} (\frac{(x_t- \sqrt{\overline \alpha_{t}}x_{t-1})^2}{\beta_t}) + (\frac{(x_{t-1} - \sqrt{\overline \alpha_{t-1}}x_{0})^2}{1-\overline \alpha_{t}}) + (\frac{(x_{t} - \sqrt{\overline \alpha_{t}}x_{0})^2}{1-\overline \alpha_{t}}))$

$= exp(-\frac{1}{2} (\frac{x_t^2-2\sqrt \alpha_tx_tx_{t-1}+\alpha_t\textcolor{red}{x_{t-1}^2}}{\beta_t})+(\frac{\textcolor{red}{x_{t-1}^2}-2\sqrt \alpha_{t-1}x_0x_{t-1}+\alpha_tx_{0}^2}{1-\overline \alpha_{t}})+(\frac{(x_{t} - \sqrt{\overline \alpha_{t}}x_{0})^2}{1-\overline \alpha_{t}}))$

$= exp(-\frac{1}{2} (\frac{\alpha_t}{\beta_t} +\frac{1}{1-\overline \alpha_{t-1}})x_{t-1}^2-2(\frac{\sqrt \alpha_t}{\beta_t}x_t+\frac{\sqrt{\overline \alpha_{t-1}}}{1-\overline \alpha_{t-1}}x_0)x_{t-1}+\mathcal C(x_t,x_0))$

> 在高斯噪声里，有高斯分布$q(z) \propto exp(-\frac{(z-\mu)^2}{2\sigma^2}) = exp(-\frac{1}{2}(\frac{1}{\sigma^2})z^2 - 2\frac{\mu}{\sigma^2}z + \frac{\mu^2}{\sigma^2})$

因此，上式$\frac{\alpha_t}{\beta_t} +\frac{1}{1-\overline \alpha_{t-1}} = \textcolor{red}{\frac{1}{\tilde \beta_t^2}}，2(\frac{\sqrt \alpha_t}{\beta_t}x_t+\frac{\sqrt{\overline \alpha_{t-1}}}{1-\overline \alpha_{t-1}}x_0)=\textcolor{red}{2\frac{\tilde \mu(x_t,x_0)}{\tilde \beta_t^2}}$

第一个式子直接通分，得到:
$$\frac{1}{\tilde \beta_t^2}=\frac{\alpha_t(1-\overline \alpha_{t-1})+\textcolor{red}{\beta_t}}{\beta_t(1-\overline \alpha_{t-1})} = \frac{\alpha_t -\textcolor{green}{\overline \alpha_{t-1}\alpha_t}+\textcolor{red}{1-\alpha_t}}{\beta_t(1-\overline \alpha_{t-1})} =\frac{1-\textcolor{green}{\overline \alpha_t}}{\beta_t(1-\overline \alpha_{t-1})}$$

然后，把这个第一个式子计算好的结果，代入第二个式子：

$\tilde \mu(x_t,x_0)=(\frac{\sqrt \alpha_t}{\beta_t}x_t + \frac{\sqrt{\overline \alpha_{t-1}}}{1-\overline \alpha_{t-1}}x_0) \times \textcolor{red}{\tilde \beta^2} = (\frac{\sqrt \alpha_t}{\beta_t}x_t + \frac{\sqrt{\overline \alpha_{t-1}}}{1-\overline \alpha_{t-1}}x_0) \times \textcolor{red}{\frac{1-\overline \alpha_t}{\beta_t(1-\overline \alpha_{t-1})}}$

$= \frac{\sqrt \alpha_t(1-\overline \alpha_{t-1})}{1-\overline \alpha_t}x_t+\frac{\sqrt {\overline \alpha_{t-1}}}{1-\overline \alpha_t} \beta_t\textcolor{orange}{x_0} = \frac{\sqrt \alpha_t(1-\overline \alpha_{t-1})}{1-\overline \alpha_t}x_t+\frac{\sqrt{\alpha_{t-1}}}{1-\overline \alpha_t} \beta_t\textcolor{orange}{\frac{x_t-\sqrt{1- \overline \alpha_t}\overline z_t}{\sqrt {\overline \alpha_t}}}$

$= (\frac{\sqrt{\alpha_t}(1-\overline \alpha_{t-1})}{1-\overline \alpha_t}+ \frac{\textcolor{red}{\beta_t}\sqrt{\overline \alpha_{t-1}}}{\textcolor{purple}{\sqrt \alpha_t(1-\overline \alpha_t)}})x_t-\frac{\sqrt {\overline \alpha_{t-1}}\sqrt{1-\overline\alpha_t}\textcolor{red}{\beta_t}\overline z_t}{\textcolor{purple}{\sqrt \alpha_t(1-\overline \alpha_t)}}$

$= (\frac{\sqrt{\alpha_t}(1-\overline \alpha_{t-1})}{1-\overline \alpha_t}+\frac{1-\alpha_t} {\sqrt \alpha_t(1-\overline \alpha_t)})x_t-\frac{\beta_t\overline z_t}{\sqrt \alpha_t(1-\overline \alpha_t)}$

$= \frac{\alpha_t(1-\overline\alpha_{t-1})+1-\alpha_t}{\sqrt\alpha_t(1-\overline\alpha_t)}x_t- \frac{\beta_t\overline z_t}{\sqrt \alpha_t(1-\overline \alpha_t)}$

$= \frac{1-\alpha_t\overline\alpha_{t-1}}{\sqrt\alpha_t(1-\overline\alpha_t)}x_t- \frac{\beta_t\overline z_t}{\sqrt \alpha_t(1-\overline \alpha_t)}$

$= \frac{1-\overline\alpha_{t}}{\sqrt\alpha_t(1-\overline\alpha_t)}x_t- \frac{\beta_t\overline z_t}{\sqrt \alpha_t(1-\overline \alpha_t)}$

由于$\alpha \beta$都是确定的，因此在给定$x_0$的条件下，反向过程真实的概率分布只和$x_t$和$\overline z_t$有关，即是一个只依赖$x_0 x_t$的函数，且满足如下概率分布：

$$q(x_{t-1}|x_t,x_0)=\mathcal N(x_{t-1}, \tilde \mu(x_t,x_0),\tilde \beta_tI) = \mathcal N(x_{t-1},\frac{1-\overline\alpha_{t}}{\sqrt\alpha_t(1-\overline\alpha_t)}x_t- \frac{\beta_t\overline z_t}{\sqrt \alpha_t(1-\overline \alpha_t)},\frac{1-\overline \alpha_{t-1}}{1-\overline \alpha_t}\beta_tI)$$

### 优化目标

扩散模型的目标是尽可能得到真实的$x_0$，即求得上文说到的$\theta$参数，使得最终得到$x_0$的概率最大，这是一个极大似然估计的问题，似然函数如下：

$$p(x_0|\theta)=p_{theta}(x_0)=\int_{x_1}\int_{x_2}\int_{x_3}\dots\int_{x_T}q(x_{1:T|x_0})\frac{p_{\theta}(x_0,x_1,\dots.x_T)}{q(x_{1:T|x_0})}d_{x_1}d_{x_2}\dots d_{x_T}=\mathbb E_{q(x_1:T|x_0)} [ \frac{p_{\theta}(x_{0:T})}{q(x_{1:T|x_0})} ] $$

之后的推导非常繁琐，可以查看[DDPM](https://arxiv.org/abs/2006.11239)原文，简单来说，由Jensen不等式可知，任一凸函数$f$始终满足函数值的期望大于等于期望的函数值，取负对数似然函数可以得到一个分变不等式。这里期望的函数就是变分下界。结合马尔可夫贝叶斯公式展开变分下界会发现中间的一串其实是KL散度的计算公式，只要固定方差就可以进一步简化KL散度。最后省略无关系数得到最终优化函数：
$$L_{t-1}^{simple}=\mathbb E_{x_0, \epsilon \sim N(0,I)}[\Vert \epsilon_{\theta}(\sqrt{\overline \alpha_t}x_0+\sqrt{1-\overline \alpha_t}\epsilon.t) \Vert ^2]$$
可以看出，在训练DDPM时，只要用一个简单的MSEloss来最小化前向过程施加噪声分布和后向过程预测的噪声分布，就能实现最终优化目标。

## 扩散模型的应用
- 计算机视觉：图像分割与目标检测、图像超分辨率（串联多个扩散模型）、图像修复、图像翻译和图像编辑。
- 时序数据预测：TimeGrad模型，使用RNN处理历史数据并保存到隐空间，对数据添加噪声实现扩散过程，处理数千维度德多元数据完成预测。
- 自然语言：使用Diffusion-LM可以应用在语句生成、语言翻译、问答对话、搜索补全、情感分析、文章续写等任务中。
- 基于文本的多模态：文本生成图像（DALLE-2、Imagen、Stable Diffusion）、文本生成视频（Make-A-Video、ControlNet Video）、文本生成3D（DiffRF）
- AI基础科学：SMCDiff（支架蛋白质生成）、CDVAE（扩散晶体变分自编码器模型）