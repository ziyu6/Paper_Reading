# Paper_Reading
主要阅读TTS、VC、LLM、ML、Multimodel相关方向的论文，努力周更！

## **TTS方向：**

### Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS):
<mark>一句话解读:一个很经典的效果非常好的端到端语音合成“小”模型，主要应用**条件变分推理**和**归一化流**，另外还提出了一个**随机时长预测器**。
主要贡献点就是第一个端到端的效果非常好的tts模型。<mark>
#### 一、Abstract:
1. 当前已有的端到端的tts模型生成效果不如传统的两阶段模型，而本文的end to end模型可以生成比现有的两阶段模型更自然的语音
2. 模型采用变分推理，加上归一化流（flow)和对抗训练过程。
3. 同时还提出了一个随机时长预测器用于从输入文本中合成具有不同韵律的语音。通过对隐变量（z)的不确定性建模和随机时长预测器，我们的方法可以表达自然的一对多关系（一段文本可以用不同的音高和节奏朗读）
4. 我们的模型实现了与真值相当的MOS（mean opinion score),并优于目前最好的tts系统
#### 二、Introduction
1. 二阶段的模型中的两个阶段是独立训练的，而基于神经网络的自回归tts系统因为其顺序生成的特点导致速度很慢。目前已有一些非自回归的方法，另外，GAN网络能够合成高质量的音频
2. 本文中，我们的贡献:
  1. 使用VAE，通过隐变量z将tts的两个模块连接起来。
  - 将归一化流应用于波形域上的条件先验分布和对抗训练
  - 提出了一种随机时长预测器, 使得模型更加自然得表达一对多的关系（文本输入可以通过多种方式以不同的变化（例如音高和持续时间）朗读。）
#### 三、Method:
1. Variational Inference 变分推断：
![image](https://github.com/user-attachments/assets/568913b7-7859-4701-bf94-442db96442bd)
![image](https://github.com/user-attachments/assets/5b8e3834-5368-45d8-ba82-cd919bdedb49)
2. 训练loss = 重建loss + KL散度loss
3. 重建loss（mel loss)：作为重建损失中的目标数据点，我们使用梅尔频谱图而不是原始波形，用 xmel 表示。我们通过解码器将潜在变量 z 上采样到波形域 ˆ y，并将 ˆ y 变换到梅尔谱图域 ˆ xmel。然后将预测梅尔谱图与目标梅尔谱图之间的 L1 损失用作重建损失：
4. KL loss：KL散度(Kullback-Leibler Divergence)是用来度量两个概率分布相似度的指标
5. 时长预测器：从文本预测持续时间
6. 对抗训练：添加一个鉴别器D用于区分解码器G生成的输出和真值波形y
7. 总loss:
![image](https://github.com/user-attachments/assets/e8a59d20-85da-4b30-aeec-016b3346b804)

#### 四、Model Architexture：
1. 所提模型的整体架构由后验编码器、先验编码器、解码器、判别器和随机持续时间预测器组成。后验编码器和判别器仅用于训练，不用于推理
2. 后验编码器：对于后验编码器，我们使用 WaveGlow和 Glow-TTS中使用的非因果 WaveNet 残差块。
3. 先验编码器：由处理因素的text enocoder和提高先验分布灵活性的归一化流flow共同组成，
4. decoder：本质上是HiFi-GAN V1 生成器
5. 鉴别器：遵循 HiFi-GAN 中提出的多周期鉴别器的鉴别器架构
6. 随机持续时间预测器：随机持续时间预测器根据条件输入 htext 估计音素持续时间的分布
![image](https://github.com/user-attachments/assets/f44bc1ad-cb4a-4108-8253-5024a772d230)

#### 五、Experiment:
1. 实验结果表明，
1）随机持续时间预测器比确定性持续时间预测器生成更真实的音素持续时间
2）我们的端到端训练方法是一种有效的方法，即使保持相似的持续时间预测器架构，也可以比其他TTS模型产生更好的样本。

#### 六、问题：
1. 变分VAE的优点是可以生成具有多样性的语音，因为隐变量 z 是随机采样的，不同的 z 可以生成不同的音调和韵律
2. KL loss的计算：
![image](https://github.com/user-attachments/assets/121c1939-0368-4571-b0fe-7484bf81b239)


