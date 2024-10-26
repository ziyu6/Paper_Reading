# Paper_Reading
主要阅读TTS、VC、LLM、ML、Multimodel相关方向的论文，努力周更！

## **TTS方向：**

### 1. Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS):
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



## **Multi-modal：**
### 1. PIRenderer：Controllable Portrait Image Generation via Semantic Neural Rendering
1. 通过 语义神经渲染 生成 可控肖像图像
2. 通过控制现有面部的运动来生成肖像图像是对社交媒体行业具有重大影响的一项重要任务。为了易于使用和直观控制，应使用语义上有意义且完全解缠的参数作为modifications。然而，许多现有技术不提供如此细粒度的控制或使用间接编辑方法，如：模仿其他个体的动作。在本文中，提出了一种 Portrait Image Neural Renderer（PIRenderer），利用三维可变形面部模型（3DMM）的参数来控制面部运动。所提出的模型可以根据直观的修改生成具有准确运动的逼真的肖像图像。直接和间接编辑任务的实验证明了该模型的优越性。同时，我们进一步扩展该模型，通过从音频输入中提取顺序运动来处理音频驱动的面部重演任务。我们证明，我们的模型可以仅从单个参考图像和驱动音频流生成具有令人信服的动作的连贯视频。我们的源代码可在 https://github.com/RenYurui/PIRender 获取。
3. 如下图，PIRender可以生成逼真的人像图片根据：tagert motion / target image / target audio
![image](https://github.com/user-attachments/assets/b732ede0-1737-45b7-9ec6-f8baa2b63965)
4. 目前大多数方法使用 间接 或 特定于某一主题的动作描述符，阻碍了模型以 直观 方式编辑肖像的能力。为了实现直观的控制，运动描述符应该具有语义意义，这需要将面部表情、头部旋转和平移表示为完全解开的变量。参数化人脸建模方法提供了使用语义参数描述 3D 人脸的强大工具。这些方法允许控制 3D 网格的形状、表情等参数。结合这些技术的先验，人们可以期望控制类似于图形渲染处理的照片级真实肖像图像的生成。
5. 本文提出了一种神经渲染模型PIRenderer。给定源肖像图像和目标 3DMM 参数，我们的模型会生成具有准确运动的照片般真实的结果。所提出的模型分为三个部分：the Mapping Network, the Warping Network, and the Editing Network。映射网络 通过目标的动作描述符p 生成 隐层向量z ; 在向量z的指导下，变形网络 估计源和期望目标之间的变形，并通过用 估计的变形扭曲 源图像 来生成粗略结果。编辑网络 从粗略图像生成最终图像。如下图：
![image](https://github.com/user-attachments/assets/cd4d10af-a9e9-4141-a294-0656a9423f0f)
6. 现有的三种方式根据控制信号的不同分为：
  - Portrait Editing via Semantic Parameterization：使用类似于计算机动画控件的语义控制空间来编辑肖像图像可以为用户提供直观的控制,X2face,StyleGAN,StyleRig,PIE ....
  - Portrait Editing via Motion Imitation: 不用语义参数来描述目标运动，而是模仿另一个人的运动。這些模型依赖于特定主题（例如地标、边缘、解析图）或运动纠缠（例如稀疏关键点）描述符，这使得他们缺乏直观编辑源肖像的能力
  - Portrait Editing via Audio： 使用音频编辑肖像图像需要从音频流和源图像生成具有令人信服的动作的连贯视频。
7. Target Motion Descriptor：在本文中，我们采用 3DMM 参数的子集作为运动描述符。对于 3DMM，面部的 3D shape S 参数化为：其中Bid代表identiy,Bexp代表expression
![image](https://github.com/user-attachments/assets/44e1a514-694e-4c7a-bbf6-c37ca8007a5a)
头部旋转和平移表示为 R ∈ SO(3) 和 t ∈ R3。通过参数集 pi == {βi, Ri, ti}，可以清楚地表达第 i 面的所需运 动。
8. Mapping network: 首先，采用映射网络 fm : P → Z 从运动描述符 p ∈ P 生成潜在向量 z ∈ Z。即：z = fm(p)；接著，学习到的潜在向量 z 通过仿射变换进一步变换，以生成控制自适应实例归一化 (AdaIN) 操作的 y = (ys, yb)。 即：AdaIN(xi, y) = ys,i * [xi − μ(xi) / σ(xi)] + yb,i  每个特征图 xi 首先被归一化，然后使用 y 的相应标量分量进行缩放和偏置。
9. Wraping network: 为了更好地保留生动的源纹理并实现更好的泛化，我们使用扭曲网络 gw 对源图像 Is 的重要信息进行空间变换。它以源图像 Is 和潜在向量 z 作为输入并生成flow field w 即：w = gw(Is, z)；当得到w后，可以利用 Iˆw = w(Is) 计算得到 粗略结果; 通过输入的source图片和生成的粗略图像结果计算loss
![image](https://github.com/user-attachments/assets/92ff4679-52e4-4b2c-9a70-21e93835fbc5)
10. Editing Network：扭曲操作引入的伪影将导致性能下降。因此，设计了一个编辑网络 ge 来修改扭曲的粗略结果 I ˆw。编辑网络以 I ˆw、Is 和 z 作为输入并生成最终预测 ˆI: Iˆ = ge(ˆIw, Is, z)。 重建loss通过计算最终预测结果^I与target 图像It之间的loss得到；
11. Extension on Audio-driven Reenactment :将音频信号直接映射到图像或其他运动描述符（例如边缘、地标）具有挑战性。与运动无关的因素（例如身份和照明）会干扰模型。因此，采用语义上有意义的参数（例如 3DMM）作为中间结果可以显著简化任务。因此，我们通过加入额外的 映射函数 fθ 来进一步改进我们的模型，以从音频生成连续的 3DMM 系数。采用归一化流来设计该模型。归一化流的核心思想是训练一个可逆、可微的非线性映射函数，将样本从简单分布映射到更复杂的分布。使用audio序列+condition向量+前面预测出来的motion descriptors向量序列，通过归一化流预测当前帧的pi
![image](https://github.com/user-attachments/assets/d0dfcac0-bed0-43fc-b301-c1346b0b2d23)
12. 实验：StyleRig 生成了具有真实细节的令人印象深刻的结果。然而，它倾向于采用保守策略生成图像：为了获得更好的图像质量，远离配送中心的运动会被削弱或忽略。同时，一些与运动无关的因素（例如眼镜、衣服）在修改过程中发生了变化。我们的就没有
![image](https://github.com/user-attachments/assets/ab126dd2-f71e-47c8-8c92-5dc3897e5ab5)
13. 我们的模型比DAVS能生成更丰富的头部运动和嘴部细节：
![image](https://github.com/user-attachments/assets/42e6418a-9e70-45a1-b8ae-06320644644f)
14. 我们提出了 PIRenderer，一种高效的肖像图像神经渲染器，能够通过语义上有意义的参数来控制面部。结合 3DMM 的先验，我们的模型可以通过根据用户指定的系数修改面部表情、头部姿势和平移来对现实世界的肖像图像进行直观的编辑。同时，它还可以执行动作模仿任务。在与主体无关的运动描述符的指导下，该模型可以生成具有维护良好的源身份的连贯视频。我们相信，通过灵活的图形控制生成神经网络可以实现许多令人兴奋的应用。音频驱动的面部重演任务的扩展提供了一个示例，并显示了这种组合的潜力。  
