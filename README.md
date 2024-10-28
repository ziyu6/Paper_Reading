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




### 2.BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data
<mark>一句话解读:一个大模型：数据量大（100kh）、参数量大（10亿参数），并设计了一个测试集衡量模型涌现能力（通过提升模型参数规模和训练数据规模，测试模型是否具有更好的性能）<mark>
#### Abstract:
1. 通过100K 小时的数据 构建 10亿参数的TTS
2. We introduce a text-to-speech (TTS) model called BASE TTS, which stands for Big Adaptive Streamable TTS with Emergent abilities.   大的、自适应的、流式的、具有涌现能力的TTS
3. 它部署了一个 10 亿参数的自回归 Transformer，将原始文本转换为离散代码（“"speechcodes"”），然后用一个基于卷积的解码器，以增量、流式传输的方式将这些speechcodes转换为波形。
4. 设计了一个专门的测试集来衡量TTS的emergent 能力。

#### Introduction：
1. BASE TTS follows the approach of casting TTS as a next-token-prediction problem, inspired by the success of LLMs。这种方法通常与大量训练数据结合应用，以实现强大的多语言和多说话人能力.
2. 我们的目标是提高总体 TTS 质量，并研究 缩放定律 如何影响模型为具有挑战性的文本输入生成适当韵律和表达的能力，类似于LLM 如何通过数据和参数 缩放 获得新能力，也就是“涌现”能力。
3. 主要贡献： 
4. BASE TTS是目前最大的TTS模型，且主观评估胜过其他baseline
5. 提出一个测试集，可以作为大规模 TTS 模型的文本理解的主观评估基准，并且在该测试集上对不同参数规模的BASE TTS做了实验，验证了随着数据集大小和参数数量的增加，质量单调提高。（即： scaling 对 emergent 能力的影响）
6. 引入一种新型的 离散语音表示方式， 能够只capture 音素和韵律信息，且在非常高的压缩（量化）等级仍能还原回高质量音频。
#### Model：
1. 模型架构：
![image](https://github.com/user-attachments/assets/95746ca7-1b7f-49f5-b71d-7b5c54f5a207)
模型主体由一个transformer-based 自回归模型构成，其输入由三部分组成：音素序列（text  embedding) + 对应的speech codes(由speech tokenizer提取得到的） + 目标说话人audio经过reference encoder得到的speaker embedding， 模型自回归的生成对应的 具有目标说话人音色的 speech codes， 再通过decoder还原为语音波形

2. WaveLM-based  speech tokenizer:（上图中的第一个结构）
我们的目标是开发包含语音和韵律信息的语音代码，但与说话者身份、录音条件和音频信号中的其他特征无关。为此，我们引入了一种基于从预训练的 WavLM 模型 中提取的特征的 语音标记器，并通过鼓励解缠说话者身份的损失进行了进一步训练。
如下图，首先将audio通过WavLM model得到提取的hidden states，接着将隐层特征分别通过content回归器和speaker回归器，都通过一个encoder，接着对content相关特征过vq离散化，对speaker相关特征过speaker extractor获得对应的speaker embedding，做了三个loss
(1) 把content相关特征和speaker embedding concat之后过decoder得到的频谱与GT做 重建Loss 
(2) 对 speaker embedding做 对比Loss : 最大化来自同一说话人的样本之间的相似性，并最小化来自不同说话人的样本之间的相似性
(3) 将 content 相关特征 通过冻结speaker extractor的网络梯度回传得到的embedding与speaker embedding做余弦相似度loss, 可以理解为content 特征 与 speaker 特征越远离，说明说话人音色解缠的越好
![image](https://github.com/user-attachments/assets/3dcc20b3-3c1d-4f65-b313-b7f22e295f4d)
3. Autoregressive speech modeling (SpeechGPT)
主体的自回归模型 是一个 GPT2 架构自回归模型，从头训的，就是基础的gpt2，没啥改变
4. Decoder部分：
主要的改进是：解码器不是将语音代码作为输入，而是将自回归 Transformer 的最后一个隐藏状态作为输入。我们这样做是因为密集的潜在表示提供了比单个语音代码更丰富的信息
![image](https://github.com/user-attachments/assets/cc9d1ffb-fc2e-48e7-af29-4ece9689a269)

#### Experiment:
1. 我们创建了一个由 10 万小时未标记的公共领域语音数据组成的数据集。整体数据集以英语数据为主（超过 90%），其次是德语、荷兰语、西班牙语。
2. 为了验证缩放定律对模型性能的影响，做了三个不同大小的模型：
![image](https://github.com/user-attachments/assets/39c35cec-bfdc-409c-b9d7-677acea50f7c)
3. 结果如下：基本上都是随着模型越大效果越好
![image](https://github.com/user-attachments/assets/048f64aa-fa0d-4f71-9e1e-d1d6fa799662)

#### Conclusion
1. 我们引入了 BASE TTS，这是一种 GPT 风格的 TTS 系统，使用新颖的基于 SSL 的 speechcode 作为中间表示和解码器。无论是在参数还是训练数据方面，这都是我们所知的同类模型中最大的。我们根据包括 Tortoise、Bark 和 YourTTS 在内的基线展示了最先进的新 TTS 结果。
2. 我们的方法指向 LTTS 模型的潜在缩放定律 [92]，其中需要更大量的语音和其他（文本、图像）数据来支持多模态目标 [93] 并在 TTS 中开辟新天地。我们的方法仍然存在一些局限性：a) BASE TTS 有时会产生幻觉和中断，我们会产生超出文本预期的额外或不完整的音频。这是自回归 LM 方法的一个固有问题，； b) 为 GPT 式 TTS 选择正确的离散表示至关重要。需要更多的研究来确定语音代码的不同属性如何转化为端到端系统质量。






## **音乐合成：**
### 1. Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer
<mark>一句话解读:使用旋律和文本编辑音乐：使用 ControlNet 进行 Diffusion Transformer<mark>
1. 使用梅尔频谱图表示和基于 UNet 的模型结构，生成音乐的质量和长度仍然存在挑战。
2. 为了解决这些限制，我们提出了一种使用 Diffusion Transformer（DiT）的新方法，并通过 ControlNet 增加了一个额外的控制分支。
3. 引入了一种新颖的 top-k constant-Q Transform representation as melody prompt,，与以前的表示相比减少了歧义。
4. 为了有效平衡文本和旋律提示的控制信号，我们采用了逐步mask melody prompt 的课程学习策略，从而使训练过程更加稳定。
5. Top-k CQT：扩展 CQT以获得前 k 个最突出的音高值，提出了更精确和灵活的旋律表示。对于立体声音频输入，我们首先计算左声道和右声道的 128 个 bin 的 CQT，然后应用 argmax 运算以保留每个声道中每帧的 4 个最突出的音高。
![image](https://github.com/user-attachments/assets/73702248-ee10-4a0a-a52d-7bd7cae0fb27)
6. 渐进式课程屏蔽策略：过于精确的旋律提示可能会将附加的音乐元素（例如音色）合并到条件信息中。另一方面，由于旋律提示在时间上与目标音频对齐，因此模型更有可能学习旋律提示和音频之间的直接关系，而不是依赖于高级文本提示。最初，所有旋律提示都被屏蔽，允许模型学习用空旋律提示生成音乐。随着训练的进行，音级方向的掩码比率逐渐减小，使模型能够逐步学习合并旋律提示。在初始全掩蔽阶段之后，保留 top-1 旋律提示，而 top-2、top-3 和 top-4 旋律提示则被随机掩蔽和打乱。
7. DiT with ControlNet: ControlNet-Transformer 复制前 N 个 Transformer 块作为控制分支。第 i 个可训练复制块的输出在通过附加的零初始化线性层后，与第 i 个冻结块的输出相结合，并用作第 (i + 1) 个冻结块的输入。本设计保持了DiT模型原有的连接结构，实现了ControlNet结构的无缝集成，同时保留了Transformer结构的核心优势。
![image](https://github.com/user-attachments/assets/3a09e325-28ee-4f60-ac26-62073a47f756)

8. 实验：客观指标表明，该模型在text to music和music editing任务上均强于baseline MusicGen
![image](https://github.com/user-attachments/assets/4db63f1d-a770-4d5a-b0a3-54e5a8d1a949)

9. 表 II 列出了文本音乐一致性和音频质量的平均 MOS 分数。主观结果清楚地表明，我们的模型在这两项任务中都优于 MusicGEN，在音乐编辑任务中优势尤其显着。
![image](https://github.com/user-attachments/assets/6c5192a2-e22a-4e11-9d88-ca7595964df7)

10. 在本文中，我们提出了一种新颖的音乐生成和编辑方法，旨在支持长格式和可变长度的音乐以及文本和旋律提示。为了实现这一目标，我们将 ControlNet 集成到 DiT 结构中，并引入一种新颖的 top-k CQT 表示作为旋律提示，提供更精确、更细粒度的旋律控制。主观和客观结果都表明，我们的模型在保持强大的文本到音乐生成能力的同时，在音乐编辑方面表现出色，并在文本和旋律提示之间实现了更加平衡的控制。







## **Multi-modal：**
### 1. PIRenderer：Controllable Portrait Image Generation via Semantic Neural Rendering
<mark>一句话解读:这篇是之前做audio driven talking head的时候看的，主要是用了3dmm参数控制面部表情、头部姿势和平移等。
我可借鉴的点就是3dmm参数＋说话时嘴部用流式audio控制<mark>
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


### 2.Audio-visual Generalized Zero-shot Learning the Easy Way
<mask>视听泛化零样本学习，理解视频中音频和视觉线索之间的复杂关系。<mask>
1. 视听泛化零样本学习是一个快速发展的领域，旨在理解视频中音频和视觉线索之间的复杂关系。总体目标是利用从可见类中获得的insight，从以前未见过的实例中识别实例.
2. Audio-visual learning：视听学习问题，从视频中学习两个不同模态之间的视听相关性。这样的跨模态对齐有利于各种视听任务。在这项工作中，我们的主要重点是学习与文本嵌入对齐的音视频表示，用于广义零样本学习，这是一个比前述任务提出更多挑战的任务。
3. 先前的方法主要利用synchronized auto-encoders同步自编码器来重建视听属性，这些属性是由交叉注意力transformers和投影文本嵌入提供的。然而，这些方法不能有效地捕获预训练语言对齐嵌入中固有的跨模态特征和类标签嵌入之间的复杂关系。
4. 为了克服这些瓶颈，我们提出了一个简单而有效的Easy Audio-Vis Generalized零样本学习框架，命名为EZ - AVGZL，它将音视频嵌入与转换后的文本表示进行对齐。它利用单个有监督的文本视听对比损失来学习视听和文本模态之间的对齐，摆脱了传统的重建跨模态特征和文本嵌入的方法。
5. 性能对比：
![image](https://github.com/user-attachments/assets/3db3dffd-c6f5-46bf-aa52-6e24fce63576)
6. Method:
![image](https://github.com/user-attachments/assets/e23b2354-bb3e-4fd5-a7f2-317276075afa)
7. 文本输入经过冻结的text encoder得到初始class embedding ti, ti通过最大可分离性和保留语义优化得到wi；另一边，video和audio分别通过video encoder和audio encoder得到相应的embedding，交叉注意力变换模块从单模态编码器中提取视觉和音频特征(vi , ai)，生成多模态表示。最后，用一个非线性相似函数被对齐表示x和相应的类嵌入wi。对于同一类别：最小化相似度score和1之间的距离；对于不同类别，最小化相似度score和0之间的距离。
8. 实验：VGGSound-GZSL，UCF-GZSL，ActivityNet-GZSL 
9. 与以前的视听广义零样本学习方法相比，我们在可见类和未见类的所有指标中都取得了最好的结果。
![image](https://github.com/user-attachments/assets/20009164-8ae3-4f7d-a153-d42d777358f0)
![image](https://github.com/user-attachments/assets/85a34379-2cfa-4fef-8330-7f05a33e04ec)
![image](https://github.com/user-attachments/assets/f47deab3-3140-445a-939b-c9247a27e851)

在这项工作中，我们提出了一个新颖而有效的框架EZ - AVGZL，它将视听表示与优化的文本嵌入对齐，以实现视听广义零样本学习。我们利用差分优化来学习具有最大可分性和语义保持性的更好分离的文本表示。此外，我们引入有监督的视听语言对齐来学习视听特征和文本嵌入之间的对应关系，从而捕获以类标签为提示的跨模态动态。在VGGSound - GZSL、UCF - GZSL和ActivityNet - GZSL数据集上的实验结果表明了本文方法相对于以往基线的优越性。广泛的消融研究也验证了类嵌入优化和超级的重要性

### 3. Faces that Speak: Jointly Synthesising Talking Face and Speech from Text
1. 会说话的面孔：从文本中联合合成会说话的面孔和语音
2. 这项工作的目标是同时从文本生成自然的说话面孔和语音输出。我们通过将人脸生成 (TFG) 和文本转语音 (TTS) 系统集成到一个统一的框架中来实现这一目标。我们解决每项任务的主要挑战：（1）生成一系列代表现实世界场景的头部姿势，以及（2）尽管同一身份的面部运动存在变化，但仍确保语音一致性。为了解决这些问题，我们引入了一种基于条件流匹配的运动采样器，它能够以有效的方式生成高质量的运动代码。此外，我们还为 TTS 系统引入了一种新颖的调节方法，该方法利用 TFG 模型中的运动消除特征来产生均匀的语音输出。我们广泛的实验表明，我们的方法可以有效地创建自然的说话面孔和语音，并与输入文本精确匹配。据我们所知，这是构建一个可以泛化到不可见身份的多模态合成系统的第一次努力。
![image](https://github.com/user-attachments/assets/8f63aee1-ea25-436f-bce7-7b3dee05732e)

### 4.FaceVerse: a Fine-grained and Detail-controllable 3D Face Morphable Model from a Hybrid Dataset
FaceVerse：来自混合数据集的细粒度和细节可控的 3D 人脸可变形模型
1. 我们展示了 FaceVerse，这是一种细粒度的 3D 神经面部模型，它由混合东亚面部数据集构建而成，其中包含 60K 融合的 RGB-D 图像和 2K 高保真 3D 头部扫描模型。提出了一种新的从粗到细的结构，以更好地利用我们的混合数据集。在粗模组中，我们从大尺度RGB-D图像中生成了一个基础参数模型，该模型能够预测不同性别、年龄等的准确粗略3D人脸模型。然后，在精细模块中，引入了使用高保真扫描模型训练的条件 StyleGAN 架构，以丰富精细的面部几何和纹理细节。请注意，与以前的方法不同，我们的基础模块和详细模块都是可变的，这使得调整 3D 面部模型的基本属性和面部细节的创新应用成为可能。此外，我们提出了一种基于可微分渲染的单图像拟合框架。丰富的实验表明，我们的方法优于最先进的方法

### Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set 
弱监督学习的精确 3D 人脸重建：从单图像到图像集
#### Abstract:
1. 最近，基于深度学习的 3D 人脸重建方法在质量和效率上都显示出了可喜的结果。然而，训练深度神经网络通常需要大量数据，而具有真实 3D 人脸形状的人脸图像却很少。在本文中，我们提出了一种新颖的深度 3D 人脸重建方法，该方法 1）利用鲁棒的混合损失函数进行弱监督学习，同时考虑低级和感知级信息进行监督，2）执行多重处理通过利用来自不同图像的互补信息进行形状聚合来重建图像人脸。我们的方法快速、准确，并且对遮挡和大姿势具有鲁棒性。我们对三个数据集进行了全面的实验，系统地将我们的方法与十五种最新方法进行比较，并展示了其最先进的性能。
#### Introduction:
1. 无监督学习的关键是可微分图像形成过程，它通过网络预测渲染人脸图像，而监督信号来源于输入图像和渲染对应图像之间的差异。
2. 针对一张图像：提出了一种混合水平损失函数，将图像级别和感知级别的loss 结合起来。我们还提出了一种新颖的基于肤色的光度误差注意策略，使我们的方法对遮挡和其他具有挑战性的外观变化（例如胡须和浓妆）具有进一步的鲁棒性。
3. 针对多张图像：以无监督的方式从多个图像中学习 3D 人脸聚合。我们训练一个简单的辅助网络来生成带有身份的回归 3D 模型系数的“置信度分数”，并通过基于置信度的聚合获得最终的身份系数。尽管没有使用明确的置信度标签，但我们的方法会自动学习支持高质量（尤其是高可见度）的照片。此外，它可以利用姿势差异来更好地融合互补信息，学习更准确的 3D 形状。
#### Preliminaries: Models and Outputs
1. 如下图，框架由用于端到端单图像 3D 重建的 重建网络 和为基于多图像的重建而设计的 置信度测量子网 组 成。
![image](https://github.com/user-attachments/assets/52451f5f-781c-4e38-b8e5-de5197c7df79)
2. 使用 3DMM，脸型 S 和纹理 T 可以用仿射模型表示：Bid、Bexp 和 Bt 分别是identity、expression和texture的 PCA bases，
![image](https://github.com/user-attachments/assets/215ea60a-7732-4212-9e30-52e04a11e8bf)
3. Training pipeline for single image 3D face reconstruction：给定一个训练的RGB图像 I，我们使用R-Net回归一个系数向量 x，用它可以通过一些简单的、可微的数学推导来解析生成重建的图像 I′。，计算I与I'之间的混合loss。
![image](https://github.com/user-attachments/assets/37cd1864-766a-4655-8021-77f7f4bc5db3)
4. Training pipeline for multi-image 3D face reconstruction with shape aggregation：
![Uploading image.png…]()














