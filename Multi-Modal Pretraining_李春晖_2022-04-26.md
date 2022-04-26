# **多模态预训练**

## **预备知识**

### **多模态预训练基本任务介绍**

这里介绍的是常见的，广泛使用的任务，后面不同的工作中又会有不同的改进和创新

#### **Cross-Modal Masked Language Modeling（MLM）**

Cross-modal MLM与BERT中的MLM相似，在Cross modal MLM中，模型基于没有被mask掉的token和视觉特征同时来预测mask掉的token

#### **Cross-Modal Masked Region Prediction（MRP）**

这种方法首先用Faster RCNN等实体检测工具，抽取出一幅图中不同的实体，抽取出来的实体也叫Region，该任务随机mask图片中的Region（一般是将该region像素值置为0），通过其他region和文本一起预测该region，这里预测又分为两种：

##### **Masked Region Classification（MRC）**

这是预测被mask掉的region的类别（为什么会有类别？因为如Faster RCNN这样的实体检测工具在抽取出实体的同时还会附带该实体的类别标签）

**Masked Region Feature Regression（MRFR）**

这里直接对被mask掉的region的像素值做回归

#### **Image-Text Matching（ITM）**

上面的Cross-modal MLM和MRP帮助预训练模型学习细粒度的image和text之间的关系，而ITM通过粗粒度地对齐图文来赋予模型能力。

具体地，构建匹配的图片文本正样本对和不匹配的图文负样本对，该任务让模型预测图文匹配关系

### **不同图片编码器的特点**

多模态预训练任务中，图片端常用两种类型的编码器：Resnet和Faster RCNN,他们有不同的特点：

- Faster RCNN
  - 可以抽取图片中包含的实体以及对应的实体类别（可以手工设定抽取个数或抽取置信度）
- Resnet
  - CNN一族的编码器，Resnet-152对整张图编码成（49，2048）的向量

### **多模态领域常见的下游任务**

**Cross Modal Matching**

- Image Text Retrieval（ITR）：给定一句话，搜索与该句子match的image；或是给定一张图，搜索与该图match的sentence

- Visual Referring Expression（VRE）：这个任务的目标是根据相应的特定的文本描述localize图片中的region

**Cross-Modal Reason**

- Visual Question Answering (VQA)：给定一张图片和一个与该图片相关的自然语言问题，计算机能产生一个正确的回答
- Natural Language for Visual Reasoning (NLVR)：这个任务给一对图片和一句语言描述，判断该语言描述对这对图片的描述是否正确
- Visual Commonsense Reasoning（VCR） 这个任务是VQA的一个特例

**Vision and Language Generation**

- Text-to-Image Generation. 给定描述文本，生成相应图片
- Multimodal Text Generation 
  - image caption
  - 多模态机器翻译

## **文章介绍（按时间顺序）**



### **2019**

这几篇文章相对比较基础

#### **VL-BERT: PRE-TRAINING OF GENERIC VISUAL-LINGUISTIC REPRESENTATIONS**

时间：2019.08

模型名称：VL-BERT

#### ![img](https://static.dingtalk.com/media/lALPDe7s3ik0ZLTNAaTNAtA_720_420.png)

该模型扩展了BERT，具体地：

- Token Embedding
  - 文本后面拼接了[IMG]字段
- Visual Feature Embedding
  - 文本token对应位置输入原图，[IMG]字段输入通过Faster RCNN抽取出来的实体

做两个任务：

- MLM：如图中文本token处被[MASK]掉，根据文本和图像region填空
- MRC：图中蓝色斜线盖住的region，根据其他文本和图像region判断盖住的区域的类别

下游任务设置：VCR、VQA



**LXMERT: Learning Cross-Modality Encoder Representations from Transformers**

时间：2019-08

模型名称：LXMERT

同期工作，不同点在于：

- 额外用一个自注意力机制建模Faster RCNN抽取出来的不同region的关系
- 额外预训练任务Image Question Answering(QA)：如果问题和图片相对应，则由模型给出问题的答案，判断答案的正误。



**VISUALBERT: A SIMPLE AND PERFORMANT BASELINE FOR VISION AND LANGUAGE**



时间：2019-08

模型名称：VisualBERT



同期工作，和VL-Bert的区别很小：

- 由token-enbedding和feature-embedding共同组成了一个embedding层
- position编码层被用于进行对齐

做两个任务：MLM、ITM

之后再在Task-Specific任务上再做预训练(有点像打比赛的操作)，最后对应下游任务fine-tune

下游任务：VQA、VCR、NLVR、VRE



**ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks**

![img](https://static.dingtalk.com/media/lALPDfmVWhfHn_3NAQ3NBSM_1315_269.png)

时间：2019-08

模型名称：ViLBERT

同期工作，两模态通过co-attention交互，做两个任务：MLM、ITM

下游任务：VQA、VCR、ITR、VRE



**Unified Vision-Language Pre-Training for Image Captioning and VQA**

时间：2019-09

模型名称：Unified VLP

动机：上面提到的预训练模型都只能处理Visual understanding任务，但是没有能力处理Visual generation任务（比如生成image caption）

这个模型：

- 既可以做understanding，又可以generation
- encoding和decoding共享一个transformer

模型结构

![img](https://static.dingtalk.com/media/lALPDfYH23MZ747NA-vNBws_1803_1003.png)

预训练的任务：

- bidirectional masked vision-language prediction
- seq2seq masked vision-language prediction

下游任务：image caption生成



**UNITER: UNiversal Image-TExt Representation Learning**

时间：2019.12

模型名称：UNITER

创新点：

- 提出conditional masking
  - 动机是之前有的工作做随机mask的时候，可能同时mask掉恰好有对应关系文本和图像region，这样学不到这部分对应关系，条件mask就是说每次只mask一个模态，要么mask文本，要么mask visual region
- 提出新任务Word Region Alignment（还没看懂）
  - 这个任务是希望word能与图片中抽取的region建立对应关系
  - 使用最优化传输Optimal Transport来做这个任务
- UNITER在六个VL任务上做的好，包括Visual Question Answering, Image-Text Retrieval, Referring Expression Comprehension, Visual Commonsense Reasoning, Visual Entailment, and NLVR



### **2020**

**InterBERT: Vision-and-Language Interaction for Multi-modal Pretraining**

时间：2020.03

模型名称：InterBERT

这篇文章主要创新点在于设计了更难的预训练任务：

- **Masked Segment Modeling**：按照 10% 的概率挑选文本 mask 的锚点，随后以锚点开始 mask 连续的单词，平均长度为3，最后预测被 mask 的单词。

- 我们随机以10%的概率选择word作为masking anchors，然后mask掉该锚点之后的0到2个词
- **Masked Region Modeling**：按照 15% 的概率挑选图像 RoI mask 的锚点，随后将被选中的 RoI 以及与该 RoI 重叠 IOU 值大于 0.4 的 RoIs 一起 mask 为全 0 向量，预测被 mask 的 RoI 类别。
- 原因是，它认为之前的MRM的方法存在信息泄漏的问题，就是说很多region重叠的比率很大，mask掉一个region后，可以从其他region中推测出来该region是什么
- **Image-Text Matching**：之前的工作中，负样本（也就是图文不匹配的样本）是随机的从训练集中选，区别他们很简单，为了强迫模型学习更强的跨模态匹配能力，我们要做的难一点。我们选择caption和图片原本的caption的TF-IDF值小于0.5的那些caption中，TF-IDF最高的top30个caption作为hard sample(而且只有20%的负样本这么构建，其余80%依然用原来的方法构建）



**XGPT: Cross-modal Generative Pre-Training for Image Captioning**

时间：2020.03

模型名称：XGPT

这篇工作专注于做多模态生成任务，提出了**三个novel的生成任务：**

- Image-conditioned Masked Language Modeling (IMLM)
- Image-conditioned Denoising Autoencoding (IDA)
- Text-conditioned Image Feature Generation (TIFG)

这篇文章创新点就在这三个任务上，看下图模型框架来理解

![img](https://static.dingtalk.com/media/lALPDefR4OAMDqHNAz3NBJw_1180_829.png)

我觉得XGPT可能很重要，因为它拥有生成整句话的能力



**Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks**

时间：2020.04

模型名称：Oscar

动机：对于图文对，图片中凸显的对象很容易被检测到，而且经常在对应文本中有所提及

做法：本文将相同语义下的物体（名词）作为图像和语言对齐的锚点（Anchor Point）从而简化图像和文本之间的语义对齐的学习任务，**也就是用Faster-RCNN检测出来的物体标签对应caption中的词建立一个关联**

![img](https://static.dingtalk.com/media/lALPDgCwV2FsPH_NAlzNBDk_1081_604.png)

实践起来很直观，如上图，预训练任务做MLM（Object Tags也算进去）和对比损失（随机将object tags替换成别的构建负样本）

下游任务：VQA、NLVR、GQA、image caption生成、image-text retrieval



**Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal-Transformers**

时间：2020.04

模型名称：Pixel-BERT

动机：可以看到上面工作中图像都用Faster RCNN来提取基于region的视觉特征，但该方法存在缺陷：

![img](https://static.dingtalk.com/media/lALPDfYH23MVMxLNAYrNA4U_901_394.png)

如图，**用Faster RCNN来提取基于region的视觉特征的情况下**：A例子，让目标检测模型取得飞机的状态较为困难；B例子中，即使可以检测到地面和女孩，但因为她们的region重叠，之后fusion embedding模型对于给定边界框来判断实际空间联系就比较困难；C例子中，视觉特征只有长颈鹿，但很难去推理出这些动物的状态。

于是，这篇工作提出Pixel-BERT来跨出边界框，用CNN类模型架构提取图片特征，在**像素和文本级别对齐语义联系**, 解决了VL任务在特定任务的视觉表征上的限制。另一方面，**减轻标注边界框的耗费**，克服视觉任务语义标签和语言语义的不平衡。

此外，该工作还提出了**随机像素采样机制：**

为了增强特征学习、防止过拟合，**受到dropout的启发**，在预训练期间随机采样特征像素。在每个迭代过程中，在抽取像素特征后，会在它们上随机采样一部分，并将它们喂给Transformer。像素随机采样对于模型训练有两方面的好处：

1. 鼓励模型从不完整的视觉输入上学习语义知识，进而增强稳健性。
2. 减少输入元素的数量，这样可以减少计算消耗且加速训练进度。

### **2021**

**Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision**

时间：2021.02

模型名称：ALIGN



在这项工作中，**作者利用了超过10亿个有噪声的图像文本对的数据集来扩展视觉和视觉语言表示学习**。作者采用了Conceptual Captions的方式来获取一个大的噪声数据集。与其不同的是，**作者没有用复杂的数据滤波和后处理步骤来清理数据集，而是只应用简单的基于数据频率的过滤。**虽然得到的数据集有噪声，但比Conceptual Captions数据集大两个数量级。作者发现，在这样的大规模噪声数据集上预训练的视觉和视觉语言表示在广泛的任务上取得了非常强的性能。

作者证明了，**语料库规模的巨大提升可以弥补数据内部存在的噪声，因此即使使用简单的学习方式，模型也能达到SOTA的特征表示**

![img](https://static.dingtalk.com/media/lALPDgCwV2F-7UHNAaTNAgQ_516_420.png) 

图像和文本编码器是通过对比损失函数学习的，将匹配的图像文本对的embedding推在一起，同时将不匹配的图像文本对的embedding分开



**ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision**

时间：2021.02

模型名称：ViLT

本文的重点放在了parameter efficient上，首先总结了以往的工作是怎么处理图像信息的：

- **region feature**：通常采用Faster R-CNN二阶段检测器提取region的特征
- **grid feature**：直接使用CNN（或resnet)提取grid的特征
- **patch projection**：将输入图片切片投影提取特征。

ViLT是首个使用patch projection来做visual embedding的方法

动机：现在VLP模型过度依赖于图像特征提取，使用区域监督和卷积结构，图像特征提取大部分都heavy

导致：

（1）计算速度和效率较低，简单地提取输入特征需要比多模态交互用到多得多的计算量

（2）表达能力的上限取决于视觉嵌入器和预定义的视觉词汇量（predefined visual vocabulary）

本文主要创新：

1. 直接对图像做patch projection, 并简化了模型结构，速度极快，性能相当

![img](https://static.dingtalk.com/media/lALPDeREYjtiP8fNATfNAtY_726_311.png)

1. 使用了whole word masking技巧。whole word masking是将连续的子词tokens进行mask的技巧，避免了只通过单词上下文进行预测。比如将“giraffe”词tokenized成3个部分["gi", "##raf", "##fe"]，可以mask成["gi", "[MASK]", "##fe"]，模型会通过mask的上下文信息[“gi”，“##fe”]来预测mask的“##raf”，就会导致不利用图像信息。
2. ViLT在fine-tuning使用了RandAugment (Cubuk et al., 2020)的图像增强策略，使用所有的原始策略，除了两个：颜色反转和剪切，因为文本通常也包含颜色信息；因为它可能会清除分散在整个图像中的小但重要的对象。



**Unifying Vision-and-Language Tasks via Text Generation**

时间：2021.02

模型名称：VL-T5

**我觉得这篇文章对于设计prompt有启示作用**

**动机：**多模态任务中，不同任务目标都可以形式化为文本表达，比如下图中四个不同的任务，都可以用Text Output来形式化模型的输出结果（这里像极了纯文本T5，也是把比如四元组抽取形式化为一句话来输出）

![img](https://static.dingtalk.com/media/lALPDf0i2LzTwu3NAZ7NAdU_469_414.png) 

这样，就可以用T5这样的文本生成模型来做任务，具体地，模型架构如下：

![img](https://static.dingtalk.com/media/lALPDgCwV2GRKEHNAQHNAvs_763_257.png)

其中，visual embedding 为faster rcnn抽取的实体，prefix和输出结果的构造根据下表：

![img](https://static.dingtalk.com/media/lALPDfmVWhf-SQ_NAdPNBeQ_1508_467.png)

当然，这里做的还都是sentence-level的任务，如何做token-level的任务还面临挑战，这个工作启发我们，如果能将token-level的任务形式化为纯文本的输出任务，就可以借助如T5这样的模型来直接生成

**Seeing Out of tHe bOx: EndtoEnd Pretraining for VisionLanguage Representation Learning**

时间：2021.04

模型名称：SOHO

**动机：**

目前SOTA的vision-language预训练模型中，都是基于已有的**目标检测器提取图像区域**，**再将提取得到的图像区域与文本进行对齐**

两种不同的image-feature：

- region feature：也即检测出来的目标，语义层级较高，但是缺点在于非目标区域会被忽略。
- grid feature：其实就是feature map，语义层级较低，优点在于可以覆盖整张图。

目前基于object detectors（通常是faster rcnn），提取到的region-based image feature有如下问题：

- regions忽略了图像的上下文信息，只能知道Box里面的信息，对于Out of Box的信息完全不知道。
- 目标检测器预先定义的类限制图像理解能力
- detection model 本身也不是完全准确的，会出现一些错漏的目标



如下图：

![img](https://static.dingtalk.com/media/lALPDgCwV2GX_bXNARXNAtA_720_277.png)

如果只是目标检测，对于实体外的缓解就忽略了，以前的模型看不到人在河岸，以为人在划船

**Contribution**

- **直接拿整张图像作为输入**，可以直接利用图文对end2end一步到位学习跨模态表征（cross-modal representation）。由于无需前置的目标检测器来找到region，推理速度快了10倍。
- 如果直接输入图像不给额外的明确的指导信息，pixel-level很难进行对齐,我们提出了一个会动态更新的visual dictionary来提取视觉特征（**这个visual dictionary还没看懂，是本文的关键**）



![img](https://static.dingtalk.com/media/lALPDeC245ahRgXNAd7NAtA_720_478.png)



**Visual parsing with self-attention for vision-and-language pre-training**

时间：2021.06

模型名称：Visual parsing

动机：visual contents 之间的关系对理解图像很重要，也对VLP中模态之间的对齐很重要。给定一个图片，图片中一个人在冲浪，VQA任务问图中人在干什么，显然人与冲浪板的关系对回答这个问题很重要，然而现有的这三种图片表示的方法都不能建模intra-vision relation。对于region和patch特征，每个unit是独立的，全局的关系并没有visual embedding中被编码。对于resnet这种学到的grid feature,卷积层的局部的感受野只能学到邻近region的局部特征

做法：于是本文用Vit先建模visual contents之间的关系，并在预训练任务中设计了独特的mask机制：随机选择一个词mask后，根据前一步得到的attention找到与它关系最大的topk个词mask掉，减弱了visual之间的关联，强迫模型从文本端学习.

**UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning**

时间：ACL2021

模型名称：UNIMO

动机：在视觉-文本多模态领域，目前比较流行的模型有ViBERT，VisualBERT和UNITER等。**但是这些模型存在的共同问题是只能利用图像-文本对数据来进行训练。高质量的图像-文本对数据是非常稀少的**，但是网络上却存在着大规模的图像或者文本的单模态数据，如果能将这些数据利用起来，则对于模型性能提升非常有益。**如图所示，对于给定的问题，如果只提供图片信息，是非常难以找到准确答案的，但是如果能够将这张图片和有效的文本信息联系起来，则可以非常容易地得到答案**。

![img](https://static.dingtalk.com/media/lALPDetfX4TkSbfNAc7NAok_649_462.png)



本文提出了UNIMO的框架来用一个模型统一处理多模态场景数据输入，包括单独的文本、图像以及对齐的图像-文本数据，处理过程如下，采用对比学习来处理三种不同的输入，对比学习过程采样采用了下面三种技术：

![img](https://static.dingtalk.com/media/lALPDefR4OAvOFXNAkbNAtA_720_582.png)

- Text Rewriting
  - 将原图文对的文本用一个文本改写模型改写成别的句子，分为sentence-level,phrase-level和word-level rewriting
    - sentence-level rewrting ：用回译的方法
    - 负样本：用TF-IDF找数据集中最相似的caption与原图构成负样本对
- Text Retrieval
  - 基于相似度，在单模态文本数据中抽取文本
- Image Retrieval
  - 基于相似度，在单模态图片数据中抽取图像

UNIMO的优点在于可以利用大量的开放域文本语料和图片集来提高视觉和文本理解能力

此外，还有SimVLM[1]（致力于设计简单的多模态预训练框架，使用前缀语言建模这一单一目标）、FLAVA[2]（Facebook发布的通用模型，支持图文双模态、纯文本、纯图片输入，可以做多模态、文本、图像三种不同形式的任务）VLMo[3]（微软21年多模态预训练工作）、ALBEF[4](作者引入了一种对比损失，通过在跨模态注意前融合(ALign BEfore Fuse)来调整图像和文本表示,并动量蒸馏方法缓解噪声问题)



从多模态预训练的研究可以发现：

1. 解决的都是sentence-level的多模态任务，没有解决token-level的工作
2. 不同的预训练任务很多，不少工作动辄几个三四个任务同时训练，这反映了多模态任务的复杂性，统一建模的挑战性



但仍有很多启发和收获：

1. 以VL-T5为代表的模型，用文本生成的方法形式化不同的多模态任务，和我期望的很贴合，有潜力
2. 实验数据这里没有贴，随着数据规模越来越大，模型的建模能力越来越强，性能也越来越好，如果能解决前述挑战，迁移到多模态ABSA上，应该能达到较好的效果。
3. 虽然多模态任务种类多，但预训练模型还是统一了不少下游任务，说明还是有很多任务共有的知识可以迁移的



一些感想：从时间顺序看多模态预训练的发展，可以发现比较亮点的工作集中在20下半年到21上半年，这可以看作这个领域的成熟期，在这之前19年的工作相对开创，额外的创新较少，比的是谁能最先建模出这个任务，而21年下半年后到现在，就都是大厂在卷数据，卷大模型，或是一些很复杂的操作了，不够直观，可能渐渐做不动了；而中间成熟期那段时间，不论是从动机上还是从idea的直觉性上，都给人眼前一亮，很有意思。



