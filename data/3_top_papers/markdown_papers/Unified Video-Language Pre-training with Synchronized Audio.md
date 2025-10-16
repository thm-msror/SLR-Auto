## <span id="page-0-0"></span>Unified Video-Language Pre-training with Synchronized Audio

Shentong Mo1,2<sup>∗</sup> Haofan Wang<sup>3</sup> Huaxia Li<sup>3</sup> Xu Tang<sup>3</sup> <sup>1</sup>CMU, <sup>2</sup>MBZUAI, <sup>3</sup>Xiaohongshu

## Abstract

Video-language pre-training is a typical and challenging problem that aims at learning visual and textual representations from large-scale data in a self-supervised way. Existing pre-training approaches either captured the correspondence of image-text pairs or utilized temporal ordering of frames. However, they do not explicitly explore the natural synchronization between audio and the other two modalities. In this work, we propose an enhanced framework for Video-Language pre-training with Synchronized Audio, termed as VLSA, that can learn tri-modal representations in a unified self-supervised transformer. Specifically, our VLSA jointly aggregates embeddings of local patches and global tokens for video, text, and audio. Furthermore, we utilize local-patch masked modeling to learn modality-aware features, and leverage global audio matching to capture audio-guided features for video and text. We conduct extensive experiments on retrieval across text, video, and audio. Our simple model pre-trained on only 0.9M data achieves improving results against state-of-the-art baselines. In addition, qualitative visualizations vividly showcase the superiority of our VLSA in learning discriminative visual-textual representations.

## 1 Introduction

When we enjoy fascinating story in a movie, not only frames with the caption are attractive, but synchronized sounds are also impressive for us. In daily life, sound appears to be in multiple scenes, such as a classroom scene with professors' speech and students' whispering, a family reunion that consists of talking and laughing. As audio and visual contents are commonly matched and synchronized, many researchers [\[1,](#page-9-0) [2,](#page-9-1) [3,](#page-9-2) [4,](#page-9-3) [5,](#page-9-4) [6,](#page-9-5) [7\]](#page-9-6) have explored the benefit of such synchronization for audio-visual tasks, such as audio-event localization [\[8,](#page-9-7) [9,](#page-9-8) [10,](#page-9-9) [11,](#page-9-10) [12,](#page-9-11) [13\]](#page-9-12), audio-visual spatialization [\[14,](#page-9-13) [15,](#page-9-14) [16,](#page-9-15) [17\]](#page-9-16), and sound source localization [\[18,](#page-9-17) [19,](#page-9-18) [20,](#page-10-0) [21,](#page-10-1) [22,](#page-10-2) [23,](#page-10-3) [24,](#page-10-4) [25,](#page-10-5) [26,](#page-10-6) [27\]](#page-10-7). However, in this work, we leverage this cross-modal synchronization between audio and videos/texts for enhancing video-language pre-training.

Video-language pre-training aims to learn visual and textual representations jointly from large-scale data. Previous work [\[28,](#page-10-8) [29,](#page-10-9) [30,](#page-10-10) [31,](#page-10-11) [32,](#page-10-12) [33,](#page-10-13) [34,](#page-10-14) [35\]](#page-10-15) either proposed to learn the correspondence of image-text pairs or utilized temporal ordering of frames. Typically, MMT [\[29\]](#page-10-9) proposed a multimodal transformer to aggregate per-frame visual features with temporal information. With the success of CLIP [\[36\]](#page-10-16) on visual and textual representation learning, CLIP2TV [\[37\]](#page-10-17) leveraged CLIP pre-trained weights with a video-text alignment module and a video-text matching module for discriminating positive and negative pairs of embeddings from text and video encoders.

However, they do not explicitly explore the natural synchronization between audio and other two modalities, and can not learn compact representations of the global correspondence across each modality. In contrast, we leverage synchronized audio for global audio matching in our unified transformer to capture discriminative global features for boosting video-language pre-training.

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: Comparison performance of text-to-video (Left), zero-shot text-to-video (Middle), and text-to-audio retrieval (Right). VLSA pre-trained on only 0.9M data achieves significant gains compared to previous video-language (MEE, HT, MMT, SupportSet, Frozen, OA-Trans, AllinOne), video-audio (TVLT), and video-language-audio (CE, VATT) methods.

Since audio and visual contents are commonly matched and synchronized, recent pre-training methods [\[38,](#page-11-0) [39,](#page-11-1) [40\]](#page-11-2) are developed to incorporate audio as an auxiliary modality for learning discriminative representations transferrable to video-text retrieval. VATT [\[38\]](#page-11-0) introduced a multimodal contrastive loss for global features of each modality to learn the alignment of video-audiotext triplets. AudioCLIP [\[39\]](#page-11-1) extended CLIP with the ESResNeXt audio model and optimized a contrastive loss between two modalities to maximize diagonal values of the scaled dot-product similarity matrix. More recently, MERLOT Reserve [\[40\]](#page-11-2) employed a contrastive span training loss between masked tokens and corresponding global uni-modal embeddings for audio, captions, and video frames. While these methods achieve promising results on video-language pre-training, they are extremely dependent on the capacity of three separate encoders with many parameters to extract discriminative representations for each modality on large-scale datasets. In addition, they do not incorporate local features into the pre-training objective. Different from them, we combine localpatch masked modeling with global-token alignment in a unified transformer to learn modality-aware features during pre-training.

To this end, we propose a novel and enhanced framework for Video-Language pre-training with Synchronized Audio, namely VLSA, that can jointly learn tri-modal representations in a unified selfsupervised transformer. Specifically, VLSA aggregates local patches and global token representations for each modality through a joint transformer. Furthermore, local-patch masked modeling is applied to patch-level embeddings of each modality to learn modality-aware local features. In addition, global audio matching is applied on both video and text modalities to capture compact features from global tokens of audio, text, and video frames.

We pre-train our VLSA on only 0.9M video-audio-transcript triplets from an intersection set of HowTo100M and AudioSet. We conduct extensive experiments on retrieval across text, video, and audio. Our simple pre-trained model achieves improving results against previous baselines on downstream tasks. In addition, qualitative visualizations vividly showcase the advantage of the proposed approach in learning compact visual-textual representations.

<span id="page-2-1"></span><span id="page-2-0"></span>Table 1: Comparison of multi-modal self-supervised pre-training on video, text, and audio or speech. "Single", "Dual", and "Triple" refer to one joint encoder, two separate encoders, and three separate encoders, respectively. Local and global denote uni-modal masked modeling and multi-modal matching.

| Method              | Publication   | Modalities          | Architecture  | Pre-train Objectives | Pre-train Datasets |
|---------------------|---------------|---------------------|---------------|----------------------|--------------------|
| VideoBERT [31]      | ICCV, 2019    | Video, Text         | Single        | local                | self-collected     |
| Frozen [32]         | ICCV, 2021    | Video, Text         | Dual          | global               | WebVid2M+CC3M      |
| OA-Trans [34]       | CVPR, 2022    | Video, Text         | Dual          | global               | WebVid2M+CC3M      |
| Region-Learner [56] | AAAI, 2023    | Video, Text         | Single        | global               | WebVid2M+CC3M      |
| AllinOne [35]       | CVPR, 2023    | Video, Text         | Single        | local+global         | WebVid2M+HowTo100M |
| VATT [38]           | NeurIPS, 2021 | Video, Text, Audio  | Single/Triple | global               | AudioSet+HowTo100M |
| AudioCLIP [39]      | ICASSP, 2022  | Video, Text, Audio  | Triple        | global               | AudioSet           |
| MERLOT Reserve [40] | CVPR, 2022    | Video, Text, Audio  | Triple        | global               | YT-Temporal-1B     |
| i-Code [57]         | AAAI, 2023    | Video, Text, Speech | Single        | local+global         | self-collected     |
| VLSA (ours)         | -             | Video, Text, Audio  | Single        | local+global         | AudioSet∩HowTo100M |

Our contributions can be summarized as follows:

- We propose a novel and enhanced framework for Video-Language pre-training with Synchronized Audio, termed as VLSA, that aggregates local patches and global visual-textual representation guided by audio through a joint transformer encoder.
- We introduce audio local-patch masked modeling to learn modality-aware interaction between audio and the other two modalities.
- We further leverage global audio matching to capture compact and discriminative video-text features without the need for video-text matching.
- Our simple unified transformer pre-trained on only 0.9M triplets achieves competitive results against baselines on text-video and text-audio retrieval.

#### 2 Related Work

**Video-Language Pre-training.** Video-language pre-training aims at learning joint visual and textual representations from captions and video frames. At the early stage, CE [28] designed a collaborative experts model to aggregate information from different pre-trained experts for video retrieval tasks. MEE [41] proposed a model with mixed embedding experts to handle missing input modalities for learning improved text-video embeddings simultaneously. In the recent years, diverse pipelines [42, 43, 44, 45, 31, 46, 47, 48, 49, 50, 51, 52, 32, 34, 35, 53] have been proposed to explore the fusion of two distinct modalities, where the correspondence of image-text pairs is usually aligned during pre-training, as shown in Table 1. Typically, ActBERT [30] leveraged a tangled transformer to learn the correspondence between global actions and local object regions from paired video sequences and text descriptions. A multi-modal transformer was introduced in MMT [29] to aggregate per-frame visual features with temporal information. CAMoE [33] utilized mixture-of-experts to capture the alignment between text and multiple video representations of action and scene. Based on CLIP [36] pre-trained weights, CLIP2TV [37] adopted a video-text alignment module and a video-text matching module to discriminate positive and negative pairs of features from text and video encoders.

While the aforementioned video-language pre-training approaches achieve promising performance on downstream tasks, they do not learn the natural synchronization between audio and the other two modalities in an explicit manner. Moreover, they can not learn compact representations of the global correspondence across each modality for retrieval tasks [54, 55] involved with audio. In this work, we aim to enhance video-language pre-training with synchronized audio by global-token matching for both text-video and text-audio retrieval.

Audio Synchronization in Pre-training. Audio synchronization in pre-training has been applied in previous works [38, 39, 40, 57] to learn discriminative representations of audio, captions, and video frames. VATT [38] introduced a multi-modal contrastive loss for global features of each modality to learn the alignment of video-audio-text triplets. AudioCLIP [39] extended CLIP with the ESResNeXt audio model and optimized a contrastive loss between two modalities to maximize diagonal values of the scaled dot-product similarity matrix. More recently, MERLOT Reserve [40] employed a contrastive span training loss between masked tokens and corresponding global unimodal embeddings for audio, captions, and video frames. i-Code [57] exploited a multi-modal fusion

<span id="page-3-1"></span><span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2: Illustration of our enhanced framework of Video-Language pre-training with Synchronized Audio (VLSA). The modality-aware patch embeddings  $\{\mathbf{x}_i^v\}_{i=1}^{VI}, \{\mathbf{x}_i^a\}_{i=1}^{A}, \{\mathbf{x}_i^t\}_{i=1}^{S},$  are extracted from each linear projection layer. The Local-Patch Masked Modeling module is applied to local-patch representations for audio spectrogram a extracted from the unified encoder, and the decoder is utilized to predict the raw audio spectrograms  $\hat{\mathbf{a}}$  for learning the interaction of audio and the other two modalities (video and text). Finally, the Global Audio Matching module (contrastive loss and binary matching loss) is leveraged on modality-aware global embeddings  $\hat{\mathbf{g}}^v, \hat{\mathbf{g}}^t, \hat{\mathbf{g}}^a$  averaged from the encoder to capture the cross-modal alignment between synchronized audio and video frames/caption sentence in an explicit manner.

network to integrate single-modality outputs of vision, speech, and language from each pre-trained encoder.

Different from these tri-modal pre-training baselines, the proposed transformer is more lightweight than comparable methods, as we do not need to use three separate encoders with many parameters to extract discriminative embeddings for each modality. In addition, we develop a fully novel and unified transformer to aggregate both local-patch masked modeling and global-token matching for learning modality-aware features. Audio-video matching and masked modeling are also addressed in a very recent work called TVLT [58], but they do not involve captions during pre-training.

#### 3 Method

Given a triplet of video sequences, captions, and synchronized audio, our target is to learn joint representations from large-scale unsupervised data. We propose a novel Video-Language pre-training framework with Synchronized Audio named VLSA for enhancing visual-textual semantics from sound itself, which mainly consists of two modules, Local-Patch Masked Modeling in Section 3.2 and Global Audio Matching in Section 3.3.

#### 3.1 Preliminaries

In this section, we first describe the problem setup and notations, and then revisit the commonly-used objective for video-language pre-training.

**Problem Setup and Notations.** Given a triplet of video sequences with a dimension of  $V \times 3 \times H \times W$ , texts with a dimension of  $S \times C$ , and synchronized audio spectrograms with a dimension of  $T \times F$ , our target is to learn discriminative representations simultaneously and evaluate them for downstream retrieval tasks. We formally denote V as the number of video frames, S as the length of text sentences, S as the length of the word dictionary, and S as the number of frequencies in audio. S and S are the height and width of each frame in the video.

**Revisit Video-Language Pre-training.** To solve the joint modeling problem, current pre-training baselines [47, 48, 49, 50, 51, 52, 32, 34, 35] introduced a visual-textual contrastive loss to learn the alignment between images and captions. Given a set of global video-language features  $\{\mathbf{g}_i^v\}_{i=1}^B, \{\mathbf{g}_i^t\}_{i=1}^B$  in a batch of size B, the contrastive loss between the cosine similarity  $\mathtt{sim}(\mathbf{g}_i^v, \mathbf{g}_i^t)$ 

of global embeddings is formulated as:

<span id="page-4-1"></span>
$$\mathcal{L}_{v \to t} = \frac{1}{B} \sum_{i}^{B} -\log \frac{\exp\left(\frac{1}{\tau} \operatorname{sim}(\mathbf{g}_{i}^{v}, \mathbf{g}_{i}^{t})\right)}{\sum_{j=1}^{B} \exp\left(\frac{1}{\tau} \operatorname{sim}(\mathbf{g}_{i}^{v}, \mathbf{g}_{j}^{t})\right)},\tag{1}$$

where  $\mathbf{g}_i^v, \mathbf{g}_j^t \in \mathbb{R}^{1 \times D}$ , and D is the size of the embedding dimension.  $\mathrm{sim}(\mathbf{g}_i^v, \mathbf{g}_i^t) = (\mathbf{g}_i^v)^\top \mathbf{g}_i^t/(\|\mathbf{g}_i^v\|\|\mathbf{g}_i^t\|)$  is the cosine similarity, and  $\tau$  is the temperature parameter.  $B^2 - B$  negative vision-language pairs are created within a training batch. Besides, they proposed to take  $N^{v,t}$  samples randomly from the whole dataset and applied an FC layer and sigmoid operator to predict the matching probability  $\mathbf{p}^{v,t}$  of image-caption pairs. The video-text matching loss is formally summarized as:

$$\mathcal{L}_{vtm} = \sum_{n=1}^{N^{v,t}} \text{BCE}\left(\mathbf{y}_n^{v,t}, \mathbf{p}_n^{v,t}\right)$$
 (2)

where  $\mathbf{y}_n^{v,t}$ ,  $\mathbf{p}_n^{v,t}$  denote the ground-truth and probability of nth visual-textual representation pairs.  $\mathrm{BCE}(\cdot)$  is a binary cross-entropy loss. If the image and caption are from the same pair, the target is 1; otherwise, it is 0. During training, the positive pair is sampled from the dataset, while the negative pair is created by replacing the text or vision in a paired sample with a randomly selected other sample. Overall, the global loss for video-language alignment can be computed by  $\mathcal{L}_{v,t} = \mathcal{L}_{v \to t} + \mathcal{L}_{t \to v} + \mathcal{L}_{vtm}$ .

However, those video-language pre-training approaches are extremely dependent on the capacity of encoders with many parameters to extract discriminative global embeddings for each modality. In addition, most frameworks do not explicitly learn the natural synchronization between audio and video/text. To address this issue, we propose a novel and unified transformer for Video-Language pre-training with Synchronized Audio, that can learn both local and global features between audio and videos/captions simultaneously, as illustrated in Figure 2.

#### <span id="page-4-0"></span>3.2 Local-Patch Masked Modeling

In order to learn local modality-aware features across three different modalities, we introduce modality-aware patch embeddings for each modality that are extracted from raw input via each linear projection layer, i.e.,  $\mathbf{x}^v \in \mathbb{R}^{(V \times I) \times D}$ ,  $\mathbf{x}^t \in \mathbb{R}^{S \times D}$ ,  $\mathbf{x}^a \in \mathbb{R}^{A \times D}$ , where I, S, A denotes the total number of patches for each video frame, each caption, and the corresponding audio. Assume the patch resolution of each frame and audio are  $P^v, P^a$ , the patch-wise raw input for video and audio are formally denoted as  $\mathbf{v} \in \mathbb{R}^{(V \times I) \times (3 \times P^v \times P^v)}$  and  $\mathbf{a} \in \mathbb{R}^{A \times (P^a \times P^a)}$ . Note that  $I = H/P^v \times W/P^v, A = T/P^a \times F/P^a$ .

With local-patch representations for each modality  $\{\mathbf{x}_i^v\}_{i=1}^{VI}, \{\mathbf{x}_i^t\}_{i=1}^{S}, \{\mathbf{x}_i^a\}_{i=1}^{A}$ , we first apply an unified transformer encoder  $\phi(\cdot)$  to aggregate patch-level features from the raw input as:

$$\begin{aligned}
&\{\hat{\mathbf{x}}^{v}\}_{i=1}^{VI}, \{\hat{\mathbf{x}}_{i}^{t}\}_{i=1}^{S}, \{\hat{\mathbf{x}}_{i}^{a}\}_{i=1}^{A} = \{\phi(\mathbf{x}_{j}, \mathbf{X}, \mathbf{X})\}_{j=1}^{VI+S+A}, \\
&\mathbf{X} = \{\mathbf{x}_{j}\}_{j=1}^{VI+S+A} = [\{\mathbf{x}_{i}^{v}\}_{i=1}^{VI}; \{\mathbf{x}_{i}^{t}\}_{i=1}^{S}; \{\mathbf{x}_{i}^{a}\}_{i=1}^{A}]
\end{aligned} \tag{3}$$

where  $[\;;\;]$  denotes the concatenation operator.  $\mathbf{x}_i^v, \mathbf{x}_i^t, \mathbf{x}_i^a \in \mathbb{R}^{1 \times D}$ , and D is the dimension of embeddings. The self-attention operator  $\phi(\cdot)$  is formulated as:

$$\phi(\mathbf{x}_j, \mathbf{X}, \mathbf{X}) = \operatorname{Softmax}(\frac{\mathbf{x}_j \mathbf{X}^\top}{\sqrt{D}}) \mathbf{X}$$
 (4)

Then, to capture the interaction across each modality, we exploit a tri-modal masking mechanism and a shared decoder to predict the masked patch of one modality, given the other two modalities as auxiliaries. Specifically, with video and text embeddings  $\{\mathbf{x}_i^v\}_{i=1}^{VI}, \{\mathbf{x}_i^t\}_{i=1}^{S}$ , we leverage a decoder to predict the raw audio spectrograms  $\hat{\mathbf{a}}$  for randomly masked audio patches. The local-level audio masked loss is computed with the mean square loss between the targeted and predicted spectrograms as:

$$\mathcal{L}_a^{\text{local}} = \frac{1}{N^a} \sum_{i \in M^a} ||\mathbf{a}_i - \hat{\mathbf{a}}_i||_2^2$$
 (5)

where  $N^a$ ,  $M^a$  denotes the total number and the set of masked patches for audio, respectively. Similar to audio, the local-level video masked loss  $\mathcal{L}_{v}^{\text{local}}$  is calculated with the mean square loss between

<span id="page-5-1"></span>the ground truth and missing pixel of frames with a random mask on all VI patches. For text masked loss  $\mathcal{L}_t^{\text{local}}$ , we use a decoder to predict the randomly masked token in S tokens, similar to BERT [59]. Note that three separate decoders for tri-modal masked prediction are parameter-shared and achieve the best performance, as observed in our experiments in Section 4.3. With optimizing the total local-level masked loss  $\mathcal{L}^{\text{local}} = \sum_{m \in \{a,v,t\}} \mathcal{L}_m^{\text{local}}$  with a shared encoder, it will capture the local-level interaction between audio and other two distinct modalities, which pushes the model to learn more discriminative embeddings.

#### <span id="page-5-0"></span>3.3 Global Audio Matching

Benefiting from the local-level masked loss above, we propose a novel and explicit global audio matching mechanism on global embeddings  $\hat{\mathbf{g}}^v, \hat{\mathbf{g}}^t, \hat{\mathbf{g}}^a \in \mathbb{R}^{1 \times D}$  for each modality in the unified transformer  $\phi(\cdot)$  to generate global modality-aware features as:

$$\hat{\mathbf{g}}^{v}; \hat{\mathbf{g}}^{t}; \hat{\mathbf{g}}^{a} = \text{AvgPool}(\{\hat{\mathbf{x}}^{v}\}_{i=1}^{VI}; \{\hat{\mathbf{x}}_{i}^{t}\}_{i=1}^{S}; \{\hat{\mathbf{x}}_{i}^{a}\}_{i=1}^{A}); \tag{6}$$

where  $\hat{\mathbf{g}}^v, \hat{\mathbf{g}}^t, \hat{\mathbf{g}}^a \in \mathbb{R}^{1 \times D}$ , and D is the dimension of each global embedding. AvgPool(·) denotes the average pooling operator. That is, we average local-level patches for three modalities along each total number of patches (VI, S, A) to generate the modality-aware global embeddings.  $[\;;\;]$  is the concatenation operator.

In order to explicitly learn the cross-modal alignment between synchronized audio and video frames, we leverage a contrastive loss similar to Eq. 1 for maximizing the cosine similarity  $\mathtt{sim}(\hat{\mathbf{g}}_i^a, \hat{\mathbf{g}}_i^v)$  of audio-video pairs from the same batch index i. By applying an FC layer and sigmoid operator to predict the alignment probability  $\mathbf{p}^{a,v} \in \mathbb{R}^{N \times 1}$  of  $N^{a,v}$  audio-video pairs that are randomly chosen from the whole pre-training dataset, we formulate the global audio-visual alignment loss as:

$$\mathcal{L}_{a \to v}^{\text{global}} = \frac{1}{B} \sum_{i}^{B} -\log \frac{\exp\left(\frac{1}{\tau} \text{sim}(\hat{\mathbf{g}}_{i}^{a}, \hat{\mathbf{g}}_{i}^{v})\right)}{\sum_{j=1}^{B} \exp\left(\frac{1}{\tau} \text{sim}(\hat{\mathbf{g}}_{i}^{a}, \hat{\mathbf{g}}_{j}^{v})\right)} \\
-\log \frac{\exp\left(\frac{1}{\tau} \text{sim}(\hat{\mathbf{g}}_{i}^{v}, \hat{\mathbf{g}}_{i}^{a})\right)}{\sum_{j=1}^{B} \exp\left(\frac{1}{\tau} \text{sim}(\hat{\mathbf{g}}_{i}^{v}, \hat{\mathbf{g}}_{j}^{a})\right)} + \sum_{n=1}^{N^{a,v}} \text{BCE}\left(\mathbf{y}_{n}^{a,v}, \mathbf{p}_{n}^{a,v}\right) \tag{7}$$

where B is the batch size.  $\mathbf{y}^{a,v} \in \mathbb{R}^{N \times 1}$  denotes a one-hot encoding and its element for the entry is 0 for non-alignment and 1 for alignment. Since one audio spectrogram is distinct in each audio-video pair, this alignment loss does not bring false alignment pairs, while most frames in a video look similar in high-level textual semantics. To boost the compactness of pre-trained global representations for retrieval, we apply similar cross-modal alignment loss  $\mathcal{L}_{a \to t}^{\mathrm{global}}$  on global tokens across audio-text pairs.

With optimizing the total global-token alignment loss  $\mathcal{L}^{\text{global}} = \mathcal{L}^{\text{global}}_{a \to v} + \mathcal{L}^{\text{global}}_{a \to t}$ , we push the model to learn more discriminative video and text representations with the benefit of synchronized audio. The overall objective of our model is simply optimized in an end-to-end manner as

$$\mathcal{L} = \mathcal{L}^{local} + \lambda \cdot \mathcal{L}^{global}$$
(8)

where  $\lambda$  denotes the weighted parameter for balancing two losses with different orders of magnitude. We use  $\lambda=5$  as the default in our experiments. During inference, we simply compute the cosine-similarity  $\sin(\hat{\mathbf{g}}^t,\hat{\mathbf{g}}^v)$ ,  $\sin(\hat{\mathbf{g}}^t,\hat{\mathbf{g}}^a)$ ,  $\sin(\hat{\mathbf{g}}^v,\hat{\mathbf{g}}^a)$  for retrieval across text-video, text-audio, and video-audio settings.

## 4 Experiments

#### 4.1 Experimental Setup

**Datasets.** HowTo100M [61] consists of 136M video clips from 1.22M YouTube videos with 134,472 hours. AudioSet [69] contains 2,084,320 clips with 632 classes from YouTube videos covering human and animal sounds, musical instruments, and common everyday environmental sounds. An intersection of HowTo100M and AudioSet with 0.9M audio-video-text triplets is used for pre-training. MSR-VTT [70] includes 10K YouTube videos with 200K description sentences and is split into 9K

<span id="page-6-1"></span><span id="page-6-0"></span>Table 2: Quantitative results of text-to-video retrieval on MSR-VTT dataset. "Single", "Dual", and "Triple" refer to one joint encoder, two separate, and three separate encoders. Bold and Underline denote the best and second result, respectively.

| Method                  | Modalities         | Architecture | Pre-train Datasets | Data Size (↓) | R@1 (↑) | R@5 (↑)     | R@10 (†)    |
|-------------------------|--------------------|--------------|--------------------|---------------|---------|-------------|-------------|
| JSFusion [60]           | Video, Text        | Dual         | AudioSet+ImageNet  | 3M            | 10.2    | 31.2        | 43.2        |
| MEE [41]                | Video, Text, Audio | Triple       | COCO+VisGenome     | 5.6M          | 14.2    | 39.2        | 53.8        |
| HT [61]                 | Video, Text        | Dual         | HowTo100M          | 136M          | 14.9    | 40.2        | 52.8        |
| CE [28]                 | Video, Text, Audio | Triple       | YouTube-8M         | 8M            | 20.9    | 48.8        | 62.4        |
| AVLnet [62]             | Video, Text, Audio | Triple       | HowTo100M          | 136M          | 27.1    | 55.6        | 66.6        |
| ActBERT [30]            | Video, Text        | Dual         | HowTo100M          | 136M          | 16.3    | 42.8        | 56.9        |
| HERO [63]               | Video, Text        | Dual         | HowTo100M          | 136M          | 16.8    | 43.4        | 57.7        |
| VidTranslate [64]       | Video, Text        | Dual         | HowTo100M          | 136M          | 14.7    | -           | 52.8        |
| NoiseEstimation [65]    | Video, Text        | Dual         | HowTo100M          | 136M          | 17.4    | 41.6        | 53.6        |
| UniVL [66]              | Video, Text        | Dual         | HowTo100M          | 136M          | 21.2    | 49.6        | 63.1        |
| MMT [29]                | Video, Text, Audio | Triple       | HowTo100M          | 136M          | 26.6    | <u>57.1</u> | 69.6        |
| ClipBERT [52]           | Video, Text        | Dual         | COCO+VisGenome     | 5.6M          | 22.0    | 46.8        | 59.9        |
| Frozen [32]             | Video, Text        | Single       | CC3M               | 3M            | 25.5    | 54.5        | 66.1        |
| TVLT [58]               | Video, Audio       | Single       | HowTo100M          | 136M          | 22.0    | -           | -           |
| Everything-At-Once [67] | Video, Text, Audio | Triple       | HowTo100M          | 136M          | 23.7    | 52.1        | 63.7        |
| AllinOne [35]           | Video, Text        | Single       | AudioSet∩HowTo100M | 0.9M          | 22.1    | 49.1        | 60.6        |
| VLSA (ours)             | Video, Text, Audio | Single       | AudioSet∩HowTo100M | 0.9M          | 27.1    | 57.3        | <u>68.9</u> |
| zero-shot:              |                    |              |                    |               |         |             |             |
| HT [61]                 | Video, Text        | Dual         | HowTo100M          | 136M          | 7.5     | 21.2        | 29.6        |
| SupportSet [68]         | Video, Text        | Dual         | HowTo100M          | 136M          | 8.7     | 23.0        | 31.1        |
| VATT [38]               | Video, Text, Audio | Single       | AudioSet+HowTo100M | 138M          | _       | _           | 29.7        |
| Frozen [32]             | Video, Text        | Dual         | CC3M+WebVid-2M     | 5.5M          | 18.7    | <u>39.5</u> | 51.6        |
| OA-Trans [34]           | Video, Text        | Dual         | WebVid-2M          | 2.5M          | 18.4    | 36.5        | 46.8        |
| Everything-At-Once [67] | Video, Text, Audio | Triple       | HowTo100M          | 136M          | 9.9     | 24.0        | 32.6        |
| VLSA (ours)             | Video, Text, Audio | Single       | AudioSet∩HowTo100M | 0.9M          | 20.4    | 40.6        | 53.8        |

for training and 1K for testing. LSMDC [71] contains 118,081 video clips with 7,408 and 1,000 videos for validation and testing. AudioCaps [72] is filtered out for 49,291 clips for training, 428 clips for validation, and 816 clips for testing. SoundDescs [55] consists of 32,979 audio clips with 23 sound categories, and is divided into 70% of the clips for training and 15% each for validation and testing. Yoocook2 [73] comprises 13K video clips of 89 cooking recipes with 9,586 clips for training and 3,350 clips for validation.

**Evaluation Metrics.** We use the standard retrieval metrics [30, 52, 64, 65, 38] to evaluate the performance of our model. Recall at rank k (R@k) measures the percentage of labels retrieved within the top k ranked predictions, and the higher value is better. k = 1, 5, 10 for text-video retrieval, and k = 1, 5, 10, 50 for text-audio retrieval. For all metrics, we report the average result of three different random seeds.

Implementation. For each audio waveform, we follow the prior work [74] and sub-sample the audio signal to 11kHz. A Short-time Fourier transform with a window size of 1022 and a hop length of 256 is further applied to generate a  $512 \times 256$  Time-Frequency representation of the audio, which is resampled to a log-frequency scale with a size of  $256 \times 256$  as the input audio spectrogram, *i.e.*, T = 256, F = 256. For each caption, the tokenizer embedding from BERT [59] is used as the input with a maximum sentence length of 40 and a vocab size of 30,522, *i.e.*, C = 30,522. For each video clip, we randomly sample 8 frames as inputs and resize each frame to  $224 \times 224$ , *i.e.*, V = 8, H = W = 224. Following prior work [75, 76], we apply a patch size of  $16 \times 16$  for both audio and video frames. With patch size  $P^v = 16, P^a = 16$ , the total number of visual and audio patches I = 196, A = 256. We use a ViT-base [77] model for the masked autoencoder same as in MAE [75]. We follow previous approaches [59, 75] to mask 15% on each word token randomly, 75% on patches of each frame independently, and 75% on patches of audio spectrograms. The model is trained for 200k steps with the AdamW [78] optimizer with a learning rate of 1e-4, a decay rate of 0.01, and a batch size of 2048. For fine-tuning, the model is trained for 20 epochs with a batch size of 256.

#### 4.2 Comparison to Prior Work

In this work, we propose a novel and effective framework for video-language pre-training with synchronized audio. In order to validate the effectiveness of the proposed VLSA, we comprehensively compare it to previous video-text (VT), video-audio (VA), and video-text-audio (VTA) pre-training baselines: i) **VT:** 1) JSFusion [60]: a very early sequence fusion model with multi-modal matching; 2) MEE [41]: a baseline with mixed embedding experts for visual-textual modalities; 3) HT [61]:

<span id="page-7-2"></span><span id="page-7-0"></span>Table 3: Quantitative results of text-audio retrieval on SoundDescs dataset. Best results are bold.

| Method      | Pre-train Datasets      | Data Size (↓) | Text-to-Audio |         |          | Audio-to-Text |         |         |          |          |
|-------------|-------------------------|---------------|---------------|---------|----------|---------------|---------|---------|----------|----------|
| Method      | inod Fre-train Datasets |               | R@1 (↑)       | R@5 (↑) | R@10 (↑) | R@50 (↑)      | R@1 (↑) | R@5 (↑) | R@10 (↑) | R@50 (↑) |
| MEE [41]    | YouTube-8M+VGGSound     | 8.2M          | 30.8          | 60.8    | 70.9     | 85.9          | 30.9    | 60.3    | 70.1     | 85.3     |
| CE [28]     | YouTube-8M+VGGSound     | 8.2M          | 31.1          | 60.6    | 70.8     | 86.0          | 30.8    | 60.3    | 69.5     | 85.4     |
| MMT [29]    | YouTube-8M+VGGSound     | 8.2M          | 30.7          | 61.8    | 72.2     | 88.8          | 31.4    | 63.2    | 73.4     | 89.0     |
| VLSA (ours) | AudioSet∩HowTo100M      | 0.9M          | 33.5          | 63.7    | 75.1     | 91.6          | 34.2    | 65.9    | 76.3     | 92.1     |

<span id="page-7-1"></span>Table 4: Quantitative results on LSMDC, AudioCaps, and Youcook2 benchmarks.

| Method             | R@1  | Method        | R@1   | Method         | R@1    |
|--------------------|------|---------------|-------|----------------|--------|
| Frozen [32]        | 15.0 | MEE [41]      | 26.6  | AVLnet [62]    | 30.7   |
| OA-Trans [34]      | 18.2 | MMT [29]      | 39.6  | TVLT [58]      | 32.8   |
| VLSA (ours)        | 19.3 | VLSA (ours)   | 42.5  | VLSA (ours)    | 36.3   |
| (a) Text-to-Video. |      | (b) Text-to-A | udio. | (c) Audio-to-V | Video. |

a simple baseline with two uni-modal encoders by learning text-video joint embedding; 4) Act-BERT [30]: a tangled transformer with the global-local correlation between actions and object regions; 5) HERO [63]: a hierarchical architecture combined with cross-modal and temporal transformer; 6) VidTranslate [64]: a generative method with a translation objective between modalities; 7) NoiseEstimation [65]: a multi-modal density estimation method for learning the cross-modal correspondence; 8) UniVL [66]: a unified pre-training model for video-language understanding and generation; 9) MMT [29]: a multi-modal transformer to aggregate per-frame visual features with temporal information; 10) ClipBERT [52]: a generic framework based on BERT with sparsely sampled short video clips for pre-training; 11) Frozen [32]: a curriculum learning-based joint transformer with attention modeling on both space and time; 12) SupportSet [68]: a generative method by recovering caption with a weighted combination of support visual features; 13) OA-Trans [34]: an object-aware pre-training transformer with bounding boxes and tags as guidance; 14) AllinOne [35]: a unified transformer that learns joint representations from raw video and textual inputs. ii) VA: 15) TVLT [58]: a very recent visual-audio pre-training framework with masked audio/video autoencoding and contrastive modeling to align video and audio. iii) VTA: 16) CE [28]: a collaborative model with multiple pre-trained experts on each modality. 17) AVLnet [62]: a self-supervised baseline that learns a shared audio-visual-textual embedding space directly from raw video and text signals; 18) VATT [38]: a unified transformer with the multi-modal contrastive loss on global features of each modality for learning the alignment of video-audio-text triplets.

For text-video retrieval, we report the quantitative comparisons of fine-tuning and zero-shot results in Table 2. As can be seen, we achieve the best results in terms of R@1 and R@5 compared to all baselines fine-tuned on MSR-VTT. In particular, the proposed VLSA significantly outperforms TVLT [58], the only video-audio pre-training approach with a single transformer, where we achieve the performance gains of 5.1 R@1. In the meanwhile, we only need 0.9M data for pre-training to achieve competitive results compared to the performance (32.7 R@1, 60.9 R@5, and 72.5 R@10) of OA-Trans [34], which reduces 64% of the least amount (2.5M) of pre-training data so far. When pre-trained on the same amount of data, the proposed VLSA achieves performance gains of 5.0 R@1, 8.2 R@5, and 8.3 R@10, compared to AllinOne [35], the unified video-language pre-training model. These improvements demonstrate the effectiveness of our method in text-video retrieval by enhancing video-language pre-training with synchronized audio.

When compared to Frozen [34], the current state-of-the-art model pre-trained on 5.5M data, the proposed VLSA achieves zero-shot results gains of 1.7 R@1, 1.1 R@5, and 2.2 R@10. Furthermore, our VLSA outperforms VATT [38] by 24.1 R@10, which implies the importance of incorporating audio into local-patch masked modeling to learn discriminative modality-aware representations. Meanwhile, the proposed approach with a unified transformer pre-trained on 0.9M video-audio-text triplets still achieves this significant gain, compared to VATT [38] pre-trained on 136M triplets and 2M video-audio pairs. These results validate the superiority of our method in learning compact cross-modal embeddings for zero-shot text-video retrieval.

In addition, significant gains in text-audio retrieval can be observed in Table 3. The proposed VLSA achieves the best performance in terms of all metrics for both text-to-audio and audio-to-text retrieval. When it comes to text-to-audio retrieval, our approach with a single transformer encoder obviously outperforms MMT with separate encoders pre-trained on 136M video-audio-text triplets by 2.8 R@1,

<span id="page-8-1"></span>Table 5: Ablation studies on Local-Patch Masked Modeling (LPMM) and Global Audio Matching (GAM).

| LPMM     | GAM | Text-to-Video |      |      | Text-to-Audio |      |      |  |
|----------|-----|---------------|------|------|---------------|------|------|--|
| LFIVIIVI | GAM | R@1           | R@5  | R@10 | R@1           | R@5  | R@10 |  |
| x        | Х   | 20.1          | 47.2 | 59.3 | 23.6          | 49.2 | 61.5 |  |
| ✓        | X   | 22.6          | 49.7 | 61.2 | 28.5          | 53.6 | 66.3 |  |
| X        | ✓   | 25.3          | 52.5 | 63.1 | 30.7          | 57.8 | 69.7 |  |
| ✓        | ✓   | 27.1          | 57.3 | 68.9 | 33.5          | 63.7 | 75.1 |  |

1.9 R@5, 2.9 R@10, and 2.8 R@50. This further indicates the effectiveness of the proposed unified framework in learning discriminative audio and textual representations. Results on more benchmarks are reported in Table 4.

#### <span id="page-8-0"></span>4.3 Experimental Analysis

In this part, we performed ablation studies to demonstrate the benefit of introducing the Local-Patch Masked Modeling and Global Audio Matching modules. In order to validate the effectiveness of local-patch masked modeling (LPMM) and global audio matching (GAM), we ablate the necessity of each module and report the quantitative results in Table 5. We can observe that adding LPMM highly improves the vanilla baseline without pre-training in terms of text-video (by 2.5 R@1, 2.5 R@5, and 1.9 R@10) and text-audio retrieval (by 1.9 R@1, 4.4 R@5, and 4.8 R10), which implies the benefit of LPMM in learning discriminative modality-aware embeddings. Meanwhile, introducing only GAM in the baseline also increases the retrieval results by significant gains (5.2 R@1, 5.3 R@5, and 3.8 R@10 for text-video; 7.1 R@1, 8.6 R@5, and 8.2 R10 for text-audio). More importantly, incorporating LPMM and GAM together highly raises the baseline by 7.0 R@1, 10.1 R@5, 9.6 R@10 for text-video, and 9.9 R@1, 14.5 R@5, 13.6 R10 for text-audio. These results demonstrate the importance of local-patch masked modeling and global audio matching in extracting compact semantics from video-audio-text triplets.

In Appendix E, we also conducted extensive experiments to explore the joint encoder/decoder, and compare the video-language embedding space learned by Global Audio Matching and Video-Text Matching, separately. These results demonstrate the effectiveness of incorporating synchronized audio into video-language pre-training in capturing more discriminative representations for both text-video and text-audio retrieval. Furthermore, the representations extracted by the proposed GAM in our VLSA are inter-modality compact for matching pairs, while inter-modality separable and intra-modality compact for non-matching pairs.

#### 5 Conclusion

In this work, we present VLSA, a novel and enhanced framework for video-language pre-training with synchronized audio that can jointly learn compact representations for video-audio-text triplets in a unified self-supervised transformer. We introduce local-patch masked modeling to learn modality-aware local features from a joint transformer encoder. Then, we leverage the joint encoder with global-token alignment to capture discriminative global features. Empirical experiments on five comprehensive cross-modal retrieval benchmarks demonstrate the significant advantage of our VLSA against previous video-language pre-training approaches. Our simple model pre-trained on only 0.9M data achieves competitive results on retrieval across text, video, and audio. Furthermore, qualitative visualizations vividly showcase the advantage of our VLSA in learning compact visual-textual representations.

**Broader Impact.** The proposed approach pre-trains representations of video-audio-text triplets from manually collected video datasets, which might cause the model to learn internal biases in the data. For instance, the model could fail to learn the correspondence between rare and noisy sounds. Therefore, these issues should be addressed for the deployment of real applications.

## References

- <span id="page-9-0"></span>[1] Shentong Mo and Pedro Morgado. Benchmarking weakly-supervised audio-visual sound localization. In *European Conference on Computer Vision (ECCV) Workshop*, 2022. [1](#page-0-0)
- <span id="page-9-1"></span>[2] Shentong Mo and Yapeng Tian. Semantic-aware multi-modal grouping for weakly-supervised audio-visual video parsing. In *European Conference on Computer Vision (ECCV) Workshop*, 2022. [1](#page-0-0)
- <span id="page-9-2"></span>[3] Shentong Mo, Jing Shi, and Yapeng Tian. DiffAVA: Personalized text-to-audio generation with visual alignment. *arXiv preprint arXiv:2305.12903*, 2023. [1](#page-0-0)
- <span id="page-9-3"></span>[4] Shentong Mo, Weiguo Pian, and Yapeng Tian. Class-incremental grouping network for continual audio-visual learning. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023. [1](#page-0-0)
- <span id="page-9-4"></span>[5] Weiguo Pian, Shentong Mo, Yunhui Guo, and Yapeng Tian. Audio-visual class-incremental learning. In *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023. [1](#page-0-0)
- <span id="page-9-5"></span>[6] Shentong Mo and Pedro Morgado. A unified audio-visual learning framework for localization, separation, and recognition. *arXiv preprint arXiv:2305.19458*, 2023. [1](#page-0-0)
- <span id="page-9-6"></span>[7] Shentong Mo, Jing Shi, and Yapeng Tian. Text-to-audio generation synchronized with videos. *arXiv preprint arXiv:2403.07938*, 2024. [1](#page-0-0)
- <span id="page-9-7"></span>[8] Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chenliang Xu. Audio-visual event localization in unconstrained videos. In *Proceedings of European Conference on Computer Vision (ECCV)*, 2018. [1](#page-0-0)
- <span id="page-9-8"></span>[9] Yan-Bo Lin, Yu-Jhe Li, and Yu-Chiang Frank Wang. Dual-modality seq2seq network for audio-visual event localization. In *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 2002–2006, 2019. [1](#page-0-0)
- <span id="page-9-9"></span>[10] Yu Wu, Linchao Zhu, Yan Yan, and Yi Yang. Dual attention matching for audio-visual event localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pages 6291–6299, 2019. [1](#page-0-0)
- <span id="page-9-10"></span>[11] Yan-Bo Lin and Yu-Chiang Frank Wang. Audiovisual transformer with instance attention for audio-visual event localization. In *Proceedings of the Asian Conference on Computer Vision (ACCV)*, 2020. [1](#page-0-0)
- <span id="page-9-11"></span>[12] Shentong Mo and Yapeng Tian. Multi-modal grouping network for weakly-supervised audiovisual video parsing. In *Proceedings of Advances in Neural Information Processing Systems (NeurIPS)*, 2022. [1](#page-0-0)
- <span id="page-9-12"></span>[13] Shentong Mo and Pedro Morgado. Unveiling the power of audio-visual early fusion transformers with dense interactions through masked modeling. *arXiv preprint arXiv:2312.01017*, 2023. [1](#page-0-0)
- <span id="page-9-13"></span>[14] Pedro Morgado, Nuno Nvasconcelos, Timothy Langlois, and Oliver Wang. Self-supervised generation of spatial audio for 360°video. In *Proceedings of Advances in Neural Information Processing Systems (NeurIPS)*, 2018. [1](#page-0-0)
- <span id="page-9-14"></span>[15] Ruohan Gao and Kristen Grauman. 2.5d visual sound. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 324–333, 2019. [1](#page-0-0)
- <span id="page-9-15"></span>[16] Changan Chen, Unnat Jain, Carl Schissler, S. V. A. Garí, Ziad Al-Halah, Vamsi Krishna Ithapu, Philip Robinson, and Kristen Grauman. Soundspaces: Audio-visual navigation in 3d environments. In *Proceedings of European Conference on Computer Vision (ECCV)*, pages 17–36, 2020. [1](#page-0-0)
- <span id="page-9-16"></span>[17] Pedro Morgado, Yi Li, and Nuno Nvasconcelos. Learning representations from audio-visual spatial alignment. In *Proceedings of Advances in Neural Information Processing Systems (NeurIPS)*, pages 4733–4744, 2020. [1](#page-0-0)
- <span id="page-9-17"></span>[18] Arda Senocak, Tae-Hyun Oh, Junsik Kim, Ming-Hsuan Yang, and In So Kweon. Learning to localize sound source in visual scenes. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4358–4366, 2018. [1](#page-0-0)
- <span id="page-9-18"></span>[19] Di Hu, Feiping Nie, and Xuelong Li. Deep multimodal clustering for unsupervised audiovisual learning. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 9248–9257, 2019. [1](#page-0-0)

- <span id="page-10-0"></span>[20] Triantafyllos Afouras, Andrew Owens, Joon Son Chung, and Andrew Zisserman. Selfsupervised learning of audio-visual objects from video. In *Proceedings of European Conference on Computer Vision (ECCV)*, pages 208–224, 2020. [1](#page-0-0)
- <span id="page-10-1"></span>[21] Rui Qian, Di Hu, Heinrich Dinkel, Mengyue Wu, Ning Xu, and Weiyao Lin. Multiple sound sources localization from coarse to fine. In *Proceedings of European Conference on Computer Vision (ECCV)*, pages 292–308, 2020. [1](#page-0-0)
- <span id="page-10-2"></span>[22] Honglie Chen, Weidi Xie, Triantafyllos Afouras, Arsha Nagrani, Andrea Vedaldi, and Andrew Zisserman. Localizing visual sounds the hard way. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 16867–16876, 2021. [1](#page-0-0)
- <span id="page-10-3"></span>[23] Shentong Mo and Pedro Morgado. Localizing visual sounds the easy way. *arXiv preprint arXiv:2203.09324*, 2022. [1](#page-0-0)
- <span id="page-10-4"></span>[24] Shentong Mo and Pedro Morgado. A closer look at weakly-supervised audio-visual source localization. In *Proceedings of Advances in Neural Information Processing Systems (NeurIPS)*, 2022. [1](#page-0-0)
- <span id="page-10-5"></span>[25] Shentong Mo and Yapeng Tian. Audio-visual grouping network for sound localization from mixtures. *arXiv preprint arXiv:2303.17056*, 2023. [1](#page-0-0)
- <span id="page-10-6"></span>[26] Shentong Mo and Yapeng Tian. AV-SAM: Segment anything model meets audio-visual localization and segmentation. *arXiv preprint arXiv:2305.01836*, 2023. [1](#page-0-0)
- <span id="page-10-7"></span>[27] Shentong Mo and Bhiksha Raj. Weakly-supervised audio-visual segmentation. In *Proceedings of Advances in Neural Information Processing Systems (NeurIPS)*, 2023. [1](#page-0-0)
- <span id="page-10-8"></span>[28] Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. Use what you have: Video retrieval using representations from collaborative experts. In *Proceedings of British Machine Vision Conference (BMVC)*, 2019. [1,](#page-0-0) [3,](#page-2-1) [7,](#page-6-1) [8,](#page-7-2) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-10-9"></span>[29] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. Multi-modal Transformer for Video Retrieval. In *Proceedings of European Conference on Computer Vision (ECCV)*, 2020. [1,](#page-0-0) [3,](#page-2-1) [7,](#page-6-1) [8,](#page-7-2) [15](#page-14-0)
- <span id="page-10-10"></span>[30] Linchao Zhu and Yi Yang. Actbert: Learning global-local video-text representations. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 8743–8752, 2020. [1,](#page-0-0) [3,](#page-2-1) [7,](#page-6-1) [8,](#page-7-2) [15](#page-14-0)
- <span id="page-10-11"></span>[31] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. VideoBERT: A joint model for video and language representation learning. *arXiv preprint arXiv:1904.01766*, 2019. [1,](#page-0-0) [3](#page-2-1)
- <span id="page-10-12"></span>[32] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In *Proceedings of IEEE International Conference on Computer Vision (ICCV)*, 2021. [1,](#page-0-0) [3,](#page-2-1) [4,](#page-3-1) [7,](#page-6-1) [8,](#page-7-2) [15,](#page-14-0) [17](#page-16-1)
- <span id="page-10-13"></span>[33] Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, and Dong Shen. Improving videotext retrieval by multi-stream corpus alignment and dual softmax loss. *arXiv preprint arXiv:2109.04290*, 2021. [1,](#page-0-0) [3](#page-2-1)
- <span id="page-10-14"></span>[34] Alex Jinpeng Wang, Yixiao Ge, Guanyu Cai, Rui Yan, Xudong Lin, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Object-aware video-language pre-training for retrieval. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. [1,](#page-0-0) [3,](#page-2-1) [4,](#page-3-1) [7,](#page-6-1) [8,](#page-7-2) [15,](#page-14-0) [17,](#page-16-1) [18](#page-17-0)
- <span id="page-10-15"></span>[35] Alex Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. All in one: Exploring unified video-language pre-training. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023. [1,](#page-0-0) [3,](#page-2-1) [4,](#page-3-1) [7,](#page-6-1) [8,](#page-7-2) [15](#page-14-0)
- <span id="page-10-16"></span>[36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. *arXiv preprint arXiv:2103.00020*, 2021. [1,](#page-0-0) [3,](#page-2-1) [18](#page-17-0)
- <span id="page-10-17"></span>[37] Zijian Gao, Jingyun Liu, Sheng Chen, Dedan Chang, Hao Zhang, and Jinwei Yuan. Clip2tv: An empirical study on transformer-based methods for video-text retrieval. *arXiv preprint arXiv:2111.05610*, 2021. [1,](#page-0-0) [3](#page-2-1)

- <span id="page-11-0"></span>[38] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. VATT: Transformers for multimodal self-supervised learning from raw video, audio and text. In *Proceedings of Advances in Neural Information Processing Systems (NeurIPS)*, 2021. [2,](#page-1-0) [3,](#page-2-1) [7,](#page-6-1) [8,](#page-7-2) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-11-1"></span>[39] Andrey Guzhov, Federico Raue, Jörn Hees, and Andreas Dengel. Audioclip: Extending clip to image, text and audio. *arXiv preprint arXiv:2106.13043*, 2021. [2,](#page-1-0) [3](#page-2-1)
- <span id="page-11-2"></span>[40] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot reserve: Neural script knowledge through vision and language and sound. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 16354–16366, 2022. [2,](#page-1-0) [3](#page-2-1)
- <span id="page-11-3"></span>[41] Antoine Miech, Ivan Laptev, and Josef Sivic. Learning a text-video embedding from incomplete and heterogeneous data. *arXiv preprint arXiv:1804.02516*, 2018. [3,](#page-2-1) [7,](#page-6-1) [8,](#page-7-2) [15](#page-14-0)
- <span id="page-11-4"></span>[42] Di Qi, Lin Su, Jia Song, Edward Cui, Taroon Bharti, and Arun Sacheti. ImageBERT: cross-modal pre-training with large-scale weak-supervised image-text data. *arXiv preprint arXiv:2001.07966*, 2020. [3](#page-2-1)
- <span id="page-11-5"></span>[43] Dandan Song, Siyi Ma, Zhanchen Sun, Sicheng Yang, and Lejian Liao. KVL-BERT: knowledge enhanced visual-and-linguistic BERT for visual commonsense reasoning. *arXiv preprint arXiv:2012.07000*, 2020. [3](#page-2-1)
- <span id="page-11-6"></span>[44] Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. Ernie-vil: Knowledge enhanced vision-language representations through scene graph. *arXiv preprint arXiv:2006.16934*, 2020. [3](#page-2-1)
- <span id="page-11-7"></span>[45] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. UNITER: learning universal image-text representations. *arXiv preprint arXiv:1909.11740*, 2019. [3](#page-2-1)
- <span id="page-11-8"></span>[46] Hao Tan and Mohit Bansal. LXMERT: Learning cross-modality encoder representations from transformers. *arXiv preprint arXiv:1908.07490*, 2019. [3](#page-2-1)
- <span id="page-11-9"></span>[47] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. *arXiv preprint arXiv:1908.03557*, 2019. [3,](#page-2-1) [4](#page-3-1)
- <span id="page-11-10"></span>[48] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In *Advances in Neural Information Processing Systems*, pages 13–23, 2019. [3,](#page-2-1) [4](#page-3-1)
- <span id="page-11-11"></span>[49] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. VL-BERT: Pretraining of generic visual-linguistic representations. In *International Conference on Learning Representations*, 2020. [3,](#page-2-1) [4](#page-3-1)
- <span id="page-11-12"></span>[50] Lei Shi, Kai Shuang, Shijie Geng, Peng Su, Zhengkai Jiang, Peng Gao, Zuohui Fu, Gerard de Melo, and Sen Su. Contrastive visual-linguistic pretraining. *arXiv preprint arXiv:2007.13135*, 2020. [3,](#page-2-1) [4](#page-3-1)
- <span id="page-11-13"></span>[51] Yicong Hong, Qi Wu, Yuankai Qi, Cristian Rodriguez-Opazo, and Stephen Gould. VLN BERT: a recurrent vision-and-language bert for navigation. In *Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021. [3,](#page-2-1) [4](#page-3-1)
- <span id="page-11-14"></span>[52] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit Bansal, and Jingjing Liu. Less is More: clipbert for video-and-language learning via sparse sampling. In *Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021. [3,](#page-2-1) [4,](#page-3-1) [7,](#page-6-1) [8,](#page-7-2) [15](#page-14-0)
- <span id="page-11-15"></span>[53] Yuying Ge, Yixiao Ge, Xihui Liu, Alex Jinpeng Wang, Jianping Wu, Ying Shan, Xiaohu Qie, and Ping Luo. Miles: Visual bert pre-training with injected language semantics for video-text retrieval. In *Proceedings of European Conference on Computer Vision (ECCV)*, page 691–708, 2022. [3](#page-2-1)
- <span id="page-11-16"></span>[54] A.-M. Oncescu, A.S. Koepke, J. Henriques, Z. Akata, and S. Albanie. Audio retrieval with natural language queries. In *Proceedings of Interspeech*, 2021. [3](#page-2-1)
- <span id="page-11-17"></span>[55] A.S. Koepke, A.-M. Oncescu, J. Henriques, Z. Akata, and S. Albanie. Audio retrieval with natural language queries: A benchmark study. *IEEE Transactions on Multimedia*, 2022. [3,](#page-2-1) [7](#page-6-1)

- <span id="page-12-0"></span>[56] Rui Yan, Mike Zheng Shou, Yixiao Ge, Alex Jinpeng Wang, Xudong Lin, Guanyu Cai, and Jinhui Tang. Video-text pre-training with learned regions. *arXiv preprint arXiv:2112.01194*, 2021. [3](#page-2-1)
- <span id="page-12-1"></span>[57] Ziyi Yang, Yuwei Fang, Chenguang Zhu, Reid Pryzant, Dongdong Chen, Yu Shi, Yichong Xu, Yao Qian, Mei Gao, Yi-Ling Chen, Liyang Lu, Yujia Xie, Robert Gmyr, Noel C. F. Codella, Naoyuki Kanda, Bin Xiao, Yuanxun Lu, Takuya Yoshioka, Michael Zeng, and Xuedong Huang. i-code: An integrative and composable multimodal learning framework. *arXiv preprint arXiv:2205.01818*, 2022. [3](#page-2-1)
- <span id="page-12-2"></span>[58] Zineng Tang, Jaemin Cho, Yixin Nie, and Mohit Bansal. Tvlt: Textless vision-language transformer. In *Proceedings of Advances in Neural Information Processing Systems (NeurIPS)*, 2022. [4,](#page-3-1) [7,](#page-6-1) [8,](#page-7-2) [15,](#page-14-0) [17](#page-16-1)
- <span id="page-12-3"></span>[59] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018. [6,](#page-5-1) [7](#page-6-1)
- <span id="page-12-7"></span>[60] Youngjae Yu, Jongseok Kim, and Gunhee Kim. A joint sequence fusion model for video question answering and retrieval. In *Proceedings of European Conference on Computer Vision (ECCV)*, pages 471–487, 2018. [7,](#page-6-1) [15](#page-14-0)
- <span id="page-12-4"></span>[61] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 2630–2640, 2019. [6,](#page-5-1) [7,](#page-6-1) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-12-8"></span>[62] Andrew Rouditchenko, Angie Boggust, David Harwath, Brian Chen, Dhiraj Joshi, Samuel Thomas, Kartik Audhkhasi, Hilde Kuehne, Rameswar Panda, Rogerio Feris, et al. Avlnet: Learning audio-visual language representations from instructional videos. *arXiv preprint arXiv:2006.09199*, 2020. [7,](#page-6-1) [8,](#page-7-2) [15,](#page-14-0) [16,](#page-15-0) [17](#page-16-1)
- <span id="page-12-9"></span>[63] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. HERO: Hierarchical encoder for Video+Language omni-representation pre-training. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 2046–2065, 2020. [7,](#page-6-1) [8,](#page-7-2) [15](#page-14-0)
- <span id="page-12-10"></span>[64] Bruno Korbar, Fabio Petroni, Rohit Girdhar, and Lorenzo Torresani. Video understanding as machine translation. *arXiv preprint arXiv:2006.07203*, 2020. [7,](#page-6-1) [8,](#page-7-2) [15](#page-14-0)
- <span id="page-12-11"></span>[65] Elad Amrani, Rami Ben-Ari, Daniel Rotman, and Alexander M. Bronstein. Noise estimation using density estimation for self-supervised multimodal learning. *arXiv preprint arXiv:2003.03186*, 2020. [7,](#page-6-1) [8,](#page-7-2) [15](#page-14-0)
- <span id="page-12-12"></span>[66] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan Duan, Tianrui Li, Jason Li, Taroon Bharti, and Ming Zhou. Univl: A unified video and language pre-training model for multimodal understanding and generation. *arXiv preprint arXiv:2002.06353*, 2020. [7,](#page-6-1) [8,](#page-7-2) [15](#page-14-0)
- <span id="page-12-13"></span>[67] Nina Shvetsova, Brian Chen, Andrew Rouditchenko, Samuel Thomas, Brian Kingsbury, Rogerio S Feris, David Harwath, James Glass, and Hilde Kuehne. Everything at once-multi-modal fusion transformer for video retrieval. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 20020–20029, 2022. [7,](#page-6-1) [15,](#page-14-0) [16,](#page-15-0) [17](#page-16-1)
- <span id="page-12-14"></span>[68] Mandela Patrick, Po-Yao Huang, Yuki Asano, Florian Metze, Alexander G Hauptmann, Joao F. Henriques, and Andrea Vedaldi. Support-set bottlenecks for video-text representation learning. In *Proceedings of International Conference on Learning Representations (ICLR)*, 2021. [7,](#page-6-1) [8,](#page-7-2) [15](#page-14-0)
- <span id="page-12-5"></span>[69] Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In *Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2017. [6,](#page-5-1) [16](#page-15-0)
- <span id="page-12-6"></span>[70] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In *Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 5288–5296, 2016. [6](#page-5-1)
- <span id="page-12-15"></span>[71] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 3202–3212, June 2015. [7](#page-6-1)

- <span id="page-13-0"></span>[72] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. AudioCaps: Generating captions for audios in the wild. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 119–132, 2019. [7](#page-6-1)
- <span id="page-13-1"></span>[73] Luowei Zhou, Chenliang Xu, and Jason J. Corso. Towards automatic learning of procedures from web instructional videos. In *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence*, page 7590–7598, 2018. [7](#page-6-1)
- <span id="page-13-2"></span>[74] Hang Zhao, Chuang Gan, Andrew Rouditchenko, Carl Vondrick, Josh McDermott, and Antonio Torralba. The sound of pixels. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 570–586, 2018. [7](#page-6-1)
- <span id="page-13-3"></span>[75] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross B. Girshick. Masked autoencoders are scalable vision learners. *arXiv preprint arXiv:2111.06377*, 2021. [7](#page-6-1)
- <span id="page-13-4"></span>[76] Alan Baade, Puyuan Peng, and David F. Harwath. MAE-AST: masked autoencoding audio spectrogram transformer. *arXiv preprint arXiv:2203.16691*, 2022. [7](#page-6-1)
- <span id="page-13-5"></span>[77] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020. [7](#page-6-1)
- <span id="page-13-6"></span>[78] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *Proceedings of International Conference on Learning Representations (ICLR)*, 2019. [7](#page-6-1)
- <span id="page-13-7"></span>[79] Jean-Baptiste Alayrac, Adrià Recasens, Rosalia Schneider, Relja Arandjelovic, Jason Ramapu- ´ ram, Jeffrey De Fauw, Lucas Smaira, Sander Dieleman, and Andrew Zisserman. Self-Supervised MultiModal Versatile Networks. In *Proceedings of Advances in Neural Information Processing Systems (NeurIPS)*, 2020. [15,](#page-14-0) [16,](#page-15-0) [17](#page-16-1)
- <span id="page-13-8"></span>[80] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. *Journal of Machine Learning Research*, 9(86):2579–2605, 2008. [18](#page-17-0)

## <span id="page-14-0"></span>Appendix

In this appendix, we provide significant differences between our VLSA and current video-text-audio pre-training baselines. In addition, we compare the proposed VLSA with those baselines in terms of text-to-video retrieval on fine-tuned and zero-shot settings. Finally, we report the quantitative comparison of computation costs with state-of-the-art methods.

## A Video-Text-Audio Pre-training Baselines

We conduct a comprehensive experimental study of existing approaches with video-text-audio pretraining. Namely, we considered:

- JSFusion [\[60\]](#page-12-7) (2018'ECCV): a very early sequence fusion model with multi-modal matching;
- MEE [\[41\]](#page-11-3) (2018'arXiv): a baseline with mixed embedding experts for visual-textual modalities;
- HT [\[61\]](#page-12-4) (2019'ICCV): a simple baseline with two uni-modal encoders by learning text-video joint embedding;
- CE [\[28\]](#page-10-8) (2019'BMVC): a simple baseline based on a collaborative model with multiple pre-trained experts on each modality to learn a joint video-text embedding.
- MMV [\[79\]](#page-13-7) (2020'NeurIPS): a multimodal versatile network with a deflation process to integrate text with audio-visual representations into a common embedding space.
- AVLnet [\[62\]](#page-12-8) (2021'Interspeech): a self-supervised baseline that learns a shared audio-visualtextual embedding space directly from raw video and text signals;
- ActBERT [\[30\]](#page-10-10) (2020'CVPR): a tangled transformer with the global-local correlation between actions and object regions;
- HERO [\[63\]](#page-12-9) (2020'EMNLP): a hierarchical architecture combined with cross-modal and temporal transformer;
- VidTranslate [\[64\]](#page-12-10) (2020'arXiv): a generative method with a translation objective between modalities;
- UniVL [\[66\]](#page-12-12) (2020'arXiv): a unified pre-training model for video-language understanding and generation;
- MMT [\[29\]](#page-10-9) (2020'ECCV): a multi-modal transformer to aggregate per-frame visual features with temporal information;
- NoiseEstimation [\[65\]](#page-12-11) (2021'AAAI): a multi-modal density estimation method for learning the cross-modal correspondence;
- ClipBERT [\[52\]](#page-11-14) (2021'CVPR): a generic framework based on BERT with sparsely sampled short video clips for pre-training;
- Frozen [\[32\]](#page-10-12) (2021'ICCV): a curriculum learning-based joint transformer with attention modeling on both space and time;
- SupportSet [\[68\]](#page-12-14) (2021'ICLR): a generative method by recovering caption with a weighted combination of support visual features;
- VATT [\[38\]](#page-11-0) (2021'NeurIPS): a unified transformer with the multi-modal contrastive loss on global features of each modality for learning the alignment of video-audio-text triplets.
- OA-Trans [\[34\]](#page-10-14) (2022'CVPR): an object-aware pre-training transformer with bounding boxes and tags as guidance;
- TVLT [\[58\]](#page-12-2) (2022'NeurIPS): a very recent visual-audio pre-training framework with masked audio/video autoencoding and contrastive modeling to align video and audio.
- Everything-At-Once [\[67\]](#page-12-13) (2022'CVPR): a modality agnostic fusion transformer that integrates embeddings from three separate encoders into a fused representation in a joined multi-modal embedding space;
- AllinOne [\[35\]](#page-10-15) (2023'CVPR): a unified transformer that learns joint representations from raw video and textual inputs.

<span id="page-15-1"></span><span id="page-15-0"></span>Table 6: **Quantitative results of text-to-video retrieval on MSR-VTT dataset.** "Single", "Dual", and "Triple" refer to one joint encoder, two separate, and three separate encoders. Bold denotes the best results.

| Method                  | Modalities         | Architecture | Pre-train Datasets | Data Size $(\downarrow)$ | R@1 (↑) | R@5 (↑) | R@10 (↑) |
|-------------------------|--------------------|--------------|--------------------|--------------------------|---------|---------|----------|
| CE [28]                 | Video, Text, Audio | Triple       | YouTube-8M         | 8M                       | 20.9    | 48.8    | 62.4     |
| AVLnet [62]             | Video, Text, Audio | Triple       | HowTo100M          | 136M                     | 27.1    | 55.6    | 66.6     |
| Everything-At-Once [67] | Video, Text, Audio | Triple       | HowTo100M          | 136M                     | 23.7    | 52.1    | 63.7     |
| VLSA (ours)             | Video, Text, Audio | Single       | AudioSet∩HowTo100M | 0.9M                     | 27.1    | 57.3    | 68.9     |
| zero-shot:              |                    |              |                    |                          |         |         |          |
| VATT [38]               | Video, Text, Audio | Single       | AudioSet+HowTo100M | 138M                     | -       | -       | 29.7     |
| MMV [79]                | Video, Text, Audio | Triple       | AudioSet+HowTo100M | 138M                     | 9.3     | 23.0    | 31.1     |
| Everything-At-Once [67] | Video, Text, Audio | Triple       | HowTo100M          | 136M                     | 9.9     | 24.0    | 32.6     |
| VLSA (ours)             | Video, Text, Audio | Single       | AudioSet∩HowTo100M | 0.9M                     | 20.4    | 40.6    | 53.8     |

#### **B** Differences between VLSA and Video-Text-Audio Pre-training Baselines

When compared to previous video-text-audio pre-training baselines, there are three significant distinct characteristics of our VLSA for addressing video-language pre-training problems, which are highlighted as follows:

- 1) Achieving competitive performance with only 0.9M triplets for pre-training. The major difference is that we use only 0.9M video-text-audio triplets for pre-training to learn disentangled video-text representations for retrieval. However, current video-text-audio pre-training approaches used 8~138M triplets, and most methods utilized all data in the whole HowTo100M [61] benchmark. Meanwhile, some strong baselines, such as VATT [38] and MMV [79], combined AudioSet [69] with HowTo100M [61] to pre-train a total of 138M data, as shown in Table 6.
- 2) Replacing Video-Text Matching with Global Audio Matching. We introduce global audio matching to replace video-text matching for extracting disentangled visual-textual representations from video-text-audio triplets. However, all previous works mainly utilized video-text matching on video frames and high-level textual semantics. Since most frames in a video look similar in high-level textual semantics, they would bring false alignment pairs during pre-training. Different from them, we leverage the global audio-video and audio-text matching to explicitly learn the cross-modal alignment between synchronized audio and video frames/captions, as one audio spectrogram is distinct in each audio-video/text pair.
- 3) Incorporating Local-Patch Masked Modeling for all modalities. We apply local-patch masked modeling on each modality through a single unified encoder and three separate decoders with shared parameters, but those video-text-audio pre-training baselines used the contrastive loss to learn the global alignment across various modalities in the joint embedding space. They do not involve the explicit masked modeling mechanism to learn local modality-aware features across three different modalities. In addition, they extracted local modality-aware embeddings through three separate encoders without shared parameters, except that VATT [38] introduced the unified encoder to extract modality-agnostic embeddings from raw signals. However, VATT [38] performs worse on the zero-shot text-to-video retrieval than our VLSA with local-patch masked modeling.

# C Comparison Results with Previous Video-Text-Audio Pre-training Baselines

Table 6 reports the comparison results with video-text-audio pre-training baselines on fine-tuned and zero-shot text-to-video retrieval on the MSR-VTT dataset. We can observe that the proposed VLSA achieves the best performance in terms of R@1, R@5, and R@10 compared to all baselines fine-tuned on MSR-VTT. In particular, our VLSA significantly outperforms Everything-At-Once [67], the recent video-text-audio pre-training approach with global alignment across three different modalities, where the performance gains of 3.4 R@1, 5.2 R@5, and 5.2 R@10 are achieved. Meanwhile, the proposed VLSA only needs 0.9M triplets for pre-training to achieve competitive results while they used the whole HowTo100M [61] with 136M triplets. Compared to CE [28] pre-trained on the data with the comparable size, we achieve performance gains of 6.2 R@1, 8.5 R@5, and 6.5 R@10, as they involved neither global audio matching nor local-patch masked modeling. These improvements validate the superiority of the proposed VLSA in enhancing video-language representations for text-video retrieval by pre-training with synchronized audio.

<span id="page-16-2"></span><span id="page-16-1"></span>

| Table 7: Quantitative  | results of computation | costs. Lower is better. |
|------------------------|------------------------|-------------------------|
| Table /. Qualiticative | results of Combutation | LOSIS. LOWEL IS DELICI. |

| Method             | Params | GPU<br>Hours | Infer<br>Latency (ms) |
|--------------------|--------|--------------|-----------------------|
| AVLnet [62] + text | 353M   | _            | 2316                  |
| TVLT [58] + text   | 283M   | _            | 2135                  |
| Frozen [32]        | 232M   | 10580        | 1989                  |
| OA-Trans [34]      | 232M   | 7680         | 1946                  |
| VLSA (ours)        | 156M   | 1350         | 1738                  |

In addition, significant gains in zero-shot text-audio retrieval can be observed. Compared to Everything-At-Once [67], we achieve significant zero-shot performance gains of 10.5 R@1, 16.6 R@5, and 21.2 R@10, which implies the importance of introducing local-patch masked modeling to learn discriminative modality-aware representations across different modalities. Meanwhile, the proposed VLSA with a unified transformer pre-trained on 0.9M video-audio-text triplets significantly outperforms MMV [79] pre-trained on 136M triplets and 2M video-audio pairs. These results further demonstrate the effectiveness of our method with global audio matching in learning compact and disentangled cross-modal embeddings for zero-shot text-video retrieval.

## D Computation Costs

For computation costs, we used 8 V100-32GB GPUs for pre-training and fine-tuning. It should be noted that the proposed LPMM & GAM modules in our VLSA take single pass cost during pre-training such that we bring much fewer computation costs with only 0.9M video-audio-transcript triplets. We report quantitative comparison results of parameters, pre-training GPU hours, and inference latency in Table 7. As can be seen, we achieve the lowest parameters compared to previous video-language pre-training baselines. In particular, the proposed VLSA significantly decreases the parameters of OA-Trans [34], the current state-of-the-art method, by 76M. Moreover, we achieve superior performance gains compared to TVLT [58], the current state-of-the-art video-audio pre-training baseline, which implies the importance of a joint transformer encoder in aggregating local and global visual-textual representations.

Meanwhile, our VLSA outperforms strong video-language pre-training approaches, Frozen [32] and OA-Trans [34], by large margins, where we achieve the pre-training cost reduction of 9230 GPU hours and 6330 GPU hours. Furthermore, when evaluating the inference latency, the proposed approach still outperforms OA-Trans [34] by 208 ms. We also achieve highly better results against TVLT [58], the joint video-audio pre-training baseline. These significant cost reductions demonstrate the efficiency of our method in pre-training a joint transformer encoder on only 0.9M video-audio-transcript triplets.

#### <span id="page-16-0"></span>**E** More Experimental Analysis

In this section, we also conducted extensive experiments to explore the joint encoder/decoder, and compare the video-language embedding space learned by Global Audio Matching and Video-Text Matching, separately.

Joint Encoder. In order to explore the effect of the modality types in a single joint encoder on both text-video and text-audio performance, we vary the combination types from {A+V+T, V+T, A+T, A+V, None}, where "None" denotes that three modality-specific encoders are used. A, V, and T denote audio, video, and text, respectively. The comparison results of the retrieval performance are shown in Figure 3. As can be seen, combining any two modalities among audio, video, and text in a joint encoder increases the results of the vanilla baseline with modality-specific encoders, which implies the importance

<span id="page-16-3"></span>![](_page_16_Figure_9.jpeg)

Figure 3: **Effect of modality types in a single joint encoder.** A, V, and T denote audio, video, and text, respectively.

of the proposed joint encoder in learning cross-modal representations for retrieval tasks. In addition,

<span id="page-17-0"></span>adding audio to the V+T joint encoder significantly outperforms the baseline by 5.6 R@1, 7.6 R@5, 6.8R@10 for text-video, and 6.7 R@1, 9.9 R@5, 8.0R@10 for text-audio. These results demonstrate the effectiveness of incorporating synchronized audio into video-language pre-training in capturing more discriminative representations for both text-video and text-audio retrieval.

<span id="page-17-1"></span>![](_page_17_Figure_1.jpeg)

Figure 4: Effect of modality types with parameter-shared decoder. A, V, and T denote audio, video, and text, respectively.

Parameter-Shared Decoders. In order to explore how the parameter-shared decoder affects the final retrieval performance, we ablate the modality types from {A+V+T, V+T, A+T, A+V, None}, where "None" denotes that three separate decoders do not share parameters. Figure [4](#page-17-1) reports the comparison results on both text-video and text-audio retrieval. We can observe that pretraining with parameter-shared decoders of any two combined modalities outperforms three separate decoders without parameters shared. In the meanwhile, using three separate decoders with parameters simultaneously shared among three modalities achieves the best performance. This

further validates the rationality of local-patch masked modeling with three separate parameter-shared decoders in enhancing video-language pre-training for retrieval tasks.

Global Audio Matching vs. Video-Text Matching. Learning discriminative video-language semantic representations is essential for us to achieve higher performance in retrieval tasks. To better evaluate the quality of video-language representations learned by global audio matching (GAM) and video-text matching (VTM), we visualize learned visual-textual representations of 1000 randomly selected matching and non-matching pairs from MSR-VTT in a common space by t-SNE [\[80\]](#page-13-8), as shown in Figure [5.](#page-18-0) As can be seen in the last column, features extracted by the proposed GAM are inter-modality compact for matching pairs. More importantly, GAM representations are inter-modality separable and intra-modality compact for non-matching pairs. These meaningful visualization results further demonstrate that our VLSA successfully extracts compact visual-textual representations during pre-training. Note that VTM achieved 23.5 R@1, 50.6 R@5, and 61.8 R@10, which is much lower than GAM (27.1 R@1, 57.3 R@5, 68.9 R@10).

## F Limitation

Although the proposed VLSA achieves superior results on both text-video and text-audio retrieval, the fine-tuning gains of our approach over R@1 and R@5 on text-video retrieval are not significant. One possible solution is to leverage CLIP [\[36\]](#page-10-16) pre-trained weights and fine-grained visual features for masking (such as object bounding boxes and tags), similar to OA-Trans [\[34\]](#page-10-14) for boosting performance. Meanwhile, we notice that if we continue training for more steps, it would be hard to see significant gains on downstream tasks. The primary cause is that we have a limited amount of video-audio-text triplets for training. Therefore, the future work is potentially to gather more triplets or to explore continual learning when it comes to new data.

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

Figure 5: Qualitative comparisons of visual-textual representations learned by VTM and GAM for matching (Top Row) and non-matching pairs (Bottom Row). Note that each spot denotes the visual/textual feature of one video/caption, and each color refers to one modality (yellow for video, green for text). The VLSA representations are much better.