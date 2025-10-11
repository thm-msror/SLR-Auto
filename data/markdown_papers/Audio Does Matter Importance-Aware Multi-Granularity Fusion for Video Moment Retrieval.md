## Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval

Junan Lin<sup>∗</sup> Zhejiang University Hangzhou, China linja@zju.edu.cn

Xiaoye Qu Shanghai Artificial Intelligence Laboratory Shanghai, China xiaoye@hust.edu.cn

Daizong Liu<sup>∗</sup> Peking University Beijing, China dzliu@stu.pku.edu.cn

Xun Yang University of Science and Technology of China Hefei, China hfutyangxun@gmail.com

Xianke Chen Zhejiang Gongshang University Hangzhou, China a397283164@163.com

Jixiang Zhu Zhejiang Gongshang University Hangzhou, China zhujx@mail.zjgsu.edu.cn

Sanyuan Zhang Zhejiang University Hangzhou, China syzhang@zju.edu.cn

## Abstract

Video Moment Retrieval (VMR) aims to retrieve a specific moment semantically related to the given query. To tackle this task, most existing VMR methods solely focus on the visual and textual modalities while neglecting the complementary but important audio modality. Although a few recent works try to tackle the joint audiovision-text reasoning, they treat all modalities equally and simply embed them without fine-grained interaction for moment retrieval. These designs are counter-practical as: Not all audios are helpful for video moment retrieval, and the audio of some videos may be complete noise or background sound that is meaningless to the moment determination. To this end, we propose a novel Importance-aware Multi-Granularity fusion model (IMG), which learns to dynamically and selectively aggregate the audio-vision-text contexts for VMR. Specifically, after integrating the textual guidance with vision and audio separately, we first design a pseudo-label-supervised audio importance predictor that predicts the importance score of the audio, and accordingly assigns weights to mitigate the interference caused by noisy audio. Then, we design a multi-granularity audio fusion module that adaptively fuses audio and visual modalities at local-, event-, and global-level, fully capturing their complementary contexts. We further propose a cross-modal knowledge distillation strategy to address the challenge of missing audio modality during inference. To evaluate our method, we further construct a new VMR

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

MM '25, October 27–31, 2025, Dublin, Ireland.

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2035-2/2025/10

<https://doi.org/10.1145/3746027.3754982>

Jianfeng Dong† Zhejiang Gongshang University Hangzhou, China dongjf24@gmail.com

dataset, i.e., Charades-AudioMatter, where audio-related samples are manually selected and re-organized from the original Charades-STA to validate the model's capability in utilizing audio modality. Extensive experiments validate the effectiveness of our method, achieving state-of-the-art with audio-video fusion in VMR methods. Our code is available at [https://github.com/HuiGuanLab/IMG.](https://github.com/HuiGuanLab/IMG)

### CCS Concepts

• Information systems → Multimedia and multimodal retrieval; Video search.

## Keywords

Video Moment Retrieval; Video Understanding; Multimodal Learning; Cross-Modal Alignment

## ACM Reference Format:

Junan Lin, Daizong Liu, Xianke Chen, Xiaoye Qu, Xun Yang, Jixiang Zhu, Sanyuan Zhang, and Jianfeng Dong. 2025. Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval. In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, [14](#page-13-0) pages. [https:](https://doi.org/10.1145/3746027.3754982) [//doi.org/10.1145/3746027.3754982](https://doi.org/10.1145/3746027.3754982)

### 1 Introduction

Video Moment Retrieval (VMR) [\[1,](#page-8-0) [12,](#page-8-1) [16,](#page-8-2) [64,](#page-9-0) [64\]](#page-9-0) aims to retrieve the part of the video that is relevant to the semantic of a given query. As a fundamental yet important task, it requires in-depth interaction between video and text semantics for accurate alignment and reasoning. Existing mainstream works [\[11,](#page-8-3) [13,](#page-8-4) [62,](#page-9-1) [63,](#page-9-2) [68,](#page-9-3) [69,](#page-9-4) [73,](#page-9-5) [74\]](#page-9-6) generally focus on naive visual and textual modalities and develop vision-text integration frameworks to retrieve the specific moment. However, in addition to the visual contexts, audio modality also contains valuable contexts within the video streams [\[3,](#page-8-5) [18,](#page-8-6) [25,](#page-8-7) [39,](#page-8-8) [40,](#page-8-9) [42,](#page-8-10) [77\]](#page-9-7). Without considering the rich complementary contexts of the audio modality, previous VMR methods fail to distinguish different activities like "laughing" and "talking" that share a similar

<sup>∗</sup>Both authors contributed equally to this research.

<sup>†</sup>Corresponding author.

visual appearance. Therefore, exploring the interaction and fusion of audio, vision, and text modalities in VMR is a promising direction with research forward.

To leverage audio information, several audio-based VMR methods [\[8,](#page-8-11) [9,](#page-8-12) [36\]](#page-8-13) have been proposed. However, these approaches typically extract features from audio, vision, and text modalities and apply a uniform aggregation strategy for joint reasoning, without considering their diverse contributions. For instance, PMI-LOC [\[9\]](#page-8-12) incorporates RGB, motion, and audio modalities, establishes pairwise interactions between modalities, UMT [\[36\]](#page-8-13) introduces a unified multimodal transformer framework for the integration of visual and audio information, while ADPN [\[8\]](#page-8-11) leverages the consistency and complementarity between audio and visual modalities for efficient audio fusion. Although these methods achieve relatively better performance than conventional VMR methods, they neglect that not all audios contribute to the final grounding as the audio of some videos may be complete noise or background sound.

In practice, audio semantics exhibit considerable complexity and diversity, varying significantly across different scenarios. In certain instances, audio context serves as a valuable complement, enhancing the alignment with text semantics and facilitating accurate reasoning. Conversely, noisy audio can lead to erroneous textual associations. As shown in Figure [1,](#page-1-0) for the first query "a person is laughing", leveraging audio context significantly aids in identifying the laughing action, which might be ambiguous using vision alone. However, for the second query "a person looks out a window", the audio modality provides no benefit and may even be detrimental, given that this action is primarily visually driven. Therefore, this motivates us to design a more dynamic audio-vision-text association framework that learns to selectively and adaptively aggregate appropriate contexts from audio and visual modalities for reasoning the specific text semantics.

To this end, we make the first attempt to tackle a flexible audiovision-text joint reasoning for the VMR task. In particular, we propose a novel Importance-aware Multi-Granularity fusion model (IMG) with three prediction branches: audio branch, visual branch, and audio-visual fusion branch. Initially, textual guidance is integrated separately with both visual and audio inputs. We then introduce an audio importance-aware module to tackle the issue of variable audio importance, which is crucial in vision-text pairs. This module is supervised by pseudo-labels derived from the retrieval loss of each branch. It effectively learns to assess the relative importance of audio compared to vision. Then, for the latter audio-vision context fusion, we design a multi-granularity fusion network, which establishes local-level and event-level to global-level audio-vision fusion, as a way to better discover key clues in audio for assisting text-specific activity understanding within video contents. In addition to using traditional retrieval loss for supervision, since the multi-modal fusion branch tends to show better performance than the individual vision/audio reasoning branch as the former fuses the positive contexts from both modalities, we also distillate the knowledge from the fusion branch to the weaker visual and audio branches, thus strengthening the performance of both branches and in-turn providing better feedback to the fusion branch to further improving the performance.

To sum up, the key contributions of our work are four-fold:

<span id="page-1-0"></span>![](_page_1_Figure_7.jpeg)

Figure 1: (Top) Audio is a critical modality, outweighing the importance of vision. (Bottom) Audio is entirely irrelevant and considered noise relative to the vision.

- We propose a novel Importance-aware Multi-Granularity fusion network (IMG) to handle the audio-incorporated VMR task, which selectively fuses audio modal information of video samples at multiple granularities for final retrieval.
- We introduce an audio importance predictor, guided by a loss-aware pseudo-importance generator during training, to identify and emphasize semantically relevant audio clues. This mechanism enables the model to selectively focus on informative audio cues while suppressing irrelevant or noisy background sounds.
- We propose a cross-modal knowledge distillation strategy, which transfers knowledge from the more effective fusion branch to the unimodal branch. This strategy enables our framework to retain strong performance even when audio information is missing during inference.
- In addition to standard benchmarks such as Charades-STA and ActivityNet Captions, we introduce a new evaluation dataset, Charades-AudioMatter, where sample's audio matter for moment retrieval. Extensive experiments on these datasets demonstrate the effectiveness of our approach, particularly in scenarios where audio cues play a complementary or dominant role.

## 2 Related Works

Video Moment Retrieval (VMR). VMR aims to retrieve a specific video segment based on a natural language query. Current approaches fall into two categories: proposal-based and proposal-free. For proposal-based [\[34,](#page-8-14) [47,](#page-9-8) [49,](#page-9-9) [57,](#page-9-10) [58,](#page-9-11) [72,](#page-9-12) [73,](#page-9-5) [75\]](#page-9-13), it is often necessary to pre-segment the candidate proposals, and the pre-segmented proposals and text are used as inputs to the cross-modal matching module for retrieval. For proposal-free [\[12,](#page-8-1) [22,](#page-8-15) [62,](#page-9-1) [63,](#page-9-2) [68–](#page-9-3)[70,](#page-9-14) [74\]](#page-9-6), they eliminate the need for predefined proposals, processing raw visual and textual features directly through cross-modal matching. Building on these paradigms, recent studies [\[24,](#page-8-16) [29,](#page-8-17) [41,](#page-8-18) [48,](#page-9-15) [54\]](#page-9-16) have explored DETR-style architectures [\[6\]](#page-8-19) to formulate VMR as a set prediction problem, enabling more flexible and end-to-end training. Further extending these trends, some works aim to unify various

video tasks (e.g., moment retrieval, highlight detection, video summarization) under a general framework [32, 61]. Meanwhile, the rapid progress of large language models (LLMs) has inspired a new wave of research that leverages their semantic reasoning capabilities to enhance VMR [20, 26, 45, 53, 56, 65]. Concurrently, audio has emerged as a valuable modality for complementing vision in VMR, e.g., PMI-LOC [9] employs RGB, motion, and audio and is designed to interact with pairs of modalities at the sequence and channel levels. UMT [36] proposes a unified multimodal transformer framework to fuse vision and audio. ADPN [8] proposes a text-guided clues miner to fill the information gap between audio-visual modalities. However, the above models overlook the inherent uncertainty of audio as a modality and the contribution of audio varies significantly depending on the specific query and video content, highlighting a need for more adaptive solutions.

**Uncertain Modal Learning.** The audio modality often exhibits uncertainty and imbalance in video comprehension tasks [15, 60]. For instance, audio in some videos may consist solely of noise or background sounds, while in text-based video tasks, the query may be entirely independent of the audio. Similar issues arise in other modalities and these modal imbalance challenges have garnered significant attention [10, 23, 33, 46, 59, 66]. To address these challenges, Li et al. [31] quantified uncertainty caused by inherent data ambiguity to enhance prediction reliability. Tellamekala et al. [50] addressed modal uncertainty in categorical sentiment recognition by introducing a modeling approach that enforces both calibration and ordinality constraints and Zhang et al. [71] explored the challenges and solutions for low-quality multimodal fusion, emphasizing the promise of dynamic multimodal learning in overcoming sample-specific, temporal, and spatial variations. Building on these developments, we introduce the audio importance predictor. Supervised by dynamic pseudo-labels derived from sample-wise loss functions, this predictor quantifies the audio modality's importance, providing a critical parameter for adaptive modal fusion.

#### 3 Method

#### 3.1 Overview

**Problem Definition.** Video moment retrieval aims to retrieve the start-end frame pair  $\{f_s, f_e\}$  of a specific segment that semantically match the textual query  $Q = \{w_i\}_{i=1}^N$  from the untrimmed video  $V = \{f_t\}_{t=1}^T$ , where  $w_i$  represents the i-th word and  $f_t$  represents the t-th frame. Additionally, for each video frame, we can extract an audio-aware clip as a complementary modality. Thus, the corresponding audio stream is represented as  $A = \{a_j\}_{j=1}^T$ , where  $a_j$  denotes the j-th audio clip, providing contextual knowledge to enhance the retrieval process.

Overall Pipline. We illustrate our proposed framework in Figure 2. Given the pre-extracted visual, audio, and text features by the corresponding encoder, our IMG model first employs feed-forward network (FFN) layers to map these features into a common latent space. Then, we employ interactions between vision-text and audiotext pairs, fusing them to derive text-semantic-activated visual and audio features. These features are then passed into the visual-audio fusion branch, where they dynamically interact to enable joint reasoning. Specifically, an audio importance predictor generates a sample-wise score, which serves as a crucial weight to determine

the audio-to-vision complementarity coefficients for the given sample pair. Then, the visual and audio features will be fed into a multi-granularity fusion module to aggregate the target-moment-related information at local-, event-, and global-levels according to the previously obtained important weight. Finally, the three-level features are concatenated and fed into the predictor to output predictions, while the visual-only feature and audio-only feature are also fed into their respective unimodal predictors. A multi-branch training with cross-modal knowledge distillation strategy is used to transfer knowledge from the fusion branch to the unimodal branch. During inference, the retrieval branch can be freely selected, with the fusion branch typically being the preferred choice.

#### 3.2 Input Representation

**Multi-Modal Feature Representation.** For audio modality, we firstly use pre-trained audio-aware CNN [21, 27] to extract its original features  $A \in \mathbb{R}^{T \times d_a}$ , then employ an audio encoder which is composed by an FFN, convolutional and transformer layers on them follow [69], textual dependency enhanced features  $A' \in \mathbb{R}^{T \times d}$ . For vision modality, we extract the original visual features  $V \in \mathbb{R}^{T \times d_v}$  by a pre-trained visual CNN [7, 52, 76] and further obtain corresponding enhanced features  $V' \in \mathbb{R}^{T \times d}$  by a visual encoder which shares the same structure as audio encoder. For textual query, we directly initialize it with GloVe embeddings [43]. Since the query may have different semantic alignments with vision and audio, we further encode it by two separate text encoders which also share the same structure as the audio encoder and obtain modal-specific enhanced text features as  $Q'_a \in \mathbb{R}^{N \times d}$  and  $Q'_n \in \mathbb{R}^{N \times d}$ .

**Vision-text/Audio-Text Fusion.** To highlight the most related contents between the vision/audio and the given textual query, we apply context-query attention [69] on each pair, resulting in fused features  $\hat{V} \in \mathbb{R}^{T \times d}$  and  $\hat{A} \in \mathbb{R}^{T \times d}$ .

#### <span id="page-2-0"></span>3.3 Importance-Aware Multi-modal Fusion

The importance-aware multi-modal fusion is a multi-granularity fusion module guided by an audio importance predictor. The predictor is trained to identify and emphasize semantically relevant audio cues, enabling the model to selectively fuse informative audio signals with visual features. Guided by the predicted importance scores, the multi-granularity fusion process effectively filters out irrelevant or noisy audio content while aggregating meaningful cross-modal information at multiple temporal levels, thereby enhancing retrieval performance.

3.3.1 Audio Importance Predictor. The Audio Importance Predictor (AIP) is a lightweight module designed to dynamically estimate the relative importance of audio for each video-query pair. Since ground-truth importance labels are unavailable, we design a loss-aware pseudo-importance generator to produce pseudo labels that serve as supervision signals during training.

**Structure**. Given the text-guided visual and audio features  $\hat{V}$  and  $\hat{A}$ , we first apply attention pooling [4] to obtain their global representations, denoted as  $\hat{V}_G \in \mathbb{R}^d$  and  $\hat{A}_G \in \mathbb{R}^d$ . These global features capture the overall semantic context of the visual and audio modalities, respectively. Next, we concatenate the two global features and feed them into a Multi-Layer Perceptron (MLP), which

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

Figure 2: The framework of our proposed importance-aware multi-granularity fusion model for video moment retrieval.

facilitates mutual feature interaction and enables the model to reason about the relative importance of audio with respect to the visual context. The audio importance score p is then predicted as:  $p = Sigmoid(MLP([\hat{A_G};\hat{V_G}]))$ , where Sigmoid denotes the sigmoid activation function, and [;] indicate the concatenation operator. This predicted score p serves as a sample-wise importance weight, guiding the subsequent multimodal fusion by modulating the contribution of the audio modality.

Training with the pseudo importance labels. In order to train the audio importance predictor, we should construct pseudo labels as supervisory signals. We draw inspiration from the observation that neural networks tend to prioritize learning from simpler samples, which typically correspond to lower training losses [2]. Based on this, we compare the retrieval losses of the audio and visual branches for each video-query pair. The modality with a lower loss is considered to provide more relevant information and is thus assigned a higher pseudo-importance score. Specifically, we compute a pseudo-importance score y' with a softmax-like normalization:

$$y = \frac{e^{\mathcal{L}_{ret}^{v}/\gamma}}{e^{\mathcal{L}_{ret}^{a}/\gamma} + e^{\mathcal{L}_{ret}^{v}/\gamma}}, y' = \begin{cases} 1 & \text{if } y \ge \epsilon_{max}, \\ y & \text{if } \epsilon_{max} > y \ge \epsilon_{min}, \\ 0 & \text{if } y < \epsilon_{min}, \end{cases}$$
(1)

where  $\mathcal{L}^a_{ret}$  and  $\mathcal{L}^v_{ret}$  represents the retrieval loss of audio branch and visual branch, respectively,  $\gamma$  is temperature coefficient. Besides,  $\epsilon_{min}$  is a lower threshold below which the audio modality is considered uninformative, and its contribution is suppressed. Conversely, values above  $\epsilon_{max}$  indicate that audio plays a dominant role in retrieval. Finally, we use a binary cross entropy loss to train AIP as:

$$\mathcal{L}_{p} = \frac{1}{B} \sum_{i=1}^{B} y_{i}' log p_{i} + (1 - y_{i}') log (1 - p_{i}),$$
 (2)

where *B* denotes batch size, and *i* represents the index of *i*-th sample.

The predicted importance score p serves as a key control parameter in the subsequent multi-granularity fusion stage, guiding the selective integration of audio and visual features. To prevent unstable predictions from misguiding early fusion, we initialize the fusion weight with a neutral value of 0.5 and gradually increase the influence of the AIP-predicted score as training progresses. This curriculum-like strategy helps the model build robust multimodal interactions while mitigating the impact of early-stage noise in the importance estimation.

3.3.2 Multi-Granularity Fusion. Given the inherently noisy and variable nature of the audio modality compared to visual signals, a simple fusion strategy may not be sufficient to fully exploit audio-visual complementarity. To address this, we propose a Multi-Granularity Fusion (MGF) module that performs hierarchical fusion from Local-, Event- and Global-perspective, and guided by the dynamically estimated audio importance score.

**Local-Level Visual-Audio Fusion.** As shown in Figure 3(a), to match the visual and audio context frame-to-clip for fine-level fusion, we construct symmetric multi-kernel 1D convolutional networks and to deeply perceive the local relationships between video frames and audio clips, as follows:

$$c_k^v = Conv1d_k(\hat{V}), \ c_k^a = Conv1d_k(\hat{A}), \tag{3}$$

where k is the kernel size of the convolutional networks. The outputs are then concatenated and encoded by an MLP layer to map the dimensionality to d, as follows:

$$\hat{V}_l = MLP([c_1^v; ...; c_n^v]), \hat{A}_l = MLP([c_1^a; ...; c_n^a]), \tag{4}$$

where  $\hat{V}_l \in \mathbb{R}^{T \times d}$  and  $\hat{A}_l \in \mathbb{R}^{T \times d}$ . From this, we obtain the audio and visual features after reinforcement at the local level.

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

Figure 3: Our proposed Multi-Granularity Fusion module: (a) Local-Level Fusion, (b) Event-Level Fusion, (c) Global-Level Fusion.

Finally, we fuse two features by element-wise addition with weight p that is derived from the audio importance predictor:

<span id="page-4-1"></span>
$$\mathcal{F}_{l} = (1 - p)LN(\hat{V}_{l}) + pLN(\hat{A}_{l}), \tag{5}$$

where  $LN(\cdot)$  is layer normalization.

**Event-Level Visual-Audio Fusion.** As shown in Figure 3(b), to match the event-aware semantics between vision and audio for activity reasoning, our event-level fusion module first employs a group of slot attention mechanism [37] to aggregate similar visual/audio clips into multiple events by using a set of learnable event slots, as follows:

$$\hat{A}_{s} = SlotAttn(\hat{A}), \hat{V}_{s} = SlotAttn(\hat{V}), \tag{6}$$

where  $\hat{A}_s \in \mathbb{R}^{e \times d}$  and  $\hat{V}_s \in \mathbb{R}^{e \times d}$  indicates that e events are extracted from the visual/audio sequence. Subsequently, origin visual/audio features enter the cross-modal transformer layer as query and visual/audio events as key and value to obtain  $\hat{A}_e$  and  $\hat{V}_e$ . Finally, we fuse to obtain visual-audio event aware features  $\mathcal{F}_e$  as same as Equation 5.

**Global-Level Visual-Audio Fusion.** As shown in Figure 3(c), to match the visual and audio context from a global perspective, we first encode visual/audio features  $\hat{V}$  and  $\hat{A}$  into global level representation with attention pooling mechanism [4]. Then, we concatenate it with each element of origin  $\hat{V}$  and  $\hat{A}$ , and an MLP layer is used to obtain  $\hat{V}_g$  and  $\hat{A}_g$ . Finally, we obtain visual-audio global aware features  $\mathcal{F}_g$  as same as Equation 5.

**Multi-Scale Feature Fusion.** Since the fused features obtained from different granularities have varying interrelationships, we adopt a set of Bi-GRUs to re-establish these inter-perceptual relationships by combining the features pairwise. Finally, we concatenate the results and pass them through MLP layers to map the dimensions back to d-dimension space, obtaining our final visual-audio fused features  $\mathcal{F}$ .

#### 3.4 Cross-modal Knowledge Distillation

The fusion branch, by jointly modeling audio and visual cues, inherently captures richer and more comprehensive semantic representations. However, in practical applications, audio signals may be missing, corrupted, or unavailable during inference. To ensure that unimodal branches retain strong retrieval capabilities under such

conditions, particularly for the visual branch, we introduce a cross-modal knowledge distillation strategy. Specifically, we treat the fusion branch as a teacher to distill knowledge into the unimodal branches, particularly the visual branch. This enables the unimodal branch to inherit modality-complementary cues from the fusion branch, thereby achieving strong retrieval performance even with visual-only input. To this end, we minimize the Kullback-Leibler (KL) divergence between the output distributions of the fusion and unimodal branches as follows:

$$\mathcal{L}_{kl} = \sum_{i=1}^{B} \tau^2 (KL(\sigma(s^s/\tau), \sigma(t^s/\tau)) + KL(\sigma(s^e/\tau), \sigma(t^e/\tau))),$$
(7)

where  $s^{s/e}$  is start or end logits predicted by the student unimodal branch,  $t^{s/e}$  is start or end logits predicted by the teacher fusion branch,  $\tau$  is temperature coefficient and  $\sigma$  is softmax function. Combining the two unimodal branches, the final KL divergence loss is the summation of the corresponding losses on the visual  $\mathcal{L}^v_{kl}$  and the audio  $\mathcal{L}^a_{kl}$ .

#### 3.5 Model Training

Following previous works [69], we exploit the moment predictor as the retrieval heads to output the start logits and end logits of the moment, and get the final prediction  $P_s$  and  $P_e$ . Take the fusion branch as an example, the retrieval loss is computed as follows:

$$\mathcal{L}_{ret}^f = CE(P_s^f, Y_s) + CE(P_e^f, Y_e), \tag{8}$$

where CE denotes cross-entropy loss,  $Y_{s/e} = \{Y_{s/e}^i\}_i \in \{0,1\}$  represents the supervision where  $Y_{s/e}$  is set to 1 only at the start/end point. By applying this loss function to the three branches (visual branch, audio branch and visual-audio fusion branch), the total retrieval loss of prediction is:

$$\mathcal{L}_{ret} = \mathcal{L}_{ret}^{v} + \mathcal{L}_{ret}^{a} + \mathcal{L}_{ret}^{f}. \tag{9}$$

In addition, following [30], we also introduce saliency loss  $\mathcal{L}_{sal}$  to the vision-text fusion features  $\hat{V}$  and audio-text fusion features  $\hat{A}$  as well as visual-audio fused features  $\mathcal{F}$ . This loss widens the distance between features within and outside of the timestamp. Finally, the overall training loss is:

$$\mathcal{L} = \mathcal{L}_{ret} + \lambda_1 \mathcal{L}_p + \lambda_2 \mathcal{L}_{kl} + \lambda_3 \mathcal{L}_{sal}, \tag{10}$$

<span id="page-5-0"></span>Table 1: Ablation studies of Audio Importance Predictor (AIP) on Charades-STA.

| Line ID | Approach             | R1@3  | R1@5  | R1@7  | mIOU  |
|---------|----------------------|-------|-------|-------|-------|
| #1      | Add                  | 74.19 | 60.97 | 43.41 | 55.02 |
| #2      | Concat               | 72.77 | 59.73 | 43.20 | 54.24 |
| #3      | Sim                  | 74.33 | 60.91 | 43.80 | 55.12 |
| #4      | Attn Entropy         | 73.76 | 60.11 | 43.23 | 55.00 |
| #5      | AIP w/o pseudo-label | 73.98 | 59.74 | 43.33 | 54.39 |
| #6      | AIP                  | 75.18 | 61.85 | 44.23 | 55.62 |

where  $\lambda_1$ ,  $\lambda_2$  and  $\lambda_3$  are the balancing parameters. During inference, we use Maximum Likelihood Estimation (MLE) to obtain the predicted  $(y^s, y^e)$  with the constraint  $y^s \leq y^e$ .

#### 4 Experiment

#### 4.1 Dataset

We conduct our experiments on two video moment retrieval benchmark datasets with audio, *i.e.*, Charades-STA [16] and ActivityNet Captions [28], as well as Chardes-AudioMatter dataset reconstructed by ours. Specifically, **Charades-STA** is a dataset about daily indoor activities. There are 12,408 and 3,720 moment annotations for training and testing, respectively. **ActivityNet Captions** dataset contains about 20k videos taken from ActivityNet. We follow the setup in [69] with 37,421 moment annotations for training, and 17,505 annotations for testing.

Moreover, to further validate the model's capability in integrating audio modality, we introduce a new dataset, **Charades-AudioMatter**, where audio matters for each test query. By reviewing both the videos and their corresponding audio, we manually select and re-organize 1,196 samples from the test set of Charades-STA in which the audio provides valuable information. This selection constitutes a new test set, while the training set of Charades-STA remained unchanged, see supplementary material for more details.

## 4.2 Evaluation Metrics

Following the previous works [16, 35, 67], we adopt " $Rn@\mu$ " and "mIoU" as the evaluation metrics. The " $Rn@\mu$ " denotes the percentage of language queries having at least one result whose Intersection over Union (IoU) with ground truth is larger than  $\mu * 0.1$  in top-n retrieved moments. "mIoU" is the average IoU over all testing samples. In our experiments, we use n=1 and  $\mu \in \{3,5,7\}$ .

#### 4.3 Ablation Study

Effectiveness of Audio Importance Predictor. To demonstrate the effectiveness of our audio importance predictor, we conducted ablation studies. In our initial design, we explored multiple approaches for fusing audio and visual modalities. As shown in Table 1, we compared various approaches with our weighted fusion based on predicted importance of AIP, including direct addition (line 1), concatenation (line 2), and weighted fusion based on cosine similarity between embeddings (line 3) and attention entropy (line 4) calculated by the last attention layer, we also compared AIP without the supervision of pseudo-label (line 5). Ultimately, our AIP demonstrated superior performance, which we attribute to our carefully designed label-supervised module that provides effective

<span id="page-5-1"></span>![](_page_5_Figure_13.jpeg)

(a) Performance curves when noisy audio (b) Average audio importance curve preis introduced.

Figure 4: During inference, as noise in the audio progressively increases, the gap between the two curves in (a) widens, suggesting that the IMG model with AIP exhibits greater robustness. Additionally, as we expected, the average audio importance in (b) decreases as noise levels rise.

<span id="page-5-2"></span>Table 2: Ablation studies of fusion strategy on Charades-STA.

| Local        | Event        | Global | R1@3  | R1@5  | R1@7  | mIOU  |
|--------------|--------------|--------|-------|-------|-------|-------|
| <b>√</b>     | -            | -      | 73.07 | 58.85 | 40.68 | 53.67 |
| -            | ✓            | -      | 74.84 | 59.92 | 41.32 | 54.83 |
| -            | -            | ✓      | 73.20 | 57.50 | 41.64 | 53.67 |
| <b>√</b>     | ✓            | -      | 74.09 | 60.08 | 42.64 | 55.08 |
| $\checkmark$ | -            | ✓      | 73.88 | 59.98 | 43.15 | 54.87 |
| -            | $\checkmark$ | ✓      | 74.28 | 60.33 | 42.93 | 55.47 |
| <b>√</b>     | ✓            | ✓      | 75.18 | 61.85 | 44.23 | 55.62 |

guidance for AIP and ultimately achieve better dynamic multimodal integration.

Robustness Analysis on Audio Importance Predictor. To assess the robustness of AIP, we introduced random gaussian noise to a subset of the test set audio samples. As shown in Figure 4, increasing the proportion of noisy audio widened the performance gap between IMG models with and without AIP. Notably, the performance of IMG without AIP fell below the baseline, while IMG with AIP exhibited a more gentle decline, proving the AIP's robustness and highlighting the detrimental impact on performance when modality importance is ignored under extreme modal imbalance. Effectiveness of Fusion Strategies. We conduct ablation studies to evaluate the visual-audio fusion strategy in Table 2. Here, each of our proposed fusion methods demonstrates a performance improvement, highlighting the effectiveness of our fusion strategy. Performance is further enhanced when two feature aspects are fused, with the fusion of three feature sets outperforming the fusion of two. These results indicate that features extracted at different granularities complement each other effectively, facilitating a more comprehensive fusion of audio information.

**Qualitative Analysis of Fusion Strategies.** This experiment will attempt to answer why our visual-audio fusion strategies is effective and can complement each other. The starting point of this structure is that we want different fusion strategies to focus on different sides of information, *e.g.*, for local-level, our expectation is to find subtle clues. In Figure 5, we categorized samples of Charades-STA into 5 equal number of categories based on the moment-to-video ratios, and we find that IMG with only local fusion show a stronger ability

<span id="page-6-0"></span>![](_page_6_Figure_2.jpeg)

Figure 5: Performance of different granularity fusion strategies at different normalized moment-to-video ratios.

<span id="page-6-1"></span>Table 3: Ablation studies on Charades-STA under conditions of using unimodal branch during inference. "CKD" denotes Cross-modal Knowledge Distillation.

| Method      | Branch | R1@3        | R1@5        | R1@7        | mIOU        |
|-------------|--------|-------------|-------------|-------------|-------------|
|             | Fusion | 75.18       | 61.85       | 44.23       | 55.62       |
| IMG         | Visual | 74.840.34↓  | 60.950.90↓  | 43.440.79↓  | 54.970.65↓  |
|             | Audio  | 60.1115.07↓ | 45.8615.99↓ | 29.3514.88↓ | 42.8512.77↓ |
|             | Fusion | 74.09       | 61.03       | 43.33       | 55.31       |
| IMG w/o CKD | Visual | 72.491.60↓  | 56.924.11↓  | 39.583.75↓  | 53.122.19↓  |
|             | Audio  | 58.0416.05↓ | 43.7017.33↓ | 25.4817.85↓ | 40.6814.63↓ |

to handle smaller ratio (i.e., more subtle moments), whereas IMG with only event fusion enhanced performance of moderate ratio, and for IMG with only global fusion are suited to deal with the case of larger ration. The different performances between different granularities lay the foundation for multi-granularity fusion and finally, our IMG achieves a good balance.

Inference with Unimodal Branch. In real-world scenarios, the audio modality may sometimes be irrelevant or unavailable, such as in surveillance footage. In such cases, the visual branch of IMG can still be employed, and cross-modal knowledge distillation strategy is expected to mitigate potential negative impacts. As shown in Table [3,](#page-6-1) IMG with CKD exhibits minimal performance degradation which confirmed that CKD can largely overcome the negative effects and IMG still demonstrates excellent performance when audio modality is missing during inference. Additionally, we extend our investigation to the inference of audio branch, it turns out that audio branch inference alone has a significant degradation in performance, so we argue that audio can only be an auxiliary modality, and comparing the models with and without the CKD, we find that CKD also improves the performance of audio branch.

Effectiveness and Flexibility of Audio Integration. As shown in Table [4,](#page-6-2) we compare our baseline model which trained using only visual branch (line 1) against our audio-integrated model (line 2), which demonstrates the effectiveness of the incorporation of audio modality. To further validate the effectiveness and flexibility of our framework, we incorporate our IAMF module (Section [3.3\)](#page-2-0) as a plug-in across advanced models (lines 3-6). Results indicate that all models achieve improvements across all metrics, with particularly notable gains on the challenging R1@7 metric, demonstrating that our approach effectively extracts meaningful information from audio modalities.

<span id="page-6-2"></span>Table 4: Effectiveness of audio integration for video moment retrieval. "↑" denotes performance improvement when audio modality is introduced.

| Line ID<br>Method |              |            | Charades-STA | ActivityNet Captions |            |  |
|-------------------|--------------|------------|--------------|----------------------|------------|--|
|                   |              | R1@7       | mIOU         | R1@7                 | mIOU       |  |
| #1                | Baseline     | 39.52      | 52.76        | 26.18                | 43.21      |  |
| #2                | Ours         | 44.234.71↑ | 55.622.86↑   | 29.473.29↑           | 45.191.98↑ |  |
| #3                | EMB [22]     | 39.25      | 53.09        | 26.07                | 45.59      |  |
| #4                | EMB + Ours   | 43.153.90↑ | 54.531.44↑   | 28.442.37↑           | 46.691.10↑ |  |
| #5                | EAMAT [62]   | 41.96      | 54.45        | 25.77                | 42.19      |  |
| #6                | EAMAT + Ours | 44.082.12↑ | 55.591.14↑   | 27.381.61↑           | 43.271.08↑ |  |

## 4.4 Performance Comparison

In Table [5,](#page-7-0) we evaluate our IMG on Charades-STA and ActivityNet Captions and compare it with existing audio-incorporated VMR methods. Furthermore, we list the results for the audio-incorporated method when trained without audio. On Charades-STA and ActivityNet Captions, our IMG achieves the best performance on all metrics. By comparing with the visual branch, we find that introducing audio greatly improves sample training which demonstrates that audio modality can play an important role in assisting VMR. Moreover, our proposed methodology achieves markedly superior performance gains compared to existing approaches, which substantiates our methodological advantage.

In Table [6,](#page-7-1) we evaluate IMG on Charades-STA, comparing it with state-of-the-art VMR methods that employ visual language models as backbones. IMG with InternVideo2 [\[55\]](#page-9-32) achieves the best performance across all metrics when the audio modality is incorporated. This highlights not only the generalization strength of our method under strong backbone settings, but also the critical role of audio cues in improving retrieval performance.

To highlight that IMG effectively mines audio modal information, we conducted experiments on Charades-AudioMatter where the audio data are more consistent and reliable. We compare our method with open-source methods that exhibit competitive performance on Charades-STA. As presented in Table [7,](#page-7-2) our IMG achieves state-of-the-art performance, particularly on R1@7, thereby establishing a substantial lead over all other compared models. This result underscores the effectiveness of our model in extracting and utilizing audio modality and compare to ADPN, IMG exhibits superior proficiency in the integration of audio.

### 4.5 Qualitative Analysis

As shown in Figure [6\(](#page-7-3)a), we observe that the action "sneeze" is not clearly visible, leading to inaccurate predictions. In contrast, the audio prominently captures the action, with an AIP-predicted importance of 0.587, which helps correct the error in the fusion branch. In Figure [6\(](#page-7-3)b), the action "sits" lacks distinct acoustic semantics, causing inaccurate inferences in the audio branch. AIP assigns an importance score of only 0.178, thereby reducing the fusion branch's reliance on audio.

We also perform qualitative analysis on Charades-AudioMatter, comparing with methods that do not introduce audio. For Figure [7\(](#page-7-4)a), the window is partially obscured by curtains, which significantly increases the difficulty of visual-only retrieval for the action

<span id="page-7-0"></span>Table 5: Comparison with audio-incorporated methods on Charades-STA and ActivityNet Captions. We use I3D [\[7\]](#page-8-30) as vision backbone with GloVe [\[43\]](#page-9-30) embeddings.

| Method                 | Audio | Charades-STA |            |            | ActivityNet Captions |            |            |            |            |
|------------------------|-------|--------------|------------|------------|----------------------|------------|------------|------------|------------|
|                        |       | R1@3         | R1@5       | R1@7       | mIOU                 | R1@3       | R1@5       | R1@7       | mIOU       |
| UMT [36]               | ✓     | -            | 48.31      | 29.25      | -                    | -          | -          | -          | -          |
| PMI-LOC w/o audio [9]  | -     | 56.84        | 41.29      | 20.11      | -                    | 60.16      | 39.16      | 18.02      | -          |
| PMI-LOC [9]            | ✓     | 58.081.24↑   | 42.631.34↑ | 21.321.21↑ | -                    | 61.221.06↑ | 40.070.91↑ | 18.290.27↑ | -          |
| QD-DETR w/o audio [41] | -     | -            | 52.77      | 31.13      | -                    | -          | -          | -          | -          |
| QD-DETR [41]           | ✓     | -            | 55.512.74↑ | 34.173.04↑ | -                    | -          | -          | -          | -<br>-     |
| ADPN w/o audio [8]     | -     | 70.35        | 55.32      | 37.47      | 51.13                | 55.72      | 39.56      | 25.20      | 41.55      |
| ADPN [8]               | ✓     | 71.991.64↑   | 57.692.37↑ | 41.103.63↑ | 52.861.73↑           | 57.161.44↑ | 41.401.84↑ | 26.311.11↑ | 42.310.76↑ |
| IMG w/o audio          | -     | 72.37        | 56.34      | 39.52      | 52.76                | 59.19      | 41.51      | 26.18      | 43.21      |
| IMG                    | ✓     | 75.182.81↑   | 61.855.51↑ | 44.234.71↑ | 55.622.86↑           | 61.502.31↑ | 45.063.55↑ | 29.473.29↑ | 45.191.98↑ |

<span id="page-7-1"></span>Table 6: Comparison with state-of-the-art methods on Charades-STA. We compare methods of using visual language models as backbone. "CLIP+SF" refers to SlowFast [\[14\]](#page-8-37) combined with CLIP [\[44\]](#page-9-33) , "IV2" denotes InternVideo2 [\[55\]](#page-9-32).

| Method            | backbone | R1@3       | R1@5       | R1@7       | mIOU       |
|-------------------|----------|------------|------------|------------|------------|
| UnLoc-L [61]      | CLIP     | -          | 60.80      | 38.40      | -          |
| Moment-DETR [30]  | CLIP+SF  | -          | 55.65      | 34.17      | -          |
| BAM-DETR [29]     | CLIP+SF  | 72.93      | 59.95      | 39.38      | 52.33      |
| QD-DETR [41]      | CLIP+SF  | -          | 57.31      | 32.55      | -          |
| TR-DETR [48]      | CLIP+SF  | -          | 57.61      | 33.52      | -          |
| UniVTG [32]       | CLIP+SF  | 70.81      | 58.01      | 35.65      | 50.10      |
| FlashVTG [5]      | CLIP+SF  | -          | 60.11      | 38.01      | -          |
| IMG w/o audio     | CLIP+SF  | 70.25      | 54.12      | 37.72      | 51.65      |
| IMG               | CLIP+SF  | 74.443.38↑ | 59.765.64↑ | 42.935.21↑ | 55.033.38↑ |
| InternVideo2 [55] | IV2      | 79.70      | 70.03      | 48.95      | 58.79      |
| FlashVTG [5]      | IV2      | -          | 70.32      | 49.87      | -          |
| SG-DETR [19]      | IV2      | -          | 70.20      | 49.50      | 59.10      |
| IMG w/o audio     | IV2      | 78.58      | 66.08      | 48.69      | 58.46      |
| IMG               | IV2      | 82.023.44↑ | 70.814.73↑ | 54.335.64↑ | 62.253.79↑ |

<span id="page-7-2"></span>Table 7: Performance comparison on Charades-AudioMatter. All methods utilize I3D [\[7\]](#page-8-30) backbone.

| Method             | Audio | R1@3       | R1@5       | R1@7       | mIOU       |
|--------------------|-------|------------|------------|------------|------------|
| SeqPAN [68]        | -     | 79.30      | 67.17      | 48.96      | 58.74      |
| EAMAT [62]         | -     | 78.30      | 68.25      | 48.88      | 58.90      |
| EMB [22]           | -     | 77.81      | 67.00      | 47.96      | 58.66      |
| ADPN w/o audio [8] | -     | 77.89      | 64.42      | 44.64      | 56.98      |
| ADPN [8]           | ✓     | 78.650.76↑ | 66.752.33↑ | 49.715.07↑ | 59.852.87↑ |
| IMG w/o audio      | -     | 77.89      | 65.92      | 47.58      | 58.35      |
| IMG                | ✓     | 82.744.85↑ | 71.936.01↑ | 54.276.69↑ | 62.764.41↑ |

"closes the window". EMB, EAMAT, SeqPAN, and IMG without audio failed to retrieve accurately as they relied solely on vision. In contrast, IMG leveraged the acoustic semantics, allowing for more accurate retrieval. For Figure [7\(](#page-7-4)b), visual-only retrieval is also challenging due to the subtle movements associated with "laugh" and minimal scene variation. However, the prominent acoustic signal of "laugh" enabled IMG to effectively pinpoint the corresponding timestamp.

<span id="page-7-3"></span>![](_page_7_Figure_9.jpeg)

Figure 6: Two samples were selected from Charades-STA.

<span id="page-7-4"></span>![](_page_7_Figure_11.jpeg)

Figure 7: Two samples were selected from Charades-AudioMatter. (a) appeared an occlusion interfering with the visual field, while (b) depicted a visually insignificant action.

### 5 Conclusion

In this paper, we propose a novel Importance-aware Multi-Granularity fusion model (IMG) to handle the flexible audio-vision-text reasoning for the VMR task. To explore the audio's uncertainty, we propose an audio importance predictor to utilize the retrieval loss of the model to generate dynamic pseudo-labels for supervision and dynamically assign weights to different samples of audio to provide better audio-context guidance. We also propose a multi-granularity visual-audio fusion network to fully fuse audio and visual modality from local- to event- and global-level for complementary learning. A new dataset Charades-AudioMatter is further introduced to validate the model's capability in integrating audio modality. Experiments have proven the effectiveness of our proposed approach.

### Acknowledgments

This work was supported by the Pioneer and Leading Goose R&D Program of Zhejiang (No. 2024C01110), National Natural Science Foundation of China (No. 62472385), Young Elite Scientists Sponsorship Program by China Association for Science and Technology (No. 2022QNRC001), Public Welfare Technology Research Project of Zhejiang Province (No. LGF21F020010), Fundamental Research Funds for the Provincial Universities of Zhejiang (No. FR2402ZD) and Zhejiang Provincial High-Level Talent Special Support Program.

## References

- <span id="page-8-0"></span>[1] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. 2017. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision. 5803–5812.
- <span id="page-8-32"></span>[2] Devansh Arpit, Stanisław Jastrzębski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, et al. 2017. A closer look at memorization in deep networks. In International conference on machine learning. PMLR, 233–242.
- <span id="page-8-5"></span>[3] Anurag Bagchi, Jazib Mahmood, Dolton Fernandes, and Ravi Kiran Sarvadevabhatla. 2021. Hear me out: Fusional approaches for audio augmented temporal action localization. arXiv preprint arXiv:2106.14118 (2021).
- <span id="page-8-31"></span>[4] Dzmitry Bahdanau. 2014. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473 (2014).
- <span id="page-8-38"></span>[5] Zhuo Cao, Bingqing Zhang, Heming Du, Xin Yu, Xue Li, and Sen Wang. 2024. FlashVTG: Feature Layering and Adaptive Score Handling Network for Video Temporal Grounding. arXiv preprint arXiv:2412.13441 (2024).
- <span id="page-8-19"></span>[6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 2020. End-to-end object detection with transformers. In European conference on computer vision. Springer, 213–229.
- <span id="page-8-30"></span>[7] Joao Carreira and Andrew Zisserman. 2017. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 6299–6308.
- <span id="page-8-11"></span>[8] Houlun Chen, Xin Wang, Xiaohan Lan, Hong Chen, Xuguang Duan, Jia Jia, and Wenwu Zhu. 2023. Curriculum-listener: Consistency-and complementarityaware audio-enhanced temporal sentence grounding. In Proceedings of the 31st ACM International Conference on Multimedia. 3117–3128.
- <span id="page-8-12"></span>[9] Shaoxiang Chen, Wenhao Jiang, Wei Liu, and Yu-Gang Jiang. 2020. Learning modality interaction for temporal sentence localization and event captioning in videos. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16. Springer, 333–351.
- <span id="page-8-24"></span>[10] Xuelian Cheng, Yiran Zhong, Yuchao Dai, Pan Ji, and Hongdong Li. 2019. Noiseaware unsupervised deep lidar-stereo fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6339–6348.
- <span id="page-8-3"></span>[11] Jianfeng Dong, Xirong Li, Chaoxi Xu, Xun Yang, Gang Yang, Xun Wang, and Meng Wang. 2022. Dual encoding for video retrieval by text. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 8 (2022), 4065–4080.
- <span id="page-8-1"></span>[12] Jianfeng Dong, Xiaoman Peng, Daizong Liu, Xiaoye Qu, Xun Yang, Cuizhu Bao, and Meng Wang. 2024. Temporal sentence grounding with relevance feedback in videos. Advances in Neural Information Processing Systems 37 (2024), 43107–43132.
- <span id="page-8-4"></span>[13] Jianfeng Dong, Minsong Zhang, Zheng Zhang, Xianke Chen, Daizong Liu, Xiaoye Qu, Xun Wang, and Baolong Liu. 2023. Dual learning with dynamic knowledge distillation for partially relevant video retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 11302–11312.
- <span id="page-8-37"></span>[14] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. 2019. Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision. 6202–6211.
- <span id="page-8-23"></span>[15] Jie Fu, Junyu Gao, Bing-Kun Bao, and Changsheng Xu. 2023. Multimodal imbalance-aware gradient modulation for weakly-supervised audio-visual video parsing. IEEE Transactions on Circuits and Systems for Video Technology (2023).
- <span id="page-8-2"></span>[16] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. 2017. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision. 5267–5275.
- <span id="page-8-41"></span>[17] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. 2017. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 776–780.
- <span id="page-8-6"></span>[18] Mandar Gogate, Kia Dashtipour, Ahsan Adeel, and Amir Hussain. 2020. CochleaNet: A robust language-independent audio-visual model for real-time speech enhancement. Information Fusion 63 (2020), 273–285.
- <span id="page-8-39"></span>[19] Aleksandr Gordeev, Vladimir Dokholyan, Irina Tolstykh, and Maksim Kuprashevich. 2024. Saliency-guided detr for moment retrieval and highlight detection. arXiv preprint arXiv:2410.01615 (2024).

- <span id="page-8-21"></span>[20] Yongxin Guo, Jingyu Liu, Mingda Li, Xiaoying Tang, Xi Chen, and Bo Zhao. 2024. VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced Video Temporal Grounding. arXiv preprint arXiv:2405.13382 (2024).
- <span id="page-8-28"></span>[21] Shawn Hershey, Sourish Chaudhuri, Daniel PW Ellis, Jort F Gemmeke, Aren Jansen, R Channing Moore, Manoj Plakal, Devin Platt, Rif A Saurous, Bryan Seybold, et al. 2017. CNN architectures for large-scale audio classification. In 2017 ieee international conference on acoustics, speech and signal processing (icassp). IEEE, 131–135.
- <span id="page-8-15"></span>[22] Jiabo Huang, Hailin Jin, Shaogang Gong, and Yang Liu. 2022. Video activity localisation with uncertainties in temporal boundary. In European Conference on Computer Vision. Springer, 724–740.
- <span id="page-8-25"></span>[23] Zhenyu Huang, Guocheng Niu, Xiao Liu, Wenbiao Ding, Xinyan Xiao, Hua Wu, and Xi Peng. 2021. Learning with noisy correspondence for cross-modal matching. Advances in Neural Information Processing Systems 34 (2021), 29406–29419.
- <span id="page-8-16"></span>[24] Jinhyun Jang, Jungin Park, Jin Kim, Hyeongjun Kwon, and Kwanghoon Sohn. 2023. Knowing where to focus: Event-aware transformer for video grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13846– 13856.
- <span id="page-8-7"></span>[25] Xun Jiang, Xing Xu, Zhiguo Chen, Jingran Zhang, Jingkuan Song, Fumin Shen, Huimin Lu, and Heng Tao Shen. 2022. Dhhn: Dual hierarchical hybrid network for weakly-supervised audio-visual video parsing. In Proceedings of the 30th ACM International Conference on Multimedia. 719–727.
- <span id="page-8-22"></span>[26] Yiyang Jiang, Wengyu Zhang, Xulu Zhang, Xiao-Yong Wei, Chang Wen Chen, and Qing Li. 2024. Prior knowledge integration via llm encoding and pseudo event regulation for video moment retrieval. In Proceedings of the 32nd ACM International Conference on Multimedia. 7249–7258.
- <span id="page-8-29"></span>[27] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley. 2020. Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020), 2880–2894.
- <span id="page-8-35"></span>[28] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. 2017. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision. 706–715.
- <span id="page-8-17"></span>[29] Pilhyeon Lee and Hyeran Byun. 2024. Bam-detr: Boundary-aligned moment detection transformer for temporal sentence grounding in videos. In European Conference on Computer Vision. Springer, 220–238.
- <span id="page-8-34"></span>[30] Jie Lei, Tamara L Berg, and Mohit Bansal. 2021. Detecting moments and highlights in videos via natural language queries. Advances in Neural Information Processing Systems 34 (2021), 11846–11858.
- <span id="page-8-27"></span>[31] Hao Li, Jingkuan Song, Lianli Gao, Xiaosu Zhu, and Hengtao Shen. 2024. Prototype-based aleatoric uncertainty quantification for cross-modal retrieval. Advances in Neural Information Processing Systems 36 (2024).
- <span id="page-8-20"></span>[32] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. 2023. Univtg: Towards unified video-language temporal grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2794–2804.
- <span id="page-8-26"></span>[33] Yijie Lin, Jie Zhang, Zhenyu Huang, Jia Liu, Zujie Wen, and Xi Peng. 2024. Multigranularity correspondence learning from long-term noisy videos. arXiv preprint arXiv:2401.16702 (2024).
- <span id="page-8-14"></span>[34] Daizong Liu, Xiaoye Qu, Jianfeng Dong, and Pan Zhou. 2021. Adaptive proposal generation network for temporal sentence localization in videos. arXiv preprint arXiv:2109.06398 (2021).
- <span id="page-8-36"></span>[35] Meng Liu, Xiang Wang, Liqiang Nie, Xiangnan He, Baoquan Chen, and Tat-Seng Chua. 2018. Attentive moment retrieval in videos. In The 41st international ACM SIGIR conference on research & development in information retrieval. 15–24.
- <span id="page-8-13"></span>[36] Ye Liu, Siyuan Li, Yang Wu, Chang-Wen Chen, Ying Shan, and Xiaohu Qie. 2022. Umt: Unified multi-modal transformers for joint video moment retrieval and highlight detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3042–3051.
- <span id="page-8-33"></span>[37] Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. 2020. Object-centric learning with slot attention. Advances in neural information processing systems 33 (2020), 11525–11538.
- <span id="page-8-40"></span>[38] Ilya Loshchilov, Frank Hutter, et al. 2017. Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101 5 (2017).
- <span id="page-8-8"></span>[39] Otniel-Bogdan Mercea, Thomas Hummel, A Sophia Koepke, and Zeynep Akata. 2022. Temporal and cross-modal attention for audio-visual zero-shot learning. In European Conference on Computer Vision. Springer, 488–505.
- <span id="page-8-9"></span>[40] Otniel-Bogdan Mercea, Lukas Riesch, A Koepke, and Zeynep Akata. 2022. Audiovisual generalised zero-shot learning with cross-modal attention and language. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10553–10563.
- <span id="page-8-18"></span>[41] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. 2023. Query-dependent video representation for moment retrieval and highlight detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 23023–23033.
- <span id="page-8-10"></span>[42] Wenwen Pan, Haonan Shi, Zhou Zhao, Jieming Zhu, Xiuqiang He, Zhigeng Pan, Lianli Gao, Jun Yu, Fei Wu, and Qi Tian. 2022. Wnet: Audio-guided video object

- segmentation via wavelet-based cross-modal denoising networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1320– 1331.
- <span id="page-9-30"></span>[43] Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 1532–1543.
- <span id="page-9-33"></span>[44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.
- <span id="page-9-18"></span>[45] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. 2024. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14313–14323.
- <span id="page-9-23"></span>[46] Yazhou Ren, Xinyue Chen, Jie Xu, Jingyu Pu, Yonghao Huang, Xiaorong Pu, Ce Zhu, Xiaofeng Zhu, Zhifeng Hao, and Lifang He. 2024. A novel federated multiview clustering method for unaligned and incomplete data fusion. Information Fusion 108 (2024), 102357.
- <span id="page-9-8"></span>[47] Xingyu Shen, Xiang Zhang, Xun Yang, Yibing Zhan, Long Lan, Jianfeng Dong, and Hongzhou Wu. 2023. Semantics-Enriched Cross-Modal Alignment for Complex-Query Video Moment Retrieval. In Proceedings of the 31st ACM International Conference on Multimedia. 4109–4118.
- <span id="page-9-15"></span>[48] Hao Sun, Mingyao Zhou, Wenjing Chen, and Wei Xie. 2024. Tr-detr: Taskreciprocal transformer for joint moment retrieval and highlight detection. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 4998–5007.
- <span id="page-9-9"></span>[49] Xin Sun, Xuan Wang, Jialin Gao, Qiong Liu, and Xi Zhou. 2022. You need to read again: Multi-granularity perception network for moment retrieval in videos. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1022–1032.
- <span id="page-9-26"></span>[50] Mani Kumar Tellamekala, Shahin Amiriparian, Björn W Schuller, Elisabeth André, Timo Giesbrecht, and Michel Valstar. 2023. COLD fusion: Calibrated and ordinal latent distribution fusion for uncertainty-aware multimodal emotion recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence (2023).
- <span id="page-9-34"></span>[51] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).
- <span id="page-9-28"></span>[52] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. 2015. Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision. 4489–4497.
- <span id="page-9-19"></span>[53] Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan Wang, Weifeng Ge, and Lifu Huang. 2024. Grounded-videollm: Sharpening finegrained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290 (2024).
- <span id="page-9-16"></span>[54] Jing Wang, Aixin Sun, Hao Zhang, and Xiaoli Li. 2023. MS-DETR: Natural Language Video Localization with Sampling Moment-Moment Interaction. arXiv preprint arXiv:2305.18969 (2023).
- <span id="page-9-32"></span>[55] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. 2024. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision. Springer, 396–416.
- <span id="page-9-20"></span>[56] Yueqian Wang, Xiaojun Meng, Jianxin Liang, Yuxuan Wang, Qun Liu, and Dongyan Zhao. 2024. Hawkeye: Training video-text llms for grounding text in videos. arXiv preprint arXiv:2403.10228 (2024).
- <span id="page-9-10"></span>[57] Zhenzhi Wang, Limin Wang, Tao Wu, Tianhao Li, and Gangshan Wu. 2022. Negative sample matters: A renaissance of metric learning for temporal grounding. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 36. 2613–2623.
- <span id="page-9-11"></span>[58] Shaoning Xiao, Long Chen, Songyang Zhang, Wei Ji, Jian Shao, Lu Ye, and Jun Xiao. 2021. Boundary proposal network for two-stage natural language video localization. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35. 2986–2994.
- <span id="page-9-24"></span>[59] Bo Xu, Shizhou Huang, Ming Du, Hongya Wang, Hui Song, Chaofeng Sha, and Yanghua Xiao. 2022. Different data, different modalities! reinforced data splitting for effective multimodal information extraction from social media posts. In Proceedings of the 29th International Conference on Computational Linguistics. 1855–1864.
- <span id="page-9-22"></span>[60] Ruize Xu, Ruoxuan Feng, Shi-Xiong Zhang, and Di Hu. 2023. Mmcosine: Multimodal cosine loss towards balanced audio-visual fine-grained learning. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 1–5.
- <span id="page-9-17"></span>[61] Shen Yan, Xuehan Xiong, Arsha Nagrani, Anurag Arnab, Zhonghao Wang, Weina Ge, David Ross, and Cordelia Schmid. 2023. Unloc: A unified framework for video localization tasks. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13623–13633.
- <span id="page-9-1"></span>[62] Shuo Yang and Xinxiao Wu. 2022. Entity-aware and motion-aware transformers for language-driven action localization in videos. arXiv preprint arXiv:2205.05854 (2022).

- <span id="page-9-2"></span>[63] Shuo Yang, Xinxiao Wu, Zirui Shang, and Jiebo Luo. 2024. Dynamic Pathway for Query-Aware Feature Learning in Language-Driven Action Localization. IEEE Transactions on Multimedia (2024).
- <span id="page-9-0"></span>[64] Xun Yang, Tianyu Chang, Tianzhu Zhang, Shanshan Wang, Richang Hong, and Meng Wang. 2024. Learning hierarchical visual transformation for domain generalizable visual matching and recognition. International Journal of Computer Vision 132, 11 (2024), 4823–4849.
- <span id="page-9-21"></span>[65] Xun Yang, Jianming Zeng, Dan Guo, Shanshan Wang, Jianfeng Dong, and Meng Wang. 2024. Robust video question answering via contrastive cross-modality representation learning. Science China Information Sciences 67, 10 (2024), 202104.
- <span id="page-9-25"></span>[66] Wenfang Yao, Kejing Yin, William K Cheung, Jia Liu, and Jing Qin. 2024. DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion with Missing Modality and Modal Inconsistency. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 16416–16424.
- <span id="page-9-31"></span>[67] Yitian Yuan, Tao Mei, and Wenwu Zhu. 2019. To find where you talk: Temporal sentence localization in video with attention based location regression. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33. 9159–9166.
- <span id="page-9-3"></span>[68] Hao Zhang, Aixin Sun, Wei Jing, Liangli Zhen, Joey Tianyi Zhou, and Rick Siow Mong Goh. 2021. Parallel attention network with sequence matching for video grounding. arXiv preprint arXiv:2105.08481 (2021).
- <span id="page-9-4"></span>[69] Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. 2020. Span-based localizing network for natural language video localization. arXiv preprint arXiv:2004.13931 (2020).
- <span id="page-9-14"></span>[70] Long Zhang, Peipei Song, Zhangling Duan, Shuo Wang, Xiaojun Chang, and Xun Yang. 2025. Video corpus moment retrieval with query-specific context learning and progressive localization. IEEE Transactions on Circuits and Systems for Video Technology (2025).
- <span id="page-9-27"></span>[71] Qingyang Zhang, Yake Wei, Zongbo Han, Huazhu Fu, Xi Peng, Cheng Deng, Qinghua Hu, Cai Xu, Jie Wen, Di Hu, et al. 2024. Multimodal fusion on low-quality data: A comprehensive survey. arXiv preprint arXiv:2404.18947 (2024).
- <span id="page-9-12"></span>[72] Songyang Zhang, Houwen Peng, Jianlong Fu, Yijuan Lu, and Jiebo Luo. 2021. Multi-scale 2d temporal adjacency networks for moment localization with natural language. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 12 (2021), 9073–9087.
- <span id="page-9-5"></span>[73] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. 2020. Learning 2d temporal adjacent networks for moment localization with natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34. 12870–12877.
- <span id="page-9-6"></span>[74] Yimeng Zhang, Xin Chen, Jinghan Jia, Sijia Liu, and Ke Ding. 2023. Textvisual prompting for efficient 2d temporal video grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14794–14804.
- <span id="page-9-13"></span>[75] Qi Zheng, Jianfeng Dong, Xiaoye Qu, Xun Yang, Yabing Wang, Pan Zhou, Baolong Liu, and Xun Wang. 2023. Progressive localization networks for language-based moment localization. ACM Transactions on Multimedia Computing, Communications and Applications 19, 2 (2023), 1–21.
- <span id="page-9-29"></span>[76] Sheng Zhou, Junbin Xiao, Qingyun Li, Yicong Li, Xun Yang, Dan Guo, Meng Wang, Tat-Seng Chua, and Angela Yao. 2025. Egotextvqa: Towards egocentric scene-text aware video question answering. In Proceedings of the Computer Vision and Pattern Recognition Conference. 3363–3373.
- <span id="page-9-7"></span>[77] Yipin Zhou and Ser-Nam Lim. 2021. Joint audio-visual deepfake detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 14800– 14809.

We report more technical details and more experimental results which are not included in the paper due to space limit:

- Detailed analysis of dataset Charades-AudioMatter including:
  - Dataset construction (Section [A.1\)](#page-10-0).
  - Statistical analysis (Section [A.2\)](#page-10-1).
- Experiments on ActivityNet Captions including:
  - Ablation studies on fusion strategies (Section [B.1\)](#page-11-0).
  - Ablation studies on additional model structures (Section [B.4\)](#page-11-1).
  - Qualitative analysis (Section [B.3\)](#page-11-2).
- Additional experiments including:
  - Experiments on hyperparameters (Section [C.1\)](#page-12-0) including threshold , temperature and others.
  - Experiments on efficiency (Section [C.2\)](#page-12-1), Event-Level Fusion module (Section [C.3\)](#page-12-2), weak supervision (Section [C.4\)](#page-12-3), failed AIP (Section [C.6\)](#page-13-1) and audio importance distribution (Section [C.5\)](#page-12-4).
- Implement details (Section [D\)](#page-13-2).

## A Charades-AudioMatter Dataset Construction

### <span id="page-10-0"></span>A.1 Dataset Construction

In this section, we introduce the dataset Charades-AudioMatter in detail. To ensure the high quality of the Charades-AudioMatter dataset and the reliability of experimental results, the dataset construction underwent a rigorous screening process. The dataset was annotated by six postgraduate students with experience in multimodal learning. Each instance was independently labeled by two annotators, with disagreements adjudicated by a third annotator. The validity and relevance of the audio data were carefully evaluated through the following processes:

Validity of the Audio. Given a sample, the audio modality is first subjected to a validity assessment. Samples containing significant background noise or lacking any sound were excluded, as such audio lacks meaningful information and cannot contribute effectively to VMR. This process is employed for rapid preliminary screening. Correlation between Audio and Query Text. After the initial screening, each sample was manually evaluated through a combination of audio and visual to determine whether the query text was dependent on the audio. For instance, query describing static actions (e.g., "sitting," "looking," "standing") were almost excluded because the audio does not provide meaningful cues for these actions. Similarly, for actions typically associated with audio cues (e.g., "laughing," "closing the door"), if the audio in a specific instance lacked sound or the sound was too faint, the sample was marked as invalid and excluded. This step ensured the relevance of audio to the text by integrating auditory judgment with semantic analysis of the text.

Temporal Alignment of Audio and Video. After the screening steps above, we evaluate the temporal alignment between visual and audio modalities. Specifically, manual timestamp annotation was performed solely based on the audio and query text, followed by IoU computation with the ground truth. Samples exhibiting an IoU score below 0.3 are discarded. This process ensures the validation of temporal consistency between audio and video modalities, effectively filtering out severely misaligned samples (e.g., those with significant audio delays or excessive offsets).

<span id="page-10-3"></span>![](_page_10_Picture_19.jpeg)

Figure 8: For the same activity of "open the door", we reviewed and listened specific samples, ultimately choosing the left one where the sound is clearly communicated, discarding the other where there is almost no corresponding sound.

<span id="page-10-2"></span>Table 8: Statistical analysis of the dataset Charades-AudioMatter. We compare the selected activity categories with the unselected one.

| Selected Activity       | Count | Unselected Activity    | Count |
|-------------------------|-------|------------------------|-------|
| open (door/cabinet/)    | 241   | sit (on bed/chair/)    | 218   |
| close (door/closet/)    | 150   | hold                   | 147   |
| put (bag/grpceries/)    | 138   | (un)dress              | 111   |
| run                     | 90    | look                   | 85    |
| turn on/off (light/tv/) | 89    | stand                  | 59    |
| throw (broom/shoes/)    | 66    | smile                  | 55    |
| take (vacuum/food/)     | 56    | watch                  | 48    |
| laugh                   | 52    | read                   | 32    |
| eat                     | 41    | awake                  | 38    |
| wash (hand/glass/)      | 29    | take a picture         | 30    |
| drink                   | 28    | play (phone/camera/)   | 23    |
| walk                    | 25    | snuggle with (pillow/) | 20    |
| cook                    | 22    | (fix/adjust) hair      | 19    |
| pour (water/coffee/)    | 16    | lay                    | 18    |
| sit down                | 13    |                        |       |
| talk                    | 10    |                        |       |

Upon completing the labeling process, a random sampling procedure was conducted to evaluate the reliability and consistency of the annotations, and the final inter-annotator agreement exceeded 95%. This rigorous, multi-step approach ensures that the dataset adheres to high-quality standards while providing a robust foundation for advancing research in the VMR task.

### <span id="page-10-1"></span>A.2 Statistical Analysis

To further demonstrate the effectiveness of our proposed dataset Charades-AudioMatter, we conduct activity category-wise analysis in Table [8.](#page-10-2) We sorted the categories of selected activities and compared them with other unselected activities. As we can see from the table, selected activities tend to have more significant differentiating sounds such as "open", "put", and "run", while unselected activities do not tend to convey sounds such as "sit on", "hold", and "look". But we can't exactly classify audio validity by activity, we give samples in Figure [8.](#page-10-3)

Table [9](#page-11-3) shows the frequency distribution of normalized moment durations in Charades-AudioMatter compared with the original Charades-STA. Our purposed Charades-AudioMatter maintains

<span id="page-11-3"></span>![](_page_11_Figure_2.jpeg)

Figure 9: Comparison between Charades-AudioMatter and Charades-STA in terms of moment duration.

<span id="page-11-4"></span>Table 9: Ablation studies of fusion strategies on ActivityNet Captions.

| Local    | Event        | Global | R1@3  | R1@5  | R1@7  | mIOU  |
|----------|--------------|--------|-------|-------|-------|-------|
| <b>√</b> | -            | -      | 60.08 | 43.70 | 27.66 | 43.96 |
| -        | $\checkmark$ | -      | 59.13 | 42.68 | 26.83 | 43.54 |
| -        | -            | ✓      | 58.57 | 42.12 | 27.00 | 43.22 |
|          | ✓            | -      | 59.21 | 43.60 | 28.71 | 44.17 |
| ✓        | -            | ✓      | 61.25 | 43.90 | 28.92 | 44.78 |
| -        | ✓            | ✓      | 59.84 | 43.69 | 28.19 | 44.05 |
| <b>√</b> | ✓            | ✓      | 61.50 | 45.06 | 29.47 | 45.19 |

<span id="page-11-5"></span>Table 10: Ablation studies of each component on ActivityNet Captions.

| Method               | R1@3  | R1@5  | R1@7  | mIOU  |
|----------------------|-------|-------|-------|-------|
| IMG w/o AIP          | 59.81 | 43.40 | 28.06 | 44.49 |
| IMG w/o pseudo-label | 58.10 | 42.00 | 27.76 | 43.52 |
| IMG w/o CKD          | 59.95 | 43.89 | 28.49 | 44.47 |
| IMG                  | 61.50 | 45.06 | 29.47 | 45.19 |

comparable diversity and roughly follows the original Charades-STA in duration distribution, which validates the rationality of our proposed dataset.

## B Experiments on ActivityNet Captions

To further verify the general effectiveness of the crucial contributions in our proposed IMG, we conduct more experiments on ActivityNet Captions.

## <span id="page-11-0"></span>**B.1** Ablation studies on fusion strategies

We verify the effectiveness of fusion strategies on ActivityNet Captions. As shown in Table 9, each of the proposed fusion strategies consistently yields performance improvements, underscoring their efficacy. These results also demonstrate that integrating features of varying granularities provides complementary benefits, leading to superior overall performance.

<span id="page-11-6"></span>![](_page_11_Figure_13.jpeg)

Figure 10: Two samples were selected from ActivityNet Captions.

<span id="page-11-7"></span>![](_page_11_Figure_15.jpeg)

Figure 11: Experiments with different hyperparameters, (a) threshold  $\epsilon_{min}$ , (b) temperature  $\gamma$ .

## **B.2** Ablation studies on additional model structures

We also verify the effectiveness of additional model structures on ActivityNet Captions. As presented in Table 10, lines 1 and 2 illustrate the performance of the model without the audio importance predictor and the pseudo-label constraint, respectively. These results indicate that the proposed pseudo-label mechanism enhances decision-making within the audio importance predictor and ultimately improves performance. Finally, line 3 quantifies the effect of ablating cross-modal knowledge distillation, further validating the contributions of this component to the overall framework.

#### <span id="page-11-2"></span>**B.3** Qualitative analysis

In order to demonstrate our proposed IMG more intuitively, we have selected examples on ActivityNet Captions for visual presentation. As shown in Figure 10(a) "dives" is visible in the frames, but "flipping" is not distinctly captured. However, both actions exhibit clear acoustic semantics, allowing the fusion branch to make a more accurate prediction. In Figure 10(b), the visual presentation is highly noticeable, while the audio consists entirely of background music, and as a result, the fusion branch is not misled by the audio.

# <span id="page-11-1"></span>B.4 Ablation studies on additional model structures

We also verify the effectiveness of additional model structures on ActivityNet Captions. As presented in Table 10, lines 1 and 2 illustrate the performance of the model without the audio importance

<span id="page-12-5"></span>Table 11: Ablation studies on hyperparameters  $\tau$  on visual branch.

| τ    | 0.5   | 1.0   | 2.0   | 4.0   |
|------|-------|-------|-------|-------|
| R1@7 | 42.10 | 42.61 | 43.44 | 42.90 |

predictor and the pseudo-label constraint, respectively. These results indicate that the proposed pseudo-label mechanism enhances decision-making within the audio importance predictor and ultimately improves performance. Finally, line 3 quantifies the effect of ablating cross-modal knowledge distillation, further validating the contributions of this component to the overall framework.

### C Additional experiments

#### <span id="page-12-0"></span>C.1 Experiments on hyperparameters

We conduct ablation studies on two critical hyperparameters, threshold  $\epsilon_{min}$  and temperature  $\gamma$ . As detailed in Figure 11, our analysis reveals that selecting an optimal threshold and temperature significantly enhances the model's ability to learn the relative importance of visual and audio features, thereby improving overall performance. Conversely, setting the threshold too low may cause the model to mistakenly assign importance to noisy semantic features, while setting it too high can lead the model to disregard valuable samples containing relevant semantic information. Similarly, an excessively low temperature coefficient results in overly rigid decision-making by the model, whereas an excessively high coefficient diminishes the model's sensitivity to the two feature types, ultimately impairing performance.

In Table 11, we conduct an ablation study on the temperature coefficient  $\tau$  in cross-modal knowledge distillation module. The results indicate that our method is relatively insensitive to  $\tau$ .

For loss-related parameters, both  $\lambda_1$  and  $\lambda_2$  are crucial to the model. We set  $\lambda_1$  = 5 and  $\lambda_2$  = 10 to balance the respective loss terms, ensuring they are on a similar scale.  $\lambda_3$  controls the auxiliary loss, and we set  $\lambda_3$  = 0.5, significantly smaller than the others. These values were determined based on grid search and empirical validation.

#### <span id="page-12-1"></span>C.2 Experiments on efficiency

As shown in the Table 12, we evaluate the efficiency of our proposed method by measuring both the FLOPs and the number of parameters during inference. Compared to several open-source methods, our model maintains low computational and parameter overhead while achieving better performance. Additionally, we also report the results of key modules in our method including Audio Importance Predictor (AIP) ,Multi-Granularity Fusion (MGF) .

#### <span id="page-12-2"></span>C.3 Experiments on Event-Level Fusion

**Slots with supervision.** Since the moment boundary label is coarse-level and does not have its contained event split, we cannot utilize it to provide explicit event-level supervision. Therefore, we adopt the unsupervised slot attention mechanism to implicitly learn the potential event contexts. Specifically, we utilize the moment boundary label to additionally supervise the global content of all events. After slot interaction, the event-level sequences are globally

<span id="page-12-6"></span>Table 12: Comparison on flops and params.

| Method      | Flops(G)            | Params(M)           | R1@7  |
|-------------|---------------------|---------------------|-------|
| EAMAT       | 9.97                | 94.12               | 41.96 |
| BAM-DETR    | 1.39                | 13.43               | 39.38 |
| FlashVTG    | 1.05                | 8.73                | 38.01 |
| QD-DETR     | 0.82                | 6.36                | 32.55 |
| Moment-DETR | 0.26                | 3.23                | 38.01 |
| ADPN        | 0.34                | 1.54                | 41.10 |
| IMG         | 0.38                | 3.31                | 44.23 |
| -AIP        | $9.87\times10^{-5}$ | $5.12\times10^{-4}$ | -     |
| -MGF        | 0.20                | 1.91                | -     |

<span id="page-12-7"></span>Table 13: Ablation studies on supervised slots.

| Method               | R1@3  | R1@5  | R1@7  | mIOU  |
|----------------------|-------|-------|-------|-------|
| slot w/ supervision  | 73.77 | 59.60 | 41.19 | 54.51 |
| slot w/o supervision | 74.84 | 59.92 | 41.32 | 54.83 |

<span id="page-12-8"></span>Table 14: Performance under different slot numbers and iterations.

| #Slot \ #Iter | 1     | 2     | 3     | 4     | 5     |
|---------------|-------|-------|-------|-------|-------|
| 2             | 40.45 | 40.75 | 41.23 | 41.01 | 40.76 |
| 3             | 40.23 | 40.92 | 41.32 | 41.20 | 41.25 |
| 4             | 39.88 | 40.45 | 41.10 | 41.22 | 40.95 |
| 5             | 39.20 | 39.70 | 40.12 | 40.70 | 40.32 |

projected into a 1D sequence via an MLP and Sigmoid for supervision with binary cross-entropy loss. As shown in Table 13, the global supervision yields a similar performance compared to the unsupervised one. We assume that this is due to the limited granularity of available supervision in VMR, while the unsupervised version can implicitly learn the potential events.

**Number of slots and iter.** Table 14 summarizes the performance with varying numbers of slots and iterations. While increasing the number of slots and iterations increases computational overhead, we find that using 2 or 3 slots with 3 or 4 iterations achieves satisfactory performance. Moreover, using 3 slots and 3 iterations offers a good balance between performance and computational cost.

#### <span id="page-12-3"></span>C.4 Experiments on weak supervision

To assess the robustness of our model under limited supervision, we further conduct evaluation using reduced training data (70%, 80%, and 90% subsets). As shown in Table 15, our model largely maintains its performance and consistently outperforms the strong baseline ADPN.

#### <span id="page-12-4"></span>C.5 Audio importance distribution

Table 16 shows the distribution of audio importance scores, where most samples fall within the range of 0.15 to 0.45, further validating that audio serves as an auxiliary modality.

<span id="page-13-3"></span><span id="page-13-0"></span>Table 15: Performance under different training set sizes to evaluate weak supervision capability.

| Method | Samples for train (%) |       |       |       |  |  |
|--------|-----------------------|-------|-------|-------|--|--|
|        | 70                    | 80    | 90    | 100   |  |  |
| ADPN   | 37.07                 | 38.78 | 39.47 | 41.10 |  |  |
| IMG    | 41.21                 | 43.29 | 43.52 | 44.23 |  |  |

<span id="page-13-4"></span>Table 16: Distribution of audio importance scores across samples.

| Score Range | < 0.15 | 0.15-0.25 | 0.25-0.35 | 0.35-0.45 | >0.45 |
|-------------|--------|-----------|-----------|-----------|-------|
| Count       | 26     | 963       | 1861      | 665       | 205   |

<span id="page-13-5"></span>Table 17: Ablation studies on AIP with zero importance.

<span id="page-13-1"></span>

| Method      | R1@3  | R1@5  | R1@7  | mIOU  |
|-------------|-------|-------|-------|-------|
| IMG         | 82.74 | 71.93 | 54.27 | 62.76 |
| IMG $(p=0)$ | 80.27 | 70.22 | 50.96 | 59.84 |

#### C.6 Experiments on failed AIP

We explore the impact when AIP mistakenly predicts audio importance to zero (p=0) on Charades-AudioMatter. As shown in Table 17, such change degrade the performance. The results not only show the importance of audio clues, but also demonstrate the effectivenesss of our proposed audio-aware desgin.

### <span id="page-13-2"></span>D Implement Details

For all datasets, we set the initial learning rate to 0.0005, and the maximum number of frames to 128. We use AdamW [38] for optimization and linear decay scheduling, and the maximum epoch number is 100 for all of them with batch size 16. We use I3D [7] as pretrained visual features on all datasets. For the audio pretraining models, following previous work [8], we utilized PANN [27], pretrained on AudioSet [17] dataset, and VGGish [21], pretrained on YouTube-100M [21] dataset, for Charades-STA/Charades-AudioMatter and ActivityNet Caption, respectively. We initialize words with 300d GloVe [43] embeddings. To further demonstrate the generalizability of our model, we also use InternVideo2 [55] and LLaMA [51] for visual and textual backbone. We set  $\epsilon_{min}$  to 0.2,  $\gamma$  to 3 for Charades-STA/Charades-AudioMatter,  $\epsilon_{min}$  to 0.1 and  $\gamma$  to 2 for ActivityNet Captions. All experiments are implemented on a single NVIDIA 3090 GPU.