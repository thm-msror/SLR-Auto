# <span id="page-0-5"></span>Question-Aware Global-Local Video Understanding Network for Audio-Visual Question Answering

Zailong Chen [,](https://orcid.org/0009-0003-8431-5471) Lei Wan[g](https://orcid.org/0000-0002-0961-0441) , *Senior Member, IEEE*, Peng Wang [,](https://orcid.org/0000-0002-5397-9115) and Peng Gao

*Abstract*— As a newly emerging task, audio-visual question answering (AVQA) has attracted research attention. Compared with traditional single-modality (e.g., audio or visual) QA tasks, it poses new challenges due to the higher complexity of feature extraction and fusion brought by the multimodal inputs. First, AVQA requires more comprehensive understanding of the scene which involves both audio and visual information; Second, in the presence of more information, feature extraction has to be better connected with a given question; Third, features from different modalities need to be sufficiently correlated and fused. To address this situation, this work proposes a novel framework for multimodal question answering task. It characterises an audiovisual scene at both global and local levels, and within each level, the features from different modalities are well fused. Furthermore, the given question is utilised to guide not only the feature extraction at the local level but also the final fusion of global and local features to predict the answer. Our framework provides a new perspective for audio-visual scene understanding through focusing on both general and specific representations as well as aggregating multimodalities by prioritizing question-related information. As experimentally demonstrated, our method significantly improves the existing audio-visual question answering performance, with the averaged absolute gain of 3.3% and 3.1% on MUSIC-AVQA and AVQA datasets, respectively. Moreover, the ablation study verifies the necessity and effectiveness of our design. Our code will be publicly released.

*Index Terms*— Audio-visual question answering, video understanding, multimodal learning, deep learning.

# I. INTRODUCTION

I N RECENT times, question answering has garnered significant attention and has exhibited its potential for various applications such as information retrieval, human-computer interaction, and visual/auditory assistance. Notably, considerable advancements have been made in the domain of visual question answering (VQA) [\[1\],](#page-9-0) [\[2\],](#page-9-1) [\[3\],](#page-9-2) [\[4\],](#page-9-3) [\[5\],](#page-9-4) [\[6\], a](#page-9-5)nd audio question answering (AQA) [\[7\],](#page-9-6) [\[8\]. W](#page-9-7)hile VQA and

Manuscript received 23 April 2023; revised 13 August 2023; accepted 10 September 2023. Date of publication 2 October 2023; date of current version 9 May 2024. This article was recommended by Associate Editor C. Herglotz. *(Corresponding author: Lei Wang.)*

Zailong Chen and Lei Wang are with the School of Computing and Information Technology, University of Wollongong, Wollongong, NSW 2522, Australia (e-mail: zc881@uowmail.edu.au; leiw@uow.edu.au).

Peng Wang is with the School of Computer Science and Engineering, University of Electronic Science and Technology of China, Chengdu 610056, China (e-mail: p.wang6@hotmail.com).

Peng Gao is with the Institute of Computer Science, Beijing Normal University–Hong Kong Baptist University United International College, Zhuhai 519000, China (e-mail: s230201705@mail.uic.edu.cn).

Color versions of one or more figures in this article are available at https://doi.org/10.1109/TCSVT.2023.3318220.

Digital Object Identifier 10.1109/TCSVT.2023.3318220

<span id="page-0-3"></span><span id="page-0-2"></span>AQA concentrate on comprehending the signal of a single modality for generating answers to the given question, the past few years have observed a growing interest in tackling more pervasive and intricate audio-visual scenarios, such as audio-visual scene-aware dialog (AVSD) [\[9\],](#page-9-8) [\[10\],](#page-9-9) [\[11\]](#page-9-10) and the more recent audio-visual question answering (AVQA) [\[12\],](#page-9-11) [\[13\],](#page-9-12) [\[14\]](#page-9-13) tasks. The availability of AVQA datasets based on audio-visual scenes has substantially broadened the application scope in the multimodal domain. For instance, the expeditious extraction of pertinent information from surveillance videos is achievable through the analysis of visual and auditory data, coupled with targeted query questions, thereby circumventing the arduous process of frame-by-frame video observation. This methodology exhibits promising potential in alleviating the temporal and labor-intensive burdens associated with managing extensive video recordings, thereby facilitating expeditious and effective acquisition of pivotal and contextually relevant details by law enforcement agencies.

The comprehension of audio-visual scenes represents a crucial prerequisite for AVQA systems. Extensive prior research has underscored the importance of global audio and visual features in facilitating scene understanding within AVQA [\[12\],](#page-9-11) [\[13\],](#page-9-12) [\[14\]. H](#page-9-13)owever, it is argued that relying solely on such global features may not be sufficient for achieving a comprehensive understanding of scenes. Instead, it is advantageous to integrate task-specific local features, e.g., focusing on critical moments relevant to the posed question [\[2\]](#page-9-1) and integrating multi-granularity information from the video [\[6\]. T](#page-9-5)o illustrate, Fig. [1](#page-1-0) depicts a scenario wherein diverse video content necessitates different attention to answer distinct questions, and even individual words within a question hold varying degrees of importance. Hence, a holistic comprehension of video content cannot be attained solely through a global or local perspective in isolation. Instead, a comprehensive understanding requires absorbing both global and task-relevant local features. Furthermore, questions should act as guiding cues to better identify key audio or visual information within the scene.

<span id="page-0-4"></span><span id="page-0-1"></span><span id="page-0-0"></span>Current AVQA methods primarily extract audio and visual features globally, without considering the importance of local feature extraction [\[12\],](#page-9-11) [\[13\],](#page-9-12) [\[14\],](#page-9-13) [\[15\]. C](#page-9-14)onsequently, subtle and task-relevant information may be overlooked. Additionally, these methods do not thoroughly investigate the significance of question for feature extraction or fusion, potentially impeding the ability to effectively answer questions. Given the complementary nature of modalities in AVQA, adequate correlation between different modalities during feature

1051-8215 © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Fig. 1. This figure is to illustrate the AVQA problem. In the context of the audio-visual question answering task, the areas and elements that necessitate focused attention vary with the specific question presented, encompassing temporal periods, visual regions, and acoustic zones. To facilitate the identification of these areas of interest for distinct questions within the same video, we have incorporated green trumpet icons, yellow highlights, and red dashed boxes into the accompanying figure.

<span id="page-1-1"></span>extraction or fusion is essential for scene comprehension, as is done in [\[16\].](#page-9-15)

We propose a new framework for AVQA to address the current situation. Our approach aims to extract both global and local video features and to fuse multimodal information in a question-aware manner to generate an accurate answer. The proposed framework consists of two stages, as illustrated in Fig. [2.](#page-2-0) In the first stage, the model extracts global and local features from the audio and visual data. The global features are extracted using the co-attention mechanism, which correlates global audio and visual features. The local features are obtained by identifying important local information from audio and visual inputs based on the given question. Additionally, the extracted audio and visual features are fused to ensure that they are integrated with each other. In the second stage, the question is used again as a guide to further refine the fusion of global and local features before they are decoded into the answer of the question by a classifier. Our framework is fully "question-aware" as the question is used to guide both feature extraction and fusion. Our approach not only provides global features to attain general, questionindependent information for overall video understanding, but also leverages local features to highlight information critical to deducing the correct answer.

Our contributions are summarised as follows:

1. The present study investigates the emerging task of AVQA and delineates its principal challenges and requirements. To address these issues and to attain more precise question answering, we propose a novel framework that aims to enhance the comprehension, extraction, and fusion of multimodal information.

- 2. Our approach surpasses prior works by characterizing video content with both global and local features that are critical for AVQA. This innovative design enables our model to comprehensively interpret the audio-visual scenes.
- 3. Our framework achieves promising results by integrating the question into the AVQA task, thereby providing guidance for feature extraction and fusion. Moreover, we emphasize sufficient correlation and fusion of global and local features within and between them.

An experimental study is carried out on the recently introduced MUSIC-AVQA [\[13\]](#page-9-12) and AVQA [\[14\]](#page-9-13) datasets, employing various questions types and modalities. The outcomes corroborate the efficacy and benefits of the proposed approach over the existing related methods, and the ablation study substantiates the indispensability and significance of our design.

## II. RELATED WORK

## *A. AVQA*

AVQA is a nascent task that has garnered considerable attention from the deep learning community due to the remarkable achievements of deep neural networks in the realm of multimodal research. The AVQA mission requires comprehensive understanding and integration of diverse modalities, leading to precise responses to distinct questions. AVSD, akin to the AVQA task, focuses on human-to-human dialogue scenarios, presenting textual captions or dialogs in conjunction with audio and visual cues. In contrast to AVSD task, which draws upon audio, visual, and dialog text from videos to facilitate comprehensive scene understanding, the AVQA task allows for the understanding of video content solely based on audio and visual information. The initial AVQA dataset, Pano-AVQA [\[12\], w](#page-9-11)as introduced to explore audio-visual question answering in panoramic videos, encompassing only two question types, namely *existential* and *location*. In order to tackle more complex question types and reason audiovisual scenarios from multiple perspectives, MUSIC-AVQA [\[13\]](#page-9-12) was unveiled, which is a large-scale dataset comprising real-life audiovisual scenarios (concerts). It embraces five questions (i.e., *counting*, *comparative*, *location*, *existential*, and *temporal*) and spans over nine question types by merging these five question aspects with different modalities. Recently, a real-life scenebased AVQA benchmark was proposed in [\[14\], w](#page-9-13)hich further expands the audiovisual scene coverage of AVQA tasks. In the classic question answering setting in the literature [\[1\],](#page-9-0) [\[10\],](#page-9-9) objective of the agent entails selecting the appropriate answer from a pool of predefined potential responses, aligning with the query question and specific input modalities. This setting also governs the two aforementioned AVQA tasks.

## *B. Deep Audio-Visual Learning*

<span id="page-1-3"></span><span id="page-1-2"></span>Several audio-visual learning methods have been proposed in recent years. Reference [\[17\]](#page-9-16) presented an audiovisual slowfast network for modeling multimodal concepts hierarchically. Reference [\[18\]](#page-9-17) proposed a multimodal bottleneck attention

<span id="page-2-0"></span>![](_page_2_Figure_2.jpeg)

Fig. 2. The proposed AVQA model. Our model can be divided into two stages. The first stage is to understand video contents at both global (general) and local (specific) levels. Furthermore, the second stage integrates a question-aware fusion module of global and local features and an answer classifier. More detailed description is provided in Section [III.](#page-2-1)

<span id="page-2-4"></span><span id="page-2-3"></span><span id="page-2-2"></span>method for efficiently fusing audio and visual inputs. Reference [\[19\]](#page-9-18) developed an architecture that projects multimodal inputs into a joint multimodal embedding space using a combinatorial loss during training. Contrastive learning was utilized in [\[20\]](#page-9-19) and [\[21\]](#page-9-20) to obtain robust audio and visual encoders and an audiovisual embedding space, respectively. These methods offer diverse perspectives on how to merge multimodalities. Reference [\[12\]](#page-9-11) proposed a combinatorial attention paradigm that combines audio, visual, and textual features for specific audio-visual question-answering tasks. Additionally, [\[13\]](#page-9-12) utilized audio features to query visual features for spatial video comprehension, followed by an attention module that identifies temporal information in the audio and visual features using the question as a clue. Reference [\[14\]](#page-9-13) introduced a hierarchical audio-visual fusing method that can be combined with existing bi-modal fusion methods to aggregate multimodal features.

Existing methods primarily analyze audio-visual scenes from a global viewpoint, thus, ignoring the significance of the task-relevant local information of the scene in the AVQA task. The present study aims to address the aforementioned limitations by extracting video features from both global and local levels and emphasizing the role of the question in AVQA task. Moreover, our proposed methodology extensively correlates and fuses data from different modalities.

## III. PROPOSED METHOD

<span id="page-2-1"></span>This section introduces our proposed model in detail. The overall architecture of the model is shown in Fig. [2.](#page-2-0)

## *A. Workflow of the Framework*

The inputs of the model consist of audio and visual signals sampled from a video, along with a question based on the video. The model is then tasked with accurately answering the given question based on multimodal inputs. The proposed model consists of the following key components: (1) separate encoders extract features of the three modalities of audio, visual, and text; (2) The first stage of the model then refines and integrates these multimodal features, effectively identifying general and question-specific features of the audio and visual signals through global and local branches; (3) contrastive learning is used atop the global and local branches to ensure alignment between audio and visual features; (4) the global and local features flow to the second stage simultaneously for final fusion, during which they are sufficiently correlated and aggregated under the guidance of the question; (5) the final fused feature is fed into a classifier to determine an appropriate answer.

The aim of the presented framework is to enhance the optimization of an objective function for attaining the better audiovisual feature representation. The objective function includes the alignment of global-level audio-visual features and local-level audio-visual features, in addition to ensuring the precision of question answering. Details are shown below.

## B. Video Understanding

1) Feature Extraction: Given an input video with both visual and audio tracks, we sample a fixed number (T) of segments from it, and this gives T positive audio-visual pairs denoted by  $\left\{A_t, V_t^+\right\}_{t=1}^T$ . Also, we randomly sample a visual sequence from another video and compose negative audio-visual pairs  $\left\{A_t, V_t^-\right\}_{t=1}^T$ . These positive and negative audio-visual pairs will be employed to align the audio-visual features coming from the same video through contrastive learning. The question text is tokenized into K individual tokens  $\left\{Q_k\right\}_{k=1}^K$ .

Here, we extract the audio feature vectors  $\{\mathbf{a}_t\}_{t=1}^T$ , positive visual feature vectors  $\{\mathbf{v}_t^+\}_{t=1}^T$ , negative visual feature vectors  $\{\mathbf{v}_t^-\}_{t=1}^T$ , and question feature vectors  $\{\mathbf{q}_k\}_{k=1}^K$  from  $\{A_t\}_{t=1}^T$ ,  $\{V_t^+\}_{t=1}^T$ ,  $\{V_t^-\}_{t=1}^T$ , and  $\{Q_k\}_{k=1}^K$ , respectively. Specifically, we adopt the commonly used networks to extract the features from different modalities, i.e., VGGish [22] for audio feature extraction, ResNet-18 [23] for visual feature extraction, and LSTM for textual input. Thus, we obtain audio feature matrix  $\mathbf{A}$ , positive visual feature matrix  $\mathbf{V}^+$ , negative visual feature matrix  $\mathbf{V}^-$ , and question feature matrix  $\mathbf{Q}$ .

<span id="page-3-1"></span>2) Multimodal Fusion and Interaction: When presented with complex, multimodal data, it is imperative to consider the intricate interactions that exist both within and between the various modalities. In order to accomplish this, we leverage the extensively utilized cross-attention mechanism [5], [24], [25], specifically implementing co-attention as illustrated in Fig. 2. We provide a thorough exposition of this module as it plays a significant role in our model.

<span id="page-3-2"></span>Given two input modalities (i.e., left-input modality  $\mathbf{M}_l$  and right-input modality  $\mathbf{M}_r$ ), co-attention module models their interaction by two channels. Before modeling the interaction information of two modalities, we employ self-attention to further capture their long-range interdependent features.

$$\mathbf{F}_l = \text{Self-att}(\mathbf{M}_l, \mathbf{M}_l); \quad \mathbf{F}_r = \text{Self-att}(\mathbf{M}_r, \mathbf{M}_r), \quad (1)$$

where  $\mathbf{F}_l$  and  $\mathbf{F}_r$  are the features of left and right channels after processing by the self-attention. Self-att is the self-attention operation.

To learn the self-influence caused by the left-input modality itself, the left channel feature  $\mathbf{F}_l$  is used to query itself, generating the self-modality attentional feature. Simultaneously, to capture the interactive influence that the right-input modality brings to the left one, the right channel feature  $\mathbf{F}_r$  is applied to query  $\mathbf{F}_l$  to generate the cross-modality attentional feature. These two features are then averaged and concatenated with  $\mathbf{F}_l$ , and the resulting concatenated feature is fed into a Feedforward Neural Network (FFN) to obtain the left attentional feature  $\mathbf{F}_L$  thereby enabling interaction between the left and right modalities. The operation can be formulated as follows.

$$\mathbf{F}_L = \text{FFN}(\text{Cat}(\mathbf{F}_l, (\text{Self-att}(\mathbf{F}_l, \mathbf{F}_l) + \text{Bi-att}(\mathbf{F}_l, \mathbf{F}_r))/2)),$$

where Cat is the concatenation operation. Bi-att is the Bi-modal attention operation.

In a similar fashion, the acquisition of the right attentional feature  $\mathbf{F}_R$  can be achieved through the introduction of the left modality into the right one. Details are shown as follows.

$$\mathbf{F}_R = \text{FFN}(\text{Cat}(\mathbf{F}_r, (\text{Self-att}(\mathbf{F}_r, \mathbf{F}_r) + \text{Bi-att}(\mathbf{F}_r, \mathbf{F}_l))/2)),$$
(3)

Note that the co-attention can be stacked multiple times in our model for adequate and hierarchical extraction and fusion of two inputs.

3) Question-Aware Feature Extraction: Effective processing of AVQA tasks necessitates the question serving as an anchor for locating crucial information within video content. We address this issue by proposing the question-aware attention mechanism, which is illustrated in Fig. 2. Herein, we present a detailed exposition of its structure.

<span id="page-3-0"></span>This module consists of three inputs, denoted as left modality  $\mathbf{M}_{l,q}$ , right modality  $\mathbf{M}_{r,q}$ , and question  $\mathbf{Q}$ . The utilization of question-aware attention allows for the capture of features that are relevant to the posed question. The process commences with the implementation of self-attention in order to capture long-range interdependent features of different inputs.

$$\mathbf{F}_{l,q} = \text{Self-att}(\mathbf{M}_{l,q}, \mathbf{M}_{l,q}), \tag{4}$$

$$\mathbf{F}_{r,q} = \text{Self-att}(\mathbf{M}_{r,q}, \mathbf{M}_{r,q}), \tag{5}$$

$$\mathbf{F}_q = \text{Self-att}(\mathbf{Q}, \mathbf{Q}),$$
 (6)

where  $\mathbf{F}_{l,q}$ ,  $\mathbf{F}_{r,q}$  and  $\mathbf{F}_q$  are the features of  $\mathbf{M}_{l,q}$ ,  $\mathbf{M}_{r,q}$ , and question  $\mathbf{Q}$  after processing by the self-attention, respectively.

To effectively locate the question-relevant information across distinct input sources, the question feature  $\mathbf{F}_q$  is used to query  $\mathbf{F}_{l,q}$  and  $\mathbf{F}_{r,q}$ . Subsequently, the obtained results are concatenated with  $\mathbf{F}_q$  to derive the question-guided features. The operation is shown as follows.

$$\mathbf{F}_{L,q} = \text{FFN}(\text{Cat}(\mathbf{F}_q, \text{Bi-att}(\mathbf{F}_{l,q}, \mathbf{F}_q))), \tag{7}$$

$$\mathbf{F}_{R,q} = \text{FFN}(\text{Cat}(\mathbf{F}_q, \text{Bi-att}(\mathbf{F}_{r,q}, \mathbf{F}_q))), \tag{8}$$

where  $\mathbf{F}_{L,q}$  and  $\mathbf{F}_{R,q}$  are the features extracted under the guidance of the question.

4) Global Branch: In this branch, integration of extracted audio and visual information is accomplished through employment of the co-attention module, leading to the acquisition of comprehensive video features. Specifically, as shown in Fig. 2, the audio (A) and visual ( $\mathbf{V}^+/\mathbf{V}^-$ ) modalities are fed into the co-attention module to sufficiently correlate with each other. Therefore, we can obtain  $\mathbf{A}_{glb}^+$  and  $\mathbf{A}_{glb}^-$  denoting global audio features by fusing A with  $\mathbf{V}^+$  and  $\mathbf{V}^-$ , respectively. Additionally,  $\mathbf{V}_{glb}^+$  and  $\mathbf{V}_{glb}^-$  represent the global visual features through separately integrating  $\mathbf{V}^+$  and  $\mathbf{V}^-$  with  $\mathbf{A}$ .

On the top of the global branch, we concatenate positive features  $\mathbf{A}_{glb}^+$  and  $\mathbf{V}_{glb}^+$  from the audio and visual channels as the global feature  $\mathbf{F}_{glb}$ .

$$\mathbf{F}_{glb} = \operatorname{Cat}(\mathbf{A}_{glb}^+, \mathbf{V}_{glb}^+). \tag{9}$$

It should be noted that the usage of negative audio-visual pairs is exclusively limited to the first stage. Given that the model deciphers visual and audio content separately through distinct channels in the first stage of the proposed framework, it becomes crucial to establish alignment between audio and visual modalities from both temporal and content perspectives. We employ contrastive learning to guarantee the alignment of positive audio-visual pairs and precise feature extraction, encompassing positive audio and visual features alongside all negative visual features present within a given batch. All negative global features in a batch are gathered into the global negative feature tensor  $\mathbf{V}_{all,glo}^-$ .

<span id="page-4-1"></span>
$$\mathcal{L}_{glb} = \mathcal{L}_{InfoNCE}(\mathbf{A}_{glb}^+, \mathbf{V}_{glb}^+, \mathbf{V}_{all,glo}^-), \tag{10}$$

where  $L_{InfoNCE}$  is the InfoNCE (in which NCE means Noise Contrastive Estimation) loss [26], which is used to implement the contrastive learning. We use  $\mathcal{L}_{glb}$  to denote the contrastive learning loss of the global branch.

5) Local Branch: In this branch, the question-aware attention is implemented to facilitate the capture of task-relevant local features from audio and visual modalities. By incorporating question-aware attention in conjunction with the co-attention, it becomes possible to model both the self-influence stemming from single modality and the interactive influence among three modalities, including the question to audio, question to visual, audio to visual, and visual to audio, with further details provided below.

Initially, we initiate the information processing pipeline by inputting the visual  $(\mathbf{V}^+/\mathbf{V}^-)$ , audio  $(\mathbf{A})$ , and text  $(\mathbf{Q})$  modalities into the question-aware attention module. This step serves to effectively identify key information that is closely associated with the posed question. Subsequently, the audio and visual information, having undergone the process of localization and refinement, is further integrated using coattention module. As a result of this integration, we are able to derive distinct local audio features, denoted as  $\mathbf{A}_{loc}^+$  (positive) and  $\mathbf{A}_{loc}^-$  (negative), which are obtained by fusing the audio feature with positive and negative visual features, respectively. Additionally, our approach also enables the extraction of local visual features, represented as  $\mathbf{V}_{loc}^+$  (positive) and  $\mathbf{V}_{loc}^-$  (negative), contributing to a comprehensive and detailed representation of the underlying data.

Next, positive local features of audio and visual modalities are concatenated as the local video feature  $\mathbf{F}_{loc}$ .

$$\mathbf{F}_{loc} = \operatorname{Cat}(\mathbf{A}_{loc}^+, \mathbf{V}_{loc}^+). \tag{11}$$

Similar to the global branch,  $\mathbf{V}_{all,loc}^-$  encompasses all local negative visual features  $\mathbf{V}_{loc}^-$  in a batch. Moreover, the contrastive learning is utilised between the positive features and negative features to align the positive audio-visual features.

$$\mathcal{L}_{loc} = \mathcal{L}_{InfoNCE}(\mathbf{A}_{loc}^+, \mathbf{V}_{loc}^+, \mathbf{V}_{all.loc}^-), \tag{12}$$

where  $\mathcal{L}_{loc}$  is the contrastive learning loss of the local branch.

#### C. Global Local Fusion and Answer Prediction

1) Global-Local Fusion: In this part, we present a fusion module that strategically integrates global and local features, guided by question features. As depicted in the second stage of

- Fig. 2, the process unfolds in three steps. Firstly, the question-aware attention module processes global ( $\mathbf{F}_{glb}$ ), local ( $\mathbf{F}_{loc}$ ), and question ( $\mathbf{Q}$ ) features, facilitating the identification of key question-relevant information. Secondly, the refined global and local features undergo fusion using the co-attention module, ensuring a comprehensive integration. Lastly, the output of the co-attention module is concatenated, yielding the final fused feature ( $\mathbf{F}_{final}$ ), which subsequently undergoes answer decoding.
- 2) Answer Prediction: At last,  $\mathbf{F}_{final}$  is used as the input of a classifier to predict the answer from an answer candidate pool, i.e., probabilities  $\mathbf{p} \in \mathbb{R}^C$ , where C is the size of the answer candidate pool.
- 3) Objective Function: AVQA is a challenging task to optimize as it requires comprehensive scene understanding and question-aware feature fusion. To enhance learning, we optimize three losses to help our model learn informative and correct representations. For the first loss, with the predicted probability vector **p** and the ground-truth label **y**, we use a cross-entropy loss to optimize the prediction accuracy:

$$\mathcal{L}_{qa} = -\sum_{c=1}^{C} y_c \log(p_c), \tag{13}$$

where  $y_c$  and  $p_c$  are c-th component of y and p.

At the same time, we use the InfoNCE loss both in global branch and local branch to ensure the alignment of extracted positive audio-visual features. As a result, the objective function of our model is defined as

$$\mathcal{L}_{AVQA} = \mathcal{L}_{qa} + \lambda_{glb} \times \mathcal{L}_{glb} + \lambda_{loc} \times \mathcal{L}_{loc}, \tag{14}$$

where  $\lambda_{glb}$  and  $\lambda_{loc}$  denote the scaling factors.

## <span id="page-4-0"></span>IV. EXPERIMENTAL RESULT

In this section, we present a comprehensive performance analysis of the proposed model by introducing our experimental setup, dataset, evaluation protocols, and baselines in Sec. A. Subsequently, we report the experimental results to showcase the efficacy of our method in Sec. B. This includes a comparison with the aforementioned baselines, an ablation study, hyperparameter exploration, and a visualized attention map of the local branch. Our findings highlight the superiority of the proposed model and offer valuable insights into its inner workings.

## A. Experimental Setup

1) Dataset and Evaluation: Audio-visual question answering is a newly emerging task. MUSIC-AVQA [13] and AVQA [14] are the latest and only two publicly released datasets. Our study focuses on the utilization of these two datasets as the testbed to investigate the performance of our model. MUSIC-AVQA [13] comprises 9, 288 videos (mainly about the concert scenarios) and 45, 867 question-answer pairs, spanning nine distinct question types, i.e., Audio questions: Counting and Comparative; Visual questions: Counting and Location, Audio Visual questions: Existential, Location, Counting, Comparative, and Temporal. While AVQA [14] consists of 57, 015 videos reflecting the real-world scenarios and 57, 335 specially-designed question-answer pairs relying

on clues from audio and visual modalities, including eight question types, i.e., *Which, Come From, Happening, Where, Why, Before Next, When*, and *Used For*. We use a predefined set of training, testing, and validation in the benchmark datasets [\[13\],](#page-9-12) [\[14\].](#page-9-13) Each original video in the dataset is about 60-second long and is divided into 60 non-overlapping segments of the same length. The sampling rates of sounds and video frames are 16 *k H z* and 1 *f ps*, respectively. Following the baseline [\[13\], w](#page-9-12)e sampled each video by taking one-second long segment every six-second long segment, constituting 10 one-second long segments. In our performance assessment, we adopt answer prediction accuracy as the primary metric, as depicted in Eq. [\(15\).](#page-5-0) This accuracy metric is determined by assessing the proportion of correctly predicted questions for each question type. The numerator *N<sup>p</sup>* in Eq. [\(15\)](#page-5-0) represents the count of accurately predicted questions, while the denominator *Nall* in Eq. [\(15\)](#page-5-0) corresponds to the total number of questions belonging to that particular question type. The algorithm objective is to predict the answer with the highest probability, which ideally corresponds to the correct answer.

$$\mathbf{Acc} = \frac{N_p}{N_{all}} \tag{15}$$

*2) Model Training:* The hyperparameters for the proposed model are pre-determined as per the following specifications. Certain training configurations are adopted from the MUSIC-AVQA baseline [\[13\], e](#page-9-12)ncompassing aspects such as a 512-D feature for visual, audio, and text modalities, an initial learning rate of 0.0001, and a comprehensive text preprocessing methodology that includes tokenization, padding, and vocabulary construction.

However, certain divergences from the baseline are introduced. Specifically, the learning rate will drop by multiplying 0.1 every eight training epochs (in contrast to every 10 epochs in [\[13\]\).](#page-9-12) The model undergoes 15 training epochs (compared to 30 epochs in [\[13\]\) w](#page-9-12)ith a mini-batch size of 24 (in contrast to 64 in [\[13\]\).](#page-9-12)

Two Nvidia RTX 3090 Ti GPUs are employed to train the proposed model with the Adam optimizer. The multihead attention mechanism utilizes four parallel attention heads (*h* = 4). Within the model architecture, the global branch consists of two stacked co-attention modules, while both the local branch and global-local fusion block have one stacked co-attention module each. We set the weights for the global and local branches to be λ*glb* = λ*loc* = 0.1, respectively, in Eq. [\(14\).](#page-4-0)

In order to ensure a fair comparison, the feature encoders used are identical to that of the baseline [\[13\]:](#page-9-12) pre-trained VGGish [\[22\]](#page-9-21) for audio, pre-trained ResNet-18 [\[23\]](#page-9-22) for visual, and an LSTM for processing questions. During training, the VGGish and ResNet-18 encoders remain frozen. The main novelty of this work lies in the aspects of video understanding and modality fusion, where significant differences emerge compared to the other counterparts.

*3) Baselines:* We assess the efficacy of our model on the MUSIC-AVQA [\[13\]](#page-9-12) and AVQA [\[14\]](#page-9-13) datasets by comparing it with multiple existing relevant methods. Specifically, we compare our approach with single-modality QA methods, such as AudioQA [\[7\], V](#page-9-6)isualQA [\[28\],](#page-9-26) and VideoQA [\[31\],](#page-9-27) [\[33\],](#page-10-0) [\[34\], t](#page-10-1)o demonstrate the benefits of multimodal perception in facilitating question answering. AudioQA methods respond questions based on audio signals without the help of visual information. On the contrary, VisualQA and VideoQA approaches answer questions based on the visual features without the audio input. Visual input of the VisualQA is typically an image, while VideoQA is based on a video segment input. Moreover, we evaluate our algorithm against existing work developed for audio-visual scene understanding to verify its performance improvement. These audio-visual scene-based methods include:

- *4) AVSD:* AVSD method answers questions through understanding the video content from visual, audio, and dialogue information. But in AVQA tasks, there is no dialogue for video understanding, so we compare AVSD method for investigating the audiovisual scene perception ability of our model without dialog assistance. Reference [\[32\]](#page-10-2) uses a multimodal attention mechanism to fuse three modalities (audio, visual, and textual) with equal contribution to answer generation.
- <span id="page-5-0"></span>*5) AVQA:* AVQA-based methods are what we focus on comparing to demonstrate the superiority of our model, including (1) Pano-AVQA [\[12\], w](#page-9-11)hich employs a cross-attention module to fuse three modalities; (2) HAVF [\[14\], w](#page-9-13)hich serves as the baseline of AVQA [\[14\]](#page-9-13) dataset by incorporating with other bi-modality fusion method. It comprises three fusion methods and ensembles three fusion outputs through an averaging strategy to generate the answer; (3) ST-AVQA [\[13\],](#page-9-12) which is the baseline method reported on the MUSIC-AVQA [\[13\]](#page-9-12) dataset. ST-AVQA locates the spatial and temporal area in audio and visual signals using an attention mechanism and fuses three modalities to predict the answer.

## *B. Results and Discussion*

In this section, we present a comprehensive evaluation of the proposed model through various analyses. Firstly, we compare its performance against relevant QA approaches to establish its effectiveness and superiority. Secondly, extensive ablation studies are conducted to gain a comprehensive understanding of inner workings of our model. Thirdly, we report the results obtained with different hyperparameters to explore the robustness of the model. Fourthly, we provide a visualized result to enhance the interpretation of the local branch. Finally, the prediction results of our model are depicted to further demonstrate the effectiveness of our model.

*1) Results on MUSIC-AVQA Dataset:* Tab. [I](#page-6-0) presents the findings of our study evaluating recent QA methods and our proposed model on the MUSIC-AVQA [\[13\]](#page-9-12) dataset. The results except the HCRN+HAVF and Ours in Tab. [I](#page-6-0) are all imported from [\[13\]. W](#page-9-12)e implement HCRN+HAVF according to the architecture depicted in [\[14\].](#page-9-13) From the last column of Tab. [I,](#page-6-0) we can clearly tell that the averaged accuracies of single-modality QA methods (i.e., AudioQA, VisualQA, and VideoQA) are significantly lower than multimodality QA approaches, such as AVSD and AVQA. These findings suggest that multimodal perception can enhance audio-visual scene understanding and improve QA performance. Furthermore,

TABLE I

<span id="page-6-0"></span>FINE-GRAINED EVALUATION PERFORMANCE (%) OF BASELINES AND OUR PROPOSED MODEL ON MUSIC-AVQA [\[13\]](#page-9-12) DATASET. THE TOP-2 RESULTS ARE HIGHLIGHTED. THE BEST AND SECOND BEST ACCURACIES OF EACH QUESTION TYPE ARE HIGHLIGHTED IN BOLD FORM AND UNDERLINE FORM, RESPECTIVELY. IN ADDITION, FOR EACH QUESTION CATEGORY, WE PRESENT THE PERFORMANCE INCREASE BROUGHT BY OUR METHOD WITH RESPECT TO THE BEST RESULT AMONG ALL THE EXISTING METHODS COMPARED IN THE TABLE

|          |                | Audio-Question |             |         | Visual Question |          |         | Audio-Visual Question |          |          |             |          | All     |         |
|----------|----------------|----------------|-------------|---------|-----------------|----------|---------|-----------------------|----------|----------|-------------|----------|---------|---------|
| Tasks    | Methods        | Counting       | Comparative | Avg.    | Counting        | Location | Avg.    | Existential           | Location | Counting | Comparative | Temporal | Avg.    | Avg.    |
| AudioQA  | FCNLSTM [7]    | 70.8           | 65.66       | 68.9    | 64.58           | 48.08    | 56.23   | 82.29                 | 59.92    | 46.20    | 62.94       | 47.45    | 60.42   | 60.81   |
|          | CONVLSTM [7]   | 73.55          | 67.17       | 71.2    | 67.17           | 55.84    | 61.44   | 82.49                 | 63.08    | 51.85    | 62.13       | 50.36    | 62.56   | 63.79   |
| VisualQA | GRU [1]        | 71.29          | 63.13       | 68.28   | 66.08           | 68.08    | 67.09   | 80.67                 | 61.03    | 51.74    | 62.85       | 57.79    | 63.03   | 65.03   |
|          | Hco_Att [27]   | 70.80          | 54.71       | 64.87   | 63.49           | 67.10    | 65.32   | 79.48                 | 59.84    | 48.8     | 56.31       | 56.33    | 60.32   | 62.45   |
|          | MCAN [28]      | 78.07          | 57.74       | 70.58   | 71.76           | 71.76    | 71.76   | 80.77                 | 65.22    | 54.57    | 56.77       | 46.84    | 61.52   | 65.83   |
|          | PSAC [29]      | 75.02          | 66.84       | 72.00   | 68.00           | 70.78    | 69.41   | 79.76                 | 61.66    | 55.22    | 61.13       | 59.85    | 63.60   | 66.62   |
| VideoQA  | HME [30]       | 73.65          | 63.74       | 69.89   | 67.42           | 70.20    | 68.83   | 80.87                 | 63.64    | 54.89    | 63.03       | 60.58    | 64.78   | 66.75   |
|          | HCRN [31]      | 71.29          | 50.67       | 63.69   | 65.33           | 64.98    | 65.15   | 54.15                 | 53.28    | 41.74    | 51.04       | 46.72    | 49.82   | 56.34   |
| AVSD     | AVSD [32]      | 72.47          | 62.46       | 68.78   | 66.00           | 74.53    | 70.31   | 80.77                 | 64.03    | 57.93    | 62.85       | 61.07    | 65.44   | 67.32   |
|          | Pano-AVQA [12] | 75.71          | 65.99       | 72.13   | 70.51           | 75.76    | 73.16   | 82.09                 | 65.38    | 61.30    | 63.67       | 62.03    | 66.97   | 69.53   |
| AVQA     | HCRN+HAVF [14] | 78.17          | 66.84       | 73.99   | 73.35           | 72.98    | 73.16   | 81.88                 | 67.98    | 49.78    | 61.94       | 63.26    | 65.33   | 68.93   |
| AVQA     | ST-AVQA [13]   | 77.78          | 67.17       | 73.87   | 73.52           | 75.27    | 74.40   | 82.49                 | 69.88    | 64.24    | 64.67       | 65.82    | 69.53   | 71.59   |
|          | Ours           | 82.60          | 71.21       | 78.40   | 81.95           | 77.71    | 79.81   | 82.39                 | 74.15    | 65.98    | 66.30       | 67.15    | 71.45   | 74.89   |
|          |                | (+4.43)        | (+4.04)     | (+4.41) | (+8.43)         | (+1.95)  | (+5.41) | (-0.10)               | (+4.27)  | (+1.74)  | (+1.63)     | (+1.33)  | (+1.92) | (+3.30) |

TABLE II

<span id="page-6-1"></span>FINE-GRAINED TESTING PERFORMANCE(%) OF QA METHODS, METHOD+HAVF, AND OUR MODEL ON AVQA [\[14\]](#page-9-13) DATASET. THE BEST AND SECOND BEST PERFORMANCE OVER EACH QUESTION TYPE ARE HIGHLIGHTED IN BOLD FORM AND UNDERLINE FORM, RESPECTIVELY

| Methods                  | Which              | Come From          | Happening          | Where              | Why                | Before Next        | When               | Used For           | Total Accuracy     |
|--------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| ACRTransformer [33]      | 82.5               | 82.8               | 79.4               | 82.5               | 54.7               | 80.0               | 47.6               | 58.8               | 81.7               |
| ACRTransformer+HAVF [14] | 88.5               | 91.7               | 83.9               | <u>84.9</u>        | 50.0               | 82.0               | 57.1               | 64.7               | 87.8               |
| HGA [34]                 | 82.1               | 84.3               | 79.5               | 83.1               | 59.3               | 82.0               | 57.1               | 88.2               | 82.2               |
| HGA+HAVF [14]            | 88.6               | 92.2               | 83.8               | 82.6               | <u>61.6</u>        | 78.0               | <u>52.4</u>        | 82.4               | 87.7               |
| HCRN [31]                | 83.7               | 84.1               | 80.2               | 80.9               | 52.3               | 74.0               | 57.1               | 70.6               | 82.5               |
| HCRN+HAVF [14]           | <u>89.8</u>        | <u>92.8</u>        | <u>86.0</u>        | 84.4               | 57.0               | 80.0               | 52.4               | 82.4               | <u>89.0</u>        |
| Ours                     | <b>90.2</b> (+0.4) | <b>93.0</b> (+0.2) | <b>88.4</b> (+2.4) | <b>85.8</b> (+0.9) | <b>68.3</b> (+6.7) | <b>84.5</b> (+2.5) | <b>60.2</b> (+3.1) | <b>90.3</b> (+2.1) | <b>92.1</b> (+3.1) |

our proposed model achieves better performance than other multimodal QA methods on most subtasks, with significant improvements observed for all audio and visual questions. Notably, our method achieves similar accuracy (82.49%) as the highest-performing method on *Existential* questions for audio-visual questions, while surpassing other methods on the remaining question types, particularly on *Location* and *Comparative* questions. The results consistently demonstrate the effectiveness of our method in audio-visual scenes and its ability to provide more accurate responses to questions.

- *2) Results on AVQA Dataset:* In this section, we report the evaluation results of our model and other counterparts on the AVQA [\[14\]](#page-9-13) dataset. The results except the last row (Ours) in Tab. [II](#page-6-1) are all imported from [\[14\]. A](#page-9-13)s shown in Tab. [II,](#page-6-1) our approach yields substantial improvements in QA accuracy across all question types when compared to multiple QA approaches. Specifically, our model achieve significant increase (+6.7%) on *Why* question type. In addition, the accuracy improvements of *Which* (+0.4%) and *Come From* (+0.2%) are relatively small. We think this is because the difficulty of these two question types is relatively low, so most of the QA methods can achieve good accuracy on them. These results underscore the effectiveness of our question-aware QA model in accurately inferring answers in audio-visual settings.
- *3) Ablation Study:* In this part, ablation studies are conducted to gain a deeper understanding of our model and to validate the indispensability and efficacy of each constituent within our framework based on MUSIC-AVQA [\[13\]](#page-9-12) dataset.

The outcomes of this study are presented in Tab. [III.](#page-7-0) Our model, as previously detailed, is composed of two stages: (i) the *First Stage*, which comprehends the video content through a combination of global and local perspectives, and (ii) the *Second Stage*, which includes a fusion module that is question-oriented, global and local feature-based, and an answer prediction block. We undertake partial ablation of crucial components within the model architecture, followed by retraining and validation procedures to ascertain the significance of these elements in shaping the overall model performance.

The first issue that we aim to investigate is what extent to which the features derived from the global and local branches can aid in the comprehension of audio-visual scenes by a model. We begin by conducting an ablation study wherein we replace the inputs of the *Second Stage* module, namely the global and local inputs, with the outputs from the audio and visual channels in the global branch after removing the local branch. The experimental results, depicted in the first row of Tab. [III](#page-7-0) under the label "Without Local," reveal that the absence of local features leads to a considerable decrease on averaged accuracy (−1.4%). Furthermore, we repeat the above-mentioned procedure by removing the global branch and present the outcomes in the second row of Tab. [III](#page-7-0) labeled as "Without Global." It can be clearly observed that the averaged accuracy drops by 1.21% after eliminating the global branch. Our findings demonstrate that both global and local features contribute significantly to the performance of the model. Specifically, the performance drops more significantly

| Methods                 | Audio Question       | Visual Question | A-V question  | All           |
|-------------------------|----------------------|-----------------|---------------|---------------|
| Without Local           | 78.09 (-0.31)        | 78.41 (-1.4)    | 69.70 (-1.75) | 73.49 (-1.4)  |
| Without Global          | 78.77 (+0.37)        | 79.73 (-0.08)   | 69.19 (-2.26) | 73.68 (-1.21) |
| Without NCE             | <b>78.83</b> (+0.43) | 77.62 (-2.19)   | 69.31 (-2.14) | 73.20 (-1.69) |
| Without Cross-attention | 78.46 (+0.06)        | 78.28 (-1.53)   | 70.23 (-1.22) | 73.82 (-1.07) |
| Dummy Question          | 76.78 (-1.62)        | 77.21 (-2.6)    | 68.94 (-2.51) | 72.52 (-2.37) |
| Full Version (Ours)     | 78.40                | 79.81           | 71.45         | 74.89         |

<span id="page-7-0"></span>TABLE III ABLATION STUDY ON THE CONTRIBUTION FROM THE KEY COMPONENTS OF OUR PROPOSED MODEL ON MUSIC-AVQA [\[13\]](#page-9-12) DATASET

<span id="page-7-1"></span>![](_page_7_Picture_4.jpeg)

Fig. 3. The cross-modality attention in the model is removed to investigate the contribution of this mechanism to our model. The left is the cross-attention module in the proposed model, and the right is the cross-attention removal module.

when the local branch is removed, and the A-V question types demonstrate a greater reliance on the global features.

The second issue that we investigate is the significance of audio-visual feature alignment for the precise derivation of video representations. To accomplish this, we eliminate the contrastive learning module in the global and local branches, as well as the InfoNCE loss L*glb* and L*loc* in Eq. [\(14\).](#page-4-0) Our findings, presented in the third row of Tab. [III,](#page-7-0) show that the averaged accuracy significantly drops by 1.69%, revealing that aligning audio-visual representations both in global and local branches can considerably enhance the performance of the model. Although removing InfoNCE loss leads to a slight accuracy increase (+0.43%) of the audio questions.

Thirdly, we examine the cruciality of the cross-modality attention mechanism in the proposed model. In this regard, we perform an ablation study by removing the cross-attention operation (as seen in Fig. [3\)](#page-7-1) and report the corresponding results in the fourth row of Tab. [III.](#page-7-0) The results illustrate that the cross-attention mechanism provides significant benefits to both visual question and audio-visual question types, although its impact on the audio question type is comparatively lower.

Furthermore, we investigate the significance of leveraging the provided question as a guide in the Second Stage of our model. To explore this, we replace the given question with a dummy question that does not contain video information. We set all the elements of this dummy question to a uniform value of "1" and see what happens. The fifth row in Tab. [III](#page-7-0) presents the experimental results, demonstrating a substantial decline in model performance when the provided question is not used as a guide any more. Specifically, the accuracy of all question types decrease significantly, with the averaged accuracy dropping by 2.37%. These results suggest that employing the given question facilitates the fusion of

<span id="page-7-2"></span>TABLE IV VARYING THE SCALING FACTORS λ*glb* AND λ*loc* IN EQ. [\(14\)](#page-4-0)

| $\lambda_{glb}$ | $\lambda_{loc}$ | Audio Question | Visual Question | A-V question | All   |
|-----------------|-----------------|----------------|-----------------|--------------|-------|
|                 | 0.01            | 78.34          | 78.45           | 69.88        | 73.64 |
|                 | 0.1             | 78.40          | 79.81           | 71.45        | 74.89 |
|                 | 0.2             | 78.09          | 77.99           | 70.31        | 73.72 |
| 0.1             | 0.3             | 78.09          | 78.57           | 70.29        | 73.86 |
|                 | 0.4             | 78.77          | 78.86           | 69.96        | 73.87 |
|                 | 0.5             | 78.03          | 78.57           | 70.47        | 73.95 |
|                 | 0.7             | 77.90          | 78.94           | 70.41        | 73.99 |
| 0.01            |                 | 78.65          | 77.46           | 70.47        | 73.76 |
| 0.1             |                 | 78.40          | 79.81           | 71.45        | 74.89 |
| 0.2             |                 | 77.72          | 78.12           | 70.11        | 73.58 |
| 0.3             | 0.1             | 78.65          | 78.36           | 70.21        | 73.86 |
| 0.4             |                 | 77.53          | 78.61           | 70.15        | 73.70 |
| 0.5             |                 | 78.46          | 77.58           | 70.39        | 73.72 |
| 0.7             |                 | 77.09          | 78.16           | 69.66        | 73.23 |

global and local features, and ultimately improves the answer generation process.

- *4) Hyperparameter Analysis:* In this part, we investigate the impact of various loss functions on our proposed method by manipulating the scaling factors λ*glb* and λ*loc* in Eq. [\(14\).](#page-4-0) Firstly, we fix the λ*glb* to 0.1 and adjust the λ*loc* from 0.01 to 0.7. Furthermore, we fix the λ*loc* to 0.1 and adjust the λ*glb* from 0.01 to 0.7. Our findings, shown in Tab. [IV,](#page-7-2) demonstrate that the performance of our model is not sensitive to these two hyperparameters. It is worth noting that we always set the scaling factor of the first term (L*qa*) in Eq. [\(14\)](#page-4-0) to 1 as the answer prediction accuracy is the key driver of model performance, whereas the impact of contrastive learning on model training is relatively minor.
- *5) Visualized Analysis of Local Branch:* Through our ablation study in Tab. [III,](#page-7-0) we show that the local branch significantly improves the model performance for most question types. We visualize the attention map of the visual channel in the local branch in Fig. [4](#page-8-0) to explore the correlation between the question words and the sampled video frames.

For example, as shown in Fig. [4 \(a\),](#page-8-0) we employ a video with two people playing different instruments and a questionanswer pair, i.e., question: *"Is the instrument on the left more rhythmic than the instrument on the right?"* answer: *"Yes"*. The heatmap illustrates varying attentional intensities between the question words and video frames, with visual features of the 31st and 37th timestamps demonstrating a stronger correlation with the question. Furthermore, specific words such as *"instrument"*, *"left"*, *"more"*, *"rhythmic"*, and *"than"*

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

Fig. 4. The attention map visualization of the video channel in the local branch. The color of the image reflects the attentional intensities between question words and video frames. The greater attentional intensity, the stronger correlation. The red color means stronger intensities and the blue color represents the weaker ones.

<span id="page-8-1"></span>![](_page_8_Figure_4.jpeg)

Fig. 5. Prediction results of our model for different questions types in the MUSIC-AVQA [\[13\]](#page-9-12) dataset. The figure provides the video frames, audio waveforms, questions (indicated by "Q"), the ground-truth answers (indicated by "A"), and the top-3 answers predicted by our model (indicated by "Top-3").

activate more visual features, indicating their crucial role in feature extraction from visual signals. Moreover, by listen to the original audio of this video, we find that the audio from the 20th to 53rd seconds clearly highlights that the violin playing on the left is more rhythmic than the accordion playing on the right. This is consistent in time with the attentional intensities shown by the attention map.

In addition, in Fig. [4 \(b\),](#page-8-0) the image captured at the 49th second, wherein no instrument is present, exhibits a weaker correlation with any of the words present in the question. Moreover, in Fig. [\(e\),](#page-8-0) the images with the accordion present show stronger association with the question.

Overall, our findings suggest that the local branch is an effective module for improving the performance of the model.

*6) Prediction Results of Our Model:* In this part, we present the predictive analysis outcomes of our model on each question type of the MUSIC-AVQA [\[13\]](#page-9-12) dataset. The results, depicted in Fig. [5,](#page-8-1) include the question, the ground-truth answer, and the top-3 model-generated results for each question type. Our findings reveal that our model adeptly deduces answers for diverse question types. Despite some instances where the correct answers do not obtain the highest probability, as evidenced by examples (e) and (f) depicted in Fig. [5,](#page-8-1) our model effectively identifies the correct answer and ranks it among the top-2 answers, thus showcasing the potential efficacy of the proposed methodology.

## V. CONCLUSION

In this paper, we investigated the AVQA task and identified the main challenges and requirements. To address these issues, we propose a novel framework for multimodal question answering, which provides a comprehensive understanding of audio-visual scenarios at both global and local levels. Our framework considers the importance of the given question in an AVQA task, which guides the feature extraction of audio and visual signals and the final fusion between global and local features. Additionally, our method sufficiently correlates and fuses multimodal data. Experimental results demonstrate the effectiveness and superiority of our design.

## *A. Limitation*

Our framework currently assumes the availability of all modalities at all times, which may not be the case in some scenarios. As such, our framework can be further improved to handle missing modality or information in the future. Furthermore, following the classic question answering setting in the literature, our proposed method constitutes a classification framework that primarily operates within the constraints of a close-set answer setting, restricting it from accommodating the diverse range of potential answers. As part of our future research endeavors, we aim to extend this framework towards the generative paradigm, thereby enabling us to effectively address the complexities inherent in more scenarios. In addition, given the extensive applications of AVQA in information retrieval, human-computer interaction, and visual/auditory assistance, we should better consider issues of fairness, transparency, and explainability within the QA framework to ensure its safe and responsible use.

# ACKNOWLEDGMENT

The authors would like to thank Guangyao Li for his helpful discussion about the MUSIC-AVQA dataset during the course of this work.

## REFERENCES

- <span id="page-9-0"></span>[\[1\] S](#page-0-0). Antol et al., "VQA: Visual question answering," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Dec. 2015, pp. 2425–2433.
- <span id="page-9-1"></span>[\[2\] T](#page-0-0). Yu, J. Yu, Z. Yu, Q. Huang, and Q. Tian, "Long-term video question answering via multimodal hierarchical memory attentive networks," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 31, no. 3, pp. 931–944, Mar. 2021.
- <span id="page-9-2"></span>[\[3\] H](#page-0-0).-T. Su et al., "End-to-end video question-answer generation with generator-pretester network," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 31, no. 11, pp. 4497–4507, Nov. 2021.
- <span id="page-9-3"></span>[\[4\] L](#page-0-0). Zhao et al., "Toward explainable 3D grounded visual question answering: A new benchmark and strong baseline," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 33, no. 6, pp. 2935–2949, Jun. 2023.
- <span id="page-9-4"></span>[\[5\] F](#page-0-0). Zhang, R. Wang, F. Zhou, and Y. Luo, "ERM: Energy-based refined-attention mechanism for video question answering," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 33, no. 3, pp. 1454–1467, Mar. 2023.

- <span id="page-9-5"></span>[\[6\] L](#page-0-0). Li et al., "Multi-granularity relational attention network for audiovisual question answering," *IEEE Trans. Circuits Syst. Video Technol.*, early access, Apr. 5, 2023, doi: [10.1109/TCSVT.2023.3264524.](http://dx.doi.org/10.1109/TCSVT.2023.3264524)
- <span id="page-9-6"></span>[\[7\] H](#page-0-1). M. Fayek and J. Johnson, "Temporal reasoning via audio question answering," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 28, pp. 2283–2294, 2020.
- <span id="page-9-7"></span>[\[8\] G](#page-0-1). Li, Y. Xu, and D. Hu, "Multi-scale attention for audio question answering," 2023, *arXiv:2305.17993*.
- <span id="page-9-8"></span>[\[9\] H](#page-0-2). Alamri, C. Hori, T. K. Marks, D. Batra, and D. Parikh, "Audio Visual Scene-aware Dialog (AVSD) track for natural language generation in DSTC7," in *Proc. AAAI Workshop*, vol. 2, 2018, pp. 1–6.
- <span id="page-9-9"></span>[\[10\]](#page-0-2) S. Kim et al., "Overview of the eighth dialog system technology challenge: DSTC8," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 29, pp. 2529–2540, 2021.
- <span id="page-9-10"></span>[\[11\]](#page-0-2) A. Shah et al., "Audio-visual scene-aware dialog and reasoning using audio-visual transformers with joint student-teacher learning," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, May 2022, pp. 7732–7736.
- <span id="page-9-11"></span>[\[12\]](#page-0-3) H. Yun, Y. Yu, W. Yang, K. Lee, and G. Kim, "Pano-AVQA: Grounded audio-visual question answering on 360◦ videos," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Oct. 2021, pp. 2031–2041.
- <span id="page-9-12"></span>[\[13\]](#page-0-3) G. Li, Y. Wei, Y. Tian, C. Xu, J.-R. Wen, and D. Hu, "Learning to answer questions in dynamic audio-visual scenarios," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2022, pp. 19086–19096.
- <span id="page-9-13"></span>[\[14\]](#page-0-3) P. Yang et al., "AVQA: A dataset for audio-visual question answering on videos," in *Proc. 30th ACM Int. Conf. Multimedia*, Oct. 2022, pp. 3480–3491.
- <span id="page-9-14"></span>[\[15\]](#page-0-4) Y.-B. Lin, Y.-L. Sung, J. Lei, M. Bansal, and G. Bertasius, "Vision transformers are parameter-efficient audio-visual learners," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2023, pp. 2299–2309.
- <span id="page-9-15"></span>[\[16\]](#page-1-1) S. Chen et al., "VALOR: Vision-Audio-Language Omni-peRception pretraining model and dataset," 2023, *arXiv:2304.08345*.
- <span id="page-9-16"></span>[\[17\]](#page-1-2) F. Xiao, Y. J. Lee, K. Grauman, J. Malik, and C. Feichtenhofer, "Audiovisual SlowFast networks for video recognition," 2020, *arXiv:2001.08740*.
- <span id="page-9-17"></span>[\[18\]](#page-1-3) A. Nagrani, S. Yang, A. Arnab, A. Jansen, C. Schmid, and C. Sun, "Attention bottlenecks for multimodal fusion," in *Proc. Adv. Neural Inf. Process. Syst.*, vol. 34, 2021, pp. 14200–14213.
- <span id="page-9-18"></span>[\[19\]](#page-2-2) N. Shvetsova et al., "Everything at once—Multi-modal fusion transformer for video retrieval," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2022, pp. 19988–19997.
- <span id="page-9-19"></span>[\[20\]](#page-2-3) R. Arandjelovic and A. Zisserman, "Look, listen and learn," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Oct. 2017, pp. 609–617.
- <span id="page-9-20"></span>[\[21\]](#page-2-4) A. Rouditchenko et al., "AVLnet: Learning audio-visual language representations from instructional videos," 2020, *arXiv:2006.09199*.
- <span id="page-9-21"></span>[\[22\]](#page-3-0) J. F. Gemmeke et al., "Audio set: An ontology and human-labeled dataset for audio events," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, Mar. 2017, pp. 776–780.
- <span id="page-9-22"></span>[\[23\]](#page-3-1) D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, "A closer look at spatiotemporal convolutions for action recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, Jun. 2018, pp. 6450–6459.
- <span id="page-9-23"></span>[\[24\]](#page-3-2) N. Xu, W. Mao, and G. Chen, "Multi-interactive memory network for aspect based multimodal sentiment analysis," in *Proc. AAAI Conf. Artif. Intell.*, 2019, vol. 33, no. 1, pp. 371–378.
- <span id="page-9-24"></span>[\[25\]](#page-3-2) G. P. Rajasekar et al., "A joint cross-attention model for audio-visual fusion in dimensional emotion recognition," 2022, *arXiv:2203.14779*.
- <span id="page-9-25"></span>[\[26\]](#page-4-1) A. van den Oord, Y. Li, and O. Vinyals, "Representation learning with contrastive predictive coding," 2018, *arXiv:1807.03748*.
- [\[27\]](#page-0-5) J. Lu, J. Yang, D. Batra, and D. Parikh, "Hierarchical question-image co-attention for visual question answering," in *Proc. Adv. Neural Inf. Process. Syst.*, vol. 29, 2016, pp. 1–9.
- <span id="page-9-26"></span>[\[28\]](#page-0-5) Z. Yu, J. Yu, Y. Cui, D. Tao, and Q. Tian, "Deep modular co-attention networks for visual question answering," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2019, pp. 6274–6283.
- [\[29\]](#page-0-5) X. Li et al., "Beyond RNNs: Positional self-attention with co-attention for video question answering," in *Proc. AAAI Conf. Artif. Intell.*, 2019, vol. 33, no. 1, pp. 8658–8665.
- [\[30\]](#page-0-5) C. Fan, X. Zhang, S. Zhang, W. Wang, C. Zhang, and H. Huang, "Heterogeneous memory enhanced multimodal attention model for video question answering," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2019, pp. 1999–2007.
- <span id="page-9-27"></span>[\[31\]](#page-0-5) T. M. Le, V. Le, S. Venkatesh, and T. Tran, "Hierarchical conditional relation networks for video question answering," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2020, pp. 9969–9978.

- <span id="page-10-2"></span>[\[32\]](#page-0-5) I. Schwartz, A. G. Schwing, and T. Hazan, "A simple baseline for audio-visual scene-aware dialog," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2019, pp. 12540–12550.
- <span id="page-10-0"></span>[\[33\]](#page-0-5) J. Zhang, J. Shao, R. Cao, L. Gao, X. Xu, and H. T. Shen, "Action-centric relation transformer network for video question answering," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 32, no. 1, pp. 63–74, Jan. 2022.
- <span id="page-10-1"></span>[\[34\]](#page-0-5) P. Jiang and Y. Han, "Reasoning with heterogeneous graph alignment for video question answering," in *Proc. AAAI Conf. Artif. Intell.*, Apr. 2020, vol. 34, no. 7, pp. 11109–11116.

![](_page_10_Picture_5.jpeg)

Zailong Chen received the B.S. degree from the Nanjing University of Posts and Telecommunications (NUPT), China, in 2015, and the M.E. degree from Hunan University, China, in 2022. He is currently pursuing the Ph.D. degree with the School of Computing and Information Technology, University of Wollongong, Australia. His research interests include deep learning, computer vision, and human–computer interaction.

![](_page_10_Picture_7.jpeg)

Lei Wang (Senior Member, IEEE) received the Ph.D. degree from Nanyang Technological University, Singapore. He is currently a Professor with the School of Computing and Information Technology, University of Wollongong, Australia. He has published more than 190 peer-reviewed papers, including those in highly regarded journals and conferences, such as IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLI-GENCE, *International Journal of Computer Vision*, CVPR, ICCV, and ECCV. His research interests

include machine learning, pattern recognition, and computer vision. He was awarded the Early Career Researcher Award by the Australian Academy of Science and Australian Research Council. He served as the General Co-Chair of DICTA 2014 and the Program Co-Chair of ACCV 2022. He served on the technical program committees for many international conferences and workshops.

![](_page_10_Picture_10.jpeg)

Peng Wang received the Ph.D. degree from the School of Information Technology and Electrical Engineering, The University of Queensland. He is a Professor with the School of Computer Science and Engineering, University of Electronic Science and Technology of China. Prior to joining UESTC, he was a Lecturer with the School of Computing and Information Technology, University of Wollongong. His major research interest lies in computer vision and deep learning, with special interest in dataefficient deep learning.

![](_page_10_Picture_12.jpeg)

Peng Gao received the B.S. degree from the Beijing University of Posts and Telecommunications (BUPT), China, in 2002, and the M.A. degree from the University of Bristol (UoB), U.K., in 2022. He is currently pursuing the Ph.D. degree with the Institute of Computer Science, Beijing Normal University–Hong Kong Baptist University United International College, Zhuhai, China. His research interests include deep learning, computer vision, and person recognition.