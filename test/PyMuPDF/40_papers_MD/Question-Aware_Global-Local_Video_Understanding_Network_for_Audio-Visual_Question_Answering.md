IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 5, MAY 2024 4109

## Question-Aware Global-Local Video Understanding Network for Audio-Visual Question Answering


[Zailong Chen, Lei Wang,](https://orcid.org/0009-0003-8431-5471) _Senior Member, IEEE_ [, Peng Wang, and Peng Gao](https://orcid.org/0000-0002-5397-9115)



_**Abstract**_ **— As a newly emerging task, audio-visual question**
**answering (AVQA) has attracted research attention. Compared**
**with traditional single-modality (e.g., audio or visual) QA tasks,**
**it poses new challenges due to the higher complexity of feature**
**extraction and fusion brought by the multimodal inputs. First,**
**AVQA requires more comprehensive understanding of the scene**
**which involves both audio and visual information; Second, in the**
**presence of more information, feature extraction has to be better**
**connected with a given question; Third, features from different**
**modalities need to be sufficiently correlated and fused. To address**
**this situation, this work proposes a novel framework for multi-**
**modal question answering task. It characterises an audiovisual**
**scene at both global and local levels, and within each level, the**
**features from different modalities are well fused. Furthermore,**
**the given question is utilised to guide not only the feature**
**extraction at the local level but also the final fusion of global and**
**local features to predict the answer. Our framework provides a**
**new perspective for audio-visual scene understanding through**
**focusing on both general and specific representations as well**
**as aggregating multimodalities by prioritizing question-related**
**information. As experimentally demonstrated, our method sig-**
**nificantly improves the existing audio-visual question answering**
**performance, with the averaged absolute gain of 3.3% and 3.1%**
**on MUSIC-AVQA and AVQA datasets, respectively. Moreover,**
**the ablation study verifies the necessity and effectiveness of our**
**design. Our code will be publicly released.**


_**Index Terms**_ **— Audio-visual question answering, video under-**
**standing, multimodal learning, deep learning.**


I. I NTRODUCTION
# I N RECENT times, question answering has garnered sig-nificant attention and has exhibited its potential for various

applications such as information retrieval, human-computer
interaction, and visual/auditory assistance. Notably, considerable advancements have been made in the domain of visual

question answering (VQA) [1], [2], [3], [4], [5], [6], and
audio question answering (AQA) [7], [8]. While VQA and


Manuscript received 23 April 2023; revised 13 August 2023; accepted
10 September 2023. Date of publication 2 October 2023; date of current
version 9 May 2024. This article was recommended by Associate Editor
C. Herglotz. _(Corresponding author: Lei Wang.)_
Zailong Chen and Lei Wang are with the School of Computing and
Information Technology, University of Wollongong, Wollongong, NSW 2522,
Australia (e-mail: zc881@uowmail.edu.au; leiw@uow.edu.au).
Peng Wang is with the School of Computer Science and Engineering,
University of Electronic Science and Technology of China, Chengdu 610056,
China (e-mail: p.wang6@hotmail.com).
Peng Gao is with the Institute of Computer Science, Beijing Normal
University–Hong Kong Baptist University United International College,
Zhuhai 519000, China (e-mail: s230201705@mail.uic.edu.cn).
Color versions of one or more figures in this article are available at
https://doi.org/10.1109/TCSVT.2023.3318220.
Digital Object Identifier 10.1109/TCSVT.2023.3318220



AQA concentrate on comprehending the signal of a single
modality for generating answers to the given question, the
past few years have observed a growing interest in tackling
more pervasive and intricate audio-visual scenarios, such as
audio-visual scene-aware dialog (AVSD) [9], [10], [11] and
the more recent audio-visual question answering (AVQA) [12],

[13], [14] tasks. The availability of AVQA datasets based on
audio-visual scenes has substantially broadened the application
scope in the multimodal domain. For instance, the expeditious
extraction of pertinent information from surveillance videos is
achievable through the analysis of visual and auditory data,
coupled with targeted query questions, thereby circumventing
the arduous process of frame-by-frame video observation. This
methodology exhibits promising potential in alleviating the
temporal and labor-intensive burdens associated with managing extensive video recordings, thereby facilitating expeditious
and effective acquisition of pivotal and contextually relevant
details by law enforcement agencies.
The comprehension of audio-visual scenes represents a crucial prerequisite for AVQA systems. Extensive prior research
has underscored the importance of global audio and visual
features in facilitating scene understanding within AVQA [12],

[13], [14]. However, it is argued that relying solely on such
global features may not be sufficient for achieving a comprehensive understanding of scenes. Instead, it is advantageous to
integrate task-specific local features, e.g., focusing on critical
moments relevant to the posed question [2] and integrating
multi-granularity information from the video [6]. To illustrate, Fig. 1 depicts a scenario wherein diverse video content
necessitates different attention to answer distinct questions,
and even individual words within a question hold varying
degrees of importance. Hence, a holistic comprehension of
video content cannot be attained solely through a global
or local perspective in isolation. Instead, a comprehensive
understanding requires absorbing both global and task-relevant
local features. Furthermore, questions should act as guiding
cues to better identify key audio or visual information within
the scene.

Current AVQA methods primarily extract audio and visual
features globally, without considering the importance of local
feature extraction [12], [13], [14], [15]. Consequently, subtle
and task-relevant information may be overlooked. Additionally, these methods do not thoroughly investigate the
significance of question for feature extraction or fusion, potentially impeding the ability to effectively answer questions.
Given the complementary nature of modalities in AVQA, adequate correlation between different modalities during feature



1051-8215 © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.


Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


4110 IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 5, MAY 2024


to enhance the comprehension, extraction, and fusion of multimodal information.

2. Our approach surpasses prior works by characterizing
video content with both global and local features that are
critical for AVQA. This innovative design enables our model
to comprehensively interpret the audio-visual scenes.
3. Our framework achieves promising results by integrating
the question into the AVQA task, thereby providing guidance
for feature extraction and fusion. Moreover, we emphasize
sufficient correlation and fusion of global and local features
within and between them.

An experimental study is carried out on the recently
introduced MUSIC-AVQA [13] and AVQA [14] datasets,
employing various questions types and modalities. The outcomes corroborate the efficacy and benefits of the proposed
approach over the existing related methods, and the ablation
study substantiates the indispensability and significance of our
design.


II. R ELATED W ORK


_A. AVQA_



Fig. 1. This figure is to illustrate the AVQA problem. In the context of the
audio-visual question answering task, the areas and elements that necessitate
focused attention vary with the specific question presented, encompassing
temporal periods, visual regions, and acoustic zones. To facilitate the identification of these areas of interest for distinct questions within the same video,
we have incorporated green trumpet icons, yellow highlights, and red dashed
boxes into the accompanying figure.


extraction or fusion is essential for scene comprehension, as is
done in [16].
We propose a new framework for AVQA to address the
current situation. Our approach aims to extract both global
and local video features and to fuse multimodal information

in a question-aware manner to generate an accurate answer.
The proposed framework consists of two stages, as illustrated
in Fig. 2. In the first stage, the model extracts global and
local features from the audio and visual data. The global
features are extracted using the co-attention mechanism, which
correlates global audio and visual features. The local features are obtained by identifying important local information
from audio and visual inputs based on the given question.
Additionally, the extracted audio and visual features are fused
to ensure that they are integrated with each other. In the
second stage, the question is used again as a guide to further
refine the fusion of global and local features before they are
decoded into the answer of the question by a classifier. Our
framework is fully “question-aware” as the question is used
to guide both feature extraction and fusion. Our approach
not only provides global features to attain general, questionindependent information for overall video understanding, but
also leverages local features to highlight information critical
to deducing the correct answer.
Our contributions are summarised as follows:

1. The present study investigates the emerging task of
AVQA and delineates its principal challenges and requirements. To address these issues and to attain more precise
question answering, we propose a novel framework that aims



AVQA is a nascent task that has garnered considerable
attention from the deep learning community due to the remarkable achievements of deep neural networks in the realm of
multimodal research. The AVQA mission requires comprehensive understanding and integration of diverse modalities,
leading to precise responses to distinct questions. AVSD,
akin to the AVQA task, focuses on human-to-human dialogue
scenarios, presenting textual captions or dialogs in conjunction
with audio and visual cues. In contrast to AVSD task, which
draws upon audio, visual, and dialog text from videos to
facilitate comprehensive scene understanding, the AVQA task
allows for the understanding of video content solely based on
audio and visual information. The initial AVQA dataset, PanoAVQA [12], was introduced to explore audio-visual question
answering in panoramic videos, encompassing only two question types, namely _existential_ and _location_ . In order to tackle
more complex question types and reason audiovisual scenarios
from multiple perspectives, MUSIC-AVQA [13] was unveiled,
which is a large-scale dataset comprising real-life audiovisual
scenarios (concerts). It embraces five questions (i.e., _counting_,
_comparative_, _location_, _existential_, and _temporal_ ) and spans
over nine question types by merging these five question
aspects with different modalities. Recently, a real-life scenebased AVQA benchmark was proposed in [14], which further
expands the audiovisual scene coverage of AVQA tasks. In the
classic question answering setting in the literature [1], [10],
objective of the agent entails selecting the appropriate answer
from a pool of predefined potential responses, aligning with
the query question and specific input modalities. This setting
also governs the two aforementioned AVQA tasks.


_B. Deep Audio-Visual Learning_


Several audio-visual learning methods have been proposed
in recent years. Reference [17] presented an audiovisual slowfast network for modeling multimodal concepts hierarchically.
Reference [18] proposed a multimodal bottleneck attention



Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


CHEN et al.: QUESTION-AWARE GLOBAL-LOCAL VIDEO UNDERSTANDING NETWORK FOR AVQA 4111


Fig. 2. The proposed AVQA model. Our model can be divided into two stages. The first stage is to understand video contents at both global (general) and
local (specific) levels. Furthermore, the second stage integrates a question-aware fusion module of global and local features and an answer classifier. More
detailed description is provided in Section III.



method for efficiently fusing audio and visual inputs. Reference [19] developed an architecture that projects multimodal
inputs into a joint multimodal embedding space using a combinatorial loss during training. Contrastive learning was utilized
in [20] and [21] to obtain robust audio and visual encoders and
an audiovisual embedding space, respectively. These methods
offer diverse perspectives on how to merge multimodalities.
Reference [12] proposed a combinatorial attention paradigm
that combines audio, visual, and textual features for specific
audio-visual question-answering tasks. Additionally, [13] utilized audio features to query visual features for spatial video
comprehension, followed by an attention module that identifies
temporal information in the audio and visual features using
the question as a clue. Reference [14] introduced a hierarchical audio-visual fusing method that can be combined with
existing bi-modal fusion methods to aggregate multimodal
features.

Existing methods primarily analyze audio-visual scenes
from a global viewpoint, thus, ignoring the significance
of the task-relevant local information of the scene in the

AVQA task. The present study aims to address the aforementioned limitations by extracting video features from both
global and local levels and emphasizing the role of the
question in AVQA task. Moreover, our proposed methodology extensively correlates and fuses data from different
modalities.



III. P ROPOSED M ETHOD


This section introduces our proposed model in detail. The
overall architecture of the model is shown in Fig. 2.


_A. Workflow of the Framework_


The inputs of the model consist of audio and visual signals
sampled from a video, along with a question based on the
video. The model is then tasked with accurately answering
the given question based on multimodal inputs. The proposed model consists of the following key components: (1)
separate encoders extract features of the three modalities of
audio, visual, and text; (2) The first stage of the model then
refines and integrates these multimodal features, effectively
identifying general and question-specific features of the audio
and visual signals through global and local branches; (3)
contrastive learning is used atop the global and local branches
to ensure alignment between audio and visual features; (4)
the global and local features flow to the second stage simultaneously for final fusion, during which they are sufficiently
correlated and aggregated under the guidance of the question;
(5) the final fused feature is fed into a classifier to determine
an appropriate answer.
The aim of the presented framework is to enhance the
optimization of an objective function for attaining the better audiovisual feature representation. The objective function



Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


4112 IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 5, MAY 2024



includes the alignment of global-level audio-visual features
and local-level audio-visual features, in addition to ensuring
the precision of question answering. Details are shown below.


_B. Video Understanding_


_1) Feature Extraction:_ Given an input video with both
visual and audio tracks, we sample a fixed number _(T )_
of segments from it, and this gives _T_ positive audio-visual
_T_
pairs denoted by � _A_ _t_ _, V_ _t_ [+] � _t_ =1 [. Also, we randomly sample]
a visual sequence from another video and compose negative
_T_
audio-visual pairs � _A_ _t_ _, V_ _t_ [−] � _t_ =1 [. These positive and negative]
audio-visual pairs will be employed to align the audio-visual
features coming from the same video through contrastive
learning. The question text is tokenized into _K_ individual
tokens { _Q_ _k_ } _k_ _[K]_ =1 [.]
Here, we extract the audio feature vectors { **a** _t_ } _t_ _[T]_ =1 [, positive]
visual feature vectors { **v** [+] _t_ [}] _t_ _[T]_ =1 [, negative visual feature vectors]
{ **v** [−] _t_ [}] _t_ _[T]_ =1 [, and question feature vectors][ {] **[q]** _[k]_ [}] _k_ _[K]_ =1 [from][ {] _[A]_ _[t]_ [}] _t_ _[T]_ =1 [,]
{ _V_ _t_ [+] [}] _t_ _[T]_ =1 [,][ {] _[V]_ _t_ [ −] [}] _t_ _[T]_ =1 [, and][ {] _[Q]_ _[k]_ [}] _k_ _[K]_ =1 [, respectively. Specifically,]
we adopt the commonly used networks to extract the features
from different modalities, i.e., VGGish [22] for audio feature
extraction, ResNet-18 [23] for visual feature extraction, and
LSTM for textual input. Thus, we obtain audio feature matrix
**A**, positive visual feature matrix **V** [+],negative visual feature
matrix **V** [−], and question feature matrix **Q** .
_2) Multimodal Fusion and Interaction:_ When presented
with complex, multimodal data, it is imperative to consider
the intricate interactions that exist both within and between the

various modalities. In order to accomplish this, we leverage
the extensively utilized cross-attention mechanism [5], [24],

[25], specifically implementing co-attention as illustrated in
Fig. 2. We provide a thorough exposition of this module as it
plays a significant role in our model.
Given two input modalities (i.e., left-input modality **M** _l_ and
right-input modality **M** _r_ ), co-attention module models their
interaction by two channels. Before modeling the interaction
information of two modalities, we employ self-attention to
further capture their long-range interdependent features.


**F** _l_ = Self-att _(_ **M** _l_ _,_ **M** _l_ _)_ ; **F** _r_ = Self-att _(_ **M** _r_ _,_ **M** _r_ _),_ (1)


where **F** _l_ and **F** _r_ are the features of left and right channels after
processing by the self-attention. Self-att is the self-attention
operation.
To learn the self-influence caused by the left-input modality
itself, the left channel feature **F** _l_ is used to query itself,
generating the self-modality attentional feature. Simultaneously, to capture the interactive influence that the right-input
modality brings to the left one, the right channel feature
**F** _r_ is applied to query **F** _l_ to generate the cross-modality
attentional feature. These two features are then averaged and
concatenated with **F** _l_, and the resulting concatenated feature
is fed into a Feedforward Neural Network (FFN) to obtain the
left attentional feature **F** _L_ thereby enabling interaction between
the left and right modalities. The operation can be formulated
as follows.


**F** _L_ = FFN _(_ Cat _(_ **F** _l_ _, (_ Self-att _(_ **F** _l_ _,_ **F** _l_ _)_ + Bi-att _(_ **F** _l_ _,_ **F** _r_ _))/_ 2 _)),_

(2)



where Cat is the concatenation operation. Bi-att is the
Bi-modal attention operation.
In a similar fashion, the acquisition of the right attentional
feature **F** _R_ can be achieved through the introduction of the left
modality into the right one. Details are shown as follows.


**F** _R_ = FFN _(_ Cat _(_ **F** _r_ _, (_ Self-att _(_ **F** _r_ _,_ **F** _r_ _)_ + Bi-att _(_ **F** _r_ _,_ **F** _l_ _))/_ 2 _)),_

(3)


Note that the co-attention can be stacked multiple times in
our model for adequate and hierarchical extraction and fusion
of two inputs.
_3) Question-Aware Feature Extraction:_ Effective processing of AVQA tasks necessitates the question serving as an
anchor for locating crucial information within video content. We address this issue by proposing the question-aware
attention mechanism, which is illustrated in Fig. 2. Herein,
we present a detailed exposition of its structure.
This module consists of three inputs, denoted as left modality **M** _l,q_, right modality **M** _r,q_, and question **Q** . The utilization
of question-aware attention allows for the capture of features
that are relevant to the posed question. The process commences with the implementation of self-attention in order to
capture long-range interdependent features of different inputs.


**F** _l,q_ = Self-att _(_ **M** _l,q_ _,_ **M** _l,q_ _),_ (4)

**F** _r,q_ = Self-att _(_ **M** _r,q_ _,_ **M** _r,q_ _),_ (5)

**F** _q_ = Self-att _(_ **Q** _,_ **Q** _),_ (6)


where **F** _l,q_, **F** _r,q_ and **F** _q_ are the features of **M** _l,q_, **M** _r,q_, and
question **Q** after processing by the self-attention, respectively.
To effectively locate the question-relevant information
across distinct input sources, the question feature **F** _q_ is used
to query **F** _l,q_ and **F** _r,q_ . Subsequently, the obtained results are
concatenated with **F** _q_ to derive the question-guided features.
The operation is shown as follows.


**F** _L_ _,q_ = FFN _(_ Cat _(_ **F** _q_ _,_ Bi-att _(_ **F** _l,q_ _,_ **F** _q_ _))),_ (7)

**F** _R,q_ = FFN _(_ Cat _(_ **F** _q_ _,_ Bi-att _(_ **F** _r,q_ _,_ **F** _q_ _))),_ (8)


where **F** _L_ _,q_ and **F** _R,q_ are the features extracted under the
guidance of the question.
_4) Global Branch:_ In this branch, integration of extracted
audio and visual information is accomplished through employment of the co-attention module, leading to the acquisition
of comprehensive video features. Specifically, as shown in
Fig. 2, the audio ( **A** ) and visual ( **V** [+] / **V** [−] ) modalities are fed
into the co-attention module to sufficiently correlate with each
other. Therefore, we can obtain **A** [+] _glb_ [and] **[ A]** [−] _glb_ [denoting global]
audio features by fusing **A** with **V** [+] and **V** [−], respectively.
Additionally, **V** [+] _glb_ [and] **[ V]** [−] _glb_ [represent the global visual features]
through separately integrating **V** [+] and **V** [−] with **A** .
On the top of the global branch, we concatenate positive
features **A** [+] _glb_ [and] **[ V]** [+] _glb_ [from the audio and visual channels as]
the global feature **F** _glb_ .


**F** _glb_ = Cat _(_ **A** [+] _glb_ _[,]_ **[ V]** [+] _glb_ _[).]_ (9)


It should be noted that the usage of negative audio-visual
pairs is exclusively limited to the first stage. Given that the



Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


CHEN et al.: QUESTION-AWARE GLOBAL-LOCAL VIDEO UNDERSTANDING NETWORK FOR AVQA 4113



model deciphers visual and audio content separately through
distinct channels in the first stage of the proposed framework,
it becomes crucial to establish alignment between audio and
visual modalities from both temporal and content perspectives.
We employ contrastive learning to guarantee the alignment
of positive audio-visual pairs and precise feature extraction,
encompassing positive audio and visual features alongside
all negative visual features present within a given batch. All
negative global features in a batch are gathered into the global
negative feature tensor **V** _all_ [−] _,glo_ [.]


_L_ _glb_ = L InfoNCE _(_ **A** [+] _glb_ _[,]_ **[ V]** [+] _glb_ _[,]_ **[ V]** _all_ [−] _,glo_ _[),]_ (10)


where L InfoNCE is the InfoNCE (in which NCE means Noise
Contrastive Estimation) loss [26], which is used to implement
the contrastive learning. We use _L_ _glb_ to denote the contrastive
learning loss of the global branch.
_5) Local Branch:_ In this branch, the question-aware attention is implemented to facilitate the capture of task-relevant
local features from audio and visual modalities. By incorporating question-aware attention in conjunction with the
co-attention, it becomes possible to model both the selfinfluence stemming from single modality and the interactive
influence among three modalities, including the question to
audio, question to visual, audio to visual, and visual to audio,
with further details provided below.
Initially, we initiate the information processing pipeline
by inputting the visual ( **V** [+] _/_ **V** [−] ), audio ( **A** ), and text ( **Q** )
modalities into the question-aware attention module. This
step serves to effectively identify key information that is
closely associated with the posed question. Subsequently, the
audio and visual information, having undergone the process
of localization and refinement, is further integrated using coattention module. As a result of this integration, we are able to
derive distinct local audio features, denoted as **A** _loc_ [+] [(positive)]
and **A** _loc_ [−] [(negative), which are obtained by fusing the audio]
feature with positive and negative visual features, respectively. Additionally, our approach also enables the extraction
of local visual features, represented as **V** _loc_ [+] [(positive) and]
**V** _loc_ [−] [(negative), contributing to a comprehensive and detailed]
representation of the underlying data.
Next, positive local features of audio and visual modalities
are concatenated as the local video feature **F** _loc_ .


**F** _loc_ = Cat _(_ **A** _loc_ [+] _[,]_ **[ V]** _loc_ [+] _[).]_ (11)


Similar to the global branch, **V** _all_ [−] _,loc_ [encompasses all local]
negative visual features **V** _loc_ [−] [in a batch. Moreover, the con-]
trastive learning is utilised between the positive features and
negative features to align the positive audio-visual features.


_L_ _loc_ = L InfoNCE _(_ **A** _loc_ [+] _[,]_ **[ V]** _loc_ [+] _[,]_ **[ V]** _all_ [−] _,loc_ _[),]_ (12)


where _L_ _loc_ is the contrastive learning loss of the local branch.


_C. Global Local Fusion and Answer Prediction_


_1) Global-Local Fusion:_ In this part, we present a fusion
module that strategically integrates global and local features,
guided by question features. As depicted in the second stage of



Fig. 2, the process unfolds in three steps. Firstly, the questionaware attention module processes global ( **F** _glb_ ), local ( **F** _loc_ ),
and question ( **Q** ) features, facilitating the identification of key
question-relevant information. Secondly, the refined global and
local features undergo fusion using the co-attention module,
ensuring a comprehensive integration. Lastly, the output of
the co-attention module is concatenated, yielding the final
fused feature ( **F** _f inal_ ), which subsequently undergoes answer
decoding.
_2) Answer Prediction:_ At last, **F** _f inal_ is used as the input
of a classifier to predict the answer from an answer candidate
pool, i.e., probabilities **p** ∈ R _[C]_, where _C_ is the size of the
answer candidate pool.
_3) Objective Function:_ AVQA is a challenging task to
optimize as it requires comprehensive scene understanding and
question-aware feature fusion. To enhance learning, we optimize three losses to help our model learn informative and
correct representations. For the first loss, with the predicted
probability vector **p** and the ground-truth label **y**, we use a
cross-entropy loss to optimize the prediction accuracy:


_L_ _qa_ = − [�] _[C]_ _c_ =1 _[y]_ _[c]_ [ log] _[(]_ _[p]_ _[c]_ _[),]_ (13)


where _y_ _c_ and _p_ _c_ are _c_ -th component of **y** and **p** .
At the same time, we use the InfoNCE loss both in global
branch and local branch to ensure the alignment of extracted
positive audio-visual features. As a result, the objective function of our model is defined as


_L_ _AVQA_ = _L_ _qa_ + _λ_ _glb_ × _L_ _glb_ + _λ_ _loc_ × _L_ _loc_ _,_ (14)


where _λ_ _glb_ and _λ_ _loc_ denote the scaling factors.


IV. E XPERIMENTAL R ESULT


In this section, we present a comprehensive performance
analysis of the proposed model by introducing our experimental setup, dataset, evaluation protocols, and baselines in
Sec. A. Subsequently, we report the experimental results to
showcase the efficacy of our method in Sec. B. This includes
a comparison with the aforementioned baselines, an ablation
study, hyperparameter exploration, and a visualized attention
map of the local branch. Our findings highlight the superiority
of the proposed model and offer valuable insights into its inner
workings.


_A. Experimental Setup_


_1) Dataset and Evaluation:_ Audio-visual question answering is a newly emerging task. MUSIC-AVQA [13] and
AVQA [14] are the latest and only two publicly released
datasets. Our study focuses on the utilization of these two
datasets as the testbed to investigate the performance of our
model. MUSIC-AVQA [13] comprises 9 _,_ 288 videos (mainly
about the concert scenarios) and 45 _,_ 867 question-answer
pairs, spanning nine distinct question types, i.e., Audio questions: _Counting_ and _Comparative_ ; Visual questions: _Counting_
and _Location_ ; Audio Visual questions: _Existential, Location,_
_Counting, Comparative_, and _Temporal_ . While AVQA [14]
consists of 57 _,_ 015 videos reflecting the real-world scenarios
and 57 _,_ 335 specially-designed question-answer pairs relying



Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


4114 IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 5, MAY 2024



on clues from audio and visual modalities, including eight
question types, i.e., _Which, Come From, Happening, Where,_
_Why, Before Next, When_, and _Used For_ . We use a predefined
set of training, testing, and validation in the benchmark
datasets [13], [14]. Each original video in the dataset is
about 60-second long and is divided into 60 non-overlapping
segments of the same length. The sampling rates of sounds and
video frames are 16 _kHz_ and 1 _f ps_, respectively. Following
the baseline [13], we sampled each video by taking one-second
long segment every six-second long segment, constituting
10 one-second long segments. In our performance assessment,
we adopt answer prediction accuracy as the primary metric,
as depicted in Eq. (15). This accuracy metric is determined
by assessing the proportion of correctly predicted questions
for each question type. The numerator _N_ _p_ in Eq. (15) represents the count of accurately predicted questions, while the
denominator _N_ _all_ in Eq. (15) corresponds to the total number
of questions belonging to that particular question type. The
algorithm objective is to predict the answer with the highest
probability, which ideally corresponds to the correct answer.

**Acc** = _[N]_ _[p]_ (15)

_N_ _all_


_2) Model Training:_ The hyperparameters for the proposed
model are pre-determined as per the following specifications.
Certain training configurations are adopted from the MUSICAVQA baseline [13], encompassing aspects such as a 512-D
feature for visual, audio, and text modalities, an initial learning rate of 0 _._ 0001, and a comprehensive text preprocessing
methodology that includes tokenization, padding, and vocabulary construction.
However, certain divergences from the baseline are introduced. Specifically, the learning rate will drop by multiplying
0 _._ 1 every eight training epochs (in contrast to every 10 epochs
in [13]). The model undergoes 15 training epochs (compared
to 30 epochs in [13]) with a mini-batch size of 24 (in contrast
to 64 in [13]).
Two Nvidia RTX 3090 Ti GPUs are employed to train
the proposed model with the Adam optimizer. The multihead attention mechanism utilizes four parallel attention heads
( _h_ = 4). Within the model architecture, the global branch
consists of two stacked co-attention modules, while both the
local branch and global-local fusion block have one stacked
co-attention module each. We set the weights for the global
and local branches to be _λ_ _glb_ = _λ_ _loc_ = 0 _._ 1, respectively,
in Eq. (14).
In order to ensure a fair comparison, the feature encoders
used are identical to that of the baseline [13]: pre-trained
VGGish [22] for audio, pre-trained ResNet-18 [23] for visual,
and an LSTM for processing questions. During training, the
VGGish and ResNet-18 encoders remain frozen. The main

novelty of this work lies in the aspects of video understanding
and modality fusion, where significant differences emerge
compared to the other counterparts.
_3) Baselines:_ We assess the efficacy of our model on the
MUSIC-AVQA [13] and AVQA [14] datasets by comparing it with multiple existing relevant methods. Specifically,
we compare our approach with single-modality QA methods,



such as AudioQA [7], VisualQA [28], and VideoQA [31],

[33], [34], to demonstrate the benefits of multimodal perception in facilitating question answering. AudioQA methods
respond questions based on audio signals without the help of
visual information. On the contrary, VisualQA and VideoQA
approaches answer questions based on the visual features
without the audio input. Visual input of the VisualQA is
typically an image, while VideoQA is based on a video
segment input. Moreover, we evaluate our algorithm against
existing work developed for audio-visual scene understanding
to verify its performance improvement. These audio-visual
scene-based methods include:

_4) AVSD:_ AVSD method answers questions through understanding the video content from visual, audio, and dialogue
information. But in AVQA tasks, there is no dialogue for video
understanding, so we compare AVSD method for investigating
the audiovisual scene perception ability of our model without
dialog assistance. Reference [32] uses a multimodal attention
mechanism to fuse three modalities (audio, visual, and textual)
with equal contribution to answer generation.
_5) AVQA:_ AVQA-based methods are what we focus on
comparing to demonstrate the superiority of our model, including (1) Pano-AVQA [12], which employs a cross-attention
module to fuse three modalities; (2) HAVF [14], which serves
as the baseline of AVQA [14] dataset by incorporating with
other bi-modality fusion method. It comprises three fusion
methods and ensembles three fusion outputs through an averaging strategy to generate the answer; (3) ST-AVQA [13],
which is the baseline method reported on the MUSIC-AVQA

[13] dataset. ST-AVQA locates the spatial and temporal area
in audio and visual signals using an attention mechanism and
fuses three modalities to predict the answer.


_B. Results and Discussion_


In this section, we present a comprehensive evaluation of the
proposed model through various analyses. Firstly, we compare
its performance against relevant QA approaches to establish
its effectiveness and superiority. Secondly, extensive ablation
studies are conducted to gain a comprehensive understanding
of inner workings of our model. Thirdly, we report the
results obtained with different hyperparameters to explore the
robustness of the model. Fourthly, we provide a visualized
result to enhance the interpretation of the local branch. Finally,
the prediction results of our model are depicted to further
demonstrate the effectiveness of our model.
_1) Results on MUSIC-AVQA Dataset:_ Tab. I presents the
findings of our study evaluating recent QA methods and
our proposed model on the MUSIC-AVQA [13] dataset. The
results except the HCRN+HAVF and Ours in Tab. I are all
imported from [13]. We implement HCRN+HAVF according
to the architecture depicted in [14]. From the last column
of Tab. I, we can clearly tell that the averaged accuracies
of single-modality QA methods (i.e., AudioQA, VisualQA,
and VideoQA) are significantly lower than multimodality QA
approaches, such as AVSD and AVQA. These findings suggest
that multimodal perception can enhance audio-visual scene
understanding and improve QA performance. Furthermore,



Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


CHEN et al.: QUESTION-AWARE GLOBAL-LOCAL VIDEO UNDERSTANDING NETWORK FOR AVQA 4115


TABLE I


F INE -G RAINED E VALUATION P ERFORMANCE (%) OF B ASELINES AND O UR P ROPOSED M ODEL ON MUSIC-AVQA [13] D ATASET . T HE T OP -2 R ESULTS
A RE H IGHLIGHTED . T HE B EST AND S ECOND B EST A CCURACIES OF E ACH Q UESTION T YPE A RE H IGHLIGHTED IN B OLD F ORM AND U NDERLINE
F ORM, R ESPECTIVELY . I N A DDITION, FOR E ACH Q UESTION C ATEGORY, W E P RESENT THE P ERFORMANCE I NCREASE B ROUGHT BY O UR
M ETHOD W ITH R ESPECT TO THE B EST R ESULT A MONG A LL THE E XISTING M ETHODS C OMPARED IN THE T ABLE


TABLE II


F INE -G RAINED T ESTING P ERFORMANCE (%) OF QA M ETHODS, M ETHOD +HAVF, AND O UR M ODEL ON AVQA [14] D ATASET . T HE B EST AND S ECOND
B EST P ERFORMANCE O VER E ACH Q UESTION T YPE A RE H IGHLIGHTED IN B OLD F ORM AND U NDERLINE F ORM, R ESPECTIVELY



our proposed model achieves better performance than other
multimodal QA methods on most subtasks, with significant
improvements observed for all audio and visual questions.
Notably, our method achieves similar accuracy (82.49%) as
the highest-performing method on _Existential_ questions for
audio-visual questions, while surpassing other methods on
the remaining question types, particularly on _Location_ and
_Comparative_ questions. The results consistently demonstrate
the effectiveness of our method in audio-visual scenes and its

ability to provide more accurate responses to questions.
_2) Results on AVQA Dataset:_ In this section, we report
the evaluation results of our model and other counterparts
on the AVQA [14] dataset. The results except the last row
(Ours) in Tab. II are all imported from [14]. As shown in
Tab. II, our approach yields substantial improvements in QA
accuracy across all question types when compared to multiple
QA approaches. Specifically, our model achieve significant
increase (+6.7%) on _Why_ question type. In addition, the
accuracy improvements of _Which_ (+0.4%) and _Come From_
(+0.2%) are relatively small. We think this is because the
difficulty of these two question types is relatively low, so most
of the QA methods can achieve good accuracy on them. These
results underscore the effectiveness of our question-aware
QA model in accurately inferring answers in audio-visual
settings.
_3) Ablation Study:_ In this part, ablation studies are conducted to gain a deeper understanding of our model and to
validate the indispensability and efficacy of each constituent
within our framework based on MUSIC-AVQA [13] dataset.



The outcomes of this study are presented in Tab. III. Our
model, as previously detailed, is composed of two stages:
(i) the _First Stage_, which comprehends the video content
through a combination of global and local perspectives, and
(ii) the _Second Stage_, which includes a fusion module that
is question-oriented, global and local feature-based, and an
answer prediction block. We undertake partial ablation of
crucial components within the model architecture, followed
by retraining and validation procedures to ascertain the significance of these elements in shaping the overall model
performance.
The first issue that we aim to investigate is what extent to
which the features derived from the global and local branches
can aid in the comprehension of audio-visual scenes by a
model. We begin by conducting an ablation study wherein
we replace the inputs of the _Second Stage_ module, namely
the global and local inputs, with the outputs from the audio
and visual channels in the global branch after removing the
local branch. The experimental results, depicted in the first
row of Tab. III under the label “Without Local,” reveal that

the absence of local features leads to a considerable decrease

on averaged accuracy (−1 _._ 4%). Furthermore, we repeat the
above-mentioned procedure by removing the global branch
and present the outcomes in the second row of Tab. III
labeled as “Without Global.” It can be clearly observed that
the averaged accuracy drops by 1 _._ 21% after eliminating the
global branch. Our findings demonstrate that both global and
local features contribute significantly to the performance of the
model. Specifically, the performance drops more significantly



Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


4116 IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 5, MAY 2024


TABLE III


A BLATION S TUDY ON THE C ONTRIBUTION F ROM THE K EY C OMPONENTS OF O UR P ROPOSED M ODEL ON MUSIC-AVQA [13] D ATASET


TABLE IV


V ARYING THE S CALING F ACTORS _λ_ _glb_ AND _λ_ _loc_ IN E Q . (14)


Fig. 3. The cross-modality attention in the model is removed to investigate
the contribution of this mechanism to our model. The left is the cross-attention
module in the proposed model, and the right is the cross-attention removal
module.



when the local branch is removed, and the A-V question types
demonstrate a greater reliance on the global features.
The second issue that we investigate is the significance of
audio-visual feature alignment for the precise derivation of
video representations. To accomplish this, we eliminate the
contrastive learning module in the global and local branches,
as well as the InfoNCE loss _L_ _glb_ and _L_ _loc_ in Eq. (14). Our
findings, presented in the third row of Tab. III, show that
the averaged accuracy significantly drops by 1 _._ 69%, revealing
that aligning audio-visual representations both in global and
local branches can considerably enhance the performance of
the model. Although removing InfoNCE loss leads to a slight
accuracy increase (+0 _._ 43%) of the audio questions.
Thirdly, we examine the cruciality of the cross-modality
attention mechanism in the proposed model. In this regard,
we perform an ablation study by removing the cross-attention
operation (as seen in Fig. 3) and report the corresponding
results in the fourth row of Tab. III. The results illustrate that

the cross-attention mechanism provides significant benefits to
both visual question and audio-visual question types, although
its impact on the audio question type is comparatively lower.
Furthermore, we investigate the significance of leveraging
the provided question as a guide in the Second Stage of our
model. To explore this, we replace the given question with
a dummy question that does not contain video information.
We set all the elements of this dummy question to a uniform value of “1” and see what happens. The fifth row in
Tab. III presents the experimental results, demonstrating a
substantial decline in model performance when the provided
question is not used as a guide any more. Specifically, the
accuracy of all question types decrease significantly, with the
averaged accuracy dropping by 2 _._ 37%. These results suggest
that employing the given question facilitates the fusion of



global and local features, and ultimately improves the answer
generation process.
_4) Hyperparameter Analysis:_ In this part, we investigate
the impact of various loss functions on our proposed method
by manipulating the scaling factors _λ_ _glb_ and _λ_ _loc_ in Eq. (14).
Firstly, we fix the _λ_ _glb_ to 0 _._ 1 and adjust the _λ_ _loc_ from 0 _._ 01 to
0 _._ 7. Furthermore, we fix the _λ_ _loc_ to 0 _._ 1 and adjust the _λ_ _glb_
from 0 _._ 01 to 0 _._ 7. Our findings, shown in Tab. IV, demonstrate
that the performance of our model is not sensitive to these
two hyperparameters. It is worth noting that we always set
the scaling factor of the first term ( _L_ _qa_ ) in Eq. (14) to 1 as
the answer prediction accuracy is the key driver of model
performance, whereas the impact of contrastive learning on
model training is relatively minor.
_5) Visualized Analysis of Local Branch:_ Through our
ablation study in Tab. III, we show that the local branch significantly improves the model performance for most question
types. We visualize the attention map of the visual channel in
the local branch in Fig. 4 to explore the correlation between
the question words and the sampled video frames.
For example, as shown in Fig. 4 (a), we employ a video
with two people playing different instruments and a questionanswer pair, i.e., question: _“Is the instrument on the left more_
_rhythmic than the instrument on the right?”_ answer: _“Yes”_ .
The heatmap illustrates varying attentional intensities between
the question words and video frames, with visual features
of the 31st and 37th timestamps demonstrating a stronger
correlation with the question. Furthermore, specific words such
as _“instrument”_, _“left”_, _“more”_, _“rhythmic”_, and _“than”_



Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


CHEN et al.: QUESTION-AWARE GLOBAL-LOCAL VIDEO UNDERSTANDING NETWORK FOR AVQA 4117


Fig. 4. The attention map visualization of the video channel in the local branch. The color of the image reflects the attentional intensities between question
words and video frames. The greater attentional intensity, the stronger correlation. The red color means stronger intensities and the blue color represents the
weaker ones.


Fig. 5. Prediction results of our model for different questions types in the MUSIC-AVQA [13] dataset. The figure provides the video frames, audio waveforms,
questions (indicated by “Q”), the ground-truth answers (indicated by “A”), and the top-3 answers predicted by our model (indicated by “Top-3”).



activate more visual features, indicating their crucial role in
feature extraction from visual signals. Moreover, by listen to
the original audio of this video, we find that the audio from the
20th to 53rd seconds clearly highlights that the violin playing
on the left is more rhythmic than the accordion playing on the
right. This is consistent in time with the attentional intensities
shown by the attention map.
In addition, in Fig. 4 (b), the image captured at the 49th
second, wherein no instrument is present, exhibits a weaker
correlation with any of the words present in the question.



Moreover, in Fig. (e), the images with the accordion present
show stronger association with the question.
Overall, our findings suggest that the local branch is an
effective module for improving the performance of the model.
_6) Prediction Results of Our Model:_ In this part, we present
the predictive analysis outcomes of our model on each question
type of the MUSIC-AVQA [13] dataset. The results, depicted
in Fig. 5, include the question, the ground-truth answer, and
the top-3 model-generated results for each question type.
Our findings reveal that our model adeptly deduces answers



Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


4118 IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 5, MAY 2024



for diverse question types. Despite some instances where
the correct answers do not obtain the highest probability,
as evidenced by examples (e) and (f) depicted in Fig. 5,
our model effectively identifies the correct answer and ranks
it among the top-2 answers, thus showcasing the potential
efficacy of the proposed methodology.


V. C ONCLUSION


In this paper, we investigated the AVQA task and identified the main challenges and requirements. To address these
issues, we propose a novel framework for multimodal question
answering, which provides a comprehensive understanding of
audio-visual scenarios at both global and local levels. Our
framework considers the importance of the given question in
an AVQA task, which guides the feature extraction of audio
and visual signals and the final fusion between global and local
features. Additionally, our method sufficiently correlates and
fuses multimodal data. Experimental results demonstrate the
effectiveness and superiority of our design.


_A. Limitation_


Our framework currently assumes the availability of all
modalities at all times, which may not be the case in some
scenarios. As such, our framework can be further improved
to handle missing modality or information in the future.
Furthermore, following the classic question answering setting in the literature, our proposed method constitutes a
classification framework that primarily operates within the
constraints of a close-set answer setting, restricting it from
accommodating the diverse range of potential answers. As part
of our future research endeavors, we aim to extend this
framework towards the generative paradigm, thereby enabling
us to effectively address the complexities inherent in more
scenarios. In addition, given the extensive applications of
AVQA in information retrieval, human-computer interaction,
and visual/auditory assistance, we should better consider issues
of fairness, transparency, and explainability within the QA
framework to ensure its safe and responsible use.


A CKNOWLEDGMENT


The authors would like to thank Guangyao Li for his helpful
discussion about the MUSIC-AVQA dataset during the course
of this work.


R EFERENCES


[1] S. Antol et al., “VQA: Visual question answering,” in _Proc. IEEE Int._
_Conf. Comput. Vis. (ICCV)_, Dec. 2015, pp. 2425–2433.

[2] T. Yu, J. Yu, Z. Yu, Q. Huang, and Q. Tian, “Long-term video question
answering via multimodal hierarchical memory attentive networks,”
_IEEE Trans. Circuits Syst. Video Technol._, vol. 31, no. 3, pp. 931–944,
Mar. 2021.

[3] H.-T. Su et al., “End-to-end video question-answer generation with
generator-pretester network,” _IEEE Trans. Circuits Syst. Video Technol._,
vol. 31, no. 11, pp. 4497–4507, Nov. 2021.

[4] L. Zhao et al., “Toward explainable 3D grounded visual question
answering: A new benchmark and strong baseline,” _IEEE Trans. Circuits_
_Syst. Video Technol._, vol. 33, no. 6, pp. 2935–2949, Jun. 2023.

[5] F. Zhang, R. Wang, F. Zhou, and Y. Luo, “ERM: Energy-based
refined-attention mechanism for video question answering,” _IEEE Trans._
_Circuits Syst. Video Technol._, vol. 33, no. 3, pp. 1454–1467, Mar. 2023.




[6] L. Li et al., “Multi-granularity relational attention network for audiovisual question answering,” _IEEE Trans. Circuits Syst. Video Technol._,
[early access, Apr. 5, 2023, doi: 10.1109/TCSVT.2023.3264524.](http://dx.doi.org/10.1109/TCSVT.2023.3264524)

[7] H. M. Fayek and J. Johnson, “Temporal reasoning via audio question answering,” _IEEE/ACM Trans. Audio, Speech, Language Process._,
vol. 28, pp. 2283–2294, 2020.

[8] G. Li, Y. Xu, and D. Hu, “Multi-scale attention for audio question
answering,” 2023, _arXiv:2305.17993_ .

[9] H. Alamri, C. Hori, T. K. Marks, D. Batra, and D. Parikh, “Audio Visual
Scene-aware Dialog (AVSD) track for natural language generation in
DSTC7,” in _Proc. AAAI Workshop_, vol. 2, 2018, pp. 1–6.

[10] S. Kim et al., “Overview of the eighth dialog system technology challenge: DSTC8,” _IEEE/ACM Trans. Audio, Speech, Language Process._,
vol. 29, pp. 2529–2540, 2021.

[11] A. Shah et al., “Audio-visual scene-aware dialog and reasoning using
audio-visual transformers with joint student-teacher learning,” in _Proc._
_IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)_, May 2022,
pp. 7732–7736.

[12] H. Yun, Y. Yu, W. Yang, K. Lee, and G. Kim, “Pano-AVQA: Grounded
audio-visual question answering on 360 [◦] videos,” in _Proc. IEEE/CVF_
_Int. Conf. Comput. Vis. (ICCV)_, Oct. 2021, pp. 2031–2041.

[13] G. Li, Y. Wei, Y. Tian, C. Xu, J.-R. Wen, and D. Hu, “Learning to answer
questions in dynamic audio-visual scenarios,” in _Proc. IEEE/CVF Conf._
_Comput. Vis. Pattern Recognit. (CVPR)_, Jun. 2022, pp. 19086–19096.

[14] P. Yang et al., “AVQA: A dataset for audio-visual question answering
on videos,” in _Proc. 30th ACM Int. Conf. Multimedia_, Oct. 2022,
pp. 3480–3491.

[15] Y.-B. Lin, Y.-L. Sung, J. Lei, M. Bansal, and G. Bertasius, “Vision
transformers are parameter-efficient audio-visual learners,” in _Proc._
_IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)_, Jun. 2023,
pp. 2299–2309.

[16] S. Chen et al., “VALOR: Vision-Audio-Language Omni-peRception
pretraining model and dataset,” 2023, _arXiv:2304.08345_ .

[17] F. Xiao, Y. J. Lee, K. Grauman, J. Malik, and C. Feichtenhofer, “Audiovisual SlowFast networks for video recognition,” 2020,
_arXiv:2001.08740_ .

[18] A. Nagrani, S. Yang, A. Arnab, A. Jansen, C. Schmid, and C. Sun,
“Attention bottlenecks for multimodal fusion,” in _Proc. Adv. Neural Inf._
_Process. Syst._, vol. 34, 2021, pp. 14200–14213.

[19] N. Shvetsova et al., “Everything at once—Multi-modal fusion transformer for video retrieval,” in _Proc. IEEE/CVF Conf. Comput. Vis._
_Pattern Recognit. (CVPR)_, Jun. 2022, pp. 19988–19997.

[20] R. Arandjelovic and A. Zisserman, “Look, listen and learn,” in _Proc._
_IEEE Int. Conf. Comput. Vis. (ICCV)_, Oct. 2017, pp. 609–617.

[21] A. Rouditchenko et al., “AVLnet: Learning audio-visual language representations from instructional videos,” 2020, _arXiv:2006.09199_ .

[22] J. F. Gemmeke et al., “Audio set: An ontology and human-labeled
dataset for audio events,” in _Proc. IEEE Int. Conf. Acoust., Speech Signal_
_Process. (ICASSP)_, Mar. 2017, pp. 776–780.

[23] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri,
“A closer look at spatiotemporal convolutions for action recognition,”
in _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._, Jun. 2018,
pp. 6450–6459.

[24] N. Xu, W. Mao, and G. Chen, “Multi-interactive memory network for
aspect based multimodal sentiment analysis,” in _Proc. AAAI Conf. Artif._
_Intell._, 2019, vol. 33, no. 1, pp. 371–378.

[25] G. P. Rajasekar et al., “A joint cross-attention model for audio-visual
fusion in dimensional emotion recognition,” 2022, _arXiv:2203.14779_ .

[26] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with
contrastive predictive coding,” 2018, _arXiv:1807.03748_ .

[27] J. Lu, J. Yang, D. Batra, and D. Parikh, “Hierarchical question-image
co-attention for visual question answering,” in _Proc. Adv. Neural Inf._
_Process. Syst._, vol. 29, 2016, pp. 1–9.

[28] Z. Yu, J. Yu, Y. Cui, D. Tao, and Q. Tian, “Deep modular co-attention
networks for visual question answering,” in _Proc. IEEE/CVF Conf._
_Comput. Vis. Pattern Recognit. (CVPR)_, Jun. 2019, pp. 6274–6283.

[29] X. Li et al., “Beyond RNNs: Positional self-attention with co-attention
for video question answering,” in _Proc. AAAI Conf. Artif. Intell._, 2019,
vol. 33, no. 1, pp. 8658–8665.

[30] C. Fan, X. Zhang, S. Zhang, W. Wang, C. Zhang, and H. Huang,
“Heterogeneous memory enhanced multimodal attention model for video
question answering,” in _Proc. IEEE/CVF Conf. Comput. Vis. Pattern_
_Recognit. (CVPR)_, Jun. 2019, pp. 1999–2007.

[31] T. M. Le, V. Le, S. Venkatesh, and T. Tran, “Hierarchical conditional relation networks for video question answering,” in _Proc._
_IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)_, Jun. 2020,
pp. 9969–9978.



Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


CHEN et al.: QUESTION-AWARE GLOBAL-LOCAL VIDEO UNDERSTANDING NETWORK FOR AVQA 4119




[32] I. Schwartz, A. G. Schwing, and T. Hazan, “A simple baseline for
audio-visual scene-aware dialog,” in _Proc. IEEE/CVF Conf. Comput._
_Vis. Pattern Recognit. (CVPR)_, Jun. 2019, pp. 12540–12550.

[33] J. Zhang, J. Shao, R. Cao, L. Gao, X. Xu, and H. T. Shen, “Action-centric
relation transformer network for video question answering,” _IEEE Trans._
_Circuits Syst. Video Technol._, vol. 32, no. 1, pp. 63–74, Jan. 2022.

[34] P. Jiang and Y. Han, “Reasoning with heterogeneous graph alignment for
video question answering,” in _Proc. AAAI Conf. Artif. Intell._, Apr. 2020,
vol. 34, no. 7, pp. 11109–11116.


**Zailong Chen** received the B.S. degree from the
Nanjing University of Posts and Telecommunications (NUPT), China, in 2015, and the M.E.
degree from Hunan University, China, in 2022. He
is currently pursuing the Ph.D. degree with the
School of Computing and Information Technology,
University of Wollongong, Australia. His research
interests include deep learning, computer vision, and
human–computer interaction.


**Lei Wang** (Senior Member, IEEE) received the
Ph.D. degree from Nanyang Technological University, Singapore. He is currently a Professor
with the School of Computing and Information
Technology, University of Wollongong, Australia.
He has published more than 190 peer-reviewed
papers, including those in highly regarded journals
and conferences, such as IEEE T RANSACTIONS
ON P ATTERN A NALYSIS AND M ACHINE I NTELLI         GENCE, _International Journal of Computer Vision_,
CVPR, ICCV, and ECCV. His research interests
include machine learning, pattern recognition, and computer vision. He was
awarded the Early Career Researcher Award by the Australian Academy of
Science and Australian Research Council. He served as the General Co-Chair
of DICTA 2014 and the Program Co-Chair of ACCV 2022. He served on
the technical program committees for many international conferences and
workshops.



**Peng Wang** received the Ph.D. degree from the
School of Information Technology and Electrical
Engineering, The University of Queensland. He is
a Professor with the School of Computer Science
and Engineering, University of Electronic Science
and Technology of China. Prior to joining UESTC,
he was a Lecturer with the School of Computing and
Information Technology, University of Wollongong.
His major research interest lies in computer vision
and deep learning, with special interest in dataefficient deep learning.


**Peng** **Gao** received the B.S. degree from the
Beijing University of Posts and Telecommunications
(BUPT), China, in 2002, and the M.A. degree from
the University of Bristol (UoB), U.K., in 2022.
He is currently pursuing the Ph.D. degree with
the Institute of Computer Science, Beijing Normal
University–Hong Kong Baptist University United
International College, Zhuhai, China. His research
interests include deep learning, computer vision, and
person recognition.



Authorized licensed use limited to: Qatar National Library. Downloaded on October 05,2025 at 15:42:18 UTC from IEEE Xplore. Restrictions apply.


