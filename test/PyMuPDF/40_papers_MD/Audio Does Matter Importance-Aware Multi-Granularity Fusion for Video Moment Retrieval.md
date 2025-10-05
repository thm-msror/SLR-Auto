## **Audio Does Matter: Importance-Aware Multi-Granularity Fusion** **for Video Moment Retrieval**



Junan Lin [∗]

Zhejiang University
Hangzhou, China
linja@zju.edu.cn


Xiaoye Qu
Shanghai Artificial Intelligence
Laboratory
Shanghai, China
xiaoye@hust.edu.cn



Daizong Liu [∗]

Peking University
Beijing, China
dzliu@stu.pku.edu.cn


Xun Yang
University of Science and Technology
of China

Hefei, China
hfutyangxun@gmail.com



Xianke Chen

Zhejiang Gongshang University
Hangzhou, China
a397283164@163.com


Jixiang Zhu
Zhejiang Gongshang University
Hangzhou, China
zhujx@mail.zjgsu.edu.cn



Sanyuan Zhang
Zhejiang University
Hangzhou, China
syzhang@zju.edu.cn


**Abstract**


Video Moment Retrieval (VMR) aims to retrieve a specific moment
semantically related to the given query. To tackle this task, most
existing VMR methods solely focus on the visual and textual modalities while neglecting the complementary but important audio
modality. Although a few recent works try to tackle the joint audiovision-text reasoning, they treat all modalities equally and simply
embed them without fine-grained interaction for moment retrieval.
These designs are counter-practical as: Not all audios are helpful for
video moment retrieval, and the audio of some videos may be complete noise or background sound that is meaningless to the moment
determination. To this end, we propose a novel Importance-aware
Multi-Granularity fusion model (IMG), which learns to dynamically
and selectively aggregate the audio-vision-text contexts for VMR.
Specifically, after integrating the textual guidance with vision and
audio separately, we first design a pseudo-label-supervised audio
importance predictor that predicts the importance score of the audio, and accordingly assigns weights to mitigate the interference
caused by noisy audio. Then, we design a multi-granularity audio
fusion module that adaptively fuses audio and visual modalities at
local-, event-, and global-level, fully capturing their complementary
contexts. We further propose a cross-modal knowledge distillation
strategy to address the challenge of missing audio modality during
inference. To evaluate our method, we further construct a new VMR


∗ Both authors contributed equally to this research.

- Corresponding author.


Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
_MM ’25, October 27–31, 2025, Dublin, Ireland._
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-2035-2/2025/10
[https://doi.org/10.1145/3746027.3754982](https://doi.org/10.1145/3746027.3754982)



Jianfeng Dong [†]

Zhejiang Gongshang University
Hangzhou, China
dongjf24@gmail.com


dataset, _i.e._, Charades-AudioMatter, where audio-related samples
are manually selected and re-organized from the original CharadesSTA to validate the model’s capability in utilizing audio modality.
Extensive experiments validate the effectiveness of our method,
achieving state-of-the-art with audio-video fusion in VMR methods.
[Our code is available at https://github.com/HuiGuanLab/IMG.](https://github.com/HuiGuanLab/IMG)


**CCS Concepts**


- **Information systems** → **Multimedia and multimodal re-**
**trieval** ; **Video search** .


**Keywords**


Video Moment Retrieval; Video Understanding; Multimodal Learning; Cross-Modal Alignment


**ACM Reference Format:**

Junan Lin, Daizong Liu, Xianke Chen, Xiaoye Qu, Xun Yang, Jixiang Zhu,
Sanyuan Zhang, and Jianfeng Dong. 2025. Audio Does Matter: ImportanceAware Multi-Granularity Fusion for Video Moment Retrieval. In _Proceedings_
_of the 33rd ACM International Conference on Multimedia (MM ’25), October_
_27–31, 2025, Dublin, Ireland._ [ACM, New York, NY, USA, 14 pages. https:](https://doi.org/10.1145/3746027.3754982)
[//doi.org/10.1145/3746027.3754982](https://doi.org/10.1145/3746027.3754982)


**1** **Introduction**


Video Moment Retrieval (VMR) [ 1, 12, 16, 64, 64 ] aims to retrieve
the part of the video that is relevant to the semantic of a given query.
As a fundamental yet important task, it requires in-depth interaction between video and text semantics for accurate alignment and
reasoning. Existing mainstream works [ 11, 13, 62, 63, 68, 69, 73, 74 ]
generally focus on naive visual and textual modalities and develop
vision-text integration frameworks to retrieve the specific moment.
However, in addition to the visual contexts, audio modality also
contains valuable contexts within the video streams [ 3, 18, 25, 39,
40, 42, 77 ]. Without considering the rich complementary contexts
of the audio modality, previous VMR methods fail to distinguish
different activities like “laughing" and “talking" that share a similar


MM ’25, October 27–31, 2025, Dublin, Ireland. Junan Lin et al.



visual appearance. Therefore, exploring the interaction and fusion
of audio, vision, and text modalities in VMR is a promising direction
with research forward.

To leverage audio information, several audio-based VMR methods [ 8, 9, 36 ] have been proposed. However, these approaches typically extract features from audio, vision, and text modalities and
apply a uniform aggregation strategy for joint reasoning, without
considering their diverse contributions. For instance, PMI-LOC [ 9 ]
incorporates RGB, motion, and audio modalities, establishes pairwise interactions between modalities, UMT [ 36 ] introduces a unified
multimodal transformer framework for the integration of visual
and audio information, while ADPN [ 8 ] leverages the consistency
and complementarity between audio and visual modalities for efficient audio fusion. Although these methods achieve relatively
better performance than conventional VMR methods, they neglect
that not all audios contribute to the final grounding as the audio of
some videos may be complete noise or background sound.
In practice, audio semantics exhibit considerable complexity and
diversity, varying significantly across different scenarios. In certain
instances, audio context serves as a valuable complement, enhancing the alignment with text semantics and facilitating accurate
reasoning. Conversely, noisy audio can lead to erroneous textual
associations. As shown in Figure 1, for the first query “a person is
laughing", leveraging audio context significantly aids in identifying
the laughing action, which might be ambiguous using vision alone.
However, for the second query “a person looks out a window", the
audio modality provides no benefit and may even be detrimental,
given that this action is primarily visually driven. Therefore, this
motivates us to design a more dynamic audio-vision-text association framework that learns to selectively and adaptively aggregate
appropriate contexts from audio and visual modalities for reasoning
the specific text semantics.
To this end, we make the first attempt to tackle a flexible audiovision-text joint reasoning for the VMR task. In particular, we propose a novel Importance-aware Multi-Granularity fusion model
(IMG) with three prediction branches: audio branch, visual branch,
and audio-visual fusion branch. Initially, textual guidance is integrated separately with both visual and audio inputs. We then
introduce an audio importance-aware module to tackle the issue of
variable audio importance, which is crucial in vision-text pairs. This
module is supervised by pseudo-labels derived from the retrieval
loss of each branch. It effectively learns to assess the relative importance of audio compared to vision. Then, for the latter audio-vision
context fusion, we design a multi-granularity fusion network, which
establishes local-level and event-level to global-level audio-vision
fusion, as a way to better discover key clues in audio for assisting
text-specific activity understanding within video contents. In addition to using traditional retrieval loss for supervision, since the
multi-modal fusion branch tends to show better performance than
the individual vision/audio reasoning branch as the former fuses
the positive contexts from both modalities, we also distillate the
knowledge from the fusion branch to the weaker visual and audio
branches, thus strengthening the performance of both branches and
in-turn providing better feedback to the fusion branch to further
improving the performance.
To sum up, the key contributions of our work are four-fold:



**Figure 1: (Top) Audio is a critical modality, outweighing the**
**importance of vision. (Bottom) Audio is entirely irrelevant**
**and considered noise relative to the vision.**


  - We propose a novel Importance-aware Multi-Granularity
fusion network (IMG) to handle the audio-incorporated VMR
task, which selectively fuses audio modal information of
video samples at multiple granularities for final retrieval.

  - We introduce an audio importance predictor, guided by a
loss-aware pseudo-importance generator during training, to
identify and emphasize semantically relevant audio clues.
This mechanism enables the model to selectively focus on
informative audio cues while suppressing irrelevant or noisy
background sounds.

  - We propose a cross-modal knowledge distillation strategy,
which transfers knowledge from the more effective fusion
branch to the unimodal branch. This strategy enables our
framework to retain strong performance even when audio
information is missing during inference.

  - In addition to standard benchmarks such as Charades-STA

and ActivityNet Captions, we introduce a new evaluation
dataset, Charades-AudioMatter, where sample’s audio matter for moment retrieval. Extensive experiments on these
datasets demonstrate the effectiveness of our approach, particularly in scenarios where audio cues play a complementary or dominant role.


**2** **Related Works**


**Video Moment Retrieval (VMR)** . VMR aims to retrieve a specific
video segment based on a natural language query. Current approaches fall into two categories: proposal-based and proposal-free.
For proposal-based [ 34, 47, 49, 57, 58, 72, 73, 75 ], it is often necessary
to pre-segment the candidate proposals, and the pre-segmented
proposals and text are used as inputs to the cross-modal matching
module for retrieval. For proposal-free [ 12, 22, 62, 63, 68 – 70, 74 ],
they eliminate the need for predefined proposals, processing raw
visual and textual features directly through cross-modal matching.
Building on these paradigms, recent studies [ 24, 29, 41, 48, 54 ] have
explored DETR-style architectures [ 6 ] to formulate VMR as a set
prediction problem, enabling more flexible and end-to-end training.
Further extending these trends, some works aim to unify various


Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval MM ’25, October 27–31, 2025, Dublin, Ireland.



video tasks (e.g., moment retrieval, highlight detection, video summarization) under a general framework [ 32, 61 ]. Meanwhile, the
rapid progress of large language models (LLMs) has inspired a new
wave of research that leverages their semantic reasoning capabilities to enhance VMR [ 20, 26, 45, 53, 56, 65 ]. Concurrently, audio
has emerged as a valuable modality for complementing vision in
VMR, _e.g._, PMI-LOC [ 9 ] employs RGB, motion, and audio and is designed to interact with pairs of modalities at the sequence and channel levels. UMT [ 36 ] proposes a unified multimodal transformer
framework to fuse vision and audio. ADPN [ 8 ] proposes a textguided clues miner to fill the information gap between audio-visual
modalities. However, the above models overlook the inherent uncertainty of audio as a modality and the contribution of audio varies
significantly depending on the specific query and video content,
highlighting a need for more adaptive solutions.
**Uncertain Modal Learning.** The audio modality often exhibits
uncertainty and imbalance in video comprehension tasks [ 15, 60 ].
For instance, audio in some videos may consist solely of noise or
background sounds, while in text-based video tasks, the query may
be entirely independent of the audio. Similar issues arise in other
modalities and these modal imbalance challenges have garnered
significant attention [ 10, 23, 33, 46, 59, 66 ]. To address these challenges, Li _et al._ [ 31 ] quantified uncertainty caused by inherent data
ambiguity to enhance prediction reliability. Tellamekala _et al._ [ 50 ]
addressed modal uncertainty in categorical sentiment recognition
by introducing a modeling approach that enforces both calibration and ordinality constraints and Zhang _et al._ [ 71 ] explored the
challenges and solutions for low-quality multimodal fusion, emphasizing the promise of dynamic multimodal learning in overcoming
sample-specific, temporal, and spatial variations. Building on these
developments, we introduce the audio importance predictor. Supervised by dynamic pseudo-labels derived from sample-wise loss
functions, this predictor quantifies the audio modality’s importance,
providing a critical parameter for adaptive modal fusion.


**3** **Method**

**3.1** **Overview**


**Problem Definition.** Video moment retrieval aims to retrieve the
start-end frame pair { _𝑓_ _𝑠_ _, 𝑓_ _𝑒_ } of a specific segment that semantically
match the textual query _𝑄_ = { _𝑤_ _𝑖_ } _𝑖_ _[𝑁]_ =1 [from the untrimmed video] _[𝑉]_ [=]
{ _𝑓_ _𝑡_ } _[𝑇]_ _𝑡_ =1 [, where] _[𝑤]_ _[𝑖]_ [represents the] _[𝑖]_ [-th word and] _[ 𝑓]_ _[𝑡]_ [represents the] _[𝑡]_ [-th]
frame. Additionally, for each video frame, we can extract an audioaware clip as a complementary modality. Thus, the corresponding
audio stream is represented as _𝐴_ = { _𝑎_ _𝑗_ } _[𝑇]_ _𝑗_ =1 [, where] _[ 𝑎]_ _[𝑗]_ [denotes the]
_𝑗_ -th audio clip, providing contextual knowledge to enhance the
retrieval process.
**Overall Pipline.** We illustrate our proposed framework in Figure 2.
Given the pre-extracted visual, audio, and text features by the corresponding encoder, our IMG model first employs feed-forward
network (FFN) layers to map these features into a common latent
space. Then, we employ interactions between vision-text and audiotext pairs, fusing them to derive text-semantic-activated visual and
audio features. These features are then passed into the visual-audio
fusion branch, where they dynamically interact to enable joint
reasoning. Specifically, an audio importance predictor generates a
sample-wise score, which serves as a crucial weight to determine



the audio-to-vision complementarity coefficients for the given sample pair. Then, the visual and audio features will be fed into a
multi-granularity fusion module to aggregate the target-momentrelated information at local-, event-, and global-levels according to
the previously obtained important weight. Finally, the three-level
features are concatenated and fed into the predictor to output predictions, while the visual-only feature and audio-only feature are
also fed into their respective unimodal predictors. A multi-branch
training with cross-modal knowledge distillation strategy is used to
transfer knowledge from the fusion branch to the unimodal branch.
During inference, the retrieval branch can be freely selected, with
the fusion branch typically being the preferred choice.


**3.2** **Input Representation**


**Multi-Modal Feature Representation.** _For audio_ modality, we
firstly use pre-trained audio-aware CNN [ 21, 27 ] to extract its original features _𝐴_ ∈ R _[𝑇]_ [×] _[𝑑]_ _[𝑎]_, then employ an audio encoder which is
composed by an FFN, convolutional and transformer layers on them
follow [ 69 ], textual dependency enhanced features _𝐴_ [′] ∈ R _[𝑇]_ [×] _[𝑑]_ . _For_
_vision_ modality, we extract the original visual features _𝑉_ ∈ R _[𝑇]_ [×] _[𝑑]_ _[𝑣]_
by a pre-trained visual CNN [ 7, 52, 76 ] and further obtain corresponding enhanced features _𝑉_ [′] ∈ R _[𝑇]_ [×] _[𝑑]_ by a visual encoder which
shares the same structure as audio encoder. _For textual query_, we
directly initialize it with GloVe embeddings [ 43 ]. Since the query
may have different semantic alignments with vision and audio, we
further encode it by two separate text encoders which also share
the same structure as the audio encoder and obtain modal-specific
enhanced text features as _𝑄_ _𝑎_ [′] ∈ R _[𝑁]_ [×] _[𝑑]_ and _𝑄_ _𝑣_ [′] ∈ R _[𝑁]_ [×] _[𝑑]_ .
**Vision-text/Audio-Text Fusion.** To highlight the most related
contents between the vision/audio and the given textual query, we
apply context-query attention [ 69 ] on each pair, resulting in fused
features _𝑉_ [ˆ] ∈ R _[𝑇]_ [×] _[𝑑]_ and _𝐴_ [ˆ] ∈ R _[𝑇]_ [×] _[𝑑]_ .


**3.3** **Importance-Aware Multi-modal Fusion**


The importance-aware multi-modal fusion is a multi-granularity
fusion module guided by an audio importance predictor. The predictor is trained to identify and emphasize semantically relevant
audio cues, enabling the model to selectively fuse informative audio
signals with visual features. Guided by the predicted importance
scores, the multi-granularity fusion process effectively filters out
irrelevant or noisy audio content while aggregating meaningful
cross-modal information at multiple temporal levels, thereby enhancing retrieval performance.


_3.3.1_ _Audio Importance Predictor._ The Audio Importance Predictor
(AIP) is a lightweight module designed to dynamically estimate
the relative importance of audio for each video-query pair. Since
ground-truth importance labels are unavailable, we design a lossaware pseudo-importance generator to produce pseudo labels that
serve as supervision signals during training.
_𝐴_ **Structure** ˆ, we first apply attention pooling [ . Given the text-guided visual and audio features 4 ] to obtain their global rep- _𝑉_ [ˆ] and
resentations, denoted as _𝑉_ [ˆ] _𝐺_ ∈ R _[𝑑]_ and _𝐴_ [ˆ] _𝐺_ ∈ R _[𝑑]_ . These global
features capture the overall semantic context of the visual and audio modalities, respectively. Next, we concatenate the two global
features and feed them into a Multi-Layer Perceptron (MLP), which


MM ’25, October 27–31, 2025, Dublin, Ireland. Junan Lin et al.


**Figure 2: The framework of our proposed importance-aware multi-granularity fusion model for video moment retrieval.**



facilitates mutual feature interaction and enables the model to rea
son about the relative importance of audio with respect to the
visual context. The audio importance score _𝑝_ is then predicted as:
_𝑝_ = _𝑆𝑖𝑔𝑚𝑜𝑖𝑑_ ( _𝑀𝐿𝑃_ ([ _𝐴_ [ˆ] _𝐺_ ; _𝑉_ [ˆ] _𝐺_ ])), where _𝑆𝑖𝑔𝑚𝑜𝑖𝑑_ denotes the sigmoid
activation function, and [ ; ] indicate the concatenation operator.
This predicted score _𝑝_ serves as a sample-wise importance weight,
guiding the subsequent multimodal fusion by modulating the contribution of the audio modality.
**Training with the pseudo importance labels.** In order to train
the audio importance predictor, we should construct pseudo labels
as supervisory signals. We draw inspiration from the observation
that neural networks tend to prioritize learning from simpler samples, which typically correspond to lower training losses [ 2 ]. Based
on this, we compare the retrieval losses of the audio and visual
branches for each video-query pair. The modality with a lower loss
is considered to provide more relevant information and is thus assigned a higher pseudo-importance score. Specifically, we compute
a pseudo-importance score _𝑦_ [′] with a softmax-like normalization:



_𝑒_ [L] _𝑟𝑒𝑡_ _[𝑣]_ [/] _[𝛾]_
_𝑦_ =

_𝑒_ [L] _𝑟𝑒𝑡_ _[𝑎]_ [/] _[𝛾]_ + _𝑒_ [L] _𝑟𝑒𝑡_ _[𝑣]_ [/] _[𝛾]_ _[,𝑦]_ [′] [ =]



1 if _𝑦_ ≥ _𝜖_ _𝑚𝑎𝑥_ _,_

 _𝑦_ if _𝜖_ _𝑚𝑎𝑥_ _> 𝑦_ ≥ _𝜖_ _𝑚𝑖𝑛_

0 if _𝑦_ _< 𝜖_ _𝑚𝑖𝑛_ _,_



_,_ (1)



The predicted importance score _𝑝_ serves as a key control parameter in the subsequent multi-granularity fusion stage, guiding
the selective integration of audio and visual features. To prevent
unstable predictions from misguiding early fusion, we initialize the
fusion weight with a neutral value of 0.5 and gradually increase the
influence of the AIP-predicted score as training progresses. This
curriculum-like strategy helps the model build robust multimodal
interactions while mitigating the impact of early-stage noise in the
importance estimation.


_3.3.2_ _Multi-Granularity Fusion._ Given the inherently noisy and
variable nature of the audio modality compared to visual signals,
a simple fusion strategy may not be sufficient to fully exploit
audio-visual complementarity. To address this, we propose a MultiGranularity Fusion (MGF) module that performs hierarchical fusion
from Local-, Event- and Global-perspective, and guided by the dynamically estimated audio importance score.
**Local-Level Visual-Audio Fusion.** As shown in Figure 3(a), to
match the visual and audio context frame-to-clip for fine-level
fusion, we construct symmetric multi-kernel 1D convolutional networks and to deeply perceive the local relationships between video
frames and audio clips, as follows:


_𝑐_ _𝑘_ _[𝑣]_ [=] _[ 𝐶𝑜𝑛𝑣]_ [1] _[𝑑]_ _[𝑘]_ [(] _[𝑉]_ [ ˆ] [)] _[, 𝑐]_ _𝑘_ _[𝑎]_ [=] _[ 𝐶𝑜𝑛𝑣]_ [1] _[𝑑]_ _[𝑘]_ [(][ ˆ] _[𝐴]_ [)] _[,]_ (3)


where _𝑘_ is the kernel size of the convolutional networks. The out
puts are then concatenated and encoded by an MLP layer to map
the dimensionality to _𝑑_, as follows:


ˆ
_𝑉_ _𝑙_ = _𝑀𝐿𝑃_ ([ _𝑐_ 1 _[𝑣]_ [;] _[ ...]_ [;] _[𝑐]_ _𝑛_ _[𝑣]_ [])] _[,]_ [ ˆ] _[𝐴]_ _𝑙_ [=] _[ 𝑀𝐿𝑃]_ [([] _[𝑐]_ _[𝑎]_ 1 [;] _[ ...]_ [;] _[𝑐]_ _𝑛_ _[𝑎]_ [])] _[,]_ (4)


where _𝑉_ [ˆ] _𝑙_ ∈ R _[𝑇]_ [×] _[𝑑]_ and _𝐴_ [ˆ] _𝑙_ ∈ R _[𝑇]_ [×] _[𝑑]_ . From this, we obtain the audio
and visual features after reinforcement at the local level.



where L _𝑟𝑒𝑡_ _[𝑎]_ [and] [ L] _𝑟𝑒𝑡_ _[𝑣]_ [represents the retrieval loss of audio branch]
and visual branch, respectively, _𝛾_ is temperature coefficient. Besides, _𝜖_ _𝑚𝑖𝑛_ is a lower threshold below which the audio modality is
considered uninformative, and its contribution is suppressed. Conversely, values above _𝜖_ _𝑚𝑎𝑥_ indicate that audio plays a dominant
role in retrieval. Finally, we use a binary cross entropy loss to train
AIP as:



L _𝑝_ = _𝐵_ [1]



_𝐵_
∑︁ _𝑦_ _𝑖_ [′] _[𝑙𝑜𝑔𝑝]_ _[𝑖]_ [+ (][1][ −] _[𝑦]_ _𝑖_ [′] [)] _[𝑙𝑜𝑔]_ [(][1][ −] _[𝑝]_ _[𝑖]_ [)] _[,]_ (2)

_𝑖_ =1



where _𝐵_ denotes batch size, and _𝑖_ represents the index of _𝑖_ -th sample.


Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval MM ’25, October 27–31, 2025, Dublin, Ireland.


**Figure 3: Our proposed Multi-Granularity Fusion module: (a) Local-Level Fusion, (b) Event-Level Fusion, (c) Global-Level Fusion.**



Finally, we fuse two features by element-wise addition with
weight _𝑝_ that is derived from the audio importance predictor:


F _𝑙_ = (1 − _𝑝_ ) _𝐿𝑁_ ( _𝑉_ [ˆ] _𝑙_ ) + _𝑝𝐿𝑁_ ( _𝐴_ [ˆ] _𝑙_ ) _,_ (5)


where _𝐿𝑁_ (·) is layer normalization.
**Event-Level Visual-Audio Fusion.** As shown in Figure 3(b), to
match the event-aware semantics between vision and audio for

activity reasoning, our event-level fusion module first employs a
group of slot attention mechanism [ 37 ] to aggregate similar visual/audio clips into multiple events by using a set of learnable
event slots, as follows:


ˆ
_𝐴_ _𝑠_ = _𝑆𝑙𝑜𝑡𝐴𝑡𝑡𝑛_ ( ˆ _𝐴_ ) _,_ ˆ _𝑉_ _𝑠_ = _𝑆𝑙𝑜𝑡𝐴𝑡𝑡𝑛_ ( _𝑉_ ˆ ) _,_ (6)


where _𝐴_ [ˆ] _𝑠_ ∈ R _[𝑒]_ [×] _[𝑑]_ and _𝑉_ [ˆ] _𝑠_ ∈ R _[𝑒]_ [×] _[𝑑]_ indicates that _𝑒_ events are
extracted from the visual/audio sequence. Subsequently, origin
visual/audio features enter the cross-modal transformer layer as
_𝑉_ query and visual/audio events as key and value to obtain ˆ _𝑒_ . Finally, we fuse to obtain visual-audio event aware features _𝐴_ [ˆ] _𝑒_ and F _𝑒_
as same as Equation 5.
**Global-Level Visual-Audio Fusion.** As shown in Figure 3(c),
to match the visual and audio context from a global perspective,
we first encode visual/audio features _𝑉_ [ˆ] and _𝐴_ [ˆ] into global level
representation with attention pooling mechanism [ 4 ]. Then, we
concatenate it with each element of origin _𝑉_ [ˆ] and _𝐴_ [ˆ], and an MLP
layer is used to obtain _𝑉_ [ˆ] _𝑔_ and _𝐴_ [ˆ] _𝑔_ . Finally, we obtain visual-audio
global aware features F _𝑔_ as same as Equation 5.
**Multi-Scale Feature Fusion.** Since the fused features obtained

from different granularities have varying interrelationships, we
adopt a set of Bi-GRUs to re-establish these inter-perceptual relationships by combining the features pairwise. Finally, we concatenate the results and pass them through MLP layers to map the
dimensions back to _𝑑_ -dimension space, obtaining our final visualaudio fused features F .


**3.4** **Cross-modal Knowledge Distillation**


The fusion branch, by jointly modeling audio and visual cues, inherently captures richer and more comprehensive semantic representations. However, in practical applications, audio signals may be
missing, corrupted, or unavailable during inference. To ensure that
unimodal branches retain strong retrieval capabilities under such



conditions, particularly for the visual branch, we introduce a crossmodal knowledge distillation strategy. Specifically, we treat the
fusion branch as a teacher to distill knowledge into the unimodal
branches, particularly the visual branch. This enables the unimodal
branch to inherit modality-complementary cues from the fusion
branch, thereby achieving strong retrieval performance even with
visual-only input. To this end, we minimize the Kullback-Leibler
(KL) divergence between the output distributions of the fusion and
unimodal branches as follows:



where _𝑠_ _[𝑠]_ [/] _[𝑒]_ is start or end logits predicted by the student unimodal
branch, _𝑡_ _[𝑠]_ [/] _[𝑒]_ is start or end logits predicted by the teacher fusion
branch, _𝜏_ is temperature coefficient and _𝜎_ is softmax function. Combining the two unimodal branches, the final KL divergence loss is
the summation of the corresponding losses on the visual L _𝑘𝑙_ _[𝑣]_ [and]
the audio L _[𝑎]_
_𝑘𝑙_ [.]


**3.5** **Model Training**


Following previous works [ 69 ], we exploit the moment predictor
as the retrieval heads to output the start logits and end logits of
the moment, and get the final prediction _𝑃_ _𝑠_ and _𝑃_ _𝑒_ . Take the fusion
branch as an example, the retrieval loss is computed as follows:

L _𝑟𝑒𝑡_ _[𝑓]_ [=] _[ 𝐶𝐸]_ [(] _[𝑃]_ _𝑠_ _[𝑓]_ _[,𝑌]_ _𝑠_ [) +] _[ 𝐶𝐸]_ [(] _[𝑃]_ _𝑒_ _[𝑓]_ _[,𝑌]_ _𝑒_ [)] _[,]_ (8)

where CE denotes cross-entropy loss, _𝑌_ _𝑠_ / _𝑒_ = { _𝑌_ _𝑠_ _[𝑖]_ / _𝑒_ [}] _[𝑖]_ [∈{] [0] _[,]_ [ 1] [}] [ rep-]
resents the supervision where _𝑌_ _𝑠_ / _𝑒_ is set to 1 only at the start/end
point. By applying this loss function to the three branches (visual
branch, audio branch and visual-audio fusion branch), the total
retrieval loss of prediction is:

L _𝑟𝑒𝑡_ = L _𝑟𝑒𝑡_ _[𝑣]_ [+ L] _𝑟𝑒𝑡_ _[𝑎]_ [+ L] _𝑟𝑒𝑡_ _[𝑓]_ _[.]_ (9)


In addition, following [ 30 ], we also introduce saliency loss L _𝑠𝑎𝑙_
to the vision-text fusion features _𝑉_ [ˆ] and audio-text fusion features
_𝐴_ ˆ as well as visual-audio fused features F . This loss widens the
distance between features within and outside of the timestamp.
Finally, the overall training loss is:


L = L _𝑟𝑒𝑡_ + _𝜆_ 1 L _𝑝_ + _𝜆_ 2 L _𝑘𝑙_ + _𝜆_ 3 L _𝑠𝑎𝑙_ _,_ (10)



L _𝑘𝑙_ =



_𝐵_
∑︁ _𝜏_ [2] ( _𝐾𝐿_ ( _𝜎_ ( _𝑠_ _[𝑠]_ / _𝜏_ ) _, 𝜎_ ( _𝑡_ _[𝑠]_ / _𝜏_ ))

_𝑖_ =1


+ _𝐾𝐿_ ( _𝜎_ ( _𝑠_ _[𝑒]_ / _𝜏_ ) _, 𝜎_ ( _𝑡_ _[𝑒]_ / _𝜏_ ))) _,_



(7)


MM ’25, October 27–31, 2025, Dublin, Ireland. Junan Lin et al.


**Table 1: Ablation studies of Audio Importance Predictor (AIP)**
**on Charades-STA.**


**Line ID** **Approach** **R1@3** **R1@5** **R1@7** **mIOU**


#1 Add 74.19 60.97 43.41 55.02

#2 Concat 72.77 59.73 43.20 54.24

#3 Sim 74.33 60.91 43.80 55.12

#4 Attn Entropy 73.76 60.11 43.23 55.00

#5 AIP w/o pseudo-label 73.98 59.74 43.33 54.39



(b) Average audio importance curve predicted by AIP.



#6 AIP **75.18** **61.85** **44.23** **55.62**


where _𝜆_ 1, _𝜆_ 2 and _𝜆_ 3 are the balancing parameters. During inference, we use Maximum Likelihood Estimation (MLE) to obtain the
predicted ( _𝑦_ _[𝑠]_ _,𝑦_ _[𝑒]_ ) with the constraint _𝑦_ _[𝑠]_ ≤ _𝑦_ _[𝑒]_ .
**4** **Experiment**

**4.1** **Dataset**


We conduct our experiments on two video moment retrieval benchmark datasets with audio, _i_ . _e_ ., Charades-STA [ 16 ] and ActivityNet
Captions [ 28 ], as well as Chardes-AudioMatter dataset reconstructed
by ours. Specifically, **Charades-STA** is a dataset about daily indoor activities. There are 12,408 and 3,720 moment annotations for
training and testing, respectively. **ActivityNet Captions** dataset
contains about 20k videos taken from ActivityNet. We follow the
setup in [ 69 ] with 37,421 moment annotations for training, and
17,505 annotations for testing.
Moreover, to further validate the model’s capability in integrating audio modality, we introduce a new dataset, **Charades-**
**AudioMatter**, where audio matters for each test query. By reviewing both the videos and their corresponding audio, we manually select and re-organize 1,196 samples from the test set of Charades-STA
in which the audio provides valuable information. This selection
constitutes a new test set, while the training set of Charades-STA
remained unchanged, see supplementary material for more details.


**4.2** **Evaluation Metrics**


Following the previous works [ 16, 35, 67 ], we adopt “R _𝑛_ @ _𝜇_ ” and
“mIoU” as the evaluation metrics. The “R _𝑛_ @ _𝜇_ ” denotes the percentage of language queries having at least one result whose Intersection over Union (IoU) with ground truth is larger than _𝜇_ ∗ 0 _._ 1 in
top- _𝑛_ retrieved moments. “mIoU” is the average IoU over all testing
samples. In our experiments, we use _𝑛_ = 1 and _𝜇_ ∈{3 _,_ 5 _,_ 7}.


**4.3** **Ablation Study**


**Effectiveness of Audio Importance Predictor.** To demonstrate
the effectiveness of our audio importance predictor, we conducted
ablation studies. In our initial design, we explored multiple approaches for fusing audio and visual modalities. As shown in Table
1, we compared various approaches with our weighted fusion based
on predicted importance of AIP, including direct addtiton (line 1),
concatenation (line 2), and weighted fusion based on cosine similarity between embeddings (line 3) and attention entropy (line
4) calculated by the last attention layer, we also compared AIP
without the supervision of pseudo-label (line 5). Ultimately, our
AIP demonstrated superior performance, which we attribute to our
carefully designed label-supervised module that provides effective



(a) Performance curves when noisy audio
is introduced.



**Figure 4: During inference, as noise in the audio progres-**
**sively increases, the gap between the two curves in (a) widens,**
**suggesting that the IMG model with AIP exhibits greater ro-**
**bustness. Additionally, as we expected, the average audio**
**importance in (b) decreases as noise levels rise.**


**Table 2: Ablation studies of fusion strategy on Charades-STA.**


**Local** **Event** **Global** **R1@3** **R1@5** **R1@7** **mIOU**


✓    -    - 73.07 58.85 40.68 53.67

        - ✓        - 74.84 59.92 41.32 54.83

        -        - ✓ 73.20 57.50 41.64 53.67


✓ ✓    - 74.09 60.08 42.64 55.08

✓    - ✓ 73.88 59.98 43.15 54.87

        - ✓ ✓ 74.28 60.33 42.93 55.47


✓ ✓ ✓ **75.18** **61.85** **44.23** **55.62**


guidance for AIP and ultimately achieve better dynamic multimodal
integration.
**Robustness Analysis on Audio Importance Predictor.** To assess the robustness of AIP, we introduced random gaussian noise
to a subset of the test set audio samples. As shown in Figure 4,
increasing the proportion of noisy audio widened the performance
gap between IMG models with and without AIP. Notably, the performance of IMG without AIP fell below the baseline, while IMG
with AIP exhibited a more gentle decline, proving the AIP’s robustness and highlighting the detrimental impact on performance when
modality importance is ignored under extreme modal imbalance.
**Effectiveness of Fusion Strategies.** We conduct ablation studies to evaluate the visual-audio fusion strategy in Table 2. Here,
each of our proposed fusion methods demonstrates a performance
improvement, highlighting the effectiveness of our fusion strategy. Performance is further enhanced when two feature aspects are
fused, with the fusion of three feature sets outperforming the fusion
of two. These results indicate that features extracted at different
granularities complement each other effectively, facilitating a more
comprehensive fusion of audio information.
**Qualitative Analysis of Fusion Strategies.** This experiment will
attempt to answer why our visual-audio fusion strategies is effective
and can complement each other. The starting point of this structure
is that we want different fusion strategies to focus on different sides
of information, _e.g._, for local-level, our expectation is to find subtle
clues. In Figure 5, we categorized samples of Charades-STA into 5
equal number of categories based on the moment-to-video ratios,
and we find that IMG with only local fusion show a stronger ability


Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval MM ’25, October 27–31, 2025, Dublin, Ireland.


**Table 4: Effectiveness of audio integration for video moment**
**retrieval. “** ↑ **” denotes performance improvement when audio**
**modality is introduced.**


**Charades-STA** **ActivityNet Captions**
**Line ID** **Method**


**R1@7** **mIOU** **R1@7** **mIOU**



**Figure 5: Performance of different granularity fusion strate-**
**gies at different normalized moment-to-video ratios.**


**Table 3: Ablation studies on Charades-STA under conditions**
**of using unimodal branch during inference. “CKD” denotes**
**Cross-modal Knowledge Distillation.**


**Method** **Branch** **R1@3** **R1@5** **R1@7** **mIOU**


Fusion 75.18 61.85 44.23 55.62

IMG Visual 74 _._ 84 0 _._ 34↓ 60 _._ 95 0 _._ 90↓ 43 _._ 44 0 _._ 79↓ 54 _._ 97 0 _._ 65↓
Audio 60 _._ 11 15 _._ 07↓ 45 _._ 86 15 _._ 99↓ 29 _._ 35 14 _._ 88↓ 42 _._ 85 12 _._ 77↓


Fusion 74.09 61.03 43.33 55.31

IMG w/o CKD Visual 72 _._ 49 1 _._ 60↓ 56 _._ 92 4 _._ 11↓ 39 _._ 58 3 _._ 75↓ 53 _._ 12 2 _._ 19↓
Audio 58 _._ 04 16 _._ 05↓ 43 _._ 70 17 _._ 33↓ 25 _._ 48 17 _._ 85↓ 40 _._ 68 14 _._ 63↓


to handle smaller ratio ( _i.e._, more subtle moments), whereas IMG
with only event fusion enhanced performance of moderate ratio,
and for IMG with only global fusion are suited to deal with the
case of larger ration. The different performances between different
granularities lay the foundation for multi-granularity fusion and
finally, our IMG achieves a good balance.
**Inference with Unimodal Branch.** In real-world scenarios, the
audio modality may sometimes be irrelevant or unavailable, such as
in surveillance footage. In such cases, the visual branch of IMG can
still be employed, and cross-modal knowledge distillation strategy
is expected to mitigate potential negative impacts. As shown in
Table 3, IMG with CKD exhibits minimal performance degradation
which confirmed that CKD can largely overcome the negative effects and IMG still demonstrates excellent performance when audio
modality is missing during inference. Additionally, we extend our
investigation to the inference of audio branch, it turns out that
audio branch inference alone has a significant degradation in performance, so we argue that audio can only be an auxiliary modality,
and comparing the models with and without the CKD, we find that
CKD also improves the performance of audio branch.
**Effectiveness and Flexibility of Audio Integration.** As shown
in Table 4, we compare our baseline model which trained using only
visual branch (line 1) against our audio-integrated model (line 2),
which demonstrates the effectiveness of the incorporation of audio
modality. To further validate the effectiveness and flexibility of our
framework, we incorporate our IAMF module (Section 3.3) as a
plug-in across advanced models (lines 3-6). Results indicate that all
models achieve improvements across all metrics, with particularly
notable gains on the challenging R1@7 metric, demonstrating that
our approach effectively extracts meaningful information from
audio modalities.



#1 Baseline 39.52 52.76 26.18 43.21

#2 Ours 44 _._ 23 4 _._ 71↑ 55 _._ 62 2 _._ 86↑ 29 _._ 47 3 _._ 29↑ 45 _._ 19 1 _._ 98↑
#3 EMB [22] 39.25 53.09 26.07 45.59

#4 EMB + Ours 43 _._ 15 3 _._ 90↑ 54 _._ 53 1 _._ 44↑ 28 _._ 44 2 _._ 37↑ 46 _._ 69 1 _._ 10↑
#5 EAMAT [62] 41.96 54.45 25.77 42.19

#6 EAMAT + Ours 44 _._ 08 2 _._ 12↑ 55 _._ 59 1 _._ 14↑ 27 _._ 38 1 _._ 61↑ 43 _._ 27 1 _._ 08↑


**4.4** **Performance Comparison**
In Table 5, we evaluate our IMG on Charades-STA and ActivityNet
Captions and compare it with existing audio-incorporated VMR
methods. Furthermore, we list the results for the audio-incorporated
method when trained without audio. On Charades-STA and Ac
tivityNet Captions, our IMG achieves the best performance on all
metrics. By comparing with the visual branch, we find that introducing audio greatly improves sample training which demonstrates
that audio modality can play an important role in assisting VMR.
Moreover, our proposed methodology achieves markedly superior
performance gains compared to existing approaches, which substantiates our methodological advantage.
In Table 6, we evaluate IMG on Charades-STA, comparing it
with state-of-the-art VMR methods that employ visual language
models as backbones. IMG with InternVideo2 [ 55 ] achieves the
best performance across all metrics when the audio modality is
incorporated. This highlights not only the generalization strength
of our method under strong backbone settings, but also the critical
role of audio cues in improving retrieval performance.
To highlight that IMG effectively mines audio modal information, we conducted experiments on Charades-AudioMatter where
the audio data are more consistent and reliable. We compare our
method with open-source methods that exhibit competitive performance on Charades-STA. As presented in Table 7, our IMG achieves
state-of-the-art performance, particularly on R1@7, thereby establishing a substantial lead over all other compared models. This
result underscores the effectiveness of our model in extracting
and utilizing audio modality and compare to ADPN, IMG exhibits
superior proficiency in the integration of audio.


**4.5** **Qualitative Analysis**
As shown in Figure 6(a), we observe that the action "sneeze" is
not clearly visible, leading to inaccurate predictions. In contrast,
the audio prominently captures the action, with an AIP-predicted
importance of 0.587, which helps correct the error in the fusion
branch. In Figure 6(b), the action "sits" lacks distinct acoustic semantics, causing inaccurate inferences in the audio branch. AIP
assigns an importance score of only 0.178, thereby reducing the
fusion branch’s reliance on audio.

We also perform qualitative analysis on Charades-AudioMatter,
comparing with methods that do not introduce audio. For Figure 7(a), the window is partially obscured by curtains, which significantly increases the difficulty of visual-only retrieval for the action


MM ’25, October 27–31, 2025, Dublin, Ireland. Junan Lin et al.


**Table 5: Comparison with audio-incorporated methods on Charades-STA and ActivityNet Captions. We use I3D [** **7** **] as vision**
**backbone with GloVe [43] embeddings.**


**Charades-STA** **ActivityNet Captions**
**Method** **Audio**


**R1@3** **R1@5** **R1@7** **mIOU** **R1@3** **R1@5** **R1@7** **mIOU**


UMT [36] ✓       - 48.31 29.25       -       -       -       -       
PMI-LOC w/o audio [9]       - 56.84 41.29 20.11       - 60.16 39.16 18.02       
PMI-LOC [9] ✓ 58 _._ 08 1 _._ 24↑ 42 _._ 63 1 _._ 34↑ 21 _._ 32 1 _._ 21↑       - 61 _._ 22 1 _._ 06↑ 40 _._ 07 0 _._ 91↑ 18 _._ 29 0 _._ 27↑       QD-DETR w/o audio [41]       -       - 52.77 31.13       -       -       -       -       
QD-DETR [41] ✓       - 55 _._ 51 2 _._ 74↑ 34 _._ 17 3 _._ 04↑       -       -       -       -       -       ADPN w/o audio [8]       - 70.35 55.32 37.47 51.13 55.72 39.56 25.20 41.55

ADPN [8] ✓ 71 _._ 99 1 _._ 64↑ 57 _._ 69 2 _._ 37↑ 41 _._ 10 3 _._ 63↑ 52 _._ 86 1 _._ 73↑ 57 _._ 16 1 _._ 44↑ 41 _._ 40 1 _._ 84↑ 26 _._ 31 1 _._ 11↑ 42 _._ 31 0 _._ 76↑

**IMG w/o audio**       - 72.37 56.34 39.52 52.76 59.19 41.51 26.18 43.21

**IMG** ✓ **75.18** 2 _._ 81↑ **61.85** 5 _._ 51↑ **44.23** 4 _._ 71↑ **55.62** 2 _._ 86↑ **61.50** 2 _._ 31↑ **45.06** 3 _._ 55↑ **29.47** 3 _._ 29↑ **45.19** 1 _._ 98↑


**Table 6: Comparison with state-of-the-art methods on**
**Charades-STA. We compare methods of using visual lan-**
**guage models as backbone. “CLIP+SF” refers to SlowFast [** **14** **]**
**combined with CLIP [** **44** **], “IV2” denotes InternVideo2 [** **55** **].**


**Method** **backbone** **R1@3** **R1@5** **R1@7** **mIOU**



UnLoc-L [61] CLIP - **60.80** 38.40 
Moment-DETR [30] CLIP+SF - 55.65 34.17 
BAM-DETR [29] CLIP+SF 72.93 59.95 39.38 52.33

QD-DETR [41] CLIP+SF - 57.31 32.55 
TR-DETR [48] CLIP+SF - 57.61 33.52 
UniVTG [32] CLIP+SF 70.81 58.01 35.65 50.10

FlashVTG [5] CLIP+SF - 60.11 38.01 

**IMG w/o audio** CLIP+SF 70.25 54.12 37.72 51.65

**IMG** CLIP+SF **74.44** 3 _._ 38↑ 59 _._ 76 5 _._ 64↑ **42.93** 5 _._ 21↑ **55.03** 3 _._ 38↑


InternVideo2 [55] IV2 79.70 70.03 48.95 58.79

FlashVTG [5] IV2 - 70.32 49.87 
SG-DETR [19] IV2 - 70.20 49.50 59.10


**IMG w/o audio** IV2 78.58 66.08 48.69 58.46

**IMG** IV2 **82.02** 3 _._ 44↑ **70.81** 4 _._ 73↑ **54.33** 5 _._ 64↑ **62.25** 3 _._ 79↑


**Table 7: Performance comparison on Charades-AudioMatter.**
**All methods utilize I3D [7] backbone.**


**Method** **Audio** **R1@3** **R1@5** **R1@7** **mIOU**


SeqPAN [68]  - 79.30 67.17 48.96 58.74

EAMAT [62]  - 78.30 68.25 48.88 58.90

EMB [22]  - 77.81 67.00 47.96 58.66

ADPN w/o audio [8]  - 77.89 64.42 44.64 56.98

ADPN [8] ✓ 78 _._ 65 0 _._ 76↑ 66 _._ 75 2 _._ 33↑ 49 _._ 71 5 _._ 07↑ 59 _._ 85 2 _._ 87↑

**IMG w/o audio**  - 77.89 65.92 47.58 58.35

**IMG** ✓ **82.74** 4 _._ 85↑ **71.93** 6 _._ 01↑ **54.27** 6 _._ 69↑ **62.76** 4 _._ 41↑


"closes the window". EMB, EAMAT, SeqPAN, and IMG without
audio failed to retrieve accurately as they relied solely on vision. In
contrast, IMG leveraged the acoustic semantics, allowing for more
accurate retrieval. For Figure 7(b), visual-only retrieval is also challenging due to the subtle movements associated with "laugh" and
minimal scene variation. However, the prominent acoustic signal
of "laugh" enabled IMG to effectively pinpoint the corresponding
timestamp.



**Figure 6: Two samples were selected from Charades-STA.**


**Figure 7: Two samples were selected from Charades-**
**AudioMatter. (a) appeared an occlusion interfering with the**
**visual field, while (b) depicted a visually insignificant action.**


**5** **Conclusion**

In this paper, we propose a novel Importance-aware Multi-Granularity
fusion model (IMG) to handle the flexible audio-vision-text reasoning for the VMR task. To explore the audio’s uncertainty, we propose
an audio importance predictor to utilize the retrieval loss of the
model to generate dynamic pseudo-labels for supervision and dynamically assign weights to different samples of audio to provide
better audio-context guidance. We also propose a multi-granularity
visual-audio fusion network to fully fuse audio and visual modality
from local- to event- and global-level for complementary learning. A
new dataset Charades-AudioMatter is further introduced to validate

the model’s capability in integrating audio modality. Experiments
have proven the effectiveness of our proposed approach.


Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval MM ’25, October 27–31, 2025, Dublin, Ireland.



**Acknowledgments**


This work was supported by the Pioneer and Leading Goose R&D
Program of Zhejiang (No. 2024C01110), National Natural Science
Foundation of China (No. 62472385), Young Elite Scientists Sponsorship Program by China Association for Science and Technology (No.
2022QNRC001), Public Welfare Technology Research Project of Zhejiang Province (No. LGF21F020010), Fundamental Research Funds
for the Provincial Universities of Zhejiang (No. FR2402ZD) and
Zhejiang Provincial High-Level Talent Special Support Program.


**References**


[1] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell,
and Bryan Russell. 2017. Localizing moments in video with natural language. In
_Proceedings of the IEEE international conference on computer vision_ . 5803–5812.

[2] Devansh Arpit, Stanisław Jastrzębski, Nicolas Ballas, David Krueger, Emmanuel
Bengio, Maxinder S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville,
Yoshua Bengio, et al . 2017. A closer look at memorization in deep networks. In
_International conference on machine learning_ . PMLR, 233–242.

[3] Anurag Bagchi, Jazib Mahmood, Dolton Fernandes, and Ravi Kiran Sarvadevabhatla. 2021. Hear me out: Fusional approaches for audio augmented temporal
action localization. _arXiv preprint arXiv:2106.14118_ (2021).

[4] Dzmitry Bahdanau. 2014. Neural machine translation by jointly learning to align
and translate. _arXiv preprint arXiv:1409.0473_ (2014).

[5] Zhuo Cao, Bingqing Zhang, Heming Du, Xin Yu, Xue Li, and Sen Wang. 2024.
FlashVTG: Feature Layering and Adaptive Score Handling Network for Video
Temporal Grounding. _arXiv preprint arXiv:2412.13441_ (2024).

[6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 2020. End-to-end object detection with
transformers. In _European conference on computer vision_ . Springer, 213–229.

[7] Joao Carreira and Andrew Zisserman. 2017. Quo vadis, action recognition? a new
model and the kinetics dataset. In _proceedings of the IEEE Conference on Computer_
_Vision and Pattern Recognition_ . 6299–6308.

[8] Houlun Chen, Xin Wang, Xiaohan Lan, Hong Chen, Xuguang Duan, Jia Jia,
and Wenwu Zhu. 2023. Curriculum-listener: Consistency-and complementarityaware audio-enhanced temporal sentence grounding. In _Proceedings of the 31st_
_ACM International Conference on Multimedia_ . 3117–3128.

[9] Shaoxiang Chen, Wenhao Jiang, Wei Liu, and Yu-Gang Jiang. 2020. Learning
modality interaction for temporal sentence localization and event captioning in
videos. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK,_
_August 23–28, 2020, Proceedings, Part IV 16_ . Springer, 333–351.

[10] Xuelian Cheng, Yiran Zhong, Yuchao Dai, Pan Ji, and Hongdong Li. 2019. Noiseaware unsupervised deep lidar-stereo fusion. In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_ . 6339–6348.

[11] Jianfeng Dong, Xirong Li, Chaoxi Xu, Xun Yang, Gang Yang, Xun Wang, and
Meng Wang. 2022. Dual encoding for video retrieval by text. _IEEE Transactions_
_on Pattern Analysis and Machine Intelligence_ 44, 8 (2022), 4065–4080.

[12] Jianfeng Dong, Xiaoman Peng, Daizong Liu, Xiaoye Qu, Xun Yang, Cuizhu Bao,
and Meng Wang. 2024. Temporal sentence grounding with relevance feedback in
videos. _Advances in Neural Information Processing Systems_ 37 (2024), 43107–43132.

[13] Jianfeng Dong, Minsong Zhang, Zheng Zhang, Xianke Chen, Daizong Liu, Xiaoye
Qu, Xun Wang, and Baolong Liu. 2023. Dual learning with dynamic knowledge
distillation for partially relevant video retrieval. In _Proceedings of the IEEE/CVF_
_International Conference on Computer Vision_ . 11302–11312.

[14] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. 2019. Slowfast networks for video recognition. In _Proceedings of the IEEE/CVF international_
_conference on computer vision_ . 6202–6211.

[15] Jie Fu, Junyu Gao, Bing-Kun Bao, and Changsheng Xu. 2023. Multimodal
imbalance-aware gradient modulation for weakly-supervised audio-visual video
parsing. _IEEE Transactions on Circuits and Systems for Video Technology_ (2023).

[16] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. 2017. Tall: Temporal
activity localization via language query. In _Proceedings of the IEEE international_
_conference on computer vision_ . 5267–5275.

[17] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence,
R Channing Moore, Manoj Plakal, and Marvin Ritter. 2017. Audio set: An ontology
and human-labeled dataset for audio events. In _2017 IEEE international conference_
_on acoustics, speech and signal processing (ICASSP)_ . IEEE, 776–780.

[18] Mandar Gogate, Kia Dashtipour, Ahsan Adeel, and Amir Hussain. 2020.
CochleaNet: A robust language-independent audio-visual model for real-time
speech enhancement. _Information Fusion_ 63 (2020), 273–285.

[19] Aleksandr Gordeev, Vladimir Dokholyan, Irina Tolstykh, and Maksim Kuprashevich. 2024. Saliency-guided detr for moment retrieval and highlight detection.
_arXiv preprint arXiv:2410.01615_ (2024).




[20] Yongxin Guo, Jingyu Liu, Mingda Li, Xiaoying Tang, Xi Chen, and Bo Zhao. 2024.
VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced
Video Temporal Grounding. _arXiv preprint arXiv:2405.13382_ (2024).

[21] Shawn Hershey, Sourish Chaudhuri, Daniel PW Ellis, Jort F Gemmeke, Aren
Jansen, R Channing Moore, Manoj Plakal, Devin Platt, Rif A Saurous, Bryan
Seybold, et al . 2017. CNN architectures for large-scale audio classification. In
_2017 ieee international conference on acoustics, speech and signal processing (icassp)_ .
IEEE, 131–135.

[22] Jiabo Huang, Hailin Jin, Shaogang Gong, and Yang Liu. 2022. Video activity
localisation with uncertainties in temporal boundary. In _European Conference on_
_Computer Vision_ . Springer, 724–740.

[23] Zhenyu Huang, Guocheng Niu, Xiao Liu, Wenbiao Ding, Xinyan Xiao, Hua Wu,
and Xi Peng. 2021. Learning with noisy correspondence for cross-modal matching.
_Advances in Neural Information Processing Systems_ 34 (2021), 29406–29419.

[24] Jinhyun Jang, Jungin Park, Jin Kim, Hyeongjun Kwon, and Kwanghoon Sohn.
2023. Knowing where to focus: Event-aware transformer for video grounding. In
_Proceedings of the IEEE/CVF International Conference on Computer Vision_ . 13846–
13856.

[25] Xun Jiang, Xing Xu, Zhiguo Chen, Jingran Zhang, Jingkuan Song, Fumin Shen,
Huimin Lu, and Heng Tao Shen. 2022. Dhhn: Dual hierarchical hybrid network
for weakly-supervised audio-visual video parsing. In _Proceedings of the 30th ACM_
_International Conference on Multimedia_ . 719–727.

[26] Yiyang Jiang, Wengyu Zhang, Xulu Zhang, Xiao-Yong Wei, Chang Wen Chen,
and Qing Li. 2024. Prior knowledge integration via llm encoding and pseudo
event regulation for video moment retrieval. In _Proceedings of the 32nd ACM_
_International Conference on Multimedia_ . 7249–7258.

[27] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D
Plumbley. 2020. Panns: Large-scale pretrained audio neural networks for audio
pattern recognition. _IEEE/ACM Transactions on Audio, Speech, and Language_
_Processing_ 28 (2020), 2880–2894.

[28] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles.
2017. Dense-captioning events in videos. In _Proceedings of the IEEE international_
_conference on computer vision_ . 706–715.

[29] Pilhyeon Lee and Hyeran Byun. 2024. Bam-detr: Boundary-aligned moment
detection transformer for temporal sentence grounding in videos. In _European_
_Conference on Computer Vision_ . Springer, 220–238.

[30] Jie Lei, Tamara L Berg, and Mohit Bansal. 2021. Detecting moments and highlights
in videos via natural language queries. _Advances in Neural Information Processing_
_Systems_ 34 (2021), 11846–11858.

[31] Hao Li, Jingkuan Song, Lianli Gao, Xiaosu Zhu, and Hengtao Shen. 2024.
Prototype-based aleatoric uncertainty quantification for cross-modal retrieval.
_Advances in Neural Information Processing Systems_ 36 (2024).

[32] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei
Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. 2023. Univtg: Towards unified video-language temporal grounding. In _Proceedings of the IEEE/CVF_
_International Conference on Computer Vision_ . 2794–2804.

[33] Yijie Lin, Jie Zhang, Zhenyu Huang, Jia Liu, Zujie Wen, and Xi Peng. 2024. Multigranularity correspondence learning from long-term noisy videos. _arXiv preprint_
_arXiv:2401.16702_ (2024).

[34] Daizong Liu, Xiaoye Qu, Jianfeng Dong, and Pan Zhou. 2021. Adaptive proposal
generation network for temporal sentence localization in videos. _arXiv preprint_
_arXiv:2109.06398_ (2021).

[35] Meng Liu, Xiang Wang, Liqiang Nie, Xiangnan He, Baoquan Chen, and Tat-Seng
Chua. 2018. Attentive moment retrieval in videos. In _The 41st international ACM_
_SIGIR conference on research & development in information retrieval_ . 15–24.

[36] Ye Liu, Siyuan Li, Yang Wu, Chang-Wen Chen, Ying Shan, and Xiaohu Qie. 2022.
Umt: Unified multi-modal transformers for joint video moment retrieval and
highlight detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision_
_and Pattern Recognition_ . 3042–3051.

[37] Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf.
2020. Object-centric learning with slot attention. _Advances in neural information_
_processing systems_ 33 (2020), 11525–11538.

[38] Ilya Loshchilov, Frank Hutter, et al . 2017. Fixing weight decay regularization in
adam. _arXiv preprint arXiv:1711.05101_ 5 (2017).

[39] Otniel-Bogdan Mercea, Thomas Hummel, A Sophia Koepke, and Zeynep Akata.
2022. Temporal and cross-modal attention for audio-visual zero-shot learning. In
_European Conference on Computer Vision_ . Springer, 488–505.

[40] Otniel-Bogdan Mercea, Lukas Riesch, A Koepke, and Zeynep Akata. 2022. Audiovisual generalised zero-shot learning with cross-modal attention and language. In
_Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ .
10553–10563.

[41] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo.
2023. Query-dependent video representation for moment retrieval and highlight
detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_ . 23023–23033.

[42] Wenwen Pan, Haonan Shi, Zhou Zhao, Jieming Zhu, Xiuqiang He, Zhigeng Pan,
Lianli Gao, Jun Yu, Fei Wu, and Qi Tian. 2022. Wnet: Audio-guided video object


MM ’25, October 27–31, 2025, Dublin, Ireland. Junan Lin et al.



segmentation via wavelet-based cross-modal denoising networks. In _Proceedings_
_of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 1320–
1331.

[43] Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove:
Global vectors for word representation. In _Proceedings of the 2014 conference on_
_empirical methods in natural language processing (EMNLP)_ . 1532–1543.

[44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al . 2021. Learning transferable visual models from natural language supervision.
In _International conference on machine learning_ . PmLR, 8748–8763.

[45] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. 2024. Timechat: A
time-sensitive multimodal large language model for long video understanding. In
_Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ .
14313–14323.

[46] Yazhou Ren, Xinyue Chen, Jie Xu, Jingyu Pu, Yonghao Huang, Xiaorong Pu, Ce
Zhu, Xiaofeng Zhu, Zhifeng Hao, and Lifang He. 2024. A novel federated multiview clustering method for unaligned and incomplete data fusion. _Information_
_Fusion_ 108 (2024), 102357.

[47] Xingyu Shen, Xiang Zhang, Xun Yang, Yibing Zhan, Long Lan, Jianfeng Dong, and
Hongzhou Wu. 2023. Semantics-Enriched Cross-Modal Alignment for ComplexQuery Video Moment Retrieval. In _Proceedings of the 31st ACM International_
_Conference on Multimedia_ . 4109–4118.

[48] Hao Sun, Mingyao Zhou, Wenjing Chen, and Wei Xie. 2024. Tr-detr: Taskreciprocal transformer for joint moment retrieval and highlight detection. In
_Proceedings of the AAAI Conference on Artificial Intelligence_, Vol. 38. 4998–5007.

[49] Xin Sun, Xuan Wang, Jialin Gao, Qiong Liu, and Xi Zhou. 2022. You need to
read again: Multi-granularity perception network for moment retrieval in videos.
In _Proceedings of the 45th International ACM SIGIR Conference on Research and_
_Development in Information Retrieval_ . 1022–1032.

[50] Mani Kumar Tellamekala, Shahin Amiriparian, Björn W Schuller, Elisabeth André,
Timo Giesbrecht, and Michel Valstar. 2023. COLD fusion: Calibrated and ordinal
latent distribution fusion for uncertainty-aware multimodal emotion recognition.
_IEEE Transactions on Pattern Analysis and Machine Intelligence_ (2023).

[51] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al . 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv_
_preprint arXiv:2307.09288_ (2023).

[52] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri.
2015. Learning spatiotemporal features with 3d convolutional networks. In
_Proceedings of the IEEE international conference on computer vision_ . 4489–4497.

[53] Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan
Wang, Weifeng Ge, and Lifu Huang. 2024. Grounded-videollm: Sharpening finegrained temporal grounding in video large language models. _arXiv preprint_
_arXiv:2410.03290_ (2024).

[54] Jing Wang, Aixin Sun, Hao Zhang, and Xiaoli Li. 2023. MS-DETR: Natural
Language Video Localization with Sampling Moment-Moment Interaction. _arXiv_
_preprint arXiv:2305.18969_ (2023).

[55] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei,
Rongkun Zheng, Zun Wang, Yansong Shi, et al . 2024. Internvideo2: Scaling
foundation models for multimodal video understanding. In _European Conference_
_on Computer Vision_ . Springer, 396–416.

[56] Yueqian Wang, Xiaojun Meng, Jianxin Liang, Yuxuan Wang, Qun Liu, and
Dongyan Zhao. 2024. Hawkeye: Training video-text llms for grounding text in
videos. _arXiv preprint arXiv:2403.10228_ (2024).

[57] Zhenzhi Wang, Limin Wang, Tao Wu, Tianhao Li, and Gangshan Wu. 2022. Negative sample matters: A renaissance of metric learning for temporal grounding. In
_Proceedings of the AAAI Conference on Artificial Intelligence_, Vol. 36. 2613–2623.

[58] Shaoning Xiao, Long Chen, Songyang Zhang, Wei Ji, Jian Shao, Lu Ye, and Jun
Xiao. 2021. Boundary proposal network for two-stage natural language video
localization. In _Proceedings of the AAAI Conference on Artificial Intelligence_, Vol. 35.
2986–2994.

[59] Bo Xu, Shizhou Huang, Ming Du, Hongya Wang, Hui Song, Chaofeng Sha,
and Yanghua Xiao. 2022. Different data, different modalities! reinforced data
splitting for effective multimodal information extraction from social media posts.
In _Proceedings of the 29th International Conference on Computational Linguistics_ .
1855–1864.

[60] Ruize Xu, Ruoxuan Feng, Shi-Xiong Zhang, and Di Hu. 2023. Mmcosine: Multimodal cosine loss towards balanced audio-visual fine-grained learning. In _ICASSP_
_2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing_
_(ICASSP)_ . IEEE, 1–5.

[61] Shen Yan, Xuehan Xiong, Arsha Nagrani, Anurag Arnab, Zhonghao Wang, Weina
Ge, David Ross, and Cordelia Schmid. 2023. Unloc: A unified framework for
video localization tasks. In _Proceedings of the IEEE/CVF International Conference_
_on Computer Vision_ . 13623–13633.

[62] Shuo Yang and Xinxiao Wu. 2022. Entity-aware and motion-aware transformers
for language-driven action localization in videos. _arXiv preprint arXiv:2205.05854_
(2022).




[63] Shuo Yang, Xinxiao Wu, Zirui Shang, and Jiebo Luo. 2024. Dynamic Pathway for
Query-Aware Feature Learning in Language-Driven Action Localization. _IEEE_
_Transactions on Multimedia_ (2024).

[64] Xun Yang, Tianyu Chang, Tianzhu Zhang, Shanshan Wang, Richang Hong, and
Meng Wang. 2024. Learning hierarchical visual transformation for domain
generalizable visual matching and recognition. _International Journal of Computer_
_Vision_ 132, 11 (2024), 4823–4849.

[65] Xun Yang, Jianming Zeng, Dan Guo, Shanshan Wang, Jianfeng Dong, and Meng
Wang. 2024. Robust video question answering via contrastive cross-modality
representation learning. _Science China Information Sciences_ 67, 10 (2024), 202104.

[66] Wenfang Yao, Kejing Yin, William K Cheung, Jia Liu, and Jing Qin. 2024. DrFuse:
Learning Disentangled Representation for Clinical Multi-Modal Fusion with
Missing Modality and Modal Inconsistency. In _Proceedings of the AAAI Conference_
_on Artificial Intelligence_, Vol. 38. 16416–16424.

[67] Yitian Yuan, Tao Mei, and Wenwu Zhu. 2019. To find where you talk: Temporal sentence localization in video with attention based location regression. In
_Proceedings of the AAAI Conference on Artificial Intelligence_, Vol. 33. 9159–9166.

[68] Hao Zhang, Aixin Sun, Wei Jing, Liangli Zhen, Joey Tianyi Zhou, and Rick
Siow Mong Goh. 2021. Parallel attention network with sequence matching for
video grounding. _arXiv preprint arXiv:2105.08481_ (2021).

[69] Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. 2020. Span-based
localizing network for natural language video localization. _arXiv preprint_
_arXiv:2004.13931_ (2020).

[70] Long Zhang, Peipei Song, Zhangling Duan, Shuo Wang, Xiaojun Chang, and Xun
Yang. 2025. Video corpus moment retrieval with query-specific context learning
and progressive localization. _IEEE Transactions on Circuits and Systems for Video_
_Technology_ (2025).

[71] Qingyang Zhang, Yake Wei, Zongbo Han, Huazhu Fu, Xi Peng, Cheng Deng,
Qinghua Hu, Cai Xu, Jie Wen, Di Hu, et al . 2024. Multimodal fusion on low-quality
data: A comprehensive survey. _arXiv preprint arXiv:2404.18947_ (2024).

[72] Songyang Zhang, Houwen Peng, Jianlong Fu, Yijuan Lu, and Jiebo Luo. 2021.
Multi-scale 2d temporal adjacency networks for moment localization with natural
language. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 44, 12
(2021), 9073–9087.

[73] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. 2020. Learning 2d
temporal adjacent networks for moment localization with natural language. In
_Proceedings of the AAAI Conference on Artificial Intelligence_, Vol. 34. 12870–12877.

[74] Yimeng Zhang, Xin Chen, Jinghan Jia, Sijia Liu, and Ke Ding. 2023. Textvisual prompting for efficient 2d temporal video grounding. In _Proceedings of the_
_IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 14794–14804.

[75] Qi Zheng, Jianfeng Dong, Xiaoye Qu, Xun Yang, Yabing Wang, Pan Zhou, Baolong
Liu, and Xun Wang. 2023. Progressive localization networks for language-based
moment localization. _ACM Transactions on Multimedia Computing, Communica-_
_tions and Applications_ 19, 2 (2023), 1–21.

[76] Sheng Zhou, Junbin Xiao, Qingyun Li, Yicong Li, Xun Yang, Dan Guo, Meng
Wang, Tat-Seng Chua, and Angela Yao. 2025. Egotextvqa: Towards egocentric
scene-text aware video question answering. In _Proceedings of the Computer Vision_
_and Pattern Recognition Conference_ . 3363–3373.

[77] Yipin Zhou and Ser-Nam Lim. 2021. Joint audio-visual deepfake detection. In
_Proceedings of the IEEE/CVF International Conference on Computer Vision_ . 14800–
14809.


Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval MM ’25, October 27–31, 2025, Dublin, Ireland.


We report more technical details and more experimental results
which are not included in the paper due to space limit:


  - Detailed analysis of dataset Charades-AudioMatter including:

**–**
Dataset construction (Section A.1).

**–**
Statistical analysis (Section A.2).

  - Experiments on ActivityNet Captions including:



**–**
Ablation studies on fusion strategies (Section B.1).

**–**
Ablation studies on additional model structures (Section B.4).

**–**
Qualitative analysis (Section B.3).

  - Additional experiments including:


**–**
Experiments on hyperparameters (Section C.1) including
threshold _𝜖_ _𝑚𝑖𝑛_, temperature _𝛾_ and others.

**–**
Experiments on efficiency (Section C.2), Event-Level Fusion module (Section C.3), weak supervision (Section C.4),
failed AIP (Section C.6) and audio importance distribution
(Section C.5).

   - Implement details (Section D).


**A** **Charades-AudioMatter Dataset Construction**

**A.1** **Dataset Construction**


In this section, we introduce the dataset Charades-AudioMatter
in detail. To ensure the high quality of the Charades-AudioMatter
dataset and the reliability of experimental results, the dataset construction underwent a rigorous screening process. The dataset was
annotated by six postgraduate students with experience in multimodal learning. Each instance was independently labeled by two
annotators, with disagreements adjudicated by a third annotator.
The validity and relevance of the audio data were carefully evaluated through the following processes:
**Validity of the Audio.** Given a sample, the audio modality is first
subjected to a validity assessment. Samples containing significant
background noise or lacking any sound were excluded, as such audio lacks meaningful information and cannot contribute effectively
to VMR. This process is employed for rapid preliminary screening.
**Correlation between Audio and Query Text.** After the initial
screening, each sample was manually evaluated through a combination of audio and visual to determine whether the query text
was dependent on the audio. For instance, query describing static
actions ( _e.g._, “sitting,” “looking,” “standing”) were almost excluded
because the audio does not provide meaningful cues for these actions. Similarly, for actions typically associated with audio cues ( _e.g._,
“laughing,” “closing the door”), if the audio in a specific instance
lacked sound or the sound was too faint, the sample was marked as
invalid and excluded. This step ensured the relevance of audio to
the text by integrating auditory judgment with semantic analysis
of the text.

**Temporal Alignment of Audio and Video.** After the screening
steps above, we evaluate the temporal alignment between visual
and audio modalities. Specifically, manual timestamp annotation
was performed solely based on the audio and query text, followed
by IoU computation with the ground truth. Samples exhibiting an
IoU score below 0.3 are discarded. This process ensures the validation of temporal consistency between audio and video modalities,
effectively filtering out severely misaligned samples (e.g., those
with significant audio delays or excessive offsets).



**Figure 8: For the same activity of “open the door", we reviewed**
**and listened specific samples, ultimately choosing the left**
**one where the sound is clearly communicated, discarding**
**the other where there is almost no corresponding sound.**


**Table 8: Statistical analysis of the dataset Charades-**
**AudioMatter. We compare the selected activity categories**
**with the unselected one.**

|Selected Activity Count|Unselected Activity Count|
|---|---|
|open (door/cabinet/...)<br>241<br>close (door/closet/...)<br>150<br>put (bag/grpceries/...)<br>138<br>run<br>90<br>turn on/of (light/tv/...)<br>89<br>throw (broom/shoes/...)<br>66<br>take (vacuum/food/...)<br>56<br>laugh<br>52<br>eat<br>41<br>wash (hand/glass/...)<br>29<br>drink<br>28<br>walk<br>25<br>cook<br>22<br>pour (water/cofee/...)<br>16<br>sit down<br>13<br>talk<br>10|sit (on bed/chair/...)<br>218<br>hold<br>147<br>(un)dress<br>111<br>look<br>85<br>stand<br>59<br>smile<br>55<br>watch<br>48<br>read<br>32<br>awake<br>38<br>take a picture<br>30<br>play (phone/camera/...)<br>23<br>snuggle with (pillow/...)<br>20<br>(fx/adjust) hair<br>19<br>lay<br>18|



Upon completing the labeling process, a random sampling procedure was conducted to evaluate the reliability and consistency of
the annotations, and the final inter-annotator agreement exceeded
95%. This rigorous, multi-step approach ensures that the dataset adheres to high-quality standards while providing a robust foundation
for advancing research in the VMR task.


**A.2** **Statistical Analysis**


To further demonstrate the effectiveness of our proposed dataset
Charades-AudioMatter, we conduct activity category-wise analysis
in Table 8. We sorted the categories of selected activities and compared them with other unselected activities. As we can see from
the table, selected activities tend to have more significant differentiating sounds such as “open", “put", and “run", while unselected
activities do not tend to convey sounds such as “sit on", “hold", and
“look". But we can’t exactly classify audio validity by activity, we
give samples in Figure 8.
Table 9 shows the frequency distribution of normalized moment
durations in Charades-AudioMatter compared with the original
Charades-STA. Our purposed Charades-AudioMatter maintains


MM ’25, October 27–31, 2025, Dublin, Ireland. Junan Lin et al.



**Figure 9: Comparison between Charades-AudioMatter and**
**Charades-STA in terms of moment duration.**


**Table 9: Ablation studies of fusion strategies on ActivityNet**
**Captions.**


**Local** **Event** **Global** **R1@3** **R1@5** **R1@7** **mIOU**


✓    -    - 60.08 43.70 27.66 43.96

       - ✓        - 59.13 42.68 26.83 43.54

       -        - ✓ 58.57 42.12 27.00 43.22


✓ ✓    - 59.21 43.60 28.71 44.17

✓    - ✓ 61.25 43.90 28.92 44.78

       - ✓ ✓ 59.84 43.69 28.19 44.05


✓ ✓ ✓ **61.50** **45.06** **29.47** **45.19**


**Table 10: Ablation studies of each component on ActivityNet**
**Captions.**


**Method** **R1@3** **R1@5** **R1@7** **mIOU**


IMG w/o AIP 59.81 43.40 28.06 44.49

IMG w/o pseudo-label 58.10 42.00 27.76 43.52

IMG w/o CKD 59.95 43.89 28.49 44.47

IMG **61.50** **45.06** **29.47** **45.19**


comparable diversity and roughly follows the original CharadesSTA in duration distribution, which validates the rationality of our
proposed dataset.


**B** **Experiments on ActivityNet Captions**


To further verify the general effectiveness of the crucial contributions in our proposed IMG, we conduct more experiments on
ActivityNet Captions.


**B.1** **Ablation studies on fusion strategies**


We verify the effectiveness of fusion strategies on ActivityNet Captions. As shown in Table 9, each of the proposed fusion strategies
consistently yields performance improvements, underscoring their
efficacy. These results also demonstrate that integrating features of
varying granularities provides complementary benefits, leading to
superior overall performance.



**Figure 11: Experiments with different hyperparameters, (a)**
**threshold** _𝜖_ _𝑚𝑖𝑛_ **, (b) temperature** _𝛾_ **.**


**B.2** **Ablation studies on additional model**

**structures**


We also verify the effectiveness of additional model structures on
ActivityNet Captions. As presented in Table 10, lines 1 and 2 illustrate the performance of the model without the audio importance
predictor and the pseudo-label constraint, respectively. These results indicate that the proposed pseudo-label mechanism enhances
decision-making within the audio importance predictor and ultimately improves performance. Finally, line 3 quantifies the effect
of ablating cross-modal knowledge distillation, further validating
the contributions of this component to the overall framework.


**B.3** **Qualitative analysis**


In order to demonstrate our proposed IMG more intuitively, we
have selected examples on ActivityNet Captions for visual presentation. As shown in Figure 10(a) "dives" is visible in the frames, but
"flipping" is not distinctly captured. However, both actions exhibit
clear acoustic semantics, allowing the fusion branch to make a
more accurate prediction. In Figure 10(b), the visual presentation is
highly noticeable, while the audio consists entirely of background
music, and as a result, the fusion branch is not misled by the audio.


**B.4** **Ablation studies on additional model**

**structures**


We also verify the effectiveness of additional model structures on
ActivityNet Captions. As presented in Table 10, lines 1 and 2 illustrate the performance of the model without the audio importance



**Figure 10: Two samples were selected from ActivityNet Cap-**
**tions.**



(a) Performance curves at different
thresholds.



(b) Performance curves at different temperatures.


Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval MM ’25, October 27–31, 2025, Dublin, Ireland.



**Table 11: Ablation studies on hyperparameters** _𝜏_ **on visual**
**branch.**


_𝜏_ **0.5** **1.0** **2.0** **4.0**


R1@7 42.10 42.61 43.44 42.90


predictor and the pseudo-label constraint, respectively. These results indicate that the proposed pseudo-label mechanism enhances
decision-making within the audio importance predictor and ultimately improves performance. Finally, line 3 quantifies the effect
of ablating cross-modal knowledge distillation, further validating
the contributions of this component to the overall framework.


**C** **Additional experiments**

**C.1** **Experiments on hyperparameters**


We conduct ablation studies on two critical hyperparameters, threshold _𝜖_ _𝑚𝑖𝑛_ and temperature _𝛾_ . As detailed in Figure 11, our analysis
reveals that selecting an optimal threshold and temperature significantly enhances the model’s ability to learn the relative importance
of visual and audio features, thereby improving overall performance.
Conversely, setting the threshold too low may cause the model to
mistakenly assign importance to noisy semantic features, while
setting it too high can lead the model to disregard valuable samples
containing relevant semantic information. Similarly, an excessively
low temperature coefficient results in overly rigid decision-making
by the model, whereas an excessively high coefficient diminishes
the model’s sensitivity to the two feature types, ultimately impairing performance.
In Table 11, we conduct an ablation study on the temperature
coefficient _𝜏_ in cross-modal knowledge distillation module. The
results indicate that our method is relatively insensitive to _𝜏_ .
For loss-related parameters, both _𝜆_ 1 and _𝜆_ 2 are crucial to the
model. We set _𝜆_ 1 = 5 and _𝜆_ 2 = 10 to balance the respective loss
terms, ensuring they are on a similar scale. _𝜆_ 3 controls the auxiliary
loss, and we set _𝜆_ 3 = 0.5, significantly smaller than the others.
These values were determined based on grid search and empirical
validation.


**C.2** **Experiments on efficiency**


As shown in the Table 12, we evaluate the efficiency of our proposed method by measuring both the FLOPs and the number of
parameters during inference. Compared to several open-source
methods, our model maintains low computational and parameter
overhead while achieving better performance. Additionally, we also
report the results of key modules in our method including Audio
Importance Predictor (AIP),Multi-Granularity Fusion (MGF) .


**C.3** **Experiments on Event-Level Fusion**


**Slots with supervision.** Since the moment boundary label is
coarse-level and does not have its contained event split, we cannot
utilize it to provide explicit event-level supervision. Therefore, we
adopt the unsupervised slot attention mechanism to implicitly learn
the potential event contexts. Specifically, we utilize the moment
boundary label to additionally supervise the global content of all
events. After slot interaction, the event-level sequences are globally



**Table 12: Comparison on flops and params.**


**Method** **Flops(G)** **Params(M)** **R1@7**


EAMAT 9.97 94.12 41.96

BAM-DETR 1.39 13.43 39.38

FlashVTG 1.05 8.73 38.01

QD-DETR 0.82 6.36 32.55

Moment-DETR 0.26 3.23 38.01

ADPN 0.34 1.54 41.10


**IMG** 0.38 3.31 **44.23**

—AIP 9 _._ 87 × 10 [−][5] 5 _._ 12 × 10 [−][4]      
—MGF 0.20 1.91     

**Table 13: Ablation studies on supervised slots.**


**Method** **R1@3** **R1@5** **R1@7** **mIOU**


slot w/ supervision 73.77 59.60 41.19 54.51

slot w/o supervision 74.84 59.92 41.32 54.83


**Table 14: Performance under different slot numbers and iter-**
**ations.**


**#Slot \ #Iter** **1** **2** **3** **4** **5**


2 40.45 40.75 41.23 41.01 40.76

3 40.23 40.92 **41.32** 41.20 41.25

4 39.88 40.45 41.10 41.22 40.95

5 39.20 39.70 40.12 40.70 40.32


projected into a 1D sequence via an MLP and Sigmoid for supervision with binary cross-entropy loss. As shown in Table 13, the
global supervision yields a similar performance compared to the
unsupervised one. We assume that this is due to the limited granularity of available supervision in VMR, while the unsupervised
version can implicitly learn the potential events.
**Number of slots and iter.** Table 14 summarizes the performance with varying numbers of slots and iterations. While increasing the number of slots and iterations increases computational overhead, we find that using 2 or 3 slots with 3 or 4 iterations achieves
satisfactory performance. Moreover, using 3 slots and 3 iterations
offers a good balance between performance and computational cost.


**C.4** **Experiments on weak supervision**


To assess the robustness of our model under limited supervision,
we further conduct evaluation using reduced training data (70%,
80%, and 90% subsets). As shown in Table 15, our model largely
maintains its performance and consistently outperforms the strong
baseline ADPN.


**C.5** **Audio importance distribution**


Table 16 shows the distribution of audio importance scores, where
most samples fall within the range of 0.15 to 0.45, further validating
that audio serves as an auxiliary modality.


MM ’25, October 27–31, 2025, Dublin, Ireland. Junan Lin et al.



**Table 15: Performance under different training set sizes to**
**evaluate weak supervision capability.**


**Samples for train (%)**
**Method**


**70** **80** **90** **100**


ADPN 37.07 38.78 39.47 41.10

IMG 41.21 43.29 43.52 44.23


**Table 16: Distribution of audio importance scores across sam-**
**ples.**


**Score Range** <0.15 0.15–0.25 0.25–0.35 0.35–0.45 >0.45


**Count** 26 963 1861 665 205


**Table 17: Ablation studies on AIP with zero importance.**


**Method** **R1@3** **R1@5** **R1@7** **mIOU**


IMG 82.74 71.93 54.27 62.76

IMG ( _𝑝_ =0) 80.27 70.22 50.96 59.84



**C.6** **Experiments on failed AIP**


We explore the impact when AIP mistakenly predicts audio importance to zero ( _𝑝_ = 0) on Charades-AudioMatter. As shown in
Table 17, such change degrade the performance. The results not
only show the importance of audio clues, but also demonstrate the
effectivenesss of our proposed audio-aware desgin.


**D** **Implement Details**


For all datasets, we set the initial learning rate to 0.0005, and the
maximum number of frames to 128. We use AdamW [ 38 ] for optimization and linear decay scheduling, and the maximum epoch
number is 100 for all of them with batch size 16. We use I3D [ 7 ]
as pretrained visual features on all datasets. For the audio pretraining models, following previous work [ 8 ], we utilized PANN

[ 27 ], pretrained on AudioSet [ 17 ] dataset, and VGGish [ 21 ], pretrained on YouTube-100M [ 21 ] dataset, for Charades-STA/CharadesAudioMatter and ActivityNet Caption, respectively. We initialize
words with 300d GloVe [ 43 ] embeddings. To further demonstrate
the generalizability of our model, we also use InternVideo2 [ 55 ]
and LLaMA [ 51 ] for visual and textual backbone. We set _𝜖_ _𝑚𝑖𝑛_ to 0.2,
_𝛾_ to 3 for Charades-STA/Charades-AudioMatter, _𝜖_ _𝑚𝑖𝑛_ to 0.1 and _𝛾_
to 2 for ActivityNet Captions. All experiments are implemented on
a single NVIDIA 3090 GPU.


