TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 1

## Unified Static and Dynamic Network: Efficient Temporal Filtering for Video Grounding


[Jingjing Hu, Dan Guo,](https://orcid.org/0009-0003-6944-0848) _Senior Member, IEEE_ [, Kun Li, Zhan Si, Xun Yang,](https://orcid.org/0000-0001-5083-2145)


[Xiaojun Chang,](https://orcid.org/0000-0002-7778-8807) _Senior Member, IEEE_ [and Meng Wang,](https://orcid.org/0000-0002-3094-7735) _Fellow, IEEE_


**Abstract** —Inspired by the activity-silent and persistent activity mechanisms in human visual perception biology, we design a Unified
Static and Dynamic Network (UniSDNet), to learn the semantic association between the video and text/audio queries in a cross-modal
environment for efficient video grounding. For static modeling, we devise a novel residual structure (ResMLP) to boost the global
comprehensive interaction between the video segments and queries, achieving more effective semantic enhancement/supplement. For
dynamic modeling, we effectively exploit three characteristics of the persistent activity mechanism in our network design for a better
video context comprehension. Specifically, we construct a diffusely connected video clip graph on the basis of 2D sparse temporal
masking to reflect the “short-term effect” relationship. We innovatively consider the temporal distance and relevance as the joint “auxiliary
evidence clues” and design a multi-kernel Temporal Gaussian Filter to expand the context clue into high-dimensional space, simulating
the “complex visual perception”, and then conduct element level filtering convolution operations on neighbour clip nodes in message
passing stage for finally generating and ranking the candidate proposals. Our UniSDNet is applicable to both _Natural Language Video_
_Grounding (NLVG)_ and _Spoken Language Video Grounding (SLVG)_ tasks. Our UniSDNet achieves SOTA performance on three widely
used datasets for NLVG, as well as three datasets for SLVG, _e.g_ ., reporting new records at 38.88% _R_ @1 _, IoU_ @0 _._ 7 on ActivityNet
Captions and 40.26% _R_ @1 _, IoU_ @0 _._ 5 on TACoS. To facilitate this field, we collect two new datasets (Charades-STA Speech and TACoS
Speech) for SLVG task. Meanwhile, the inference speed of our UniSDNet is 1.56 _×_ faster than the strong multi-query benchmark. Code
[is available at: https://github.com/xian-sh/UniSDNet.](https://github.com/xian-sh/UniSDNet)


**Index Terms** —Natural Language Video Grounding, Spoken Language Video Grounding, Video Moment Retrieval, Video Understanding, Vision and Language


✦



**1** **I** **NTRODUCTION**
# T queried moment retrieval (MR), as a fundamental and E mporal Video Grounding (TVG), also called language
challenging task in video understanding, has gained importance with the surge of online videos, attracting significant
attention from both academia and industry in recent years.
Generally, the TVG task refers to a natural language sentence as a query, with the goal of locating the accurate video
segment that semantically corresponds to the query [3], [4],


_•_ _J. Hu, D. Guo, K. Li, and M. Wang are with Key Laboratory of_
_Knowledge Engineering with Big Data (HFUT), Ministry of Educa-_
_tion, School of Computer Science and Information Engineering (School_
_of Artificial Intelligence), Hefei University of Technology (HFUT),_
_and Intelligent Interconnected Systems Laboratory of Anhui Province_
_(HFUT), Hefei, 230601, China (e-mail: xianhjj623@gmail.com; guo-_
_dan@hfut.edu.cn; kunli.hfut@gmail.com; eric.mengwang@gmail.com)._

_•_ _Z. Si is with the Department of Chemistry and Centre for Atomic_
_Engineering of Advanced Materials, Anhui University, Hefei, Anhui_
_230601, P.R. China (e-mail: naa0528@163.com)._

_•_ _X. Yang and X. Chang are with the Department of Electronic Engineering_
_and Information Science, School of Information Science and Technology,_
_University of Science and Technology of China, Hefei 230026, China (e-_
_mail: xyang21@ustc.edu.cn; xjchang@ustc.edu.cn)._

_•_ _D. Guo and M. Wang are also with the Institute of Artificial Intelligence,_
_Hefei Comprehensive National Science Center, Hefei, 230026, China._

_•_ _Corresponding authors: D. Guo, X. Yang, X. Chang, M. Wang._


_This work was supported in part by the National Natural Science Foun-_
_dation of China (62272144, 72188101, 62020106007, and U20A20183),_
_and the Major Project of Anhui Province (202203a05020011, 2408085J040,_
_202423k09020001). Fundamental Research Funds for the Central Universities_
_(JZ2024HGTG0309, JZ2024AHST0337 and JZ2023YQTD0072)._



Fig. 1. A schematic illustration of the biology behind how people understand the events of a video during solving video grounding tasks. Firstly,
according to the theory of GNW ( _Global Neuronal Workspace_ ) [1], the
brain engages in static multimodal information association to achieve
semantic complements between multimodalities. Then the focus will
be brought to the dynamic perception of the video content along the
timeline, and during which three characteristics will be expressed: 1)
Short-term Effect: the most recent perceptions have a high impact on
the present; 2) Relevance Clues: semantically scenes will provide clues
to help understand the current scene; 3) Perception Complexity: visual
perception is high-dimensional and non-linear [2].


and the task is named _Natural Language Video Grounding_
_(NLVG)_ . With the development of Automatic Speech Recognition (ASR) and Text-to-speech (TTS), speech is becoming an essential medium for Human-Computer Interaction
(HCI). _Spoken Language Video Grounding (SLVG)_ [5] has also
gained a lot of attention. We find that whether using text
or speech as a query, the key to solving TVG lies in video
understanding and cross-modal interaction. Our work is














TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 2


**Dynamic Temporal Filtering**







vacuum, and he vacuums the floors.



rooms and jumping excitedly because he wants to be picked up.





**Static Global Context**


Fig. 2. An illustrating example for the video grounding task (query: text or audio). This video is described by four queries (events), all of which
have separate semantic contexts and temporal dependencies. Other queries can provide a global context (antecedents and consequences) for the
current query ( _e.g_ ., query _Q_ 4 ). Besides, historical similar scenarios (such as in the blue dashed box) help to discover relevant event clues (time
and semantic clues) for understanding the current scenario (blue solid box).



devoted to multimodal semantics-driven video understanding, namely, how to aggregate multimodal information for
better video understanding?
In this work, we revisit solving TVG tasks through
the lens of human visual perception biology [1], [2], as
illustrated in Fig. 1. We observe that humans quickly comprehend queried events in a video, a process linked to the
Global Neuronal Workspace (GNW) theory and dynamic
visual perception theory in the brain’s prefrontal cortex
(PFC) [1], [2]. These theories describe the interplay between
_activity-silent and persistent activity mechanisms_ in the PFC [2].
The GNW theory suggests that when the brain processes
multi-source data, it creates shallow correlations, allowing
for semantic complementation between multimodal information. This step might not need overly complex deep
networks for multimodal interactions between video and
language. After that, the brain might pay attention on
correlating as much useful information as possible. It will
then focus on the video content and conduct dynamic visual
perception that is transmitted along the **Timeline Main**
**Clue** and exhibits **three characteristics** : **1) Short-term Effect:**
nearby perceptions strongly affect current perceptions; **2)**
**Auxiliary Evidence (Relevance) Cues:** semantically relevant scenes in the video provide auxiliary time and semantic
cues; **3) Perception Complexity:** the perception process is
time-series associative and complex, demonstrating highdimensional nonlinearity [2].
Inspired by the above biological theories, we view the
process of video grounding as the two-stage cross-modal
semantic aggregation, beginning with the global feature
interactions of video and language in _text or audio modality_,
followed by a deeper video semantic purification based on
the dynamic visual perception of the video, and thus design
a unified static and dynamic framework for both NLVG
and SLVG tsaks. **For the static stage**, static multimodal
information will be comprehensively handled based on the
language and video features and semantic connections between them are learned. **For the dynamic stage**, we further
consider the aforementioned three characteristics of visual
perception transmission, and integrate the key ideas of them
into our model design. Specifically, as the example shown
in Fig. 2, we first comprehensively communicate multiple
queries and video clips to obtain contextual information for



the current query ( _e.g_ ., _Q_ 4 ) and associate different queries to
understand video scenes ( _e.g_ ., query _Q_ 2 supplements query
_Q_ 4 with more contextual information, in terms of semantics). This video-query understanding process is deemed as
a _static global interaction_ . Then we design a visual perception
network to imitate _dynamic context information transmission_
in the video with a dynamic filter generation network. We
build a sparely connected relationship (blue arrow in Fig. 2)
between video clips to reflect “Short-term Effect” ( _e.g_ ., the
video frames in the two dashed boxes closest to the solid
blue box have the greatest impact on the current solid box
frame, in terms of temporal direction and action continuity),
and collect “Evidence (Relevance) Clues” ( _e.g_ ., the orange
and green video clips in the dashed boxes contain the cause
and course of the whole video event, providing the time
and semantic clues for current query sub-event) from these
neighbor clips (blue dashed box in Fig. 2) by conducting a
high-dimensional temporal Gaussian filtering convolution
(in Section 3.3, imitating visual Perception Complexity).
Technically, existing methods primarily focus on solving a certain methodological aspect of Temporal Video
Grounding tasks, such as learning self-modality language
and video representation [5], [6], multimodal fusion [7], [8],
cross-modal interaction [9], [10], candidate generation of
proposals [11], [12], proposal-based cross-modal matching

[13], [14], target moment boundary regression [15], [16],
_etc_ . Most current methods prefer to unilaterally consider
the static feature interactions by employing the attention
computation [5], [7], [15], [17]–[21] or graph convolution

[9], [10], [16], [22] and relation computation [6], [11]–[14],

[23]–[27] to associate the query and related video clips,
rather than comprehensively expressing both static and dynamic visual perception simultaneously. Our work actually
proposes a new paradigm for a two-stage unified staticdynamic semantic complementary new architecture.
In this paper, we propose a novel **Unified Static and**
**Dynamic Networks (UniSDNet)** for both NLVG and SLVG.
The overview of UniSDNet is shown in Fig. 3. Specifically,
**for static modeling**, we propose a Static Semantic Supplement Network (S [3] Net), which contains a purely multilayer perceptron within the residual structure (ResMLP) and
serves as a static multimodal feature aggregator to capture
the association between queries and associate queries with


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 3



video clips. Unlike the traditional transformer attention [28]
network, this is a non-attention architecture that constitutes
an efficient feedforward and facilitates data training for
easy optimization of model performance-complexity tradeoffs (in Section 3.2). **For the dynamic modeling**, we design
a Dynamic Temporal Filtering Network (DTFNet) based
on a Gaussian filtering GCN architecture to capture more
useful contextual information in the video sequence (in
Section 3.3). We firstly construct a diffusely connected video
clip graph to reflect the “ _short-term effect_ ” relationship between video clip nodes. Then we redesign the aggregation of
messages from neighboring nodes of the graph network by
innovatively introducing the joint clue of the relative temporal distance _r_ between the nodes and the relevance weight of
the node _a_ for measuring _relevance between nodes_ . We employ
the multi-kernel Temporal Gaussian Filter to extend the joint
clue to high-dimensional space, and by performing highdimensional Gaussian filtering convolution operations on
neighbor nodes, we imitate _visual perception complexity_ and
model fine-grained context correlations of video clips.
Notably, our proposed UniSDNet method shows encouraging performance and high inference efficiency in both
NLVG and SLVG tasks, as shown in Section 5.4. Particularly,
our model achieves higher efficiency (as shown in Fig. 7 and
Table 8). For example, our proposed UniSDNet-M achieves
10. 31% performance gain on the _R_ @1 _, IoU_ @0 _._ 5 metric
while being 1.56× faster than multi-query training SOTA
methods PTRM [14] and MMN [25], and, notably, the static
and dynamic modules of UniSDNet-M are parameterized
only by 0.53M and 0.68M (Table 3), respectively.
Our main contributions are summarized as follows:


_•_ We make a new attempt in solving video grounding
tasks from the perspective of visual perception biology
and propose a Unified Static and Dynamic Networks
(UniSDNet), where the static module is a fully interactive ResMLP network that provides a global crossmodal environment for multiple queries and the video,
and a Dynamic Temporal Filter Network (DTFNet)
learns the fine context of the video with query attached.

_•_ In dynamic network DTFNet, we innovatively integrate
dynamic visual perception transmission biology mechanisms into the node message aggregation process of
the graph network, including a newly proposed joint
clue of relative temporal distance _r_ and the node relevance weight _a_, and a multi-kernel Temporal Gaussian
Filtering approach.

_•_ In order to facilitate the research about the spoken language video grounding, we collect the new CharadesSTA Speech and TACoS Speech datasets with diverse
speakers.

_•_ We conduct experiments on three public datasets for
NLVG and one public dataset and two new datasets
for SLVG, and verify the effectiveness of the proposed
method. The SOTA performance on NLVG and SLVG
tasks demonstrates the generalization of our model.


**2** **R** **ELATED WORKS**


Temporal Video Grounding (TVG) includes Natural Language Video Grounding (NLVG) and Spoken Language
Video Grounding (SLVG). NLVG uses text to locate video



moments, while SLVG relies on spoken language. NLVG is
widely studied due to advancements in natural language
processing, with most existing works focus on it [9]–[14],

[16], [20]–[22], [24]–[27], [29], [30]. SLVG, on the other hand,
has gained attention recently due to its flexible speech-based
querying. However, NLVG methods cannot be directly applied to SLVG without performance loss [5], [31], and few
works address SLVG, leaving room for improvement. In this
work, we consider both NLVG and SLVG tasks.


**2.1** **Natural Language Video Grounding (NLVG)**


Generally, existing popular methods for solving _NLVG_ can
be categorized into two main approaches: proposal-free [5]–

[8], [15], [17]–[19], [23], [32] and proposal-based [9]–[14],

[16], [20]–[22], [24]–[27], [29], [30], [33] methods, with detailed comparative methods listed in Section 5.2. _1) Proposal-_
_free_ methods directly regress the target temporal span based
on multimodal features. These proposal-free methods are
mainly often divided into two main categories: Attentionbased models [7], [17]–[19], [34] and Transformer-based
models [35]–[39]. _2) Proposal-based methods_ use a two-stage
strategy of “generate and rank”. First, they generate video
moment proposals, and then rank them to obtain the best
match. Herein, 2D-TAN [11] is the first solution depositing
possible candidate proposals via a 2D temporal map for
temporal grounding and MMN [25] further optimizes it
for NLVG by introducing metric learning to align language
and video modalities. Because of the elegance of 2D-TAN,
we incorporate the concept of 2D temporal map modeling
into our model, buffering the possible candidate clues.
Our approach is a proposal-based architecture method.
Otherwise, some proposal-based methods also focus on
using Attention-based [8], [13], [23], [24], [29], [30] and
Transformer-based [6], [20] architectures to address textvideo interaction and modal semantic extraction in NLVG
tasks. Additionally, some approaches utilize Graph-based
architectures [9], [10], [16], [22] for modeling static interactions between video clips. Although existing NLVG methods have made significant strides in video grounding, but
they rely on single, static architectures [6], [8]–[10], [13],

[16], [20], [22]–[24], [29], [30], limiting their ability to capture
dynamic interactions as the video progresses.
No matter what, regardless of proposal-free or proposalbased manner, previous methods primarily emphasize feature learning with cross-modal attention [7], [12], [15]–[20],
multi-level feature fusion [14], [23], relational computation [11], [13], [24]–[26], _etc_ .; all the works are conducted _in a_
_relatively static global perceptual mechanism mode_ . Additionally,
more and more methods are dedicated to capturing _the_
_dynamics of the video_ . On one side, temporal feature modeling
are studied, such as using RNN to learn the temporal
video relationship [34], [40] and conditional video feature
manipulation [27]. On the other side, graph methods are
explored for relational learning. For instance, CSMGAN [9]
integrates RNN for video temporal capture followed by fullconnected graph for cross-modal interaction. RaNet [22] and
CRaNet [10] initially utilize the GC-NeXt [41] to aggregate
the temporal and semantic context of the video, and then a
specially designed semantic graph network is used for crossmodal relational modeling. The current graph models [9],


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 4




[10], [16], [22] with Graph Attention Networks (GAN) and
Graph Convolutional Networks (GCN) are prominent for
modeling static interactions between video clips. They
overemphasize the correlation between video clip nodes
but ignore the intrinsic high-dimensional time-series nature
of the video In this work, we examine both static feature
interactions and dynamic video representation in a unified
video grounding framework, considering them in light of
the motivations behind human visual perception. In the
effort to achieve this, we design a lightweight ResMLP
network for static semantic complements and exploit the
relational learning in a video clip graph. Especially, we
fresh sparse masking strategy in a 2D temporal map to
build a diffusive connected video clip graph with dynamic
Temporal Gaussian filtering for video grounding. Extensive
experiments in Section 5 prove that this artifice is available
for both NLVG and SLVG tasks. Such an integrated approach also offers broader applicability across both NLVG
and SLVG tasks.


**2.2** **Spoken Language Video Grounding (SLVG)**


To the best of our knowledge, the only available SLVG
works at present are VGCL [5] and SIL [31], both of them
have been assessed using the _ActivityNet Speech_ dataset
that has collected in VGCL’s work. The VGCL proposes a
proposal-free method that utilizes CPC [42] as the audio
decoder and transformer encoder as the video encoder
to guide audio decoding with the curriculum learning.
The SIL proposes the acoustic-semantic pre-training to improve spoken language understanding and the acousticvisual contrastive learning to maximize acoustic-visual mutual information. VGCL firstly explore whether the virgin
speech rather than text language can highlight relevant
moments in unconstrained videos and propose the SLVG
task. Compared to NLVG, the challenge of SLVG lies in
the discretization of speech semantics and the audio-video
interaction. The new task demonstrates that text annotations
are not necessary to pilot the machine to understand video.
Recently, with the development of audio pre-training, a
breakthrough has been made in the discretization feature
representation of speech [43]–[45].In this work, we focus
on the audio-video interaction challenge of SLVG through
the proposed UniSDNet. More importantly, to facilitate the
research of SLVG, we collect two new audio description
datasets named Charades-STA Speech and TACoS Speech
that originate from the NLVG datasets of Charades-STA [3]
and TACoS [46]. For more details, please refer to Section 4.


**3** **P** **ROPOSED** **F** **RAMEWORK**


**3.1** **Task Definition & Framework Overview**


The goal of the NLVG ( _natural language video grounding_ ) and
SLVG ( _spoken language video grounding_ ) is to predict the temporal boundary ( _t_ _[s]_ _, t_ _[e]_ ) of the specific moment in the video
in response to a given query in text or audio modality. Denote the input video as _V_ = _{v_ _i_ _}_ _[T]_ _i_ =1 _[∈]_ [R] _[T][ ×][d]_ _[v]_ [, where] _[ d]_ _[v]_ [ and]
_T_ are the feature dimension and total number of video clips,
respectively. Each video has an annotation set of _{Q, M}_,
in which _Q_ is a _M_ -query set in the _text_ or _audio_ modality
and _M_ represents the corresponding video moments of the



queried events, denoted as _Q_ = _{q_ _i_ _}_ _[M]_ _i_ =1 _[∈]_ [R] _[M]_ _[×][d]_ _[q]_ [, and]
_M_ = _{_ ( _t_ _[s]_ _i_ _[, t]_ _[e]_ _i_ [)] _[}]_ _[M]_ _i_ =1 [, where] [ (] _[t]_ _[s]_ _i_ _[, t]_ _[e]_ _i_ [)] [ represents the starting and]
ending timestamps of the _m_ -th query, _d_ _[q]_ is the dimension
of query feature, and _M_ is the query number.
In this paper, we present a unified framework, named
**Uni** fied **S** tatic and **D** ynamic **Net** work ( **UniSDNet** ), for both
NLVG and SLVG tasks, focusing on video content understanding in the multimodal environment. Fig. 3 illustrates
the overview of our proposed architecture. Our UniSDNet
comprises the _**Static Semantic Supplement Network (S**_ [3] _**Net)**_
and _**Dynamic Temporal Filtering Network (DTFNet)**_ . It
adopts a two-stage information aggregation strategy, beginning with a global interaction mode to perceive all multimodal information, followed by a graph filter to purify key
visual information. Finally, we extract enhanced semantic
features of the video clip for high-quality 2D video moment proposals generation. In the following subsections, we
introduce the core modules, S [3] Net (Section 3.2), DTFNet
(Section 3.3), and 2D proposal generation (Section 3.4) of
our proposed unified framework.


**3.2** **Static Semantic Supplement Network**


The static network S [3] Net is inspired by the concept of the
global neuronal workspace (GNW) [1] in the human brain,
which aggregates the multimodal information in the first
stage of visual event recognition. In terms of the functionality of the static network for video understanding, it provides more video descriptions information and significantly
fills the gap between vision-language modalities, aiding in
understanding video content.
Technically, the S [3] Net can be seen as a fully interactive and associative process involving static queries and
video features. From the aforementioned perspective, we
have designed the static semantic supplement network
S [3] Net (as shown in Fig. 3) by integrating the MLP into
the residual structure (ResMLP). The incorporation of a
multilayer perceptron within ResMLP enables the fulfillment of static feature’s linear interaction requirement for
achieving multimodal information aggregation. This setup
constitutes an efficient feedforward network that facilitates
data training and allows for easy optimization of model performance/complexity trade-offs. Additionally, employing a
linear layer offers the advantage of having long-range filters
at each layer [47].
Before feature interaction, we utilize pre-trained models
(C3D [48], GloVe [49], Data2vec [50], _etc_ .) to extract the
original video and query features, which are then linearly
converted into a unified feature space. This yields video
and query features _F_ _V_ _∈_ R _[T][ ×][d]_ and _F_ _Q_ _∈_ R _[M]_ _[×][d]_, respectively, with _F_ _VQ_ = [ _F_ _V_ _||F_ _Q_ ] _∈_ R [(] _[T]_ [ +] _[M]_ [)] _[×][d]_ . Inspired
by the existing multi-modal Transformers work [51]–[53],
we independently add position embeddings for video and
queries, to distinguish modality-specific information. More
ablation studies on adding position embeddings are discussed in Appendix B.2. Specifically, we incorporate the
position embedding [28] _P_ _V_ _∈_ R _[T][ ×][d]_ for video feature and
_P_ _Q_ _∈_ R _[M]_ _[×][d]_ for query feature, and concatenate them into
_P_ _VQ_ = [ _P_ _V_ _||P_ _Q_ ] _∈_ R [(] _[T]_ [ +] _[M]_ [)] _[×][d]_ . We use MLPBlock, which
is a combination of a LayerNorm layer, a Linear layer, a


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 5



**Static Semantic Supplement Network** **Dynamic Temporal Filtering Network** **2D Proposal Generation**



_**v**_ **1**
_**v**_ **2**
_**v**_ _**3**_


…

_**v**_ _**N**_


_**q**_ **1**
_**q**_ **2**

_**q**_ … _**M**_























Fig. 3. **The architecture of the Unified Static and Dynamic Network (UniSDNet).** It mainly consists of static and dynamic networks: Static
Semantic Supplement Network (S [3] Net) and Dynamic Temporal Filtering Network (DTFNet). **S** [3] **Net** concatenates video clips and multiple queries
into a sequence and encodes them through a lightweight single-stream ResMLP network. **DTFNet** is a 2-layer graph network with a dynamic
Gaussian filtering convolution mechanism, which is designed to control message passing between nodes by considering temporal distance and
semantic relevance as the Gaussian filtering clues when updating node features. The role of 2D temporal map is to retain possible candidate
proposals and represent them by aggregating the features of each proposal moment. Finally, we perform semantic matching between the queries
and proposals and rank the best ones as the predictions.



ReLU activation layer, and a Linear layer, to obtain the static
interactive video clip features _F_ [ˆ] _V_ and query features _F_ [ˆ] _Q_ :



_F_ ˜ _VQ_ = _F_ _VQ_ + LayerNorm( _F_ _VQ_ ) + _P_ _VQ_ _,_
_F_ ˆ _VQ_ = LayerNorm( ˜ _F_ _VQ_ + MLPBlock( ˜ _F_ _VQ_ )) _,_
_F_ ˆ _V_ = ˆ _F_ _VQ_ [1 : _T_ ; :] _∈_ R _[T][ ×][d]_ _,_
_F_ ˆ _Q_ = ˆ _F_ _VQ_ [ _T_ + 1 : _T_ + _M_ ; :] _∈_ R _[M]_ _[×][d]_ _._



(1)



emulate the human visual perception process, we introduce
a new message passing approach between video clip nodes
and propose a Dynamic Temporal Filtering Graph Network
(DTFNet as depicted in Figs. 3 and 4).
To imitate the Short-term Effect, we construct a diffusive
connected graph based on the 2D temporal video clip map
(please see “Graph Construction” below). For discovering
Auxiliary Evidence Cues, we integrate the message passed
from each node’s neighbors by measuring the relative temporal distance and the semantic relevance in the graph (as
explained in the filter clue introduced in “How to construct
_F_ _filter_ ?” below). Finally, we employ a multi-kernel Gaussian filter-generator to expand the auxiliary evidence clues
to a high-dimensional space, simulating the complex visual
perception capabilities of humans (explained in the filter
function in “How to construct _F_ _filter_ ?” below).


_3.3.1_ _Graph Construction_
Let us denote a video graph _G_ = ( _G_ _V_ _, G_ _E_ ) to represent the
relation in the video _V_ . In the graph _G_, node _v_ _i_ is the _i_ -th
video clip and edge _e_ _ij_ _∼_ ( _v_ _i_ _, v_ _j_ ) _∈G_ _E_ represents whether
_v_ _j_ is _v_ _i_ ’s connective neighbor. We obtain _F_ [ˆ] _V_ from the S [3] Net
(in Eq. 1) and take it as the initialization of clip nodes in the
graph, namely the initial node embedding of the graph is set
to _G_ _V_ (0) = ˆ _F_ _V_ _∈_ R _T ×d_ . For the graph edge set _G_ _E_, we utilize
a diffusive connecting strategy [11] based on the temporal
distance of two nodes, to determine the edge status _e_ _ij_ . The
temporal distance between node _v_ _j_ and node _v_ _i_ is defined as
_r_ _ij_ = _∥j −_ _i∥_, setting the hyperparameter _k_, for the current
node _v_ _i_, we define the **short distance** as 0 _≤_ _r_ _ij_ _< k_ and the
**long distance** as _r_ _ij_ _≥_ _k_ . Based on these two distances, there
are two types of edge connections: (1) Dense connectivity for
nodes with a short distance: when 0 _≤_ _r_ _ij_ _< k_, we densely
connect two nodes, _i.e_ ., _G_ _E_ _short_ = _{e_ _ij_ _|_ 0 _≤_ _r_ _ij_ _< k}_ . (2)
Sparse connectivity for nodes with a long distance: when
_r_ _ij_ _≥_ _k_, we connect them at exponentially spaced intervals,
_i.e_ ., the following conditions should be met when _e_ _ij_ exists:



Note that UniSDNet can accommodate any number of
queries as inputs during training. Proving a single query
input is the traditional training mode for NLVG and SLVG
tasks. When multiple queries are fed as inputs, there are
interactions among the queries, within the video (across
multiple video clips), and between the queries and video.
This approach enables the learning of self-modal and crossmodal semantic associations between video and queries
without semantic constraints, allowing the model to leverage the complementary effects among multiple queries related to the same video content. The semantics, either in a
single query or multiple queries, can offer more comprehensive semantic supplementation for a effective and efficient
understanding of the entire video content.


**3.3** **Dynamic Temporal Filtering Network**


The second stage (DTFNet) of UniSDNet dynamically filters
out important video content, inspired by the dynamic visual
perception mechanism observed in human activity [2], as
introduced in Section 1. We imitate the three characteristics
of this visual perception mechanism by learning a video
graph network. We restate the key points of these three
characteristics here: **1) Short-term Effect:** nearby perceptions strongly affect current perceptions; **2) Auxiliary Ev-**
**idence (Relevance) Cues:** semantically relevant scenes in
the video provide auxiliary time and semantic cues; **3) Per-**
**ception Complexity:** the perception process is time-series
associative and complex, demonstrating high-dimensional
nonlinearity [2]. These characteristics play a crucial role in
assisting individuals in locating queried events within the
video, which have been explained in Fig. 1 and Fig. 2.
Graph neural networks have shown efficacy in facilitating
intricate information transmission between nodes [54]. To



_G_ _E_ _long_ = _{e_ _ij_ _},_ _s.t._







_i_ mod 2 _[n]_ [+1] = 0

_r_ _ij_ mod (2 _[n]_ _k_ ) = 0



_r_ _ij_ mod (2 _[n]_ _k_ ) = 0 _,_ (2)

2 _[n]_ _k ≤_ _r_ _ij_ _<_ 2 _[n]_ [+1] _k_


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 6



where _n_ = (0 _,_ 1 _, · · ·, ⌈log_ 2 _[T]_ _k_ _[⌉−]_ [1)] [.] _[ ⌈·⌉]_ [is the ceil function.]

we obtain a sparsely connected edge set _G_ _E_ = _G_ _E_ _short_ _∪_
_G_ _E_ _long_ . Please note that we model forward along the timeline, resulting in that the edge set _G_ _E_ is reflected as a upper
triangular adjacency matrix. For more explanation and discussion on the edge construction, please see Appendix B.1.


_3.3.2_ _Temporal Filtering Graph Learning_

We build _L_ -layer graph filtering convolutions in our implementation. During training, the node embedding _G_ _V_ ( _l_ ) =
_{v_ _i_ [(] _[l]_ [)] _[}]_ _i_ _[T]_ =1 [is optimized at each graph layer,] [ 1] _[ ≤]_ _[l][ ≤]_ _[L]_ [. In]
this part, we introduce a Gaussian Radial **Filter-Generator**
_F_ _filter_ shown in Fig. 4 to imitate the dynamic flashback
process of video for visual perception. There are two core
technical difficulties to be resolved below.
**How to construct** _F_ _filter_ **?** Since visual perception is
transmitted along the timeline, we consider the relative time
interval between nodes as the primary clue. Additionally,
similar scenes work appropriately on the comprehension
of current scene, so we take into account the semantic
relevance between graph nodes as auxiliary clue. Specifically, we compute the two clues of the relative temporal
distance _r_ _ij_ of node _v_ _j_ and node _v_ _i_ ( _r_ _ij_ = _||j −_ _i||_ ) and
the relevance weight _a_ _ij_ of this two-node pair measured
by the _cos_ ( _·_ ) similarity function. We combine them as the
joint clue _d_ _ij_ = (1 _−_ _a_ _ij_ ) _· r_ _ij_ . To mimic the dynamic
nature, continuity (high dimensionality), and non-linearity
(complexity) of visual perception transmission, we use the
filter-generating network to dynamically generate highdimensional filter operators that control message passing
between nodes, rather than directly applying the simple
discrete scalar _d_ _ij_ to compute message aggregation weights,
which is insufficient to express these properties. The filtergenerator (as illustrated in Fig. 4) is given in the form of
_F_ _filter_ ( _d_ _ij_ ) : R _→_ R _[h]_ . Gaussian function has already been
exploited in deep neural networks, such as Gaussian kernel
grouping [55], learnable Gaussian fucntion [21], Gaussian
radial basis function [56] that have been proven to be effective in simulating high-dimensional nonlinear information
in various scenes. Inspired by these works, we adopt multikernel Gaussian radial basis to extend the influence of the
clue _d_ _ij_ into high-dimensional space, thereby reflecting the
continuous complexity of the perception process. Specifically, we design a temporal Gaussian basis function to
build the _F_ _filter_ and expand the joint clue _d_ _ij_ to a high
dimension vector _f_ _ij_ _∈_ R _[h]_ in message passing process.
We express the form of a single kernel temporal Gaussian
as _ϕ_ ( _d_ _ij_ _, z_ ) = exp( _−γ_ ( _d_ _ij_ _−_ _z_ ) [2] ), where _γ_ is a Gaussian
coefficient that reflects the amplitude of Gaussian kernel
function and controls the gradient descent speed of the
function value, _z_ is a bias we added to avoid a plateau at the
beginning of training due to the highly correlated Gaussian
filters. Furthermore, we expand it to multiple-kernel Gaussian function Φ( _d_ _ij_ _, Z_ ) = exp( _−γ_ ( _d_ _ij_ _−_ _z_ _k_ ) [2] ) _, k ∈_ [1 _, h_ ] to
fully represent the complex nonlinear of video perception.
Based on the single kernel term, we construct _h_ kernel
functions, more studies on the settings of ( _γ, z, h_ ) are in
the Section 5.5.3. The way we generate the filter _f_ _ij_ of node
_v_ _j_ to node _v_ _i_ through the multi-kernel Gaussian filer is:


_f_ _ij_ = _F_ _filter_ ( _d_ _ij_ ) = ( _ϕ_ 1 ( _d_ _ij_ ) _, ϕ_ 2 ( _d_ _ij_ ) _, · · ·, ϕ_ _h_ ( _d_ _ij_ )) _._ (3)



Joint relevance weight
and relative temporal _**a**_ _**ij**_
distance of nodes _**r**_ _**ij**_









Multi-Kernel

Gaussian Radial

Basis Functions





(a) Message Aggregation (b) Filter-Generator


Fig. 4. The process of (a) node message aggregation in the Dynamic
Temporal Filtering graph and (b) dynamic filter-generator _Filter_, which
is built based on the joint clue of relevance weight _a_ _ij_ and relative
temporal distance _r_ _ij_ between two nodes. This joint clue is expanded
into high dimensions representation through a multi-kernel Gaussian
radial basis function.


**How to update the nodes in graph** _G_ _V_ **?** In the stage of
message passing on _l_ -th layer, we update each node representation by aggregating its neighbor node message to obtain _G_ _V_ ( _l_ ) . For node _v_ _i_, its neighbor set is _{v_ _j_ _| v_ _j_ _∈N_ ( _v_ _i_ ) _}_
corresponding to the adjacency map _G_ _E_ . With the multikernel Gaussian filter _f_ _ij_, the update of node feature _v_ _i_ on
_l_ -th graph layer is described as:



� FFN 1 ( _f_ _ij_ ) _⊙_ FFN 0 ( _v_ _j_ [(] _[l][−]_ [1)] )

_j∈N_ ( _v_ _i_ )



_v_ _i_ [(] _[l]_ [)] = _σ_







�
 _j∈N_ (







_,_ (4)




where _⊙_ represents element-wise multiplication and _σ_ is
a ReLU activation function. So far, a video graph with
spatiotemporal context correlation of video clips is learned.


**3.4** **2D Proposal Generation**


**Proposal Generation.** After obtaining the updated video
clip features from the above DTFNet module, we implement
the moment sampling [11] on the features to generate a 2D
temporary proposal map _M_ [2D] _∈_ R _[T][ ×][T][ ×][d]_ that indicates
all candidate moments (2D Proposal Generation in Fig. 3).
The element _m_ _ij_ in the map _M_ [2D] indicates the candidate
proposal [ _v_ _i_ _, · · ·, v_ _j_ ] . For each moment _m_ _ij_, we consider all
the clips in the moment interval and the boundary feature
is further added to the moment representation (Eq. 5).
Afterwards, a stack of 2D convolutions is used to encode
the moment feature. For the detailed ablation studies about
the moment sampling strategy, please refer to Section 5.5.4.


_m_ _ij_ = MaxPool( _v_ _i_ _[L]_ _[, v]_ _i_ _[L]_ +1 _[,][ · · ·][, v]_ _j_ _[L]_ [) +] _[ v]_ _i_ _[L]_ [+] _[ v]_ _j_ _[L]_ _[∈]_ [R] _[d]_ _[,]_ (5)

_M_ [2D] = _CNN_ ( _m_ _ij_ ) _∈_ R _[T][ ·][T][ ·][d]_ _._


**Modality Alignment Measurement.** We calculate the relevance of each { _query_, _moment proposal_ } pair according to
the semantic similarity, generating new 2D moment score
maps for the _M_ -queries. Specifically, a 1 _×_ 1 convolution
and an FFN are respectively used to project the moment
feature _M_ [2] _[D]_ and the query feature _F_ [ˆ] _Q_ into the same dimensional vectors _S_ _[M]_ _∈_ R _[T][ ×][T][ ×][d]_ and _S_ _[Q]_ _∈_ R _[M]_ _[×][d]_ . Following
MMN [25], we use cosine similarity to measure the semantic
similarity between queries and moment proposals, it is


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 7



defined as _S_ [˜] = CoSine( _S_ _[M]_ _, S_ _[Q]_ ) . Thereby, _M_ similarity
score maps for input _M_ queries are computed:


_S_ _[M]_ = Norm(Conv2d 1 _×_ 1 ( _M_ [2D] )) _∈_ R _[T][ ·][T][ ·][d]_ _,_



_S_ _[Q]_ = Norm(FNN( _F_ [ˆ] _Q_ )) _∈_ R _[M]_ _[·][d]_ _,_
_S_ ˜ = CoSine( _S_ _[M]_ _, S_ _[Q]_ ) = _{s_ ˜ [1] _,_ ˜ _s_ [2] _, · · ·,_ ˜ _s_ _[M]_ _} ∈_ R [(] _[T][ ×][T]_ [ )] _[·][M]_ _,_



(6)



_Speech dataset_ [5] is publicly available, an extension of _Ac-_
_tivityNet Captions_ dataset used for NLVG task. To accelerate
SLVG development, we collect **two new Speech datasets** :
_**Charades-STA Speech**_ **and** _**TACoS Speech**_ based on the
original Charades-STA and TACoS datasets.


**4.1** **Existing Datasets for NLVG Task and SLVG Task**


The dataset benchmarks used for the NLVG task consist
of the untrimmed video and its annotations (text sentence descriptions and video moment pairs). _**(1) Activi-**_
_**tyNet Captions**_ [57] dataset includes 19,209 videos sourced
from YouTube’s open domain collection, initially proposed
by [57] for dense video captioning task and later utilized for video grounding task. The dataset is divided
according to the partitioning scheme in [7], [11]; it comprises 37,417, 17,505, and 17,031 sentence-moment pairs for
training, validation, and testing, respectively. _**(2) Charades-**_
_**STA**_ [3] dataset consists of 9,848 relatively short indoor
videos from Charades dataset [59] originally designed for
action recognition and localization. It is extended by [3] to
include language descriptions for the NLVG task, including
12,408 and 3,720 sentence-moment pairs for training and
testing, respectively. _**(3) TACoS**_ [46] dataset focuses on 127
activities within a kitchen, constructed based on the MPIICompositive dataset [60]. Following the split outlined in

[11], the dataset includes 10,146, 4,589, and 4,083 sentencemoment pairs for training, validation and testing, respectively. Compared to _ActivityNet Captions_ and _Charades-STA_,
the _TACoS_ features longer and more annotated queries for
each video, with an average of 286.59s and 130.53 per video
in the training set.
Currently, there is only one dataset, _ActivityNet Speech_
proposed by Xia _et al_ . [5], publicly available for the SLVG
task. The dataset is collected based on the ActivityNet Captions dataset [57], consisting of 37,417, 17,505, and 17,031
audio-moment pairs for training, validation, and testing (as
the same split as in [57]), where audio is obtained by 58
volunteers (28 male and 30 female) reading the text fluently
in a clean surrounding environment.


**4.2** **New Collected Datasets for SLVG Task**


In this work, we collected two new datasets to facilitate
SLVG research. Unlike the _ActivityNet Speech_ [5] with manual text-to-speech reading, we use _machine simulation_ to
synthesize audio subtitle datasets and release two **new**
_**Charades-STA Speech**_ **and** _**TACoS Speech**_ **datasets** [1] . The
considerations for adopting the machine simulation are:

_•_ **High-quality synthesised voice.** Thanks to advancements in text-to-speech (TTS) technology [58], [61], TTS
is capable of closely simulating the human voice, effectively capturing and expressing intricate voice characteristics, including speaking style and tone, and generating a high-quality synthesised voice.

_•_ **Diverse readers.** We randomly select a “reader” from
the CMU ARCTIC database [2] to “read” text sentences


1. _Charades-STA Speech_ [dataset is available at https://zenodo.org/](https://zenodo.org/records/8019213)
[records/8019213 and](https://zenodo.org/records/8019213) _TACoS Speech_ [dataset is available at https://](https://zenodo.org/records/8022063)
[zenodo.org/records/8022063](https://zenodo.org/records/8022063)

[2. CMU ARCTIC database is available at http://www.festvox.org/](http://www.festvox.org/cmu_arctic/)
[cmu_arctic/](http://www.festvox.org/cmu_arctic/)



where for each query _q_ _i_, the proposal corresponding to the
maximum value in ˜ _s_ _[i]_ is selected as the best match for the
given query _q_ _i_ . There are some other semantic similarity
functions for measuring modal alignment. Please refer to
Appendix B.3 for relevant ablation study.


**3.5** **Training and Inference**


Our UniSDNet is proposal-based, thereby we optimize the
score map _S_ [˜] with IoU regression loss and contrastive
learning loss. Following 2D-TAN [11], we compute the
groundtruth IoU Map IoU [GT] = _{IoU_ _[i]_ _}_ _[M]_ _i_ =1 _[∈]_ [R] [(] _[T][ ×][T]_ [ )] _[·][M]_

corresponding to queries. That is, we compute the value
of intersection over union between each candidate moment
and the target moment ( _t_ _[s]_ _gt_ _[, t]_ _[e]_ _gt_ [)] [, and scale this value to (0,1),]
with total _N_ moment scores. The IoU prediction loss is



_L_ _iou_ = _N_ [1]



_N_
� ( _iou_ _i_ _·_ log( _y_ _i_ ) + (1 _−_ _iou_ _i_ ) _·_ log(1 _−_ _y_ _i_ )) _,_ (7)

_j_ =1



where _iou_ _i_ is the groundtruth from IoU [GT], and _y_ _i_ is the
predicted IoU value from _S_ [˜] in Eq. 6.
Besides, we adopt contrastive learning [25] as an auxiliary constraint, to fully utilize the positive and negative
samples between queries and moments to provide more
supervised signals. The noise contrastive estimation [42] is
used to estimate two conditional distributions _p_ ( _q|m_ ) and
_p_ ( _m|q_ ) . The former represents the probability that a query _q_
matches the video moment _m_ when giving _m_, and the latter
represents the probability that a video moment _m_ matches
the query _q_ when giving _q_ .



_L_ _contra_ = _−_ ( �



� log _p_ ( _m_ _q_ _|q_ ) + �

_q∈Q_ _[B]_ _m∈M_



� log _p_ ( _q_ _m_ _|m_ )) _,_ (8)

_m∈M_ _[B]_



where _Q_ _[B]_ and _M_ _[B]_ are the sets of queries and moments
in a training batch. _m_ _q_ _∈{m_ [+] _q_ _[, m]_ _[−]_ _q_ _[}]_ [,] _[ m]_ [+] _q_ [is the moment]
matched to query _q_ (solo positive sample) and _m_ _[−]_ _q_ [denotes]
the moment unmatched to _q_ in the training batch (multiple
negative samples). The definition of _q_ _m_ _∈{q_ _m_ [+] _[, q]_ _m_ _[−]_ _[}]_ [ for]
moment _m_ is similar to that of _m_ _q_ _∈{m_ [+] _q_ _[, m]_ _[−]_ _q_ _[}]_ [. The ob-]
jective of contrastive learning is to guide the representation
learning of video and queries and effectively capture mutual
matching information between modalities. As a result, the
total loss is _L_ = _L_ _iou_ + _L_ _contra_ . Non-Maximum Suppression
(NMS) threshold is 0.5 during inference.


**4** **D** **ATASETS**


To validate the effectiveness of our proposed unified static
and dynamic framework for both NLVG and SLVG tasks,
we conduct experiments on the popular video grounding
benchmarks. There are three classic benchmarks for NLVG
task, _i.e_ ., _ActivityNet Captions_ [57], _Charades-STA_ [3], and
_TACoS_ [46] datasets. For SLVG task, only the _ActivityNet_


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 8


TABLE 1
Data statistics of three widely used datasets for NLVG task, ActivityNet Captions, Charades-STA and TACoS datasets.

|Datasets|Domain|# Videos<br>Train Val Test|# Sentences<br>Train Val Test|Average Length<br>Video Words Moment|Average Queries per Video<br>Train Val Test|
|---|---|---|---|---|---|
|ActivityNet Captions [57]<br>Charades-STA [3]<br>TACoS [46]|Open<br>Indoors<br>Cooking|10,009<br>4,917<br>4,885<br>5,336<br>-<br>1,334<br>75<br>27<br>25|37,421<br>17,505<br>17,031<br>12,408<br>-<br>3,720<br>9,790<br>4,436<br>4001|117.60s<br>14.41<br>37.14s<br>30.60s<br>7.22<br>8.09s<br>286.59s<br>9.42<br>27.88s|3.74<br>3.56<br>3.49<br>2.33<br>-<br>2.79<br>130.53<br>164.30<br>160.04|



TABLE 2
Data statistics of datasets for SLVG task. _Charades-STA Speech_ _[∗]_ and _TACoS Speech_ _[∗]_ are new datasets collected by us, using machine
simulation [58] from CMU ARCTIC database, offering more diverse pronunciations than AcitivityNet Speech.


|Datasets|Domain|# Videos<br>Train Val Test|# Audios<br>Train Val Test|Average Length<br>Video Audio Moment|Audio Source|
|---|---|---|---|---|---|
|ActivityNet Speech [5]<br>**_Charades-STA Speech_**_∗_<br>**_TACoS Speech_**_∗_<br>|Open<br>Indoors<br>Cooking|10,009<br>4,917<br>4,885<br>5,336<br>-<br>1,334<br>75<br>27<br>25|37,421<br>17,505<br>17,031<br>12,408<br>-<br>3,720<br>9790<br>4436<br>4001|117.60s<br>6.22s<br>37.14s<br>30.60s<br>**2.33s**<br>8.09s<br>286.59s<br>**2.89s**<br>27.88s|58 Volunteers<br>**3,869 Readers**<br>**126 Readers**|





(a) ActivityNet Captions









(a) ActivityNet Captions



(b) TACoS



(b) TACoS



(c) Charades-STA



(c) Charades-STA



Fig. 5. Statistics on the query number size of each video in training set
for NLVG&SLVG datasets (1k=1,000). The datasets can be divided into
three categories: large query size (TACoS & _TACoS Speech_, most sizes
are 110), middle query size (ActivityNet Captions & ActivityNet Speech,
most sizes are 3), and small query size (Charades-STA & _Charades-STA_
_Speech_, most sizes are 1, and the query description is often ambiguous
and semantically insufficient as the video is too short with mostly 30s
duration for manually annotating events).


in Charades-STA and TACoS datasets. The database
contains 7,931 vocal embeddings with different English
pronunciation characteristics.

_•_ **Cost savings and high-quality annotation.** With the
strong ability of TTS technology to prevent errors like
word mispronunciations, incoherent sentence delivery,
and audio-text mismatches caused by manual annotation, the necessity for manual text reading, recording,
and file annotation processes is mitigated. Machine
emulation reduces the cost of manual annotation and
avoids manual reading errors.


Based on the above considerations, we adopt the TTS
technology “microsoft/speecht5_tts” [3] to collect the audio
description of the text query with a random virtual “reader”
in the CMU ARCTIC database to guarantee the diversity of
voice, style, and tone. Compared to the _ActivityNet Speech_
dataset, the _Charades-STA Speech_ and _TACoS Speech_ datasets
we collected have more diverse pronunciations. The average
of each speech recording is 2.33 seconds and 2.89 seconds in
the _Charades-STA Speech_ and _TACoS Speech_ datasets, respectively. It is important to note that the partitioning of both the
_Charades-STA Speech_ and _TACoS Speech_ datasets is consistent
with their source datasets _Charades-STA_ [3] and _TACoS_ [46].


[3. Source code of Microsoft TTS technology is available at https://](https://huggingface.co/microsoft/speecht5_tts)
[huggingface.co/microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts)



Fig. 6. Statistics on the query length ( _counted by word number_ ) in
training set for NLVG&SLVG datasets (1k=1000). The query length of
the ActivityNet Captions dataset are generally long (mostly 14 words
and mostly 6s per query), having more detailed descriptions compared
to the other two datasets.


We have summarized the statistics of these two new datasets

for SLVG task in Table 2.


**4.3** **Datasets Analysis**


First of all, please note that the SLVG datasets are derived from the NLVG datasets, sharing the same video and
query sentence. The main difference between them is the
modality of query used: SLVG use audio-moment pairs,
while NLVG use text-moment pairs. The datasets exhibit
distinct characteristics in the following aspects: _**(1) Video**_
_**Duration**_ . The average video duration is counted in Table 2
with the datasets ActivityNet, Charades-STA, and TACoS
of 117.60s, 30.60s, and 286.59s, respectively. The minimum
video duration in the Charades-STA implies a stricter judgment of event boundaries than the other two datasets. _**(2)**_
_**Query Length**_ _(counted by word number in a text sentence or_
_audio duration)._ Generally, the longer the audio duration,
the more words in the text annotation, and the richer the
information provided by the query to describe the video.
Notably, the ActivityNet Speech dataset has longer queries
(mostly 14 words and mostly 6s per query as shown in
Fig. 6), providing more detailed descriptions. _**(3) Query**_
_**Number**_ . Fig. 5 shows the distributions of the video’s query
numbers, the datasets can be divided into three categories:
large (TACoS), medium (ActivityNet Captions), and small
(Charades-STA). Particularly, the Charades-STA is minimal
with at most 1 query per video, suggesting a potential
limitation in description detail provided for the video.


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 9


TABLE 3
The hyperparameter settings of UniSDNet framework for different NLVG&SLVG datasets with the specific pre-extracted video features. It is worth
noting that the number of parameters in the static (S [3] Net) and dynamic (DTFNet) modules of UniSDNet is extremely small on all datasets.


|Datasets|#Clips|Static S3Net<br>Hidden size|Dynamic DTFNet<br>#Layers Hidden size|2D Proposal Generation<br>#Layers Kernel size Hidden size|#Parameters<br>S3Net 3.2 DTFNet 3.3 Proposal Generation 3.4|
|---|---|---|---|---|---|
|ActivityNet Captions (C3D)|64|1024|2<br>256|4<br>9<br>512|**0.53M**<br>**0.68M**<br>76.79M|
|Charades-STA (VGG)<br>Charades-STA (C3D)<br>Charades-STA (I3D)|16<br>16<br>64|1024<br>1024<br>1024|2<br>512<br>2<br>512<br>2<br>256|3<br>5<br>512<br>3<br>5<br>512<br>2<br>17<br>512|**1.05M**<br>**2.68M**<br>20.19M<br>**1.05M**<br>**2.68M**<br>20.19M<br>**0.53M**<br>**0.68M**<br>113.91M|
|TACoS (C3D)|128|1024|2<br>256|3<br>5<br>512|**0.53M**<br>**0.68M**<br>16.65M|



**5** **E** **XPERIMENTS**


**5.1** **Experimental setup**


**Evaluation Metrics.** Following the convention in the video
grounding and video moment retrieval tasks [3], [7], [19],
we compute the “ _R_ @ _h, IoU_ @ _u_ ” and “ _mIoU_ ” for performance evaluation of both NLVG and SLVG tasks. The metric
“ _R_ @ _h, IoU_ @ _u_ ” denotes the percentage of samples that have
at least one correct answer in the top- _h_ choices, where the
criterion for correctness is that the moment IoU between
the predicted result and the groundtruth is greater than a
threshold _u_ . Mathematically, “ _R_ @ _h, IoU_ @ _u_ ” is defined as:



which includes 500-dim C3D feature [48] on ActivityNet
Captions, 4096-dim VGG feature [64] on Charades-STA, and
500-dim C3D feature on TACoS from [9]. Besides, there are
currently other popular C3D feature and I3D feature [65]
available on Charades-STA, so we also use the 4096-dim
C3D feature from [18] and 1024-dim I3D feature provided
by [19]. Following previous work [25], we use the GloVe [49]
and BERT [62] to extract textual feature. For the audio
feature, we use the HuggingFace [66] implementation of
Data2vec [50] with pre-trained model “facebook/data2vecaudio-base-960h” for SLVG. Specifically, we set the audio
sampling rate to 16,000 Hz, and use the python audio standard library “librosa” to read the original audio and input
it into the Data2vec model to obtain the audio sequence
embedding. Additionally, we use LayerNorm and AvgPool
operations to aggregate the entire audio representation. The
feature dimensions of both text and audio are 768.
**Training and Inference Settings.** In this work, we delve
into both single-query and multi-query training. For the _M_ query annotations _Q_ = _{q_ _i_ _}_ _[M]_ _i_ =1 [associated with video] _[ V]_ [, we]
specify the number of queries fed into model training at a
time to be _m_ . When _m_ = 1, this corresponds to single query
training, designated as **UniSDNet-S** . Conversely, for multiquery training, where _m >_ 1, specifically when _m_ = _M_,
all queries relating to video _V_ are simultaneously fed into
the model, referred to as **UniSDNet-M** . It is important
to underscore that during the inference phase, regardless
of UniSDNet-S or UniSDNet-M, _the evaluation process is a_
_fair single query input_ that determines the prediction of a
uniquely corresponding moment, consistent with the conventional settings of the NLVG & SLVG tasks [3], [7], [19].
We use the AdamW [67] to optimize the proposed
model. For ActivityNet Captions and TACoS datasets, the
learning rate and batch size are set to 8 _×_ 10 _[−]_ [4] and 12,
respectively. For Charades-STA dataset, we set the learning rate and batch size to 1 _×_ 10 _[−]_ [4], and 48, respectively.
We train the model (whether UniSDNet-S or UniSDNet-M)
with the upper-limit of 15 epochs on ActivityNet Captions
and Charades-STA datasets and 200 epochs on TACoS. All
experiments are conducted with a GeForce RTX 2080Ti GPU.


**5.2** **Comparison with state-of-the-arts for NLVG Task**


We compare our UniSDNet with the state-of-the-art methods for _**NLVG**_ and divide them into two groups. **1) Proposal-**
**free methods** : VSLNet [17], LGI [19], DRN [18], CPNet [7], VSLNet-L [15], BPNet [23], VGCL [5], METML [6],
MA3SRN [8]. **2) Proposal-based methods** : 2D-TAN [11],
CSMGAN [9], MS-2D-TAN [12], MSAT [20], RaNet [22],
I [2] N [24], FVMR [13], SCDM [29], MMN [25], MGPN [30],



_R_ @ _h, IoU_ @ _u_ = [1]

_N_ _q_



_N_ _q_
� _r_ ( _h, u, q_ _i_ ) _,_ (9)


_i_ =1



where _N_ _q_ denotes the number of queries in the test set
and _q_ _i_ represents the _i_ -th query. In the top _h_ predicted
moments of query _q_ _i_, if the moment IoU between prediction and groundtruth is larger than _u_, _r_ ( _h, u, q_ _i_ ) equals 1;
otherwise, _r_ ( _h, u, q_ _i_ ) =0. Specifically, we set _h ∈{_ 1 _,_ 5 _}_ and
_u ∈{_ 0 _._ 3 _,_ 0 _._ 5 _,_ 0 _._ 7 _}_ . Also, we use _mIoU_, the average IoU
between the prediction and groundtruth across the test set,
as an indicator to compare overall performance:



_mIoU_ = [1]

_N_ _q_



_N_ _q_
� _IoU_ _i_ _,_ (10)


_i_ =1



where _N_ _q_ is the total number of queries, and _IoU_ _i_ is the IoU
value of the predicted moment for the _i_ -th query.
**Hyperparameter Settings.** Table 3 shows hyperparameter settings of UniSDNet. For data preparation, we evenly
sample 64 and 128 video clips for ActivityNet Captions
dataset with C3D features, and 16, 16, and 64 video clips
for the Charades-STA dataset with VGG, C3D, and I3D
features, respectively. In the static module, we conduct two
ResMLP blocks ( _N_ =2), and feature hidden size is set to
1024. In the dynamic module, DTFNet has two graph layers,
Based on the average clips of target moments in training
set, hyperparameter _k_ in Eq. 2 – dividing value between
short and long distances in video graph – is set to 16. More
discussion and ablation studies of _k_ are in Appendix B.1. We
empirically set hyperparameter _γ_ to 10.0, Gaussian kernels
number _h_ to 50, and generate _h_ biases with equal steps
from 0 with step 0.1. For dynamic filter _F_ _filter_, settings
of convolution layers, kernel size, and hidden size for 2D
proposal generation are listed in Table 3. Parameters size
of S [3] Net (Section 3.2), DTFNet (Section 3.3) and proposal
generation (Section 3.4) are also provided in Table 3.
**Implementation Details.** For a fair comparison, we utilize the same video features provided by 2D-TAN [11],


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 10


TABLE 4
Comparison with the state-of-the-arts on the _ActivityNet Captions_ and _TACoS_ datasets for _NLVG_ task. _‡_ denotes multi-query training mode, others
are single-query training mode. UniSDNet-S is single-query training result, and UniSDNet-M is multi-query training result. We evaluate our model
with two different text feature: GloVe [49] and BERT [62].







|Col1|Methods|Venue|Text|Video|ActivityNet Captions|Col7|Col8|TACoS|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||**Methods**|**Venue**|**Text**|**Video**|**R@1, IoU@**<br>**0.3**<br>**0.5**<br>**0.7**|**R@5, IoU@**<br>**0.3**<br>**0.5**<br>**0.7**|**mIoU**|**R@1, IoU@**<br>**0.3**<br>**0.5**<br>**0.7**|**R@5, IoU@**<br>**0.3**<br>**0.5**<br>**0.7**|**mIoU**|
|proposal-free|VSLNet [17]<br>LGI [19]<br>CPNet [7]<br>VSLNet-L [15]<br>VGCL [5]<br>METML [6]<br>MA3SRN [8]|_ACL’20_<br>_CVPR’20_<br>_AAAI’21_<br>_TPAMI’21_<br>_ACM MM’22_<br>_EACL’23_<br>_TMM’23_|GloVe<br>-<br>GloVe<br>GloVe<br>GloVe<br>BERT<br>GloVe|C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>I3D<br>C3D+Object|63.16<br>43.22<br>26.16<br>58.52<br>41.51<br>23.07<br>-<br>40.56<br>21.63 -<br>-<br>43.86<br>27.51<br>60.57<br>42.96<br>25.68<br>60.61<br>43.74<br>27.04<br>-<br>51.97<br>31.39|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>84.05<br>68.11|43.19<br>41.13<br>40.65<br>44.06<br>43.34<br>44.05<br>-|29.61<br>24.27<br>20.03<br>-<br>-<br>-<br>42.61<br>28.29<br>-<br>47.11<br>36.34<br>**26.42**<br>-<br>-<br>-<br>-<br>-<br>-<br>47.88<br>37.65<br>-|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>66.02<br>54.27<br>-|24.11<br>-<br>28.69<br>36.61<br>-<br>-<br>-|
|proposal-based|2D-TAN [11]<br>_AAAI’20_<br>CSMGAN [9]<br>_ACM MM’20_<br>MS-2D-TAN [12]<br>_TPAMI ’21_<br>MSAT [20]<br>_CVPR’21_<br>RaNet [22]<br>_EMNLP’21_<br>I2N [24]<br>_TIP’21_<br>FVMR [13]<br>_ICCV’21_<br>SCDM [29]<br>_TPAMI’22_<br>MGPN [30]<br>_SIGIR’22_<br>SPL [16]<br>_ACM MM’22_<br>DCLN [26]<br>_ICMR’22_<br>CRaNet [10]<br>_TCSVT’23_<br>PLN [27]<br>_ACM MM’23_<br>M2DCapsN [33]<br>_TNNLS’23_<br>MMN_‡_ [25]<br>_AAAI’22_<br>PTRM_‡_ [14]<br>_AAAI’23_<br>DFM_‡_ [63]<br>_ACM MM’23_<br>**UniSDNet-S (Ours)**<br>**UniSDNet-S (Ours)**<br>**UniSDNet-M (Ours)**<br>**UniSDNet-M (Ours)**|2D-TAN [11]<br>_AAAI’20_<br>CSMGAN [9]<br>_ACM MM’20_<br>MS-2D-TAN [12]<br>_TPAMI ’21_<br>MSAT [20]<br>_CVPR’21_<br>RaNet [22]<br>_EMNLP’21_<br>I2N [24]<br>_TIP’21_<br>FVMR [13]<br>_ICCV’21_<br>SCDM [29]<br>_TPAMI’22_<br>MGPN [30]<br>_SIGIR’22_<br>SPL [16]<br>_ACM MM’22_<br>DCLN [26]<br>_ICMR’22_<br>CRaNet [10]<br>_TCSVT’23_<br>PLN [27]<br>_ACM MM’23_<br>M2DCapsN [33]<br>_TNNLS’23_<br>MMN_‡_ [25]<br>_AAAI’22_<br>PTRM_‡_ [14]<br>_AAAI’23_<br>DFM_‡_ [63]<br>_ACM MM’23_<br>**UniSDNet-S (Ours)**<br>**UniSDNet-S (Ours)**<br>**UniSDNet-M (Ours)**<br>**UniSDNet-M (Ours)**|GloVe<br>GloVe<br>GloVe<br>-<br>GloVe<br>GloVe<br>GloVe<br>GloVe<br>GloVe<br>GloVe<br>GloVe<br>GloVe<br>GloVe<br>GloVe<br>BERT<br>BERT<br>BERT<br>GloVe<br>BERT<br>GloVe<br>BERT|C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D<br>C3D|59.45<br>44.51<br>26.54<br>68.52<br>49.11<br>29.15<br>61.04<br>46.16<br>29.21<br>-<br>48.02<br>31.78<br>-<br>45.59<br>28.67<br>-<br>-<br>-<br>60.63<br>45.00<br>26.85<br>55.25<br>36.90<br>20.28<br>-<br>47.92<br>30.47<br>-<br>52.89<br>32.04<br>65.58<br>44.41<br>24.80<br>-<br>47.27<br>30.34<br>59.65<br>45.66<br>29.28<br>61.53<br>47.03<br>29.99<br>-<br>48.59<br>29.26<br>66.41<br>50.44<br>31.18<br>-<br>45.92<br>32.18<br>68.59<br>52.73<br>31.08<br>68.66<br>52.35<br>32.25<br>74.07<br>57.67<br>35.64<br>**75.85**<br>**60.75**<br>**38.88**|85.53<br>77.13<br>61.96<br>87.68<br>77.43<br>59.63<br>87.30<br>78.80<br>60.85<br>-<br>78.02<br>63.18<br>-<br>75.93<br>62.97<br>-<br>-<br>-<br>86.11<br>77.42<br>61.04<br>78.79<br>66.84<br>42.92<br>-<br>78.15<br>63.56<br>-<br>82.65<br>67.21<br>84.65<br>74.04<br>56.67<br>-<br>78.84<br>63.51<br>85.66<br>76.65<br>63.06<br>-<br>76.64<br>62.83<br>-<br>79.50<br>64.76<br>-<br>-<br>-<br>-<br>-<br>-<br>89.57<br>84.19<br>72.52<br>89.74<br>83.35<br>70.61<br>90.49<br>84.46<br>72.47<br>**91.17**<br>**85.34**<br>**74.01**|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>44.12<br>-<br>-<br>47.68<br>-<br>50.13<br>50.22<br>53.68<br>**55.47**|37.29<br>25.32<br>-<br>33.90<br>27.09<br>-<br>45.61<br>35.77<br>23.44<br>48.79<br>37.57<br>-<br>43.34<br>33.54<br>-<br>31.47<br>29.25<br>-<br>41.48<br>29.12<br>-<br>27.64<br>23.27<br>-<br>48.81<br>36.74<br>-<br>42.73<br>32.58<br>-<br>44.96<br>28.72<br>-<br>47.86<br>37.02<br>-<br>43.89<br>31.12<br>-<br>46.41<br>32.58<br>-<br>39.24<br>26.17<br>-<br>-<br>-<br>-<br>40.04<br>28.57<br>14.77<br>51.44<br>36.37<br>23.47<br>53.46<br>36.24<br>23.48<br>53.59<br>38.34<br>23.79<br>**55.56**<br>**40.26**<br>24.12|57.81<br>45.04<br>-<br>53.98<br>41.22<br>-<br>69.11<br>57.31<br>36.09<br>67.63<br>57.91<br>-<br>67.33<br>55.09<br>-<br>52.65<br>46.08<br>-<br>64.53<br>50.00<br>-<br>40.06<br>33.49<br>-<br>71.46<br>59.24<br>-<br>64.30<br>50.17<br>-<br>66.13<br>51.91<br>-<br>70.78<br>58.39<br>-<br>65.11<br>52.89<br>-<br>66.32<br>52.91<br>-<br>62.03<br>47.39<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>76.56<br>61.06<br>36.22<br>76.96<br>63.06<br>36.34<br>**79.01**<br>**64.83**<br>36.89<br>77.08<br>64.01<br>**37.02**|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>29.70<br>-<br>-<br>-<br>27.35<br>35.83<br>36.47<br>37.54<br>**38.88**|


TABLE 5
Comparison with the state-of-the-arts on the _Charades-STA_ dataset for _NLVG_ task. _‡_ denotes multi-query training mode. Both MMN and our
method originate from the exploitation of 2D temporal map. The single-query (UniSDNet-S) and multi-query (UniSDNet-M) training results on this
dataset are closest compared to the other two NLVG datasets, due to its distribution of the query number concentrated in 1 (as shown in Fig. 5).








|Video Feature: VGG Video Feature: C3D Video Feature: I3D<br>Methods<br>R@1, IoU@ R@5, IoU@ R@1, IoU@ R@5, IoU@ R@1, IoU@ R@5, IoU@<br>mIoU mIoU mIoU<br>0.5 0.7 0.5 0.7 0.5 0.7 0.5 0.7 0.5 0.7 0.5 0.7|Col2|Col3|Col4|Col5|Video Feature: C3D|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
|**Methods**<br>**Video Feature: VGG**<br>**Video Feature: C3D**<br>**Video Feature: I3D**<br>**R@1, IoU@**<br>**R@5, IoU@**<br>**mIoU**<br>**R@1, IoU@**<br>**R@5, IoU@**<br>**mIoU**<br>**R@1, IoU@**<br>**R@5, IoU@**<br>**mIoU**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**|**Methods**<br>**Video Feature: VGG**<br>**Video Feature: C3D**<br>**Video Feature: I3D**<br>**R@1, IoU@**<br>**R@5, IoU@**<br>**mIoU**<br>**R@1, IoU@**<br>**R@5, IoU@**<br>**mIoU**<br>**R@1, IoU@**<br>**R@5, IoU@**<br>**mIoU**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**|**Methods**<br>**Video Feature: VGG**<br>**Video Feature: C3D**<br>**Video Feature: I3D**<br>**R@1, IoU@**<br>**R@5, IoU@**<br>**mIoU**<br>**R@1, IoU@**<br>**R@5, IoU@**<br>**mIoU**<br>**R@1, IoU@**<br>**R@5, IoU@**<br>**mIoU**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**<br>**0.5**<br>**0.7**|**R@5, IoU@**<br>**0.5**<br>**0.7**|**mIoU**|**R@1, IoU@**<br>**0.5**<br>**0.7**|**R@5, IoU@**<br>**0.5**<br>**0.7**|**mIoU**|**R@1, IoU@**<br>**0.5**<br>**0.7**|**R@5, IoU@**<br>**0.5**<br>**0.7**|**R@5, IoU@**<br>**0.5**<br>**0.7**|
|proposal-<br>free|DRN [18]<br>LGI [19]<br>BPNet [23]<br>CPNet [7]|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-|45.40<br>26.40<br>-<br>-<br>38.25<br>20.51<br>40.32<br>22.47|88.01<br>55.38<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>38.03<br>37.36|53.09<br>31.75<br>59.46<br>35.48<br>50.75<br>31.64<br>60.27<br>38.74|89.06<br>60.05<br>-<br>-<br>-<br>-<br>-<br>-|-<br>51.38<br>46.34<br>52.00|
|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|proposal-based<br>2D-TAN [11]<br>42.80<br>23.25<br>80.54<br>54.14<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>MS-2D-TAN [12]<br>45.65<br>27.20<br>**86.72**<br>56.42<br>-<br>41.10<br>23.25<br>81.53<br>48.55<br>-<br>60.08<br>37.39<br>89.06<br>59.17<br>-<br>FVMR [13]<br>-<br>-<br>-<br>-<br>-<br>38.16<br>18.22<br>82.18<br>44.96<br>-<br>55.01<br>33.74<br>89.17<br>57.24<br>-<br>I2N [24]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>56.61<br>34.14<br>81.48<br>55.19<br>-<br>CPL [21]<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>49.05<br>22.61<br>84.71<br>52.37<br>-<br>PLN [27]<br>45.43<br>26.26<br>86.32<br>57.02<br>41.28<br>-<br>-<br>-<br>-<br>-<br>56.02<br>35.16<br>87.63<br>62.34<br>49.09<br>PTRM_‡_ [14]<br>47.77<br>28.01<br>-<br>-<br>42.77<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>CRaNet [10]<br>47.12<br>27.39<br>83.51<br>58.33<br>-<br>-<br>-<br>-<br>-<br>-<br>60.94<br>**41.32**<br>**89.97**<br>65.19<br>-<br>M2DCapsN [33]<br>43.17<br>25.13<br>79.35<br>55.86<br>-<br>40.81<br>23.98<br>77.93<br>53.52<br>-<br>55.03<br>31.61<br>84.33<br>63.71<br>-<br>MMN_‡_ [25]<br>47.31<br>27.28<br>83.74<br>58.41<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**UniSDNet-S (Ours)**<br>47.34<br>27.45<br>84.68<br>58.41<br>43.32<br>48.71<br>27.31<br>82.77<br>57.58<br>43.16<br>59.41<br>38.58<br>89.52<br>70.65<br>52.07<br>**UniSDNet-M (Ours)**<br>**48.41**<br>**28.33**<br>84.76<br>**59.46**<br>**44.41**<br>**49.57**<br>**28.39**<br>84.70<br>**58.49**<br>**44.29**<br>**61.02**<br>39.70<br>**89.97**<br>**73.20**<br>**52.69**|



SPL [16], DCLN [26], CPL [21], PTRM [14], CRaNet [10],
PLN [27], M [2] DCapsN [33], DFM [63]. The best and secondbest results are marked in **bold** and underlined in experimental tables. The detailed test results of _R_ @1 _, IoU_ @ {0.3,
0.5, 0.7} on three NLVG datasets are reported in Table 4
and Table 5. Since most works do not report _R_ @1 _, IoU_ @0 _._ 1
performance, we have removed it from the table. Notably,
our method performs well on all metrics on the three NLVG
datasets. For more prediction distributions of our model and
other existing methods on NLVG task, see Appendix A.


_5.2.1_ _Results on the ActivityNet Captions dataset_


The ActivityNet Captions is the largest open domain dataset
for NLVG. As shown in Table 4, our UniSDNet-S has
achieved satisfactory performance to current SOTA meth


ods, but at a very low cost of 0.53M for static modules
and 0.68M for dynamic modules (Table 3). If the -M (multiquery) mode is utilized in training, there will be a significant
increase in performance (UniSDNet-M achieves the best
performance with scores of 38.88 and 55.47 in terms of
_R_ @1 _, IoU_ @0 _._ 7, and _mIoU_, respectively), note that regardless of UniSDNet-S and -M, they are tested in the same
fair way, _i.e_ ., single-query reasoning at a time. And a lot
of work has also released M-query training modes such
as MMN [25], PTRM [14] and DFM [68], but their performances are significantly worse than these of our UniSDNetM due to our efficient modelling of multimodal information. Since we adopt a proposal-based backend to favor
modal alignment between the video moment and the query,
we prefer to compare our method with recently proposed


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 11


TABLE 6
Comparison with state-of-the-art methods on three datasets for _SLVG_ task, in which _Charades-STA Speech_ _[∗]_ and _TACoS Speech_ _[∗]_ are our new
collected datasets, described in Section 4. _‡_ denotes multi-query training mode, _[†]_ denotes our reproduced results using the released code.




















|Dataset Method|Audio Feature|Video Feature|R@1, IoU@<br>0.3 0.5 0.7|R@5, IoU@<br>0.3 0.5 0.7|mIoU|
|---|---|---|---|---|---|
|ActivityNet Speech [5]<br>VGCL [5]<br>ISL [31]<br>VSLNet [17]<br>VSLNet_†_<br>MMN_‡_ [12]_†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>|CPC [42]<br>Mel Spectrogram<br>Mel Spectrogram<br>Data2vec [44]<br>Data2vec<br>Data2vec<br>Data2vec|C3D|49.80<br>30.05<br>16.63<br>49.46<br>30.26<br>15.22<br>46.75<br>29.08<br>16.24<br>51.02<br>30.38<br>17.45<br>51.98<br>35.69<br>20.77<br>64.83<br>47.82<br>27.49<br>**72.27**<br>**56.29**<br>**33.29**|-<br>-<br>-<br>82.28<br>63.73<br>35.48<br>-<br>-<br>-<br>-<br>-<br>-<br>85.46<br>75.29<br>56.87<br>90.69<br>84.16<br>72.12<br>**90.41**<br>**84.28**<br>**72.42**|35.36<br>34.52<br>34.01<br>37.04<br>37.81<br>47.31<br>**52.22**|
|ActivityNet Speech [5]<br>VGCL [5]<br>ISL [31]<br>VSLNet [17]<br>VSLNet_†_<br>MMN_‡_ [12]_†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>|Data2vec|I3D|53.06<br>32.43<br>17.69<br>53.23<br>35.53<br>20.09<br>64.16<br>49.28<br>27.94<br>**69.83**<br>**54.93**<br>**33.20**|-<br>-<br>-<br>83.77<br>72.76<br>55.88<br>90.05<br>83.38<br>67.09<br>**90.38**<br>**84.21**<br>**71.76**|37.22<br>38.24<br>47.47<br>**51.19**|
|_Charades-STA Speech∗_4<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>|Data2vec|VGG|50.27<br>38.76<br>23.25<br>56.16<br>42.74<br>24.14<br>59.19<br>45.08<br>25.91<br>**60.73**<br>**46.37**<br>**26.72**|-<br>-<br>-<br>91.25<br>80.96<br>55.97<br>92.02<br>82.47<br>57.34<br>**92.66**<br>**82.31**<br>**57.66**|35.78<br>39.15<br>41.26<br>**42.28**|
|_Charades-STA Speech∗_4<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>|Data2vec|C3D|52.42<br>40.70<br>22.36<br>52.28<br>39.44<br>21.80<br>56.37<br>41.85<br>24.06<br>**58.20**<br>**43.66**<br>**25.05**|-<br>-<br>-<br>85.24<br>74.16<br>48.23<br>86.61<br>76.24<br>52.39<br>**92.23**<br>**82.15**<br>**55.86**|36.91<br>36.09<br>39.21<br>**40.56**|
|_Charades-STA Speech∗_4<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>|Data2vec|I3D|65.46<br>47.55<br>28.98<br>64.27<br>51.75<br>31.26<br>67.37<br>53.63<br>33.87<br>**67.45**<br>**53.82**<br>**34.49**|-<br>-<br>-<br>93.46<br>85.90<br>62.69<br>94.54<br>87.45<br>67.77<br>**94.81**<br>**87.90**<br>**69.30**|45.40<br>45.84<br>48.13<br>**48.27**|
|_TACoS Speech∗_4<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**|Data2vec|VGG|29.39<br>20.59<br>10.92<br>30.12<br>20.07<br>11.62<br>38.94<br>23.07<br>11.02<br>**40.29**<br>**26.34**<br>**12.85**|-<br>-<br>-<br>56.24<br>40.64<br>22.17<br>68.13<br>50.31<br>24.97<br>**67.36**<br>**51.41**<br>**26.24**|21.10<br>21.21<br>27.59<br>**28.40**|
|_TACoS Speech∗_4<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**|Data2vec|C3D|38.14<br>27.87<br>16.35<br>31.72<br>23.82<br>12.55<br>47.04<br>31.77<br>17.42<br>**51.66**<br>**37.77**<br>**20.44**|-<br>-<br>-<br>59.16<br>45.36<br>22.89<br>73.78<br>60.88<br>32.69<br>**76.38**<br>**63.48**<br>**33.64**|27.28<br>22.58<br>33.25<br>**36.86**|
|_TACoS Speech∗_4<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**<br>VSLNet~~_†_~~<br>MMN_‡†_<br>**UniSDNet-S**<br>**UniSDNet-M**|Data2vec|I3D|30.54<br>18.87<br>10.67<br>29.39<br>20.37<br>10.82<br>40.11<br>25.19<br>11.37<br>**41.74**<br>**26.34**<br>**12.25**|-<br>-<br>-<br>54.46<br>42.41<br>21.14<br>67.58<br>50.36<br>24.62<br>**69.26**<br>**51.26**<br>**24.94**|19.88<br>20.86<br>27.93<br>**29.27**|



proposal-based methods, especially MMN [25], PTRM [14],
_etc_ . And our research on recent NLVG work has found that
proposal-based methods predominate, as shown in Table 4.
Compared to other proposal-based methods, our UniSDNetM performs the best and has substantial improvements in all
metrics due to the unique static and dynamic modes.


_5.2.2_ _Results on the TACoS dataset_

TACoS (Cooking dataset) has the longest video length (approx. 5 min) and the highest number of events ( _>_ 100)
per video (more details in Table 1). As shown in Table 4, the proposed UniSDNet-S (BERT) performs well with
_R_ @1 _, IoU_ @0 _._ 3 being 53.46, and UniSDNet-M achieves the
best results across all metrics ( _e.g_ ., 38.88 on _mIoU_ ), indicating that our model is better able to construct multiquery multimodal environmental semantics for video understanding. For proposal-based method MSAT [20] with
good performance of 37.57 on _R_ @1 _, IoU_ @0 _._ 5 . It focuses
only on static feature interactions with a transformer encoder. In contrast, our UniSDNet-M uses the lightweight
MLP- and dynamic GCN-based network to construct deeper
cross-modal associations, and performs better than MSAT,
achieving improvements of 6.77 and 2.69 in _R_ @1 _, IoU_ @0 _._ 3
and _R_ @1 _, IoU_ @0 _._ 5 metrics, respectively.


_5.2.3_ _Results on the Charades-STA dataset_

For the Charades-STA dataset, we report the fair comparison
results of our method under VGG, C3D, and I3D features in



Table 5. Notably, the different characteristics of CharadesSTA compared to the other two NLVG datasets are analysed in Fig. 5, Fig. 6 and Section 4.3, including smallest
query number size, shortest query length and shortest video
duration with an average of 30.60s, so that more subtle
human movements need to be identified, resulting in that
the models are sensitive to different visual features. Despite under this limitation, for the VGG and C3D visual
features, our method achieves the best performance on the
stringent metric _R_ @1, _e.g_ ., 28.33 and 28.39 _R_ @1 _, IoU_ @0 _._ 7
on VGG and C3D feature, respectively. For the I3D video
features, our UniSDNet-M achieves an outstanding record
in _R_ @1 _, IoU_ @0 _._ 5 and _R_ @5 _, IoU_ @0 _._ 7, that are 61.02 and
73.20, demonstrating the robustness and generalization of
our model. Moreover, we specifically make a fair comparison of ours with MMN [25] based on the same 2D temporal
proposal map. Compared with MMN, our UniSDNet has
improvements of 1.05 _↑_ in _R_ @1 _, IoU_ @0 _._ 7 with VGG feature.


**5.3** **Comparison with state-of-the-arts for SLVG Task**


We compare our UniSDNet with the state-of-the-art methods for SLVG, including VGCL [5], SIL [31], VSLNet [17]
and MMN [25] methods, where VGCL and SIL both have
been assessed on the _ActivityNet Speech_ dataset. In order to
make fair comparison and add richer results, we reconstruct
VSLNet [17] and MMN [25] models for the SLVG task,


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 12



where VSLNet is a classic proposal-free method, and MMN
is a classic proposal-based method.
In addition, existing NLVG methods evaluated different
video features in the experiments, including VGG, C3D, and
I3D fatures. To validate our UniSDNet on the SLVG task
dataset effect, we evaluate it using all existing available
video features. Since there is no existing work reporting
VGG video feature results on the ActivityNet Captions
and ActivityNet Speech datasets, we followed them and
do not report this result. And in Table 6, except for the
video features presented in the implementation details, the
other video features on different datasets are taken from

the MS-2D-TAN [12]. The detailed test results on ActivityNet Speech dataset and our newly collected two datasets
Charades-STA Speech and TACoS Speech are listed in Table 6. Our method perform the best stably under different
features. See Appendix A for visualizations of the results.


_5.3.1_ _Results on the ActivityNet Speech dataset_
The results on the ActivityNet Speech dataset are delineated
in Table 6, where we evaluate a broader array of audio features, including Contrastive Predictive Coding (CPC) [42],
Mel Spectrogram, and Data2vec [44] audio features. This
analysis aims to elucidate the variations in performance
attributable to different pre-extracted audio features. It is
observed that our UniSDNet-S and UniSDNet-M achieves
state-of-the-art performances across all evaluated metrics
( _e.g_ ., 33.29 on _R_ @1 _, IoU_ @0 _._ 7 ). Compared to VGCL [5] and
ISL [31], our UniSDNet-M exhibits a remarkable enhancement, improving by approximately 20 points in _mIoU_ .
This significant gain underscores the superior efficacy of
our integrated static and dynamic framework in addressing
the SLVG task. The reconstructed VSLNet method, which
utilizes Data2vec audio features, demonstrates an improvement of approximately 1 point in _R_ @1 _, IoU_ @0 _._ 7 compared
to the VSLNet method that uses audio Mel Spectrogram as
input. When we account for the differences in input audio
features and utilize the common Data2vec audio features,
our UniSDNet-M outperforms VSLNet and MMN with
scores of 15.84 and 12.52 on _R_ @1 _IoU_ @0 _._ 7, respectively. This
highlights the effectiveness of our method in associating
cross-modal information between audio and video.


_5.3.2_ _Results on Two New Speech datasets_
To advance research in SLVG, we conduct experiments on
newly collected datasets, _Charades-STA Speech_ and _TACos_
_Speech_, as detailed in Section 4 (Table 2) and depicted in Table 6. Our UniSDNet-M achieves SOTA performance across
all evaluated SLVG datasets, ( _e.g_ ., _R_ @1 _, IoU_ @0 _._ 7 of 34.49
and 20.44 on the Charades-STA Speech and TACoS Speech,
respectively). This underscores its exceptional versatility
across a variety of dataset environments. When compared
to VSLNet, our UniSDNet-M exhibits superior performance,
enhancing the _mIoU_ by margins of 2.87 and 9.58 on the
Charades-STA Speech and TACoS Speech datasets, respectively. Furthermore, in a direct comparison with the baseline model MMN, our UniSDNet-M demonstrates significantly better performance, with _mIoU_ improvements of
3.13 and 14.28 on the Charades-STA Speech and TACoS
Speech datasets, respectively. These improvements further
highlight the efficacy of our static and dynamic framework



in bridging cross-modal information between audio and
video, showcasing not only its accuracy but also its ability
to effectively associate diverse modalities.


**5.4** **Model Efficiency**


To better distinguish our model from other proposal-based
models, we conduct an efficiency comparison on the ActivityNet Captions dataset in both single-query and multiquery training modes. The results are presented in Table 8.
Additionally, the specific parameters of the various modules
within our UniSDNet are elaborated in Fig. 7 and Table 3.
From the analysis in Table 8, it is evident that our UniSDNet
offers moderate parameters and exhibits the fastest inference speed 0.009 s/query, regardless of the training mode
(single-query or multi-query). It is worth noting that our
UniSDNet-M has only half the number of model parameters
compared to the proposal-based multi-query MMN and
PTRM models. Nonetheless, our UniSDNet-M achieves a
remarkable 35.71% improvement in efficiency over MMN.
Compared to the PTRM approach that employs multiquery training, our UniSDNet exhibits notable accuracy
enhancement, with an increase of 10.31% in _R_ @1 _, IoU_ @0 _._ 5 .
Meanwhile, under single-query training, UniSDNet-S also
has 9.02% of performance gain on R@1, IoU@0.5 while being
4.67× faster than single-query training SOTA method.


**5.5** **Ablation Studies**


In this section, we conduct in-depth ablation to analyze
each component and specified parameter of UniSDNet. The
experiments are conducted in multi-query training mode.


_5.5.1_ _Ablation Study on Static and Dynamic Modules_

We remove the static (Section 3.2) and dynamic modules
(Section 3.3) separately to investigate their contribution to
cross-modal associativity modeling in our model. The results of NLVG and SLVG are reported in Table 7. In NLVG,
the single static module outperforms the baseline (without
static and dynamic modules) with improvements of 4.91 and
7.48 in _R_ @1 _, IoU_ @0 _._ 7 and _mIoU_, respectively. In addition,
the single dynamic module exhibits improvements of 7.41
and 8.32 than the baseline on _R_ @1 _, IoU_ @0 _._ 7 and _mIoU_,
which demonstrates its effectiveness of dynamic temporal
modeling in the video. When combining the static and
dynamic modules, all the performance metrics are further
improved, such as setting new SOTA records 38.88 in
_R_ @1 _, IoU_ @0 _._ 7 and 55.47 in _mIoU_ for NLVG. In SLVG, we
can observe similar conclusions. These results demonstrate
that both static and dynamic modules indeed have a mutual
promoting effect on improving accuracy.


_5.5.2_ _Ablation Study on Static Network Variants_

In the static network, transformer architecture [28] or the
recent S4 architecture [69] can also be used as long-range
filter. We have tested the effect of Transformer or S4 as a
static network as shown in Table 9. From the results, in
terms of performance and efficiency, Transformer is close to
our method, but our results are better. We speculate that the
reason is that our network also includes the second stage
of graph filtering. The static network uses a lightweight


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 13


TABLE 7
Ablation studies of the static (Section 3.2) and dynamic (Section 3.3) modules on the _ActivityNet Captions_ and _ActivityNet Speech_ datasets.






|Task Static Dynamic|R@1, IoU@0.3 R@1, IoU@0.5 R@1, IoU@0.7|R@5, IoU@0.3 R@5, IoU@0.5 R@5, IoU@0.7|
|---|---|---|
|**NLVG**<br>✗<br>✗<br>✓<br>✗<br>✗<br>✓<br>✓<br>✓|61.22<br>44.46<br>26.76<br>72.32<br>55.18<br>31.67<br>72.74<br>55.99<br>34.17<br>**75.85**<br>**60.75**<br>**38.88**|87.19<br>78.63<br>63.60<br>90.99<br>84.65<br>71.46<br>90.51<br>83.95<br>71.42<br>**91.16**<br>**85.34**<br>**74.01**|
|**SLVG**<br>✗<br>✗<br>✓<br>✗<br>✗<br>✓<br>✓<br>✓|53.63<br>35.91<br>20.51<br>64.83<br>47.82<br>27.49<br>63.77<br>49.68<br>29.32<br>**72.27**<br>**56.29**<br>**33.29**|84.71<br>74.21<br>55.95<br>90.19<br>84.16<br>72.12<br>89.84<br>83.33<br>70.30<br>**90.41**<br>**84.28**<br>**72.42**|






|Infer. Speed<br>Method<br>(s/query)|R@1, IoU@<br>0.3 0.5 0.7|mIoU|
|---|---|---|
|Transformer [28]<br>0.024<br>S4 [69]<br>0.030<br>**Our S**3**Net**<br>0.009|75.17<br>59.98<br>38.38<br>70.41<br>55.11<br>34.93<br>**75.85**<br>**60.75**<br>**38.88**|54.97<br>51.40<br>**55.47**|










|Query|Method Model Size Infer. Speed (s/query) R@1, IoU@0.5|
|---|---|
|**Single**|2D-TAN [11]<br>21.62M<br>0.061<br>44.51<br>MS-2D-TAN [12]<br>479.46M<br>0.141<br>46.16<br>MSAT [20]<br>37.19M<br>0.042<br>48.02<br>MGPN [30]<br>5.12M<br>0.115<br>47.92<br>**UniSDNet-S (Ours)**<br>76.52M<br>**0.009**<br>**52.35**|
|**Multi**|MMN [25]<br>152.22M<br>0.014<br>48.59<br>PTRM [14]<br>152.25M<br>0.038<br>50.44<br>**UniSDNet-M (Ours)**<br>**76.52M**<br>**0.009**<br>**60.75**|












|55.18<br>47.82|58.14 58.78<br>53.29 53.96|NLVG SLVG<br>60.75<br>59.52 59.16<br>56.29<br>55.03 54.95|
|---|---|---|
|~~w/o Graph~~<br>~~GCN~~<br>~~GAT~~<br>~~D~~<br>~~MLP~~<br>~~Ours~~|~~w/o Graph~~<br>~~GCN~~<br>~~GAT~~<br>~~D~~<br>~~MLP~~<br>~~Ours~~|~~w/o Graph~~<br>~~GCN~~<br>~~GAT~~<br>~~D~~<br>~~MLP~~<br>~~Ours~~|







|2 Multi-query 1.56x UniSDNet-M<br>0 Single-query<br>8 Diameter<br>10.31%<br>6 2 8 32 64 ×10M<br>4 UniSDNet-S<br>4.67x<br>2<br>PTRM<br>9.02%<br>0<br>MGPN<br>8<br>MSAT<br>MMN<br>6 MS-2D-TAN<br>4 2D-TAN<br>8 16 25 71 111|Multi-query UniSDNet-M<br>1.56x<br>Single-query<br>Diameter<br>10.31%<br>2 8 32 64 ×10M<br>UniSDNet-S<br>4.67x<br>PTRM<br>9.02%<br>MGPN<br>MSAT<br>MMN<br>MS-2D-TAN<br>2D-TAN|Col3|Col4|Col5|
|---|---|---|---|---|
|8 16 25<br>71<br>111<br>4<br>6<br>8<br>0<br>2<br>4<br>6<br>8<br>0<br>2<br>2D~~-T~~AN<br>MS~~-~~2D~~-T~~AN<br>~~MSAT~~<br>MMN<br>MGPN<br>PTRM<br>UniSDNet~~-~~M<br>UniSDNet~~-~~S<br>10.31%<br>1.56x<br>9.02%<br>4.67x<br>Multi~~-~~query<br>Single~~-~~query<br>Diameter<br>2 8<br>32<br>64 ×10M|2D~~-T~~AN<br>MS~~-~~2D~~-T~~AN<br>~~MSAT~~<br>MMN<br>MGPN<br>PTRM<br>UniSDNet~~-~~M<br>UniSDNet~~-~~S<br>10.31%<br>1.56x<br>9.02%<br>4.67x<br>Multi~~-~~query<br>Single~~-~~query<br>Diameter<br>2 8<br>32<br>64 ×10M|25<br>7|1<br>1|11|


Transformer as a static network increases the weight and
instability factors [47] of the network.


_5.5.3_ _Dynamic Network Variants and Hyperparameters_


**Different Graph Networks.** Our dynamic network implementation is based on the graph structure. We compare it
with the currently popular graph structures, GCN [70] and
GAT [54], and test other variants of our graph filter, namely
**D** and **MLP** . Additionally, our proposed temporal filtering
graph contains more parametric details, which are analyzed





weight for GAT. The variant _d_ [(] _ij_ _[l]_ [)] [= (1] _[ −]_ _[a]_ [(] _ij_ _[l]_ [)] [)] _[ · ||][j][ −]_ _[i][|| ∈]_
R has been defined in Section 3.3, which denotes the joint
clue of temporal distance and relevance between two nodes.
In particular, MLP( _d_ [(] _ij_ _[l]_ [)] [)] _[ ∈]_ [R] _[h]_ [ and] [ Φ(] _[d]_ [(] _ij_ _[l]_ [)] [)] _[ ∈]_ [R] _[h]_ [ are two]

different ways of expanding the _d_ [(] _ij_ _[l]_ [)] [dimension.]
Observing Fig. 8, **w/o Graph** denotes the Dynamic
Network is removed from the whole framework, and its
performance is the worst. The vanilla **GCN** tracts all the
neighbor nodes equally with a convolution operation to
aggregate neighbor information. **GAT** is a weighted attention aggregation method [54]. Our method outperforms






TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 14



(b) Content, Boundary (Addition)







Cat **Conv or MaxPool**



TABLE 10
Different Gaussian kernel number _h_ and step _z_ on the ActivityNet
Captions dataset.

|#Kernels Step|R@1, IoU@<br>0.3 0.5 0.7|R@5, IoU@<br>0.3 0.5 0.7|mIoU|
|---|---|---|---|
|25<br>0.1<br>50<br>0.1<br>100<br>0.1<br>200<br>0.1|75.12<br>60.20<br>38.02<br>**75.62**<br>**60.75**<br>**38.88**<br>74.88<br>59.54<br>38.53<br>74.28<br>59.60<br>38.62|91.20<br>85.82<br>74.68<br>**90.94**<br>**85.34**<br>**74.01**<br>91.29<br>85.96<br>74.75<br>91.33<br>85.91<br>75.05|54.91<br>**55.47**<br>54.99<br>54.96|
|25<br>0.2<br>50<br>0.2<br>100<br>0.2<br>200<br>0.2|75.11<br>60.01<br>38.13<br>75.12<br>60.31<br>38.66<br>75.30<br>59.73<br>38.47<br>74.69<br>59.99<br>39.03|91.25<br>85.48<br>74.31<br>90.95<br>85.11<br>73.86<br>91.65<br>86.07<br>75.16<br>91.30<br>85.63<br>74.86|54.99<br>55.25<br>55.13<br>55.18|



TABLE 11
Different Gaussian coefficient _γ_ on the ActivityNet Captions dataset.

|R@1, IoU@ R@5, IoU@<br>Gaussian Coeffciient mIoU<br>0.3 0.5 0.7 0.3 0.5 0.7|Col2|Col3|Col4|
|---|---|---|---|
|5.0<br>**10.0**<br>25.0<br>50.0<br>75.0|75.76<br>60.80<br>39.23<br>**75.85**<br>**60.75**<br>**38.88**<br>75.87<br>60.77<br>39.30<br>75.84<br>60.98<br>38.83<br>75.74<br>60.57<br>38.63|91.14<br>85.43<br>74.33<br>**91.16**<br>**85.34**<br>**74.01**<br>91.16<br>85.23<br>74.06<br>91.04<br>85.27<br>73.98<br>90.98<br>85.26<br>73.86|55.51<br>**55.47**<br>55.52<br>55.51<br>55.29|
|average<br>75.81<br>60.77<br>38.97<br>91.10<br>85.31<br>74.05<br>55.46<br>standard deviation<br>0.06<br>0.15<br>0.28<br>0.08<br>0.08<br>0.17<br>0.10|average<br>75.81<br>60.77<br>38.97<br>91.10<br>85.31<br>74.05<br>55.46<br>standard deviation<br>0.06<br>0.15<br>0.28<br>0.08<br>0.08<br>0.17<br>0.10|average<br>75.81<br>60.77<br>38.97<br>91.10<br>85.31<br>74.05<br>55.46<br>standard deviation<br>0.06<br>0.15<br>0.28<br>0.08<br>0.08<br>0.17<br>0.10|average<br>75.81<br>60.77<br>38.97<br>91.10<br>85.31<br>74.05<br>55.46<br>standard deviation<br>0.06<br>0.15<br>0.28<br>0.08<br>0.08<br>0.17<br>0.10|



**GCN** and **GAT** by 2.61 and 1.97 on _R_ @1 _, IoU_ @0 _._ 5 for
NLVG, and by 3.00 and 2.33 on _R_ @1 _, IoU_ @0 _._ 5 for SLVG,
respectively. For **D** and **MLP**, we discuss the Gaussian filter
setup in our method. In the setting of **D**, we directly use
the message aggregation wight _f_ _ij_ [(] _[l]_ [)] [= 1] _[/]_ [(] _[d]_ [(] _ij_ _[l]_ [)] [+1)] [ to replace]
_f_ _ij_ [(] _[l]_ [)] [=] _[ F]_ _[filter]_ [(] _[d]_ [(] _ij_ _[l]_ [)] [)] [ in Eq.][ 3][, which indicates that we still con-]
sider the same joint clue of temporal distance and relevance
between two nodes _d_ [(] _ij_ _[l]_ [)] [but remove the Gaussian filtering]
calculation from our method. This replacement results in a
decrease of 1.23 and 1.26 on _R_ @1 _, IoU_ @0 _._ 5 for NLVG and
SLVG, respectively. **MLP** uses the operation MLP( _d_ [(] _ij_ _[l]_ [)] [)] [ to]

replace the Gaussian basis function _ϕ_ ( _d_ [(] _ij_ _[l]_ [)] [)] [ in Eq.][ 3][. In this]
way, we realize the convolution kernel rather than Gaussian
kernel in the dynamic filter. Compared to **Ours**, **MLP** has a
decreased performance of 1.59 and 1.34 on _R_ @1 _, IoU_ @0 _._ 5
for NLVG and SLVG, respectively. Overall, our proposed
dynamic filtering network offers irreplaceable benefits.
**Hyperparameters in the Dynamic Temporal Filter** _F_ _filter_ **.**
In this work, we employ the multi-kernel Gaussian Φ( _x_ ) =
exp( _−γ_ ( _x−z_ _k_ ) [2] ) _, k ∈_ [1 _, h_ ] (Section 3.3), and there are three
variables ( _z_ _k_ _, h, γ_ ): different bias _z_ _k_ for total _h_ Gaussian
kernels and a Gaussian coefficient _γ_ . To meet the constraint
of nonlinear correlated Gaussian kernels, we randomly set
biases at equal intervals ( _e.g_ ., 0.1 or 0.2) starting from 0.0,
sweep the value of from 25 to 200 and set the global range
of _z_ _k_ values to [0 _,_ 5] in our experiments, as shown in
Table 10. And we can find that the best setting is _h_ = 50, we
speculate that our method achieves the best results when
number of Gaussian kernels _h_ is close to the number of
graph nodes. Gaussian coefficient _γ_ reflects the amplitude
of the Gaussian kernel function that controls the gradient
descent speed of the function value. It can be found that
from Table 11, when _γ_ = 25 _._ 0, our model achieves the best
performance with _mIoU_ at 55.52. We also list the average
and standard deviation of the five experimental results and
select _γ_ = 10 _._ 0 as the empirical setting as its results are







(a) Content



|1 2|3|4 5|6 7|8|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||


(c) Content, Boundary (Concatenation)



Fig. 9. Different feature sampling strategies for 2D proposal generation.
(a) Only the content feature. (b) The content and boundary features are
fused by addition operation. (c) The content and boundary features are
fused with concatenation operation.

















|7<br>0<br>6<br>5<br>0<br>0 4<br>0 3<br>0 2<br>0|4.50 74.15 75.09 74.78 75.<br>5.05<br>60.75|Col3|Col4|
|---|---|---|---|
|0<br>0<br>0<br>0<br>0<br>0<br>7<br>5<br>3<br>6<br>4<br>2|9.23<br>58.85<br>59.82<br>59.90<br>60.<br>|9.23<br>58.85<br>59.82<br>59.90<br>60.<br>|9.23<br>58.85<br>59.82<br>59.90<br>60.<br>|
|0<br>0<br>0<br>0<br>0<br>0<br>7<br>5<br>3<br>6<br>4<br>2|38.88<br>8.22<br>37.78<br>38.32<br>38.11<br>38.<br>8.59<br>9.26|38.88<br>8.22<br>37.78<br>38.32<br>38.11<br>38.<br>8.59<br>9.26|38.88<br>8.22<br>37.78<br>38.32<br>38.11<br>38.<br>8.59<br>9.26|
|0<br>0<br>0<br>0<br>0<br>0<br>7<br>5<br>3<br>6<br>4<br>2|UniSDNet<br>MMN|UniSDNet<br>MMN|UniSDNet<br>MMN|
|0<br>0<br>0<br>0<br>0<br>0<br>7<br>5<br>3<br>6<br>4<br>2|1<br>2<br>3|4<br>5|6|


Fig. 10. The results across different graph layer on the AcvitivtyNet
Captions dataset for NLVG. From top to bottom, the metrics are
_R_ @1 _, IoU_ @0 _._ 3, _IoU_ @0 _._ 5, and _IoU_ @0 _._ 7, respectively.


closest to the average. To summarize, in our experiments,
the final settings of variables ( _h, γ_ ) are set to 50 and 10.0,
and _z_ _k_ is set at an equal interval of 0.1.
**Dynamic Graph Layer.** We investigate the influence of the
graph layer of our dynamic module. As shown in Fig. 10,
we observe that our model achieves the best result ( _e.g_ .,
_R_ @1 _, IoU_ @0 _._ 7 is 38.88) when the total number of graph
layer is set to 2. It is speculated that on the basis of informative context modelling by the static module, two-layers
dynamic graph module is enough for relational learning of
the video. Additionally, graph convolutional networks generally experience over-smoothing problem as the number of
layers increases, leading to a performance decline [71]. Our

_∼_
model exhibits good stability on the 1 6-th graph layers.


_5.5.4_ _Ablation Study on Proposals Generation_


To analyze the sensitivity of the feature sampling strategy
for 2D proposals generation, we evaluate the effects of
moment content and boundary features. As shown in Fig. 9,
we conduct experiments with different proposal generation
strategies: (a) only the content feature; (b) the addition of
content and two boundary features; (c) the concatenation
of content and boundary features. Here, the content feature
refers to Gen( _v_ _i_ _[L]_ _[, v]_ _i_ _[L]_ +1 _[,][ · · ·][, v]_ _j_ _[L]_ [)] [ with] [ Gen] [ being 1D Conv]
or MaxPool [11], where _v_ _i_ _[L]_ and _v_ _j_ _[L]_ [are the start] _[ i]_ [-th and]
ending _j_ -th video clip features, respectively. The experimental results for both NLVG and SLVG tasks are summarized
in Table 12. For NLVG, the MaxPool strategy outperforms
convolution, _e.g_ ., 38.88 _vs_ . 38.20 in terms of _R_ @1 _, IoU_ @0 _._ 7


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 15


TABLE 12
Comparison of different proposal generation strategies on the ActivityNet Captions and ActivityNet Speech datasets.


**Task** **Generation** **Features** **Fusion** **R@1, IoU@** **R@5, IoU@** **mIoU**



NLVG


SLVG




|R@1, IoU@<br>0.3 0.5 0.7|R@5, IoU@<br>0.3 0.5 0.7|mIoU|
|---|---|---|
|75.30<br>60.27<br>38.20<br>75.85<br>60.70<br>38.75<br>74.76<br>60.30<br>38.80<br>75.62<br>60.40<br>38.99<br>**75.85**<br>**60.75**<br>**38.88**<br>75.13<br>59.96<br>38.25|90.86<br>85.16<br>73.17<br>90.85<br>85.05<br>73.25<br>90.70<br>84.96<br>73.00<br>90.94<br>85.22<br>73.97<br>**91.16**<br>**85.34**<br>**74.01**<br>91.26<br>85.59<br>73.91|55.13<br>55.41<br>55.15<br>55.39<br>**55.47**<br>54.98|
|71.02<br>55.24<br>32.88<br>72.27<br>56.29<br>33.29<br>71.45<br>55.79<br>33.20<br>71.26<br>55.25<br>**33.74**<br>**72.60**<br>**56.64**<br>32.61<br>69.85<br>53.96<br>32.05|90.38<br>84.25<br>71.38<br>90.41<br>84.28<br>72.42<br>90.55<br>84.16<br>71.48<br>90.49<br>84.29<br>72.46<br>**90.82**<br>**84.89**<br>**72.48**<br>90.36<br>84.12<br>72.24|51.66<br>**52.22**<br>51.76<br>51.80<br>52.04<br>50.68|











TABLE 13
Performance comparison on QVHighlights _test_ split. _∗_ : introduce audio
modality.







(a) NLVG (on ActivityNet Captions)



|60<br>55.43 56.29|Col2|Col3|
|---|---|---|
|50<br>55<br>|47.82<br>52.62<br>54.77<br><br><br><br><br><br><br><br>VSLNet(Single)<br>30.38|47.82<br>52.62<br>54.77<br><br><br><br><br><br><br><br>VSLNet(Single)<br>30.38|
|50<br>55<br>|47.82<br>52.62<br>54.77<br><br><br><br><br><br><br><br>VSLNet(Single)<br>30.38|~~Full~~<br>r|


(b) SLVG (on ActivityNet Speech)









Fig. 11. Results of different training query number _m_ of a video for
NLVG and SLVG. Because the query number distribution of ActivityNet
Captions is concentrated in the [3,8] (Fig. 5), we test _m_ = 1 _,_ 3 _,_ 5 _,_ 8 .
When _m_ = 1, the training mode is single-query, and when _m >_ 1,
the training mode is multi-query. “Full” represents all query inputs for a
video simultaneously during training. A more detailed comparison of the
single-query and multi-query mode methods is given in Table 8.


when the model using content feature. Additionally, addition performs better than concatenation, _e.g_ ., 55.47 _vs_ . 54.98
when the model uses the content and boundary features.
SLVG shows similar results. Therefore, we use content and
boundary features to generate proposals through MaxPool
and Conv for both NLVG and SLVG.


_5.5.5_ _Ablation Study on Training Mode_
In this work, we adopt two training mode: single-query and
multi-query training, as described in experimental setup
part (Section 5.1, Training and Inference Mode). The number
of queries is an important variable, in order to explore the
effect of our UniSDNet in single-query and multi-query
modes, we conduct experiments with different number of
queries on the ActivityNet Captions and ActivityNet Speech
datasets. The results are shown in Fig. 11. It can be observed
that for single-query training, our model is comparable with
state-of-art MSAT [20] and VSLNet [17], achieving scores
of 52,35 and 47.82 on _R_ @1 _, IoU_ @0 _._ 7 in the NLVG and
SLVG tasks, respectively. As the query number upper limit
increases, the performance of our model significantly improves, which demonstrates the effectiveness of our model
in utilizing multimodal information.


**5.6** **Extended Evaluation on QVHighlights Dataset**


We also validate our model on the most recently publicized NLVG dataset QVHighlights [38] for multi-tasks:



|MR HD<br>Method R1 mAP >= Very Good<br>@0.5 @0.7 @0.5 @0.75 Avg. mAP HIT@1|Col2|
|---|---|
|BeautyThumb [72]<br>DVSE [73]<br>MCN [4]<br>CAL [35]<br>XML+ [36]<br>CLIP [37]<br>Moment-DETR [38]<br>UMT_∗_[39]<br>MH-DETR [74]<br>QD-DETR [75]<br>UniVTG [76]<br>**UniSDNet (Ours)**|-<br>-<br>-<br>-<br>-<br>14.36<br>20.88<br>-<br>-<br>-<br>-<br>-<br>18.75<br>21.79<br>11.41<br>2.72<br>24.94<br>8.22<br>10.67<br>-<br>-<br>25.49<br>11.54<br>23.40<br>7.65<br>9.89<br>-<br>-<br>46.69<br>33.46<br>47.89<br>34.67<br>34.90<br>35.38<br>55.06<br>16.88<br>5.19<br>18.11<br>7.00<br>7.67<br>31.30<br>61.04<br>52.89<br>33.02<br>54.82<br>29.40<br>30.73<br>35.69<br>55.60<br>56.23<br>41.18<br>53.83<br>37.01<br>36.12<br>38.18<br>59.99<br>60.05<br>42.48<br>60.75<br>38.13<br>38.38<br>38.22<br>60.51<br>62.40<br>44.98<br>62.52<br>39.88<br>39.86<br>38.94<br>62.40<br>58.86<br>40.86<br>57.60<br>35.59<br>35.47<br>38.20<br>60.96<br>**63.49**<br>**46.63**<br>**62.86**<br>**42.51**<br>**41.33**<br>**39.80**<br>**64.66**|


both moment retrieval (MR, also called temporal video
grounding) and highlight detection (HD) tasks. Following
the practice [38], [39], the commonly used metric for moment retrieval is Recall@K, IoU=[0.5, 0.7], and mean average precision (mAP). HIT@1 is also used to evaluate the
highlight detection by computing the hit ratio of the highestscored clip. The other settings such as pre-extracted Slowfast
video and CLIP text features, the number of transformer
decoder layers and loss weights are the same with MomentDETR [38]. The comparison with exiting works are listed in
Table 13. From the results, our model achieves superior performance to state-of-art models, achieving _R_ @1 _, IoU_ @0 _._ 7 of
63.03 for MR, and HIT@1 of 62.56 for HD, demonstrating its
strong universality for both tasks.


**5.7** **Qualitative Results**


We provide the qualitative results of our UniSDNet on the
ActivityNet Captions dataset with a video named “v_q81HV1_gGo” for NLVG, as shown in Fig. 12. MMN [25] exhibits significant semantic bias, making it impossible to
distinguish between _Q_ 2 and _Q_ 3 . Our **Only Static** accurately
predicts the moments, which is thanks to the effective static
learning of the semantic association between queries and
video moments. Our **Only Dynamic** performs well in the
three queries too, thanks to the fine dynamic learning of the
video sequence context. The results of the full model **Ours**
for all queries are the closest to **Groundtruth (GT)** . It shows
that the full model can integrate the advantageous aspects


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 16


**Q1:** "A man is talking outside a house."
**Q2:** " He puts a ring over his finger and pulls out
an arrow."

**Q3:** " He then demonstrates how to shoot the

arrow with the bow."





**GroundTruth of IoU Score Map**









**Q1** **Q2** **Q3**







end time















**Predicted Score Map of Full Model**



















Fig. 12. Qualitative examples of our UniSDNet. The right figures display the groundtruth IoU maps and the predicted score maps by our UniSDNet.



of static (differentiating different query semantics and supplementing video semantics) and dynamic (differentiating
and associating the related contexts in the video) modules
to achieve more accurate target moment prediction. The
quantitative results confirm the effectiveness of our unified
static and static methods in solving both NLVG and SLVG
tasks. More examples are unfolded in Appendix C.


**6** **F** **UTURE** **D** **IRECTIONS**


As a fundamental cross-modal task, TVG research remains focusing on effectively integrating multimodal data
for accurate temporal localization. Language-queried video
grounding dominates current research due to advanced
language models. In the future, several promising directions
can advance TVG: First, expanding to _more flexible query_
_modes_ – incorporating audio, images, and video clips –
can enhance the model’s ability to handle diverse inputs
and improve generalization. Second, addressing _fine-grained_
_video grounding_ is essential for real-world applications, requiring detailed temporal-spatial interactions and complex
scene dynamics capture, by developing larger fine-grained
datasets and more sophisticated models. Third, _long-form_
_video understanding_, remains challenging, as current methods
are typically designed for short videos struggle with extended duration content. Additionally, leveraging advances
in large vision-language models (VLMs) like GPT-4V can
better align visual and textual features, and explore more
complementary modality information. Finally, improving
model efficiency in computation and memory is crucial for
scaling TVG systems to larger datasets and more complex
scenarios.


**7** **C** **ONCLUSION**


In this paper, we propose a novel Unified Static and Dynamic Network (UniSDNet) for efficient video grounding.
We can adopt either single-query or multi-query mode and
achieve model performance/complexity trade-offs; it benefits from both “static” and “dynamic” associations between
queries and video semantics in a cross-modal environment.
We adopt a ResMLP architecture that comprehensively considers mutual semantic supplement through video-queries
interaction (static mode). Afterwards, we utilize a dynamic



Temporal Gaussian filter convolution to model nonlinear high-dimensional visual semantic perception (dynamic
mode). The static and dynamic manners complement each
other, ensuring effective 2D temporary proposal generation.
We also contribute two new Charades-STA Speech and
TACoS Speech datasets for SLVG task. UniSDNet is evaluated on both NLVG and SLVG. For both of them we achieve

new state-of-the-art results. We believe that our work is a
new attempt and inspire similar video tasks in the design of
neural networks guided by visual perception biology.


**R** **EFERENCES**


[1] G. Deco, D. Vidaurre, and M. L. Kringelbach, “Revisiting the
global workspace orchestrating the hierarchical organization of the
human brain,” _Nature human behaviour_, vol. 5, no. 4, pp. 497–511,
2021.

[2] J. Barbosa, H. Stein, R. L. Martinez, A. Galan-Gadea, S. Li, J. Dalmau, K. C. Adam, J. Valls-Solé, C. Constantinidis, and A. Compte,
“Interplay between persistent activity and activity-silent dynamics
in the prefrontal cortex underlies serial biases in working memory,” _Nature neuroscience_, vol. 23, no. 8, pp. 1016–1024, 2020.

[3] J. Gao, C. Sun, Z. Yang, and R. Nevatia, “TALL: temporal activity
localization via language query,” in _ICCV_, 2017, pp. 5277–5285.

[4] L. Anne Hendricks, O. Wang, E. Shechtman, J. Sivic, T. Darrell, and
B. Russell, “Localizing moments in video with natural language,”
in _ICCV_, 2017, pp. 5803–5812.

[5] Y. Xia, Z. Zhao, S. Ye, Y. Zhao, H. Li, and Y. Ren, “Videoguided curriculum learning for spoken video grounding,” in _ACM_
_Multimedia_, 2022, pp. 5191–5200.

[6] C. Rodriguez, E. Marrese-Taylor, B. Fernando, H. Takamura, and
Q. Wu, “Memory-efficient temporal moment localization in long
videos,” in _EACL_, 2023, pp. 1901–1916.

[7] K. Li, D. Guo, and M. Wang, “Proposal-free video grounding with
contextual pyramid network,” in _AAAI_, vol. 35, no. 3, 2021, pp.
1902–1910.

[8] D. Liu, X. Fang, W. Hu, and P. Zhou, “Exploring optical-flowguided motion and detection-based appearance for temporal sentence grounding,” _IEEE Trans. Multim._, vol. 25, pp. 8539–8553,
2023.

[9] D. Liu, X. Qu, X.-Y. Liu, J. Dong, P. Zhou, and Z. Xu, “Jointly crossand self-modal graph attention network for query-based moment
localization,” in _ACM Multimedia_, 2020, pp. 4070–4078.

[10] X. Sun, J. Gao, Y. Zhu, X. Wang, and X. Zhou, “Video moment
retrieval via comprehensive relation-aware network,” _IEEE Trans._
_Circuits Syst. Video Technol._, vol. 33, no. 9, pp. 5281–5295, 2023.

[11] S. Zhang, H. Peng, J. Fu, and J. Luo, “Learning 2d temporal adjacent networks for moment localization with natural language,” in
_AAAI_, 2020, pp. 12 870–12 877.

[12] S. Zhang, H. Peng, J. Fu, Y. Lu, and J. Luo, “Multi-scale 2d temporal adjacency networks for moment localization with natural
language,” _IEEE Trans. Pattern Anal. Mach. Intell._, vol. 44, no. 12,
pp. 9073–9087, 2021.


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 17




[13] J. Gao and C. Xu, “Fast video moment retrieval,” in _ICCV_, 2021,
pp. 1523–1532.

[14] M. Zheng, S. Li, Q. Chen, Y. Peng, and Y. Liu, “Phrase-level
temporal relationship mining for temporal sentence localization,”
in _AAAI_, 2023, pp. 3669–3677.

[15] H. Zhang, A. Sun, W. Jing, L. Zhen, J. T. Zhou, and R. S. M.
Goh, “Natural language video localization: A revisit in span-based
question answering framework,” _IEEE Trans. Pattern Anal. Mach._
_Intell._, vol. 44, no. 8, pp. 4252–4266, 2021.

[16] D. Liu and W. Hu, “Skimming, locating, then perusing: A humanlike framework for natural language video localization,” in _ACM_
_Multimedia_, 2022, pp. 4536–4545.

[17] H. Zhang, A. Sun, W. Jing, and J. T. Zhou, “Span-based localizing
network for natural language video localization,” in _ACL_, 2020,
pp. 6543–6554.

[18] R. Zeng, H. Xu, W. Huang, P. Chen, M. Tan, and C. Gan, “Dense
regression network for video grounding,” in _ICCV_, 2020, pp.
10 287–10 296.

[19] J. Mun, M. Cho, and B. Han, “Local-global video-text interactions
for temporal grounding,” in _ICCV_, 2020, pp. 10 810–10 819.

[20] M. Zhang, Y. Yang, X. Chen, Y. Ji, X. Xu, J. Li, and H. T. Shen,
“Multi-stage aggregated transformer network for temporal language localization in videos,” in _ICCV_, 2021, pp. 12 669–12 678.

[21] M. Zheng, Y. Huang, Q. Chen, Y. Peng, and Y. Liu, “Weakly
supervised temporal sentence grounding with gaussian-based
contrastive proposal learning,” in _ICCV_, 2022, pp. 15 555–15 564.

[22] J. Gao, X. Sun, M. Xu, X. Zhou, and B. Ghanem, “Relation-aware
video reading comprehension for temporal language grounding,”
in _EMNLP_, 2021, pp. 3978–3988.

[23] S. Xiao, L. Chen, S. Zhang, W. Ji, J. Shao, L. Ye, and J. Xiao, “Boundary proposal network for two-stage natural language video localization,” in _AAAI_, vol. 35, no. 4, 2021, pp. 2986–2994.

[24] K. Ning, L. Xie, J. Liu, F. Wu, and Q. Tian, “Interaction-integrated
network for natural language moment localization,” _IEEE Trans._
_Image Process._, vol. 30, pp. 2538–2548, 2021.

[25] Z. Wang, L. Wang, T. Wu, T. Li, and G. Wu, “Negative sample matters: A renaissance of metric learning for temporal grounding,” in
_AAAI_, 2022, pp. 2613–2623.

[26] B. Zhang, B. Jiang, C. Yang, and L. Pang, “Dual-channel localization networks for moment retrieval with natural language,” in
_ICMR_, 2022, pp. 351–359.

[27] Q. Zheng, J. Dong, X. Qu, X. Yang, Y. Wang, P. Zhou, B. Liu,
and X. Wang, “Progressive localization networks for languagebased moment localization,” _ACM Trans. Multim. Comput. Com-_
_mun. Appl._, vol. 19, no. 2, pp. 1–21, 2023.

[28] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.
Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”
in _NeurIPS_, vol. 30, 2017.

[29] Y. Yuan, L. Ma, J. Wang, W. Liu, and W. Zhu, “Semantic conditioned dynamic modulation for temporal sentence grounding in
videos,” _IEEE Trans. Pattern Anal. Mach. Intel._, vol. 44, no. 5, pp.
2725–2741, 2022.

[30] X. Sun, X. Wang, J. Gao, Q. Liu, and X. Zhou, “You need to read
again: Multi-granularity perception network for moment retrieval
in videos,” in _SIGIR_, 2022, pp. 1022–1032.

[31] Y. Wang, W. Lin, S. Zhang, T. Jin, L. Li, X. Cheng, and Z. Zhao,
“Weakly-supervised spoken video grounding via semantic interaction learning,” in _ACL_, 2023, pp. 10 914–10 932.

[32] K. Li, D. Guo, and M. Wang, “Vigt: proposal-free video grounding
with a learnable token in the transformer,” _Science China Informa-_
_tion Sciences_, vol. 66, no. 10, p. 202102, 2023.

[33] N. Liu, X. Sun, H. Yu, F. Yao, G. Xu, and K. Fu, “M [2] dcapsn: Multimodal, multichannel, and dual-step capsule network for natural
language moment localization,” _IEEE Trans. Neural Networks Learn._
_Syst._, 2023.

[34] Y. Yuan, T. Mei, and W. Zhu, “To find where you talk: Temporal
sentence localization in video with attention based location regression,” in _AAAI_, 2019, pp. 9159–9166.

[35] V. Escorcia, M. Soldan, J. Sivic, B. Ghanem, and B. C. Russell,
“Temporal localization of moments in video collections with natural language,” _CoRR_, vol. abs/1907.12763, 2019.

[36] J. Lei, L. Yu, T. L. Berg, and M. Bansal, “TVR: A large-scale dataset
for video-subtitle moment retrieval,” in _ECCV_, 2020, pp. 447–463.

[37] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark _et al._, “Learning transferable visual models from natural language supervision,” in _ICML_,
2021, pp. 8748–8763.




[38] J. Lei, T. L. Berg, and M. Bansal, “Detecting moments and highlights in videos via natural language queries,” in _NeurIPS_, vol. 34,
2021, pp. 11 846–11 858.

[39] Y. Liu, S. Li, Y. Wu, C.-W. Chen, Y. Shan, and X. Qie, “Umt: Unified
multi-modal transformers for joint video moment retrieval and
highlight detection,” in _ICCV_, 2022, pp. 3042–3051.

[40] C. Rodriguez, E. Marrese-Taylor, F. S. Saleh, H. Li, and S. Gould,
“Proposal-free temporal moment localization of a naturallanguage query in video using guided attention,” in _ICCV_, 2020,
pp. 2464–2473.

[41] M. Xu, C. Zhao, D. S. Rojas, A. Thabet, and B. Ghanem, “G-tad:
Sub-graph localization for temporal action detection,” in _ICCV_,
2020, pp. 10 156–10 165.

[42] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning
with contrastive predictive coding,” _CoRR_, vol. abs/1807.03748,
2018.

[43] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0:
A framework for self-supervised learning of speech representations,” in _NeurIPS_, vol. 33, 2020, pp. 12 449–12 460.

[44] A. Baevski, W. Hsu, Q. Xu, A. Babu, J. Gu, and M. Auli, “data2vec:
A general framework for self-supervised learning in speech, vision
and language,” in _ICML_, vol. 162, 2022, pp. 1298–1312.

[45] C. Wang, Y. Wu, Y. Qian, K. Kumatani, S. Liu, F. Wei, M. Zeng,
and X. Huang, “Unispeech: Unified speech representation learning
with labeled and unlabeled data,” in _ICML_, 2021, pp. 10 937–
10 947.

[46] M. Regneri, M. Rohrbach, D. Wetzel, S. Thater, B. Schiele, and
M. Pinkal, “Grounding action descriptions in videos,” _Trans. Assoc._
_Comput. Linguistics_, vol. 1, pp. 25–36, 2013.

[47] H. Touvron, P. Bojanowski, M. Caron, M. Cord, A. El-Nouby,
E. Grave, G. Izacard, A. Joulin, G. Synnaeve, J. Verbeek, and
H. Jégou, “Resmlp: Feedforward networks for image classification
with data-efficient training,” _IEEE Trans. Pattern Anal. Mach. Intel._,
vol. 45, no. 4, pp. 5314–5321, 2023.

[48] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, “Learning spatiotemporal features with 3d convolutional networks,” in
_ICCV_, 2015, pp. 4489–4497.

[49] J. Pennington, R. Socher, and C. D. Manning, “Glove: Global
vectors for word representation,” in _EMNLP_, 2014, pp. 1532–1543.

[50] A. Baevski, W.-N. Hsu, Q. Xu, A. Babu, J. Gu, and M. Auli,
“Data2vec: A general framework for self-supervised learning in
speech, vision and language,” in _ICML_, 2022, pp. 1298–1312.

[51] P. Wang, S. Wang, J. Lin, S. Bai, X. Zhou, J. Zhou, X. Wang, and
C. Zhou, “One-peace: Exploring one general representation model
toward unlimited modalities,” _arXiv preprint arXiv:2305.11172_,
2023.

[52] W. Kim, B. Son, and I. Kim, “Vilt: Vision-and-language transformer
without convolution or region supervision,” in _ICML_ . PMLR,
2021, pp. 5583–5594.

[53] J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and
Y. Wu, “Coca: Contrastive captioners are image-text foundation
models,” _arXiv preprint arXiv:2205.01917_, 2022.

[54] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Liò, and
Y. Bengio, “Graph attention networks,” _CoRR_, vol. abs/1710.10903,
2017.

[55] F. Long, T. Yao, Z. Qiu, X. Tian, J. Luo, and T. Mei, “Gaussian
temporal awareness networks for action localization,” in _ICCV_,
2019, pp. 344–353.

[56] K. Schütt, P. Kindermans, H. E. S. Felix, S. Chmiela, A. Tkatchenko,
and K. Müller, “Schnet: A continuous-filter convolutional neural
network for modeling quantum interactions,” in _NeurIPS_, 2017,
pp. 991–1001.

[57] R. Krishna, K. Hata, F. Ren, L. Fei-Fei, and J. Carlos Niebles,
“Dense-captioning events in videos,” in _ICCV_, 2017, pp. 706–715.

[58] J. Ao, R. Wang, L. Zhou, C. Wang, S. Ren, Y. Wu, S. Liu, T. Ko,
Q. Li, Y. Zhang _et al._, “Speecht5: Unified-modal encoder-decoder
pre-training for spoken language processing,” in _ACL_, 2022, pp.
5723–5738.

[59] G. A. Sigurdsson, G. Varol, X. Wang, A. Farhadi, I. Laptev, and
A. Gupta, “Hollywood in homes: Crowdsourcing data collection
for activity understanding,” in _ECCV_, 2016, pp. 510–526.

[60] M. Rohrbach, M. Regneri, M. Andriluka, S. Amin, M. Pinkal,
and B. Schiele, “Script data for attribute-based recognition of
composite activities,” in _ECCV_, 2012, pp. 144–157.

[61] V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu,
A. Elkahky, Z. Ni, A. Vyas, M. Fazel-Zarandi, A. Baevski, Y. Adi,


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 18



X. Zhang, W. Hsu, A. Conneau, and M. Auli, “Scaling speech
technology to 1, 000+ languages,” _CoRR_, vol. abs/2305.13516, 2023.

[62] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert, a distilled
version of BERT: smaller, faster, cheaper and lighter,” _CoRR_, vol.
abs/1910.01108, 2019.

[63] X. Wang, Z. Wu, H. Chen, X. Lan, and W. Zhu, “Mixup-augmented
temporally debiased video grounding with content-location disentanglement,” in _ACM Multimedia_, 2023, pp. 4450–4459.

[64] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” vol. abs/1409.1556, 2014.

[65] J. Carreira and A. Zisserman, “Quo vadis, action recognition? a
new model and the kinetics dataset,” in _ICCV_, 2017, pp. 6299–
6308.

[66] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi,
P. Cistac, T. Rault, R. Louf, M. Funtowicz, and J. Brew, “Huggingface’s transformers: State-of-the-art natural language processing,”
_CoRR_, vol. abs/1910.03771, 2019.

[67] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” _CoRR_, vol. abs/1711.05101, 2017.

[68] W. Jing, A. Sun, H. Zhang, and X. Li, “MS-DETR: natural language
video localization with sampling moment-moment interaction,” in
_ACL_, 2023, pp. 1387–1400.

[69] A. Gu, K. Goel, and C. Ré, “Efficiently modeling long sequences
with structured state spaces,” _CoRR_, vol. abs/2111.00396, 2021.

[70] T. N. Kipf and M. Welling, “Semi-supervised classification with
graph convolutional networks,” vol. abs/1609.02907, 2016.

[71] Q. Li, Z. Han, and X.-M. Wu, “Deeper insights into graph convolutional networks for semi-supervised learning,” in _AAAI_, 2018,
pp. 3538–3545.

[72] Y. Song, M. Redi, J. Vallmitjana, and A. Jaimes, “To click or not to
click: Automatic selection of beautiful thumbnails from videos,”
in _CIKM_, 2016, pp. 659–668.

[73] W. Liu, T. Mei, Y. Zhang, C. Che, and J. Luo, “Multi-task deep
visual-semantic embedding for video thumbnail selection,” in
_ICCV_, 2015, pp. 3707–3715.

[74] Y. Xu, Y. Sun, Y. Li, Y. Shi, X. Zhu, and S. Du, “MH-DETR: video
moment and highlight detection with cross-modal transformer,”
_CoRR_, vol. abs/2305.00355, 2023.

[75] W. Moon, S. Hyun, S. Park, D. Park, and J.-P. Heo, “Querydependent video representation for moment retrieval and highlight detection,” in _ICCV_, 2023, pp. 23 023–23 033.

[76] K. Q. Lin, P. Zhang, J. Chen, S. Pramanick, D. Gao, A. J. Wang,
R. Yan, and M. Z. Shou, “Univtg: Towards unified video-language
temporal grounding,” in _ICCV_, 2023, pp. 2794–2804.


**Jingjing Hu** received the B.E. degree in the
Internet of Things from Northeastern University,
China, in 2022. She is currently pursuing the
master’s degree in the School of Computer Science and Information Engineering, Hefei University of Technology, China. Her research interests
include multimedia content analysis, computer
vision. She currently serves as a reviewer of
ACM Multimedia conference.


**Dan Guo** (Senior Member, IEEE) is currently a
Professor with the School of Computer Science
and Information Engineering, Hefei University of
Technology, China. Her research interests include computer vision, machine learning, and intelligent multimedia content analysis. She serves
as a PC Member and for top-tier conferences
and prestigious journals in multimedia and artificial intelligence, like ACM Multimedia, IJCAI,
AAAI, CVPR and ECCV. She also serves as a
SPC Member for IJCAI 2021.



**Kun Li** is currently pursuing the Ph.D. degree in
the School of Computer Science and Information Engineering, Hefei University of Technology,
China. His research interests include multimedia content analysis, computer vision, and video
understanding. He regularly serves as a PC
Member for top-tier conferences in multimedia
and artificial intelligence, like ACM Multimedia,
IJCAI, AAAI, CVPR, ICCV, and ECCV.


**Zhan** **Si** received the B.E. degree from
Shenyang University of Chemical Technology,
China, in 2022. He is currently pursuing the
master’s degree in the School of Chemistry and
Chemical Engineering, Anhui University, China.
His research interests include AI for science and
computational chemistry.


**Xun Yang** is currently a Professor with the Department of Electronic Engineering and Information Science, University of Science and Technology of China (USTC). His research interests include information retrieval, cross-media analysis
and reasoning, and computer vision. He served
as the Area Chair for the ACM Multimedia 2022.
He also serves as the Associate Editor for the
IEEE TRANSACTIONS ON BIG DATA journal.


**Xiaojun Chang** (Senior Member, IEEE) is currently a Professor at the School of Information Science and Technology (USTC). He has
spent most of his time working on exploring
multiple signals (visual, acoustic, and textual) for
automatic content analysis in unconstrained or
surveillance videos. He has achieved top performances in various international competitions,
such as TRECVID MED, TRECVID SIN, and
TRECVID AVS.


**Meng Wang** (Fellow, IEEE) is currently a Professor with the Hefei University of Technology,
China. His current research interests include
multimedia content analysis, computer vision,
and pattern recognition. He was a recipient of
the ACM SIGMM Rising Star Award 2014. He
is an Associate Editor of the IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, the IEEE TRANSACTIONS ON
CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, and the IEEE TRANSACTIONS ON
NEURAL NETWORKS AND LEARNING SYSTEMS.


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 19


**A** **PPENDIX** **A**

**O** **VERALL** **P** **REDICTION** **A** **NALYSIS FOR BOTH** **NLVG** **AND** **SLVG T** **ASKS**


Fig. 13 shows the temporal distribution of target moments on the ActivtyNet Captions, Charades-STA, and TACoS datasets
for NLVG task, the distribution of target moments varies among the these datasets, and our method has good predictive
performance than MMN [25] on all these datasets, indicating that the model has good robustness. Fig. 14 shows the
temporal distribution of target moments on the ActivtyNet Speech, Charades-STA Speech, and TACoS Speech datasets for
SLVG task, it is clear to see that when audio is used as the query, the MMN approach is clearly missing some important
moment regions in the predicted response on the ActivityNet Speech and TACoS Speech datasets, whereas our approach
responds more comprehensively to the moment peak regions shown by groundtruth. It is worth noting that the distribution
of video grounding results using text and audio as queries differ for both MMN and our UniSDNet-M. It is reasonable to
assume that the predictions using text are closest to groundtruth, text-based TVG outperforms audio-based TVG due to its
superior representation capability from pre-training technique.





































(a) Groundtruth of the target ~~m~~ oment distribution for NLVG.



























(b) Predicted moment distri ~~b~~ ution by MMN [25] for NLVG.





















(c) The distribution of predicted moment by our UniSDNet-M for NLVG.


Fig. 13. The distribution of predicted moments by our UniSDNet-M and MMN [25] on ActivtyNet Captions, Charades-STA, and TACoS datasets
for NLVG task. While MMN’s predictions are more centrally biased towards regions of high density, our model fits the true distribution of the target
moments to a greater extent.


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 20





































(a) Groundtruth of the target ~~m~~ oment distribution for SLVG.



































(b) Predicted moment distribution by MMN [25] for SLVG.























(c) The distribution of predicted moment by our UniSDNet-M for SLVG.


Fig. 14. The distribution of predicted moments by our UniSDNet-M and MMN [25] on ActivtyNet Speech, Charades-STA Speech, and TACoS
Speech datasets for SLVG task. While the prediction of MMN clearly ignores some important regions, _e.g_ ., on the ActivtyNet Speech dataset, very
few results correspond to the left centre and right centre of the distributions for [start index = 0, end index = 15] and [start index = 40, end index =
60], our model also fits the true distribution of the target moments to a greater extent when using the spoken language as the query.


**A** **PPENDIX** **B**

**A** **DDITIONAL** **E** **XPERIMENTAL** **R** **ESULTS**


In this section, we conduct a series of ablation studies to evaluate the hyperparameter _k_ in graph construction, as well as
various model settings including the method of adding positional encodings in the feature encoding stage and the semantic
matching function in the model’s decoding stage.


**B.1** **Ablation Study on Hyperparameters** _k_ **in Graph Construction**


The hyperparameter _k_ in Eq. 2 of the main paper, the dividing value between short and long distances in the video graph,
is a empirical parameter, which is tuned on validation set with the final model are tested on test set. Table 14 shows the
ablation experiments on hyperparameter _k_, and Fig. 15 visualizes the video graph connectivity matrix with different _k_
value. From the experimental results, the optimal value of _k_ on all three datasets is 16. Either too small or too large a
_k_ value can impair performance, a small _k_ value overly focuses on short-distance information, neglecting long-distance


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 21


dependencies in videos, while a large _k_ value adds more redundant edges, increasing the difficulty for the model to
recognize video events.



















Fig. 15. The adjacency matrices for different _k_ values (4, 8, 16, and 32), with the number of video clips _T_ fixed at 64 (row _i_ corresponds to node
_v_ _i_, column _i_ corresponds to node _v_ _j_ ). Each sub-figure is a binary value _{_ 0 _,_ 1 _}_ that shows valid connections between start and end indexes in the
video graph.


TABLE 14
Ablation study on hyperparameter _k_ for NLVG task. _L_ _V_ denotes the average duration of videos in a dataset. _T_ is the number of sampled clips,
which is consistent with the settings in [9], [11], [19] for experiment fairness. Here, we use currently popular video features (ActivityNet Captions,
C3D) [11], (Charades-STA, I3D) [19], (TACoS, C3D) [9] respectively.












|Dataset LV (s) T k|R@1<br>IoU@0.1 IoU@0.3 IoU@0.5 IoU@0.7|R@5<br>IoU@0.1 IoU@0.3 IoU@0.5 IoU@0.7|mIoU|
|---|---|---|---|
|117.60<br>64<br>4<br>ActivityNet<br>8<br>Captions<br>16<br>32<br>64|89.79<br>73.59<br>57.92<br>35.50<br>90.12<br>74.97<br>60.03<br>37.20<br>**90.28**<br>**75.85**<br>**60.75**<br>**38.88**<br>90.04<br>74.87<br>60.05<br>36.92<br>89.99<br>74.79<br>59.67<br>37.17|96.26<br>90.59<br>84.53<br>73.20<br>96.26<br>90.75<br>84.66<br>73.21<br>**96.32**<br>**91.17**<br>**85.34**<br>**74.01**<br>96.09<br>90.72<br>84.45<br>72.77<br>96.08<br>90.42<br>84.48<br>72.17|53.45<br>54.53<br>**55.47**<br>54.48<br>54.44|
|30.60<br>64<br>4<br>Charades-<br>8<br>STA<br>16<br>32<br>64|78.04<br>70.94<br>58.15<br>37.42<br>78.44<br>71.29<br>58.44<br>38.98<br>**79.44**<br>**72.18**<br>**61.02**<br>**39.70**<br>78.17<br>70.83<br>58.23<br>39.33<br>78.68<br>71.53<br>58.76<br>38.33|98.06<br>95.91<br>89.60<br>70.99<br>97.82<br>95.81<br>89.68<br>72.72<br>**97.55**<br>**95.35**<br>**89.97**<br>**73.20**<br>97.72<br>95.97<br>90.30<br>73.23<br>98.06<br>96.42<br>89.73<br>71.96|50.99<br>51.60<br>**52.69**<br>51.67<br>51.66|
|TACoS<br>286.59<br>128<br>4<br>8<br>16<br>32<br>64|68.03<br>53.34<br>37.69<br>21.94<br>69.78<br>53.19<br>38.64<br>22.14<br>**70.78**<br>**55.56**<br>**40.26**<br>**24.12**<br>66.73<br>53.06<br>37.72<br>21.87<br>68.93<br>51.59<br>34.82<br>18.22|88.63<br>76.63<br>63.03<br>35.67<br>87.83<br>75.43<br>63.16<br>35.14<br>**89.85**<br>**77.08**<br>**64.01**<br>**37.02**<br>88.75<br>75.88<br>63.31<br>35.34<br>88.58<br>76.66<br>63.01<br>32.94|37.17<br>37.84<br>**38.88**<br>36.90<br>35.50|



**B.2** **Ablation Study on Adding Position Embeddings for Video and Query**


The function of position embedding (PE) is to help the model understand the relative position and order of different
elements in the sequence, and thus better capture the semantic information in the sequence. We add sine position
embedding [28] to the input video clip and query sequence, in order to enhance the temporal relationships between video
sequences and the logical relationships between queries. Considering that most existing multimodal Transformers add
independent PEs to different modalities, in order to distinguish modality-specific information, and arcitecturally, the static
module with ResMLP structure of our model is similar to Transformer in processing multi-modal sequences in parallel [47].
We simply follow existing work, adding independent PEs for video and queries, as shown in Fig. 16. Specifically, we denote
the PE for each video clip _v_ _i_ or query _q_ _i_ as:



_PE_ ( _o_ _i_ ) =



_sin_ ( _i/_ 10000 _[j/d]_ ) _,_ if _j_ is even

_,_ (11)

� _cos_ ( _i/_ 10000 _[j/d]_ ) _,_ if _j_ is odd



where _PE_ ( _o_ _i_ ) _∈_ R [1] _[×][d]_, _o_ _i_ denotes _v_ _i_ or _q_ _i_, and _j_ varies from 1 to _d_ dimension. We set up two different ways to add PE:
adding Independent PE and adding Joint PE. These two ways of adding PE correspond to “w/. Independent PE” and “w/.
Joint PE” in Table 15, respectively. The results demonstrate that the inclusion of PEs significantly improves the model’s
performance. On the ActivityNet Captions dataset, the _R_ @1 _, IoU_ @0 _._ 7 score improved from 30.16% to 36.96%. Similarly, on
the TACoS dataset, the _R_ @1 _, IoU_ @0 _._ 5 score increased from 30.94% to 36.84%. The setting of “w/. Independent PE” gives
better results than that of “w/. Joint PE”, which demonstrates the superiority of adding independent PE.


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 22



_**Query**_



_**Video Clip**_



_**Query**_



_**Video Clip**_



**2** _**v**_ **3** _**v**_ **4** _**v**_ **5** **6** **7**



**1**



_**v**_ _**1**_ **2** _**v**_ _**2**_ **3** _**v**_ _**3**_ **4** _**v**_ _**4**_ **1** _**q**_ _**1**_ **2** _**q**_ _**2**_ **3** _**q**_ _**3**_ **1**



**Position**
**2** _**v**_ _**2**_ **3** _**v**_ _**3**_ **4** _**v**_ _**4**_ **1** _**q**_ _**1**_ **2** _**q**_ _**2**_ **3** _**q**_ _**3**_ **1** _**v**_ _**1**_ **2** _**v**_ _**2**_ **3** _**v**_ _**3**_ **4** _**v**_ _**4**_ **5** _**q**_ _**1**_ **6** _**q**_ _**2**_ **7** _**q**_ _**3**_ **Embedding**



_**v**_ _**1**_ **2** _**v**_ _**2**_ **3** _**v**_ _**3**_ **4** _**v**_ _**4**_ **5** _**q**_ _**1**_ **6** _**q**_ _**2**_ **7** _**q**_ _**3**_



**(a) Adding independent PEs for different features.** **(b) Adding joint PEs for different features.**


Fig. 16. Illustration of two different ways to add position embedding (PE) for different modality features (query and video).


TABLE 15
**Ablation Study on adding position embedding for different modality features.** The setting of “w/o. PE” refers to the model without any position
embedding; the setting of “w/. Joint PE” refers to the model with the joint position encoding added to all modalities; the setting of “w/. Independent
PE” refers to the model with independent positional embedding for different modalities (queries and video clips).

|Dataset Model Setting|R@1<br>IoU@0.1 IoU@0.3 IoU@0.5 IoU@0.7|R@5<br>IoU@0.1 IoU@0.3 IoU@0.5 IoU@0.7|mIoU|
|---|---|---|---|
|ActivityNet<br>w/o. PE<br>Captions<br>w/. Joint PE<br>w/. Independent PE|88.42<br>70.11<br>54.12<br>30.16<br>89.88<br>74.18<br>58.29<br>36.96<br>**90.28**<br>**75.85**<br>**60.75**<br>**38.88**|95.93<br>89.89<br>81.43<br>67.82<br>96.14<br>90.12<br>83.67<br>72.08<br>**96.32**<br>**91.17**<br>**85.34**<br>**74.01**|49.91<br>53.96<br>**55.47**|
|Charades-<br>w/o. PE<br>STA<br>w/. Joint PE<br>w/. Independent PE|71.99<br>63.60<br>50.56<br>31.18<br>77.96<br>70.81<br>57.80<br>36.53<br>**79.44**<br>**72.18**<br>**61.02**<br>**39.70**|97.61<br>94.68<br>86.53<br>64.78<br>98.09<br>96.08<br>90.13<br>71.18<br>**97.55**<br>**95.35**<br>**89.97**<br>**73.20**|45.09<br>50.81<br>**52.69**|
|TACoS<br>w/o. PE<br>w/. Joint PE<br>w/. Independent PE|62.48<br>45.21<br>30.94<br>16.97<br>65.96<br>50.81<br>36.84<br>19.70<br>**70.78**<br>**55.56**<br>**40.26**<br>**24.12**|88.30<br>75.01<br>58.69<br>31.22<br>90.85<br>79.18<br>65.28<br>35.02<br>**89.85**<br>**77.08**<br>**64.01**<br>**37.02**|31.99<br>35.59<br>**38.88**|



**B.3** **Ablation Study on Modality Alignment Measurement Method**


In this part, we investigate different cross-modal semantic similarity matching methods. The ablation study in Table 16
compares cosine similarity used in “Section 3.4 2D Proposal Generation”, Eq. 6 of the main paper, with other similarity
measures, including **(1) Mean Hadamard product** : Hada Mean ( _S_ _[M]_ _, S_ _[Q]_ ) = _d_ 1 � _di_ =1 [(] _[S]_ _i_ _[M]_ _⊙_ _S_ _i_ _[Q]_ [)] [.] **[ (2) Euclidean distance]**

measures the straight-line distance between two vectors and is defined as: E-Dis ( _S_ _[M]_ _, S_ _[Q]_ ) = � ~~�~~ _di_ =1 [(] _[S]_ _i_ _[M]_ _−_ _S_ _i_ _[Q]_ [)] [2] [.] **[ (3)]**

**Manhattan distance**, also known as L1 distance, is calculated as: M-Dis ( _S_ _[M]_ _, S_ _[Q]_ ) = [�] _[d]_ _i_ =1 _[|][S]_ _i_ _[M]_ _−_ _S_ _i_ _[Q]_ _[|]_ [. From Table][ 16][, it]
is evident that cosine similarity performs best, and the Hadamard product provides competitive results. Based on these
findings, we confirm that cosine similarity is an effective measure for our semantic matching module. Nevertheless, the
alternative similarity measures provide valuable insights and potential areas for further exploration.


TABLE 16
**Ablation Study on different similarity measure functions.** We report the experimental results of similarity measures: “cosine" CoSine ( _·, ·_ ),
“Mean Hadamard product” Hada Mean ( _·, ·_ ), “Euclidean distance” E-Dis ( _·, ·_ ), and “Manhattan distance” M-Dis ( _·, ·_ ) .






|Semantic Matching<br>Dataset<br>Measure Method|R@1<br>IoU@0.1 IoU@0.3 IoU@0.5 IoU@0.7|R@5<br>IoU@0.1 IoU@0.3 IoU@0.5 IoU@0.7|mIoU|
|---|---|---|---|
|ActivityNet<br>CoSine(_·, ·_)<br>Captions<br>HadaMean(_·, ·_)<br>E-Dis(_·, ·_)<br>M-Dis(_·, ·_)|**90.28**<br>**75.85**<br>**60.75**<br>**38.88**<br>89.08<br>74.23<br>58.89<br>36.85<br>89.37<br>74.68<br>58.40<br>35.83<br>89.06<br>73.21<br>56.68<br>34.01|**96.32**<br>**91.17**<br>**85.34**<br>**74.01**<br>95.94<br>90.59<br>84.73<br>73.16<br>95.96<br>90.32<br>84.16<br>71.26<br>96.29<br>90.66<br>84.45<br>72.93|**55.47**<br>54.04<br>53.57<br>52.69|
|Charades-<br>CoSine(_·, ·_)<br>STA<br>HadaMean(_·, ·_)<br>E-Dis(_·, ·_)<br>M-Dis(_·, ·_)|**79.44**<br>**72.18**<br>**61.02**<br>**39.70**<br>78.68<br>71.53<br>58.76<br>38.33<br>78.01<br>70.59<br>57.28<br>37.37<br>77.72<br>69.95<br>57.47<br>37.26|**97.55**<br>**95.35**<br>**89.97**<br>**73.20**<br>98.06<br>96.42<br>89.73<br>71.96<br>98.31<br>96.29<br>90.27<br>71.64<br>97.77<br>95.43<br>89.25<br>70.97|**52.69**<br>51.66<br>50.82<br>50.48|
|TACoS<br>CoSine(_·, ·_)<br>HadaMean(_·, ·_)<br>E-Dis(_·, ·_)<br>M-Dis(_·, ·_)|**70.78**<br>**55.56**<br>**40.26**<br>**24.12**<br>69.51<br>54.19<br>38.59<br>23.03<br>68.03<br>53.34<br>37.69<br>22.94<br>66.58<br>53.11<br>37.44<br>22.87|**89.85**<br>**77.08**<br>**64.01**<br>**37.02**<br>89.65<br>78.78<br>64.48<br>35.87<br>89.63<br>77.63<br>64.03<br>35.67<br>88.78<br>75.43<br>62.76<br>35.09|**38.88**<br>37.60<br>37.17<br>36.90|



**A** **PPENDIX** **C**

**M** **ORE** **V** **ISUALIZATION OF** **P** **REDICTION** **R** **ESULTS**


In order to clearly demonstrate the specific role of our proposed unified static and dynamic networks in cross-modal video
grounding, we provide more challenging visualization cases in this section as a supplement to Sec. 5.7.


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 23


**C.1** **Visualization on ActivityNet Captions for NLVG**


**Video Sample with Complex Scene Transitions.** The ActivityNet Captions dataset contains a large amount of open-world
videos with more shot transitions. We choose typical samples of this type for visualisation and analysis. As shown in Fig 17
(a), there are multiple scene transitions in video sample “ID: v_rKtktLDSOpA” from the ActivityNet Captions dataset and
different events have serious intersection in the temporal sequence of video. For example, there is an intersection between
the end of the moment corresponding to _Q_ 1 and the beginning of the moment corresponding to _Q_ 2 and another big
intersection exists between the moments corresponding to _Q_ 2 and _Q_ 3 . From Fig. 17, **MMN** [25] makes a serious prediction
for _Q_ 1, locating the moment corresponding to _Q_ 2 . Meanwhile, when predicting _Q_ 3, **MMN** omits the temporal region
intersected with _Q_ 2 but correct temporal region also belonged to the moment of _Q_ 3 for the final prediction. Compared
to **MMN**, our **Only Static** and **Only Dynamic** predict more accurate moments for each query, and they can accurately
comprehend the intersection of _Q_ 2 and _Q_ 3 . **Only Static** performs better at identifying transitions, while **Only Dynamic**
performs better at recognizing overlapping events. Our **Full Model** performs best in these challenging scenarios because
it combines the advantages of **Only Static** and **Only Dynamic** .
**Video Sample with Similar Scenes.** For the NLVG task that employs textual queries, it is also challenging to use
the semantic guidance of the text to distinguish some video clips that are similar in the front and back frames (without
transitions). As shown in Fig. 17 (b), the frames in video sample “ID: v_UajYunTsr70” from the ActivityNet Captions dataset
also have high similarity, you can find it to locate the corresponding moment corresponding to _Q_ 1 : “ _A cat is sitting on top_
_of a white sheet._ ” **MMN** is basically unable to distinguish the video content for the three different queries. It almost predicts
the entire video for each query. Even through our **Only Static** performs poorly in this situation too, our **Only Dynamic**
performs much better than MMN. Finally, our **Full model** locates the most accurate target moment. This is thanks to our
model that combines the advantages of static and dynamic modules, especially for that the latter learns a tighter contextual
correlation of video in this case.


**C.2** **Visualization on ActivityNet Speech for SLVG.**


We also provide quantitative results of our UniSDNet on SLVG to demonstrate the effectiveness of our model in the video
grounding task based on spoken language.
**Video Sample with Noisy Background.** When using audio as a query, we prefer to analyze how well the model
understands the interaction between audio and video by performing visualizations of video cases that contain more
background noise. We instantiate the video sample “ID: v_FsS8cQbfKTQ” from the ActivityNet Speech dataset in Fig. 18
(a) using audio queries under noisy background interference. We can see that **MMN** predicts the video clips corresponding
to _Q_ 2 and _Q_ 3 with significant deviations, and the predicted moments totally do not intersect with **GT** at all. This video
is a challenging case. Compared to **MMN**, **Only Static** and **Only Dynamic** coverage the queried moment but have
somewhat boundary shifts, exhibits a strong advantage, as it correctly predicts the relative positions of all events has a
large intersection ratio with **GT** video clips. Compared to **MMN**, our **Full model** exhibits the best prediction results for all
queries, as it correctly predicts the queried moment and has a large intersection ratio with **GT** video clips. From the 2D
map in the figure, it can be seen that our model still performs well in video grounding task based on audio queries, fully
demonstrating the generalization of our model.
**The Videos with Continuous and Varied Actions.** Similarly to the NLVG task, we analyse the video case without
transitions but with continuous action changes for the SLVG task to quantify the model’s ability of identifying event
boundaries. Taking video sample “ID: v_UJwWjTvDEpQ” from the ActivityNet Speech dataset in Fig. 18 (b) as an example,
the video shows a scene with a clean background, but in which a boy’s actions are continuously changing. In this case,
for different event divisions, it is necessary to finely distinguish the contentual semantics of the boy’s actions and the
differences between them. **MMN** fails to recognize such densly varied actions and incorrectly assigns the entire video as
the answer ( _e.g_ ., _Q_ 1 and _Q_ 2 ). **Our Static** predicts the approximate location of each event. **Our Dynamic** exhibits excellent
performance in distinguishing the semantics of continuous actions, it not only correctly distinguishes the semantic centers
of three events, but also more accurately predicts the boundaries of each event, compared to **MMN** and **Our Static** .
Inspiring, **Full Model** achieves the most accurate prediction of the location and semantic boundaries of events, this is
thanks to the combination of static and dynamic modes, which deepens the understanding of video context and enables
the model to distinguish different action semantics.


**C.3** **More Visualization of Plethoric Multi-query Cases**


**Visualization Examples on the TACoS Dataset.** Taking the video sample “ID: s27-d50” in Fig. 19 (a) as an example,
we provide the grounding results of our model and MMN. Note that the total duration of the video is 82.11 s, which
includes 119 query descriptions. Limited by page size and layout, we select and show 6 very challenging queries here. The
video depicts a person cooking in a kitchen. **MMN** experiences a significant prediction error in the moment corresponding
to the query _Q_ 88 . On the contrary, our **Full model** accurately determines the relative positions of the video segments
corresponding to all queries. The qualitative results highlight the effectiveness of learning semantic associations between
multi-queries ( _i.e_ ., multi-queries contextualization) for cross-modal video grounding.
**Visualization Examples on the Charades-STA Dataset.** The video sample “ID: U5T4M” in Fig. 19 (b) has a duration
of 19.58 s, which describes the indoor activities of a person, and contains 7 queries. Our **Full model** infers the localization


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 24



**v_rKtktLDSOpA.mp4**



(a)



**Q1:** “The camera is blurry then focuses on people playing polo.”
**Q2:** “Two teams play against each other.”,
**Q3:** “A woman video records the proceedings.”,













**GT** **Full Model** **GT** **Full Model** **GT** **Full Model**





end time end time end time end time







**Q1** **Q2** **Q3**



(b)



**Q1:** “A cat is sitting on top of a white sheet.”
**Q2:** “It is licking its paws over and over again.”



**v_UajYunTsr70.mp4**









**GT** **Full Model** **GT** **Full Model** **GT** **Full Model**











**Q1** **Q2** **Q3**


Fig. 17. Qualitative examples on ActivityNet Captions for NLVG. **(a)** The video contains complex scene transitions and overlap. **(b)** The video scenes
that are difficult to distinguish. **MMN** makes significant errors in predicting the location range of the queried events, _i.e_ ., _Q_ 1 and _Q_ 3 in cases (a)
and (b), respectively. Our **Only Static** has an advantage in predicting transitions ( _Q_ 1 in case (a)), our **Only Dynamic** performs better in predicting
overlapping. It is difficult to distinguish scenarios ( _Q_ 2 and _Q_ 3 in both cases (a) and (b)). Our **Full Model** performs best in both challenging scenarios,
as it combines the advantages of static (query semantic differentiation) and dynamic (video sequence context association) modules.


results of all queries corresponding to the video at once. In all queries, _Q_ 1 and _Q_ 2 are similar descriptions of an event,
respectively. The same situation also includes queries of _Q_ 4 and _Q_ 5, _Q_ 6 and _Q_ 7 . Our **Full model** accurately predicts the
boundaries of each query, and effectively distinguishing the semantics among similar but with slightly different events.


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 25



(a)



**Q1:** FsS8cQbfKTQ_val_2_1.wav                   ( A man is running while holding a pole on a track. )
**Q2:** FsS8cQbfKTQ_val_2_2.wav                   ( He uses the pole to vault through the air. )



















**GT** **Full Model** **GT** **Full Model** **GT** **Full Model**









end time end time end time end time end time end time

**Q1** **Q2** **Q3**



(b)



**Q1:** UJwWjTvDEpQ_val_2_1.wav                  ( A young child is seen speaking to the camera while holding a guitar. )
**Q2:** UJwWjTvDEpQ_val_2_2.wav                  ( He moves the guitar around while showing it off to the camera. )
**Q3:** UJwWjTvDEpQ_val_2_3.wav ( He plays a bit and continues to speak. ) **v_UJwWjTvDEpQ.mp4**











**GT** **Full Model** **GT** **Full Model** **GT** **Full Model**











**Q1** **Q2** **Q3**


Fig. 18. Qualitative examples on ActivityNet Speech for SLVG. **(a)** The scenes that contains a noisy background. **(b)** The Videos with Continuous
and Varied Actions. **MMN** makes significant errors in predicting the location ( _Q_ 2 and _Q_ 3 in case (a)) and location coverage areas of events ( _Q_ 1 and
_Q_ 2 in case (b)). These two cases are challenging. Encouragingly, our **Full Model** achieves the best performance in these video grounding cases
based on audio queries, which confirms the effectiveness and generalization of our unified static and dynamic methods in this task.


TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 26



**Q64** "The person cracks the egg carefully over the larger glass."
**Q88** "The person pours the yolk into the small glass."



(a)


(b)



**Q7:** "The person gets out two glasses."
**Q32:** "The person procures a small glass from the cupboard."



**Q1:** "person opens refrigerator grabs milk.",
**Q2:** "a person opened a refrigerator.",
**Q3:** "person closed the refrigerator door.",
**Q4:** "person starts eating it.",



**Q5:** "person eats it.",
**Q6:** "person walked over to a chair to sit down.",
**Q7:** "person sits in a chair."





















**U5T4M.mp4**

























Fig. 19. Quantitative examples of plethoric multi-query cases. **(a)** Examples on the TACoS dataset for NLVG. **(b)** Examples on the Charades-STA
dataste for NLVG. **MMN** has a significant semantic bias when predicting _Q_ 7 in case (a), and _Q_ 4 _, Q_ 5 _, Q_ 7 in case (b), there is also a large positional
deviation in predicting _Q_ 88 in case (a), and _Q_ 1 _, Q_ 3 in case (b). Our **Full Model** correctly predicts the location of all the queried events, and the
predicted moment interval is closest to that of **GT**, this is thanks to model capacity of mutual learning of video and multiple queries and effectively
capturing the video context associated with multiple queries.


