_LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_ **1**

## **Use What You Have: Video Retrieval Using** **Representations From Collaborative Experts**



Yang Liu*
yangl@robots.ox.ac.uk


Samuel Albanie*

albanie@robots.ox.ac.uk


Arsha Nagrani*
arsha@robots.ox.ac.uk


Andrew Zisserman

az@robots.ox.ac.uk



Visual Geometry Group
University of Oxford
UK



**Abstract**


The rapid growth of video on the internet has made searching for video content using
natural language queries a significant challenge. Human-generated queries for video datasets
‘in the wild’ vary a lot in terms of degree of specificity, with some queries describing ‘specific
details’ such as the names of famous identities, content from speech, or text available on the
screen. Our goal is to condense the multi-modal, extremely high dimensional information from
videos into a single, compact video representation for the task of video retrieval using free-form
text queries, where the degree of specificity is open-ended.
For this we exploit existing knowledge in the form of pre-trained semantic embeddings
which include ‘general’ features such as motion, appearance, and scene features from visual
content. We also explore the use of more ‘specific‘ cues from ASR and OCR which are intermittently available for videos and find that these signals remain challenging to use effectively
for retrieval. We propose a _collaborative experts_ model to aggregate information from these
different pre-trained experts and assess our approach empirically on five retrieval benchmarks:
MSR-VTT, LSMDC, MSVD, DiDeMo, and ActivityNet. Code and data can be found at www.
robots.ox.ac.uk/~vgg/research/collaborative-experts/ . This paper
contains a correction to results reported in the previous version.

### **1 Introduction**


Videos capture the world in two important ways beyond a simple image: first, video contains
temporal information – semantic concepts, actions and interactions evolve over time; Second, video
may also contain information from multiple modalities, such as an accompanying audio track.
This makes videos both richer and more informative, but also more challenging to represent. Our
goal in this paper is to embed the information from multiple modalities and multiple time steps
of a video segment into a compact fixed-length representation. Such a compact representation can
then be used for a number of video understanding tasks, such as video retrieval, clustering and


_⃝_ c 2019. The copyright of this document resides with its authors.
It may be distributed unchanged freely in print or electronic forms.
Details of the results correction can found in the appendix.
*Equal contribution.


**2** _LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_







Figure 1: (Left): Unconstrained videos ‘in the wild’ convey information in various different ways,
including (clockwise from upper-left), clues from distinctive speech, names of individuals on
screen, other text clues embedded in the video and audio. (Right): For the five video datasets
considered in this work, the chart portrays the video-level availability of “expert” embeddings
from different domains (with potentially multiple experts per domain): certain generic embeddings
can almost always be extracted via pretrained object/action/scene classification networks. Other
features such as sounds, faces, speech and OCR are less consistently available and are more
challenging to exploit (Sec. 4.3).


summarisation. In particular, we focus on retrieval; our objective is to be able to retrieve video
clips using a free form text query that may contain both general and specific information.
Learning a robust and compact representation _tabula rasa_ for this task is made extremely
challenging by the high dimensionality of the sensory data contained in videos—to do so with
discriminative training would require prohibitively expensive textual annotation of a vast number
of videos. The primary hypothesis underpinning our approach is the following: _the discriminative_
_content of the multi-modal video embedding can be well approximated by the set of semantic repre-_
_sentations of the video data learnt by individual experts (in audio, scenes, actions, etc)_ . In essence,
this approximation enables us to exploit knowledge from existing individual sources where the cost
of annotation is significantly reduced (e.g. classification labels for objects and scenes in images,
labels for actions in videos etc.) and where consequently, there exist very large-scale labelled
datasets. These large-scale datasets can then be used to train independent “experts” for different
perception tasks, which in turn provide a robust, low-dimensional basis for the discriminative
query-content approximation described above.
The two key aspects of this idea that we explore in this paper are: (i) _General and specific_
_features:_ in addition to using generic video descriptors (e.g. objects and actions) we investigate
encodings of quite specific information from the clip, for example, text from overlaid captions
and text from speech to provide effective coverage of the “queryable content” of the video (Fig. 1,
left). While such features may be highly discriminative for humans, they may not always be
available (Fig. 1, right) and as we show through experiments (Sec. 4.3), making good use of these
cues is challenging. We therefore also propose (ii) _Collaborative experts:_ a framework that seeks
to make effective use of embeddings from different ‘experts’ (e.g. objects, actions, speech) by
learning their combination in order to render them more discriminative. Each expert is filtered via
a simple dynamic attention mechanism that considers its relation to all other experts to enable their
collaboration. This pairwise approach enables, for instance, the sound of a dog barking to inform
the modulation of the RGB features, selecting the features that have encoded the concept of the
dog. As we demonstrate in the sequel, this idea yields improvements in the retrieval performance.
Concretely, we make the following three contributions: (i) We propose the _Collaborative_


_LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_ **3**


_Experts_ framework for learning a joint embedding of video and text by combining a collection of
pretrained embeddings into a single, compact video representation. Our joint video embeddings are
independent of the retrieval text-query and can be pre-computed offline and indexed for efficient
retrieval; (ii) We explore the use of both _general_ video features such as motion, image classification
and audio features, and _specific_ video features such as text embedded on screen and speech obtained
using OCR and ASR respectively. We find that strong generic features deliver good performance,
but that specific, rarely available features remain challenging to use for retrieval [1] . (iii) We assess
the performance of the representation produced by combining all available cues on a number of
retrieval benchmarks, in several cases achieving an advance over prior work.

### **2 Related Work**


**Cross-Modal Embeddings:** A range of prior work has proposed to jointly embed images and text
into the same space [ 16, 17, 18, 30, 44 ], enabling cross-modal retrieval. More recently, several
works have also focused on audio-visual cross-modal embeddings [ 2, 43 ], as well as audio-text
embeddings [ 7 ]. Our goal in this work, however, is to embed videos and natural language sentences
(sometimes multiple sentences) into the same semantic space, which is made more challenging
by the high dimensional content of videos.
**Video-Text Embeddings:** While a large number of works [ 12, 45, 46, 55, 62 ] have focused on
learning visual semantic embeddings for video and language, many of these existing approaches
are based on image-text embedding methods by design and typically focus on single visual frames.
Mithun el al. [ 42 ] observe that a simple adaptation of a state-of-the-art image-text embedding
method [ 16 ] by mean-pooling features from video frames provides a better result than many prior
video-text retrieval approaches [ 12, 45 ]. However, such methods do not take advantage of the rich
and varied additional information present in videos, including motion dynamics, speech and other
background sounds, which may influence the concepts in human captions to a considerable extent.
Consequently, there has been a growing interest in fusing information from other modalities—

[ 39, 42 ] utilise the audio stream (but do not exploit speech content) and use models pretrained for
action recognition to extract motion features. These methods do not make use of speech-to-text
or OCR for additional cues, which have nevertheless been used successfully to understand videos
in other domains, particularly lecture retrieval [ 49, 63 ] (where the videos consist of slide shows)
and news broadcast [ 21 ] retrieval, where a large fraction of the content is displayed on screen in
the form of text. Our approach draws particular inspiration from the powerful joint embedding
proposed by [ 39 ] (which in turn, builds on the classical Mixtures-of-Experts model [ 28 ]) and
extends it to investigate additional cues (such as speech and text) and make more effective use of
pretrained features via the robust collaborative gating mechanism described in Sec. 3.
**Annotation scarcity:** A key challenge for video-retrieval is the small size of existing training
datasets, due to the high cost of annotating videos with natural language. We therefore propose
to use the knowledge from existing embeddings pretrained on a wide variety of other tasks. This
idea is not new: semantic projections of visual inputs in the form of ‘experts’ was used by [ 15 ]
for the task of image retrieval and has also been central to modern video retrieval methods such as

[ 39, 42 ]. More recently, alternative approaches to addressing the issue of annotation scarcity have
been explored, which include self-supervised [ 54 ] and weakly-supervised [ 70 ] video-text models.


1 Note that this finding differs from the previous version of this paper (see appendix A.1).


**4** _LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_


“Expert”
Features













































Figure 2: (Left): The proposed Collaborative Experts framework for learning a joint video-text
embedding (coloured boxes denote learnable parameters). The information provided by each
pretrained “expert” (potentially with multiple experts from a single domain) is temporally
aggregated as it enters the video encoder and then refined through the use of a collaborative gating
mechanism (right) to obtain the video-embedding (for visual clarity, we show the interaction of
just a single expert with three others, though in practice all experts are used—see Sec. 3.1 for
details). Note that to maintain retrieval efficiency, collaboration occurs only between video experts
(the text-query and video embeddings are computed independently).

### **3 Collaborative Experts**


Given a set of videos with corresponding text captions, we would like to create a pair of functions
_φ_ _v_ and _φ_ _t_ that map sensory video data and text into a joint embedding space that respects this
correspondence—embeddings for paired text and video should lie close together, while embeddings for text and video that do not match should lie far apart. We would also like _φ_ _v_ and _φ_ _t_ to
be independent of each other to enable efficient retrieval: the process of querying then reduces to a
distance comparison between the embedding of the query and the embeddings of the collection to
be searched (which can be pre-computed offline). The proposed Collaborative Experts framework
for learning these functions is illustrated in Fig. 2. In this work, we pay particular attention to the
design of the video encoder _φ_ _v_ and the process of combining information from different video
modalities (Sec. 3.1). To complete the framework, we then discuss how the query text is encoded
and the ranking loss used to learn the joint embedding space (Sec. 3.2).


**3.1** **Video Encoder**


To construct the video encoder _φ_ _v_, we draw on a collection of pretrained, single-modality experts. These operate on the video sensory data **v** and project it to a collection of _n_ variable-length
task-specific embeddings _{_ Ψ [(] var [1][)] [(] **[v]** [)] _[,...,]_ [Ψ] [(] var _[n]_ [)] [(] **[v]** [)] _[}]_ [. Here] [ Ψ] [(] var _[i]_ [)] [represents the] _[ i]_ _[th]_ [ expert (we use]
the “var” subscript to denote a variable-length output when applied to a sequence of frames)


_LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_ **5**


whose parameters have been learned on a prior task such as object classification and then frozen.
Each element of this collection is then aggregated along its temporal dimension to produce
fixed-size, task-specific embeddings per video _{_ Ψ [(][1][)] ( **v** ) _,...,_ Ψ [(] _[n]_ [)] ( **v** ) _}_ . Any temporal aggregation
function may be used here—in this work, we use simple average pooling to aggregate “slow”
visual features such as objects and scenes, and NetVLAD [ 3 ] to aggregate more dynamic audio and word features (see Sec. 4.1 for further details). Next, to enable their combination, we
apply linear projections to transform these task-specific embeddings to a common dimensionality. Our goal when fusing the resulting representations together into a single condensed video
representation is to capture the valuable complementary information between task-specific projections while simultaneously filtering out irrelevant noise and resolving individual expert conflicts
on a _per-sample basis_ . To do so, this we propose a collaborative gating module, described

next.


**Collaborative Gating:** The collaborative gating module comprises two operations: (1) Prediction
of attention vectors for every expert projection _T_ = _{T_ [(][1][)] ( **v** ) _,...,T_ [(] _[n]_ [)] ( **v** ) _}_ ; and (2) modulation of
expert responses. Inspired by the relational reasoning module proposed by [ 51 ] for visual question
answering, we define the attention vector of the _i_ _[th]_ expert projection _T_ _i_ as follows:

### T [(] [i] [)] ( v )= h φ ( ∑ g θ (Ψ [(] [i] [)] ( v ), Ψ [(] [j] [)] ( v ))), (1)

_j_ = _̸_ _i_


where functions _h_ _φ_ and _g_ _θ_ are used to model the pairwise relationship between projection Ψ [(] _[i]_ [)] and
projection Ψ [(] _[j]_ [)] . Of these, _g_ _θ_ is used to infer pairwise task relationships, while _h_ _φ_ maps the sum
of all pairwise relationships into a single attention vector. In this work, we instantiate both _h_ _φ_ and
_g_ _θ_ as multi-layer perceptrons (MLPs). Note that the functional form of Equation (1) dictates that
the attention vector of any expert projection should consider the potential relationships between all
pairs associated with this expert. That is to say, the quality of each expert Ψ [(] _[j]_ [)] should contribute in
determining and selecting the information content from Ψ [(] _[i]_ [)] in the final decision. It is also worth
noting that the collaborative gating module uses the same functions _g_ _θ_ and _h_ _φ_ (shared weights) to
compute all pairwise relationships. This mode of operation encourages greater generalisation, since
_g_ _θ_ and _h_ _φ_ are encouraged not to over-fit to features of any particular pair of tasks. After the attention
vectors _T_ = _{T_ [(][1][)] ( **v** ) _,...,T_ [(] _[n]_ [)] ( **v** ) _}_ have been computed, each expert projection is modulated follows:


Ψ [(] _[i]_ [)] ( **v** )=Ψ [(] _[i]_ [)] ( **v** ) _◦σ_ ( _T_ [(] _[i]_ [)] ( **v** )) _,_ (2)


where _σ_ is an element-wise sigmoid activation and _◦_ is the element-wise multiplication
(Hadamard product). This gating function re-calibrates the strength of different activations of
Ψ [(] _[i]_ [)] ( **v** ) and selects which information is highlighted or suppressed, providing the model with a
powerful mechanism for dynamically filtering content from different experts. A diagram of the
mechanism is shown in Fig. 2 (right). The final video embedding is then obtained by passing the
modulated responses of each expert through a Gated Embedding Module (GEM) [ 39 ] (note that
this operation produces l2-normalized outputs) before concatenating the outputs together into a
single fixed-length vector.


**3.2** **Text Query Encoder and Training Loss**


To construct the text embeddings, query sentences are first mapped to a sequence of feature vectors
with pretrained contextual word-level embeddings (see Sec. 4.1 for details)—as with the video
experts, the parameters of this first stage are frozen. These are then aggregated, again using


**6** _LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_


NetVLAD [ 3 ]. Following aggregation, we follow the text encoding architecture proposed by [ 39 ],
which projects the aggregated features to separate subspaces for each expert using GEMs (as
with the video encoder, producing l2-normalized outputs). Each projection is then scaled by a
mixture weight (one scalar weight per expert projection), which is computed by applying a single
linear layer to the aggregated text-features, and passing the result through a softmax to ensure
that the mixture weights sum to one (see [ 39 ] for further details). Finally, the scaled outputs are
concatenated, producing a vector of dimensionality that matches that of the video embedding.
With the video encoder _φ_ _v_ and text encoder _φ_ _t_ as described, the similarity _s_ _i_ _[j]_ [of the] _[ i]_ _[th]_ [ video,] **[ v]** _[i]_ [,]
and the and _j_ _[th]_ caption, **t** _j_, can then be directly computed as the cosine of the angle between their
respective embeddings _φ_ _v_ ( **v** _i_ ) _[T]_ _φ_ _t_ ( **t** _j_ ) . During optimisation, the parameters of the video encoder
(including the collaborative gating module) and text query encoder (the coloured regions of Fig. 2)
are learned jointly. Training proceeds by sampling a sequence of minibatches of corresponding
video-text pairs _{_ **v** _i_ _,_ **t** _i_ _}_ _[N]_ _i_ = _[B]_ 1 [and minimising a] _[ Bidirectional Max-margin Ranking Loss]_ [ [][53][]:]


_̸_



_L_ _r_ = [1]

_N_ _B_ _̸_



_N_ _B_
### ∑ max(0,m + s i [j] [−][s] i [i] [)+][max][(][0] [,][m] [+] [s] [i] j [−][s] i [i] [)] (3)

_i_ =1 _,_ _j_ = _̸_ _i_



_̸_


where _N_ _B_ is the batch size, and _m_ is a fixed constant which is set as a hyperparameter. When
assessing retrieval performance, at test time the embedding distances are simply computed via their
inner product, as described above.


**3.2.1** **Missing Experts**


When a set of expert features are missing, such as when there is no speech in the audio track, we
simply zero-pad the missing experts when estimating the similarity score. To compensate for the implicit scaling introduced by missing experts (the similarity is effectively computed between shorter
embeddings), we follow the elegant approach proposed by [ 39 ] and simply remove the mixture
weights for missing experts, then renormalise the remaining weights such that they sum to one.

### **4 Experiments**


In this section, we evaluate our model on five benchmarks for video retrieval tasks. The description
of datasets, implementation details and evaluation metric are provided in Sec. 4.1. A comprehensive
comparison on general video retrieval benchmarks is reported in Sec. 4.2. We present an ablation
study in Sec. 4.3 to explore how the performance of the proposed method is affected by different
model configurations, including the aggregation methods, importance of different experts and
number of captions in training.


**4.1** **Datasets, Implementation Details and Metrics**


**Datasets:** We perform experiments on five video datasets: MSR-VTT [ 61 ], LSMDC [ 50 ],
MSVD [ 8 ], DiDeMo [ 1 ] and ActivityNet-captions [ 32 ], covering a challenging set of domains
which include videos from YouTube, personal collections and movies.
**Expert Features:** In order to capture the rich content of a video, we draw on existing powerful
representations for a number of different semantic tasks. These are first extracted at a frame-level,
then aggregated to produce a single feature vector per modality per video. _RGB “object”_ framelevel embeddings of the visual data are generated with two models: an SENet-154 model [ 24 ]


_LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_ **7**


(pretrained on ImageNet for the task of image classification), and a ResNext-101 [ 60 ] pretrained
on Instagram hashtags [ 37 ]. _Motion_ embeddings are generated using the I3D inception model [ 6 ]
and a 34-layer R(2+1)D model [ 56 ] trained on IG-65m [ 19 ]. _Face_ embeddings are extracted in two
stages: (1) Each frame is passed through an SSD face detector [ 4, 35 ] to extract bounding boxes; (2)
The image region of each box is passed through a ResNet50 [ 22 ] that has been trained for the task
of face classification on the VGGFace2 dataset [ 5 ]. _Audio_ embeddings are obtained with a VGGish
model, trained for audio classification on the YouTube-8m dataset [ 23 ]. _Speech-to-Text_ features
are extracted using the Google Cloud speech API, to extract word tokens from the audio stream,
which are then encoded via pretrained word2vec embeddings [ 41 ]. _Optical Character Recognition_
is done in two stages: (1) Each frame is passed through the Pixel Link [ 10 ] text detection model
to extract bounding boxes for text; (2) The image region of each box is passed through a model

[ 36 ] that has been trained for scene text recognition on the Synth90K dataset[ 27 ]. The text is then
encoded via a pretrained word2vec embedding model [41].
**Temporal Aggregation:** We adopt a simple approach to aggregating the features described above.
For appearance, motion, scene and face embeddings, we average frame-level features along the
temporal dimension to produce a single feature vector per video (we found max-pooling to perform
similarly). For speech, audio and OCR features, we adopt the NetVLAD mechanism proposed
by [ 3 ], which has proven effective in the retrieval setting [ 38 ]. As noted in Sec. 3.1, all aggregated
features are projected to a common size (768 dimensions).
**Text:** Each word is encoded using pretrained word2vec word embeddings [ 41 ] and then passed
through a pretrained OpenAI-GPT model [ 48 ] to extract contextual word embeddings. Finally, the
word embeddings in each sentence are aggregated using NetVLAD.
**Dataset-specific details:** Except where noted otherwise for ablation purposes, we use each of the
embeddings described above for the MSR-VTT, ActivityNet and DiDeMo datasets. For MSVD,
we extract the subset of features which do not require an audio stream (since no audio is available
with the dataset). For LSMDC, we re-use the existing face, text and audio features made available
by [39], and combine them with the remaining features described above.
**Training Details:** The CE framework is implemented with PyTorch [ 47 ]. Optimisation is performed with the Lookahead solver [ 29 ] in combination with RAdam [ 34 ] (implementation by [ 59 ]).
Optimisation settings and the hyperparameter selection procedure is described in the appendix.
**Evaluation Metrics:** We follow prior work (e.g. [ 12, 39, 42, 65, 66 ]) and report standard retrieval
metrics (where existing work enables comparison) including median rank (lower is better), mean
rank (lower is better) and R@K (recall at rank K—higher is better). When computing videoto-sentence metrics for datasets with multiple independent sentences per video (MSR-VTT and
MSVD), we follow the evaluation protocol used in prior work [ 13, 14, 42 ] which corresponds to
reporting the minimum rank among all valid text descriptions for a given video query. For each
benchmark, we report the mean and standard deviation of three randomly seeded runs.


**4.2** **Comparison to Prior State-of-the-Art**


We first compare the proposed method with the existing state-of-the-art on the MSR-VTT benchmark for the tasks of sentence-to-video and video-to-sentence retrieval Tab. 1. Driven by strong
expert features, we observe that Collaborative Experts (CE) consistently improves retrieval performance for both sentence and video queries. We next evaluate the performance of the CE framework
on the LSMDC benchmark for sentence-to-video retrieval (Tab. 2, left) and observe that CE matches
or outperforms all prior work, including the prior state-of-the-art method [ 39 ] which incorporates
additional training images and captions from the COCO benchmark during training, but uses fewer
experts. We observe similar trends in the results for the MSVD retrieval benchmark (Tab. 2, right).


**8** _LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_

|Col1|Col2|Text =⇒Video|Video =⇒Text|
|---|---|---|---|
|Method|Test-set|R@1<br>R@5<br>R@10<br>MdR<br>MnR|R@1<br>R@5<br>R@10<br>MdR<br>MnR|
|JSFusion [65]<br>**CE**|1k-A<br>1k-A|10.2<br>31.2<br>43.2<br>13<br>-<br>**20.9**_±_1_._2** 48.8**_±_0_._6** 62.4**_±_0_._8<br>**6**_±_0<br>**28.2**_±_0_._8|-<br>-<br>-<br>-<br>-<br>**20.6**_±_0_._6** 50.3**_±_0_._5** 64.0**_±_0_._2** 5.3**_±_0_._6** 25.1**_±_0_._8|
|MoEE [39]<br>MoEECOCO [39]<br>**CE**|1k-B<br>1k-B<br>1k-B|13.6<br>37.9<br>51.0<br>10<br>-<br>14.2<br>39.2<br>53.8<br>9<br>-<br>**18.2**_±_0_._7** 46.0**_±_0_._4** 60.7**_±_0_._2<br>**7**_±_0<br>**35.3**_±_1_._1|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**18.0**_±_0_._8** 46.0**_±_0_._5** 60.3**_±_0_._5** 6.5**_±_0_._5** 30.6**_±_1_._2|
|VSE [42]<br>VSE++ [42]<br>Mithun et al. [42]<br>W2VV [13]<br>Dual Enc. [14]<br>E2E [40]<br>**CE**|Full<br>Full<br>Full<br>Full<br>Full<br>Full<br>Full|5.0<br>16.4<br>24.6<br>47<br>215.1<br>5.7<br>17.1<br>24.8<br>65<br>300.8<br>7.0<br>20.9<br>29.7<br>38<br>213.8<br>6.1<br>18.7<br>27.5<br>45<br>-<br>7.7<br>22.0<br>31.8<br>32<br>-<br>9.9<br>24.0<br>32.4<br>29.5<br>-<br>**10**_._**0**_±_0_._1** 29**_._**0**_±_0_._3** 41**_._**2**_±_0_._2** 16**_±_0<br>**86**_._**8**_±_0_._3|7.7<br>20.3<br>31.2<br>28<br>185.8<br>10.2<br>25.4<br>35.1<br>25<br>228.1<br>12.5<br>32.1<br>42.4<br>16<br>134.0<br>11.8<br>28.9<br>39.1<br>21<br>-<br>13.0<br>30.8<br>43.3<br>15<br>-<br>-<br>-<br>-<br>-<br>-<br>**15**_._**6**_±_0_._3** 40**_._**9**_±_1_._4** 55**_._**2**_±_1_._0** 8**_._**3**_±_0_._6** 38**_._**1**_±_1_._8|



Table 1: Retrieval with sentences and videos on the MSR-VTT dataset. R @ k denotes recall @ k

(higher is better), MdR and MnR denote median rank and mean rank resp. (lower is better).
Standard deviations are reported from three randomly seeded runs. 1k-A and 1k-B denote test
sets of 1000 randomly sampled text-video pairs used by [65] and [39] resp.


In Tab. 4, we compare with prior work on the ActivityNet paragraph-video retrieval benchmark (note
that we compare to methods which use the same level of annotation as our approach i.e. video-level
annotation), and see that CE is competitive. Finally, in Tab. 3 we provide a comparison with previously reported numbers on the DiDeMo benchmark and see that CE again outperforms prior work.






|Col1|Text =⇒Video|
|---|---|
|Method|R@1<br>R@5<br>R@10<br>MdR|
|Yu et al. [64]†<br>CCA [31] (rep. by [39])<br>JSFusion [65]‡<br>MoEE [39]<br>MoEECOCO [39]<br>**CE**|3.6<br>14.7<br>23.9<br>50<br>7.5<br>21.7<br>31.0<br>33<br>9.1<br>21.2<br>34.1<br>36<br>9.3<br>25.1<br>33.4<br>27<br>10.1<br>25.6<br>34.6<br>27<br>**11.2**_±_0_._4**26.9**_±_1_._1**34.8**_±_2_._0<br>**25.3**_±_3_._1|


|Col1|Text =⇒Video|
|---|---|
|Method|R@1<br>R@5<br>R@10<br>MdR<br>MnR|
|CCA ([62])<br>JMDV [62]<br>VSE [30] ([42])<br>VSE++ [16] ([42])<br>Multi. Cues [42]<br>**CE**|-<br>-<br>-<br>-<br>245.3<br>-<br>-<br>-<br>-<br>236.3<br>12.3<br>30.1<br>42.3<br>14<br>57.7<br>15.4<br>39.6<br>53.0<br>9<br>43.8<br>**20**_._**3**<br>47.8<br>61.1<br>**6**<br>28.3<br>19_._8_±_0_._3**49**_._**0**_±_0_._3**63**_._**8**_±_0_._1<br>**6**_±_0_._0** 23**_._**1**_±_0_._3|



Table 2: Text-to-Video retrieval results on the LSMDC dataset (left) and the MSVD dataset (right).

 - _,_ - denote the winners of the 2016 and 2017 LSMDC challenges, respectively.

|Col1|Text =⇒Video|Video =⇒Text|
|---|---|---|
|Method|R@1<br>R@5<br>R@50<br>MdR<br>MnR|R@1<br>R@5<br>R@50<br>MdR<br>MnR|
|S2VT [57] ([66])<br>FSE [66]<br>**CE**|11.9<br>33.6<br>76.5<br>13<br>-<br>13_._9_±_0_._7 36_±_0_._8 78_._9_±_1_._6<br>11<br>-<br>**16.1**_±_1_._4** 41.1**_±_0_._4** 82.7**_±_0_._3<br>**8.3**_±_0_._6<br>**43.7**_±_3_._6|13.2<br>33.6<br>76.5<br>15<br>-<br>13_._1_±_0_._5 33_._9_±_0_._4 78_._0_±_0_._8<br>12<br>-<br>**15.6**_±_1_._3** 40.9**_±_0_._4** 82.2**_±_1_._3<br>**8.2**_±_0_._3<br>**42.4**_±_3_._3|



Table 3: Comparison of paragraph-video retrieval methods trained with video-level information
on the DiDeMo dataset.


**4.3** **Ablation Studies**


In this section, we provide ablation studies to empirically assess: (1) the effectiveness of the
proposed collaborative experts framework vs other aggregation strategies; (2) the importance of
using of a diverse range of experts with differing levels of specificity; (3) the relative value of using
experts in comparison to simply having additional annotated training data.


_LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_ **9**

|Col1|Text =⇒Video|Video =⇒Text|
|---|---|---|
|Method|R@1<br>R@5<br>R@50<br>MdR<br>MnR|R@1<br>R@5<br>R@50<br>MdR<br>MnR|
|LSTM-YT [58] ([66])<br>NOCTXT [57] ([66])<br>DENSE [32]<br>FSE [66]<br>HSE(4SEGS) [66]†<br>CE|0.0<br>4.0<br>24.0<br>102<br>-<br>5.0<br>14.0<br>32.0<br>78<br>-<br>14.0<br>32.0<br>65.0<br>34<br>-<br>18_._2_±_0_._2 44_._8_±_0_._4 89_._1_±_0_._3 7<br>-<br>**20.5**<br>**49.3**<br>-<br>-<br>-<br>18_._2_±_0_._3 47_._7_±_0_._6 91_._4_±_0_._4 6_±_0<br>23_._1_±_0_._5|0.0<br>7.0<br>38.0<br>98<br>-<br>7.0<br>18.0<br>45.0<br>56<br>-<br>18.0<br>36.0<br>74.0<br>32<br>16_._7_±_0_._8 43_._1_±_1_._1 88_._4_±_0_._3 7<br>-<br>**18.7**<br>**48.1**<br>-<br>-<br>17_._7_±_0_._6 46_._6_±_0_._7 90_._9_±_0_._2 6_±_0<br>24_._4_±_0_._5|



Table 4: Comparison of paragraph-video retrieval methods trained with video-level information
on the ActivityNet-captions dataset (val1 test-split).


**Aggregation method:** We compare the use of collaborative experts with several other baselines
(with access to the same experts) for embedding aggregation including: (1) simple expert concatenation; (2) CE without projecting to a common dimension, without mixture weights and
without the collaborative gating module described in Sec. 3.1; (3) the state of the art MoEE [ 39 ]
method (equivalent to CE without the common projection and collaborative gating) and (4) CE
without collaborative gating. The results, presented in Tab. 6 (left), demonstrate the contribution
of collaborative gating which improves performance and leads to a more efficient parameterisation
than the prior state of the art.
**Importance of different experts:** The value of different experts is assessed in Tab. 5 (note that
since several experts are not present in all videos, we combine them with features produced by
a “scene” expert pretrained on Places365 [ 69 ]—the expert with the lowest performance that is
consistently available as a baseline to enable a more meaningful comparison). There is considerable
variance in the effect produced by different choices of expert. Using stronger features within a given
modality (pretraining on Instagram [ 37 ] rather than Kinetics [ 6 ] (resp. ImageNet) [ 11 ] for actions
(resp. object) experts can yield a significant boost in performance). The cues from scarce features
(such as speech, face and OCR) which are often missing from videos (see Fig. 1, right) provide significantly weaker cues and bring a limited improvement to performance when used in combination.



Text = _⇒_ Video
Experts R@1 R@5 R@10 MdR MnR
Scene 4 _._ 0 _±_ 0 _._ 1 14 _._ 1 _±_ 0 _._ 1 22 _._ 4 _±_ 0 _._ 3 50 _._ 0 _±_ 1 _._ 0 201 _._ 3 _±_ 1 _._ 6
Scene+Speech 4 _._ 6 _±_ 0 _._ 1 15 _._ 5 _±_ 0 _._ 2 24 _._ 4 _±_ 0 _._ 2 44 _._ 7 _±_ 1 _._ 2 183 _._ 6 _±_ 1 _._ 7
Scene+Audio 5 _._ 6 _±_ 0 _._ 0 18 _._ 7 _±_ 0 _._ 1 28 _._ 2 _±_ 0 _._ 1 33 _._ 7 _±_ 0 _._ 6 140 _._ 8 _±_ 0 _._ 3
Scene+Action(KN) 5 _._ 3 _±_ 0 _._ 3 17 _._ 6 _±_ 0 _._ 8 27 _._ 1 _±_ 0 _._ 9 36 _._ 0 _±_ 1 _._ 7 158 _._ 7 _±_ 1 _._ 6
Scene+Obj(IN) 5 _._ 0 _±_ 0 _._ 2 16 _._ 6 _±_ 0 _._ 7 25 _._ 5 _±_ 1 _._ 0 40 _._ 7 _±_ 2 _._ 1 173 _._ 1 _±_ 3 _._ 3
Scene+Obj(IG) 7 _._ 2 _±_ 0 _._ 1 22 _._ 3 _±_ 0 _._ 3 33 _._ 0 _±_ 0 _._ 2 25 _._ 3 _±_ 0 _._ 6 125 _._ 1 _±_ 0 _._ 1
Scene+Action(IG) 6 _._ 8 _±_ 0 _._ 1 21 _._ 7 _±_ 0 _._ 1 32 _._ 4 _±_ 0 _._ 1 25 _._ 7 _±_ 0 _._ 6 122 _._ 1 _±_ 0 _._ 3
Scene+OCR 4 _._ 1 _±_ 0 _._ 1 14 _._ 1 _±_ 0 _._ 1 22 _._ 2 _±_ 0 _._ 2 50 _._ 3 _±_ 1 _._ 2 203 _._ 1 _±_ 4 _._ 4
Scene+Face 4 _._ 1 _±_ 0 _._ 1 14 _._ 2 _±_ 0 _._ 3 22 _._ 4 _±_ 0 _._ 4 49 _._ 7 _±_ 0 _._ 6 194 _._ 2 _±_ 5 _._ 1



Text = _⇒_ Video
Experts R@1 R@5 R@10 MdR MnR
Scene 4 _._ 0 _±_ 0 _._ 1 14 _._ 1 _±_ 0 _._ 1 22 _._ 4 _±_ 0 _._ 3 50 _._ 0 _±_ 1 _._ 0 201 _._ 3 _±_ 1 _._ 6
Prev.+Speech 4 _._ 6 _±_ 0 _._ 1 15 _._ 5 _±_ 0 _._ 2 24 _._ 4 _±_ 0 _._ 2 44 _._ 7 _±_ 1 _._ 2 183 _._ 6 _±_ 1 _._ 7
Prev.+Audio 5 _._ 8 _±_ 0 _._ 1 19 _._ 0 _±_ 0 _._ 3 28 _._ 8 _±_ 0 _._ 2 32 _._ 3 _±_ 0 _._ 6 136 _._ 8 _±_ 1 _._ 2
Prev.+Action(KN) 6 _._ 7 _±_ 0 _._ 2 21 _._ 8 _±_ 0 _._ 4 32 _._ 5 _±_ 0 _._ 5 25 _._ 3 _±_ 0 _._ 6 115 _._ 9 _±_ 1 _._ 0
Prev.+Obj(IN) 7 _._ 5 _±_ 0 _._ 1 23 _._ 4 _±_ 0 _._ 0 34 _._ 1 _±_ 0 _._ 2 23 _._ 7 _±_ 0 _._ 6 111 _._ 9 _±_ 0 _._ 6
Prev.+Obj(IG) 9 _._ 5 _±_ 0 _._ 2 27 _._ 7 _±_ 0 _._ 1 39 _._ 4 _±_ 0 _._ 1 18 _._ 0 _±_ 0 _._ 0 92 _._ 6 _±_ 0 _._ 4
Prev.+Action(IG) 9 _._ 9 _±_ 0 _._ 1 28 _._ 6 _±_ 0 _._ 3 40 _._ 7 _±_ 0 _._ 1 17 _._ 0 _±_ 0 _._ 0 86 _._ 4 _±_ 0 _._ 4
Prev.+OCR 10 _._ 0 _±_ 0 _._ 1 28 _._ 8 _±_ 0 _._ 2 40 _._ 9 _±_ 0 _._ 2 16 _._ 7 _±_ 0 _._ 6 87 _._ 3 _±_ 0 _._ 8
Prev.+Face 10 _._ 0 _±_ 0 _._ 1 29 _._ 0 _±_ 0 _._ 3 41 _._ 2 _±_ 0 _._ 2 16 _._ 0 _±_ 0 _._ 0 86 _._ 8 _±_ 0 _._ 3



Table 5: **The importance of different experts** (Left): The value of different experts in
combination with a baseline set for text-video retrieval and (right) their cumulative effect on
MSR-VTT (here Prev. denotes the experts used in the previous row).


**Number of Captions in training:** An emerging idea in our community is that many machine
perception tasks might be solved through the combination of simple models and large-scale training
sets, reminiscent of the “big-data” hypothesis [ 20 ]. In this section, we perform an ablation study
to assess the relative importance of access to pretrained experts and additional video description
annotations. To do so, we measure the performance of the CE model as we vary (1) the number
of descriptions available per-video during training and (2) the number of experts it has access to.
The results are shown in Tab. 6 (right). We observe that increasing the number of training captions


**10** _LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_



Aggreg. R@1 R@5 R@10 MdR Params
Concat 0 _._ 0 _±_ 0 _._ 0 0 _._ 0 _±_ 0 _._ 0 0 _._ 0 _±_ 0 _._ 0 1495 _._ 5 _±_ 0 _._ 0 369.72k
CE - MW,P,CG 8 _._ 5 _±_ 0 _._ 1 25 _._ 9 _±_ 0 _._ 3 37 _._ 6 _±_ 0 _._ 2 19 _._ 0 _±_ 0 _._ 0 246.22M
MoEE [39] 9 _._ 6 _±_ 0 _._ 1 28 _._ 0 _±_ 0 _._ 2 39 _._ 7 _±_ 0 _._ 2 17 _._ 7 _±_ 0 _._ 6 400.41 M
CE - CG 9 _._ 7 _±_ 0 _._ 1 28 _._ 1 _±_ 0 _._ 2 40 _._ 2 _±_ 0 _._ 1 17 _._ 0 _±_ 0 _._ 0 181.07 M
CE 10 _._ 0 _±_ 0 _._ 1 29 _._ 0 _±_ 0 _._ 3 41 _._ 2 _±_ 0 _._ 2 16 _._ 0 _±_ 0 _._ 0 183.45 M



Expert Num. Captions R@1 R@5 R@10 MdR
Obj(IN) 1 2 _._ 6 _±_ 0 _._ 1 9 _._ 3 _±_ 0 _._ 4 15 _._ 0 _±_ 0 _._ 7 101 _._ 3 _±_ 15 _._ 5
Obj(IN) 20 4 _._ 9 _±_ 0 _._ 1 16 _._ 5 _±_ 0 _._ 2 25 _._ 3 _±_ 0 _._ 4 40 _._ 7 _±_ 1 _._ 2
All 1 4 _._ 8 _±_ 0 _._ 2 16 _._ 2 _±_ 0 _._ 5 25 _._ 0 _±_ 0 _._ 7 43 _._ 3 _±_ 4 _._ 0
All 20 10 _._ 0 _±_ 0 _._ 1 29 _._ 0 _±_ 0 _._ 3 41 _._ 2 _±_ 0 _._ 2 16 _._ 0 _±_ 0 _._ 0



Table 6: (Left): Aggregation methods for text-video retrieval on MSR-VTT; (Right): The relative
value of training with additional captions vs the value of experts.


per-video from 1 to 20 brings an improvement in performance, approximately comparable to
adding the full collection of experts, suggesting that indeed, adding experts can help to compensate
for a paucity of labelled data. When multiple captions and multiple experts are both available, they
naturally lead to the most robust embedding. Some qualitative examples of videos retrieved by
the multiple-expert, multiple-caption system are provided in Fig. 3.



Query: Guy working on his engine with multiple parts
(GT rank: 29)


Similarity: 0.60 Similarity: 0.59 Similarity: 0.55


Query: Awareness of mosquitoe bites by doctors
(GT rank: 2)



Query: shiny black sports car drives very slowly down road through orange
and white safety cones (GT rank: 10)


Similarity: 0.48 Similarity: 0.46 Similarity: 0.46


Query: Query: Awareness of mosquitoe bites by doctors
_(without CE)_ - (GT rank: 7)



Similarity: 0.38



Similarity: 0.36 Similarity: 0.34 Similarity: 0.34 Similarity: 0.33 Similarity: 0.23



Figure 3: **Qualitative Results on MSR-VTT:** For each query, we show frames from the top
three ranked videos (where present, the ground truth video is indicated by a green box around
the similarity score). Top row: (left) Even for imperfect rankings, the model retrieves reasonable
videos; Failure case (right) the embeddings can fail to differentiate between certain signals (in
this case, ranking cars of the wrong colour above the ground truth video). Bottom row: (left) the
videos retrieved by the proposed model (which assigns its second highest similarity to the correct
video); (right) removing the proposed CE component produces a nosier ranking.

### **5 Conclusion**


In this work, we introduced collaborative experts, a framework for learning a joint video-text
embedding for efficient retrieval. We have shown that using a range of pretrained features and
combining them through an appropriate gating mechanism can boost retrieval performance. In
future work, we plan to explore the use of collaborative experts for other video understanding tasks
such as clustering and summarisation.


**Acknowledgements:** Funding for this research is provided by the EPSRC Programme Grant
Seebibyte EP/M013774/1 and EPSRC grant EP/R03298X/1. A.N. is supported by a Google
PhD Fellowship. We would like to thank Antoine Miech, YoungJae Yu and Bowen Zhang for
their assistance with experiment details. We would like to particularly thank Valentin Gabeur for
identifying a bug in the software implementation that was responsible for the inaccurate results
reported in the initial version of the paper. We would also like to thank Zak Stone and Susie Lim
for their help with cloud computing.


_LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_ **11**

### **References**


[1] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell.
Localizing moments in video with natural language. In _Proceedings of the IEEE International_
_Conference on Computer Vision_, pages 5803–5812, 2017.


[2] Relja Arandjelovic and Andrew Zisserman. Look, listen and learn. In _Proceedings of the IEEE_
_International Conference on Computer Vision_, pages 609–617, 2017.


[3] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla, and Josef Sivic. Netvlad: Cnn architecture
for weakly supervised place recognition. In _Proceedings of the IEEE Conference on Computer Vision_
_and Pattern Recognition_, pages 5297–5307, 2016.


[4] G. Bradski. The OpenCV Library. _Dr. Dobb’s Journal of Software Tools_, 2000.


[5] Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman. VGGFace2: A dataset for recognising faces
across pose and age. In _Proc. Int. Conf. Autom. Face and Gesture Recog._, 2018.


[6] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics
dataset. In _proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, pages
6299–6308, 2017.


[7] Gal Chechik, Eugene Ie, Martin Rehn, Samy Bengio, and Dick Lyon. Large-scale content-based audio
retrieval from text queries. In _Proceedings of the 1st ACM international conference on Multimedia_
_information retrieval_, pages 105–112. ACM, 2008.


[8] David L Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In
_Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human_
_Language Technologies-Volume 1_, pages 190–200. Association for Computational Linguistics, 2011.


[[9] J. H. Daniel. ig65m-pytorch. https://github.com/moabitcoin/ig65m-pytorch, 2019.](https://github.com/moabitcoin/ig65m-pytorch)


[10] Dan Deng, Haifeng Liu, Xuelong Li, and Deng Cai. Pixellink: Detecting scene text via instance
segmentation. In _Thirty-Second AAAI Conference on Artificial Intelligence_, 2018.


[11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_,
pages 248–255. Ieee, 2009.


[12] Jianfeng Dong, Xirong Li, and Cees GM Snoek. Word2visualvec: Image and video to sentence
matching by visual feature prediction. _arXiv preprint arXiv:1604.06838_, 2016.


[13] Jianfeng Dong, Xirong Li, and Cees GM Snoek. Predicting visual features from text for image and
video caption retrieval. _IEEE Transactions on Multimedia_, 20(12):3377–3388, 2018.


[14] Jianfeng Dong, Xirong Li, Chaoxi Xu, Shouling Ji, and Xun Wang. Dual dense encoding for
zero-example video retrieval. _Proceedings of the IEEE Conference on Computer Vision and Pattern_
_Recognition_, 2019.


[15] Matthijs Douze, Arnau Ramisa, and Cordelia Schmid. Combining attributes and fisher vectors for
efficient image retrieval. In _CVPR 2011_, pages 745–752. IEEE, 2011.


[16] Fartash Faghri, David J Fleet, Jamie Ryan Kiros, and Sanja Fidler. Vse++: Improved visual-semantic
embeddings. _arXiv preprint arXiv:1707.05612_, 2(7):8, 2017.


[17] Ali Farhadi, Mohsen Hejrati, Mohammad Amin Sadeghi, Peter Young, Cyrus Rashtchian, Julia
Hockenmaier, and David Forsyth. Every picture tells a story: Generating sentences from images. In
_European conference on computer vision_, pages 15–29. Springer, 2010.


**12** _LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_


[18] Andrea Frome, Greg S Corrado, Jon Shlens, Samy Bengio, Jeff Dean, Tomas Mikolov, et al. Devise: A
deep visual-semantic embedding model. In _Advances in neural information processing systems_, pages
2121–2129, 2013.


[19] Deepti Ghadiyaram, Du Tran, and Dhruv Mahajan. Large-scale weakly-supervised pre-training for
video action recognition. In _Proceedings of the IEEE Conference on Computer Vision and Pattern_
_Recognition_, pages 12046–12055, 2019.


[20] Alon Halevy, Peter Norvig, and Fernando Pereira. The unreasonable effectiveness of data. 2009.


[21] Alexander G Hauptmann, Rong Jin, and Tobun Dorbin Ng. Multi-modal information retrieval from
broadcast video using ocr and speech recognition. In _Proceedings of the 2nd ACM/IEEE-CS joint_
_conference on Digital libraries_, pages 160–161. ACM, 2002.


[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks.
In _European conference on computer vision_, pages 630–645. Springer, 2016.


[23] Shawn Hershey, Sourish Chaudhuri, Daniel P. W. Ellis, Jort F. Gemmeke, Aren Jansen, Channing Moore,
Manoj Plakal, Devin Platt, Rif A. Saurous, Bryan Seybold, Malcolm Slaney, Ron Weiss, and Kevin
Wilson. Cnn architectures for large-scale audio classification. In _International Conference on Acoustics,_
_Speech and Signal Processing (ICASSP)_ . 2017. URL [https://arxiv.org/abs/1609.09430](https://arxiv.org/abs/1609.09430) .


[24] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, and Enhua Wu. Squeeze-and-excitation networks. _IEEE_
_transactions on pattern analysis and machine intelligence_, 2019.


[25] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected
convolutional networks. In _Proceedings of the IEEE conference on computer vision and pattern_
_recognition_, pages 4700–4708, 2017.


[26] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by
reducing internal covariate shift. _arXiv preprint arXiv:1502.03167_, 2015.


[27] Max Jaderberg, Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Synthetic data and artificial
neural networks for natural scene text recognition. _arXiv preprint arXiv:1406.2227_, 2014.


[28] Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. _Neural_
_computation_, 6(2):181–214, 1994.


[29] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint_
_arXiv:1412.6980_, 2014.


[30] Ryan Kiros, Ruslan Salakhutdinov, and Richard S Zemel. Unifying visual-semantic embeddings with
multimodal neural language models. _arXiv preprint arXiv:1411.2539_, 2014.


[31] Benjamin Klein, Guy Lev, Gil Sadeh, and Lior Wolf. Associating neural word embeddings with deep
image representations using fisher vectors. In _Proceedings of the IEEE Conference on Computer Vision_
_and Pattern Recognition_, pages 4437–4446, 2015.


[32] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning
events in videos. In _Proceedings of the IEEE International Conference on Computer Vision_, pages
706–715, 2017.


[33] Liam Li, Kevin Jamieson, Afshin Rostamizadeh, Ekaterina Gonina, Moritz Hardt, Benjamin Recht, and
Ameet Talwalkar. Massively parallel hyperparameter tuning. _arXiv preprint arXiv:1810.05934_, 2018.


[34] Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei
Han. On the variance of the adaptive learning rate and beyond. _arXiv preprint arXiv:1908.03265_, 2019.


_LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_ **13**


[35] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and
Alexander C Berg. Ssd: Single shot multibox detector. In _European conference on computer vision_,
pages 21–37. Springer, 2016.


[36] Yang Liu, Zhaowen Wang, Hailin Jin, and Ian Wassell. Synthetically supervised feature learning for
scene text recognition. In _Proceedings of the European Conference on Computer Vision (ECCV)_, pages
435–451, 2018.


[37] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin
Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In
_Proceedings of the European Conference on Computer Vision (ECCV)_, pages 181–196, 2018.


[38] Antoine Miech, Ivan Laptev, and Josef Sivic. Learnable pooling with context gating for video
classification. _arXiv preprint arXiv:1706.06905_, 2017.


[39] Antoine Miech, Ivan Laptev, and Josef Sivic. Learning a text-video embedding from incomplete and
heterogeneous data. _arXiv preprint arXiv:1804.02516_, 2018.


[40] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman.
End-to-end learning of visual representations from uncurated instructional videos. _arXiv preprint_
_arXiv:1912.06430_, 2019.


[41] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations
in vector space. _arXiv preprint arXiv:1301.3781_, 2013.


[42] Niluthpol Chowdhury Mithun, Juncheng Li, Florian Metze, and Amit K Roy-Chowdhury. Learning
joint embedding with multimodal cues for cross-modal video-text retrieval. In _Proceedings of the 2018_
_ACM on International Conference on Multimedia Retrieval_, pages 19–27. ACM, 2018.


[43] Arsha Nagrani, Samuel Albanie, and Andrew Zisserman. Learnable pins: Cross-modal embeddings
for person identity. In _Proceedings of the European Conference on Computer Vision (ECCV)_, pages
71–88, 2018.


[44] Hyeonseob Nam, Jung-Woo Ha, and Jeonghee Kim. Dual attention networks for multimodal reasoning
and matching. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_,
pages 299–307, 2017.


[45] Mayu Otani, Yuta Nakashima, Esa Rahtu, Janne Heikkilä, and Naokazu Yokoya. Learning joint
representations of videos and sentences with web image search. In _European Conference on Computer_
_Vision_, pages 651–667. Springer, 2016.


[46] Yingwei Pan, Tao Mei, Ting Yao, Houqiang Li, and Yong Rui. Jointly modeling embedding and
translation to bridge video and language. In _Proceedings of the IEEE conference on computer vision_
_and pattern recognition_, pages 4594–4602, 2016.


[47] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming
Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.


[48] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language
understanding by generative pre-training. 2018. URL [https://s3-us-west-2.amazonaws.](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
[com/openai-assets/research-covers/language-unsupervised/language_](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

[understanding_paper.pdf.](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


[49] N Radha. Video retrieval using speech and text in video. In _2016 International Conference on Inventive_
_Computation Technologies (ICICT)_, volume 2, pages 1–6. IEEE, 2016.


**14** _LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_


[50] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In
_Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 3202–3212, 2015.


[51] Adam Santoro, David Raposo, David G Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia,
and Timothy Lillicrap. A simple neural network module for relational reasoning. In _Advances in neural_
_information processing systems_, pages 4967–4976, 2017.


[52] Baoguang Shi, Xiang Bai, and Cong Yao. An end-to-end trainable neural network for image-based
sequence recognition and its application to scene text recognition. _IEEE transactions on pattern_
_analysis and machine intelligence_, 39(11):2298–2304, 2017.


[53] Richard Socher, Andrej Karpathy, Quoc V Le, Christopher D Manning, and Andrew Y Ng. Grounded
compositional semantics for finding and describing images with sentences. _Transactions of the_
_Association for Computational Linguistics_, 2:207–218, 2014.


[54] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. Videobert: A joint
model for video and language representation learning. _arXiv preprint arXiv:1904.01766_, 2019.


[55] Atousa Torabi, Niket Tandon, and Leonid Sigal. Learning language-visual embedding for movie
understanding with natural-language. _arXiv preprint arXiv:1609.08124_, 2016.


[56] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer
look at spatiotemporal convolutions for action recognition. In _Proceedings of the IEEE conference_
_on Computer Vision and Pattern Recognition_, pages 6450–6459, 2018.


[57] Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, and Kate
Saenko. Translating videos to natural language using deep recurrent neural networks. _arXiv preprint_
_arXiv:1412.4729_, 2014.


[58] Subhashini Venugopalan, Marcus Rohrbach, Jeffrey Donahue, Raymond Mooney, Trevor Darrell, and
Kate Saenko. Sequence to sequence-video to text. In _Proceedings of the IEEE international conference_
_on computer vision_, pages 4534–4542, 2015.


[59] Less Wright. Project title. [https://github.com/lessw2020/](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
[Ranger-Deep-Learning-Optimizer, 2019.](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)


[60] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual
transformations for deep neural networks. In _Proceedings of the IEEE conference on computer vision_
_and pattern recognition_, pages 1492–1500, 2017.


[61] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video
and language. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_,
pages 5288–5296, 2016.


[62] Ran Xu, Caiming Xiong, Wei Chen, and Jason J Corso. Jointly modeling deep video and compositional
text to bridge vision and language in a unified framework. In _Twenty-Ninth AAAI Conference on_
_Artificial Intelligence_, 2015.


[63] Natsuo Yamamoto, Jun Ogata, and Yasuo Ariki. Topic segmentation and retrieval system for
lecture videos based on spontaneous speech recognition. In _Eighth European Conference on Speech_
_Communication and Technology_, 2003.


[64] Youngjae Yu, Hyungjin Ko, Jongwook Choi, and Gunhee Kim. Video captioning and retrieval models
with semantic attention. _arXiv preprint arXiv:1610.02947_, 6(7), 2016.


_LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_ **15**


[65] Youngjae Yu, Jongseok Kim, and Gunhee Kim. A joint sequence fusion model for video question
answering and retrieval. In _Proceedings of the European Conference on Computer Vision (ECCV)_,
pages 471–487, 2018.


[66] Bowen Zhang, Hexiang Hu, and Fei Sha. Cross-modal and hierarchical modeling of video and text.
In _Proceedings of the European Conference on Computer Vision (ECCV)_, pages 374–390, 2018.


[67] Michael Zhang, James Lucas, Jimmy Ba, and Geoffrey E Hinton. Lookahead optimizer: k steps
forward, 1 step back. In _Advances in Neural Information Processing Systems_, pages 9593–9604, 2019.


[68] Yujie Zhong, Relja Arandjelovi´c, and Andrew Zisserman. Ghostvlad for set-based face recognition.
In _Asian Conference on Computer Vision_, pages 35–50. Springer, 2018.


[69] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million
image database for scene recognition. _IEEE Transactions on Pattern Analysis and Machine Intelligence_,
2017.


[70] Dimitri Zhukov, Jean-Baptiste Alayrac, Ramazan Gokberk Cinbis, David Fouhey, Ivan Laptev,
and Josef Sivic. Cross-task weakly supervised learning from instructional videos. _arXiv preprint_
_arXiv:1903.08225_, 2019.

### **A Supplementary Material**


**A.1** **Paper update, result corrections and summary of differences**


Following the release of the initial version of this paper (which can be viewed for reference at
[https://arxiv.org/abs/1907.13487v1](https://arxiv.org/abs/1907.13487v1) ), a bug was discovered in our open-source
software implementation which resulted in: (i) an overestimate of model performance; (ii) inaccurate conclusions about the relative importance of different experts on retrieval performance.
This correction to the paper contains repeats of each of the experiments reported in the initial
paper, with the following changes: (1) the removal of the bug which affected previous results; (2) a
systematic approach to hyperparameter selection (discussed in more detail below); (3) the inclusion
of additional “expert” pretrained features (described in Sec. A.5) to assess the influence of feature
strength within a modality. In addition to results, the written analysis has also been updated to
reflect the corresponding changes in results. The authors would like to express their gratitude to
Valentin Gabeur who identified the bug in the software implementation and enabled this correction.
**Bug details** : The bug caused information about feature availability in the ground truth target
video to become available to the query encoder during both training and testing when computing
embedding distances. The leak occurred through incorrect weighting of the embedding distances
due to: (1) a leaking broadcasting operation in an existing open-source library [ 39 ] that was
imported into our codebase; (2) incorrect NaN handling (introduced in our codebase), producing
the same effect. The bug has now been patched in each of the open-source codebases that were
known to have used this implementation.


**A.2** **Detailed Description of Datasets**


**MSR-VTT [61]:** This large-scale dataset comprises approximately 200K unique video-caption
pairs (10K YouTube video clips, each accompanied by 20 different captions). The dataset is
particularly useful because it contains a good degree of video diversity, but we noted a reasonably high degree of label noise (there are a number of duplicate annotations in the provided


**16** _LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_


captions). The dataset allocates 6513, 497 and 2990 videos for training, validation and testing, respectively. To enable a comparison with as many methods as possible, we also report
results across other train/test splits used in prior work [ 39, 65 ]. In particular, when comparing
with [ 39 ] (on splits which do not provide a validation set), we follow their evaluation protocol, measuring performance after training has occurred for a fixed number of epochs (100 in
total).
**MSVD [8]:** The MSVD dataset contains 80K English descriptions for 1,970 videos sourced from
YouTube with a large number of captions per video (around 40 sentences each). We use the standard
split of 1,200, 100, and 670 videos for training, validation, and testing [ 58, 62 ] [2] . Differently from
the other datasets, the MSVD videos do not have audio streams.
**LSMDC [50]:** This dataset contains 118,081 short video clips extracted from 202 movies. Each
video has a caption, either extracted from the movie script or from transcribed DVS (descriptive
video services) for the visually impaired. The validation set contains 7408 clips and evaluation
is performed on a test set of 1000 videos from movies disjoint from the training and val sets, as
outlined by the Large Scale Movie Description Challenge (LSMDC). [3]

**ActivityNet-captions [32]:** ActivityNet Captions consists of 20K videos from YouTube, coupled
with approximately 100K descriptive sentences. We follow the paragraph-video retrieval protocols
described in [ 66 ] training up to 200 epochs and reporting performance on val1 (this train/test
split allocates 10,009 videos for training and 4,917 videos for testing).
**DiDeMo [1]:** DiDeMo contains 10,464 unedited, personal videos in diverse visual settings with
roughly 3-5 pairs of descriptions and distinct moments per video. The videos are collected in an
open-world setting and include diverse content such as pets, concerts, and sports games. The total
number of sentences is 40,543. While the moments are localised with time-stamp annotations, we
do not use time stamps in this work.


**A.3** **Optimisation details and hyperparameter selection**


For each dataset, a grid search was first performed (using the Lookahead solver [ 59, 67 ]) over
batch sizes (16, 32, 64, 128, 256), learning rates (0.1, 0.01) and weight decay (1E-3, 5E-5) for each
dataset using a single expert to determine appropriate optimisation parameters. Next, an experiment
on MSR-VTT compared several choices for the dimensionality of the projection operation applied
to the features (described in Sec. 3.1) (choosing among 512, 768 and 1024 dimensions), which
suggested that 768 was most effective. This was then fixed for all remaining experiments (this
represents a difference from the original paper, in which 512 was used). Further ablations (provided
below) indicate that performance is not sensitive to this hyperparameter. Next, Asynchronous
Hyperband [ 33 ] was used to select all remaining hyperparameters on MSR-VTT by partially evaluating 1k configurations on the validation sets for each dataset. These hyperparameters consisted of:
the number of VLAD clusters and ghost clusters [68] used for different experts, the zero-padding
length applied to variable-length experts, the margin hyperparameter _m_ in Eq. 3, the Collaborative
Gating architecture (whether to use batch normalization [ 26 ], the number of layers used to form
the MLP, and the choice of activation function). The architecture choices were then fixed for all
datasets. Note that to ensure a fair comparison on MSR-VTT with the MoEE method of [ 39 ]
in Tab. 6, MoEE was also provided with a budget of 1k sampled configurations. To determine
zero-padding, margin and VLAD clusters for DiDeMo, MSVD and LSMDC further Asynchronous


2 Note: referred to by [42] as the JMET-JMDV split
3 https://sites.google.com/site/describingmovies/lsmdc-2017


_LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_ **17**


Hyperband searches were conducted, each with a budget of 500 sampled configurations. Since,
differently from the other datasets with available validation and test sets, the validation set itself
is used to assess performance on ActivityNet, hyperparameters were copied from the DiDeMo
configuration. The configurations, experts, pretrained models and logs for each of the experiments
reported in this paper are made available as part of the updated open-source implementation at
www.robots.ox.ac.uk/~vgg/research/collaborative-experts/.


**A.4** **Ablation Studies - Full Tables**

|Text =⇒Video|Video =⇒Text|
|---|---|
|Experts<br>R@1<br>R@5<br>R@10<br>MdR<br>MnR|R@1<br>R@5<br>R@10<br>MdR<br>MnR|
|Scene<br>4_._0_±_0_._1 14_._1_±_0_._1 22_._4_±_0_._3 50_._0_±_1_._0 201_._3_±_1_._6<br>Scene+Speech<br>4_._6_±_0_._1 15_._5_±_0_._2 24_._4_±_0_._2 44_._7_±_1_._2 183_._6_±_1_._7<br>Scene+Audio<br>5_._6_±_0_._0 18_._7_±_0_._1 28_._2_±_0_._1 33_._7_±_0_._6 140_._8_±_0_._3<br>Scene+Action(KN) 5_._3_±_0_._3 17_._6_±_0_._8 27_._1_±_0_._9 36_._0_±_1_._7 158_._7_±_1_._6<br>Scene+Obj(IN)<br>5_._0_±_0_._2 16_._6_±_0_._7 25_._5_±_1_._0 40_._7_±_2_._1 173_._1_±_3_._3<br>Scene+Obj(IG)<br>7_._2_±_0_._1 22_._3_±_0_._3 33_._0_±_0_._2 25_._3_±_0_._6 125_._1_±_0_._1 <br>Scene+Action(IG)<br>6_._8_±_0_._1 21_._7_±_0_._1 32_._4_±_0_._1 25_._7_±_0_._6 122_._1_±_0_._3<br>Scene+OCR<br>4_._1_±_0_._1 14_._1_±_0_._1 22_._2_±_0_._2 50_._3_±_1_._2 203_._1_±_4_._4<br>Scene+Face<br>4_._1_±_0_._1 14_._2_±_0_._3 22_._4_±_0_._4 49_._7_±_0_._6 194_._2_±_5_._1|5_._6_±_0_._6 18_._2_±_0_._6 27_._7_±_0_._3 39_._0_±_0_._0 247_._0_±_10_._1<br>6_._0_±_0_._2 20_._4_±_0_._5 30_._3_±_1_._0 33_._0_±_2_._0 222_._6_±_9_._9<br>8_._2_±_0_._4 24_._8_±_0_._4 36_._0_±_0_._1 21_._7_±_0_._6 127_._9_±_5_._9<br>7_._3_±_0_._6 22_._3_±_1_._4 33_._4_±_1_._7 25_._2_±_2_._0 151_._7_±_11_._6<br>6_._9_±_0_._5 21_._2_±_0_._9 31_._1_±_1_._9 28_._7_±_3_._8 188_._3_±_4_._7<br>10_._1_±_0_._3 29_._7_±_0_._5 41_._9_±_0_._7 15_._2_±_0_._9<br>91_._3_±_2_._4<br>9_._4_±_0_._3 27_._8_±_0_._6 40_._1_±_1_._1 17_._2_±_1_._1<br>87_._8_±_4_._2<br>5_._4_±_0_._5 18_._6_±_1_._2 26_._6_±_1_._2 40_._0_±_1_._0 292_._6_±_9_._9<br>5_._6_±_1_._0 17_._9_±_0_._7 26_._7_±_0_._8 39_._1_±_2_._6 273_._5_±_6_._3|



Table 7: Ablation study of importance of each expert when combined with Scene features.

|Text =⇒Video|Video =⇒Text|
|---|---|
|Experts<br>R@1<br>R@5<br>R@10<br>MdR<br>MnR|R@1<br>R@5<br>R@10<br>MdR<br>MnR|
|Scene<br>4_._0_±_0_._1 14_._1_±_0_._1 22_._4_±_0_._3 50_._0_±_1_._0 201_._3_±_1_._6<br>Prev.+Speech<br>4_._6_±_0_._1 15_._5_±_0_._2 24_._4_±_0_._2 44_._7_±_1_._2 183_._6_±_1_._7<br>Prev.+Audio<br>5_._8_±_0_._1 19_._0_±_0_._3 28_._8_±_0_._2 32_._3_±_0_._6 136_._8_±_1_._2<br>Prev.+Action(KN) 6_._7_±_0_._2 21_._8_±_0_._4 32_._5_±_0_._5 25_._3_±_0_._6 115_._9_±_1_._0<br>Prev.+Obj(IN)<br>7_._5_±_0_._1 23_._4_±_0_._0 34_._1_±_0_._2 23_._7_±_0_._6 111_._9_±_0_._6 <br>Prev.+Obj(IG)<br>9_._5_±_0_._2 27_._7_±_0_._1 39_._4_±_0_._1 18_._0_±_0_._0 92_._6_±_0_._4<br>Prev.+Action(IG)<br>9_._9_±_0_._1 28_._6_±_0_._3 40_._7_±_0_._1 17_._0_±_0_._0 86_._4_±_0_._4<br>Prev.+ OCR<br>10_._0_±_0_._1 28_._8_±_0_._2 40_._9_±_0_._2 16_._7_±_0_._6 87_._3_±_0_._8<br>Prev.+ Face<br>10_._0_±_0_._1 29_._0_±_0_._3 41_._2_±_0_._2 16_._0_±_0_._0 86_._8_±_0_._3|5_._6_±_0_._6 18_._2_±_0_._6 27_._7_±_0_._3 39_._0_±_0_._0 247_._0_±_10_._1<br>6_._0_±_0_._2 20_._4_±_0_._5 30_._3_±_1_._0 33_._0_±_2_._0 222_._6_±_9_._9<br>8_._6_±_0_._2 26_._1_±_0_._6 37_._8_±_0_._8 19_._8_±_0_._8 117_._7_±_2_._9<br>9_._9_±_0_._4 28_._6_±_0_._7 41_._7_±_0_._8 15_._7_±_0_._6<br>77_._9_±_5_._2<br>11_._2_±_0_._3 32_._1_±_0_._8 45_._4_±_0_._6 13_._7_±_0_._6<br>68_._0_±_1_._4<br>14_._7_±_0_._6 38_._9_±_0_._8 53_._1_±_1_._0 9_._3_±_0_._6<br>45_._6_±_2_._1<br>15_._5_±_0_._6 40_._1_±_1_._2 54_._4_±_1_._3 8_._7_±_0_._6<br>39_._4_±_0_._9<br>15_._2_±_0_._1 41_._1_±_0_._6 54_._6_±_0_._7 8_._5_±_0_._5<br>38_._5_±_0_._6<br>15_._6_±_0_._3 40_._9_±_1_._4 55_._2_±_1_._0 8_._3_±_0_._6<br>38_._1_±_1_._8|



Table 8: Ablation study of the importance experts on the MSR-VTT dataset.

|Col1|Text =⇒Video|Video =⇒Text|
|---|---|---|
|Expert<br>Num. Caps|R@1<br>R@5<br>R@10<br>MdR<br>MnR|R@1<br>R@5<br>R@10<br>MdR<br>MnR|
|Obj(IN)<br>1<br>Obj(IN)<br>20<br>All<br>1<br>All<br>20|2_._6_±_0_._1<br>9_._3_±_0_._4 15_._0_±_0_._7 101_._3_±_15_._5 321_._1_±_35_._1<br>4_._9_±_0_._1 16_._5_±_0_._2 25_._3_±_0_._4<br>40_._7_±_1_._2<br>169_._1_±_1_._4<br>4_._8_±_0_._2 16_._2_±_0_._5 25_._0_±_0_._7<br>43_._3_±_4_._0<br>183_._1_±_19_._6<br>10_._0_±_0_._1 29_._0_±_0_._3 41_._2_±_0_._2<br>16_._0_±_0_._0<br>86_._8_±_0_._3|3_._7_±_0_._3 13_._5_±_0_._6 20_._8_±_0_._4 60_._0_±_2_._0 304_._9_±_15_._8<br>6_._9_±_0_._6 21_._0_±_0_._3 31_._3_±_0_._3 30_._0_±_1_._7 201_._6_±_9_._5<br> 8_._4_±_0_._5 25_._6_±_0_._7 37_._1_±_0_._2 20_._3_±_0_._6<br>87_._2_±_6_._7<br>15_._6_±_0_._3 40_._9_±_1_._4 55_._2_±_1_._0 8_._3_±_0_._6<br>38_._1_±_1_._8|



Table 9: Ablation study of the number of captions in training on MSR-VTT


**A.5** **Implementation Details**


**Object** frame-level embeddings of the visual data are generated with two models, _Obj(IN)_ and
_Obj(IG)_ . _Obj(IN)_ is an SENet-154 model [ 24 ] (pretrained on ImageNet for the task of image classification) from frames extracted at 25 fps, where each frame is resized to 224 _×_ 224 pixels. _Obj(IG)_
is a ResNext-101 [ 60 ] pretrained on Instagram hashtags [ 37 ], using the same frame preparation


**18** _LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_

|Text =⇒Video|Video =⇒Text|Col3|
|---|---|---|
|Dimension<br>R@1<br>R@5<br>R@10<br>MdR<br>MnR|R@1<br>R@5<br>R@10<br>MdR<br>MnR|Params.|
|384<br>9_._4_±_0_._2 27_._8_±_0_._4 39_._8_±_0_._4 17_._7_±_0_._6 88_._8_±_0_._5 <br>512<br>9_._8_±_0_._3 28_._6_±_0_._4 40_._6_±_0_._4 17_._0_±_0_._0 88_._0_±_0_._7 <br>640<br>10_._1_±_0_._1 28_._8_±_0_._1 40_._9_±_0_._2 16_._7_±_0_._6 87_._6_±_0_._2 <br>768<br>10_._0_±_0_._1 29_._0_±_0_._3 41_._2_±_0_._2 16_._0_±_0_._0 86_._8_±_0_._3 <br>1024<br>9_._9_±_0_._1 28_._6_±_0_._3 40_._7_±_0_._4 17_._0_±_0_._0 87_._6_±_1_._1|14_._0_±_0_._5 38_._7_±_0_._5 52_._7_±_1_._4 9_._3_±_0_._6 41_._8_±_1_._0<br>14_._8_±_0_._4 40_._4_±_0_._6 53_._9_±_0_._4 8_._8_±_0_._3 38_._8_±_1_._5 <br>15_._6_±_0_._6 41_._3_±_0_._7 55_._0_±_0_._5 8_._3_±_0_._6 37_._3_±_1_._8 <br>15_._6_±_0_._3 40_._9_±_1_._4 55_._2_±_1_._0 8_._3_±_0_._6 38_._1_±_1_._8 <br>14_._7_±_0_._4 40_._7_±_0_._8 54_._4_±_0_._3 8_._5_±_0_._5 39_._1_±_1_._7|88.62M<br>119.51M<br>151.12M<br>183.45M<br>250.27M|



Table 10: Ablation study of the importance of model capacity by varying the shared embedding
dimension used by CE on MSR-VTT.


as _Obj(IN)_ . Features are collected from the final global average pooling layer of both models, and
have a dimensionality of 2048.
**Action** embeddings are similarly generated from two models, _Action(KN)_ and _Action(IG)_ . _Ac-_
_tion(KN)_ is an I3D inception model that computes features following the procedure described by

[ 6 ]. Frames extracted at 25fps and processed with a window length of 64 frames and a stride of
25 frames. Each frame is first resized to a height of 256 pixels (preserving aspect ratio), before a
224 _×_ 224 centre crop is passed to the model. Each temporal window produces a (1024x7)-matrix
of features. _Action(IG)_ is a 34-layer R(2+1)D model [ 56 ] trained on IG-65m [ 19 ] which processes
clips of 8 consecutive 112 _×_ 112 pixel frames, extracted at 30 fps (we use the implementation
provided by [9]).
**Face** embeddings are extracted in two stages: (1) Each frame (also extracted at 25 fps) is resized
to 300 _×_ 300 pixels and passed through an SSD face detector [ 4, 35 ] to extract bounding boxes;
(2) The image region of each box is resized such that the minimum dimension is 224 pixels and
a centre crop is passed through a ResNet50 [ 22 ] that has been trained for task of face classification on the VGGFace2 dataset [ 5 ], producing a 512-dimensional embedding for each detected
face.

**Audio** embeddings are obtained with a VGGish model, trained for audio classification on the
YouTube-8m dataset [ 23 ]. To produce the input for this model, the audio stream of each video
is re-sampled to a 16kHz mono signal, converted to an STFT with a window size of 25ms
and a hop of 10ms with a Hann window, then mapped to a 64 bin log mel-spectrogram. Finally, the features are parsed into non-overlapping 0.96s collections of frames (each collection
comprises 96 frames, each of 10ms duration), which is mapped to a 128-dimensional feature

vector.

**Scene** embeddings of 2208 dimensions are extracted from 224 _×_ 224 pixel centre crops of frames
extracted at 1fps using a DenseNet-161 [ 25 ] model pretrained on Places365 [ 69 ]. **Speech to Text**
The audio stream of each video is re-sampled to a 16kHz mono signal. We then obtained transcripts
of the spoken speech for MSR-VTT, MSVD and ActivityNet using the Google Cloud Speech
to Text API [4] from the resampled signal. The language for the API is specified as English. For
reference, of the 10,000 videos contained in MSR-VTT, 8,811 are accompanied by audio streams.
Of these, we detected speech in 5,626 videos.
**Optical Character Recognition** is extracted in two stages: (1) Each frame is resized to 800 _×_ 400
pixels) and passed through Pixel Link [ 10 ] text detection model to extract bounding boxes for texts;
(2) The image region of each box is resized to 32 _×_ 256 and then pass through a model [ 36, 52 ]
that has been trained for text of scene text recognition on the Synth90K dataset[ 27 ], producing
a character sequence for each detect box. They are then encoded via a pretrained word2vec
embedding model [41].


4 https://cloud.google.com/speech-to-text/


_LIU, ALBANIE, NAGRANI, ZISSERMAN: COLLABORATIVE EXPERTS_ **19**


**Text** We encode each word using the Google News [5] trained word2vec word embeddings [ 41 ].
All the word embeddings are then pass through a pretrained OpenAI-GPT model to extract the
context-specific word embeddings (i.e., not only learned based on word concurrency but also
the sequential context). Finally, all the word embeddings in each sentence are aggregated using
NetVLAD.


5 GoogleNews-vectors-negative300.bin.gz found at: https://code.google.com/archive/p/word2vec/


