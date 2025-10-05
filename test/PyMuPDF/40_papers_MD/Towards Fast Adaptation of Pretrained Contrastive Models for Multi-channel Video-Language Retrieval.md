## **Towards Fast Adaptation of Pretrained Contrastive Models for Multi-channel** **Video-Language Retrieval**

**Xudong Lin** [1] **Simran Tiwari** [1] **, Shiyuan Huang** [1] **, Manling Li** [2]

**Mike Zheng Shou** [3] **, Heng Ji** [2] **, Shih-Fu Chang** [1]

1 Columbia University 2 UIUC 3 National University of Singapore
xudong.lin@columbia.edu



**Abstract**


_Multi-channel video-language retrieval require models_
_to understand information from different channels (e.g._
_video_ + _question, video_ + _speech) to correctly link a video_
_with a textual response or query. Fortunately, contrastive_
_multimodal models are shown to be highly effective at align-_
_ing entities in images/videos and text, e.g., CLIP [26]; text_
_contrastive models are extensively studied recently for their_
_strong ability of producing discriminative sentence embed-_
_dings, e.g., SimCSE [8]._ _However, there is not a clear_
_way to quickly adapt these two lines to multi-channel video-_
_language retrieval with limited data and resources. In this_
_paper, we identify a principled model design space with two_
_axes: how to represent videos and how to fuse video and text_
_information. Based on categorization of recent methods, we_
_investigate the options of representing videos using contin-_
_uous feature vectors or discrete text tokens; for the fusion_
_method, we explore the use of a multimodal transformer or_
_a pretrained contrastive text model. We extensively evalu-_
_ate the four combinations on five video-language datasets._
_We surprisingly find that discrete text tokens coupled with_
_a pretrained contrastive text model yields the best perfor-_
_mance, which can even outperform state-of-the-art on the_
_iVQA and How2QA datasets without additional training on_
_millions of video-text data. Further analysis shows that this_
_is because representing videos as text tokens captures the_
_key visual information and text tokens are naturally aligned_
_with text models that are strong retrievers after the con-_
_trastive pretraining process. All the empirical analysis es-_
_tablishes a solid foundation for future research on afford-_
_able and upgradable multimodal intelligence._


**1. Introduction**


From retrieving a trending video on TikTok with natural
language descriptions to asking a bot to solve your technical problem with the question and a descriptive video,



AI agents handling multi-channel video-language retrievalstyle tasks have been increasingly demanded in this postsocial-media era. These tasks require the agent to fuse information from multiple channels, i.e., video and text to
retrieve a text response or return a multi-channel sample
for a text query. To power such agents, a popular approach [13, 15, 20, 40, 46] consists of two rounds of pretraining: (1) The 1st round is to obtain unimodal pretrained models, such as visual-only encoders [5, 9, 22, 26]
(e.g., S3D,CLIP) and text-only encoders [6,18,28,30] (e.g.,
BERT) (2) The 2nd round aims at pretraining on visualtext dataset - specifically, researchers leverage techniques
like masked token modeling [13, 46] or contrastive learning [15, 20, 39, 40] to align and fuse unimodal features from
model pretrained in the 1st round.
Such methods achieve good performance on multichannel retrieval-style tasks but they suffer from two major limitations: 1) huge amounts of data and computational
resources are required for the second-round “pretraining”,
which significantly limits the research exploration without
such resources; 2) the domain of video data used in the second round “pretraining” has to be strongly correlated with
downstream tasks [13], which may restrict such methods
from being generally applicable.
To alleviate such limitations, we study a novel problem:
fast adaptation of pretrained contrastive models on multichannel video-language retrieval under limited resources.
Specifically, we propose to adapt both contrastive multimodal models [21,26] and contrastive text models [8,28] to
enjoy their strong encoding ability and discriminative embedding space. There has been tremendous progress recently on large-scale contrastive multimodal models [21,
22, 26]. Through pretraining on millions of images/videos,
these models are highly effective at _encoding visual inputs_
_and linking entities across modalities_ . Meanwhile, contrastive text models [8, 28] have been also densely studied to obtain _discriminative sentence embeddings_ . These
models are shown to perform well on challenging text retrieval tasks such as semantic search [24], which requires


the model to _understand both language and real-world_
_knowledge_ [8, 28]. Such an ability allows the model to
retrieve a text response or encode a text query in multichannel video-language retrieval-style tasks. Thereby, once
the video information is effectively incorporated, it eliminates the necessity of second-round “pretraining” on large
scale multimodal datasets, enabling fast adaptation to any
video-language retrieval-style tasks.
We first conduct a systematic analysis on potential model
designs, as shown in Figure 1. We identify the model design space with two design principles: how to represent the
video information and how to fuse this video information

with questions or other text such as speech. To represent
video information, we could either adopt **Continuous Fea-**
**tures** which is commonly used in existing work [23, 40],
or project videos into unified **Text Tokens** [16] from various modalities. To fuse information from multiple channels,
i.e., video and question/speech, there are two potential options, namely, a **Multimodal Transformer** [40] or a **Text**
**Transformer** [23]. Hence, there are four combinations derived from this model design space, namely, **Continuous**
**Features + Multimodal Transformer** [40], **Continuous**
**Features + Text Transformer**, **Text Tokens + Multimodal**
**Transformer**, **Text Tokens + Text Transformer** .
Our exploration of this model design space results into
a simple yet effective approach that allows fast adaptation
of pretrained contrastive models, which first leverages contrastive multimodal models to retrieve a sequence of **Text**
**Tokens** for the visual input and then feeds these tokens together with other text to contrastive **Text Transformer** for
answer retrieval. Its fast adaptation ability not only comes
from the ability of linking entities across modalities of the
contrastive multimodal model, but it also enjoys the natural
alignment with contrastive text models to produce discriminative embeddings. To the best of our knowledge, this is the
first proposal to adapt pretrained contrastive models in this
manner, although each individual design choice may have
been adopted previously. We further conduct in-depth analysis to understand 1) the trade-off between data efficiency
and accuracy, 2) the impact of pretrained contrastive text
model, and 3) the possible limitation of this framework.
The contribution could be summarized three-fold:


  - We identified the principled model design space for
fast adaption of pretrained contrastive multimodal
models and pretrained contrastive text models.


  - We conducted extensive experiments on five videolanguage datasets, observed a consistent trend across
these four variants, and even obtained state-of-the-art
performance (e.g., **6.5%** improvement on How2QA)
with the proposed Text Tokens + Text Transformer
variant **without using millions of extra multimodal**
**data samples**, which is essential to democratize the



community of video+language.


  - The proposed Text Tokens + Text Transformer variant scales significantly better than the other variants,
w.r.t. the quality of pretrained text models. The
[code will be released at https://github.com/](https://github.com/XudongLinthu/upgradable-multimodal-intelligence)
[XudongLinthu/upgradable- multimodal-](https://github.com/XudongLinthu/upgradable-multimodal-intelligence)
[intelligence to facilitate future research.](https://github.com/XudongLinthu/upgradable-multimodal-intelligence)


**2. Related Work**


**Pretrained Contrastive Models.** In this paper, we mainly
consider two types of contrastive models: 1). contrastive
multimodal models [15, 17, 21, 26, 35], which typically consist of a visual encoder and a text encoder, and learn to map
visual and text embeddings into a common space. They
sample positive/negative pairs from aligned/unaligned image/video and text, and train the visual and text encoders
with a contrastive objective in a self-supervised manner.
With access to large-scale multimodal data (e.g., 400 million web image-text pairs [26]), they are shown superior
in linking entities across modalities; 2) Unlike general language models [6, 27], contrastive text models [8, 28], which
learn discriminative sentence embeddings from large-scale
text data (e.g., 2 billion sentence pairs [29]). They construct
positive and negative sentence pairs and train the text encoder in a either supervised or unsupervised manner.
**Multi-channel Vision-Language Learning.** Interests have
been raised in multi-channel VL applications, where the
model is asked to understand from both visual and language
inputs to output a response (e.g., a phrase). Several works
seek to get the response in a generative way [12,16,47], i.e.,
using a generative model to predict a sequence. On the other
hand, retrieval-style [1, 2, 19, 43] accomplishes the tasks in
a discriminative way, where discriminative embeddings are
produced to select the best answer from an answer candidate pool. In the paper, we focus on retrieval-style tasks,
e.g., video question answering [13, 38, 40, 45] and multichannel text-to-video retrieval [37, 48].
**Adapting Pretrained Models to VL Applications.** Existing works [23, 33, 40, 41] explored different ways to adapt
pretrained models to various vision-language tasks. For example, Shen _et al._ [31] explored directly taking visual features from pretrained CLIP [26] image encoders and feed
them to existing vision-language models [33] and then retrain the whole model using the new CLIP visual features.
This empirical analysis shows that visual features learned
by contrastive multimodal models are better for visionlanguage tasks. Just-ask [40] is a recent method for openended video question answering. The authors found the
multimodal transformer gets significantly improved after
a second-round pretraining on 69M video-question-answer
triplets. All these methods require second-round pretraining on another large task-specific dataset, which requires


𝑒𝑒 𝑣𝑣,𝑡𝑡 𝑒𝑒 𝑣𝑣,𝑡𝑡 𝑒𝑒 𝑣𝑣,𝑡𝑡 𝑒𝑒 𝑣𝑣,𝑡𝑡










|ℋ|Col2|Col3|
|---|---|---|
||||
|𝒢|ℱ𝑉|ℱ𝑉|











|𝒢𝒢|Col2|Col3|
|---|---|---|
||||
||**Tokenization**|**Tokenization**|
||**Tokenization**||
||ℱ𝑉|ℱ𝑉|


d






|𝒢𝒢|Col2|Col3|
|---|---|---|
||||
||𝒫|𝒫|
||𝒫||
||ℱ𝑉|ℱ𝑉|








|ℋ|Col2|Col3|
|---|---|---|
||||
|𝒢|𝒢|𝒢|
|𝒢|𝒢||
|𝒢|**Tokenization**|**Tokenization**|
|𝒢|**Tokenization**||
|𝒢|ℱ𝑉|ℱ𝑉|





a b c











Figure 1. a. **Continuous Features + Multimodal Transformer** . This variant uses the pretrained video encoder to represent input video
segment as continuous feature vectors. A randomly initialized multimodal transformer is used to fuse information of the video and textual
input. b. **Continuous Features + Text Transformer** . Instead of a multimodal transformer, we use the pretained contrastive text model to
fuse projected video features and textual input. c. **Text Tokens + Multimodal Transformer** . It uses the pretrained video-text encoders
to retrieve top-k text tokens from a predefined vocabulary to describe the video content. Then we feed the text tokens separately to the
pretrained text model, to obtain contextualized embeddings for a multimodal transformer to fuse with contextualized text embeddings of
other textual input. d. **Text Tokens + Text Transformer** . We further remove the multimodal transformer and directly use the pretrained
text model to fuse video and question.



additional engineering process and high demand of computational resources; on the other hand, we eliminate this
process by efficiently leveraging the pretrained contrastive
text models.


**3. Technical Approach**


In this section, we first include some preliminaries, and
then introduce the four models in the model design space in
detail. In the end, we describe the training technique and
the implementation details.


**3.1. Preliminaries**


**Multi-channel Video-language Retrieval-style Tasks.**
These tasks require the model to fuse information from multiple channels, i.e., video and text to retrieve a text response
or return a multi-channel sample for a text query. Specifically, we mainly consider open-ended video question answering and multi-channel text-video retrieval. In openended video question answering, given a video _v_ and question _t_ as input, the model is required to retrieve the correct
answer _a_ _k_ from a large answer corpus A = _{a_ 1 _, .., a_ _|_ A _|_ _}_ . In



multi-channel text-video retrieval, given a text query _a_, the
model is required to retrieve the most relevant video _v_ _i_ with
associated speech text _t_ _i_, which are from a corpus of multichannel videos V = _{v_ 1 _, t_ 1 _, .., v_ _|_ V _|_ _, t_ _|_ V _|_ _}_ . For simplicity,
in the model descriptions, we will adopt open-ended video
question answering for illustration.


**Pretrained Contrastive Multimodal Models.** We mainly
leverage pretrained video-text contrastive models. It consists of a video encoder _F_ _V_ : R _[H][×][W][ ×]_ [3] _[×][F]_ _−→_ R _[D]_ and
a text encoder _F_ _T_ : W _[L]_ _−→_ R _[D]_, where _H, W, F_ are the
height, width and number of frames of the video, _L_ is the
length of the sentence, _D_ is the dimension of the common
embedding space and W is the set of all the words. Note that
in all of the models, the pretrained contrastive multimodal
models are **frozen** for fast adaptation.


**Pretrained Contrastive Text Models.** The idea is to train
a text encoder _G_ : W _[L]_ _−→_ R _[D]_ to push the output embeddings of two relevant sentences to be similar and the irrelevant to be dissimilar. The pretrained contrastive text models
will be **updated** in all of the models. Since all the actual instances of text models are transformer-based, text models


and text transformers are used interchangeably.
**Multimodal Transformer and Projector.** It is a shallow
transformer model [34] _H_ : R _[L][×][D]_ _−→_ R _[L][×][D]_ . We direct
the audience for details of the transformer model to [34].
Inspired by prefix tuning [23], we use the same transformer
architecture for the projector _P_ : R _[L][×][D]_ _−→_ R _[L][×][D]_, which
is used to project video feature vectors to the input space of
the pretrained text model.


**3.2. Model Variants in the Design Space**


For all the four variants, there is a separate encoder _G_ _A_
to encode answers, which is initialized from pretrained text
models. In the following, we mainly focus on the difference
of these four variants, i.e., the video representation and the
multimodal fusion design.


**a.** **Continuous Features + Multimodal Transformer**


As shown in Figure 1, in this variant, we first directly use
the pretrained video encoder _F_ _V_ to represent input video
segment _v_ as continuous feature vectors. The pretrained
contrastive text model _G_ is used to extract contextualized
text embeddings from the question _t_ . A randomly initialized multimodal transformer _H_ is used to fuse information

of the video and the question. Overall the model could be
expressed as follows:


_e__ { _v_, _t_ } = _\_ _m_ a _t_ hc _al {H} (\mathcal {G} (t), \mathcal {F}_V(v) ),_ (1)


where _G_ ( _t_ ) are actually concatenated with _F_ _V_ ( _v_ ) along the
length axis before they are fed into the multimodal transformer.


**b.** **Continuous Features + Text Transformer**


Inspired by prefix tuning work [23], in this variant, we
leverage a transformer projector _P_ to project the continuous
features from the video encoder _F_ _V_ into the input space of
the text transformer. Then the text transformer will fuse the

information from both the video _v_ and the question _t_,


_e__ { _v_, _t}_ = _\_ m _a_ thc _al {G} ( t, \mathcal {P}(\mathcal {F}_V(v)) ),_ (2)


where the question _t_ actually first passes the embedding
layer of _G_ and then is concatenated with projected video
features _P_ ( _F_ _V_ ( _v_ )).


**c.** **Text Tokens + Multimodal Transformer**


We propose the Text Token retrieval process as follows.
First, we construct the word vectors from a predefined word
vocabulary W; specifically, for each word _w_ _i_, we use the
text encoder to obtain its word vector _F_ _T_ ( _w_ _i_ ). For the input video _v_, we encode it with the pretrained video encoder
_F_ _V_ ( _v_ ). Then we compare the similarity between the video
and each of the words to retrieve _k_ most similar words,


_{_
_w_ _ _{ v,1 }_ _,.._, w_ v,k _=_ _\_ _a_ r _[g]_ _\_ m _a_ x __i^k \mathcal {F}_T(w_i)^\top \mathcal {F}_V(v)._ (3)
_}_



Then the retrieved words are fed into the pretrained text
model with the question in parallel to obtain the contextualized embeddings, which are finally concatenated to be fed
into the multimodal transformer,


_e__ { _v_, _t_ } = _\_ m _a_ _th_ c _a l { H_ _} (_ \m _athcal {G} (t), \mathcal {G} (w_{v,1},.., w_{v,k}) )._ (4)


Note that to maintain similar parameter usage, we share
the weights between the text model _G_ taking question as
input and the text model _G_ taking video words as input.


**d.** **Text Tokens + Text Transformer**


In this variant, we follow the same text token retrieval
process and then further simplify the model to use pretrained text model for modality fusion. Formally, it can be
described as,


_e__ { _v_, _t}_ _=_ \ _m ath c_ _al_ { _G} (t, w_{v,1},.., w_{v,k}),_ (5)


where the question _t_ and all the retrieved words are concatenated as a whole text sequence and then input to the
pretrained text model. The removal of the multimodal transformer enables this variant to directly benefit from the pretrained discriminative sentence embedding space of _G_ .


**3.3. Training**


During training, we leverage the NCE loss [40] to optimize the parameters _θ_ of _G_, _G_ _A_ _H_, and _P_ (if included in the
model). In the actual training process, we calculate loss in
a mini-batch manner. For simplicity, we omit average over
a batch in the following equations.
For open-ended video question answering, we optimize
the following objective



\ _i_ n _ \th e _t_ _[a]_ _-_ _[\]_ _[l]_ [o] _[g]_ [\] f
_m_ _[x]_



_-_ _e_{v,t} ^\top \mathcal {G}_A (a_l) \right )}{\sum _i \exp \left ( e_{v,t}^\top \mathcal {G}_A (a_i) \right ) },_ (6)

_a_ [c {] ~~\~~ _e_ _[x]_ _p \_ _[l]_ _[e]_ [f] _[t]_ [(]



~~r~~



where _l_ is the ground-truth answer index.
For multi-channel text-video retrieval, we adopt the symmetric version of the loss [26], Specifically,




[g] lig n _e_ _[d]_ _}_ _\_ _[m]_ _[i]_ [n] _[_]_ [\] t

i [n {a] _-_ _[\]_ _[1]_



\ _e_ [g]
_b_ i



~~h~~



_e_ [ta ] ~~&~~ _-_ _[\]_ _fra_ _[c]_ [{] _[1]_ _[}]_ [{] ~~2~~



(7)



} ( \ log _\_ _[f]_ _rac_ _[{]_ [\] _[e]_ _[x]_ [p]



_a_ p _\mathcal {G}_A (a_l) \right )}{\sum _i \exp \left ( e_{v,t}^\top \mathcal {G}_A (a_i) \right ) } \\ &+ \log \frac {\exp \left ( e_{v,t} ^\top \mathcal {G}_A (a_l) \right )}{\sum _i \exp \left ( e_{v_i,t_i}^\top \mathcal {G}_A (a_l) \right ) }), \end {aligned}_

_l_ [eft] _(_ _e_ ___ _{v_ _,_ _[t]_ _[}]_ _[^]_ _[\]_ [t] ~~o~~



~~\~~



where _e_ _v,t_ and _a_ _l_ are a pair of embeddings of multi-channel
video and text query, and _e_ _v_ _i_ _,t_ _i_ is the representation of the ith multi-channel video in the mini-batch. Note that although
_G_ _A_ is also initialized from the pretrained text model, it does
not share parameters with _G_ during training, following the
setting of [40].


Open-ended Acc(%) Multi-Choice Acc(%) Retrieval AveR (%)
Model
iVQA ActivityNet-QA How2QA YouCook II VATEX


Conti. + Multi. 22.4 36.9 79.2 41.9 69.4

Conti. + Text 23.2 37.3 80.4 46.2 72.7

Text + Multi. 23.4 37.1 79.4 40.4 67.5

**Text + Text** **31.6** **38.7** **82.9** **49.7** **74.8**


Table 1. Comparison of the four model variants on various multi-channel video-text retrieval-style tasks. Acc is short for accuracy. AveR
is short for average recall@ _{_ 1,5,10 _}_ .


**3.4. Implementation Details**


30



We use MPNet (all-mpnet-base-v2) [29, 32] as _G_ when
not specified. It is ranked first by Sentence Transformers [29] at the time of writing (November, 2022). We use the
pretrained models from [21] for _F_ _T_ and _F_ _V_ when not specified. When using CLIP [26], we follow FrozenBiLM [42].
The resulted video feature vector is at 1 vector per 1.5 seconds. For _H_ and _P_, we use the same shallow multimodal
transformer as in [40], which has 2 transformer blocks with
512/768 as the embedding dimension. We follow the training procedure and hyper-parameter settings for all the variants to conduct fair comparisons. Details are provided in
the supplementary material [1] .
We find that the vocabulary W can be effectively constructed by parsing all the query/answer sentences in the
downstream datasets with spaCy [11] to obtain a set of
unique verbs and nouns. We use _k_ = 15 words retrieved
from the vocabulary for each feature vector in the video.
We follow [40] to sub-sample video feature vectors when
the video is too long to fit the memory. We further apply a max pooling with kernel size as 5 to sub-sample the
retrieved words to avoid too many repetitions of words of
neighbouring segmenting.


**4. Experimental Results**


In this section, we will first introduce the datasets and
evaluation metric, then we organize the following subsections by answering a set of important questions of the four
model variants. Extra ablations are provided in the supplementary material for hyper-parameter selection.


**4.1. Dataset and Evaluation Metric**


We select 5 commonly used multi-channel retrieval-style
datasets: **iVQA [40]**, **How2QA [13]**, **ActivityNet-QA [45]**,
**YouCook II [49]**, and **VATEX [37]** as the main evaluation
datasets. For the first three datasets, accuracy is used as
the evaluation metric and for the other two, an average of
recall@ _{_ 1,5,10 _}_ is used. Details are provided in the supplementary material.
**MSRVTT-QA and MSVD-QA [38]** . These two datasets
are automatically generated from video-caption pairs. The


1 [https://arxiv.org/abs/2206.02082](https://arxiv.org/abs/2206.02082)



25


20


15


10





20 40 60 80 100

# of Training Videos (%)


Figure 2. Comparison of the four variants under few-shot setting
on the iVQA dataset.


data is too noisy compared to the aforementioned manually
annotated datasets but this actually could serve as a testbed
to see under what circumstances our variants won’t work

well. We follow [40] for experimental settings and evaluation metric on these two datasets.


**4.2. Which of the variants performs the best?**


As shown in Table 1, the **Text Tokens + Text Trans-**
**former** consistently performs better than the other three
variants when we directly tune these models without
second-round large-scale multimodal “pretraining”. We
suppose this is because of this variant can easily benefit
from both pretrained contrastive multimodal models and
contrastive text models without struggling to align the space
of them or training a multimodal transformer from scratch.
We also observe that on downstream task from similar do
mains with that of the pretrained contrastive multimodal
model [21], our proposed variant usually enjoys higher improvement, e.g., on iVQA and YouCook II.
To further understand the behavior of these four variants

with limited training data for downstream adaptation, we
explore the few-shot setting. We sample a subset of iVQA
and train the four variants with the same sub-sampled set
for the same number of iterations with that of the full-shot

setting. In Figure 2, we consistently observe a large margin
between **Text Tokens + Text Transformer** and the other


Model _F_ _V_ _, F_ _T_ Extra MM Samples ∆ GPU hours iVQA ActivityNet How2QA


MERLOT [46]       - 180M       -       - 41.4       SiaSamRea [44]       - 5.6M + 80K       -       - 39.8 84.1


VQA-T [41] S3D [21] 69M + 3M 350 + 30 35.2 **39.0** **85.3**
Conti. + Multi. S3D [21] 69M 400 35.4 38.9 84.4
Conti. + Multi. ( _+ ASR_ ) S3D [21] 69M 400 36.0 38.9 84.8
Text + Text (Ours) S3D [21] 0 0 31.6 38.7 82.9
Text + Text ( _+ ASR_, Ours) S3D [21] 0 0 **36.8** 38.8 84.6


FrozenBiLM [42] CLIP [26] 10M 160 39.7 43.2 81.5
FrozenBiLM [42] ( _+ ASR_ ) CLIP [26] 10M 160 39.6 **43.2** 86.7
Text + Text (Ours) CLIP [26] 0 0 36.9 41.4 92.4
Text + Text ( _+ ASR_, Ours) CLIP [26] 0 0 **40.2** 41.4 **93.2**


Table 2. Comparison with the state-of-the-art on iVQA, ActivityNet and How2VQA in terms of accuracy and efficiency. Extra MM
Samples indicate the number of video-text samples that are needed in the second-round pretraining. ∆ GPU hours refer to the additional
computation required for the second-round pretraining. _+ ASR_ indicates the use of ASR texts of the video as additional inputs. Note that
our variant typically requires 0.5 GPU hours for training. Methods in gray enjoy a more costly end-to-end training process.



variants. Continuous Features + Multimodal Transformer

generally performs the worst on the iVQA dataset and also
other datasets, which implicitly verifies our hypothesis that
it is crucial to directly leverage the aligned representation
space and the discriminative output space.


**4.3. Comparison with state-of-the-art on accuracy-**
**efficiency trade-off**


We aim to examine the trade-off between downstream

performance and the additional resource requirements,
through comparison between our best variant with state-ofthe-art models that leverage huge amounts of data and computational resources for pretraining the multimodal transformer [41, 42]. As shown in Table 2, with **three or-**
**ders of magnitude smaller amount GPU training time**,
our model can actually achieve comparable or even better results than the state-of-the-art. When using pretrained
_F_ _V_ _, F_ _T_ from [21], our model can even outperform the
state-of-the-art by 0.8% when with ASR as input.
When switching the _F_ _V_ _, F_ _T_ to CLIP [26], our **Text To-**
**kens + Text Transformer** even significantly outperforms
than models that enjoys a more costly end-to-end secondround “pretraining” process like MERLOT [46] or SiaSamRea [44]. Most encouragingly, our model also outperforms
the state-of-the-art FrozenBiLM on How2QA ( **+6.5%** ) and
iVQA ( **+0.5%** ), which uses 10 million samples for secondround “pretraining” and uses a much larger text model.
We also would like to highlight that our **Text Tokens +**
**Text Transformer** _easily benefits from a better pretrained_
_F_ _V_ _, F_ _T_ as we observe significant improvement for our
model, which is encouraging for future development of our
proposed model when better pretrained multimodal contrastive models are available. Comparisons on the other two
datasets are separately provided in the supplementary material as the compared methods in Table 2 are not state-of


the-art on them.


**4.4. How well does** **Text Tokens + Text Transformer**
**benefit from pretrained text models?**


To comprehensively understand the effect of pretrained
contrastive text models, we investigate two sources of models, as shown in Figure 3 and Figure 4. SBERT [29] is
a widely-used library with models trained on paired sentences. We measure the quality of pretrained text models with the average performance over 20 sentence embedding/semantic search tasks. SimCSE [8] explores unsupervised contrastive learning of transformer models and provides various models. We measure the quality of pretrained
text models with the average performance over 7 semantic
textual similarity (STS) tasks.
In Figure 3, we first study the four variants trained on
the same data with different model sizes. MiniLM [36]
(paraphrase-MiniLM [29]) with _{_ 3 _,_ 6 _,_ 12 _}_ layers is a suitable testbed as the model architecture is identical but the

number of layers changes. We obtain a clear positive correlation between the quality of text models and the accuracy on iVQA. We then study the same model trained with
different size of data: paraphrase-mpnet-base-v2, multiqa-mpnet-base-dot-v1, and all-mpnet-base-v2, which are
trained on 83 million, 215 million, and 2 billion sentence
pairs, respectively. Again our proposed variant scales well
with the performance of the text models.
The scalability w.r.t. the quality of pretrained text model
of the proposed **Text Tokens + Text Transformer** is surprisingly superior compared the other variants. As shown in
Figure 3, among all the four variants, the proposed **Text To-**
**kens + Text Transformer** has the most positive correlation
between the multimodal task performance on iVQA and
the quality of the pretrained text models. We also observe
that both models using pretrained text transformer for mul

32


30


28


26


24


22





20


18


52 54 56 58 60 62

Average Performance on sbert.net (%)


Figure 3. Results of varying the pretrained text model in the four variants on the iVQA dataset. The performance of all SBERT models are
obtained from [29].



29.2


29.0


28.8


28.6


28.4


28.2





76 77 78 79 80 81 82

Average Performance on STS tasks (%)


Figure 4. Results of varying the pretrained text model in the **Text**
**Tokens + Text Transformer** on the iVQA dataset. The semantic
textual similarity (STS) performance of SimCSE models is obtained from [8].


timodal fusion have a positive correlation and both models
using multimodal transformer for multimodal fusion don’t
benefit from the improvement of the pretrained text models.
From this comparison, we conclude that only the proposed
**Text Tokens + Text Transformer** can well enjoy the improvement of pretrained text models.
We also verify the necessity of using pretrained contrastive text models by directly tuning the MPNet from [32]
without contrastive pretraining. It is not shown in Figure 3
as its text performance is not evaluated by sbert.net. This
model only achieves 22 _._ 0%, which is significantly lower
than the three contrastively pretrained versions.
Then we compare both the unsupervised and supervised
BERT/RoBERTa models from SimCSE. As shown in Figure 4, as STS performance increases, the overall downstream performance on iVQA is improved, despite incoherence across different model architectures. As a summary,
even with various model architectures, training methods,
supervision sources and dataset sizes, we consistently ob


serve positive correlation, which is highly encouraging for
further upgradability of the proposed **Text Tokens + Text**
**Transformer** in the future.


**4.5. Model Interpretation**


To understand why our proposed **Text Tokens + Text**
**Transformer** performs well, we select iVQA to first check
a simple statistic: the proportion of test samples that have at
least one word overlapped between the answer and retrieved
text tokens of the video. We find that the proportion is actually 66 _._ 4%, which partially explains why we obtain a huge
improvement on the iVQA dataset.
We visualize one successful case and one failure case in

Figure 5. In Figure 5a, the model is not able to retrieve
the answer word “apron” for the video but based on the
rich kitchen-related context words, the text-transformer can
still answer the question correctly. This indicates that our
proposed model can accommodate imperfection of the tokenization process. In Figure 5b, although “soup” is retrieved
as a text token for the video in the first few segments of the
video, the crucial video segment required fine-grained temporal understanding of the video in the last segment and the
text tokens after max pooling is dominated by other foodrelated words. But overall, according the aforementioned
statistic, this Text Tokens + Text Transformer approach certainly enjoys the high explainability in its design, compared
with other using Continuous Features.


**4.6. Results on Automatically Generated Datasets**


We evaluate the proposed **Text Tokens + Text Trans-**
**former** on the automatically generated MSRVTT-QA and
MSVD-QA datasets. We find that Text Tokens + Text
Transformer (39.7%, 40.9%) performs similarly with the
Continuous Features + Multimodal Transformer baseline

(39.6%, 41.2%). To quantify the imperfection of question generation, we carefully convey a manual study on


Figure 5. Visualization of **Text Tokens + Text Transformer** on iVQA. “First” and “then” indicate the temporal order and they have
minimal effect on performance ( _<_ 0 _._ 5%).



50 randomly sampled question-answer-video triplets in the
MSRVTT-QA dataset. We found that the 6% of the questions are not exactly aligned with the video, e.g., wrong
entity descriptions/wrong action descriptions. 24% of the
questions have grammatical errors. 10% of the questions
are ambiguous. These results indicate that when the text
queries have very low-quality, the well-trained text embedding space may suffer from the style change of the input
language. More discussion is included in the supplemen
tary.


**5. Conclusion**


We aim at a novel and challenging problem: fast adaption of pretrained contrastive multimodal models and pretrained contrastive text models for multi-channel video
language retrieval under limited data and computation resources. We systematically evaluate four variants of models
from our identified principled model design space. Through
extensive analysis, we find the variant with text tokens as
video representations and contrastive text model for multimodal fusion achieve the best performance. Without millions of multimodal data for pretraining and three orders of
magnitude more training time, our model achieves compa


rable performance with the state-of-the-art model. We further show that this simple yet effective variant can be easily
improved by using better pretrained contrastive multimodal
models or pretrained contrastive text models, which uncovers the great potential of our proposed model to democratize the video-language research community from heavy dependence on huge data and computation resources towards
upgradable multimodal intelligence.


**6. Acknowledgement**


We thank the anonymous reviewers, Richard Zemel,
other DVMMers and colleagues for their feedback. This paper is based upon work supported by U.S. DARPA KAIROS
Program No. FA8750-19-2-1004. The views and conclusions contained herein are those of the authors and should

not be interpreted as necessarily representing the official
policies, either expressed or implied, of DARPA, or the U.S.
Government. The U.S. Government is authorized to re
produce and distribute reprints for governmental purposes
notwithstanding any copyright annotation therein. Mike
Zheng Shou does not receive any funding for this work.


**References**


[1] Peter Anderson, Xiaodong He, Chris Buehler, Damien
Teney, Mark Johnson, Stephen Gould, and Lei Zhang.
Bottom-up and top-down attention for image captioning and
visual question answering. In _Proceedings of the IEEE con-_
_ference on computer vision and pattern recognition_, pages
6077–6086, 2018. 2

[2] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret
Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh.
Vqa: Visual question answering. In _Proceedings of the IEEE_
_international conference on computer vision_, pages 2425–
2433, 2015. 2

[3] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. _arXiv preprint arXiv:1607.06450_,
2016. 11

[4] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem,
and Juan Carlos Niebles. Activitynet: A large-scale video
benchmark for human activity understanding. In _Proceed-_
_ings of the ieee conference on computer vision and pattern_
_recognition_, pages 961–970, 2015. 11

[5] Joao Carreira and Andrew Zisserman. Quo vadis, action
recognition? a new model and the kinetics dataset. In _CVPR_,
2017. 1

[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. Bert: Pre-training of deep bidirectional
transformers for language understanding. _arXiv preprint_
_arXiv:1810.04805_, 2018. 1, 2

[7] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and
Kaiming He. Slowfast networks for video recognition. In
_Proceedings of the IEEE/CVF international conference on_
_computer vision_, pages 6202–6211, 2019. 12

[8] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse:
Simple contrastive learning of sentence embeddings. _arXiv_
_preprint arXiv:2104.08821_, 2021. 1, 2, 6, 7

[9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In _CVPR_,
2016. 1

[10] Michael Heilman and Noah A Smith. Good question! statistical ranking for question generation. In _Human Language_
_Technologies: The 2010 Annual Conference of the North_
_American Chapter of the Association for Computational Lin-_
_guistics_, pages 609–617, 2010. 12

[11] Matthew Honnibal and Ines Montani. spacy 2: Natural
language understanding with bloom embeddings, convolutional neural networks and incremental parsing. _To appear_,
7(1):411–420, 2017. 5, 12

[12] Hung Le, Doyen Sahoo, Nancy Chen, and Steven Hoi. Multimodal transformer networks for end-to-end video-grounded
dialogue systems. In _Proceedings of the 57th Annual Meet-_
_ing of the Association for Computational Linguistics_, pages
5612–5623, 2019. 2

[13] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu,
and Jingjing Liu. Hero: Hierarchical encoder for video+
language omni-representation pre-training. _arXiv preprint_
_arXiv:2005.00200_, 2020. 1, 2, 5, 11, 13

[14] Linjie Li, Jie Lei, Zhe Gan, Licheng Yu, Yen-Chun Chen,
Rohit Pillai, Yu Cheng, Luowei Zhou, Xin Eric Wang,



William Yang Wang, et al. Value: A multi-task benchmark
for video-and-language understanding evaluation. _arXiv_
_preprint arXiv:2106.04632_, 2021. 11, 13

[15] Manling Li, Ruochen Xu, Shuohang Wang, Luowei Zhou,
Xudong Lin, Chenguang Zhu, Michael Zeng, Heng Ji, and
Shih-Fu Chang. Clip-event: Connecting text and images
with event structures. _arXiv preprint arXiv:2201.05078_,
2022. 1, 2

[16] Xudong Lin, Gedas Bertasius, Jue Wang, Shih-Fu Chang,
Devi Parikh, and Lorenzo Torresani. Vx2text: End-to-end
learning of video-based text generation from multimodal inputs. In _Proceedings of the IEEE/CVF Conference on Com-_
_puter Vision and Pattern Recognition_, pages 7005–7015,
2021. 2

[17] Xudong Lin, Fabio Petroni, Gedas Bertasius, Marcus
Rohrbach, Shih-Fu Chang, and Lorenzo Torresani. Learning
to recognize procedural activities with distant supervision.
In _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_, pages 13853–13863, 2022.
2

[18] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar
Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized
bert pretraining approach. _arXiv preprint arXiv:1907.11692_,
2019. 1

[19] Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh.
Hierarchical question-image co-attention for visual question
answering. In _Advances in neural information processing_
_systems_, pages 289–297, 2016. 2

[20] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei,
Nan Duan, and Tianrui Li. Clip4clip: An empirical study
of clip for end to end video clip retrieval. _arXiv preprint_
_arXiv:2104.08860_, 2021. 1

[21] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan
Laptev, Josef Sivic, and Andrew Zisserman. End-to-end
learning of visual representations from uncurated instructional videos. In _Proceedings of the IEEE/CVF Conference_
_on Computer Vision and Pattern Recognition_, pages 9879–
9889, 2020. 1, 2, 5, 6, 11, 12

[22] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac,
Makarand Tapaswi, Ivan Laptev, and Josef Sivic.
Howto100m: Learning a text-video embedding by watching
hundred million narrated video clips. In _Proceedings of the_
_IEEE/CVF International Conference on Computer Vision_,
pages 2630–2640, 2019. 1

[23] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. _arXiv preprint_
_arXiv:2111.09734_, 2021. 2, 4

[24] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and Li Deng. Ms marco:
A human generated machine reading comprehension dataset.
In _CoCo@ NIPS_, 2016. 1

[25] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hern´andez ´Abrego, Ji Ma, Vincent Y Zhao, Yi
Luan, Keith B Hall, Ming-Wei Chang, et al. Large
dual encoders are generalizable retrievers. _arXiv preprint_
_arXiv:2112.07899_, 2021. 13


[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_,
pages 8748–8763. PMLR, 2021. 1, 2, 4, 5, 6, 12, 13

[27] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei
Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _arXiv preprint_
_arXiv:1910.10683_, 2019. 2

[28] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence
embeddings using siamese bert-networks. In _Proceedings of_
_the 2019 Conference on Empirical Methods in Natural Lan-_
_guage Processing_ . Association for Computational Linguistics, 11 2019. 1, 2

[29] Nils Reimers and Iryna Gurevych. Sentence Transformers.

[https://www.sbert.net/, 2019. 2, 5, 6, 7, 11](https://www.sbert.net/)

[30] Victor Sanh, Lysandre Debut, Julien Chaumond, and
Thomas Wolf. Distilbert, a distilled version of bert: smaller,
faster, cheaper and lighter. _arXiv preprint arXiv:1910.01108_,
2019. 1

[31] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal,
Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt
Keutzer. How much can clip benefit vision-and-language
tasks? _arXiv preprint arXiv:2107.06383_, 2021. 2

[32] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan
Liu. Mpnet: Masked and permuted pre-training for language
understanding. In H. Larochelle, M. Ranzato, R. Hadsell,
M. F. Balcan, and H. Lin, editors, _Advances in Neural Infor-_
_mation Processing Systems_, volume 33, pages 16857–16867.
Curran Associates, Inc., 2020. 5, 7

[33] Hao Tan and Mohit Bansal. Lxmert: Learning crossmodality encoder representations from transformers. _arXiv_
_preprint arXiv:1908.07490_, 2019. 2

[34] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need. In _Advances in neural_
_information processing systems_, pages 5998–6008, 2017. 4

[35] Jinpeng Wang, Yixiao Ge, Guanyu Cai, Rui Yan, Xudong
Lin, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Objectaware video-language pre-training for retrieval. In _Proceed-_
_ings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pages 3313–3322, 2022. 2

[36] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang,
and Ming Zhou. Minilm: Deep self-attention distillation for
task-agnostic compression of pre-trained transformers. _Ad-_
_vances in Neural Information Processing Systems_, 33:5776–
5788, 2020. 6

[37] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang
Wang, and William Yang Wang. Vatex: A large-scale, highquality multilingual dataset for video-and-language research.
In _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_, 2019. 2, 5, 11

[38] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang,
Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and mo


tion. In _Proceedings of the 25th ACM international confer-_
_ence on Multimedia_, pages 1645–1653, 2017. 2, 5


[39] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko,
Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and
Christoph Feichtenhofer. Videoclip: Contrastive pre-training
for zero-shot video-text understanding. _arXiv preprint_
_arXiv:2109.14084_, 2021. 1


[40] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and
Cordelia Schmid. Just ask: Learning to answer questions
from millions of narrated videos. In _Proceedings of the_
_IEEE/CVF International Conference on Computer Vision_,
pages 1686–1697, 2021. 1, 2, 4, 5, 11, 12


[41] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and
Cordelia Schmid. Learning to answer visual questions from
web videos. _IEEE Transactions on Pattern Analysis & Ma-_
_chine Intelligence_, (01):1–1, 2022. 2, 6


[42] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev,
and Cordelia Schmid. Zero-shot video question answering
via frozen bidirectional language models. _arXiv preprint_
_arXiv:2206.08155_, 2022. 5, 6


[43] Zekun Yang, Noa Garcia, Chenhui Chu, Mayu Otani, Yuta
Nakashima, and Haruo Takemura. Bert representations for
video question answering. In _The IEEE Winter Confer-_
_ence on Applications of Computer Vision_, pages 1556–1565,
2020. 2


[44] Weijiang Yu, Haoteng Zheng, Mengfei Li, Lei Ji, Lijun Wu,
Nong Xiao, and Nan Duan. Learning from inside: Selfdriven siamese sampling and reasoning for video question
answering. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S.
Liang, and J. Wortman Vaughan, editors, _Advances in Neural_
_Information Processing Systems_, volume 34, pages 26462–
26474. Curran Associates, Inc., 2021. 6


[45] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting
Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for
understanding complex web videos via question answering.
In _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_, pages 9127–9134, 2019. 2,
5, 11


[46] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu,
Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. _Advances_
_in Neural Information Processing Systems_, 34, 2021. 1, 6


[47] Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J Corso, and Jianfeng Gao. Unified vision-language
pre-training for image captioning and vqa. _arXiv preprint_
_arXiv:1909.11059_, 2019. 2


[48] Luowei Zhou, Chenliang Xu, and Jason J Corso. Towards
automatic learning of procedures from web instructional
videos. In _Thirty-Second AAAI Conference on Artificial In-_
_telligence_, 2018. 2


[49] Luowei Zhou, Chenliang Xu, and Jason J Corso. Towards
automatic learning of procedures from web instructional
videos. In _AAAI Conference on Artificial Intelligence_, pages
7590–7598, 2018. 5, 11


**A. Discussion on Reasons Why Text + Text Per-**
**forms Best**


We think the main reason that the proposed Text + Text
variant can outperform other variants is that it is hard and
resource-intensive to learn the alignment between the text
embedding space and the visual feature space, and train a
good multimodal transformer even with strong pretrained
text models as initialization. There are two lines of evi
dence we observed: first, as discussed in the next section,
these two spaces are significantly different even in terms
of statistics, which is verified by the fact that without the
trick to initialize the projected visual features to have the
same mean and variance with text embeddings, Conti. +
Text would suffer from a huge performance drop of 4%;
second, from results in the main paper, it is clear that Conti.
+ Multi. seriously rely on the huge amount of video-text
samples to align the embedding space and train the multimodal transformer. The Text + Text variant exactly avoids
these two processes in its design.


**B. Additional Implementation Details**


We use a learning rate of 0.00005, learning rate decay
of 0.9 and batch size of 256. We trained our model for

20 epochs on iVQA and MSVD-QA and for 30 epochs on
How2QA, ActivityNet-QA and MSRVTT-QA on 2 Nvidia
V100 GPUs. On VATEX and YouCook II, the number
epochs is 10 and 100, respectively. We find that the Masked
Language Modeling objective is not helpful when we use
MPNet as the language transformer, and thus, we only use
the contrastive loss as described in the main paper. We use
a gradient clipping of 1, following [29]. The other hyperparameters were directly borrowed from Just-ask [40].
In the implementation of Continuous Features + Text
Transformer, we find it important to initialize the last LayerNorm [3] in the projector with parameters of the LayerNorm after the embedding layer of MPNet so that the
projected video features are aligned with the textual embeddings. Without this initialization trick, the model only
achieves 19.8% on iVQA (when with the initialization, it
achieves 23.2%).
We use a public python library [2] to obtain the automatic
speech transcripts from YouTube for the videos we used
in iVQA, How2QA, ActivityNet-QA, YouCook II and VATEX. We also used subtitles from [14] for videos with
speech but without ASR. Eventually, about 90%, 70%,
30%, 90%, and 50% of the videos have associated speech
transcripts, respectively.
We fix the random seed in all the experiments and we
do not observe significant change of accuracy ( _<_ 0 _._ 5%)


2 [https://pypi.org/project/youtube- transcript-](https://pypi.org/project/youtube-transcript-api/)
[api/](https://pypi.org/project/youtube-transcript-api/)



Number of Tokens 60k-word Answer-word


10 25.9 29.8


15 27.3 30.9


20 26.8 31.1


25 26.5 31.6


30 26.2 30.9


Table 3. Accuracy on varying the number of text tokens and 60kword vocabulary and the answer-word vocabulary on the iVQA
dataset


when changing the random seed. The code will be re[leased at https://github.com/XudongLinthu/](https://github.com/XudongLinthu/upgradable-multimodal-intelligence)
[upgradable-multimodal-intelligence for any](https://github.com/XudongLinthu/upgradable-multimodal-intelligence)
other details and results on additional datasets.


**C. Details about Datasets and Evaluation Met-**

**ric**


**iVQA [40]** . It contains 10,000 instructional videos. Each
video is annotated with one question and five corresponding
answers. We follow the official split to use 6,000, 2,000,
and 2,000 videos for training, validation, and testing, respectively. We follow [40] to calculate accuracy with five
annotations per question.
**How2QA [13]** . The dataset contains 44,007 QA pairs that
are annotated from 9,035 videos. We follow [40] to use the
train and validation split for training and testing. Note that
in this dataset, each question and answer pair are manually
annotated with three negative answers so we actually don’t
need to retrieve from a huge answer set. The metric used
for this dataset is accuracy.
**ActivityNet-QA [45]** . It contains 58,000 QA pairs manually annotated from 5,800 videos from the ActivityNet [4]
dataset depicting a wide range of complex human activities.
The official split of 32,000, 18,000 and 8,000 QA pairs for
training, validation and testing respectively is adopted.
**YouCook II [49]** . It is a instructional video dataset containing 2,000 long videos of 89 recipes. We follow [21] to
use the temporal boundary of steps to formulate a pair as a
step description and the corresponding video segment. The
resulted number of training and testing pairs are 10,387 and
3,411.
**VATEX [37]** . We only take videos and English captions
from it to evaluate retrieval performance. Due to the fact
that only 50% of the videos have ASR and many ASRs only
contain English stop words, we only keep the videos with at
least 5 non-stop words to make sure the task is still multichannel. The resulted number of training and testing pairs
are 36,680 and 4,190. When comparing with HERO [13],
we train and evaluate our model under its data split.


**D. Experiments on** _**k**_ **and the Vocabulary.**


In this section, we report the results when varying _k_ and
the vocabulary for retrieving text tokens, as shown in Table 3. 60k-word vocabulary contains all 65,000 the words
that are used in the language model of [21]. The answerword vocabulary is constructed by collecting all the unique
verbs and nouns from parsing all the query/answer sentences in the downstream datasets with spaCy [11].


We observe that when larger than 15 words are retrieved
for each segment in the video, the benefit of retrieving more
tokens starts to be marginal. Therefore, we use _k_ = 15 for
all the other datasets as less tokens also help to further accelerate the training process. But we use _k_ = 25 for the iVQA
dataset as this turns out to be the optimal value when using the answer-word vocabulary on the iVQA dataset. This
also indicates that the performance on other downstream
datasets could be further improved if we optimize the number of tokens.


We also observe a consistent improvement for all the
number of tokens when changing from the 60k-word vocabulary to the answer-word vocabulary. Therefore, we construct the answer-word vocabulary for each dataset separately, which have a size from 3K to 25K words, depending
on the dataset. Note that for retrieval datasets, we use the
words from all the text queries.


**E. Additional Comparison with the State-of-**
**the-art**


As shown in Table 4, we provide additional comparison
with state-of-the-art on the two retrieval datasets. We ob
serve that our proposed **Text Tokens + Text Transformer**
performs slightly better than HERO, which requires a lot
more data and computational resources to train the model
properly. Note that our model only uses pretrained S3D [21]
(on YouCook II) or CLIP [26] (on VATEX) to retrieve text
tokens as additional representation of the video but HERO
that uses features from both 3D extractors [7,21] and 2D extractors [26]. Note that we found that on YouCook II, using
_K_ = 25 helps to improve the results.


AT-ST HERO is annotated as gray because it further
leverages about 730K extra **annotated** multimodal samples
to perform multi-task training and then it is fine-tuned for
specific datasets, which is not directly comparable to our
setting. Interestingly, the additional annotated data helps
AT-ST HERO to significantly improves the performance
on VATEX but the performance on YouCook II is actually
even lower. Overall, when without such multi-task learning,
our proposed method still achieves comparable results with
state-of-the-art on these datasets.



**F. Additional Discussion on MSRVTT-QA and**
**MSVD-QA**


We list a few question-answer samples from the test set
of MSRVTT-QA in Table 5. The questions are not natural
sentences and the answers are not informative. To quantify the imperfection of question generation, we carefully
convey a manual study on 50 randomly sampled questionanswer-video triplets in the MSRVTT-QA dataset. We
found that the 6% of the questions are not exactly aligned
with the video, e.g., wrong entity descriptions/wrong action
descriptions. 24% of the questions have grammatical errors. 10% of the questions are ambiguous. For the answers,
we found that 12% of them are either not aligned with the
video or too ambiguous to determine correctness. 32% of
the answers are not informative enough but they are roughly
aligned with the question and the video.
A side evidence is that the Table 7 in [40] compares
between the method used to generate MSRVTT-QA and
MSVD-QA and the method used in Just-ask for questionanswer generation. The resulted model pretrained with generation method of [40] significantly outperforms the model
pretrained with data generated from method [10], by a large
relative margin of 20% to 1000% of zero-shot accuracy on
three manually annotated datasets, which indicates the unsatisfactory quality of MSRVTT-QA and MSVD-QA generated by [10].
Both the qualitative and quantitative results motivate us
to not use them as the datasets to assess the performance of
the four variants.


**G. Additional Visualisation**


As shown in Figure 6, the answer words or some of the
answer words are retrieved as one of the text tokens to de
scribe the video. As discussed in the main paper, we find
that 64% of the videos have at least one word overlap between the retrieved text tokens and the answers. This kind

of examples show the high explainability of the proposed
method as it is clear these important key words are the input to the model to make the final prediction.


**H. Discussion on Social Impact, Limitation**
**and Future Work**


We first argue that the tasks we handle in this work are
potentially helpful for visually-impaired people to better
handle daily life as our model can be used to help them
understand the ongoing events with text queries.
The datasets we used are mostly based on videos from
YouTube. Therefore, they may contain personal information but our algorithm is not designed to specifically leverage certain private information. Overall, we do not expect
negative societal impact from the designed algorithms but


Model Extra MM Samples ∆ GPU hours YouCook II VATEX


AT-ST HERO [13, 14] 7.6M + 700K           - 45.3 80.0


HERO [13, 14] 7.6M 8,000 49.5 **63.4**


Text + Text (Ours) 0 0 **50.6** 60.1


Table 4. Comparison with the state-of-the-art on YouCook II and VATEX in terms of accuracy and efficiency. Extra MM Samples indicate
the number of video-text samples that are needed in the second-round pretraining. ∆ GPU hours refer to the additional computation
required for the second-round pretraining. Note that our variant typically requires 1 GPU hour for training. Average recall@ _{_ 1,5,10 _}_ (%)
is reported. AT-ST HERO is annotated as gray because it further leverages about 730K extra **annotated** multimodal samples to perform
multi-task training, which is not directly comparable to our setting. Note that even HERO is not completely comparable with our model,
as it enjoys both video-level and frame-level feature extractors.



**Time**


**Frames**



0.36 0.40 0.41 0.45 0.46 1.04 **Question**


**Text**

**Tokens**


**Predicted**

**Answer**


**Ground Truth**

**Answers**



What is the woman wrapping around a needle?


First. crucifix. earbud. knitting. thread chain. pullover.
crochet needle. yarn ball. knitted. designer chain. sweater.
crochet hook. kerchief. yarn. crocheting. crochet.

Then. knitting. pullover. yarn ball. knitting nettle. knitted.
knitting needle. crocheting. designer chain. crochet hook.
knitting pin. knit scarf. sweater. knitting stick. knitting
needles. woolen thread.

Then. earbud. yarn ball. crochet needle. thread chain.
knitting. crucifix. knitted. crochet hook. left ring finger.
needle. knitting needle. tapestry. index finger. crochet.
crocheting.


yarn


yarn, yarn, yarn, yarn, yarn



Figure 6. Visualisation of a successful case for **Text Tokens + Text Transformer** on the iVQA dataset. In this example, the answer “yarn”
is retrieved as one of the text tokens to describe the video.


Question Answer


what is a clip doing? show
who is showing something in a computer? someone
what can singing? pain
what explains colors? video


Table 5. Question-answer samples from the test set of MSRVTTQA.


the dataset we use for training may lead the models to produce biased or undesired results.

One limitation is on the imperfection of the visual words
retrieval process. We have already shown using CLIP [26]
significantly helps to improve the performance in the main
paper. We leave further using better pretrained multimodal
contrastive models to future work. We also leave the utiliza
tion of recent larger pretrained contrastive text models [25]
(with billions of parameters) to future research.


