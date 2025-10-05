## **Unified Video-Language Pre-training with** **Synchronized Audio**

**Shentong Mo** [1] _[,]_ [2] _[∗]_ **Haofan Wang** [3] **Huaxia Li** [3] **Xu Tang** [3]

1 CMU, 2 MBZUAI, 3 Xiaohongshu


**Abstract**


Video-language pre-training is a typical and challenging problem that aims at learning visual and textual representations from large-scale data in a self-supervised way.
Existing pre-training approaches either captured the correspondence of image-text
pairs or utilized temporal ordering of frames. However, they do not explicitly
explore the natural synchronization between audio and the other two modalities. In
this work, we propose an enhanced framework for Video-Language pre-training
with Synchronized Audio, termed as VLSA, that can learn tri-modal representations
in a unified self-supervised transformer. Specifically, our VLSA jointly aggregates
embeddings of local patches and global tokens for video, text, and audio. Furthermore, we utilize local-patch masked modeling to learn modality-aware features,
and leverage global audio matching to capture audio-guided features for video
and text. We conduct extensive experiments on retrieval across text, video, and
audio. Our simple model pre-trained on only 0.9M data achieves improving results
against state-of-the-art baselines. In addition, qualitative visualizations vividly
showcase the superiority of our VLSA in learning discriminative visual-textual
representations.


**1** **Introduction**


When we enjoy fascinating story in a movie, not only frames with the caption are attractive, but synchronized sounds are also impressive for us. In daily life, sound appears to be in multiple scenes, such
as a classroom scene with professors’ speech and students’ whispering, a family reunion that consists
of talking and laughing. As audio and visual contents are commonly matched and synchronized, many
researchers [ 1, 2, 3, 4, 5, 6, 7 ] have explored the benefit of such synchronization for audio-visual tasks,
such as audio-event localization [ 8, 9, 10, 11, 12, 13 ], audio-visual spatialization [ 14, 15, 16, 17 ], and
sound source localization [ 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 ]. However, in this work, we leverage
this cross-modal synchronization between audio and videos/texts for enhancing video-language
pre-training.


Video-language pre-training aims to learn visual and textual representations jointly from large-scale
data. Previous work [ 28, 29, 30, 31, 32, 33, 34, 35 ] either proposed to learn the correspondence of
image-text pairs or utilized temporal ordering of frames. Typically, MMT [ 29 ] proposed a multimodal transformer to aggregate per-frame visual features with temporal information. With the success
of CLIP [ 36 ] on visual and textual representation learning, CLIP2TV [ 37 ] leveraged CLIP pre-trained
weights with a video-text alignment module and a video-text matching module for discriminating
positive and negative pairs of embeddings from text and video encoders.


However, they do not explicitly explore the natural synchronization between audio and other two
modalities, and can not learn compact representations of the global correspondence across each
modality. In contrast, we leverage synchronized audio for global audio matching in our unified
transformer to capture discriminative global features for boosting video-language pre-training.


Preprint.


Figure 1: **Comparison performance of text-to-video (Left), zero-shot text-to-video (Middle), and**
**text-to-audio retrieval (Right).** VLSA pre-trained on only 0.9M data achieves significant gains
compared to previous video-language (MEE, HT, MMT, SupportSet, Frozen, OA-Trans, AllinOne),
video-audio (TVLT), and video-language-audio (CE, VATT) methods.


Since audio and visual contents are commonly matched and synchronized, recent pre-training
methods [ 38, 39, 40 ] are developed to incorporate audio as an auxiliary modality for learning
discriminative representations transferrable to video-text retrieval. VATT [ 38 ] introduced a multimodal contrastive loss for global features of each modality to learn the alignment of video-audiotext triplets. AudioCLIP [ 39 ] extended CLIP with the ESResNeXt audio model and optimized
a contrastive loss between two modalities to maximize diagonal values of the scaled dot-product
similarity matrix. More recently, MERLOT Reserve [ 40 ] employed a contrastive span training loss
between masked tokens and corresponding global uni-modal embeddings for audio, captions, and
video frames. While these methods achieve promising results on video-language pre-training, they
are extremely dependent on the capacity of three separate encoders with many parameters to extract
discriminative representations for each modality on large-scale datasets. In addition, they do not
incorporate local features into the pre-training objective. Different from them, we combine localpatch masked modeling with global-token alignment in a unified transformer to learn modality-aware
features during pre-training.


To this end, we propose a novel and enhanced framework for Video-Language pre-training with
Synchronized Audio, namely VLSA, that can jointly learn tri-modal representations in a unified selfsupervised transformer. Specifically, VLSA aggregates local patches and global token representations
for each modality through a joint transformer. Furthermore, local-patch masked modeling is applied
to patch-level embeddings of each modality to learn modality-aware local features. In addition, global
audio matching is applied on both video and text modalities to capture compact features from global
tokens of audio, text, and video frames.


We pre-train our VLSA on only 0.9M video-audio-transcript triplets from an intersection set of
HowTo100M and AudioSet. We conduct extensive experiments on retrieval across text, video,
and audio. Our simple pre-trained model achieves improving results against previous baselines
on downstream tasks. In addition, qualitative visualizations vividly showcase the advantage of the
proposed approach in learning compact visual-textual representations.


2


Table 1: **Comparison of multi-modal self-supervised pre-training on video, text, and audio**
**or speech.** “Single”, “Dual”, and “Triple” refer to one joint encoder, two separate encoders, and
three separate encoders, respectively. Local and global denote uni-modal masked modeling and
multi-modal matching.


Method Publication Modalities Architecture Pre-train Objectives Pre-train Datasets


VideoBERT [31] ICCV, 2019 Video, Text Single local self-collected
Frozen [32] ICCV, 2021 Video, Text Dual global WebVid2M+CC3M
OA-Trans [34] CVPR, 2022 Video, Text Dual global WebVid2M+CC3M
Region-Learner [56] AAAI, 2023 Video, Text Single global WebVid2M+CC3M
AllinOne [35] CVPR, 2023 Video, Text Single local+global WebVid2M+HowTo100M
VATT [38] NeurIPS, 2021 Video, Text, Audio Single/Triple global AudioSet+HowTo100M
AudioCLIP [39] ICASSP, 2022 Video, Text, Audio Triple global AudioSet
MERLOT Reserve [40] CVPR, 2022 Video, Text, Audio Triple global YT-Temporal-1B
i-Code [57] AAAI, 2023 Video, Text, Speech Single local+global self-collected
VLSA (ours) – Video, Text, Audio Single local+global AudioSet _∩_ HowTo100M


Our contributions can be summarized as follows:


    - We propose a novel and enhanced framework for Video-Language pre-training with Synchronized Audio, termed as VLSA, that aggregates local patches and global visual-textual
representation guided by audio through a joint transformer encoder.


    - We introduce audio local-patch masked modeling to learn modality-aware interaction between audio and the other two modalities.


    - We further leverage global audio matching to capture compact and discriminative video-text
features without the need for video-text matching.


    - Our simple unified transformer pre-trained on only 0.9M triplets achieves competitive results
against baselines on text-video and text-audio retrieval.


**2** **Related Work**


**Video-Language Pre-training.** Video-language pre-training aims at learning joint visual and textual
representations from captions and video frames. At the early stage, CE [ 28 ] designed a collaborative
experts model to aggregate information from different pre-trained experts for video retrieval tasks.
MEE [ 41 ] proposed a model with mixed embedding experts to handle missing input modalities for
learning improved text-video embeddings simultaneously. In the recent years, diverse pipelines [ 42,
43, 44, 45, 31, 46, 47, 48, 49, 50, 51, 52, 32, 34, 35, 53 ] have been proposed to explore the fusion
of two distinct modalities, where the correspondence of image-text pairs is usually aligned during
pre-training, as shown in Table 1. Typically, ActBERT [ 30 ] leveraged a tangled transformer to learn
the correspondence between global actions and local object regions from paired video sequences and
text descriptions. A multi-modal transformer was introduced in MMT [ 29 ] to aggregate per-frame
visual features with temporal information. CAMoE [ 33 ] utilized mixture-of-experts to capture the
alignment between text and multiple video representations of action and scene. Based on CLIP [ 36 ]
pre-trained weights, CLIP2TV [ 37 ] adopted a video-text alignment module and a video-text matching
module to discriminate positive and negative pairs of features from text and video encoders.


While the aforementioned video-language pre-training approaches achieve promising performance
on downstream tasks, they do not learn the natural synchronization between audio and the other two
modalities in an explicit manner. Moreover, they can not learn compact representations of the global
correspondence across each modality for retrieval tasks [ 54, 55 ] involved with audio. In this work,
we aim to enhance video-language pre-training with synchronized audio by global-token matching
for both text-video and text-audio retrieval.


**Audio Synchronization in Pre-training.** Audio synchronization in pre-training has been applied
in previous works [ 38, 39, 40, 57 ] to learn discriminative representations of audio, captions, and
video frames. VATT [ 38 ] introduced a multi-modal contrastive loss for global features of each
modality to learn the alignment of video-audio-text triplets. AudioCLIP [ 39 ] extended CLIP with
the ESResNeXt audio model and optimized a contrastive loss between two modalities to maximize
diagonal values of the scaled dot-product similarity matrix. More recently, MERLOT Reserve [ 40 ]
employed a contrastive span training loss between masked tokens and corresponding global unimodal embeddings for audio, captions, and video frames. i-Code [ 57 ] exploited a multi-modal fusion


3


**Model Input**





**Video Frames** " +,








|Col1|!! "|
|---|---|
|project<br>{!<br>"}<br>+,||
|project<br>{!<br>"}<br>+,||








|Col1|Col2|Col3|sked Modeling|
|---|---|---|---|
||||Decoder<br>Decoder<br><br>Share Parameter<br>Share Parameter<br>#"|
|||||
|Encoder||||





**Audio Spectrogram**


“I want to play in
the living room.”


**Transcript Sentence**


|project|!! #|
|---|---|
|{!<br>#}<br>*||
|{!<br>#}<br>*||



{! !



!$ } !&')









"


121 35 58 234


word embedding








































|project<br>$)|!!$|
|---|---|
|project<br><br>$<br>)||
|project<br><br>$<br>)||
|project<br><br>$<br>)||









Figure 2: **Illustration of our enhanced framework of Video-Language pre-training with Syn-**
**chronized Audio (VLSA).** The modality-aware patch embeddings _{_ **x** _[v]_ _i_ _[}]_ _[V I]_ _i_ =1 [,] _[ {]_ **[x]** _[a]_ _i_ _[}]_ _[A]_ _i_ =1 [,] _[ {]_ **[x]** _[t]_ _i_ _[}]_ _[S]_ _i_ =1 [, are]
extracted from each linear projection layer. The Local-Patch Masked Modeling module is applied
to local-patch representations for audio spectrogram **a** extracted from the unified encoder, and the
decoder is utilized to predict the raw audio spectrograms ˆ **a** for learning the interaction of audio and
the other two modalities (video and text). Finally, the Global Audio Matching module (contrastive
loss and binary matching loss) is leveraged on modality-aware global embeddings ˆ **g** _[v]_ _,_ ˆ **g** _[t]_ _,_ ˆ **g** _[a]_ averaged from the encoder to capture the cross-modal alignment between synchronized audio and video
frames/caption sentence in an explicit manner.


network to integrate single-modality outputs of vision, speech, and language from each pre-trained
encoder.


Different from these tri-modal pre-training baselines, the proposed transformer is more lightweight
than comparable methods, as we do not need to use three separate encoders with many parameters
to extract discriminative embeddings for each modality. In addition, we develop a fully novel and
unified transformer to aggregate both local-patch masked modeling and global-token matching for
learning modality-aware features. Audio-video matching and masked modeling are also addressed in
a very recent work called TVLT [58], but they do not involve captions during pre-training.


**3** **Method**


Given a triplet of video sequences, captions, and synchronized audio, our target is to learn joint
representations from large-scale unsupervised data. We propose a novel Video-Language pre-training
framework with Synchronized Audio named VLSA for enhancing visual-textual semantics from
sound itself, which mainly consists of two modules, Local-Patch Masked Modeling in Section 3.2
and Global Audio Matching in Section 3.3.


**3.1** **Preliminaries**


In this section, we first describe the problem setup and notations, and then revisit the commonly-used
objective for video-language pre-training.


**Problem Setup and Notations.** Given a triplet of video sequences with a dimension of _V ×_ 3 _×H_ _×W_,
texts with a dimension of _S × C_, and synchronized audio spectrograms with a dimension of _T × F_,
our target is to learn discriminative representations simultaneously and evaluate them for downstream
retrieval tasks. We formally denote _V_ as the number of video frames, _S_ as the length of text sentences,
_C_ as the length of the word dictionary, and _F_ as the number of frequencies in audio. _H_ and _W_ are
the height and width of each frame in the video.


**Revisit Video-Language Pre-training.** To solve the joint modeling problem, current pre-training
baselines [ 47, 48, 49, 50, 51, 52, 32, 34, 35 ] introduced a visual-textual contrastive loss to learn
the alignment between images and captions. Given a set of global video-language features
_{_ **g** _i_ _[v]_ _[}]_ _[B]_ _i_ =1 _[,][ {]_ **[g]** _i_ _[t]_ _[}]_ _[B]_ _i_ =1 [in a batch of size] _[ B]_ [, the contrastive loss between the cosine similarity] `[ sim]` [(] **[g]** _i_ _[v]_ _[,]_ **[ g]** _i_ _[t]_ [)]


4


of global embeddings is formulated as:



~~�~~ _Bj_ =1 [exp] ~~�~~ _τ_ 1



1 _,_ (1)

_τ_ `[sim]` [(] **[g]** _i_ _[v]_ _[,]_ **[ g]** _j_ _[t]_ [)] ~~�~~



_L_ _v→t_ = _B_ [1]



_B_
�



1
_−_ exp � _τ_
_i_ log ~~�~~ _B_ =1 [exp]



1

_τ_ `[sim]` [(] **[g]** _i_ _[v]_ _[,]_ **[ g]** _i_ _[t]_ [)] �



where **g** _i_ _[v]_ _[,]_ **[ g]** _j_ _[t]_ _∈_ R [1] _[×][D]_, and _D_ is the size of the embedding dimension. `sim` ( **g** _i_ _[v]_ _[,]_ **[ g]** _i_ _[t]_ [) =]
( **g** _i_ _[v]_ [)] _[⊤]_ **[g]** _i_ _[t]_ _[/]_ [(] _[∥]_ **[g]** _i_ _[v]_ _[∥∥]_ **[g]** _i_ _[t]_ _[∥]_ [)] [ is the cosine similarity, and] _[ τ]_ [ is the temperature parameter.] _[ B]_ [2] _[ −]_ _[B]_ [ neg-]
ative vision-language pairs are created within a training batch. Besides, they proposed to take _N_ _[v,t]_
samples randomly from the whole dataset and applied an FC layer and sigmoid operator to predict
the matching probability **p** _[v,t]_ of image-caption pairs. The video-text matching loss is formally
summarized as:



_L_ _vtm_ =



_N_ _[v,t]_
� BCE � **y** _n_ _[v,t]_ _[,]_ **[ p]** _[v,t]_ _n_ � (2)


_n_ =1



where **y** _n_ _[v,t]_ _[,]_ **[ p]** _[v,t]_ _n_ [denote the ground-truth and probability of] _[ n]_ [th visual-textual representation pairs.]
BCE( _·_ ) is a binary cross-entropy loss. If the image and caption are from the same pair, the target
is 1; otherwise, it is 0. During training, the positive pair is sampled from the dataset, while the
negative pair is created by replacing the text or vision in a paired sample with a randomly selected
other sample. Overall, the global loss for video-language alignment can be computed by _L_ _v,t_ =
_L_ _v→t_ + _L_ _t→v_ + _L_ _vtm_ .


However, those video-language pre-training approaches are extremely dependent on the capacity of
encoders with many parameters to extract discriminative global embeddings for each modality. In
addition, most frameworks do not explicitly learn the natural synchronization between audio and
video/text. To address this issue, we propose a novel and unified transformer for Video-Language
pre-training with Synchronized Audio, that can learn both local and global features between audio
and videos/captions simultaneously, as illustrated in Figure 2.


**3.2** **Local-Patch Masked Modeling**


In order to learn local modality-aware features across three different modalities, we introduce
modality-aware patch embeddings for each modality that are extracted from raw input via each
linear projection layer, _i.e._, **x** _[v]_ _∈_ R [(] _[V][ ×][I]_ [)] _[×][D]_, **x** _[t]_ _∈_ R _[S][×][D]_, **x** _[a]_ _∈_ R _[A][×][D]_, where _I, S, A_ denotes
the total number of patches for each video frame, each caption, and the corresponding audio.
Assume the patch resolution of each frame and audio are _P_ _[v]_ _, P_ _[a]_, the patch-wise raw input for
video and audio are formally denoted as **v** _∈_ R [(] _[V][ ×][I]_ [)] _[×]_ [(3] _[×][P]_ _[ v]_ _[×][P]_ _[ v]_ [)] and **a** _∈_ R _[A][×]_ [(] _[P]_ _[ a]_ _[×][P]_ _[ a]_ [)] . Note that
_I_ = _H/P_ _[v]_ _× W/P_ _[v]_ _, A_ = _T/P_ _[a]_ _× F/P_ _[a]_ .

With local-patch representations for each modality _{_ **x** _[v]_ _i_ _[}]_ _[V I]_ _i_ =1 _[,][ {]_ **[x]** _[t]_ _i_ _[}]_ _[S]_ _i_ =1 _[,][ {]_ **[x]** _[a]_ _i_ _[}]_ _[A]_ _i_ =1 [, we first apply an]

_·_
unified transformer encoder _ϕ_ ( ) to aggregate patch-level features from the raw input as:

_{_ **x** ˆ _[v]_ _}_ _[V I]_ _i_ =1 _[,][ {]_ **[x]** [ˆ] _[t]_ _i_ _[}]_ _[S]_ _i_ =1 _[,][ {]_ **[x]** [ˆ] _[a]_ _i_ _[}]_ _[A]_ _i_ =1 [=] _[ {][ϕ]_ [(] **[x]** _[j]_ _[,]_ **[ X]** _[,]_ **[ X]** [)] _[}]_ _[V I]_ _j_ =1 [+] _[S]_ [+] _[A]_ _,_

(3)
**X** = _{_ **x** _j_ _}_ _[V I]_ _j_ =1 [+] _[S]_ [+] _[A]_ = [ _{_ **x** _[v]_ _i_ _[}]_ _[V I]_ _i_ =1 [;] _[ {]_ **[x]** _[t]_ _i_ _[}]_ _[S]_ _i_ =1 [;] _[ {]_ **[x]** _[a]_ _i_ _[}]_ _[A]_ _i_ =1 []]


where [ ; ] denotes the concatenation operator. **x** _[v]_ _i_ _[,]_ **[ x]** _[t]_ _i_ _[,]_ **[ x]** _[a]_ _i_ _[∈]_ [R] [1] _[×][D]_ [, and] _[ D]_ [ is the dimension of]
embeddings. The self-attention operator _ϕ_ ( _·_ ) is formulated as:



_ϕ_ ( **x** _j_ _,_ **X** _,_ **X** ) = Softmax( **[x]** _[j]_ **[X]** _[⊤]_



~~_√_~~



) **X** (4)
_D_



Then, to capture the interaction across each modality, we exploit a tri-modal masking mechanism
and a shared decoder to predict the masked patch of one modality, given the other two modalities as
auxiliaries. Specifically, with video and text embeddings _{_ **x** _[v]_ _i_ _[}]_ _[V I]_ _i_ =1 _[,][ {]_ **[x]** _[t]_ _i_ _[}]_ _[S]_ _i_ =1 [, we leverage a decoder]
to predict the raw audio spectrograms ˆ **a** for randomly masked audio patches. The local-level audio
masked loss is computed with the mean square loss between the targeted and predicted spectrograms

as:
1 ˆ
_L_ [local] _a_ = _N_ _[a]_ � _||_ **a** _i_ _−_ **a** _i_ _||_ 2 [2] (5)

_i∈M_ _[a]_


where _N_ _[a]_ _, M_ _[a]_ denotes the total number and the set of masked patches for audio, respectively. Similar
to audio, the local-level video masked loss _L_ [local] _v_ is calculated with the mean square loss between


5


the ground truth and missing pixel of frames with a random mask on all _V I_ patches. For text
masked loss _L_ [local] _t_, we use a decoder to predict the randomly masked token in _S_ tokens, similar to
BERT [ 59 ]. Note that three separate decoders for tri-modal masked prediction are parameter-shared
and achieve the best performance, as observed in our experiments in Section 4.3. With optimizing
the total local-level masked loss _L_ [local] = [�] _m∈{a,v,t}_ _[L]_ _m_ [local] with a shared encoder, it will capture the

local-level interaction between audio and other two distinct modalities, which pushes the model to
learn more discriminative embeddings.


**3.3** **Global Audio Matching**


Benefiting from the local-level masked loss above, we propose a novel and explicit global audio
matching mechanism on global embeddings ˆ **g** _[v]_ _,_ ˆ **g** _[t]_ _,_ ˆ **g** _[a]_ _∈_ R [1] _[×][D]_ for each modality in the unified
transformer _ϕ_ ( _·_ ) to generate global modality-aware features as:


ˆ ˆ
**g** _[v]_ ; ˆ **g** _[t]_ ; ˆ **g** _[a]_ = AvgPool( _{_ **x** _[v]_ _}_ _[V I]_ _i_ =1 [;] _[ {]_ **[x]** [ˆ] _[t]_ _i_ _[}]_ _[S]_ _i_ =1 [;] _[ {]_ **[x]** [ˆ] _[a]_ _i_ _[}]_ _[A]_ _i_ =1 [);] (6)


where ˆ **g** _[v]_ _,_ ˆ **g** _[t]_ _,_ ˆ **g** _[a]_ _∈_ R [1] _[×][D]_, and _D_ is the dimension of each global embedding. AvgPool( _·_ ) denotes
the average pooling operator. That is, we average local-level patches for three modalities along each
total number of patches ( _V I_, _S_, _A_ ) to generate the modality-aware global embeddings. [ ; ] is the
concatenation operator.


In order to explicitly learn the cross-modal alignment between synchronized audio and video frames,
we leverage a contrastive loss similar to Eq. 1 for maximizing the cosine similarity `sim` (ˆ **g** _i_ _[a]_ _[,]_ [ ˆ] **[g]** _i_ _[v]_ [)] [ of]
audio-video pairs from the same batch index _i_ . By applying an FC layer and sigmoid operator to
predict the alignment probability **p** _[a,v]_ _∈_ R _[N]_ _[×]_ [1] of _N_ _[a,v]_ audio-video pairs that are randomly chosen
from the whole pre-training dataset, we formulate the global audio-visual alignment loss as:



~~�~~ _Bj_ =1 [exp] ~~�~~ _τ_ 1



1

_τ_ `[sim]` [(ˆ] **[g]** _i_ _[a]_ _[,]_ [ ˆ] **[g]** _j_ _[v]_ [)] ~~�~~



_L_ [global] _a→v_ [=] _B_ [1]



_B_
�



1
_−_ exp � _τ_
_i_ log ~~�~~ _B_ =1 [exp]



1

_τ_ `[sim]` [(] **[g]** [ˆ] _i_ _[a]_ _[,]_ [ ˆ] **[g]** _i_ _[v]_ [)] �



(7)



1
_−_ exp � _τ_
log



+

1

_τ_ `[sim]` [(ˆ] **[g]** _i_ _[v]_ _[,]_ [ ˆ] **[g]** _j_ _[a]_ [)] ~~�~~



1

_τ_ `[sim]` [(] **[g]** [ˆ] _i_ _[v]_ _[,]_ [ ˆ] **[g]** _i_ _[a]_ [)] �



_N_ _[a,v]_
� BCE ( **y** _n_ _[a,v]_ _[,]_ **[ p]** _[a,v]_ _n_ [)]


_n_ =1



~~�~~ _Bj_ =1 [exp] ~~�~~ _τ_ 1



where _B_ is the batch size. **y** _[a,v]_ _∈_ R _[N]_ _[×]_ [1] denotes a one-hot encoding and its element for the entry is 0
for non-alignment and 1 for alignment. Since one audio spectrogram is distinct in each audio-video
pair, this alignment loss does not bring false alignment pairs, while most frames in a video look
similar in high-level textual semantics. To boost the compactness of pre-trained global representations
for retrieval, we apply similar cross-modal alignment loss _L_ [global] _a→t_ [on global tokens across audio-text]
pairs.


With optimizing the total global-token alignment loss _L_ [global] = _L_ [global] _a→v_ [+] _[ L]_ [global] _a→t_ [, we push the model]
to learn more discriminative video and text representations with the benefit of synchronized audio.
The overall objective of our model is simply optimized in an end-to-end manner as


_L_ = _L_ [local] + _λ · L_ [global] (8)


where _λ_ denotes the weighted parameter for balancing two losses with different orders of magnitude.
We use _λ_ = 5 as the default in our experiments. During inference, we simply compute the cosinesimilarity `sim` (ˆ **g** _[t]_ _,_ ˆ **g** _[v]_ ) _,_ `sim` (ˆ **g** _[t]_ _,_ ˆ **g** _[a]_ ) _,_ `sim` (ˆ **g** _[v]_ _,_ ˆ **g** _[a]_ ) for retrieval across text-video, text-audio, and videoaudio settings.


**4** **Experiments**


**4.1** **Experimental Setup**


**Datasets.** HowTo100M [ 61 ] consists of 136M video clips from 1.22M YouTube videos with 134,472
hours. AudioSet [ 69 ] contains 2,084,320 clips with 632 classes from YouTube videos covering
human and animal sounds, musical instruments, and common everyday environmental sounds. An
intersection of HowTo100M and AudioSet with 0.9M audio-video-text triplets is used for pre-training.
MSR-VTT [ 70 ] includes 10K YouTube videos with 200K description sentences and is split into 9K


6


Table 2: **Quantitative results of text-to-video retrieval on MSR-VTT dataset.** “Single”, “Dual”,
and “Triple” refer to one joint encoder, two separate, and three separate encoders. Bold and Underline
denote the best and second result, respectively.


**Method** **Modalities** **Architecture** **Pre-train Datasets** **Data Size (** _↓_ **)** **R@1 (** _↑_ **)** **R@5 (** _↑_ **)** **R@10 (** _↑_ **)**


JSFusion [60] Video, Text Dual AudioSet+ImageNet 3M 10.2 31.2 43.2
MEE [41] Video, Text, Audio Triple COCO+VisGenome 5.6M 14.2 39.2 53.8
HT [61] Video, Text Dual HowTo100M 136M 14.9 40.2 52.8
CE [28] Video, Text, Audio Triple YouTube-8M 8M 20.9 48.8 62.4
AVLnet [62] Video, Text, Audio Triple HowTo100M 136M **27.1** 55.6 66.6
ActBERT [30] Video, Text Dual HowTo100M 136M 16.3 42.8 56.9
HERO [63] Video, Text Dual HowTo100M 136M 16.8 43.4 57.7
VidTranslate [64] Video, Text Dual HowTo100M 136M 14.7  - 52.8
NoiseEstimation [65] Video, Text Dual HowTo100M 136M 17.4 41.6 53.6
UniVL [66] Video, Text Dual HowTo100M 136M 21.2 49.6 63.1
MMT [29] Video, Text, Audio Triple HowTo100M 136M 26.6 57.1 **69.6**
ClipBERT [52] Video, Text Dual COCO+VisGenome 5.6M 22.0 46.8 59.9
Frozen [32] Video, Text Single CC3M 3M 25.5 54.5 66.1
TVLT [58] Video, Audio Single HowTo100M 136M 22.0  -  Everything-At-Once [67] Video, Text, Audio Triple HowTo100M 136M 23.7 52.1 63.7
AllinOne [35] Video, Text Single AudioSet _∩_ HowTo100M **0.9M** 22.1 49.1 60.6
VLSA (ours) Video, Text, Audio Single AudioSet _∩_ HowTo100M **0.9M** **27.1** **57.3** 68.9
_zero-shot:_

HT [61] Video, Text Dual HowTo100M 136M 7.5 21.2 29.6
SupportSet [68] Video, Text Dual HowTo100M 136M 8.7 23.0 31.1
VATT [38] Video, Text, Audio Single AudioSet+HowTo100M 138M – – 29.7
Frozen [32] Video, Text Dual CC3M+WebVid-2M 5.5M 18.7 39.5 51.6
OA-Trans [34] Video, Text Dual WebVid-2M 2.5M 18.4 36.5 46.8
Everything-At-Once [67] Video, Text, Audio Triple HowTo100M 136M 9.9 24.0 32.6
VLSA (ours) Video, Text, Audio Single AudioSet _∩_ HowTo100M **0.9M** **20.4** **40.6** **53.8**


for training and 1K for testing. LSMDC [ 71 ] contains 118,081 video clips with 7,408 and 1,000
videos for validation and testing. AudioCaps [ 72 ] is filtered out for 49,291 clips for training, 428
clips for validation, and 816 clips for testing. SoundDescs [ 55 ] consists of 32,979 audio clips with 23
sound categories, and is divided into 70% of the clips for training and 15% each for validation and
testing. Yoocook2 [ 73 ] comprises 13K video clips of 89 cooking recipes with 9,586 clips for training
and 3,350 clips for validation.


**Evaluation Metrics.** We use the standard retrieval metrics [ 30, 52, 64, 65, 38 ] to evaluate the
performance of our model. Recall at rank _k_ (R@ _k_ ) measures the percentage of labels retrieved within
the top _k_ ranked predictions, and the higher value is better. _k_ = 1, 5, 10 for text-video retrieval, and _k_
= 1, 5, 10, 50 for text-audio retrieval. For all metrics, we report the average result of three different
random seeds.


**Implementation.** For each audio waveform, we follow the prior work [ 74 ] and sub-sample the audio
signal to 11kHz. A Short-time Fourier transform with a window size of 1022 and a hop length of
256 is further applied to generate a 512 _×_ 256 Time-Frequency representation of the audio, which
is resampled to a log-frequency scale with a size of 256 _×_ 256 as the input audio spectrogram, _i.e._,
_T_ = 256 _, F_ = 256 . For each caption, the tokenizer embedding from BERT [ 59 ] is used as the
input with a maximum sentence length of 40 and a vocab size of 30,522, _i.e._, _C_ = 30 _,_ 522 . For
each video clip, we randomly sample 8 frames as inputs and resize each frame to 224 _×_ 224, _i.e._,
_V_ = 8 _, H_ = _W_ = 224 . Following prior work [ 75, 76 ], we apply a patch size of 16 _×_ 16 for both
audio and video frames. With patch size _P_ _[v]_ = 16 _, P_ _[a]_ = 16, the total number of visual and audio
patches _I_ = 196 _, A_ = 256 . We use a ViT-base [ 77 ] model for the masked autoencoder same as in
MAE [ 75 ]. We follow previous approaches [ 59, 75 ] to mask 15% on each word token randomly, 75%
on patches of each frame independently, and 75% on patches of audio spectrograms. The model is
trained for 200k steps with the AdamW [ 78 ] optimizer with a learning rate of 1e-4, a decay rate of
0.01, and a batch size of 2048. For fine-tuning, the model is trained for 20 epochs with a batch size of
256.


**4.2** **Comparison to Prior Work**


In this work, we propose a novel and effective framework for video-language pre-training with
synchronized audio. In order to validate the effectiveness of the proposed VLSA, we comprehensively
compare it to previous video-text (VT), video-audio (VA), and video-text-audio (VTA) pre-training
baselines: i) **VT:** 1) JSFusion [ 60 ]: a very early sequence fusion model with multi-modal matching;
2) MEE [ 41 ]: a baseline with mixed embedding experts for visual-textual modalities; 3) HT [ 61 ]:


7


Table 3: **Quantitative results of text-audio retrieval on SoundDescs dataset.** Best results are bold.


**Text-to-Audio** **Audio-to-Text**
**Method** **Pre-train Datasets** **Data Size (** _↓_ **)**
**R@1 (** _↑_ **)** **R@5 (** _↑_ **)** **R@10 (** _↑_ **)** **R@50 (** _↑_ **)** **R@1 (** _↑_ **)** **R@5 (** _↑_ **)** **R@10 (** _↑_ **)** **R@50 (** _↑_ **)**


MEE [41] YouTube-8M+VGGSound 8.2M 30.8 60.8 70.9 85.9 30.9 60.3 70.1 85.3
CE [28] YouTube-8M+VGGSound 8.2M 31.1 60.6 70.8 86.0 30.8 60.3 69.5 85.4
MMT [29] YouTube-8M+VGGSound 8.2M 30.7 61.8 72.2 88.8 31.4 63.2 73.4 89.0
VLSA (ours) AudioSet _∩_ HowTo100M **0.9M** **33.5** **63.7** **75.1** **91.6** **34.2** **65.9** **76.3** **92.1**


Table 4: **Quantitative results on LSMDC, AudioCaps, and Youcook2 benchmarks.**



**Method** **R@1**


Frozen [32] 15.0
OA-Trans [34] 18.2


(a) Text-to-Video.



**Method** **R@1**


MEE [41] 26.6
MMT [29] 39.6

(b) Text-to-Audio.



**Method** **R@1**


AVLnet [62] 30.7
TVLT [58] 32.8


(c) Audio-to-Video.



a simple baseline with two uni-modal encoders by learning text-video joint embedding; 4) ActBERT [ 30 ]: a tangled transformer with the global-local correlation between actions and object
regions; 5) HERO [ 63 ]: a hierarchical architecture combined with cross-modal and temporal transformer; 6) VidTranslate [ 64 ]: a generative method with a translation objective between modalities;
7) NoiseEstimation [ 65 ]: a multi-modal density estimation method for learning the cross-modal
correspondence; 8) UniVL [ 66 ]: a unified pre-training model for video-language understanding
and generation; 9) MMT [ 29 ]: a multi-modal transformer to aggregate per-frame visual features
with temporal information; 10) ClipBERT [ 52 ]: a generic framework based on BERT with sparsely
sampled short video clips for pre-training; 11) Frozen [ 32 ]: a curriculum learning-based joint transformer with attention modeling on both space and time; 12) SupportSet [ 68 ]: a generative method by
recovering caption with a weighted combination of support visual features; 13) OA-Trans [ 34 ]: an
object-aware pre-training transformer with bounding boxes and tags as guidance; 14) AllinOne [ 35 ]:
a unified transformer that learns joint representations from raw video and textual inputs. ii) **VA:** 15)
TVLT [ 58 ]: a very recent visual-audio pre-training framework with masked audio/video autoencoding
and contrastive modeling to align video and audio. iii) **VTA:** 16) CE [ 28 ]: a collaborative model
with multiple pre-trained experts on each modality. 17) AVLnet [ 62 ]: a self-supervised baseline that
learns a shared audio-visual-textual embedding space directly from raw video and text signals; 18)
VATT [ 38 ]: a unified transformer with the multi-modal contrastive loss on global features of each
modality for learning the alignment of video-audio-text triplets.


For text-video retrieval, we report the quantitative comparisons of fine-tuning and zero-shot results
in Table 2. As can be seen, we achieve the best results in terms of R@1 and R@5 compared to
all baselines fine-tuned on MSR-VTT. In particular, the proposed VLSA significantly outperforms
TVLT [ 58 ], the only video-audio pre-training approach with a single transformer, where we achieve
the performance gains of 5.1 R@1. In the meanwhile, we only need 0.9M data for pre-training to
achieve competitive results compared to the performance (32.7 R@1, 60.9 R@5, and 72.5 R@10)
of OA-Trans [ 34 ], which reduces 64% of the least amount (2.5M) of pre-training data so far. When
pre-trained on the same amount of data, the proposed VLSA achieves performance gains of 5.0 R@1,
8.2 R@5, and 8.3 R@10, compared to AllinOne [ 35 ], the unified video-language pre-training model.
These improvements demonstrate the effectiveness of our method in text-video retrieval by enhancing
video-language pre-training with synchronized audio.


When compared to Frozen [ 34 ], the current state-of-the-art model pre-trained on 5.5M data, the
proposed VLSA achieves zero-shot results gains of 1.7 R@1, 1.1 R@5, and 2.2 R@10. Furthermore,
our VLSA outperforms VATT [ 38 ] by 24.1 R@10, which implies the importance of incorporating
audio into local-patch masked modeling to learn discriminative modality-aware representations.
Meanwhile, the proposed approach with a unified transformer pre-trained on 0.9M video-audio-text
triplets still achieves this significant gain, compared to VATT [ 38 ] pre-trained on 136M triplets and
2M video-audio pairs. These results validate the superiority of our method in learning compact
cross-modal embeddings for zero-shot text-video retrieval.


In addition, significant gains in text-audio retrieval can be observed in Table 3. The proposed VLSA
achieves the best performance in terms of all metrics for both text-to-audio and audio-to-text retrieval.
When it comes to text-to-audio retrieval, our approach with a single transformer encoder obviously
outperforms MMT with separate encoders pre-trained on 136M video-audio-text triplets by 2.8 R@1,


8


Table 5: **Ablation studies on Local-Patch Masked Modeling (LPMM) and Global Audio Match-**
**ing (GAM).**


**Text-to-Video** **Text-to-Audio**
**LPMM** **GAM**
**R@1** **R@5** **R@10** **R@1** **R@5** **R@10**


✗ ✗ 20.1 47.2 59.3 23.6 49.2 61.5

✓ ✗ 22.6 49.7 61.2 28.5 53.6 66.3

✗ ✓ 25.3 52.5 63.1 30.7 57.8 69.7

✓ ✓ **27.1** **57.3** **68.9** **33.5** **63.7** **75.1**


1.9 R@5, 2.9 R@10, and 2.8 R@50. This further indicates the effectiveness of the proposed unified
framework in learning discriminative audio and textual representations. Results on more benchmarks
are reported in Table 4.


**4.3** **Experimental Analysis**


In this part, we performed ablation studies to demonstrate the benefit of introducing the Local-Patch
Masked Modeling and Global Audio Matching modules. In order to validate the effectiveness of
local-patch masked modeling (LPMM) and global audio matching (GAM), we ablate the necessity of
each module and report the quantitative results in Table 5. We can observe that adding LPMM highly
improves the vanilla baseline without pre-training in terms of text-video (by 2.5 R@1, 2.5 R@5,
and 1.9 R@10) and text-audio retrieval (by 1.9 R@1, 4.4 R@5, and 4.8 R10), which implies the
benefit of LPMM in learning discriminative modality-aware embeddings. Meanwhile, introducing
only GAM in the baseline also increases the retrieval results by significant gains (5.2 R@1, 5.3 R@5,
and 3.8 R@10 for text-video; 7.1 R@1, 8.6 R@5, and 8.2 R10 for text-audio). More importantly,
incorporating LPMM and GAM together highly raises the baseline by 7.0 R@1, 10.1 R@5, 9.6
R@10 for text-video, and 9.9 R@1, 14.5 R@5, 13.6 R10 for text-audio. These results demonstrate
the importance of local-patch masked modeling and global audio matching in extracting compact
semantics from video-audio-text triplets.


In Appendix E, we also conducted extensive experiments to explore the joint encoder/decoder, and
compare the video-language embedding space learned by Global Audio Matching and Video-Text
Matching, separately. These results demonstrate the effectiveness of incorporating synchronized
audio into video-language pre-training in capturing more discriminative representations for both
text-video and text-audio retrieval. Furthermore, the representations extracted by the proposed GAM
in our VLSA are inter-modality compact for matching pairs, while inter-modality separable and
intra-modality compact for non-matching pairs.


**5** **Conclusion**


In this work, we present VLSA, a novel and enhanced framework for video-language pre-training
with synchronized audio that can jointly learn compact representations for video-audio-text triplets in
a unified self-supervised transformer. We introduce local-patch masked modeling to learn modalityaware local features from a joint transformer encoder. Then, we leverage the joint encoder with
global-token alignment to capture discriminative global features. Empirical experiments on five
comprehensive cross-modal retrieval benchmarks demonstrate the significant advantage of our VLSA
against previous video-language pre-training approaches. Our simple model pre-trained on only 0.9M
data achieves competitive results on retrieval across text, video, and audio. Furthermore, qualitative
visualizations vividly showcase the advantage of our VLSA in learning compact visual-textual
representations.


**Broader Impact.** The proposed approach pre-trains representations of video-audio-text triplets
from manually collected video datasets, which might cause the model to learn internal biases in the
data. For instance, the model could fail to learn the correspondence between rare and noisy sounds.
Therefore, these issues should be addressed for the deployment of real applications.


9


**References**


[1] Shentong Mo and Pedro Morgado. Benchmarking weakly-supervised audio-visual sound
localization. In _European Conference on Computer Vision (ECCV) Workshop_, 2022. 1


[2] Shentong Mo and Yapeng Tian. Semantic-aware multi-modal grouping for weakly-supervised
audio-visual video parsing. In _European Conference on Computer Vision (ECCV) Workshop_,
2022. 1


[3] Shentong Mo, Jing Shi, and Yapeng Tian. DiffAVA: Personalized text-to-audio generation with
visual alignment. _arXiv preprint arXiv:2305.12903_, 2023. 1


[4] Shentong Mo, Weiguo Pian, and Yapeng Tian. Class-incremental grouping network for continual
audio-visual learning. In _Proceedings of the IEEE/CVF International Conference on Computer_
_Vision (ICCV)_, 2023. 1


[5] Weiguo Pian, Shentong Mo, Yunhui Guo, and Yapeng Tian. Audio-visual class-incremental
learning. In _IEEE/CVF International Conference on Computer Vision (ICCV)_, 2023. 1


[6] Shentong Mo and Pedro Morgado. A unified audio-visual learning framework for localization,
separation, and recognition. _arXiv preprint arXiv:2305.19458_, 2023. 1


[7] Shentong Mo, Jing Shi, and Yapeng Tian. Text-to-audio generation synchronized with videos.
_arXiv preprint arXiv:2403.07938_, 2024. 1


[8] Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chenliang Xu. Audio-visual event
localization in unconstrained videos. In _Proceedings of European Conference on Computer_
_Vision (ECCV)_, 2018. 1


[9] Yan-Bo Lin, Yu-Jhe Li, and Yu-Chiang Frank Wang. Dual-modality seq2seq network for
audio-visual event localization. In _IEEE International Conference on Acoustics, Speech and_
_Signal Processing (ICASSP)_, pages 2002–2006, 2019. 1


[10] Yu Wu, Linchao Zhu, Yan Yan, and Yi Yang. Dual attention matching for audio-visual event
localization. In _Proceedings of the IEEE International Conference on Computer Vision (ICCV)_,
pages 6291–6299, 2019. 1


[11] Yan-Bo Lin and Yu-Chiang Frank Wang. Audiovisual transformer with instance attention for
audio-visual event localization. In _Proceedings of the Asian Conference on Computer Vision_
_(ACCV)_, 2020. 1


[12] Shentong Mo and Yapeng Tian. Multi-modal grouping network for weakly-supervised audiovisual video parsing. In _Proceedings of Advances in Neural Information Processing Systems_
_(NeurIPS)_, 2022. 1


[13] Shentong Mo and Pedro Morgado. Unveiling the power of audio-visual early fusion transformers
with dense interactions through masked modeling. _arXiv preprint arXiv:2312.01017_, 2023. 1


[14] Pedro Morgado, Nuno Nvasconcelos, Timothy Langlois, and Oliver Wang. Self-supervised
generation of spatial audio for 360°video. In _Proceedings of Advances in Neural Information_
_Processing Systems (NeurIPS)_, 2018. 1


[15] Ruohan Gao and Kristen Grauman. 2.5d visual sound. In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition (CVPR)_, pages 324–333, 2019. 1


[16] Changan Chen, Unnat Jain, Carl Schissler, S. V. A. Garí, Ziad Al-Halah, Vamsi Krishna
Ithapu, Philip Robinson, and Kristen Grauman. Soundspaces: Audio-visual navigation in 3d
environments. In _Proceedings of European Conference on Computer Vision (ECCV)_, pages
17–36, 2020. 1


[17] Pedro Morgado, Yi Li, and Nuno Nvasconcelos. Learning representations from audio-visual
spatial alignment. In _Proceedings of Advances in Neural Information Processing Systems_
_(NeurIPS)_, pages 4733–4744, 2020. 1


[18] Arda Senocak, Tae-Hyun Oh, Junsik Kim, Ming-Hsuan Yang, and In So Kweon. Learning to
localize sound source in visual scenes. In _Proceedings of the IEEE Conference on Computer_
_Vision and Pattern Recognition (CVPR)_, pages 4358–4366, 2018. 1


[19] Di Hu, Feiping Nie, and Xuelong Li. Deep multimodal clustering for unsupervised audiovisual
learning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_
_(CVPR)_, pages 9248–9257, 2019. 1


10


[20] Triantafyllos Afouras, Andrew Owens, Joon Son Chung, and Andrew Zisserman. Selfsupervised learning of audio-visual objects from video. In _Proceedings of European Conference_
_on Computer Vision (ECCV)_, pages 208–224, 2020. 1


[21] Rui Qian, Di Hu, Heinrich Dinkel, Mengyue Wu, Ning Xu, and Weiyao Lin. Multiple sound
sources localization from coarse to fine. In _Proceedings of European Conference on Computer_
_Vision (ECCV)_, pages 292–308, 2020. 1


[22] Honglie Chen, Weidi Xie, Triantafyllos Afouras, Arsha Nagrani, Andrea Vedaldi, and Andrew
Zisserman. Localizing visual sounds the hard way. In _Proceedings of the IEEE/CVF Conference_
_on Computer Vision and Pattern Recognition (CVPR)_, pages 16867–16876, 2021. 1


[23] Shentong Mo and Pedro Morgado. Localizing visual sounds the easy way. _arXiv preprint_
_arXiv:2203.09324_, 2022. 1


[24] Shentong Mo and Pedro Morgado. A closer look at weakly-supervised audio-visual source
localization. In _Proceedings of Advances in Neural Information Processing Systems (NeurIPS)_,
2022. 1


[25] Shentong Mo and Yapeng Tian. Audio-visual grouping network for sound localization from
mixtures. _arXiv preprint arXiv:2303.17056_, 2023. 1


[26] Shentong Mo and Yapeng Tian. AV-SAM: Segment anything model meets audio-visual localization and segmentation. _arXiv preprint arXiv:2305.01836_, 2023. 1


[27] Shentong Mo and Bhiksha Raj. Weakly-supervised audio-visual segmentation. In _Proceedings_
_of Advances in Neural Information Processing Systems (NeurIPS)_, 2023. 1


[28] Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. Use what you have: Video
retrieval using representations from collaborative experts. In _Proceedings of British Machine_
_Vision Conference (BMVC)_, 2019. 1, 3, 7, 8, 15, 16


[29] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. Multi-modal Transformer
for Video Retrieval. In _Proceedings of European Conference on Computer Vision (ECCV)_, 2020.
1, 3, 7, 8, 15


[30] Linchao Zhu and Yi Yang. Actbert: Learning global-local video-text representations. In
_Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_,
pages 8743–8752, 2020. 1, 3, 7, 8, 15


[31] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. VideoBERT: A
joint model for video and language representation learning. _arXiv preprint arXiv:1904.01766_,
2019. 1, 3


[32] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video
and image encoder for end-to-end retrieval. In _Proceedings of IEEE International Conference_
_on Computer Vision (ICCV)_, 2021. 1, 3, 4, 7, 8, 15, 17


[33] Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, and Dong Shen. Improving videotext retrieval by multi-stream corpus alignment and dual softmax loss. _arXiv preprint_
_arXiv:2109.04290_, 2021. 1, 3


[34] Alex Jinpeng Wang, Yixiao Ge, Guanyu Cai, Rui Yan, Xudong Lin, Ying Shan, Xiaohu Qie,
and Mike Zheng Shou. Object-aware video-language pre-training for retrieval. In _Proceedings_
_of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, 2022. 1, 3,
4, 7, 8, 15, 17, 18


[35] Alex Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Xudong Lin, Guanyu Cai, Jianping Wu,
Ying Shan, Xiaohu Qie, and Mike Zheng Shou. All in one: Exploring unified video-language
pre-training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition (CVPR)_, 2023. 1, 3, 4, 7, 8, 15


[36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya
Sutskever. Learning transferable visual models from natural language supervision. _arXiv_
_preprint arXiv:2103.00020_, 2021. 1, 3, 18


[37] Zijian Gao, Jingyun Liu, Sheng Chen, Dedan Chang, Hao Zhang, and Jinwei Yuan. Clip2tv:
An empirical study on transformer-based methods for video-text retrieval. _arXiv preprint_
_arXiv:2111.05610_, 2021. 1, 3


11


[38] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and
Boqing Gong. VATT: Transformers for multimodal self-supervised learning from raw video,
audio and text. In _Proceedings of Advances in Neural Information Processing Systems (NeurIPS)_,
2021. 2, 3, 7, 8, 15, 16


[39] Andrey Guzhov, Federico Raue, Jörn Hees, and Andreas Dengel. Audioclip: Extending clip to
image, text and audio. _arXiv preprint arXiv:2106.13043_, 2021. 2, 3


[40] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi,
Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot reserve: Neural script
knowledge through vision and language and sound. In _Proceedings of the IEEE/CVF Conference_
_on Computer Vision and Pattern Recognition (CVPR)_, pages 16354–16366, 2022. 2, 3


[41] Antoine Miech, Ivan Laptev, and Josef Sivic. Learning a text-video embedding from incomplete
and heterogeneous data. _arXiv preprint arXiv:1804.02516_, 2018. 3, 7, 8, 15


[42] Di Qi, Lin Su, Jia Song, Edward Cui, Taroon Bharti, and Arun Sacheti. ImageBERT: cross-modal
pre-training with large-scale weak-supervised image-text data. _arXiv preprint arXiv:2001.07966_,
2020. 3


[43] Dandan Song, Siyi Ma, Zhanchen Sun, Sicheng Yang, and Lejian Liao. KVL-BERT: knowledge
enhanced visual-and-linguistic BERT for visual commonsense reasoning. _arXiv preprint_
_arXiv:2012.07000_, 2020. 3


[44] Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. Ernie-vil:
Knowledge enhanced vision-language representations through scene graph. _arXiv preprint_
_arXiv:2006.16934_, 2020. 3


[45] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng,
and Jingjing Liu. UNITER: learning universal image-text representations. _arXiv preprint_
_arXiv:1909.11740_, 2019. 3


[46] Hao Tan and Mohit Bansal. LXMERT: Learning cross-modality encoder representations from
transformers. _arXiv preprint arXiv:1908.07490_, 2019. 3


[47] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A
simple and performant baseline for vision and language. _arXiv preprint arXiv:1908.03557_,
2019. 3, 4


[48] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic
visiolinguistic representations for vision-and-language tasks. In _Advances in Neural Information_
_Processing Systems_, pages 13–23, 2019. 3, 4


[49] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. VL-BERT: Pretraining of generic visual-linguistic representations. In _International Conference on Learning_
_Representations_, 2020. 3, 4


[50] Lei Shi, Kai Shuang, Shijie Geng, Peng Su, Zhengkai Jiang, Peng Gao, Zuohui Fu, Gerard
de Melo, and Sen Su. Contrastive visual-linguistic pretraining. _arXiv preprint arXiv:2007.13135_,
2020. 3, 4


[51] Yicong Hong, Qi Wu, Yuankai Qi, Cristian Rodriguez-Opazo, and Stephen Gould. VLN BERT:
a recurrent vision-and-language bert for navigation. In _Proceedings of IEEE/CVF Conference_
_on Computer Vision and Pattern Recognition (CVPR)_, 2021. 3, 4


[52] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit Bansal, and Jingjing Liu.
Less is More: clipbert for video-and-language learning via sparse sampling. In _Proceedings of_
_IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, 2021. 3, 4, 7, 8,
15


[53] Yuying Ge, Yixiao Ge, Xihui Liu, Alex Jinpeng Wang, Jianping Wu, Ying Shan, Xiaohu Qie,
and Ping Luo. Miles: Visual bert pre-training with injected language semantics for video-text
retrieval. In _Proceedings of European Conference on Computer Vision (ECCV)_, page 691–708,
2022. 3


[54] A.-M. Oncescu, A.S. Koepke, J. Henriques, Z. Akata, and S. Albanie. Audio retrieval with
natural language queries. In _Proceedings of Interspeech_, 2021. 3


[55] A.S. Koepke, A.-M. Oncescu, J. Henriques, Z. Akata, and S. Albanie. Audio retrieval with
natural language queries: A benchmark study. _IEEE Transactions on Multimedia_, 2022. 3, 7


12


[56] Rui Yan, Mike Zheng Shou, Yixiao Ge, Alex Jinpeng Wang, Xudong Lin, Guanyu Cai, and
Jinhui Tang. Video-text pre-training with learned regions. _arXiv preprint arXiv:2112.01194_,
2021. 3

[57] Ziyi Yang, Yuwei Fang, Chenguang Zhu, Reid Pryzant, Dongdong Chen, Yu Shi, Yichong Xu,
Yao Qian, Mei Gao, Yi-Ling Chen, Liyang Lu, Yujia Xie, Robert Gmyr, Noel C. F. Codella,
Naoyuki Kanda, Bin Xiao, Yuanxun Lu, Takuya Yoshioka, Michael Zeng, and Xuedong
Huang. i-code: An integrative and composable multimodal learning framework. _arXiv preprint_
_arXiv:2205.01818_, 2022. 3

[58] Zineng Tang, Jaemin Cho, Yixin Nie, and Mohit Bansal. Tvlt: Textless vision-language
transformer. In _Proceedings of Advances in Neural Information Processing Systems (NeurIPS)_,
2022. 4, 7, 8, 15, 17

[59] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of
deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_,
2018. 6, 7

[60] Youngjae Yu, Jongseok Kim, and Gunhee Kim. A joint sequence fusion model for video
question answering and retrieval. In _Proceedings of European Conference on Computer Vision_
_(ECCV)_, pages 471–487, 2018. 7, 15

[61] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and
Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million
narrated video clips. In _Proceedings of the IEEE/CVF International Conference on Computer_
_Vision (ICCV)_, pages 2630–2640, 2019. 6, 7, 15, 16

[62] Andrew Rouditchenko, Angie Boggust, David Harwath, Brian Chen, Dhiraj Joshi, Samuel
Thomas, Kartik Audhkhasi, Hilde Kuehne, Rameswar Panda, Rogerio Feris, et al. Avlnet:
Learning audio-visual language representations from instructional videos. _arXiv preprint_
_arXiv:2006.09199_, 2020. 7, 8, 15, 16, 17

[63] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. HERO:
Hierarchical encoder for Video+Language omni-representation pre-training. In _Proceedings of_
_the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_, pages
2046–2065, 2020. 7, 8, 15

[64] Bruno Korbar, Fabio Petroni, Rohit Girdhar, and Lorenzo Torresani. Video understanding as
machine translation. _arXiv preprint arXiv:2006.07203_, 2020. 7, 8, 15

[65] Elad Amrani, Rami Ben-Ari, Daniel Rotman, and Alexander M. Bronstein. Noise estimation using density estimation for self-supervised multimodal learning. _arXiv preprint_
_arXiv:2003.03186_, 2020. 7, 8, 15

[66] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan Duan, Tianrui Li, Jason Li, Taroon
Bharti, and Ming Zhou. Univl: A unified video and language pre-training model for multimodal
understanding and generation. _arXiv preprint arXiv:2002.06353_, 2020. 7, 8, 15

[67] Nina Shvetsova, Brian Chen, Andrew Rouditchenko, Samuel Thomas, Brian Kingsbury, Rogerio S Feris, David Harwath, James Glass, and Hilde Kuehne. Everything at once-multi-modal
fusion transformer for video retrieval. In _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition (CVPR)_, pages 20020–20029, 2022. 7, 15, 16, 17

[68] Mandela Patrick, Po-Yao Huang, Yuki Asano, Florian Metze, Alexander G Hauptmann, Joao F.
Henriques, and Andrea Vedaldi. Support-set bottlenecks for video-text representation learning.
In _Proceedings of International Conference on Learning Representations (ICLR)_, 2021. 7, 8, 15

[69] Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled
dataset for audio events. In _Proceedings of IEEE International Conference on Acoustics, Speech_
_and Signal Processing (ICASSP)_, 2017. 6, 16

[70] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for
bridging video and language. In _Proceedings of IEEE Conference on Computer Vision and_
_Pattern Recognition (CVPR)_, pages 5288–5296, 2016. 6

[71] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie
description. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_
_(CVPR)_, pages 3202–3212, June 2015. 7


13


[72] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. AudioCaps: Generating captions for audios in the wild. In _Proceedings of the 2019 Conference of the North American_
_Chapter of the Association for Computational Linguistics: Human Language Technologies,_
_Volume 1 (Long and Short Papers)_, pages 119–132, 2019. 7

[73] Luowei Zhou, Chenliang Xu, and Jason J. Corso. Towards automatic learning of procedures
from web instructional videos. In _Proceedings of the Thirty-Second AAAI Conference on_
_Artificial Intelligence_, page 7590–7598, 2018. 7

[74] Hang Zhao, Chuang Gan, Andrew Rouditchenko, Carl Vondrick, Josh McDermott, and Antonio
Torralba. The sound of pixels. In _Proceedings of the European Conference on Computer Vision_
_(ECCV)_, pages 570–586, 2018. 7

[75] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross B. Girshick. Masked
autoencoders are scalable vision learners. _arXiv preprint arXiv:2111.06377_, 2021. 7

[76] Alan Baade, Puyuan Peng, and David F. Harwath. MAE-AST: masked autoencoding audio
spectrogram transformer. _arXiv preprint arXiv:2203.16691_, 2022. 7

[77] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.
An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint_
_arXiv:2010.11929_, 2020. 7

[78] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _Proceedings of_
_International Conference on Learning Representations (ICLR)_, 2019. 7

[79] Jean-Baptiste Alayrac, Adrià Recasens, Rosalia Schneider, Relja Arandjelovi´c, Jason Ramapuram, Jeffrey De Fauw, Lucas Smaira, Sander Dieleman, and Andrew Zisserman. Self-Supervised
MultiModal Versatile Networks. In _Proceedings of Advances in Neural Information Processing_
_Systems (NeurIPS)_, 2020. 15, 16, 17

[80] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. _Journal of Machine_
_Learning Research_, 9(86):2579–2605, 2008. 18


14


**Appendix**


In this appendix, we provide significant differences between our VLSA and current video-text-audio
pre-training baselines. In addition, we compare the proposed VLSA with those baselines in terms
of text-to-video retrieval on fine-tuned and zero-shot settings. Finally, we report the quantitative
comparison of computation costs with state-of-the-art methods.


**A** **Video-Text-Audio Pre-training Baselines**


We conduct a comprehensive experimental study of existing approaches with video-text-audio pretraining. Namely, we considered:


    - JSFusion [ 60 ] (2018’ECCV): a very early sequence fusion model with multi-modal matching;

    - MEE [ 41 ] (2018’arXiv): a baseline with mixed embedding experts for visual-textual modalities;

    - HT [ 61 ] (2019’ICCV): a simple baseline with two uni-modal encoders by learning text-video
joint embedding;

    - CE [ 28 ] (2019’BMVC): a simple baseline based on a collaborative model with multiple
pre-trained experts on each modality to learn a joint video-text embedding.

    - MMV [ 79 ] (2020’NeurIPS): a multimodal versatile network with a deflation process to
integrate text with audio-visual representations into a common embedding space.

    - AVLnet [ 62 ] (2021’Interspeech): a self-supervised baseline that learns a shared audio-visualtextual embedding space directly from raw video and text signals;

    - ActBERT [ 30 ] (2020’CVPR): a tangled transformer with the global-local correlation between actions and object regions;

    - HERO [ 63 ] (2020’EMNLP): a hierarchical architecture combined with cross-modal and
temporal transformer;

    - VidTranslate [ 64 ] (2020’arXiv): a generative method with a translation objective between
modalities;

    - UniVL [ 66 ] (2020’arXiv): a unified pre-training model for video-language understanding
and generation;

    - MMT [ 29 ] (2020’ECCV): a multi-modal transformer to aggregate per-frame visual features
with temporal information;

    - NoiseEstimation [ 65 ] (2021’AAAI): a multi-modal density estimation method for learning
the cross-modal correspondence;

    - ClipBERT [ 52 ] (2021’CVPR): a generic framework based on BERT with sparsely sampled
short video clips for pre-training;

    - Frozen [ 32 ] (2021’ICCV): a curriculum learning-based joint transformer with attention
modeling on both space and time;

    - SupportSet [ 68 ] (2021’ICLR): a generative method by recovering caption with a weighted
combination of support visual features;

    - VATT [ 38 ] (2021’NeurIPS): a unified transformer with the multi-modal contrastive loss on
global features of each modality for learning the alignment of video-audio-text triplets.

    - OA-Trans [ 34 ] (2022’CVPR): an object-aware pre-training transformer with bounding boxes
and tags as guidance;

    - TVLT [ 58 ] (2022’NeurIPS): a very recent visual-audio pre-training framework with masked
audio/video autoencoding and contrastive modeling to align video and audio.

    - Everything-At-Once [ 67 ] (2022’CVPR): a modality agnostic fusion transformer that integrates embeddings from three separate encoders into a fused representation in a joined
multi-modal embedding space;

    - AllinOne [ 35 ] (2023’CVPR): a unified transformer that learns joint representations from
raw video and textual inputs.


15


Table 6: **Quantitative results of text-to-video retrieval on MSR-VTT dataset.** “Single”, “Dual”,
and “Triple” refer to one joint encoder, two separate, and three separate encoders. Bold denotes the
best results.


**Method** **Modalities** **Architecture** **Pre-train Datasets** **Data Size (** _↓_ **)** **R@1 (** _↑_ **)** **R@5 (** _↑_ **)** **R@10 (** _↑_ **)**


CE [28] Video, Text, Audio Triple YouTube-8M 8M 20.9 48.8 62.4
AVLnet [62] Video, Text, Audio Triple HowTo100M 136M **27.1** 55.6 66.6
Everything-At-Once [67] Video, Text, Audio Triple HowTo100M 136M 23.7 52.1 63.7
VLSA (ours) Video, Text, Audio Single AudioSet _∩_ HowTo100M **0.9M** **27.1** **57.3** **68.9**
_zero-shot:_
VATT [38] Video, Text, Audio Single AudioSet+HowTo100M 138M – – 29.7
MMV [79] Video, Text, Audio Triple AudioSet+HowTo100M 138M 9.3 23.0 31.1
Everything-At-Once [67] Video, Text, Audio Triple HowTo100M 136M 9.9 24.0 32.6
VLSA (ours) Video, Text, Audio Single AudioSet _∩_ HowTo100M **0.9M** **20.4** **40.6** **53.8**


**B** **Differences between VLSA and Video-Text-Audio Pre-training Baselines**


When compared to previous video-text-audio pre-training baselines, there are three significant
distinct characteristics of our VLSA for addressing video-language pre-training problems, which are
highlighted as follows:


1) **Achieving competitive performance with only 0.9M triplets for pre-training.** The major
difference is that we use only 0.9M video-text-audio triplets for pre-training to learn disentangled
video-text representations for retrieval. However, current video-text-audio pre-training approaches
used 8 _∼_ 138M triplets, and most methods utilized all data in the whole HowTo100M [ 61 ] benchmark.
Meanwhile, some strong baselines, such as VATT [ 38 ] and MMV [ 79 ], combined AudioSet [ 69 ] with
HowTo100M [61] to pre-train a total of 138M data, as shown in Table 6.


2) **Replacing Video-Text Matching with Global Audio Matching.** We introduce global audio
matching to replace video-text matching for extracting disentangled visual-textual representations
from video-text-audio triplets. However, all previous works mainly utilized video-text matching on
video frames and high-level textual semantics. Since most frames in a video look similar in high-level
textual semantics, they would bring false alignment pairs during pre-training. Different from them, we
leverage the global audio-video and audio-text matching to explicitly learn the cross-modal alignment
between synchronized audio and video frames/captions, as one audio spectrogram is distinct in each
audio-video/text pair.


3) **Incorporating Local-Patch Masked Modeling for all modalities.** We apply local-patch masked
modeling on each modality through a single unified encoder and three separate decoders with shared
parameters, but those video-text-audio pre-training baselines used the contrastive loss to learn the
global alignment across various modalities in the joint embedding space. They do not involve the
explicit masked modeling mechanism to learn local modality-aware features across three different
modalities. In addition, they extracted local modality-aware embeddings through three separate
encoders without shared parameters, except that VATT [ 38 ] introduced the unified encoder to extract
modality-agnostic embeddings from raw signals. However, VATT [ 38 ] performs worse on the
zero-shot text-to-video retrieval than our VLSA with local-patch masked modeling.


**C** **Comparison Results with Previous Video-Text-Audio Pre-training**
**Baselines**


Table 6 reports the comparison results with video-text-audio pre-training baselines on fine-tuned and
zero-shot text-to-video retrieval on the MSR-VTT dataset. We can observe that the proposed VLSA
achieves the best performance in terms of R@1, R@5, and R@10 compared to all baselines fine-tuned
on MSR-VTT. In particular, our VLSA significantly outperforms Everything-At-Once [ 67 ], the recent
video-text-audio pre-training approach with global alignment across three different modalities, where
the performance gains of 3.4 R@1, 5.2 R@5, and 5.2 R@10 are achieved. Meanwhile, the proposed
VLSA only needs 0.9M triplets for pre-training to achieve competitive results while they used the
whole HowTo100M [ 61 ] with 136M triplets. Compared to CE [ 28 ] pre-trained on the data with
the comparable size, we achieve performance gains of 6.2 R@1, 8.5 R@5, and 6.5 R@10, as they
involved neither global audio matching nor local-patch masked modeling. These improvements
validate the superiority of the proposed VLSA in enhancing video-language representations for
text-video retrieval by pre-training with synchronized audio.


16


Table 7: **Quantitative results of computation costs.** Lower is better.


**GPU** **Infer**
**Method** **Params**
**Hours** **Latency (ms)**


AVLnet [62] + text 353M – 2316
TVLT [58] + text 283M – 2135
Frozen [32] 232M 10580 1989
OA-Trans [34] 232M 7680 1946


In addition, significant gains in zero-shot text-audio retrieval can be observed. Compared to
Everything-At-Once [ 67 ], we achieve significant zero-shot performance gains of 10.5 R@1, 16.6
R@5, and 21.2 R@10, which implies the importance of introducing local-patch masked modeling
to learn discriminative modality-aware representations across different modalities. Meanwhile, the
proposed VLSA with a unified transformer pre-trained on 0.9M video-audio-text triplets significantly
outperforms MMV [ 79 ] pre-trained on 136M triplets and 2M video-audio pairs. These results further
demonstrate the effectiveness of our method with global audio matching in learning compact and
disentangled cross-modal embeddings for zero-shot text-video retrieval.


**D** **Computation Costs**


For computation costs, we used 8 V100-32GB GPUs for pre-training and fine-tuning. It should
be noted that the proposed LPMM & GAM modules in our VLSA take single pass cost during
pre-training such that we bring much fewer computation costs with only 0.9M video-audio-transcript
triplets. We report quantitative comparison results of parameters, pre-training GPU hours, and
inference latency in Table 7. As can be seen, we achieve the lowest parameters compared to previous
video-language pre-training baselines. In particular, the proposed VLSA significantly decreases the
parameters of OA-Trans [ 34 ], the current state-of-the-art method, by 76M. Moreover, we achieve
superior performance gains compared to TVLT [ 58 ], the current state-of-the-art video-audio pretraining baseline, which implies the importance of a joint transformer encoder in aggregating local
and global visual-textual representations.


Meanwhile, our VLSA outperforms strong video-language pre-training approaches, Frozen [ 32 ] and
OA-Trans [ 34 ], by large margins, where we achieve the pre-training cost reduction of 9230 GPU hours
and 6330 GPU hours. Furthermore, when evaluating the inference latency, the proposed approach still
outperforms OA-Trans [ 34 ] by 208 ms. We also achieve highly better results against TVLT [ 58 ], the
joint video-audio pre-training baseline. These significant cost reductions demonstrate the efficiency
of our method in pre-training a joint transformer encoder on only 0.9M video-audio-transcript triplets.


**E** **More Experimental Analysis**


In this section, we also conducted extensive experiments to explore the joint encoder/decoder, and
compare the video-language embedding space learned by Global Audio Matching and Video-Text
Matching, separately.


**Joint Encoder.** In order to explore the effect of
the modality types in a single joint encoder on
both text-video and text-audio performance, we
vary the combination types from _{_ A+V+T, V+T,
A+T, A+V, None _}_, where “None” denotes that
three modality-specific encoders are used. A,
V, and T denote audio, video, and text, respectively. The comparison results of the retrieval
performance are shown in Figure 3. As can be
seen, combining any two modalities among au
Figure 3: **Effect of modality types in a single**

dio, video, and text in a joint encoder increases **joint encoder.** A, V, and T denote audio, video,
the results of the vanilla baseline with modality
and text, respectively.

specific encoders, which implies the importance
of the proposed joint encoder in learning cross-modal representations for retrieval tasks. In addition,


17


adding audio to the V+T joint encoder significantly outperforms the baseline by 5.6 R@1, 7.6 R@5,
6.8R@10 for text-video, and 6.7 R@1, 9.9 R@5, 8.0R@10 for text-audio. These results demonstrate
the effectiveness of incorporating synchronized audio into video-language pre-training in capturing
more discriminative representations for both text-video and text-audio retrieval.


**Parameter-Shared Decoders.** In order to explore how the parameter-shared decoder affects
the final retrieval performance, we ablate the
modality types from _{_ A+V+T, V+T, A+T, A+V,
None _}_, where “None” denotes that three separate decoders do not share parameters. Figure 4
reports the comparison results on both text-video
and text-audio retrieval. We can observe that pretraining with parameter-shared decoders of any
two combined modalities outperforms three sep
Figure 4: **Effect of modality types with** arate decoders without parameters shared. In the
**parameter-shared decoder.** A, V, and T denote meanwhile, using three separate decoders with
audio, video, and text, respectively. parameters simultaneously shared among three

modalities achieves the best performance. This
further validates the rationality of local-patch masked modeling with three separate parameter-shared
decoders in enhancing video-language pre-training for retrieval tasks.


**Global Audio Matching vs. Video-Text Matching.** Learning discriminative video-language
semantic representations is essential for us to achieve higher performance in retrieval tasks. To
better evaluate the quality of video-language representations learned by global audio matching
(GAM) and video-text matching (VTM), we visualize learned visual-textual representations of 1000
randomly selected matching and non-matching pairs from MSR-VTT in a common space by tSNE [ 80 ], as shown in Figure 5. As can be seen in the last column, features extracted by the proposed
GAM are inter-modality compact for matching pairs. More importantly, GAM representations are
inter-modality separable and intra-modality compact for non-matching pairs. These meaningful
visualization results further demonstrate that our VLSA successfully extracts compact visual-textual
representations during pre-training. Note that VTM achieved 23.5 R@1, 50.6 R@5, and 61.8 R@10,
which is much lower than GAM (27.1 R@1, 57.3 R@5, 68.9 R@10).


**F** **Limitation**


Although the proposed VLSA achieves superior results on both text-video and text-audio retrieval,
the fine-tuning gains of our approach over R@1 and R@5 on text-video retrieval are not significant.
One possible solution is to leverage CLIP [ 36 ] pre-trained weights and fine-grained visual features for
masking (such as object bounding boxes and tags), similar to OA-Trans [ 34 ] for boosting performance.
Meanwhile, we notice that if we continue training for more steps, it would be hard to see significant
gains on downstream tasks. The primary cause is that we have a limited amount of video-audio-text
triplets for training. Therefore, the future work is potentially to gather more triplets or to explore
continual learning when it comes to new data.


18


Figure 5: **Qualitative comparisons of visual-textual representations learned by VTM and GAM**
**for matching (Top Row) and non-matching pairs (Bottom Row).** Note that each spot denotes the
visual/textual feature of one video/caption, and each color refers to one modality (yellow for video,
green for text). The VLSA representations are much better.


19


