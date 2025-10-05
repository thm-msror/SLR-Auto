## **Local-Global Video-Text Interactions for Temporal Grounding**

Jonghwan Mun [1] _[,]_ [2] Minsu Cho [1] Bohyung Han [2]

1 Computer Vision Lab., POSTECH, Korea
2 Computer Vision Lab., ASRI, Seoul National University, Korea
1 2
_{_ jonghwan.mun,mscho _}_ @postech.ac.kr bhhan@snu.ac.kr



**Abstract**


_This paper addresses the problem of text-to-video tem-_
_poral grounding, which aims to identify the time interval in_
_a video semantically relevant to a text query. We tackle this_
_problem using a novel regression-based model that learns_
_to extract a collection of mid-level features for semantic_
_phrases in a text query, which corresponds to important se-_
_mantic entities described in the query (e.g., actors, objects,_
_and actions), and reflect bi-modal interactions between the_
_linguistic features of the query and the visual features of_
_the video in multiple levels. The proposed method effec-_
_tively predicts the target time interval by exploiting contex-_
_tual information from local to global during bi-modal inter-_
_actions. Through in-depth ablation studies, we find out that_
_incorporating both local and global context in video and_
_text interactions is crucial to the accurate grounding. Our_
_experiment shows that the proposed method outperforms the_
_state of the arts on Charades-STA and ActivityNet Captions_
_datasets by large margins, 7.44% and 4.61% points at Re-_
_call@tIoU=0.5 metric, respectively. Code is available._ [1]


**1. Introduction**


As the amount of videos in the internet grows explosively, understanding and analyzing video contents ( _e.g_ ., action classification [3, 7, 28] and detection [18, 22, 26, 27,
31, 34, 36, 38]) becomes increasingly important. Furthermore, with the recent advances of deep learning on top of
large-scale datasets [2, 16, 17, 23], research on video content understanding is moving towards multi-modal problems ( _e.g_ ., video question answering [17, 23], video captioning [16, 24]) involving text, speech, and sound.
This paper addresses the problem of text-to-video temporal grounding, which aims to localize the time interval in
a video corresponding to the expression in a text query. Our
main idea is to extract multiple _semantic phrases_ from the
text query and align them with the video using _local and_


1 https://github.com/JonghwanMun/LGI4temporalgrounding



(c) Our approach


Video Segments Query Proposal Semantic Phrases


Figure 1. Video-to-text temporal grounding. (a) An example
where the target time interval (red box) consists of multiple parts
related to semantic phrases in a text query. (b) _Scan-and-localize_
framework that localizes the target time interval by comparing individual proposals with the whole semantics of the query. (c) Our
method that regresses the target time interval with the bi-modal
interactions in three levels between video segments and semantic
phrases identified from a query.


_global interactions_ between linguistic and visual features.
We define the semantic phrase as a sequence of words that
may describe a semantic entity such as an actor, an object,
an action, a place, etc. Fig. 1(a) shows an example of temporal grounding, where a text query consists of multiple semantic phrases corresponding to actors ( _i.e_ ., ‘the woman’)
and actions ( _i.e_ ., ‘mixed all ingredients’, ‘put it in a pan’,
‘put it in the oven’). This example indicates that a text query
can be effectively grounded onto a video by identifying relevant semantic phrases from the query and properly aligning
them with corresponding parts of the video.
Leveraging such semantic phrases of a text query, however, has never been explored in temporal grounding. Most
existing methods [1, 4, 5, 8, 9, 20, 32, 37] tackle the problem typically in the _scan-and-localize_ framework, which in



time


Query: the woman mixed all ingredients, put it in a pan and put it in the oven

(a) An example of text-to-video temporal grounding


Proposal Generation Proposal Selection by Matching


(b) Conventional approach



Segment-level
Modality Fusion



Local Context Global Context
Regression
Modeling Modeling


a nutshell compares a query with all candidate proposals of
time intervals and selects the one with the highest matching
score as shown in Fig. 1(b). During the matching procedure, they rely on a single global feature of the query rather
than finer-grained features in a phrase level, thus missing
important details for localization. Recent work [35] formulates the task as an attentive localization by regression and
attempts to extract semantic features from a query through
an attention scheme. However, it is still limited to identifying the most discriminative semantic phrase without understanding comprehensive context.
We propose a novel regression-based method for temporal grounding as depicted in Fig. 1(c), which performs
local-global video-text interactions for in-depth relationship
modeling between semantic phrases and video segments.
Contrary to the existing approaches, we first extract linguistic features for semantic phrases in a query using sequential query attention. Then, we perform video-text interaction in three levels to effectively align the semantic phrase
features with segment-level visual features of a video: 1)
segment-level fusion across the video segment and semantic phrase features, which highlights the segments associated with each semantic phrase, 2) local context modeling,
which helps align the phrases with temporal regions of variable lengths, and 3) global context modeling, which captures relations between phrases. Finally, we aggregate the
fused segment-level features using temporal attentive pooling and regress the time interval using the aggregated fea
ture.

The main contributions are summarized as follows:


_•_ We introduce a sequential query attention module that
extracts representations of multiple and distinct semantic phrases from a text query for the subsequent
video-text interaction.


_•_ We present an effective local-global video-text interaction algorithm that models the relationship between
video segments and semantic phrases in multiple levels, thus enhancing final localization by regression.


_•_ We conduct extensive experiments to validate the effectiveness of our method and show that it outperforms the state of the arts by a large margin on both
Charades-STA and ActivityNet Captions datasets.


**2. Related Work**


**2.1. Temporal Action Detection**


Recent temporal action detection methods often rely on
the state-of-the-art object detection and segmentation techniques in the image domain, and can be categorized into the
following three groups. First, some methods [22, 26] perform frame-level dense prediction and determine time intervals by pruning frames based on their confidence scores



and grouping adjacent ones. Second, proposal-based techniques [27, 31, 36, 38] extract all action proposals and refine their boundaries for action detection. Third, there exist some approaches [18, 34] based on single-shot detection
like SSD [21] for fast inference. In contrast to the action
detection task, which is limited to localizing a single action
instance, temporal grounding on a video by a text requires
to localize more complex intervals that would involve more
than two actions depending on the description in sentence
queries.


**2.2. Text-to-Video Temporal Grounding**


Since the release of two datasets for text-to-video temporal grounding, referred to as DiDeMo and Charades-STA,
various algorithms [1, 8, 9, 20, 37] have been proposed
within the _scan-and-localize_ framework, where candidate
clips are obtained by scanning a whole video based on sliding windows and the best matching clip with an input text
query is eventually selected. As the sliding window scheme
is time-consuming and often contains redundant candidate
clips, more effective and efficient methods [4, 5, 32] are
proposed as alternatives; a LSTM-based single-stream network [4] is proposed to perform frame-by-word interactions
and the clip proposal generation based methods [5, 32] are
proposed to reduce the number of redundant candidate clips.
Although those methods successfully enhance processing
time, they still need to observe full videos, thus, reinforcement learning is introduced to observe only a fraction of
frames [29] or a few clips [12] for temporal grounding.


On the other hand, proposal-free algorithms [10, 25, 35]
have also been proposed. Inspired by the recent advance in
text-based machine comprehension, Ghosh _et al_ . [10] propose to directly identify indices of video segments corresponding to start and end positions, and Opazo _et al_ . [25]
improve the method by adopting a query-guided dynamic
filter. Yuan _et al_ . [35] present a co-attention based location
regression algorithm, where the attention is learned to focus
on video segments within ground-truth time intervals.


ABLR [35] is the most similar to our algorithm in the
sense that it formulates the task as the attention-based lo
cation regression. However, our approach is different from
ABLR in the following two aspects. First, ABLR focuses
only on the most discriminative semantic phrase in a query
to acquire visual information, whereas we consider multiple
ones for more comprehensive estimation. Second, ABLR
relies on coarse interactions between video and text inputs
and often fails to capture fine-grained correlations between
video segments and query words. In contrast, we perform
a more effective multi-level video-text interaction to model

correlations between semantic phrases and video segments.


𝒇𝒇 **pos** **Lookup table** **E** **Embedding layer** **M** **MLP layer** ⊕ **Element-wise addition** ⨀ **Element-wise multiplication**



⊗ **[Matrix multiplication]**



Figure 2. Overall architecture of our algorithm. Given a video and a text query, we encode them to obtain segment-level visual features,
word-level and sentence-level textual features (Section 3.2). We extract a set of semantic phrase features from the query using the Sequential
Query Attention Network (SQAN) (Section 3.3). Then, we obtain semantics-aware segment features based on the extracted phrase features
via local-global video-text interactions (Section 3.4). Finally, we directly predict the time interval from the summarized video features using
the temporal attention (Section 3.5). We train the model using the regression loss and two additional attention-related losses (Section 3.6).



**3. Proposed Method**





Next, the Sequential Query Attention Network (SQAN)





precise localization in temporal grounding. To realize this
idea, we introduce a differentiable module _f_ e to represent
a query as a set of semantic phrases and incorporate localglobal video-text interactions for in-depth understanding of
the phrases within a video, which leads to a new objective
as follows:


_θ_ _[∗]_ = arg max E[log _p_ _θ_ ( _C|V, f_ e ( _Q_ ))] _._ (2)
_θ_


Fig. 2 illustrates the overall architecture of the proposed
method. We first compute segment-level visual features
combined with their embedded timestamps, and then derive word- and sentence-level features based on the query.



where _d_ denotes feature dimension.


**Video encoding** An untrimmed video is divided into a sequence of segments with a fixed length ( _e.g_ ., 16 frames),
where two adjacent segments overlap each other for a half
of their lengths. We extract the features from individual

_·_
segments using a 3D CNN module, denoted by _f_ v ( ), after the uniform sampling of _T_ segments, and feed the fea


LSTM is applied to the embedded word features. A wordlevel feature at the _l_ -th position is obtained by the concatenation of hidden states in both directions, which is given by
**w** _l_ = [ _[⃗]_ **h** _l_ ; **h** _l_ ] _∈_ R _[d]_, while a sentence-level feature **q** is pro


**w** _l_ = [ **h** _l_ ; **h** _l_ ] _∈_ R _[d]_, while a sentence-level feature **q** is pro
vided by the concatenation of the last hidden states in both
the forward and backward LSTMs, _i.e_ ., **q** = [ _[⃗]_ **h** _L_ _,_ **h** 1 ] _∈_ R _[d]_



**h** 1 ] _∈_ R _[d]_


tures to an embedding layer followed by a ReLU function
to match their dimensions with query features. Formally, let
**S** = [ **s** 1 _, ...,_ **s** _T_ ] _∈_ R _[d][×][T]_ be a matrix that stores the _T_ sampled segment features in its columns [2] . If the input videos
are short and the number of segments is less than _T_, the
missing parts are filled with zero vectors. We append the
temporal position embedding of each segment to the corresponding segment feature vector as done in [6] to improve
accuracy in practice. This procedure leads to the following
equation for video representation:


**S** = ReLU( **W** seg _f_ v ( _V_ )) + _f_ pos ( **W** pos _,_ [1 _, ..., T_ ]) _,_ (3)


where **W** seg _∈_ R _[d][×][d]_ _[v]_ denotes a learnable segment feature
embedding matrix while _f_ pos ( _·, ·_ ) is a lookup table defined
by an embedding matrix **W** pos _∈_ R _[d][×][T]_ and a timestamp
vector [1 _, . . ., T_ ]. Note that _d_ _v_ is the dimension of feature
provided by _f_ v ( _·_ ). Since we formulate the given task as
a location regression problem, the position encoding is a
crucial step for identifying semantics at diverse temporal
locations in the subsequent procedure.


**3.3. Sequential Query Attention Network (SQAN)**


SQAN, denoted by _f_ e ( _·_ ) in Eq. (2), plays a key role in
identifying semantic phrases describing semantic entities
( _e.g_ ., actors, objects, and actions) that should be observed
in videos for precise localization. Since there is no groundtruth for semantic phrases, we learn their representations
in an end-to-end manner. To this end, we adopt an attention mechanism with an assumption that semantic phrases
are defined by a sequence of words in a query as shown in
Fig. 1(a). Those semantic phrases can be extracted independently of each other. Note, however, that since our goal is
to obtain _distinct_ phrases, we extract them by sequentially
conditioning on preceding ones as in [13, 33].
Given _L_ word-level features **E** = [ **w** 1 _, ...,_ **w** _L_ ] _∈_ R _[d][×][L]_

and a sentence-level feature **q** _∈_ R _[d]_, we extract _N_ semantic phrase features _{_ **e** [(1)] _, . . .,_ **e** [(] _[N]_ [)] _}_ . In each step _n_, a guidance vector **g** [(] _[n]_ [)] _∈_ R _[d]_ is obtained by embedding the vector
that concatenates a linearly transformed sentence-level feature and the previous semantic phrase feature **e** [(] _[n][−]_ [1)] _∈_ R _[d]_,
which is given by


**g** [(] _[n]_ [)] = ReLU( **W** g ([ **W** q [(] _[n]_ [)] **q** ; **e** [(] _[n][−]_ [1)] ])) _,_ (4)


where **W** g _∈_ R _[d][×]_ [2] _[d]_ and **W** q [(] _[n]_ [)] _∈_ R _[d][×][d]_ are learnable embedding matrices. Note that we use different embedding
matrix **W** q [(] _[n]_ [)] at each step to attend more readily to different
aspects of the query. Then, we obtain the current semantic


2 Although semantic phrases are sometimes associated with spatiotemporal regions in a video, for computational efficiency, we only consider temporal relationship between phrases and a video, and use spatially
pooled representation for each segment.



phrase feature **e** [(] _[n]_ [)] by estimating the attention weight vector **a** [(] _[n]_ [)] _∈_ R _[L]_ over word-level features and computing a
weighted sum of the word-level features as follows:


_α_ _l_ [(] _[n]_ [)] = **W** qatt (tanh( **W** g _α_ **g** [(] _[n]_ [)] + **W** w _α_ **w** _l_ )) _,_ (5)

**a** [(] _[n]_ [)] = softmax([ _α_ 1 [(] _[n]_ [)] _[, ..., α]_ _L_ [(] _[n]_ [)] [])] _[,]_ (6)



where **W** qatt _∈_ R [1] _[×]_ _[d]_ 2, **W** g _α_ _∈_ R _d_ 2 _[×][d]_ and **W** w _α_ _∈_ R _d_ 2 _[×][d]_

are learnable embedding matrices in the query attention
layer, and _α_ _l_ [(] _[n]_ [)] is the confidence value for the _l_ -th word at
the _n_ -th step.


**3.4. Local-Global Video-Text Interactions**


Given the semantic phrase features, we perform videotext interactions in three levels with two objectives: 1) individual semantic phrase understanding, and 2) relation modeling between semantic phrases.


**Individual semantic phrase understanding** Each semantic phrase feature interacts with individual segment features in two levels: _segment-level modality fusion_ and _local_
_context modeling_ . During the segment-level modality fusion, we encourage the segment features relevant to the semantic phrase features to be highlighted and the irrelevant
ones to be suppressed. However, segment-level interaction
is not sufficient to understand long-range semantic entities
properly since each segment has a limited field-of-view of
16 frames. We thus introduce the local context modeling
that considers neighborhood of individual segments.
With this in consideration, we perform the segment-level
modality fusion similar to [14] using the Hadamard product
while modeling the local context based on a residual block
(ResBlock) that consists of two temporal convolution layers. Note that we use kernels of large bandwidth ( _e.g_ ., 15)
in the ResBlock to cover long-range semantic entities. The
whole process is summarized as follows:


**˜m** [(] _i_ _[n]_ [)] = **W** m [(] _[n]_ [)] [(] **[W]** s [(] _[n]_ [)] **s** _i_ _⊙_ **W** e [(] _[n]_ [)] **e** [(] _[n]_ [)] ) _,_ (8)

**M** [(] _[n]_ [)] = ResBlock([ ˜ **m** [(] 1 _[n]_ [)] _[, ...,]_ [ ˜] **[m]** [(] _T_ _[n]_ [)] [])] _[,]_ (9)


where **W** m [(] _[n]_ [)] _∈_ R _[d][×][d]_, **W** s [(] _[n]_ [)] _∈_ R _[d][×][d]_ and **W** e [(] _[n]_ [)] _∈_ R _[d][×][d]_

are learnable embedding matrices for segment-level fusion, and _⊙_ is the Hadamard product operator. Note that
**m** ˜ [(] _i_ _[n]_ [)] _∈_ R _[d]_ stands for the _i_ -th bi-modal segment feature
after segment-level fusion, and **M** [(] _[n]_ [)] _∈_ R _[d][×][T]_ denotes a
semantics-specific segment feature for the _n_ -th semantic
phrase feature **e** [(] _[n]_ [)] .


**Relation modeling between semantic phrases** After obtaining a set of _N_ semantics-specific segment features,



**e** [(] _[n]_ [)] =



_L_
� **a** [(] _l_ _[n]_ [)] **w** _l_ _,_ (7)


_l_ =1


_{_ **M** [(1)] _, ...,_ **M** [(] _[N]_ [)] _}_, independently, we take contextual and
temporal relations between semantic phrases into account.
For example, in Fig. 1(a), understanding ‘it’ in a semantic
phrase of ‘put it in a pan’ requires the context from another
phrase of ‘mixed all ingredients.’ Since such relations can
be defined between semantic phrases with a large temporal
gap, we perform _global context modeling_ by observing all
the other segments.
For the purpose, we first aggregate _N_ segment features
specific to semantic phrases, _{_ **M** [(1)] _, ...,_ **M** [(] _[N]_ [)] _}_, using attentive pooling, where the weights are computed based on
the corresponding semantic phrase features, as shown in
Eq. (10) and (11). Then, we employ Non-Local block [30]
(NLBlock) that is widely used to capture global context.
The process of global context modeling is given by


**c** = softmax(MLP satt ([ **e** [(1)] _, ...,_ **e** [(] _[N]_ [)] ])) _,_ (10)



˜
**R** =



_N_
� **c** [(] _[n]_ [)] **M** [(] _[n]_ [)] _,_ (11)


_n_ =1



**R** = NLBlock( **R** [˜] ) (12)



� T



= **R** [˜] + ( **W** rv **R** [˜] ) softmax



( **W** rq **R** [˜] ) [T] ( **W** rk **R** [˜] )
~~_√_~~ _d_
�



_,_



where MLP satt denotes a multilayer perceptron (MLP) with
a hidden layer of _[d]_ 2 [-dimension and] **[ c]** _[ ∈]_ [R] _[N]_ [ is a weight]

vector for the _N_ semantics-specific segment features. **R** [˜] _∈_
R _[d][×][T]_ is the aggregated feature via attentive pooling, and
**R** _∈_ R _[d][×][T]_ is the final semantics-aware segment features using the proposed local-global video-text interactions. Note that **W** rq _∈_ R _[d][×][d]_, **W** rk _∈_ R _[d][×][d]_ and **W** rv _∈_
R _[d][×][d]_ are learnable embedding matrices in the NLBlock.


**3.5. Temporal Attention based Regression**


Once the semantics-aware segment features are obtained,
we summarize the information while highlighting important
segment features using temporal attention, and finally predict the time interval ( _t_ _[s]_, _t_ _[e]_ ) using an MLP as follows:


**o** = softmax(MLP tatt ( **R** )) _,_ (13)



Figure 3. Visualization of query attention weights (left) without
the distinct query attention loss and (right) with it. SQAN successfully extracts semantic phrases corresponding to actors and actions
across different steps.


_L_ tag, and 3) distinct query attention loss _L_ dqa, and the total
loss is given by


_L_ = _L_ reg + _L_ tag + _L_ dqa _._ (16)


**Location regression loss** Following [35], the regression
loss is defined as the sum of smooth L 1 distances between
the normalized ground-truth time interval ( _t_ [ˆ] _[s]_ _,_ _t_ [ˆ] _[e]_ ) _∈_ [0 _,_ 1]
and our prediction ( _t_ _[s]_ _, t_ _[e]_ ) as follows:


_L_ reg = smooth L1 ( _t_ [ˆ] _[s]_ _−_ _t_ _[s]_ ) + smooth L1 ( _t_ [ˆ] _[e]_ _−_ _t_ _[e]_ ) _,_ (17)


where smooth L1 ( _x_ ) is defined as 0 _._ 5 _x_ [2] if _|x| <_ 1 and _|x| −_
0 _._ 5 otherwise.


**Temporal attention guidance loss** Since we directly
regress the temporal positions from temporally attentive
features, the quality of temporal attention is critical. Therefore, we adopt the temporal attention guidance loss proposed in [35], which is given by


_T_
� _i_ =1 **[o]** [ˆ] _[i]_ [ lo][g(] **[o]** _[i]_ [)]
_L_ tag = _−_ ~~�~~ _Ti_ =1 **[o]** [ˆ] _[i]_ _,_ (18)


where ˆ **o** _i_ is set to 1 if the _i_ -th segment is located within
the ground-truth time interval and 0 otherwise. The attention guidance loss makes the model obtain higher attention
weights for the segments related to the text query.


**Distinct query attention loss** Although SQAN is designed to capture different semantic phrases in a query, we
observe that the query attention weights in different steps
are often similar as depicted in Fig. 3. Thus, we adopt a
regularization term introduced in [19] to enforce query attention weights to be distinct along different steps:


_L_ dqa = _||_ ( **A** [T] **A** ) _−_ _λI||_ _F_ [2] _[,]_ (19)


where **A** _∈_ R _[L][×][N]_ is the concatenated query attention
weights across _N_ steps and _|| · ||_ _F_ denotes Frobenius norm
of a matrix. The loss encourages attention distributions to
have less overlap by making the query attention weights at



**v** =



_T_
� **o** _i_ **R** _i_ _,_ (14)


_i_ =1



_t_ _[s]_ _, t_ _[e]_ = MLP reg ( **v** ) _,_ (15)


where **o** _∈_ R _[T]_ and **v** _∈_ R _[d]_ are attention weights for segments and summarized video feature, respectively. Note
that MLP tatt and MLP reg have _[d]_ 2 [- and] _[ d]_ [-dimensional hidden]

layers, respectively.


**3.6. Training**


We train the network using three loss terms—1) location regression loss _L_ reg, 2) temporal attention guidance loss


Table 1. Performance comparison with other algorithms on the
Charades-STA dataset. The bold-faced numbers mean the best

performance.

|Method|R@0.3 R@0.5 R@0.7 mIoU|
|---|---|
|Method|R@0.3<br>R@0.5<br>R@0.7<br>mIoU|
|Random<br>CTRL [8]<br>SMRL [29]<br>SAP [5]<br>ACL [9]<br>MLVI [32]<br>TripNet [11]<br>RWM [12]<br>ExCL [10]<br>MAN [37]<br>PfTML-GA [25]|-<br>8.51<br>3.03<br>-<br>-<br>21.42<br>7.15<br>-<br>-<br>24.36<br>9.01<br>-<br>-<br>27.42<br>13.36<br>-<br>-<br>30.48<br>12.20<br>-<br>54.70<br>35.60<br>15.80<br>-<br>51.33<br>36.61<br>14.50<br>-<br>-<br>36.70<br>-<br>-<br>65.10<br>44.10<br>22.60<br>-<br>-<br>46.53<br>22.72<br>-<br>67.53<br>52.02<br>33.74<br>-|
|Ours|**72.96**<br>**59.46**<br>**35.48**<br>**51.38**|



two different steps decorrelated. Note that _λ ∈_ [0 _,_ 1] controls the extent of overlap between query attention distributions; when _λ_ is close to 1, the attention weights are learned
to be the one-hot vector. Fig. 3 clearly shows that the regularization term encourages the model to focus on distinct
semantic phrases across query attention steps.


**4. Experiments**


**4.1. Datasets**


**Charades-STA** The dataset is collected from the Cha
rades dataset for evaluating text-to-video temporal grounding by [8], which is composed of 12,408 and 3,720 time
interval and text query pairs in training and test set, respectively. The videos are 30 seconds long on average and the
maximum length of a text query is set to 10.


**ActivityNet Captions** This dataset, which has originally
been constructed for dense video captioning, consists of 20k
YouTube videos with an average length of 120 seconds. It
is divided into 10,024, 4,926, and 5,044 videos for training, validation, and testing, respectively. The videos contain 3.65 temporally localized time intervals and sentence
descriptions on average, where the average length of the descriptions is 13.48 words. Following the previous methods,
we report the performance of our algorithm on the combined two validation set (denoted by _val_ ~~_1_~~ and _val_ ~~_2_~~ ) since
annotations of the test split is not publicly available.


**4.2. Metrics**


Following [8], we adopt two metrics for the performance
comparison: 1) Recall at various thresholds of the temporal
Intersection over Union (R@tIoU) to measure the percentage of predictions that have tIoU with ground-truth larger
than the thresholds, and 2) mean averaged tIoU (mIoU). We
use three tIoU threshold values, _{_ 0 _._ 3 _,_ 0 _._ 5 _,_ 0 _._ 7 _}_ .



Table 2. Performance comparison with other algorithms on the
ActivityNet Captions dataset. The bold-faced numbers denote the
best performance.

|Method|R@0.3 R@0.5 R@0.7 mIoU|
|---|---|
|Method|R@0.3<br>R@0.5<br>R@0.7<br>mIoU|
|MCN [1]<br>CTRL [8]<br>ACRN [20]<br>MLVI [32]<br>TGN [4]<br>TripNet [11]<br>PfTML-GA [25]<br>ABLR [35]<br>RWM [12]|21.37<br>9.58<br>-<br>15.83<br>28.70<br>14.00<br>-<br>20.54<br>31.29<br>16.17<br>-<br>24.16<br>45.30<br>27.70<br>13.60<br>-<br>45.51<br>28.47<br>-<br>-<br>45.42<br>32.19<br>13.93<br>-<br>51.28<br>33.04<br>19.26<br>37.78<br>55.67<br>36.79<br>-<br>36.99<br>-<br>36.90<br>-<br>-|
|Ours|**58.52**<br>**41.51**<br>**23.07**<br>**41.13**|



**4.3. Implementation Details**


For the 3D CNN modules to extract segment features for
Charades-STA and ActivityNet Captions datasets, we employ I3D [3] [3] and C3D [28] [4] networks, respectively, while
fixing their parameters during a training step. We uniformly
sample _T_ (= 128) segments from each video. For query
encoding, we maintain all word tokens after lower-case
conversion and tokenization; vocabulary sizes are 1,140
and 11,125 for Charades-STA and ActivityNet Captions
datasets, respectively. We truncate all text queries that have
maximum 25 words for ActivityNet Captions dataset. For
sequential query attention network, we extract 3 and 5 semantic phrases and set _λ_ in Eq. (19) to 0.3 and 0.2 for Charades and ActivityNet Captions datasets, respectively. In all
experiments, we use Adam [15] to learn models with a minibatch of 100 video-query pairs and a fixed learning rate of
0.0004. The feature dimension _d_ is set to 512.


**4.4. Comparison with Other Methods**


We compare our algorithm with several recent methods,
which are divided into two groups: _scan-and-localize_ methods, which include MCN [1], CTRL [8], SAP [5], ACL [9],
ACRN [20], MLVI [32], TGN [4], MAN [37], TripNet [11],
SMRL [29], and RWM [12], and proposal-free algorithms
such as ABLR [35], ExCL [10], and PfTML-GA [25].

Table 1 and Table 2 summarize the results on Charades
STA and ActivityNet Captions datasets, respectively, where
our algorithm outperforms all competing methods. It is noticeable that the proposed technique surpasses the state-ofthe-art performances by 7.44% and 4.61% points in terms
of R@0.5 metric, respectively.


**4.5. In-Depth Analysis**


For a better understanding of our algorithm, we analyze
the contribution of the individual components.


3 https://github.com/piergiaj/pytorch-i3d
4 http://activity-net.org/challenges/2016/download.html#c3d


Table 3. Results of main ablation studies on the Charades-STA dataset. The bold-faced numbers means the best performance.


|Method|Query Information<br>sentence (q) phrase (e)<br>√|Loss Terms<br>+L +L<br>tag dqa<br>√ √|R@0.3 R@0.5 R@0.7 mIoU|
|---|---|---|---|
|LGI<br>LGI w/o_ L_dqa<br>LGI w/o_ L_tag<br>LGI–SQAN<br>LGI–SQAN w/o_ L_tag|_√_<br>_√_<br>~~_√_~~<br>_√_|_√_<br>_√_<br>~~_√_~~|**72.96**<br>**59.46**<br>**35.48**<br>**51.38**<br>71.42<br>58.28<br>34.30<br>50.24<br>61.91<br>47.12<br>24.62<br>42.43<br>71.02<br>57.34<br>33.25<br>49.52<br>57.66<br>43.33<br>22.74<br>39.53|



**59.5**


**59.0**


**58.5**


**58.0**


**57.5**



**2** **3** **4**


**Number of semantic phrases**



**2** **3** **4** **5** **6** **7**


**Number of semantic phrases**





**41.50**


**41.25**


**41.00**


**40.75**


**40.50**









(a) Charades-STA (b) ActivityNet Captions


Figure 4. Ablation studies with respect to the number of extracted
semantic phrases.





**59.50**


**59.00**


**58.50**


**58.00**





**0.1** **0.2** **0.3** **0.4** **0.5** **0.6** **0.7** **0.8** **0.9** **1**


**Lambda**



**0.1** **0.2** **0.3** **0.4** **0.5** **0.6** **0.7** **0.8** **0.9** **1**


**Lambda**















**41.50**


**41.25**


**41.00**


**40.75**


**40.50**















(a) Charades-STA (b) ActivityNet Captions


Figure 5. Ablation studies across with respect to _λ_ values.


**4.5.1** **Main Ablation Studies**


We first investigate the contribution of sequential query attention network (SQAN) and loss terms on the CharadesSTA dataset. In this experiment, we train five variants
of our model: 1) LGI: our full model performing localglobal video-text interactions based on the extracted semantic phrase features by SQAN and being learned using all loss terms, 2) LGI w/o _L_ dqa : LGI learned without
distinct query attention loss _L_ dqa, 3) LGI w/o _L_ tag : LGI
learned without temporal attention guidance loss _L_ tag, 4)
LGI–SQAN: a model localizing a text query with sentencelevel feature **q** without SQAN, 5) LGI–SQAN w/o _L_ tag :
LGI–SQAN learned without _L_ tag . Note that the architecture of LGI–SQAN is depicted in supplementary material.

Table 3 summarizes the results where we observe the fol
lowing. First, extracting semantic phrase features from the
query (LGI) is more effective for precise localization than
simply relying on the sentence-level representation (LGI–
SQAN). Second, regularizing the query attention weights
for distinctiveness, _i.e_ ., using _L_ dqa, enhances performance
by capturing distinct constituent semantic phrases. Third,
temporal attention guidance loss _L_ tag improves the accuracy of localization by making models focus on segment
features within the target time interval. Finally, it is no


Table 4. Performance comparison by varying the combinations
of modules in local and global context modeling on the CharadesSTA dataset. The bold-faced numbers mean the best performance.


Local Context Global Context R@0.5

          -           - 40.86

Masked NL (b=1, w=15)   - 42.66
Masked NL (b=4, w=15)   - 45.78
Masked NL (b=4, w=31)   - 47.80
ResBlock (k=3)    - 43.95
ResBlock (k=7)    - 46.24
ResBlock (k=11)    - 49.78
ResBlock (k=15)    - 50.54
          - NLBlock (b=1) 48.12
          - NLBlock (b=2) 48.95
          - NLBlock (b=4) 48.52
Masked NL (b=1, w=15) NLBlock (b=1) 50.11
Masked NL (b=4, w=15) NLBlock (b=1) 53.92
Masked NL (b=4, w=31) NLBlock (b=1) 54.81
ResBlock (k=7) NLBlock (b=1) 55.00
ResBlock (k=15) NLBlock (b=1) **57.34**


ticeable that LGI–SQAN already outperforms the state-ofthe-art method at R@0.5 ( _i.e_ ., 52.02% vs.57.34%), which
shows the superiority of the proposed local-global videotext interactions in modeling relationship between video
segments and a query.
We also analyze the impact of two hyper-parameters in
SQAN—the number of semantic phrases ( _N_ ) and controlling value ( _λ_ ) in _L_ dqa —on the two datasets. Fig. 4 presents
the results across the number of semantic phrases in SQAN,
where performances increase until certain numbers (3 and
5 for Charades-STA and ActivityNet Captions datasets, respectively) and decrease afterward. This is because larger
N makes models capture shorter phrases and fail to describe proper semantics. As shown in Fig. 5, the controlling value _λ_ of 0.2 and 0.3 generally provides good performances while higher _λ_ provides worse performance by
making models focus on one or two words as phrases.


**4.5.2** **Analysis on Local-Global Video-Text Interaction**


We perform in-depth analysis for local-global interaction on
the Charades-STA dataset. For this experiment, we employ
LGI–SQAN (instead of LGI) as our base algorithm to save
training time.


|Option|R@0.5|
|---|---|
|Option|R@0.5|
|Local-Global-Fusion<br>Local-Fusion-Global<br>Fusion-Local-Global|46.96<br>53.47<br>57.34|


(a) Performance comparison depending on
the location of segment-level modality fusion in the video-text interaction.



|Option|R@0.5 Query:|
|---|---|
|Option|R@0.5|
|Addition<br>Concatenation<br>Hadamard Product|46.75<br>48.15<br>57.34|


(b) Performance comparison with respect
to fusion methods.


Table 5. Ablations on the Charades-STA dataset.



|Option an mixed all ingredient|R@0.5 ts, put it in a|
|---|---|
|Option|R@0.5|
|None<br>Position Embedding|45.70<br>57.34|


(c) Impact of position embedding for video
encoding.


time



**Impact of local and global context modeling** We study
the impact of local and global context modeling by varying
the kernel size ( _k_ ) in the residual block (ResBlock) and the
number of blocks ( _b_ ) in Non-Local block (NLBlock). For
local context modeling, we also adopt an additional module
referred to as a masked Non-Local block (Masked NL) in
addition to ResBlock; the mask restricts attention region to
a local scope with a fixed window size _w_ centered at individual segments in the NLBlock.
Table 4 summarizes the results, which imply the following. First, the performance of the model using only
segment-level modality fusion without context modeling is
far from the state-of-the-art performance. Second, incorporating local or global context modeling improves performance by enhancing the alignment of semantic phrases with
the video. Third, a larger scope of local view in local context modeling further improves performance, where ResBlock is more effective than Masked NL according to our
observation. Finally, incorporating both local and global
context modeling results in the best performance gain of
16.48% points. Note that while the global context modeling has a capability of local context modeling by itself, it
turns out to be difficult to model local context by increasing
the number of NLBlocks; a combination of Masked NL and
NLBlock outperforms the stacked NLBlocks, showing the
importance of explicit local context modeling.


**When to perform segment-level modality fusion** Table 5(a) presents the results from three different options for
the modality fusion phase. This result implies that early fusion is more beneficial for semantics-aware joint video-text
understanding and leads to the better accuracy.


**Modality fusion method** We compare different fusion
operations—addition, concatenation, and Hadamard product. For concatenation, we match the output feature dimension with that of the other methods by employing an additional embedding layer. Table 5(b) shows that Hadamard
product achieves the best performance while the other two
methods perform much worse. We conjecture that this is
partly because Hadamard product acts as a gating operation rather than combines two modalities, and thus helps
the model distinguish segments relevant to semantic phrases
from irrelevant ones.



Query: she then sands down the table and dips a brush into paint

|GT|Col2|
|---|---|
|GI|attention<br>LGI|
|GI-SQAN|attention<br>LGI-SQAN|



Figure 6. Visualization of predictions of two models (LGI and
LGI–SQAN) and their temporal attention weights **o** computed before regression.


**Impact of position embedding** Table 5(c) presents the
effectiveness of the position embedding in identifying se
|antic entities at diverse temp|oral locations and improving|
|---|---|
|e accuracy of temporal grou|nding.|
|||



**4.5.3** **Qualitative Results**


Fig. 6 illustrates the predictions and the temporal attention weights **o** for LGI and LGI–SQAN. Our full model
(LGI) provides more accurate locations than LGI–SQAN
through query understanding in a semantic phrase level,
which makes video-text interaction more effective. More

examples with visualization of temporal attention weights,
query attention weights **a** and predictions are presented in
our supplementary material.


**5. Conclusion**


We have presented a novel local-global video-text interaction algorithm for text-to-video temporal grounding via
constituent semantic phrase extraction. The proposed multilevel interaction scheme is effective in capturing relationships of semantic phrases and video segments by modeling
local and global contexts. Our algorithm achieves the stateof-the-art performance in both Charades-STA and ActivityNet Captions datasets.


**Acknowledgments** This work was partly supported by
IITP grant funded by the Korea government (MSIT) (20160-00563, 2017-0-01780), and Basic Science Research Program (NRF-2017R1E1A1A01077999) through the NRF
funded by the Ministry of Science, ICT. We also thank
Tackgeun You and Minsoo Kang for valuable discussion.


**References**


[1] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef
Sivic, Trevor Darrell, and Bryan Russell. Localizing Moments in Video with Natural Language. In _ICCV_, 2017.

[2] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem,
and Juan Carlos Niebles. Activitynet: A Large-Scale Video
Benchmark for Human Activity Understanding. In _CVPR_,
2015.

[3] Joao Carreira and Andrew Zisserman. Quo Vadis, Action
Recognition? a New Model and the Kinetics Dataset. In
_CVPR_, 2017.

[4] Jingyuan Chen, Xinpeng Chen, Lin Ma, Zequn Jie, and
Tat-Seng Chua. Temporally Grounding Natural Sentence in
Video. In _EMNLP_, 2018.

[5] Shaoxiang Chen and Yu-Gang Jiang. Semantic Proposal
for Activity Localization in Videos via Sentence Query. In
_AAAI_, 2019.

[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. BERT: Pre-Training of Deep Bidirectional
Transformers for Language Understanding. _arXiv preprint_
_arXiv:1810.04805_, 2018.

[7] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and
Kaiming He. Slowfast Networks for Video Recognition. In
_ICCV_, 2019.

[8] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia.
Tall: Temporal Activity Localization via Language Query.
In _CVPR_, 2017.

[9] Runzhou Ge, Jiyang Gao, Kan Chen, and Ram Nevatia.
MAC: Mining Activity Concepts for Language-based Temporal Localization. In _WACV_, 2019.

[10] Soham Ghosh, Anuva Agarwal, Zarana Parekh, and Alexander Hauptmann. ExCL: Extractive Clip Localization
Using Natural Language Descriptions. _arXiv preprint_
_arXiv:1904.02755_, 2019.

[11] Meera Hahn, Asim Kadav, James M Rehg, and Hans Peter
Graf. Tripping through Time: Efficient Localization of Activities in Videos. _arXiv preprint arXiv:1904.09936_, 2019.

[12] Dongliang He, Xiang Zhao, Jizhou Huang, Fu Li, Xiao Liu,
and Shilei Wen. Read, Watch, and Move: Reinforcement
Learning for Temporally Grounding Natural Language Descriptions in Videos. In _AAAI_, 2019.

[13] Drew A Hudson and Christopher D Manning. Compositional
Attention Networks for Machine Reasoning. In _ICLR_, 2018.

[14] Jin-Hwa Kim, Kyoung-Woon On, Woosang Lim, Jeonghee
Kim, Jung-Woo Ha, and Byoung-Tak Zhang. Hadamard
Product for Low-Rank Bilinear Pooling. In _ICLR_, 2017.

[15] Diederik P Kingma and Jimmy Ba. Adam: A Method for
Stochastic Optimization. In _ICLR_, 2015.

[16] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and
Juan Carlos Niebles. Dense-Captioning Events in Videos. In
_ICCV_, 2017.

[17] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara Berg.
TVQA: Localized, Compositional Video Question Answering. In _EMNLP_, 2018.

[18] Tianwei Lin, Xu Zhao, and Zheng Shou. Single Shot Temporal Action Detection. In _ACMMM_, 2017.




[19] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos,
Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A
Structured Self-Attentive Sentence Embedding. In _ICLR_,
2017.

[20] Meng Liu, Xiang Wang, Liqiang Nie, Xiangnan He, Baoquan Chen, and Tat-Seng Chua. Attentive Moment Retrieval
in Videos. In _The 41st International ACM SIGIR Conference_
_on Research & Development in Information Retrieval_ .

[21] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian
Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C
Berg. SSD: Single Shot Multibox Detector. In _ECCV_, 2016.

[22] Alberto Montes, Amaia Salvador, Santiago Pascual, and
Xavier Giro-i Nieto. Temporal Activity Detection in
Untrimmed Videos with Recurrent Neural Networks. _arXiv_

_preprint arXiv:1608.08128_, 2016.

[23] Jonghwan Mun, Paul Hongsuck Seo, Ilchae Jung, and Bohyung Han. Marioqa: Answering Questions by Watching
Gameplay Videos. In _ICCV_, 2017.

[24] Jonghwan Mun, Linjie Yang, Zhou Ren, Ning Xu, and Bohyung Han. Streamlined Dense Video Captioning. In _CVPR_,
2019.

[25] Cristian Rodriguez Opazo, Edison Marrese-Taylor, Fatemeh Sadat Saleh, Hongdong Li, and Stephen Gould.
Proposal-free Temporal Moment Localization of a NaturalLanguage Query in Video using Guided Attention. _arXiv_
_preprint arXiv:1908.07236_, 2019.

[26] Zheng Shou, Jonathan Chan, Alireza Zareian, Kazuyuki
Miyazawa, and Shih-Fu Chang. CDC: Convolutional-DeConvolutional Networks for Precise Temporal Action Localization in Untrimmed Videos. In _CVPR_, 2017.

[27] Zheng Shou, Dongang Wang, and Shih-Fu Chang. Temporal
Action Localization in Untrimmed Videos via Multi-Stage
Cnns. In _CVPR_, 2016.

[28] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani,
and Manohar Paluri. Learning Spatiotemporal Features with
3D Convolutional Networks. In _ICCV_, 2015.

[29] Weining Wang, Yan Huang, and Liang Wang. LanguageDriven Temporal Activity Localization: A Semantic Matching Reinforcement Learning Model. In _CVPR_, 2019.

[30] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-Local Neural Networks. In _CVPR_, 2018.

[31] Huijuan Xu, Abir Das, and Kate Saenko. R-C3D: Region
Convolutional 3d Network for Temporal Activity Detection.
In _ICCV_, 2017.

[32] Huijuan Xu, Kun He, Bryan A Plummer, Leonid Sigal, Stan
Sclaroff, and Kate Saenko. Multilevel Language and Vision
Integration for Text-to-Clip Retrieval. In _AAAI_, 2019.

[33] Sibei Yang, Guanbin Li, and Yizhou Yu. Dynamic Graph Attention for Referring Expression Comprehension. In _ICCV_,
2019.

[34] Serena Yeung, Olga Russakovsky, Greg Mori, and Li FeiFei. End-to-End Learning of Action Detection From Frame
Glimpses in Videos. In _CVPR_, 2016.

[35] Yitian Yuan, Tao Mei, and Wenwu Zhu. To Find Where You
Talk: Temporal Sentence Localization in Video with Attention Based Location Regression. In _AAAI_, 2019.


[36] Runhao Zeng, Wenbing Huang, Mingkui Tan, Yu Rong,
Peilin Zhao, Junzhou Huang, and Chuang Gan. Graph Convolutional Networks for Temporal Action Localization. In
_ICCV_, 2019.

[37] Da Zhang, Xiyang Dai, Xin Wang, Yuan-Fang Wang, and
Larry S Davis. Man: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment. In _CVPR_, 2019.

[38] Yue Zhao, Yuanjun Xiong, Limin Wang, Zhirong Wu, Xiaoou Tang, and Dahua Lin. Temporal Action Detection with
Structured Segment Networks. In _ICCV_, 2017.


**6. Supplementary Material**


This supplementary document first presents the architecture of our model without semantic phrase extraction ( _i.e_ .,
LGI–SQAN) used for in-depth analysis on the local-global
video-text interactions. We also present additional qualitative examples of our algorithm.


**6.1. Architectural Details of LGI–SQAN**


Compared to our full model (LGI), LGI–SQAN does
not explicitly extract semantic phrases from a query as presented in Fig. 7; it performs local-global video-text interactions based on the sentence-level feature representing whole
semantics of the query.
In our model, the sentence-level feature ( **q** ) is copied
to match its dimension with the temporal dimension ( _T_ )
of segment-level features ( **S** ). Then, as done in our full
model, we perform local-global video-text interactions—1)
segment-level modality fusion, 2) local context modeling,
and 3) global context modeling—followed by the temporal attention based regression to predict the time interval

[ _t_ _[s]_ _, t_ _[e]_ ]. Note that we adopt a masked non-local block or
a residual block for local context modeling, and a non-local
block for global context modeling, respectively.


**6.2. Visualization of More Examples**


Fig. 8 and Fig. 9 illustrate additional qualitative results
in the Charades-STA and ActivityNet Captions datasets,
respectively; we present two types of attention weights—
temporal attention weights **o** (T-ATT) and query attention
weights **a** (Q-ATT)—and predictions (Pred.). T-ATT shows
that our algorithm successfully attends to relevant segments
to the input query while Q-ATT depicts that our sequential query attention network favorably identifies semantic
phrases from the query describing actors, objects, actions,
etc. Note that our model often predicts accurate time intervals even from the noisy temporal attention.
Fig. 10 demonstrates the failure cases of our algorithm.
As presented in the first example of Fig. 10, our method
fails to localize the query on the confusing video, where a
man looks like smiling at multiple time intervals. However,
note that the temporal attention of our method captures the



Figure 7. Illustration of architecture of LGI–SQAN. In LGI–
SQAN, we use sentence-level feature **q** to interact with video.


segments relevant to the query at diverse temporal locations
in a video. In addition, as presented in the second example
of Fig. 10, our model sometimes fails to extract proper semantic phrases; ‘wooden’ and ‘floorboards’ are captured at
different steps although ‘wooden floorboards’ is more natural, which results in the inaccurate localization.


Q-ATT


Q-ATT


Q-ATT


Q-ATT



Query: a person runs down the hall


GT


Pred.


T-ATT


Query: the person puts the mirror in the box


GT


Pred.


T-ATT


Query: person eating from a bag


GT


Pred.


T-ATT


Query: person begins to undress


GT


Pred.


T-ATT



time


time


time


time



Figure 8. Qualitative results of our algorithm on the Charades-STA dataset. T-ATT and Q-ATT stand for temporal attention weights and
query attention weights, respectively.


Q-ATT


Q-ATT


Q-ATT


Q-ATT



Query: the winner celebrate and people congratulates the winner


GT


Pred.


T-ATT



time


time



Query: a large group of men are seen skating around the ice with young child following behind


GT


Pred.


T-ATT


time


Query: the guy puts the clippings in his palm


GT


Pred.


T-ATT


time


Query: the man puts his hand on the top of the pipe


GT


Pred.


T-ATT



Figure 9. Qualitative results of our algorithm on the ActivityNet Captions dataset. T-ATT and Q-ATT stand for temporal attention weights
and query attention weights, respectively.


Q-ATT


Q-ATT



Query: another person is smiling


GT


Pred.


T-ATT


Query: a man is installing wooden floorboards


GT


Pred.


T-ATT



time


time



Figure 10. Failure case of our algorithm. Examples in the first and second row are obtained from the Charades-STA and Activity Captions
datasets, respectively.


