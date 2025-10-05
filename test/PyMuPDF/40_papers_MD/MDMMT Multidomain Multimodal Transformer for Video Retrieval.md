## MDMMT: Multidomain Multimodal Transformer for Video Retrieval

A P REPRINT


**Maksim Dzabraev** [1] _[,]_ [2] **, Maksim Kalashnikov** [1] **, Stepan Komkov** [1] _[,]_ [2] **, Aleksandr Petiushko** [1] _[,]_ [2]

1 Lomonosov Moscow State University
2 Huawei Moscow Research Center
dzabraev.maksim@intsys.msu.ru, kalashnikov.maxim@intsys.msu.ru,
stepan.komkov@intsys.msu.ru, petyushko.alexander1@huawei.com



**ABSTRACT**


We present a new state-of-the-art on the text to video retrieval task on MSRVTT and LSMDC benchmarks where
our model outperforms all previous solutions by a large
margin. Moreover, state-of-the-art results are achieved
with a single model on two datasets without finetuning.
This multidomain generalisation is achieved by a proper
combination of different video caption datasets. We show
that training on different datasets can improve test results
of each other. Additionally we check intersection between
many popular datasets and found that MSRVTT has a significant overlap between the test and the train parts, and
the same situation is observed for ActivityNet.


_**Keywords**_ video, language, retrieval, multi-modal,
cross-modal, temporality, transformer, attention


**1** **Introduction**


Video is a quite popular data format, 500+ hours of video
are uploaded on YouTube every minute. Many personal
mobile phones have gigabytes of video. Since video format
gets more popular every year the importance of modern
search methods is increasing as well.


In this work we present our research about text to video
retrieval task. In this task system should return for a given
textual query the most relevant video segments from a
gallery. The query is a textual description of what we
want to find in the video. The query may describe objects,
actions, sounds, ..., and relations between them.


Such search methods are a promising direction for mobile
devices because every year manufacturers increase the
available memory on devices. The large part of the memory
is filled by media data. For end users it is getting difficult
to search for a video made one or two years ago. But users
can easily describe the content of the video using natural
language, which can be effectively used as a search query.



Figure 1: Two types of fusion


This type of approaches have access to all input data from
the beginning of its processing and can make a strong
verdict about data. But these approaches have a significant
drawback because it is not scalable: every new query the
search system should calculate the full forward pass for
this query and for each video segment from the gallery.


Another direction is two stream neural networks [24], [8],
where a textual query and a video are processed by two
different neural networks. As a result the networks produce embeddings inside the same embedding space, where
semantically close textual queries and video segments will
be placed next to each other. The schematic illustration is
presented in Fig. 1b.


The two stream approach is scalable, it allows to precompute video embeddings for all videos from the gallery,
and to do only one forward pass with the text network for



There are two major directions which allow calculate the
relevance between a textual search query and a video segment. The first direction is single stream approaches [32],
where a query and a video together are given to a network and then become fused from the beginning of the
processing. The schematic illustration of this approach is
presented in Fig. 1a.



(a) Scheme for a single-stream
neural network.



(b) Scheme for a two-stream
neural network.


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT



each new query and then to compute the cosine similarity
between the new query embedding and all precomputed
embeddings.


To make a strong video retrieval solution it is important to
show to the model a lot of situations, actions and objects
from real life. There exist a lot of video datasets, but none
of them cover a significant portion of real life situations.
One of the first steps to tackle this problem is to formulate
the rules for combining different existing datasets to a
single large train database.


Text to video retrieval is a modern direction, where one
of the first works was published at 2016 [33]. One of the
most universal solution for video retrieval task is Multi
Modal Transformer [8] architecture which uses BERT [4]
backbone for a video network. It allows in a natural way to
process the temporal dependencies inside the multi modal
data source.


To train a text to video retrieval neural network the training
database should consists of pairs: (a video segment, a
textual description of this video segment). Traditionally
such sort of datasets was created for a video captioning
task. But it turns out that these datasets perfectly can
be used for a video retrieval task. One of the first video
captioning dataset was MSVD, which was created in 2010.
Today there exist more than a dozen of different video
captioning datasets.


The most popular datasets for text to video retrieval is
MSRVTT [39], ActivityNet [17] and LSMDC [29]. Many
researchers test their solutions mostly on these three
datasets.


Our main contributions in this work are the following:


- We present a new state-of-the-art (SotA) result on
MSRVTT and LSMDC benchmarks;

- We present a model which shows good results on three
different benchmarks without finetuning: MSRVTT
(SotA), LSMDC (SotA) and ActivityNet at the same
time;

- We present a practical approach which helps us to find
the overlap between the train and the test parts of used
datasets.


**2** **Related work**


**2.1** **Datasets**


MSRVTT [39] was created in 2016. This dataset is traditionally used by researchers as the main dataset for testing
text to video retrieval models. This dataset consists of
10k video segments, each segment has 20 captions. The
authors collected 257 popular search queries and gathered
from YouTube 118 most relevant videos for each of them.
The dataset has 42 hours of video. The captions were made
by 1327 amazon workers.


Today there are three different test/train splits. The official
split is called **full** split, where the train part has 7k videos



and the test part has 3k videos. There are two important
properties of this split: 1. there are no two video segments
cropped from the same video so as the first segment is
placed in the train part and the second segment is placed in
the test part; 2. there are no two video segments, retrieved
from the same query so as the first one is placed in the
train part and the second one is placed in the test part.


Another two splits are called **1k-A** [40] (sometimes called
jsfusion) and **1k-B** [21] (sometimes called miech). Both
of them have different 1k videos for testing. They were
created by randomly sampling 1k videos from the original
test part (full split). 1k-A train part consists of the original
train split and the rest of the videos from the test part, so it
has 1k videos for the test part and 9k videos for the train
part. 1k-B has 1k videos for the test part and 6.5k videos
for the train. Additionally both splits use only one caption
per segment (instead of 20 captions).


Unfortunately 1k-A and 1k-B mixed up the train and test
parts. This led to violation in properties 1. and 2. which
the full split satisfies.


Another problem is that all these splits have the overlap
between the test and train parts, see C.2 for details. To
be strict we remove the overlap between the test part and
the train part of MSRVTT full split. We called this split
MSRVTT **full clean**, and refer to it as M _c_ . It is worth
to mention that we do not modify the test part, we only
remove some videos from the train part.


The Large Scale Movie Description Challenge
(LSMDC) [29] is the extension of two independent
datasets: MPII Movie Description Dataset (MPIIMD) [28], and Montreal Video Annotation Dataset
(M-VAD) [34].


Video segments for this dataset were cropped from movies,
where movie textualized transcriptions were used as captions. A movie transcription is an audio description of a
video segment that helps blind people to watch movies by
describing what happens, who appears in this time, what
is on background right now and so on.


In this work for testing we use LSMDC public test, which
consists of 1k video segments.


ActivityNet captions dataset [17] consists of 20k videos
and 100k captions, where captions cover the full video
length for the most of videos, and neighbour captions may
intersect. The annotations were made with Amazon Me
chanical Turk.


The situation when some video segments may overlap
makes a problem for text to video retrieval testing. Suppose
we have two video-caption pairs ( _S_ 1 _, C_ 1 ) and ( _S_ 2 _, C_ 2 )
where the video segment _S_ 1 has a non empty overlap with
the video segment _S_ 2 . Now suppose that for query _C_ 1 the
system returns the video segment _S_ 2 . Is it mistake or not?
What to do in this case?


Many previous works used ActivityNet test dataset in a
paragraph retrieval mode. In this mode all captions for all



2


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT


_̸_



video segments are concatenated, then the concatenated
text is used as a textual query and the whole video should
be retrieved for this query. Such mode has two drawbacks.
The first one is that paragraph retrieval is not a classical
video retrieval mode, it is another task. One can ask: if
a model is good in paragraph retrieval will it be good
for video retrieval? The second drawback is that queries
will be long, video segments will be long (compared to
a classical video retrieval mode). This issue requires to
enlarge the input for the model.


Another way to use the test part of ActivityNet is just to
sample once a single random segment from each video. As
a result we will have non intersected video segments and
captions with usual length. We use ActivityNet test part
in this way. We take all videos from val1 and val2 parts,
and sample a single random segment from each video. All
results on ActivityNet are reported on this split.


Additionally in this work the following datasets are used:
NIST TRECVID Twitter vines [1], TGIF [18], MSVD [2],
YouCook2 [43], Something-something V2 [10], Kinetics
700 [31], HowTo100M [23].


**2.2** **Prior Art**


A dominant approach to train video retrieval models is
contrastive learning. The idea of this approach is that we
have a set of pairs (video _i_ _,_ text _i_ ) and elements of each
pair should be placed next to each other in some metric
space: distance(video _i_ _,_ text _i_ ) _→_ 0, at the same time the
element video _i_ should be far from all other text _j_ ) _, j ̸_ = _i_ :
distance(video _i_ _,_ text _j_ ) _→_ + _∞_ . The bi-directional maxmargin ranking loss [13] represents this idea.


When training data have a lot of noise the MIL NCE
loss [22] can be applied in the training procedure. Suppose that we know that a video _i_ should be close to one of
(or several) texts text _i_ 1, ..., text _ik_ . This approach tries to
reduce the distance between the video _i_ and all text _i_ 1, ..., _̸_
text _ik_ at the same time.


All video captions datasets have the following problem.
Suppose the distance between (video _i_ _,_ text _i_ ) is to be minimized while the distance between (video _i_ _,_ text _j_ ) _, j ̸_ = _i_
is to be maximized, but text _i_ and text _j_ are quite similar
(from the semantical point of view). Maybe the optimal
scenario in this situation is to minimize the distance between (video _i_ _,_ text _j_ ) _, j ̸_ = _i_ . In [25] the authors show the
approach which deals with this problem.


As far as an input video is the temporal sequence of tokens
(frames or video segments) it is important to efficiently
aggregate the information from all tokens. Many ideas for
such aggregation in the previous works are borrowed from
the natural language processing. Convolution filters for
aggregation are used in [25], a transformer encoder as a
video aggregator is used in [8], many different aggregation
functions are tested in [26].


We think that the most promising aggregation method is
Multi Modal Transformer (MMT) [8]. MMT is a two



stream solution designed for a text to video retrieval task.
The extraction of features from the input video stream is
done in the following way. An input video is preprocessed
by several pretrained frozen neural networks (these networks are called experts). Original solution uses seven
modalities: motion, RGB, scene, face, OCR, speech, audio, and one pretrained network for each modality is used.
The motion modality is processed with video recognition
networks like S3D, SlowFast, irCSN, where several input
frames are used as a single input. The RGB modality uses
a single frame as an input. The audio modality uses the raw
input sound from a video. After embeddings are extracted
from input data by these experts, it will be augmented by
adding positional encoding tokens (representing time) and
expert tokens. Then the augmented embeddings are passed
through MMT backbone. MMT backbone is a standard
transformer encoder architecture. Each input modality produces one embedding, so in total there are seven output
embedding from MMT.


For encoding the textual query the authors use pretrained
BERT model where the output [CLS] token is used. The
output is postprocessed with shallow networks (one network per modality) to extract the modality related information, in total seven feature vectors will be produced. In
addition to embeddings from the text query seven weights
representing how much the query describes one of seven
modalities are produced. For example, if a query does
not represent the sound, the small weight for the audio
modality should be produced.


The final similarity score is done by a sum of seven
weighted dot products of embeddings.


The MMT is trained with the bi-directional max-margin
ranking loss [13]:


_̸_



�

_j_ = _̸_ _i_



1

_B_

_̸_



_B_
�


_i_ =1 _̸_



max(0 _, s_ _ij_ _−s_ _ii_ + _m_ )+max(0 _, s_ _ji_ _−s_ _ii_ + _m_ )
� �

_̸_



_̸_


where _B, s_ _ij_ _, m_ represent the batch size, the similarity
between the _i_ -th query and the _j_ -th video inside this batch,
and some predefined margin correspondingly.


**3** **Methodology**


Our work is mostly based on MMT. We use the same loss
and a similar architecture, but with different hyperparameters. In this work we study the following questions:


- Which publicly available pretrained motion expert is the
best for text to video retrieval nowadays, Sec. 3.1.

- How to combine several video caption datasets in order to train a strong model without specialisation for a
particular dataset, Sec. 3.2.

- How to find and prevent the overlap between the test and
train parts when combining datasets, Sec. C.



_̸_


3


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT



**3.1** **Motion experts**


The MMT video backbone does not process the raw input video stream, and instead the input video stream is
processed by one or more pretrained experts, where each
expert produces time series of features. The most important modality is motion: a motion expert processes several
video frames as a single input unit and extracts the information about actions and objects within a segment.


We may say that the motion modality is the basis of the
MMT. If a motion expert doesn’t extract some information,
there is a high probability that MMT won’t know about
some events in the video stream. That’s why improving
the motion expert is very important.


We consider several best solutions from Kinetics [15]
benchmark as well as several promising video recognition models and check which one works in the best way as
a motion expert. We present all details in Sec. 4.2.


**3.2** **Dataset creation**


It is possible to train a video retrieval model by two means.
The first way is the way of specialization for a single
domain. For example: create model that will work good
only for MSRVTT benchmark (or domain) but at the same
time this model will show poor results on other datasets
(domains). In this way MMT [8] was trained. The authors
trained three different models for MSRVTT, ActivityNet
and LSMDC datasets. Each of these three networks works
good on domain _X_ if and only if it was trained on _X_, but
at the same time works poor on another domain _Y ̸_ = _X_ .
A proof of this statement we provide in Tab. 6.


The second way is to create a model that will work good
for all domains at the same time. We use this way.


Obviously the model trained in the first way can’t work
good with real users, because the event when a user writes
a search query similar to some caption from a small train
database is very rare.


The second drawback here is that each video retrieval train
dataset is not that big, and it causes the situation that model
doesn’t see many words and real life situations during
training. For example, MSRVTT has only 9k videos and
200k captions in total for training, obviously this is not
enough to train a neural network that will know most of
real life situations, different items and persons. To tackle
with this problem we can take several datasets with videos
and captions and concatenate it.


Different datasets have the different number of videos, the
different number of captions per video, some datasets may
have long captions, some may have short captions, different rules for creating captions were used by human writers,
and so on. Due to these factors some datasets may contain
more information and require longer training time, some
datasets may contain less information and require shorter
training time. On the other hand, if we use long training
time for a small dataset it could lead to overfitting on this



dataset (the data will be memorized). The "information
sizes" of some used datasets are illustrated in Fig. 2.



466h


102h


41h











10K 50K 100K 166K

|h<br>h|ActivityNet<br>TGIF<br>LSMDC<br>YC2 MSRVTT<br>Vines MSVD|
|---|---|
|||


number of unique captions


Figure 2: Radius of the ball represent the “information
size” of dataset. The biggest balls have more diversity in
data.


Fig. 2 is made with a simple algorithm. We take the original training procedure of MMT and for a given dataset
we change the number of examples that will be shown to a
network during training. We define the radius of the ball
as the number of training examples after which the performance gets saturated (i.e. increasing the training time does
not give the better model).


The key question is: what is the proper way for sampling
examples from several datasets taking into account the
different information size?


We use these obvious rules:


1. If a dataset _X_ is larger than _Y_, we should sample from
_X_ more often than from _Y_ ;
2. Training on _X_ and _Y_ combined requires longer train
than training solely on _X_ or _Y_ ;
3. Training on _X_ and _Y_ combined may require a deeper
model than for _X_ or _Y_ .


If we achieve the same results on _X_ after combining _X_
and _Y_ it is still good because model gets better on _Y_ .
Our experiments show that the proper usage of rules 1–3
often improves the results for a specific test dataset (e.g.
MSRVTT) after extending the train dataset.


We managed to combine the following datasets: MSRVTT,
ActivityNet, LSMDC, TwitterVines, YouCook2, MSVD,
TGIF and Something to something V2 (SomethingV2). In
total we increase the number of video segments by 40 times
and the number of unique captions by 4 times compared
with MSRVTT dataset. In Tab. 1 we summarize the sizes of
used datasets. We separate SomethingV2 dataset from all
other datasets because: 1. all video segments are created
artificially, 2. the structure of text captions is quite limited.
At the same time videos for all other datasets are collected



4


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT



from the Internet and captions being created by humans
have quite a rich structure.



Dataset



Num Num Num Has
video pairs unique YouTube
captions Id



MSRVTT 10k 200k 167k Yes
ActivityNet 14k 70k 69k Yes
LSMDC 101k 101k 101k No

TwitterVines 6.5k 23k 23k No

YouCook2 1.5k 12k 12k Yes

MSVD 1.5k 80k 64k Yes

TGIF 102k 125k 125k No

**Sum above** **236k** **611k** **561k** —
SomethingV2 193k 193k 124k No
**Sum above** **429k** **804k** **685k** —

Table 1: The "Num video" column represents the number
of video clips in the dataset, the "Num pairs" column represents the total number of video-caption pairs, the "Num
unique captions" column represents the number of unique
captions in the dataset.


**3.3** **Intersection**


It is important to extend the training database carefully, not
allowing the addition to the train part of video segments
that already exist in the test part.


To find the intersection between the test part and the train
part we use the two stage filtration. The first stage is to use
the YouTube ID, if it is available. We should not allow to
use in the test and train parts simultaneously any two video
segments sampled from the same video. In the second
stage we compute the similarity score between each video
from the test part and each video from the train part, then
we manually assess the pairs with the highest scores. In
total we assessed more than 100K pairs of the most relevant
segments, see Sec. C.1 for details.


We found the significant overlap between the MSRVTT
1k-A test and train parts, and the similar situation is with
the 1k-B test and train parts and the less significant overlap
is found between the MSRVTT full split test and train
parts. The similar situation is with the ActivityNet train
and validation 1,2 parts.


Additionally we estimate (but did not find) the overlap
between HowTo100M and MSRVTT, and found that it
may be significant. Our approach allows to approximately
estimate the total number of videos in the intersection
without finding the exact intersection, please see the details
in Sec. C.1.2. The similar estimation is for ActivityNet
and Kinetics700, an our approximation shows that there
may be a significant overlap, see all details in Sec. C.1.



MSRVTT + ActivityNet +
LSMDC + TwitterVines +

MALVYMTS

YouCook2 + MSVD + TGIF +
Something to Something V2

Table 2: The left column represents the abbreviate name
for the set of datasets from the right column.


**4** **Experiments**


**4.1** **Architecture**


We use exactly the same neural network architecture as
original MMT [8], our method is significantly based on
their codebase. The difference is in the following: 1. we
use the more aggressive dropout equals to 0.2 for the text
BERT and the video BERT (against the original value of
0.1); 2. we found that the deeper and wider transformer
encoder for a video network gives better results — we use
6 layers and 8 heads for the motion only modality and 9
layers and 8 heads for the motion + audio setting (against
4 layer and 4 head in the original implementation).


**4.2** **Stronger motion experts**


As the input data for MMT is embeddings from experts,
the obvious question can arise: if a better expert is used,
will we have a stronger model? To answer this question
we train MMT on MSRVTT dataset with the only motion
modality. For motion experts we try several architectures
pretrained on different datasets, these models are presented
in Tab. 3. We take the architectures which show the best results on Kinetics 400 benchmark having publicly available
pretrained weights: [38] [7] [36] [35] [9] .


The results in Tab. 3 are made with the same hyperparameters as in [8]. For the train dataset we use only MSRVTT



Abbreviate Composition


M MSRVTT full split
M _c_ MSRVTT full clean split, see Sec. 2.1
M 1k-A MSRVTT 1k-A split
M 1k-B MSRVTT 1k-B split
A ActivityNet
A val1 ActivityNet val1 validation set
A val2 ActivityNet val2 validation set
A _p/r_ ActivityNet paragraph retrieval, see Sec. 2.1
L LSMDC

K Kinetics700

V Twitter Vines

Y YouCook2

HT100M HowTo100M


MSRVTT + ActivityNet +
MALV
LSMDC + TwitterVines



MALVYMT



MSRVTT + ActivityNet +
LSMDC + TwitterVines +

YouCook2 + MSVD + TGIF



5


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT



full clean split. The first line in Tab. 3 represents the motion
feature extractor from the original MMT paper.


As we can see, usually stronger models provide better
results, but not always. Refer to r(2+1)d 152 rows, this
network demonstrates one of the best performance on Kinetics 400 benchmark, but works poorly as motion expert.
Maybe this network is over specialized for Kinetics 400.
More shallow analogue of r(2+1)d 152 is r(2+1)d 34 which
shows much better results.


An interesting observation is that the best results are
achieved with the networks trained in the unsupervised
manner. CLIP and models trained on IG65M outperform
all other models trained on Kinetics in the supervised manner. Another weakly supervised dataset is Sports1M [14].
Models trained on this dataset provide weak embeddings
similar to the weak s3d model trained on Kinetics dataset.
The CLIP [27] (ViT-B/32) image feature extractor with
a large margin outperforms all other models. The model
s3dg MIL-NCE is a video encoder from the work [22].
This network was trained from scratch on HowTo100M

dataset.


As we show in Sec. C Kinetics dataset has an overlap with
MSRVTT dataset, and we don’t know whether it affects to
overfitting or not. Also it is worth to mention that IG65M
and CLIP datasets are not publicly available, so we do not
know if there is an overlap with MSRVTT and other video
retrieval datasets.


For more details about our usage of pretrained video experts please refer to Sec. A.


**4.3** **Datasets combination**


In this section we show our experiments about the combination of different datasets. Nowadays video-caption
datasets are not big enough to capture all real life situations, also some datasets may be biased. The combination
of different datasets may help to tackle this problem.


Our experiments show that the proper combination of
datasets allows to train a single model that can capture
the knowledge from all used datasets. Important thing here
is that in most cases the model trained on the combination
of datasets is better than the model trained on a single
dataset.


In our experiments we combine all datasets presented in
Tab. 5. The important thing is how to sample minibatches
during training. In our experiments we first sample a
dataset, then we uniformly sample a video segment, if
this sampled video segment has more than one caption
we sample a single caption uniformly. Column weight
in Tab. 5 describes the probability of sampling the corresponding dataset. To obtain the probability of sampling
the dataset with the weight _w_ we should divide _w_ by the
sum of all weights.


The weights for all datasets are manually adjusted. It is
important to find a good weight combination, because if



some weight will be larger than needed, this dataset will
be overseen and as a result the performance will be lower
comparing to the optimal case. The opposite case is when
a small weight was selected, this causes the situation when
during training a network does not see the required number
of examples from this dataset.


For experiments in this section we use MMT with the only
motion modality. Embeddings for the motion modality
are computed with irCSN152 pretrained on IG65M. All
configurations are trained with 50 epochs and different
number examples per epoch. The initial learning rate is 5e5. After each epoch we multiply learning rate by 0.95. The
MALVYMTS (see Tab. 2 for abbreviations.) configuration
is trained with 150K examples per epoch. Configurations
with the less number of datasets are trained with the less
number of examples per epoch. The number of examples
per epoch can be represented as a product of 150K by a sum
of normalized weights (weights from Tab. 5 divided by a
sum of all weights) for each dataset (the initial sum equals
to 1): 150 _K_ = 150 _K ×_ ( _p_ MSRVTT + _p_ ActivityNet + _p_ LSMDC +
_p_ Twitter Vines + _p_ YouCook2 + _p_ MSVD + _p_ TGIF + _p_ Something V2 ) . If
some dataset is removed from the training, we remove the
corresponding coefficient from this sum, so the resulting
length will be 150K multiplied by a value less than 1.


As far as we use the configurations M _c_, A, L as the baselines, we need to be sure that the results for these configurations are the optimal values. In addition to the rule
described above we try several values for a number of examples per epoch parameter, and report the results for the
best found value.


Tab. 6 summarizes our experiments on the datasets combination (for more details please refer to Sec. B). The main
point here is that the proper combination of datasets leads
to the best solution.


**4.4** **Final result**


In this section we compare our solution with the prior art.
Our two best solution uses three modalities: the audio, the
motion and the RGB. To fuse modalities we use MMT architecture with 9 layers and 8 heads. As a feature extractor
for the audio stream the vggish [12] network is used. For
the video encoding we use CLIP ViT-B/32 (RGB modality) and irCSN152 (motion modality) pretrained on IG65M
dataset. The details about preprocessing videos for both
networks are presented in Sec. A.


Additionally we report separate results for motion + audio
encoders and RGB + audio encoders because we do not

know whether the IG65M or CLIP train database has a
significant overlap with any of the test datasets or not.


All our models presented in Tab. 4,7 and 8 are trained
based on the pretrain HowTo100M model. We present the
details about pretraining in Sec. E.


The results for MSRVTT are presented in Tab. 7. As
we can see our solution MDMMT(MALVYMTS) L9H8
CLIP+irCSN152+audio significantly outperforms all pre


6


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT


Text _→_ Video
Video expert Dataset
R@1 _↑_ R@5 _↑_ R@10 _↑_ MnR _↓_ MdR _↓_


s3d Kinetics 600 7.7 _±_ 0 _._ 1 24.0 _±_ 0 _._ 2 34.9 _±_ 0 _._ 2 129.6 _±_ 1 _._ 0 23.7 _±_ 0 _._ 5
SlowFast 32x2 R101 Kinetics 600 9.3 _±_ 0 _._ 1 27.5 _±_ 0 _._ 1 39.1 _±_ 0 _._ 1 110.8 _±_ 1 _._ 1 18.7 _±_ 0 _._ 5
ipCSN152 IG65M 9.5 _±_ 0 _._ 1 27.9 _±_ 0 _._ 2 39.6 _±_ 0 _._ 2 106.1 _±_ 1 _._ 1 18.0 _±_ 0 _._ 0
ipCSN152 IG65M _→_ K400 8.3 _±_ 0 _._ 1 25.2 _±_ 0 _._ 1 36.5 _±_ 0 _._ 2 124.3 _±_ 0 _._ 2 21.0 _±_ 0 _._ 0
ipCSN152 Sports1M 7.4 _±_ 0 _._ 2 22.4 _±_ 0 _._ 1 32.7 _±_ 0 _._ 2 140.6 _±_ 1 _._ 0 27.0 _±_ 0 _._ 0
ipCSN152 Sports1M _→_ K400 7.8 _±_ 0 _._ 1 24.2 _±_ 0 _._ 1 35.2 _±_ 0 _._ 1 129.9 _±_ 0 _._ 2 23.0 _±_ 0 _._ 0
irCSN152 IG65M 9.5 _±_ 0 _._ 1 27.9 _±_ 0 _._ 2 39.5 _±_ 0 _._ 2 105.5 _±_ 0 _._ 4 18.0 _±_ 0 _._ 0
irCSN152 IG65M _→_ K400 8.4 _±_ 0 _._ 1 25.3 _±_ 0 _._ 1 36.5 _±_ 0 _._ 2 120.4 _±_ 0 _._ 4 21.0 _±_ 0 _._ 0
irCSN152 Sports1M 6.9 _±_ 0 _._ 1 21.6 _±_ 0 _._ 1 31.6 _±_ 0 _._ 1 141.9 _±_ 0 _._ 4 28.7 _±_ 0 _._ 5
irCSN152 Sports1M _→_ K400 7.7 _±_ 0 _._ 1 24.1 _±_ 0 _._ 1 35.1 _±_ 0 _._ 1 127.6 _±_ 0 _._ 6 23.0 _±_ 0 _._ 0
r(2+1)d 152 IG65M 5.7 _±_ 0 _._ 1 18.5 _±_ 0 _._ 1 27.8 _±_ 0 _._ 1 178.5 _±_ 1 _._ 5 37.7 _±_ 0 _._ 9
r(2+1)d 152 IG65M _→_ K400 5.5 _±_ 0 _._ 1 18.1 _±_ 0 _._ 1 27.3 _±_ 0 _._ 1 184.1 _±_ 1 _._ 2 39.3 _±_ 0 _._ 5
r(2+1)d 152 Sports1M _→_ K400 5.3 _±_ 0 _._ 1 17.3 _±_ 0 _._ 1 26.0 _±_ 0 _._ 1 193.4 _±_ 3 _._ 6 42.3 _±_ 0 _._ 5
r(2+1)d 34 IG65M 9.1 _±_ 0 _._ 2 27.2 _±_ 0 _._ 2 38.7 _±_ 0 _._ 2 108.1 _±_ 0 _._ 0 19.0 _±_ 0 _._ 0
r(2+1)d 34 IG65M _→_ K400 8.2 _±_ 0 _._ 2 25.3 _±_ 0 _._ 3 36.7 _±_ 0 _._ 1 120.8 _±_ 0 _._ 7 21.0 _±_ 0 _._ 0
CLIP CLIP **14.4** _±_ 0 _._ 1 **37.4** _±_ 0 _._ 3 **50.2** _±_ 0 _._ 3 **70.3** _±_ 0 _._ 3 **10.3** _±_ 0 _._ 5
s3dg MIL-NCE HowTo100M 8.6 _±_ 0 _._ 4 26.3 _±_ 0 _._ 5 37.9 _±_ 0 _._ 7 104.4 _±_ 2 _._ 2 19.3 _±_ 0 _._ 5


Table 3: Comparison of the best available pretrain models as the motion experts for MMT. IG65M _→_ K400 means that
model was trained on IG65M and then fine tuned on Kinetics400. Results for each experiment are computed over three
runs with random seeds. The results are reported on MSRVTT full clean split.


ActivityNet text _→_ video
model
R@1 _↑_ R@5 _↑_ R@10 _↑_ MnR _↓_ MdR _↓_


CLIP [27] 0.02 0.06 0.2 2210 2251
MMT (A p/r ) motion+audio [8] 7.3 22.5 31 283.9 30
Ours MDMMT(M _c_ ALVYMTS) L9H8 irCSN152+audio 15.1 _±_ 0 _._ 1 38.3 _±_ 0 _._ 1 51.5 _±_ 0 _._ 3 92.4 _±_ 2 _._ 3 10.0 _±_ 0 _._ 0
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+audio 17.7 _±_ 0 _._ 1 41.6 _±_ 0 _._ 3 54.3 _±_ 0 _._ 2 76.0 _±_ 1 _._ 0 8.3 _±_ 0 _._ 5
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+irCSN152+audio **20.1** _±_ 0 _._ 5 **45.1** _±_ 0 _._ 5 **58.0** _±_ 0 _._ 6 **70.8** _±_ 0 _._ 1 **7.0** _±_ 0 _._ 0


Table 4: Test results on our split (see Sec. 2.1) on ActivityNet.



Dataset Weight


MSRVTT 140
ActivityNet 100
LSMDC 70

Twitter Vines 60

YouCook2 9

MSVD 9

TGIF 102
Something V2 169
Table 5: These datasets were used in our train procedure.
The "Weight" column describes how often we sample examples from the dataset. The probability of obtaining an
example from the dataset with the weight _w_ equals to _w_
divides by a sum of all weights.


vious solutions on all splits: full, 1k-A and 1k-B. Our
solution is better than the previous SotA (on R@5) on
8.7%, 10.5% and 14.4% on full, 1k-A and 1k-B correspondingly. It is also worth to mention that our MDMMT
(using only the motion, the RGB and the audio modalities)



Test Text _→_ Video R@5 _↑_
Dataset
MSRVTT ActivityNet LSMDC


M _c_ 29.0 _±_ 0 _._ 2 _13.4_ _±_ 0 _._ 3 _12.9_ _±_ 0 _._ 6
A _14.7_ _±_ 0 _._ 1 30.9 _±_ 0 _._ 6 _10.4_ _±_ 0 _._ 3
L _8.8_ _±_ 0 _._ 1 _7.2_ _±_ 0 _._ 2 24.7 _±_ 0 _._ 6
M _c_ ALV 32.1 _±_ 0 _._ 1 32.0 _±_ 0 _._ 2 26.5 _±_ 0 _._ 7
M _c_ ALVYMT 33.8 _±_ 0 _._ 1 32.3 _±_ 0 _._ 2 27.3 _±_ 0 _._ 4
M _c_ ALVYMTS **34.5** _±_ 0 _._ 1 **32.4** _±_ 0 _._ 5 **27.4** _±_ 0 _._ 6

Table 6: See abbreviations for the first column in Tab. 2.
The first three rows M _c_,A,L report the quality of models
trained on a single domain, and tested on other domains.
_Italic_ means that the model did not see data from this domain during training. In this table the only motion modality
(irCSN152) is used.


outperforms the original MMT (the motion, the RGB and
the audio and 4 other modalities) by 8.7%, 10.5% and 14.4
(R@5) on full, 1k-A and 1k-B correspondingly.


We also report the results for the original CLIP [27]. The
CLIP model has an image encoder and a text encoder, both



7


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT


pretrained in an unsupervised way. To test the CLIP model
we take a single frame from the middle of the video (this
is the original testing protocol for CLIP). The row CLIP
agg [26] represents the usage of CLIP model with several
frames using some specific aggregation procedure from
this work.


In Tab. 8 we report the results on LSMDC. On this benchmark we outperform the previous SotA solution by 8.6%.


As we mention in Sec. 2.1 we do not use the standard
ActivityNet paragraph retrieval test protocol. Instead we
use the text to video retrieval protocol. To compare our
solution with the previous work we take the previous SotA
approach (MMT) in text to video retrieval and test it on
our split. The results are reported in Tab. 4. Our solution
outperforms MMT by 22.6%. The row MMT (A p/r ) motion+audio means that this network was trained only on
ActivityNet dataset with the paragraph retrieval mode. It is
also worth to mention that CLIP shows very bad results on
this benchmark. We try to aggregate with the mean pooling of 2, 4 and 16 uniformly taken embeddings, take the
first 10, 20 and 70 words from a caption, and no method
improves the results.


The important property of our model is that we train a
single model and test it on different test sets. The authors
of previous SotA approach (MMT) trained three different
models for MSRVTT, ActivityNet and LSMDC, while in
Tab. 6 we show that the model trained in such a manner
has poor generalization and can show good performance
on the test part of the dataset _X_ if and only if it was trained
on the train part of the dataset _X_ .


**5** **Conclusions and Discussion**


In this work we present a new text to video retrieval stateof-the-art model on MSRVTT and LSMDC benchmarks.
We do not use ActivityNet dataset in the paragraph retrieval
mode as many previous works do, so we can’t compare
with them. But we show that on ActivityNet in the video
retrieval mode we outperform the previous state-of-the-art
model (MMT) by a large margin. Our model has captured
knowledge from many video caption datasets, thus it is
able to show the best results on several datasets at the same
time without finetuning.


We also present a practical approach to find the overlap
between two different video datasets. Using this approach
we find the overlap between several datasets. Especially
we find a large overlap between the MSRVTT test and
train parts, and between the ActivityNet test and train
parts. Removing this overlap from the MSRVTT train part
significantly decreases the performance of previous best
models on MSRVTT benchmark.


_Acknowledgments._ We would like to thank Andrey
Ivanyuta and other colleagues from Intelligent Systems
and Data Science Lab for helping to find the overlap between datasets.


8


model



MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT


MSRVTT text _→_ video
R@1 _↑_ R@5 _↑_ R@10 _↑_ MnR _↓_ MdR _↓_



Random baseline 0.0 0.2 0.3 1500 1500

VSE [24] 5.0 16.4 24.6 — 47
VSE++ [24] 5.7 17.1 24.8 — 65
Multi Cues [24] 7.0 20.9 29.7 — 38
W2VV [5] 6.1 18.7 27.5 — 45
Dual Enc. [6] 7.7 22.0 31.8 — 32
CE [19] 10.0 _±_ 0 _._ 1 29.0 _±_ 0 _._ 3 41.2 _±_ 0 _._ 2 86.8 _±_ 0 _._ 3 16.0 _±_ 0 _._ 0
MMT (M) 7mod [8] 10.7 _±_ 0 _._ 2 31.1 _±_ 0 _._ 1 43.4 _±_ 0 _._ 2 88.2 _±_ 0 _._ 7 15.0 _±_ 0 _._ 0
CLIP [27] 15.1 31.8 40.4 184.2 21
CLIP agg [26] 21.5 41.1 50.4 — **4**
Ours MDMMT(MALVYMTS) L9H8 irCSN152+audio 15.7 _±_ 0 _._ 1 38.8 _±_ 0 _._ 1 51.1 _±_ 0 _._ 2 76.0 _±_ 0 _._ 7 10.0 _±_ 0 _._ 0
Ours MDMMT(MALVYMTS) L9H8 CLIP+audio 21.7 _±_ 0 _._ 2 47.6 _±_ 0 _._ 3 59.8 _±_ 0 _._ 1 55.9 _±_ 0 _._ 2 6.0 _±_ 0 _._ 0
Ours MDMMT(MALVYMTS) L9H8 CLIP+irCSN152+audio **23.1** _±_ 0 _._ 1 **49.8** _±_ 0 _._ 1 **61.8** _±_ 0 _._ 1 **52.8** _±_ 0 _._ 2 6.0 _±_ 0 _._ 0

MMT (M _c_ ) 7mod [8] 10.4 _±_ 0 _._ 1 30.2 _±_ 0 _._ 4 42.3 _±_ 0 _._ 2 89.4 _±_ 0 _._ 6 15.7 _±_ 0 _._ 5

Ours MDMMT(M _c_ ALVYMTS) L9H8 irCSN152+audio 15.8 _±_ 0 _._ 1 38.9 _±_ 0 _._ 1 51.0 _±_ 0 _._ 1 76.4 _±_ 0 _._ 5 10.0 _±_ 0 _._ 0
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+audio 21.5 _±_ 0 _._ 1 47.4 _±_ 0 _._ 2 59.6 _±_ 0 _._ 1 57.7 _±_ 0 _._ 4 6.0 _±_ 0 _._ 0
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+irCSN152+audio **22.8** _±_ 0 _._ 2 **49.5** _±_ 0 _._ 1 **61.5** _±_ 0 _._ 1 **53.8** _±_ 0 _._ 3 **6.0** _±_ 0 _._ 0

Random baseline 0.1 0.5 1.0 500.0 500.0



Random baseline



Random baseline 0.1 0.5 1.0 500.0 500.0

JSFusion [40] 10.2 31.2 43.2 — 13
E2E [22] 9.9 24.0 32.4 — 29.5
HT [23] 14.9 40.2 52.8 — 9
CE [19] 20.9 _±_ 1 _._ 2 48.8 _±_ 0 _._ 6 62.4 _±_ 0 _._ 8 28.2 _±_ 0 _._ 8 6.0 _±_ 0 _._ 0
CLIP [27] 22.5 44.3 53.7 61.7 8
MMT (M 1k-A ) 7mod [8] 26.6 _±_ 1 _._ 0 57.1 _±_ 1 _._ 0 69.6 _±_ 0 _._ 2 24.0 _±_ 0 _._ 8 4.0 _±_ 0 _._ 0
AVLnet[30] 27.1 55.6 66.6 — 4
SSB [25] 30.1 58.5 69.3 — 3.0
CLIP agg [26] 31.2 53.7 64.2 — 4
Ours MDMMT(M 1k-A ALVYMTS) L9H8 irCSN152+audio 31.3 _±_ 0 _._ 1 60.4 _±_ 1 _._ 2 71.8 _±_ 1 _._ 0 24.0 _±_ 0 _._ 4 3.0 _±_ 0 _._ 0
Ours MDMMT(M 1k-A ALVYMTS) L9H8 CLIP+audio 38.9 _±_ 1 _._ 0 68.3 _±_ 0 _._ 7 78.8 _±_ 0 _._ 2 17.3 _±_ 0 _._ 5 **2.0** _±_ 0 _._ 0
Ours MDMMT(M 1k-A ALVYMTS) L9H8 CLIP+irCSN152+audio **38.9** _±_ 0 _._ 6 **69.0** _±_ 0 _._ 1 **79.7** _±_ 0 _._ 6 **16.5** _±_ 0 _._ 4 **2.0** _±_ 0 _._ 0

Random baseline 0.1 0.5 1.0 500.0 500.0

MEE [21] 13.6 37.9 51.0 — 10.0
JPose [37] 14.3 38.1 53.0 — 9
MEE-COCO [21] 14.2 39.2 53.8 — 9.0
CE [19] 18.2 _±_ 0 _._ 7 46.0 _±_ 0 _._ 4 60.7 _±_ 0 _._ 2 35.3 _±_ 1 _._ 1 7.0 _±_ 0 _._ 0
MMT (M 1k-B ) 7mod [8] 24.5 _±_ 0 _._ 5 54.4 _±_ 0 _._ 8 68.0 _±_ 0 _._ 5 26.6 _±_ 0 _._ 2 4.7 _±_ 0 _._ 5
CLIP [27] 24.5 46.2 56.8 60.9 7
Ours MDMMT(M 1k-B ALVYMTS) L9H8 irCSN152+audio 28.8 _±_ 0 _._ 9 58.8 _±_ 0 _._ 3 71.2 _±_ 0 _._ 3 28.5 _±_ 0 _._ 5 3.7 _±_ 0 _._ 5
Ours MDMMT(M 1k-B ALVYMTS) L9H8 CLIP+audio 35.1 _±_ 0 _._ 1 66.5 _±_ 0 _._ 9 77.6 _±_ 0 _._ 3 21.5 _±_ 0 _._ 4 2.7 _±_ 0 _._ 5
Ours MDMMT(M 1k-B ALVYMTS) L9H8 CLIP+irCSN152+audio **37.4** _±_ 1 _._ 5 **68.8** _±_ 0 _._ 4 **79.4** _±_ 0 _._ 4 **21.3** _±_ 0 _._ 4 **2.0** _±_ 0 _._ 0



Table 7: Results on MSRVTT dataset.


LSMDC text _→_ video
model
R@1 _↑_ R@5 _↑_ R@10 _↑_ MnR _↓_ MdR _↓_


CT-SAN [41] 5.1 16.3 25.2 — 46
JSFusion [40] 9.1 21.2 34.1 — 36
MEE [21] 9.3 25.1 33.4 — 27
MEE-COCO [21] 10.1 25.6 34.6 — 27
CE [19] 11.2 _±_ 0 _._ 4 26.9 _±_ 1 _._ 1 34.8 _±_ 2 _._ 0 96.8 _±_ 5 _._ 0 25.3 _±_ 3 _._ 1
CLIP agg [26] 11.3 22.7 29.2 — 56.5
CLIP [27] 12.4 23.7 31.0 142.5 45
MMT (L) 7mod [8] 12.9 _±_ 0 _._ 1 29.9 _±_ 0 _._ 7 40.1 _±_ 0 _._ 8 75.0 _±_ 1 _._ 2 19.3 _±_ 0 _._ 2
Ours MDMMT(M _c_ ALVYMTS) L9H8 irCSN152+audio 13.1 _±_ 0 _._ 5 31.3 _±_ 0 _._ 3 40.1 _±_ 0 _._ 0 74.5 _±_ 0 _._ 7 19.3 _±_ 0 _._ 5
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+audio 17.2 _±_ 0 _._ 6 34.9 _±_ 0 _._ 4 45.3 _±_ 1 _._ 0 65.6 _±_ 0 _._ 8 14.0 _±_ 0 _._ 8
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+irCSN152+audio **18.8** _±_ 0 _._ 7 **38.5** _±_ 0 _._ 4 **47.9** _±_ 0 _._ 7 **58.0** _±_ 1 _._ 1 **12.3** _±_ 0 _._ 5


Table 8: Test results on LSMDC public test (1k video)


9


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT



**References**


[1] George Awad et al. “TRECVID 2020: comprehensive campaign for evaluating video retrieval tasks
across multiple application domains”. In: _Proceed-_
_ings of TRECVID 2020_ . NIST, USA. 2020.

[2] David Chen and William Dolan. “Collecting Highly
Parallel Data for Paraphrase Evaluation”. In: _Pro-_
_ceedings of the 49th Annual Meeting of the Associ-_
_ation for Computational Linguistics: Human Lan-_
_guage Technologies_ . Portland, Oregon, USA: Association for Computational Linguistics, June 2011,
pp. 190–200. URL : `[https://www.aclweb.org/](https://www.aclweb.org/anthology/P11-1020)`
`[anthology/P11-1020](https://www.aclweb.org/anthology/P11-1020)` .

[3] J. Deng et al. “ImageNet: A Large-Scale Hierarchical Image Database”. In: _CVPR09_ . 2009.

[4] Jacob Devlin et al. _BERT: Pre-training of Deep Bidi-_
_rectional Transformers for Language Understand-_
_ing_ . 2019. arXiv: `[1810.04805 [cs.CL]](https://arxiv.org/abs/1810.04805)` .

[5] Jianfeng Dong, Xirong Li, and Cees G. M. Snoek.
“Predicting Visual Features From Text for Image and
Video Caption Retrieval”. In: _IEEE Transactions on_
_Multimedia_ 20.12 (2018), 3377–3388. ISSN : 19410077. DOI : `[10.1109/tmm.2018.2832602](https://doi.org/10.1109/tmm.2018.2832602)` . URL :
```
   http://dx.doi.org/10.1109/TMM.2018.
```

`[2832602](http://dx.doi.org/10.1109/TMM.2018.2832602)` .

[6] Jianfeng Dong et al. _Dual Encoding for Zero-_
_Example Video Retrieval_ . 2019. arXiv: `[1809.06181](https://arxiv.org/abs/1809.06181)`

`[[cs.CV]](https://arxiv.org/abs/1809.06181)` .

[7] Christoph Feichtenhofer et al. _SlowFast Networks_
_for Video Recognition_ . 2019. arXiv: `[1812.03982](https://arxiv.org/abs/1812.03982)`

`[[cs.CV]](https://arxiv.org/abs/1812.03982)` .

[8] Valentin Gabeur et al. _Multi-modal Transformer_
_for Video Retrieval_ . 2020. arXiv: `[2007 . 10639](https://arxiv.org/abs/2007.10639)`

`[[cs.CV]](https://arxiv.org/abs/2007.10639)` .

[9] Deepti Ghadiyaram et al. _Large-scale weakly-_
_supervised pre-training for video action recognition_ .
2019. arXiv: `[1905.00561 [cs.CV]](https://arxiv.org/abs/1905.00561)` .

[10] Raghav Goyal et al. _The "something something"_
_video database for learning and evaluating vi-_
_sual common sense_ . 2017. arXiv: `[1706 . 04261](https://arxiv.org/abs/1706.04261)`

`[[cs.CV]](https://arxiv.org/abs/1706.04261)` .

[11] Kaiming He et al. _Deep Residual Learning for_
_Image Recognition_ . 2015. arXiv: `[1512 . 03385](https://arxiv.org/abs/1512.03385)`

`[[cs.CV]](https://arxiv.org/abs/1512.03385)` .

[12] Shawn Hershey et al. _CNN Architectures for Large-_
_Scale Audio Classification_ . 2017. arXiv: `[1609 .](https://arxiv.org/abs/1609.09430)`
`[09430 [cs.SD]](https://arxiv.org/abs/1609.09430)` .

[13] Andrej Karpathy, Armand Joulin, and Li Fei-Fei.
“Deep Fragment Embeddings for Bidirectional Image Sentence Mapping”. In: _Proceedings of the 27th_
_International Conference on Neural Information_
_Processing Systems - Volume 2_ . NIPS’14. Montreal,
Canada: MIT Press, 2014, 1889–1897.

[14] Andrej Karpathy et al. “Large-scale Video Classification with Convolutional Neural Networks”. In:
_CVPR_ . 2014.


10




[15] Will Kay et al. _The Kinetics Human Action Video_
_Dataset_ . 2017. arXiv: `[1705.06950 [cs.CV]](https://arxiv.org/abs/1705.06950)` .

[16] Giorgos Kordopatis-Zilos et al. “Near-duplicate
video retrieval with deep metric learning”. In: _Pro-_
_ceedings of the IEEE International Conference on_
_Computer Vision Workshops_ . 2017, pp. 347–356.

[17] Ranjay Krishna et al. “Dense-Captioning Events in
Videos”. In: _International Conference on Computer_
_Vision (ICCV)_ . 2017.

[18] Yuncheng Li et al. “TGIF: A New Dataset and
Benchmark on Animated GIF Description”. In: _The_
_IEEE Conference on Computer Vision and Pattern_
_Recognition (CVPR)_ . 2016.

[19] Yang Liu et al. _Use What You Have: Video Retrieval_
_Using Representations From Collaborative Experts_ .
2020. arXiv: `[1907.13487 [cs.CV]](https://arxiv.org/abs/1907.13487)` .

[20] Dhruv Kumar Mahajan et al. “Exploring the Limits of Weakly Supervised Pretraining”. In: _ECCV_ .
2018.

[21] Antoine Miech, Ivan Laptev, and Josef Sivic. _Learn-_
_ing a Text-Video Embedding from Incomplete and_
_Heterogeneous Data_ . 2020. arXiv: `[1804.02516](https://arxiv.org/abs/1804.02516)`

`[[cs.CV]](https://arxiv.org/abs/1804.02516)` .

[22] Antoine Miech et al. _End-to-End Learning of Vi-_
_sual Representations from Uncurated Instructional_
_Videos_ . 2020. arXiv: `[1912.06430 [cs.CV]](https://arxiv.org/abs/1912.06430)` .

[23] Antoine Miech et al. “HowTo100M: Learning a
Text-Video Embedding by Watching Hundred Million Narrated Video Clips”. In: _ICCV_ . 2019.

[24] Niluthpol Chowdhury Mithun et al. “Learning joint
embedding with multimodal cues for cross-modal
video-text retrieval”. In: _Proceedings of the 2018_
_ACM on International Conference on Multimedia_
_Retrieval_ . 2018, pp. 19–27.

[25] Mandela Patrick et al. _Support-set bottlenecks for_
_video-text representation learning_ . 2021. arXiv:
`[2010.02824 [cs.CV]](https://arxiv.org/abs/2010.02824)` .

[26] Jesús Andrés Portillo-Quintero, José Carlos OrtizBayliss, and Hugo Terashima-Marín. _A Straightfor-_
_ward Framework For Video Retrieval Using CLIP_ .
2021. arXiv: `[2102.12443 [cs.CV]](https://arxiv.org/abs/2102.12443)` .

[27] Alec Radford et al. “Learning Transferable Visual
Models From Natural Language Supervision”. In:
_Image_ 2 (), T2.

[28] Anna Rohrbach et al. _A Dataset for Movie Descrip-_
_tion_ . 2015. arXiv: `[1501.02530 [cs.CV]](https://arxiv.org/abs/1501.02530)` .

[29] Anna Rohrbach et al. _Movie Description_ . 2016.
arXiv: `[1605.03705 [cs.CV]](https://arxiv.org/abs/1605.03705)` .

[30] Andrew Rouditchenko et al. _AVLnet: Learning_
_Audio-Visual Language Representations from In-_
_structional Videos_ . 2020. arXiv: `[2006 . 09199](https://arxiv.org/abs/2006.09199)`

`[[cs.CV]](https://arxiv.org/abs/2006.09199)` .

[31] Lucas Smaira et al. _A Short Note on the Kinetics-_
_700-2020 Human Action Dataset_ . 2020. arXiv:
`[2010.10864 [cs.CV]](https://arxiv.org/abs/2010.10864)` .


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT




[32] Chen Sun et al. _VideoBERT: A Joint Model for_
_Video and Language Representation Learning_ . 2019.
arXiv: `[1904.01766 [cs.CV]](https://arxiv.org/abs/1904.01766)` .

[33] Atousa Torabi, Niket Tandon, and Leonid Sigal. _Learning Language-Visual Embedding for_
_Movie Understanding with Natural-Language_ . 2016.
arXiv: `[1609.08124 [cs.CV]](https://arxiv.org/abs/1609.08124)` .

[34] Atousa Torabi et al. _Using Descriptive Video Ser-_
_vices to Create a Large Data Source for Video_
_Annotation Research_ . 2015. arXiv: `[1503.01070](https://arxiv.org/abs/1503.01070)`

`[[cs.CV]](https://arxiv.org/abs/1503.01070)` .

[35] Du Tran et al. _A Closer Look at Spatiotemporal_
_Convolutions for Action Recognition_ . 2018. arXiv:
`[1711.11248 [cs.CV]](https://arxiv.org/abs/1711.11248)` .

[36] Du Tran et al. _Video Classification with Channel-_
_Separated Convolutional Networks_ . 2019. arXiv:
`[1904.02811 [cs.CV]](https://arxiv.org/abs/1904.02811)` .

[37] Michael Wray et al. _Fine-Grained Action Re-_
_trieval Through Multiple Parts-of-Speech Embed-_
_dings_ . 2019. arXiv: `[1908.03477 [cs.CV]](https://arxiv.org/abs/1908.03477)` .

[38] Saining Xie et al. _Rethinking Spatiotemporal Fea-_
_ture Learning: Speed-Accuracy Trade-offs in Video_
_Classification_ . 2018. arXiv: `[1712.04851 [cs.CV]](https://arxiv.org/abs/1712.04851)` .

[39] Jun Xu et al. “MSR-VTT: A Large Video Description Dataset for Bridging Video and Language”. In:
IEEE International Conference on Computer Vision
and Pattern Recognition (CVPR), 2016.

[40] Youngjae Yu, Jongseok Kim, and Gunhee Kim. _A_
_Joint Sequence Fusion Model for Video Question_
_Answering and Retrieval_ . 2018. arXiv: `[1808.02559](https://arxiv.org/abs/1808.02559)`

`[[cs.CV]](https://arxiv.org/abs/1808.02559)` .

[41] Youngjae Yu et al. _End-to-end Concept Word Detec-_
_tion for Video Captioning, Retrieval, and Question_
_Answering_ . 2017. arXiv: `[1610.02947 [cs.CV]](https://arxiv.org/abs/1610.02947)` .

[42] Bolei Zhou et al. “Places: A 10 million Image
Database for Scene Recognition”. In: _IEEE Transac-_
_tions on Pattern Analysis and Machine Intelligence_
(2017).

[43] Luowei Zhou, Nathan Louis, and Jason J. Corso.
_Weakly-Supervised Video Object Grounding from_
_Text by Loss Weighting and Object Interaction_ . 2018.
arXiv: `[1805.02834 [cs.CV]](https://arxiv.org/abs/1805.02834)` .


**A** **Pretrain experts usage**


The important data preparing stage is how to sample frames
from a video to achieve the best performance. For s3d experiments the input video is converted to 30 frames per second, for all other experiments we convert the input video
to 32 frames per second. As a result we compute a single
embedding for each second, having 1 second window with
1 second shift (no overlapping).


The input frame size is important. We use the different
sizes for the different models. For each model we use the
recommended input size. For s3d we resize a video to
256 on the short side and then take a 224x224 center crop.



For SlowFast 32x2 R101 we resize a video to 256 on the
short side and then take a 256x256 center crop. For ipCSN
152 and irCSN 152 we resize a video to 224 on the short
side and take a 224x224 center crop. For r(2+1)d 152 and
r(2+1)d 34 we resize a video to 112 on the short side and
then take a 112x112 center crop.


Pretrained models for ipCSN, irCSN and r(2+1)d are available here [1], for SlowFast 32x2 R101 here [2], and for s3d
here [3] .


For the CLIP model [27] we resize a video to 224 on the
short side and take a center crop, then we extract 1 frame
per second. We use a publicly available image encoder.
We do not use the text encoder from CLIP.


Model s3dg MIL-NCE is a video encoder from the
work [22]. This network was trained from scratch on
HowTo100M dataset. For this network we resize the input
video stream to the size of 228x228 pixels, then take a
center crop.


**B** **Datasets combination**


In Fig. 3,4,5 we present 6 models. Abbreviations M _c_ ALV,
M _c_ ALVYMTS and M _c_ ALVYMTS represent the same
three models on these figures. The first model, called
M _c_, is trained on the MSRVTT full clean split only, the
second one, called A, is trained on ActivityNet only.
And the third model, called L, is trained on LSMDC
only. These three models are taken as baselines. Adding
more datasets should be not worse than these baseline.
The forth model is called M _c_ ALV. This model is trained
on the combination of MSRVTT, ActivityNet, LSMDC
and TwitterVines. As we can see M _c_ _→_ M _c_ ALV gives
+3.07% on MSRVTT (full clean split), A _→_ M _c_ ALV gives
+1.06% on ActivityNet, and L _→_ M _c_ ALV gives +1.77% on
LSMDC. The next model is called M _c_ ALVYMT and it is
trained on combination of MSRVTT, ActivityNet, LSMDC,
TwitterVines, YouCook2, MSVD, TGIF. The transitions
M _c_ _→_ M _c_ ALVYMT, A _→_ M _c_ ALVYMT, L _→_ M _c_ ALVYMT
give +4.85%, +1.45% and +2.63% correspondingly. The
last transitions M _c_ _→_ M _c_ ALVYMTS, A _→_ M _c_ ALVYMTS,
L _→_ M _c_ ALVYMTS slightly improve the performance on
ActivityNet and LSMDC and significantly improve the
performance on MSRVTT. Finally, the combination of all
datasets gives +5.5% for MSRVTT, +1.47% for ActivityNet and +2.74% for LSMDC.


**C** **Test and train intersection**


In this section we present our analysis of overlapping of
popular text to video datasets. Since we compose the train
dataset from several different datasets it is important to be
sure that there is no the same video segment in the train
part and in the test part. Our aim is to find the overlap


1. https://github.com/facebookresearch/VMZ
2. https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md
3. https://github.com/princeton-vl/d3dhelper/blob/master/d3d_helper.ipynb



11


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT







34.5


33.8


32.1


29.0



27.4


26.5


24.7



M _c_



M _c_ ALV



M _c_ ALVYMT



M _c_ ALVYMTS



M _c_ ALVYMT M _c_ ALVYMTS



L



M _c_ ALV



Figure 3: Increasing R@5 metric on the MSRVTT full
clean split while enriching the train part.


32.4


32.0


30.9


|ActivityNet R@5 ↑|Col2|
|---|---|
|||
|||
|||



M _c_ ALVYMT M _c_ ALVYMTS



A



M _c_ ALV



Figure 4: Increasing R@5 metric on the ActivityNet test
set while enriching the train part.


between the train part of used datasets — MSRVTT, ActivityNet, LSMDC, YouCook2, MSVD, TGIF, TwitterVines,
HowTo100M, Kinetics700 and the test parts of MSRVTT,
ActivityNet and LSMDC, and then to remove found duplicates from the train parts.


Note that for training we use Something to Something V2
dataset, but we do not try to find overlap between it and
test datasets because this dataset is artificially created, thus
the probability to find duplicates is very low.


We decided to find the overlap only for MSRVTT, ActivityNet and LSMDC because these are the most popular
datasets and we do not have enough human resources to
find the overlap for the test part of all other datasets.


Our cleaning method consists of two stages. The first stage
is to match video segments by the YouTube ID (if the ID is



Figure 5: Increasing R@5 metric on the LSMDC test set
while enriching the train part.


available) and remove from train parts all video segments
that have the corresponding pair in test parts. In Tab. 1
the information about the availability of YouTube IDs in
datasets is presented. We collect the YouTube ID for all
videos from MSRVTT full test and ActivityNet validation
1,2 and remove corresponding video segments from the
train part.


The second stage is based on matching frames by embeddings. For each video we compute several embeddings
then we compute the similarity between each video from
the train part and the test part. After we manually assess
several thousands of video segments with highest scores
for each pair of datasets. Then we extend found duplicates
by either the YouTube ID or the internal dataset ID. This
means that if a video _V_ 1 is marked as a duplicate and a
video _V_ 2 is not marked as a duplicate, but they have the
same YouTube ID or same internal dataset ID, we will
remove _V_ 1 and _V_ 2 from the train part. In case of LSMDC
we do not have the YouTube ID, but have the name of the
movie from which the video segment was taken, so if a
video segment _V_ 1 is marked as a duplicate, we remove
all segments taken from the movie of _V_ 1 . The detailed
description of the second stage is described in Sec. C.1.


Surprisingly we found that the MSRVTT test has a significant overlap with the MSRVTT train part. This problem is
relevant for the full, 1k-A and 1k-B splits. The ActivityNet
dataset suffers from the same problem.


For large datasets like HowTo100M and Kinetics700 we
can not find the whole intersection, but we estimate the
approximate number of videos in the intersection. We
found that HowTo100M may have about 300 (10% of the
MSRVTT full test part) video segments that can be in the
MSRVTT full test part.


The similar situation is about Kinetics700 and ActivityNet
datasets. Kinetics700 may have approximately 500-600



12


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT



video segments (10% of the ActivityNet test) that may have
duplicates in ActivityNet validation 1,2. Another problem
with the Kinetics dataset is that many motion models are
pretrained on it.


This circumstance means that researchers should carefully
use HowTo100M and Kinetics700 along with MSRVTT
and ActivityNet correspondingly, because for today we
don’t know whether a neural network overfits for some
portion of this intersection or not.


All duplicates can be considered as two groups of pairs.
Pairs from the first group have the same videos, but different brightness, aspect ratio, size, presence/absence of
a logo and so on. The second group has pairs with quite
similar videos, for example it can be the same person on
the same background, doing the same things, but wearing
different clothes. We think that it is better to remove such
videos from the train part to prevent overfitting. Several
found examples are presented in Fig. 6.


**C.1** **Near duplicate video search**


**C.1.1** **Approach**


In this section we explain our approach that is used to find
the same or quite similar video segments in test and train
parts.


Suppose we have two sets of videos _Q_ = _{q_ 1 _, ...., q_ _k_ _}_ and
_G_ = _{g_ 1 _, ..., g_ _n_ _}_ called the query set and the gallery set.
We want to find all pairs ( _q_ _i_ _, g_ _j_ ) where _q_ _i_ and _g_ _j_ have a
common video segment.


From each _q_ _i_ and _g_ _j_ we extract 1 frame per second. Each
video is then represented by a sequence of pictures: _q_ _i_ =

[ _q_ _i_ [1] _[, ...., q]_ _i_ _[s]_ _[i]_ []] [ and] _[ g]_ _[j]_ [ = [] _[g]_ _j_ [1] _[, ..., g]_ _j_ _[p]_ _[j]_ []] [. Then a 2D pretrained]
neural network is used to extract features from each image:
_q_ ¯ _a_ _[b]_ [=] `[ neuralnet]` [(] _[q]_ _a_ _[b]_ [)][ and][ ¯] _[g]_ _a_ _[b]_ [=] `[ neuralnet]` [(] _[g]_ _a_ _[b]_ [)][.]


Then we compute the matrix of cosines between the fea
tures from Q and G: _s_ _[ab]_ _ij_ [=] _||q_ ¯ _<_ _a_ _[b]_ _q||_ ¯ _i_ _[a]_ 2 _[,]_ _||_ _[g]_ [¯] _g_ ¯ _j_ _[b]_ _a_ _[b]_ _[>]_ _||_ 2 [.]


Now each pair ( _q_ _i_ _, g_ _j_ ) is represented by the matrix:


_g_ _j_ [1] _..._ _g_ _j_ _[p]_ _[j]_
_q_ _i_ [1] _s_ [11] _ij_ _..._ _s_ _ij_ ~~[1]~~ ~~_[p]_~~ _[j]_

_..._
_q_ _i_ _[s]_ _[i]_ _s_ _[s]_ _ij_ _[i]_ [1] _..._ _s_ _[s]_ _ij_ _[i]_ _[p]_ _[j]_


Suppose that videos _q_ _i_ and _g_ _j_ are intersected at time moments _t_ _q_ and _t_ _g_, it is naturally to assume that the next several seconds _t_ _q_ +1 _, ..., t_ _q_ + _K −_ 1 and _t_ _g_ +1 _, ..., t_ _g_ + _K −_ 1
( _K ≤_ min( _s_ _i_ _, p_ _j_ ) ) represent the same video segment.
Motivated by this fact we compute the mean cosine for
each interval of K seconds (we use K=4): _S_ _ij_ _[t]_ _[q]_ _[t]_ _[j]_ =

_s_ _tqtgij_ + _..._ + _s_ _itqj_ + _K−_ 1 _,tg_ + _K−_ 1 . The sum in the numerator is

_K_
the sum of diagonal elements started with _s_ _[t]_ _ij_ _[q]_ _[t]_ _[j]_ .


We define the intersection score between ( _q_ _i_ _, g_ _j_ ) as


13



**S** **ij** = _a_ =1 max _,...,s_ _i_ _−K_ _S_ _ij_ _[ab]_
_b_ =1 _,...,p_ _j_ _−K_


and the corresponding video segments as


( **a** _,_ **a** + _K_ ) _,_ ( **b** _,_ **b** + _K_ )


where



**a** _,_ **b** = argmax
_a_ =1 _,...,s_ _i_ _−K_
_b_ =1 _,...,p_ _j_ _−K_



_S_ _[ab]_
_ij_



Finally we sorted all **S** **ij** in the descending order and manually assess candidate pairs.


**C.1.2** **Number of pairs to assess**


Suppose we search duplicates in datasets Q and G and we
have seen N pairs with the highest scores and find M pairs
with duplicates. The important question is: what is the
total number of duplicates and how many percents of them
have we found.


For each pair of Q and G we construct the following test
procedure. The first step is to augment Q, and let us call
the result of augmentation as Q [ˆ] . To augment a dataset we
apply two transformations: 1. we randomly crop a side of
each video, where each side can be 70%–100% of original
side length (aspect ratio can be changed); 2. we randomly
shift the start of the video by a random value between 0
and 1 seconds.


Having Q, Q [ˆ] and G we compute sets of positive and negative scores: Pos and Neg. The Pos is the set of scores
between i-th video from Q and the corresponding augmented video from Q [ˆ] . Neg is the set of scores between
each video from Q and G. Having Pos and Neg sets we can
plot a curve, where _x_ axis represents the fraction of found
pairs with duplicates and Y axis represents the number of
negative pairs that we need to assess to find fraction _x_ of
positive pairs, call this curve _F_ ( _x_ ) . We present the algorithm that computes _F_ using Pos and Neg sets in Lst. 1.
Suppose we have seen _N_ + _M_ pairs and have found _M_
pairs with duplicates. The total number of pairs with duplicates can be estimated as _M/F_ _[−]_ [1] ( _N_ ) . By the definition
_F_ ( _x_ ) connects the fraction of found positive pairs with the
number of seen negative pairs. The value _F_ _[−]_ [1] ( _N_ ) represents approximation of the fraction of found positive pairs.
So if we know, that _M_ is approximately 100 _∗F_ _[−]_ [1] ( _N_ )% of
positive pairs, then we can approximately compute 100%
of positive pairs as _M/F_ _[−]_ [1] ( _N_ ).


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT


Figure 6: The left image is taken from the MSRVTT test split and the right one from MSRVTT Train. The numbers in
the upper left corner represent the MSRVTT video ID. The faces are blurred in order to avoid legal claims.


```
# first element is highest
P = np.sort(P)[:: -1] # Pos
N = np.sort(N)[:: -1] # Neg
xs = []
ys = []
for x, p in enumerate(P):
     # how many negative scores
     # greater than p ?
     j = np.searchsorted(N, p)
     xs.append(x)
     ys.append(j)

```

Listing 1: Numpy pseudocode for building the search curve
_F_ ( _x_ )


14



**C.1.3** **Best 2D feature extractor**


The key component of a duplicate search system is a feature extractor. A good feature extractor significantly reduces the number of pairs for manual assessment. To
compare different 2D feature extractors we use the following test procedure. The test consists of two datasets.
The first dataset is the train part from the MSRVTT full
split. The second dataset is random 596k videos from the
HowTo100M dataset. From each video of the taken part of
HowTo100M we take a random 30 seconds segment. We
apply random augmentation to MSRVTT, as described
in Sec. C.1.2. Define MSRVTT as _Q_, the augmented


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT



MSRVTT dataset as _Q_ [ˆ] and the taken part of HowTo100M
as G. For each feature extractor we compute curve _F_ ( _x_ ),
as described in Sec. C.1.2.


The best expert has the lowest curve. For example, if we
want to find 95% of duplicates, we should see many of
candidates, some of them are duplicates, but majority of
them are not. So, the value _F_ (0 _._ 95) is the approximation
of how many not duplicates we need to see to find 95% of
duplicates. Ideally _F_ (0 _._ 95) = 0, where all seen candidates
are duplicates. So, a lower value _F_ (0 _._ 95) requires to see
less number of false candidates, that’s why the lower curve
is better.


We consider several feature extractors: resnet18 and
resnet101 [11] pretrained on ImageNet [3], resnet50
pretrained on Places365 [42] and resnext101-32x8d,
resnext101-32x32d, resnext101-32x48d pretrained on one
billion images from Instagram [20] and finetuned on ImageNet. We report search curves _F_ ( _x_ ) for these pretrained
networks in Fig. 7.


There exist networks [27] [16] trained especially for match
the duplicate frames or video segments, but they are not
publicly available.



10 [6]





10 [5]


10 [4]

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||resn<br>resn<br>|et18-imag<br>et50-plac<br>|enet<br>es365<br>||||
||ve pairs|resn<br>resn<br>resn<br>resn<br>|t101-im<br>ext101-32<br>ext101-32<br>ext101-32|genet<br>x8d-wsl<br>x16d-wsl<br>x48d-wsl||||
||# of negati|||||||
||||ratio of|positive p|airs|||
|||||||||



0 0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1


Figure 7: Search curves _F_ for different pretrained models.
Curve F is used to estimate the minimal number of negative
pairs (y = _F_ ( _x_ ) ) that human assessors need to inspect
before they find the fraction x of positive pairs. The lower
the curve _F_ the better (need to inspect manually less pairs).
The curves are built with the query set Q = MSRVTT
full train, the gallery set G = random 596k videos from
HowTo100M.


As we see resnext101-32x48d-wsl shows the best result.
We use this network for searching for duplicates.


It is worth to mention that here we just compare different
networks on a fixed benchmark, and pick the best one. But
the search curve _F_ ( _x_ ) significantly depends on data. This


15



curve should be estimated for each used pair of datasets _Q_
and _G_ .


**C.1.4** **Black frames**


Often two consecutive video segments are glued with several black frames. The cosine similarity of embeddings
of two black or near black frames are close to 1. In this
case the most probable candidates for duplicates are black
video segments. To prevent this we apply the following
rule. Suppose we have a frame _U_ and the unit length embedding _v_ computed from _U_ . We find the prevalent color
in _U_ and compute the area _S_ 0 filled by this color. Then
we compute the value _S_ 0 _/_ ( _hw_ ), where _h_ and _w_ are the
height and width of _U_ . If this fraction is greater than 0.7
we define _µ_ _v_ = 1 _−_ _S_ 0 _/_ ( _hw_ ), otherwise _µ_ _v_ = 1 . To
calculate similarity between embeddings _v_ 1 and _v_ 2 we use
weighted cosine similarity: _µ_ 1 _µ_ 2 cos _v_ 1 _, v_ 2, instead of classical cosine similarity. This rule removes majority of all
near black frames from the most relevant candidates for
duplicates.


**C.1.5** **Screensavers detection**


Many videos from ActivityNet, HowTo100m, YouCook2
contain screensavers at the beginning or at the end. It
causes a problem like mentioned above with near black
frames, because most of relevant proposals are the same
screensavers, but the video content of the remainder video
part are different.


Using the system described in Sec. C.1.6 we search for duplicates in the ActivityNet dataset, where a lot of the most
relevant segments are screensavers. We collect several hundreds of screensavers and then compute embeddings for
each of them. Let us call the resulting set of embeddings as
E. Then we apply the following rule: if some embedding
_v_ has the similarity greater that 0.9 to one of embeddings
from E, we set _v_ = 0 . So if the video segment has a
part of a screensaver, it will never be in the most relevant
proposals.


**C.1.6** **GUI**


The important part of the video duplicate search system is
the user interface. Without ergonomic and fast interface it
is impossible to assess tens thousands of video pairs. Our
system is presented in Fig. 8.


The system shows video pairs with the highest scores on
top. A user needs to scroll down a web page (new videos
are loaded dynamically with ajax), and if a video duplicate
is detected, a user should press the _Duplicate_ button, if
there are no duplicates in the current viewport, no action is
required. When a user scrolls a web page, all non-duplicate
pairs automatically are saved to a log file. Additionally
several users at the same time can assess video pairs.


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT


Separate results for the first and the second stages are
reported in Sec. C.2.1.


Note that columns "test" and "train" in Tab. 9 may have
different values. Consider the situation when the test part
have a video segment A, and the train part have two video
segments A1 and A2. And both are marked as duplicates
with A. In this case the video segment A brings +1 to the
"test" column and A1, A2 bring +2 to the "train" column.



Figure 8: Web system used to find duplicates. Images on
the first and third row are not duplicates and the second
row contains duplicate.


M A L
dataset
test train test train test train


M 114 223 6 10 0 0

A 10 6 127 163 0 0

L 6 2744 0 0 0 0

YouCook2 13 27 7 10 0 0

MSVD 1 1 1 1 1 1

TGIF 6 8 0 0 0 0

Twitter Vines 3 3 0 0 0 0

Kinetics700 4 5 456 464 0 0

HowTo100M 177 154 209 209 0 0

Table 9: The leftmost column represents train parts of
datasets, and the upper row represents test parts of datasets.
Column "test" means how many video segments are in
the test part that have the corresponding pair in the train
part either with the same YouTube ID or manually marked
as a duplicate. Column "train" represents the number of
video segments in the train part that have corresponding
pair in the test dataset either with the same YouTube ID or
manually marked as a duplicate. All segments counted in
the "train" column are removed from the train part. For example consider the column "A" and the row "M". train=10
means that the MSRVTT train part contains 10 video segments that have a pair in the ActivityNet test part. These
10 videos must be removed from train part when dataset
are combined. test=6 means that ActivityNet test has 6
video segments that have a pair in the MSRVTT test part.


**C.2** **Cleaning results**


Recall that our cleaning method consists of two stages. In
the first stage we throw out from the train part all video
segments that have a pair with the same YouTube ID in
test parts of MSRVTT or ActivityNet. The second stage
is matching video segments by embeddings and manually
assess several thousands pairs with the highest score.


In Tab. 9 we report how many duplicates are found for
each pairs of datasets. This table represents the final result
after applying these two stages.



The most problematic datasets in terms of the number of
duplicates are MSRVTT and ActivityNet. These datasets
overlap with itself (e.g. MSRVTT test overlap with
MSRVTT train). We found more than 100 duplicate
pairs for both of them. Other problematic datasets are
HowTo100M and Kinetics700, these datasets are large,
so we can’t assess the required number of video pairs
to find 95% or 99% of duplicates. But we can assess a
smaller number of pairs and using search curves _F_ (see
Sec. C.1.2) can extrapolate this value to 100%. We found
that HowTo100M may have the intersection with MSRVTT
test full by about 300 videos (10% of the MSRVTT test
full). The similar situation is about the ActivityNet test set
and Kinetics700, the intersection could be near 500-600
videos (10% of the ActivityNet test set).


In Tab. 10 we report results on MSRVTT for MMT retraining with no cleaning, after cleaning by the YouTube ID and
cleaning combination by the YouTube ID and the manual
assessment. The manual cleaning for 1k-A and 1k-B is
incomplete because we only do cleaning for the full split.
The following situation takes place for 1k-A, 1k-B splits:
when 1k videos from the full test are taken for test and
the remaining 2k videos are moved to the train part, the
additional overlapping is introduced, because these 1k and
2k videos are overlapping. We do not remove this overlap
in this research.


no by ID +
split by ID
clean manual


full 31.1 _±_ 0 _._ 1 31.1 _±_ 0 _._ 1 30.2 _±_ 0 _._ 4
1k-A 54.8 _±_ 0 _._ 5 50.7 _±_ 0 _._ 9 49.4 _±_ 0 _._ 5
1k-B 51.1 _±_ 0 _._ 9 46.1 _±_ 0 _._ 1 46.4 _±_ 0 _._ 6

Table 10: Comparison for original MMT trained (7 modalities) on MSRVTT without cleaning, with cleaning by the
YouTube ID only, and with cleaning by the YouTube ID
plus the manual assessment.


As you can see after cleaning the performance is significantly decreased on 1k-A and 1k-B splits for original
MMT.


**C.2.1** **Intersection by YouTube ID and embeddings**


In Tab. 11 we report the intersection by the YouTube ID
between test parts of MSRVTT (full, 1k-A, 1k-B) and
ActivityNet with train parts of MSRVTT (full, 1k-A, 1kB), ActivityNet, Kinetics700, YouCook2, HowTo100m,
MSVD.



16


MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT



data M M 1k-a M 1k-b A

set test train test train test train test train


M 0 0 0 0 104 179 2 4
M 1k-a 2362 1990 372 415 827 1007 2 4
M 1k-b 1689 1367 563 634 380 407 2 4
A 0 0 0 0 0 0 0 0

K 5 4 1 1 0 0 408 408

Y 8 4 2 2 2 2 3 3

HT100M 147 117 39 38 57 53 175 175

MSVD 3 1 2 1 0 0 1 1

Table 11: First stage. The leftmost column represents train
parts of datasets, and the upper row represents test parts
of datasets. Column "test" represents number of video
segments in test part that have corresponding video in train
part with the same YouTube ID. Column "train" represents
number of video in train part that have corresponding pair
in test part with the same ID. For example: if we combine M and YouCook2, we should remove 4 video from
YouCook2 train.


It is worth to mention that MSRVTT 1k-A test and 1k-B
test have a large overlap ratio by the YouTube ID with the
1k-A train and the 1k-B train parts correspondingly. Both
splits have the overlap ratio of about 38% between the train
part and the test part. We also emphasize that the original
MSRVTT full split does not overlap by the YouTube ID
between the test and train parts.


data M A L

set seen found total seen found total seen found total


M 10k 114 114 1k 6 6 1k 0 0

A 10k 10 10 15k 127 142 1k 0 0

L 3k 6 6 2k 0 0 — — —

Y 2k 13 13 1k 7 7 1k 0 0

MSVD 1k 1 1 1k 1 1 1k 1 1

T 2k 6 6 2k 0 0 3k 0 0

V 2k 3 3 0k 0 0 1k 0 0

K 2k 1 2 30k 227 539 2k 0 0

HT100M 5k 15 320 — — — — — —

Table 12: Second stage. The leftmost column represents
train parts of datasets, and the upper row represents test
parts of datasets. Column "seen" represents the number of
video segments that we manually assess for a given pair of
datasets. Column "found" represents the number of videos
in the test part for which there exists the corresponding
duplicate video segment in the train part. Column "total"
represents the approximately estimated total number of
videos from the test part that have a duplicate pair in the
train part. Symbol "—" means that the intersection is not
computed because it requires too much human resources.


In Tab. 12 we report the statistics for the second deduplication stage (searching by embeddings). We do not compute
an intersection for MSRVTT 1k-A and 1k-B splits.



In this table we present the number of manually found
duplicates and the estimated maximum number of duplicates for a given pair of datasets. We managed to find the
intersection for almost all pairs of datasets.


The maximum number of duplicates is computed based
on the search curve _F_ ( _x_ ) . As we told in Sec. C.1.2 the
search curve significantly depends on data. We compute
the search curve for all pairs of datasets in Tab. 12. The
search curve for each particular pair of datasets is build
exactly in the same way as described in Sec. C.1.2. For
example, to compute the search curve for MSRVTT test
and ActivityNet train we define MSRVTT test as _Q_, ActivityNet train as _G_, then augment _Q_ to produce _Q_ [ˆ], and use
the algorithm described in Sec. C.1.2.


Using the column "seen" from Tab. 12 we can compute
how many pairs need to be assessed to find the full overlap between datasets. For example, inspect 5k pairs for
HowTo100M dataset and MSRVTT (the row "HT100M"
and the column "M"), we found 15 duplicates, so the approximate maximum number of duplicates is 320: 5k *
(320 / 15) = 106k. So, to find the full overlap using the
current version of algorithm it is needed to manually assess
106k video pairs and it is too much, that’s why we do not
find full intersection for this specific pair of datasets.


**D** **Hyperparameters**


To train our best networks (MMT(MALVYMTS)
L9H8 CLIP+audio, MDMMT(MALVYMTS) L9H8
irCSN152+audio and MMT(MALVYMTS) L9H8
CLIP+irCSN152+audio) we use 50 epochs and define a
single epoch as 150K examples per GPU (in total 1.2M
examples per epoch on 8 GPUs). We use Adam optimizer
without weight decay, the initial value for a learning rate is
5e-5, after each epoch we multiply the learning rate by
0.95. Batch size of 32 examples per GPU is used. We
do not exchange embeddings between GPUs. We use
bi-directional max-margin ranking loss with margin 0.05.
In Bert and the video transformer encoder we use dropout
0.2 in attention and in FFN block. We use 8 Nvidia V100
32GB GPUs. The training time is about 14 hours.


**E** **Pretrained model**


The well known method to boost the performance in video
retrieval tasks is to use a pretrained model. First the neural
network is trained on some large dataset, then at second
stage it is finetuned for target target dataset. In video
retrieval task HowTo100M dataset is often used for pretraining. In this work we use HowTo100M for pretraining
in the same way.


In our training procedure we use 8 Nvidia V100 32Gb
GPUs, we train for 200 epochs where one epoch is defined
as 80k examples on each GPU (in total network sees 640k
examples on 8 GPUs per epoch). We use batch size 64
for each GPU and do not exchange embeddings between



17


model



MDMMT: Multidomain Multimodal Transformer for Video Retrieval A P REPRINT


MSRVTT full clean text _→_ video
R@1 _↑_ R@5 _↑_ R@10 _↑_ MnR _↓_ MdR _↓_



Ours MDMMT(M _c_ ALVYMTS) L9H8 irCSN152+audio yes 15.8 _±_ 0 _._ 1 38.9 _±_ 0 _._ 1 51.0 _±_ 0 _._ 1 76.4 _±_ 0 _._ 5 10.0 _±_ 0 _._ 0
Ours MDMMT(M _c_ ALVYMTS) L9H8 irCSN152+audio no 14.5 _±_ 0 _._ 1 36.8 _±_ 0 _._ 3 48.8 _±_ 0 _._ 3 82.2 _±_ 0 _._ 6 11.0 _±_ 0 _._ 0
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+audio yes 21.5 _±_ 0 _._ 1 47.4 _±_ 0 _._ 2 59.6 _±_ 0 _._ 1 57.7 _±_ 0 _._ 4 6.0 _±_ 0 _._ 0
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+audio no 20.0 _±_ 0 _._ 1 45.1 _±_ 0 _._ 1 57.3 _±_ 0 _._ 1 63.1 _±_ 0 _._ 1 7.0 _±_ 0 _._ 0


Table 13: Performance on the MSRVTT full clean split with and without pretrained model (HowTo100m).



model



ActivityNet text _→_ video
R@1 _↑_ R@5 _↑_ R@10 _↑_ MnR _↓_ MdR _↓_



Ours MDMMT(M _c_ ALVYMTS) L9H8 irCSN152+audio yes 15.1 _±_ 0 _._ 1 38.3 _±_ 0 _._ 1 51.5 _±_ 0 _._ 3 92.4 _±_ 2 _._ 3 10.0 _±_ 0 _._ 0
Ours MDMMT(M _c_ ALVYMTS) L9H8 irCSN152+audio no 12.0 _±_ 0 _._ 1 33.7 _±_ 0 _._ 4 46.3 _±_ 0 _._ 3 119.9 _±_ 2 _._ 1 13.0 _±_ 0 _._ 0
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+audio yes 17.7 _±_ 0 _._ 1 41.6 _±_ 0 _._ 3 54.3 _±_ 0 _._ 2 76.0 _±_ 1 _._ 0 8.3 _±_ 0 _._ 5
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+audio no 15.2 _±_ 0 _._ 3 37.9 _±_ 0 _._ 3 50.1 _±_ 0 _._ 2 93.4 _±_ 2 _._ 0 10.3 _±_ 0 _._ 5


Table 14: Performance on ActivityNet with and without pretrained model (HowTo100m). The performance reported for
the text to video retrieval task on our own subset of the original ActivityNet test part. See Sec. 2.1 for details.



model



LSMDC text _→_ video
R@1 _↑_ R@5 _↑_ R@10 _↑_ MnR _↓_ MdR _↓_



Ours MDMMT(M _c_ ALVYMTS) L9H8 irCSN152+audio yes 13.1 _±_ 0 _._ 5 31.3 _±_ 0 _._ 3 40.1 _±_ 0 _._ 0 74.5 _±_ 0 _._ 7 19.3 _±_ 0 _._ 5
Ours MDMMT(M _c_ ALVYMTS) L9H8 irCSN152+audio no 12.6 _±_ 0 _._ 7 30.2 _±_ 1 _._ 5 39.6 _±_ 0 _._ 9 76.1 _±_ 0 _._ 8 19.7 _±_ 1 _._ 3
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+audio yes 17.2 _±_ 0 _._ 6 34.9 _±_ 0 _._ 4 45.3 _±_ 1 _._ 0 65.6 _±_ 0 _._ 8 14.0 _±_ 0 _._ 8
Ours MDMMT(M _c_ ALVYMTS) L9H8 CLIP+audio no 16.2 _±_ 1 _._ 1 35.4 _±_ 1 _._ 3 45.1 _±_ 0 _._ 7 64.9 _±_ 1 _._ 9 14.7 _±_ 0 _._ 5


Table 15: Performance on LSMDC with and without pretrained model (HowTo100m).


GPU. Initial learning rate is 5e-5. After each epoch we
multiply learning rate by 0.98. We use the full HowTo00M
dataset. The model is trained either with two modalities:
motion/RGB and audio or with three modalities: motion,
RGB and audio, depending on how many modalities are
used in final model. The total training time is about 24
hours. We use bi-directional max-margin ranking loss with
margin 0.05.


In Tab. 13, 14 and 15 we compare two our models:
MDMMT(M _c_ ALVYMTS) L9H8 irCSN152+audio and
MDMMT(M _c_ ALVYMTS) L9H8 CLIP+audio when they
are trained from the pretrained model or not. In these three
tables we present the same four models (no special finetuning for the target dataset) tested on different datasets.


As we can see in Tab. 13 the pretrained model increases R1
metric by 1% and R5 by 2%. The pretrained model also
increase performance on ActivityNet dataset, see Tab. 14.
For R1 metric the improvement is about 2% and for R5
metric is about 4%. For LSMDC dataset, see Tab 15,
we have approximately the same results with and without
pretraining.


18


