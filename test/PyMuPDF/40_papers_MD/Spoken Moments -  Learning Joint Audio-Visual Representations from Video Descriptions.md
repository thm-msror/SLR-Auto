## **Spoken Moments: Learning Joint Audio-Visual Representations from Video** **Descriptions**



Mathew Monfort [*]

MIT


mmonfort@mit.edu



SouYoung Jin [*]
MIT


souyoung@mit.edu



Alexander Liu

MIT


alexhliu@mit.edu



David Harwath

UT Austin


harwath@cs.utexas.edu


Aude Oliva

MIT


oliva@mit.edu



Rogerio Feris
IBM Research


rsferis@us.ibm.com



James Glass

MIT


glass@csail.mit.edu



**Abstract**


_When people observe events, they are able to abstract key_
_information and build concise summaries of what is hap-_
_pening. These summaries include contextual and seman-_
_tic information describing the important high-level details_
_(what, where, who and how) of the observed event and ex-_
_clude background information that is deemed unimportant_
_to the observer. With this in mind, the descriptions people_
_generate for videos of different dynamic events can greatly_
_improve our understanding of the key information of inter-_
_est in each video. These descriptions can be captured in_
_captions that provide expanded attributes for video labeling_
_(e.g. actions/objects/scenes/sentiment/etc.) while allowing_
_us to gain new insight into what people find important or_
_necessary to summarize specific events. Existing caption_
_datasets for video understanding are either small in scale or_
_restricted to a specific domain. To address this, we present_
_the Spoken Moments (S-MiT) dataset of 500k spoken cap-_
_tions each attributed to a unique short video depicting a_
_broad range of different events. We collect our descriptions_
_using audio recordings to ensure that they remain as natu-_
_ral and concise as possible while allowing us to scale the_
_size of a large classification dataset. In order to utilize our_
_proposed dataset, we present a novel Adaptive Mean Mar-_
_gin (AMM) approach to contrastive learning and evaluate_
_our models on video/caption retrieval on multiple datasets._
_We show that our AMM approach consistently improves our_
_results and that models trained on our Spoken Moments_
_dataset generalize better than those trained on other video-_
_caption datasets._


_[http://moments.csail.mit.edu/spoken.html](http://moments.csail.mit.edu/spoken.html)_


  - equal contribution



**1. Introduction**


Video understanding has typically been focused on action recognition and object tracking as the temporal aspect of videos lends itself strongly to the task of representing motion, a key component of an action. Breaking
down video analysis to simple tasks, such as action recognition, allows for efficient data annotation for building large
datasets to train deep learning models [31, 46, 21] which
has been extremely successful for images with object annotations [35]. A main difficulty is that, in contrast to an
image, a video often captures an interaction between agents
and objects that evolves over time. These interactions can
be as simple as “a person picking up a glass of water”, but
even in this case three different objects (“person”, “glass”
and “water”) are included in the interaction. Additionally,
the video may also continue to depict the “person drinking from a glass” and the “person putting the glass back
down on the table”. These sequential events present additional challenges for video datasets where single annotations may not be sufficient to explain the events depicted.
Multi-label approaches to video annotation have attempted
to address this problem by labeling multiple actions in a
video [47, 22, 73]. However, these methods focus on single domain annotations, such as actions or objects, and do
not capture additional contextual information, such as “person _angrily_ putting down the _dirty_ glass on a _rusted_ table”,
which can change the interpretation of an event and how it
fits into a sequence of observations.


A solution for capturing more fully the content of video
is to annotate multiple actions or objects in each video

[22, 72, 47, 50]. However labels like “drinking”, “glass”,
only provide a portion of the information needed to interpret the veracity of the event. Additional narratives may
include intuitive descriptions and intentions, such as “an exhausted man picks up a dirty glass of water and drinks from



1


it before angrily putting it down on a table” which would
dramatically change the event interpretation. The full lingual description combines these actions with adjectives and
nouns (objects) that contextualize the events depicted leading to a better understanding of the video. This is our goal
in providing a new large scale dataset for training models
for full video understanding.
We introduce a large scale video caption dataset, Spoken Moments in Time (S-MiT), to allow large deep learning models for video understanding to learn contextual information. Most existing video description datasets [71,
60, 33, 20, 80] are limited in size when compared to the
large datasets for action recognition [31, 46, 21]. A likely
cause is the increased cost of collecting full text descriptions for videos compared to single label annotations. Recent work in image captioning [25] addressed this problem
by collecting audio descriptions for a large set of images
from the Places dataset [77]. Collecting spoken captions is
faster and more efficient due to the low overhead of speaking compared to typing. In addition, recording of spontaneous speech rather than typed text can produce more natural descriptions of an event.An automatic speech recognition (ASR) system was then used to transcribe the spoken
descriptions to text captions. In this work, both audio, text
and video models were jointly trained via contrastive learning to learn joint cross-modal representations. We build
on this approach and compare models that learn directly
from the spoken captions to models that include a trained
ASR model which feeds generated text transcriptions into
an NLP language model. We then jointly train caption and
visual models (based on concatenated video and image features) using a novel Adaptive Mean Margin (AMM) approach to contrastive learning to align the visual and caption representations. We evaluate our models on multiple
datasets for video/caption retrieval and show that a model
trained using AMM on S-MiT achieves the best general performance across four datasets.

Altogether, our novel contributions include:


1. The large-scale **Spoken Moments in Time dataset** (SMiT) which includes 500k pairs of video clips and corresponding audio descriptions. This new dataset represents the largest video description dataset available
and will serve as a new benchmark for the community.


2. **Benchmark models** with aligned spoken caption and
video representations learned via contrastive learning.
We compare approaches that learn directly from the
spoken descriptions as well as approaches that include
ASR transcriptions that feed into different language
models to generate caption representations.


3. An **Adaptive Mean Margin** (AMM) approach to
cross-entropy based contrastive learning.



**2. Related work**


**2.1. Video Understanding**


The field of video understanding has recently seen fast
progress partly due to the availability of large scale video
datasets including ActivityNet [6], Kinetics [31], Moments
in Time [46, 47] and YouTube-8M [1]. These large datasets
are used to pretrain models that are fine-tuned on smaller action recognition datasets such as UCF101 [62] and HMDB

[36]. With the increased availability of large scale video
datasets, many different models have been proposed to improve performance on a number of video understanding
tasks. Two-stream convolutional neural networks (CNNs)
combine optical flow with RGB frames to capture both temporal and spatial information [61]. I3D models [8] combine
3D CNNs [65], which use a 3D kernel to learn temporal
information from a frame sequence, with optical flow to
form a two-stream 3D network “inflated” from 2D filters
pre-trained on ImageNet [16]. More recently a temporal
shift module has been proposed to integrate temporal information into 2D models by shifting frame representations
across the temporal dimension [40].
Recently multi-modal visual understanding methods
have received significant attention [25, 64, 44, 67, 29, 4].
The DAVEnet model [25] has been proposed for jointly
learning aligned representations between images and spoken captions, and has been extended to align frame-wise
video representations with synchronized audio narration
for cross-modal audio-visual concept learning [4]. Here,
we build on the motivation from this paper and **learn**
**aligned representations between videos and unsynchro-**
**nized spoken descriptions** using the S-MiT Dataset.


**2.2. Caption Datasets**


There have been a number of different datasets released

for providing language descriptions of visual information.
Flickr8k [28] and Flickr30k [49] include 8k and 30k images respectively each sourced from Flickr. Each image is
associated with 5 text captions describing what is in the image. An additional set of 5 audio captions per image in
both sets was recently collected for learning joint embeddings between speech and images [25]. The Visual Genome
dataset [34] includes captions for multiple regions of more
than 180k images allowing for fine-grained descriptions of
each image. The Places Audio Caption dataset [26] contains
approximately 400k images from the Places 205 [78] image
dataset with audio captions of people verbally describing
each image. MS COCO [11] is a large image dataset for
object recognition, segmentation, and captioning which includes roughly 1 million captions for 160k Flickr images.
Conceptual Captions [59] contains 3.3M images with captions generated from HTML attributes associated with web
based images. The Stock3M dataset [70] includes 3.2 mil


2


Figure 1: **Examples from the Spoken Moments Dataset:** The dataset is composed of videos and the corresponding spoken captions. We
show some examples of the text transcriptions, automatically generated using the public Google ASR engine.



lion images each with a crowdsourced caption.


Beyond the numerous datasets available or image captioning [28, 49, 34, 11, 59, 70], including those that provide spoken descriptions [26, 25], there are a variety of
different video caption datasets available. A number of
these datasets are related to cooking [51, 52, 55, 13, 12]
including YouCook [14] and YouCook II [80] which include 2k videos from YouTube each with multiple captions annotated at different segments of each video. MPIIMovie Description Corpus [53] contains transcribed audio
descriptions from 94 Hollywood movies split into 68k clips
where each clip is paired with a sentence from the movie
scripts and an audio description of the visual content in each
clip. Similarly, Large Scale Movie Description Challenge
(LSMDC) dataset [54] contains 200 movies with 120K sentence descriptions. VideoStory [20] contains 20k social media videos where each video contains a paragraph length
description. The ActivityNet Captions dataset [33] has 20k
videos with 100k text descriptions. The Microsoft Video
Description (MSVD) dataset [9] contains 2k YouTube clips
with a 10-25 second duration and an average of 41 single
sentence descriptions per clip. MSR-Video to Text (MSRVTT) [71] contains 7k videos split into 10k clips with 20
captions per video.


HowTo100M [45] contains 136 million clips sourced
from 1.22 million instructional videos with narrations generated from subtitles associated with each video. However,
the subtitles are not human verified captions and the content
is constrained to instructional videos. Since the text associ
ated with the clips in the HowTo100M dataset are transcriptions of a narrator completing a task in the video, the short
text phrases from the subtitles occasionally share noisy associations with the reference clip. In Section 5, and Table
2, we decided to compare our contributions using strict caption datasets as we are proposing a large-scale human annotated caption dataset with full human generated descriptions
for each video.


VaTeX [69] contains 41k videos sourced from the
Kinetics-600 dataset [31, 7] annotated with 10 English captions and 10 Chinese captions for multilingual captioning.
VaTeX is the most similar to our proposed dataset in that it
is sourced from an existing video dataset for action recognition and the captions are directly annotated.



In this work, we present a new dataset, _Spoken Moments_
_in Time (S-MiT)_, which includes spoken audio captions for
500k unique three second clips each with different source
videos from the Moments in Time dataset [46, 47]. In
addition to vast increase in scale over other video-caption
datasets, a major contribution is that we are using spoken
descriptions rather than text. This allows us to train spoken
caption models to directly align with video models. This
is not possible with the other large video caption datasets
and allows for spoken caption models to be analyzed with
matching video information. We also show that models
trained on our S-MiT dataset generalize much better in retrieval to the video-caption pairs in other datasets. This is
due to the large coverage, diversity and scale of our proposed dataset.


**2.3. Cross Modal Contrastive Learning**


Cross modal learning has been used to jointly selfsupervise audio-visual models [3, 48, 76] with synchronized information while NLP approaches have been leveraged to align joint representations for both visual and language modalities using spoken and text descriptions [2, 79].
This is typically done via Contrastive Learning where the
alignment between positive pairs (language and visual input) is trained to be stronger than those of non-positive pairs

[24]. For visual representations, a triplet based max-margin
loss is commonly used to discriminate representations between positive and negative pairs [74, 75, 18]. Semihard negative mining [58] and a dot-product based similarity score have been used to jointly learn audio-visual embeddings between images and spoken captions [25] while
batch-wise cross-entropy approaches to contrastive learning have been used to increase the amount of information
utilized in learning by considering all negative examples in
a mini-batch [66, 10]. Work on bidirectional speech/image
retrieval using audio descriptions of images integrated ideas
from max-margin contrastive learning and added a margin
into the cross-entropy loss [29]. SimCLR [10] added a nonlinear projection head that maps paired representations into
a common space allowing for stronger representations.

A pretrained language model has recently been used
to improve cross-modality learning with language and visual input pairs. VilBERT [43] added a pretrained BERT



3


[17] transformer to capture semantic language representations associated with object detection proposals from a pretrained faster RCNN network. VideoBERT [63] extended
BERT to jointly learn the visual and linguistic domain by
generating tokenized visual words. Inspired by this prior
work, we propose adding a **pretrained language model**
**that maps word predictions from a trained ASR model**
**to semantic language features** in order to generate rich
spoken caption representations. We then utilize an MLP _[̸]_
to project these caption representations, and our video representations, to an aligned joint representation which can be
used for video/caption retrieval (see Section 5).


**2.3.1** **Optimization Approaches**


A common approach to optimization in contrastive learning settings is to use a similarity based loss function. We
formulate the contrastive loss as, _L_ = _L_ _vc_ + _L_ _cv_, where
the goal is to maximize the discrimination between positive
and negative paired captions _c_ and videos _v_ . The loss is split
into two tasks where _L_ _vc_ forms pairs from a fixed video and
each caption in a sampled mini-batch, while _L_ _cv_ fixes the
caption and forms pairs with each video in the mini-batch.
Below we discuss different approaches of _L_ _xy_, where _x_ and
_y_ are interchangeable with _v_ and _c_ .
Semi-hard negative mining (SHN) [58] has been used
for learning aligned cross-modal embeddings using a triplet
loss [25, 30]. This is an improvement over hard negative mining [19] since a sampled negative example is constrained to be less similar to the anchor than the positive
sample while still being within the margin and thus contributing a loss at each step with the margin _M_ = 1,
_L_ _xy_ = max( _S_ ( _x_ _i_ _, y_ _j_ ) _−S_ ( _x_ _i_ _, y_ _i_ )+ _M,_ 0), where _S_ ( _x_ _i_ _, y_ _j_ )
is a similarity score for the representations of _x_ _i_ and _y_ _j_,
with _x_ _i_ and _y_ _i_ forming a positive pair.
Noise contrastive estimation (NCE) [23] has been applied to contrastive learning [10, 66] by using a loglikelihood based loss function that learns to discriminate

between positive and negative pairs of feature embeddings,


_[̸]_


_̸_



_N_ refers to the number of classes being discriminated. For
aligning captions to visual information, the class size can be
considered unbounded as each caption represents a slightly
different representation that we want to discriminate leading to a max margin size of 1. Concretely, MMS proposes
adding a margin to Equation 1,


_[̸]_


_[̸]_


_̸_



_e_ _[S]_ [(] _[x]_ _[i]_ _[,y]_ _[i]_ [)] _[−][M]_ + ~~[�]~~ _[B]_ _j_ =1 _[I]_ _[i][̸]_ [=] _[j]_ _[e]_ _[S]_ [(] _[x]_ _[i]_ _[,y]_ _[j]_ [)] _[,]_ (2)


_[̸]_


_̸_



_L_ _xy_ = _−_ _B_ [1] _[̸]_


_[̸]_


_̸_



_B_
�

_[̸]_


_[̸]_


_̸_



_e_ _[S]_ [(] _[x]_ _[i]_ _[,y]_ _[i]_ [)] _[−][M]_

� _log_

_i_ =1 _e_ _[S]_ [(] _[x]_ _[i]_ _[,y]_ _[i]_ [)] _[−][M]_ + ~~[�]~~ _[B]_ _j_ =1 _[I]_ _[i][̸]_


_[̸]_


_̸_




_[̸]_


where the margin, _M_, starts as 0.001 and is exponentially
increased by a factor of 1.002 every 1000 training steps.
We propose extending the idea of an increasing margin
in MMS to an adaptive setting that does not require setting
the initial value of the margin or the growth rate. We refer to
this approach as an Adaptive Mean Margin (AMM) where
the margin is set as the mean distance between the positive
pair and the set of negative pairs in a batch. We describe
AMM in more detail in Section 4.3.


**3. The Spoken Moments Dataset**


We begin with the Moments in Time dataset [46] as it includes over 1 million videos sourced from a number of dif
ferent video hosting sites with strong inter & intra-varietal
variation in terms of the number of events depicted in each
video. Further, the videos are all cut to 3 seconds allowing for a concise description to effectively capture the localized information of each event. Here we refer to concise

descriptions as those that focus on key events depicted in
the video and does not imply partial descriptions. In data
collection, annotators may watch a video as many times as
desired. During recording, we block the annotators from
seeing/hearing the video to encourage descriptions of important memorable events rather than every specific detail.
This approach does not preclude the annotators from describing sequential or simultaneous events as shown in our
qualitative examples (see Figure 1). We describe our annotation approach in more detail in the supplementary material.


**3.1. Dataset Statistics**


_[̸]_

Our proposed Spoken Moments dataset contains 500k

_̸_ videos randomly chosen from the Multi-Moments in Time

(M-MiT) training set and all of the 10k videos from the validation set. Each video in the training set contains at least
one audio description. We transcribed each audio recording using the public Google Automatic Speech Recognition (ASR) engine to generate text captions for each video.
When analyzing these transcriptions, we build a picture of
the coverage and diversity of our captions. Table 2 (left)
shows that our captions have an average length of 18 words
with a unique vocabulary of 50,570 words consisting of
20,645 nouns, 12,523 adjectives and 7,436 verbs with a




_[̸]_


_L_ _xy_ = _−_ _B_ [1] _[̸]_


_̸_




_[̸]_


_B_

_e_ _[S]_ [(] _[x]_ _[i]_ _[,y]_ _[i]_ [)]

� _i_ =1 _log_ ~~�~~ _Bj_ =1 _[I]_ _[i][̸]_ [=] _[j]_ _[e]_ _[S]_ [(] _[x]_ _[i]_ _[,y]_ _[j]_ [)] _[,]_ (1)


_̸_




_[̸]_


_[̸]_


where _I_ _i_ = _̸_ _j_ is an indicator function that we only considers
negative pairs in the denominator. This has been shown to
improve feature alignment compared to SHN [10].
Masked Margin Softmax Loss (MMS) [29] and Large
Margin Cosine Loss (LMCL) [68] incorporate a positive
margin into the contrastive learning framework in order to
improve feature discrimination among non-paired embeddings. MMS uses a monotonically increasing margin to allow for initial learning to begin to converge before a large
alteration to the loss is added. LMCL proposes a theoretical limit on the maximum margin size of 1 _−_ _cos_ [2] _N_ _[π]_ [where]




_[̸]_


_[̸]_


_̸_


4


Type Total Average Unique
Words 5,618,064 18.01 50,570
Verbs 492,941 1.58 7,436
Nouns 1,365,305 4.37 20,645
Adjectives 386,039 1.24 12,523




|Type|Dataset Coverage|
|---|---|
|Objects|ImageNet<br>69.2%<br>MS-COCO<br>100%|
|Actions|Kinetics<br>85.1%<br>Moments in Time<br>96.2%|
|Scenes|Places365<br>47.4%|


|Dataset|Clips Videos Captions Words Vocab Domain Spoken|
|---|---|
|TACoS [51]<br>YouCook II [80]<br>MSVD [9]<br>Charades [60]<br>MPII-MD [53]<br>MSR-VTT [71]<br>ActivityNet Captions [33]<br>VideoStory [20]<br>Epic-Kitchens [13, 12]<br>Vatex-en [69]<br>**Spoken Moments**|7,206<br>127<br>18,227<br>146,771<br>28,292 Cooking<br>15,400<br>2,000<br>15,400<br>121,418<br>2,583<br>Cooking<br>1,970<br>1,970<br>70,028<br>607,339<br>13,010 General<br>10,000<br>10,000<br>27,800<br>645,636<br>32,804 General<br>68,337<br>94<br>68,375<br>653,467<br>24,549 General<br>10,000<br>7,180<br>200,000<br>1,856,523 29,316 General<br> 100,000<br>20,000<br>100,000<br>1,348,000 15,564 General<br>123,000<br>20,000<br>123,000<br>1,633,226<br>-<br>General<br>76,885<br>633<br>76,885<br>227,974<br>1,737<br>Cooking<br>41,300<br>41,300<br>413,000<br>4,994,768 44,103 General<br>**515,912 459,742**<br>**515,912**<br>**5,618,064 50,570 General**<br>✓|



Figure 2: **Dataset Statistics:** On the top-left we show the total and average number of words, verbs, nouns and adjectives in our captions as
well as the number of unique examples of each. On the bottom-left we show the percentage of the class vocabulary from different datasets
that occur in our captions. On the right we compare our proposed Spoken Moments dataset to existing video caption datasets. The word
count and vocabulary for S-MiT are generated using ASR transcriptions.



total word count of 5.6 million. Table 2 (right) shows a
comparison of our Spoken Moments dataset to other existing datasets for video captioning. Our dataset will be the
largest public dataset in terms of video clips, source videos,
total number of captions, total words in the captions and
the vocabulary set of unique words occurring in the captions. The increase in vocabulary size is important as it
shows that our increase in the number of videos over previous datasets does not simply include repeated events but
covers a novel breadth of information. We can see the opposite effect of this in YouCook II [80] where the restricted
domain of cooking videos results in a limited vocabulary
used in the descriptions.


To understand how this vocabulary covers the class labels typically used for training computer vision models, we
examined whether these labels exist in our vocabulary. Table 2 (right) shows that we have strong coverage of the two
largest action recognition datasets for video understanding
(Kinetics [31] and M-MiT [47]). We expected a large coverage of the events in M-MiT as we sourced our videos from
this dataset and the action labels themselves are fairly general (e.g. “running” and “cooking”). For Kinetics, the labels are commonly tied to a noun preceded by a verb (e.g.
“brushing hair”). For these labels we consider them to exist in our dataset if both the verb and noun are in the same

caption. For example, “A boy is in a bathroom brushing his
teeth” would cover the class “brushing teeth”. With this approach we see a 85.1% coverage of the classes in Kinetics
and a 96.2% coverage of the classes in M-MiT showing a
strong level of event diversity. Similarly we see a strong
overlap of the object classes in MS-COCO [41] (100%) and
ImageNet [16] (69.2%) in our captions. ImageNet coverage
is likely lower due to the specific labels used for many of
its classes (e.g. “coucal”). Still, 69.2% coverage means
692 ImageNet classes appear in our captions. Similarly,
Places [78] scene labels are very specific and don’t necessarily match the language used in our descriptions. For example, an “abbey” will typically be described as a “church”
or “monastery” in our captions. We did not account for all
of the synonyms possible and are only considering direct



Figure 3: **Architecture:** Videos and captions are fed into the
video/caption models where the outputs are used to compute a
similarity matrix, _S_, which is used to compute a loss, _L_ .


matches in our captions. Even so we are able to find a 47.4%
coverage of the scene labels in Places365 in our dataset.
Here we provide information on some additional characteristics of our data that may be of interest. While we do
not release demographic info of our annotators or captions,
about 57% of the spoken captions were recorded by male
voices and 43% female. For the audio streams of the videos,
roughly 51% include natural sound, 5% have music as the
audio and 44% have no audio. This is consistent with the

M-MiT dataset [47] from which we source our videos. Additionally, we found that less than 3% of the videos contain
captions that describe non-visible events (e.g. a car horn
when no car is visible in the video frames). For this reason
we have chosen to focus our approach on learning a strong
visual model in Section 4.


**4. Learning Audio-Visual Representations**


In order to learn from the large set of spoken captions in
the proposed S-MiT dataset, we adopt a cross-modal architecture used in prior work [45, 25, 56] which is composed
of a video model and a caption model as depicted in Figure 3. Specifically, we take _N_ video-caption pairs as input
and encode each modality into a 4096-D feature vector. We
do this by adding a multilayer perceptron (MLP) as a projection head on top of both the video and the caption model.
This projection head is composed of two linear layers followed by gated linear units (GLU) [15]. We then compute
the dot product between the video and caption representa


5


tions to produce an _N_ x _N_ similarity matrix, _S_, which is
used to compute our contrastive loss for training. In Section 4.3, we describe our modified approach to margined
contrastive learning which uses an Adaptive Mean Margin
(AMM) which automatically adjusts itself during learning
to improve the optimization signal during training.


**4.1. Video Model**


Following prior work [45], we use two encoders to represent input videos: image & video encoders. Specifically,
we use a ResNet-152 [27] pretrained on ImageNet [35] and
a temporal shift module (TSM) ResNet-50 model [40] pretrained on M-MiT [47]. Each encoder outputs a 2048-D feature vector after max-pooling over the temporal dimension

_∼_
(8 frames for the TSM ( 3 fps) and 3 frames for the image
model (1 fps)). We concatenate the two 2048-D vectors and
feed the concatenated vector into an MLP projection head
to get the final 4096-D visual representation. We examine
the effect of using the image and video encoders as well as
different pretrained models in the supplementary material.


**4.2. Caption Model**


**4.2.1** **Language Caption Model**


Prior work in learning joint representations between audio
captions and visual models has shown that utilizing ASR
transcriptions greatly improves results [25]. We build on
this idea and use the predicted words from a pretrained
ASR model (e.g. Google’s public ASR engine) to train our
models. Concretely, we examine the effect of using different pretrained language models stacked on top of the ASR
model predictions. We begin by comparing the results of
using Fasttext [5], BART [37] and BERT [17] models to
generate semantic and contextual word representations for
our captions. During training, we randomly select 10 words
from each caption to be included in training. In the case of
the BART and BERT models, this selection happens after
the full transformer model has been applied to avoid altering
the results from the self-attention mechanisms. If less than

10 words occur in a caption then we allow words to be sampled multiple times in the random selection. This training
augmentation allows different words in each caption to be
represented differently at different training iterations. We
examine the effect of this approach in the supplementary
material. In test, we use the full transcription as input into
the language model. We average the word representations
from the output of the language model to generate a single
representation for each caption which we align to the video
representations described in the previous section.


**4.2.2** **Spoken Caption Model**


We also train caption models with raw spoken captions in- _̸_
stead of the corresponding transcription. For each caption,



we randomly sample 10 seconds of speech for training and
compute the 40-dimensional log Mel spectrogram to serve
as the input of spoken caption model. The input is fed into a
spoken caption model where we consider ResDavenet [25]
(which is designed specifically for speech) and two ImageNet ResNet [27] models (ResNet-34, ResNet-50). For
the ResNet models, we modify the first convolutional layer
to take the 1-channel input so that spectrogram can be processed. In addition, the wav2vec [57] model, which takes
raw waveform as the input, is also involved in our experiments. Spoken captions are first fed into the pre-trained
wav2vec model, which produces 512-D vectors per 210 ms.
We then feed them into a learnable ResStack, taken from
ResDavenet, to learn representations of spoken captions.


**4.3. Adaptive Mean Margin**


We train our model using the contrastive loss with a similar setting to MMS (Equation 2). The only difference is that
we replace the margin, _M_, with an adaptive margin based
on the difference between the similarity of the positive pair
and the set of negative pairs in each batch.
The challenge in using the MMS margin for mini-batch
sampled contrastive learning is that the initial margin and
growth schedule are difficult to tune for a specific dataset
and similarity metric. Additionally, depending on the sampled pairs in a mini-batch, the margin calculated may be too
weak if the positive pair is much more similar than the sampled negative pairs and too strong if it is very similar to the
negative pairs. The approach to monotonically increase the
margin during training is meant to address this as the positive and negative pairs will share similar alignment early in
training and begin to diverge closer to convergence. However, variable rates of convergence of different models on
different datasets make this growth rate difficult to tune and
this approach does not account for differences in the negative samples that appear in different mini-batches. To address this, we propose an adaptive margin based on relative
batch-wise similarity scores.
Class labels have been proposed to be used for generating adaptive margins based on class similarity between
positive and negative pairs [38, 42]. Likewise, prior work
explored a non-class dependant approach for an adaptive
similarity-based margin for human pose estimation [39]
where the mean joint error between a positive pose and a
hard sampled negative pose was used as a margin with the
triplet loss. This adaptively increases the margin when the
sampled negative pair is dissimilar to the positive pair in order to maximize the learning signal on less aligned negative
samples. We follow a similar intuition and simply replace
_M_ in Equation 2 with


_̸_



1
_M_ _xy_ = _α_ � _S_ ( _x_ _i_ _, y_ _i_ ) _−_ _̸_
_B −_ 1



_B_
� _I_ _i_ = _̸_ _j_ _S_ ( _x_ _i_ _, y_ _j_ )� _,_ (3)

_j_ =1



_̸_


6


|Language<br>Caption Model|Caption to Video<br>R@1 R@5 R@10 mAP|Video to Caption<br>R@1 R@5 R@10 mAP|Mean<br>R@1 R@5 R@10 mAP|
|---|---|---|---|
|Fasttext [5]<br>BERT [17]<br>BART [37]|17.1_±_0.8<br>44.0_±_0.6<br>57.2_±_0.5<br>30.2_±_0.5<br>25.9_±_0.6<br>55.5_±_1.2<br>67.0_±_1.1<br>39.7_±_0.7<br>**33.1**_±_0.9<br>**65.5**_±_1.5<br>**76.6**_±_1.3<br>**47.8**_±_1.1|24.1_±_0.5<br>49.9_±_0.6<br>61.8_±_1.3<br>36.6_±_0.3<br>33.3_±_1.4<br>62.1_±_1.0<br>72.0_±_0.6<br>46.5_±_1.2<br>**43.8**_±_0.7<br>**71.5**_±_1.2<br>**80.9**_±_1.6<br>**56.4**_±_0.7|20.6_±_0.5<br>46.9_±_0.6<br>59.5_±_0.8<br>33.4_±_0.4<br>29.6_±_0.8<br>58.8_±_1.0<br>69.5_±_0.8<br>43.1_±_0.8<br>**38.4**_±_0.4<br>**68.5**_±_1.3<br>**78.7**_±_1.4<br>**52.1**_±_0.8|


Table 1: **Language Caption Model Comparison on Video/Caption Retrieval:** Here we compare the video/caption retrieval results on
the test set of the Spoken Moments dataset using models trained with three different language models.







|Dataset|Loss|Caption to Video<br>R@1 R@5 R@10 mAP|Video to Caption<br>R@1 R@5 R@10 mAP|Mean<br>R@1 R@5 R@10 mAP|
|---|---|---|---|---|
|Vatex [69]|NCE<br>SHN<br>MMS<br>AMM|43.6_±_1.4<br>77.4_±_1.4<br>86.5_±_1.4<br>58.4_±_1.2<br>19.6_±_1.4<br>50.2_±_1.5<br>63.9_±_0.6<br>33.8_±_1.1<br>46.2_±_1.5<br>79.7_±_0.8<br>88.1_±_0.8<br>60.7_±_1.0<br>**48.7**_±_1.4<br>**82.0**_±_0.9<br>**89.3**_±_1.1<br>**63.0**_±_1.0|39.4_±_1.3<br>74.3_±_1.0<br>84.7_±_0.8<br>54.7_±_1.0<br>22.9_±_1.0<br>54.0_±_0.9<br>68.8_±_1.2<br>37.6_±_0.9<br>42.0_±_0.7<br>**77.7**_±_0.7<br>**86.8**_±_0.3<br>57.5_±_0.6<br>**43.0**_±_0.7<br>77.4_±_1.1<br>85.8_±_0.7<br>**58.3**_±_0.6|41.5_±_1.2<br>75.8_±_1.1<br>85.6_±_1.1<br>56.5_±_1.0<br>21.3_±_0.9<br>52.1_±_0.8<br>66.3_±_0.8<br>35.7_±_0.7<br>44.1_±_1.1<br>78.7_±_0.7<br>87.4_±_0.5<br>59.1_±_0.7<br>**45.9**_±_1.0<br>**79.7**_±_0.4<br>**87.5**_±_0.8<br>**60.7**_±_0.6|
|ActivityNet [33]|NCE<br>SHN<br>MMS<br>AMM|11.8_±_0.6<br>35.4_±_1.0<br>50.6_±_0.8<br>23.8_±_0.4<br>9.9_±_0.9<br>31.2_±_1.3<br>45.2_±_0.9<br>20.9_±_0.9<br>12.0_±_0.7<br>35.5_±_1.0<br>49.2_±_0.8<br>23.9_±_0.6<br>**17.2**_±_1.1<br>**46.1**_±_1.4<br>**60.0**_±_0.8<br>**30.6**_±_0.6|16.7_±_0.8<br>43.0_±_1.2<br>57.1_±_1.2<br>29.5_±_0.8<br>13.7_±_1.1<br>38.5_±_0.9<br>53.4_±_0.9<br>25.9_±_1.0<br>16.2_±_0.4<br>42.4_±_0.9<br>56.5_±_1.6<br>28.8_±_0.6<br>**20.9**_±_1.1<br>**50.1**_±_1.3<br>**62.4**_±_0.8<br>**34.3**_±_0.6|14.3_±_0.6<br>39.2_±_0.8<br>53.8_±_1.0<br>26.7_±_0.5<br>11.8_±_0.9<br>34.9_±_0.8<br>49.3_±_0.7<br>23.4_±_0.9<br>14.1_±_0.4<br>39.0_±_0.2<br>52.8_±_1.2<br>26.4_±_0.2<br>**19.1**_±_1.0<br>**48.1**_±_1.2<br>**61.2**_±_0.6<br>**32.5**_±_0.6|
|MSR-VTT [71]|NCE<br>SHN<br>MMS<br>AMM|20.7_±_0.9<br>51.0_±_0.7<br>66.6_±_1.2<br>35.0_±_0.4<br>11.3_±_0.2<br>32.0_±_1.0<br>44.9_±_1.4<br>21.9_±_0.3<br>17.6_±_1.1<br>46.5_±_0.9<br>61.6_±_0.9<br>31.5_±_0.6<br>**25.7**_±_0.8<br>**61.0**_±_0.8<br>**75.6**_±_0.7<br>**41.6**_±_0.6|30.7_±_1.4<br>65.1_±_0.7<br>78.2_±_1.3<br>46.1_±_1.2<br>22.1_±_0.9<br>54.5_±_1.6<br>68.9_±_1.4<br>37.0_±_1.1<br>28.3_±_1.1<br>63.1_±_1.4<br>76.1_±_0.9<br>43.8_±_1.1<br>**32.5**_±_1.5<br>**67.5**_±_1.7<br>**80.1**_±_1.4<br>**48.0**_±_1.2|25.7_±_1.0<br>58.1_±_0.6<br>72.4_±_1.2<br>40.6_±_0.7<br>16.7_±_0.5<br>43.3_±_0.5<br>56.9_±_0.9<br>29.5_±_0.5<br>23.0_±_0.9<br>54.8_±_0.6<br>68.9_±_0.7<br>37.6_±_0.6<br>**29.1**_±_0.8<br>**64.2**_±_1.0<br>**77.9**_±_1.0<br>**44.8**_±_0.8|
|S-MiT|NCE<br>SHN<br>MMS<br>AMM|**33.1**_±_0.9<br>**66.9**_±_1.9<br>**77.6**_±_1.2<br>**47.9**_±_0.7<br>23.1_±_1.3<br>55.4_±_1.6<br>69.3_±_1.3<br>37.7_±_1.1<br>26.5_±_1.3<br>58.3_±_1.4<br>72.0_±_0.9<br>41.1_±_1.1<br>**33.1**_±_0.9<br>65.5_±_1.5<br>76.6_±_1.3<br>47.8_±_1.1|43.0_±_0.8<br>**71.8**_±_0.9<br>80.7_±_1.2<br>55.8_±_0.7<br>41.4_±_1.1<br>70.8_±_0.9<br>79.5_±_1.0<br>54.5_±_0.7<br>43.3_±_1.3<br>71.2_±_1.4<br>79.9_±_0.8<br>55.8_±_1.2<br>**43.8**_±_0.7<br>71.5_±_1.2<br>**80.9**_±_1.6<br>**56.4**_±_0.7|38.0_±_0.5<br>**69.3**_±_1.4<br>**79.1**_±_1.1<br>51.8_±_0.6<br>32.3_±_0.9<br>63.1_±_1.1<br>74.4_±_1.1<br>46.1_±_0.8<br>34.9_±_1.2<br>64.8_±_1.2<br>76.0_±_0.8<br>48.5_±_1.1<br>**38.4**_±_0.4<br>68.5_±_1.3<br>78.7_±_1.4<br>**52.1**_±_0.8|


Table 2: **Loss Function Comparison for Video/Caption Retrieval:** Models trained on four datasets with different loss functions are
compared. The proposed AMM loss function consistently achieves the best performance.







|Spoken<br>Caption Model|Loss|Caption to Video<br>R@1 R@5 R@10 mAP|Video to Caption<br>R@1 R@5 R@10 mAP|Mean<br>R@1 R@5 R@10 mAP|
|---|---|---|---|---|
|ResDavenet [25]|NCE<br>SHN<br>MMS<br>AMM|30.7_±_0.6<br>57.1_±_0.6<br>67.6_±_1.0<br>42.9_±_0.8<br>30.2_±_1.1<br>56.9_±_0.8<br>66.8_±_0.5<br>42.6_±_1.0<br>32.1_±_1.1<br>58.9_±_1.0<br>68.6_±_1.5<br>44.4_±_0.8<br>**34.8**_±_1.1<br>**62.0**_±_1.1<br>**70.4**_±_1.2<br>**47.0**_±_1.1|29.3_±_1.0<br>55.8_±_1.2<br>66.2_±_1.4<br>41.8_±_0.9<br>31.0_±_1.2<br>57.2_±_0.8<br>67.1_±_0.9<br>43.2_±_1.0<br>32.3_±_1.3<br>57.9_±_1.1<br>68.1_±_1.5<br>44.3_±_1.2<br>**34.6**_±_1.5<br>**60.8**_±_1.6<br>**70.0**_±_0.9<br>**46.8**_±_1.2|30.0_±_0.8<br>56.4_±_0.9<br>66.9_±_1.2<br>42.3_±_0.8<br>30.6_±_1.1<br>57.0_±_0.8<br>67.0_±_0.7<br>42.9_±_1.0<br>32.2_±_1.2<br>58.4_±_1.0<br>68.4_±_1.5<br>44.3_±_1.0<br>**34.7**_±_1.2<br>**61.4**_±_1.4<br>**70.2**_±_1.1<br>**46.9**_±_1.1|
|Wav2Vec [57]|NCE<br>SHN<br>MMS<br>AMM|32.6_±_0.7<br>60.4_±_0.8<br>70.3_±_1.6<br>45.3_±_0.8<br>27.8_±_1.0<br>54.2_±_1.7<br>64.9_±_1.8<br>40.1_±_1.0<br>33.6_±_0.6<br>60.5_±_1.2<br>**71.4**_±_1.1<br>46.1_±_0.7<br>**35.0**_±_0.4<br>**61.7**_±_0.9<br>71.0_±_0.9<br>**47.1**_±_0.6|30.9_±_1.0<br>59.6_±_0.9<br>69.8_±_1.1<br>43.9_±_0.8<br>28.4_±_0.7<br>53.7_±_1.6<br>64.2_±_1.7<br>40.4_±_0.8<br>33.4_±_1.0<br>60.5_±_1.7<br>**70.3**_±_1.1<br>45.7_±_0.8<br>**34.7**_±_1.5<br>**61.1**_±_0.9<br>70.2_±_0.9<br>**46.8**_±_1.2|31.8_±_0.7<br>60.0_±_0.8<br>70.0_±_1.3<br>44.6_±_0.8<br>28.1_±_0.8<br>53.9_±_1.6<br>64.6_±_1.7<br>40.2_±_0.9<br>33.5_±_0.6<br>60.5_±_1.4<br>**70.8**_±_1.1<br>45.9_±_0.7<br>**34.8**_±_0.9<br>**61.4**_±_0.9<br>70.6_±_0.8<br>**47.0**_±_0.9|
|ResNet-34|NCE<br>SHN<br>MMS<br>AMM|32.2_±_1.3<br>59.7_±_1.4<br>70.3_±_1.3<br>44.8_±_1.1<br>32.7_±_1.1<br>60.3_±_1.3<br>71.0_±_1.1<br>45.5_±_1.0<br>35.3_±_1.0<br>62.5_±_1.2<br>72.8_±_1.8<br>47.7_±_0.6<br>**36.3**_±_0.5<br>**63.9**_±_1.7<br>**73.7**_±_1.6<br>**48.9**_±_0.8|32.8_±_1.8<br>58.8_±_1.3<br>69.2_±_1.9<br>45.1_±_1.4<br>33.1_±_1.0<br>60.1_±_1.5<br>70.1_±_1.3<br>45.6_±_0.9<br>36.7_±_0.9<br>62.2_±_0.8<br>72.1_±_1.6<br>48.6_±_0.9<br>**37.5**_±_1.7<br>**63.5**_±_1.9<br>**73.7**_±_1.6<br>**49.6**_±_1.5|32.5_±_1.4<br>59.2_±_1.3<br>69.7_±_1.5<br>45.0_±_1.2<br>32.9_±_1.0<br>60.2_±_1.4<br>70.6_±_1.2<br>45.6_±_0.9<br>36.0_±_0.7<br>62.3_±_1.0<br>72.5_±_1.6<br>48.2_±_0.7<br>**36.9**_±_1.1<br>**63.7**_±_1.7<br>**73.7**_±_1.5<br>**49.2**_±_1.2|
|ResNet-50|NCE<br>SHN<br>MMS<br>AMM|32.7_±_0.6<br>60.8_±_1.9<br>70.6_±_1.6<br>45.6_±_0.8<br>33.9_±_0.6<br>60.1_±_1.4<br>70.9_±_1.3<br>45.8_±_0.7<br>37.2_±_0.9<br>65.4_±_0.6<br>75.1_±_1.3<br>50.0_±_0.7<br>**39.5**_±_1.3<br>**65.7**_±_1.5<br>**75.5**_±_1.3<br>**51.6**_±_1.1|33.1_±_1.0<br>59.4_±_1.5<br>69.6_±_1.4<br>45.5_±_0.9<br>34.0_±_1.2<br>60.6_±_1.8<br>70.1_±_1.4<br>46.0_±_1.1<br>37.8_±_1.3<br>64.6_±_1.1<br>74.2_±_0.9<br>50.1_±_1.1<br>**40.1**_±_0.7<br>**66.3**_±_1.1<br>**74.5**_±_1.2<br>**52.0**_±_0.7|32.9_±_0.5<br>60.1_±_1.7<br>70.1_±_1.4<br>45.5_±_0.8<br>34.0_±_0.8<br>60.3_±_1.5<br>70.5_±_1.3<br>45.9_±_0.8<br>37.5_±_1.0<br>65.0_±_0.8<br>74.7_±_1.1<br>50.0_±_0.9<br>**39.8**_±_0.9<br>**66.0**_±_1.2<br>**75.**0_±_1.1<br>**51.8**_±_0.8|


Table 3: **Spoken Caption Model Comparison:** Models trained with different spoken caption architectures and different loss functions are
compared for video/caption retrieval on the S-MiT test set. The proposed AMM loss function consistently achieves the highest performance
while ResNet-50 is found to be significantly stronger than the other architectures.

|Trained On|Evaluated On|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|**Trained On**|**Vatex**<br>R@1<br>R@5<br>R@10<br>mAP|**ActivityNet**<br>R@1<br>R@5<br>R@10<br>mAP|**MSR-VTT**<br>R@1<br>R@5<br>R@10<br>mAP|**S-MiT**<br>R@1<br>R@5<br>R@10<br>mAP|**Mean**<br>R@1<br>R@5<br>R@10<br>mAP|
|**Vatex**<br>**ActivityNet**<br>**MSR-VTT**<br>**S-MiT**|**45.9**<br>**79.7**<br>**87.5**<br>**60.7**<br>25.0<br>56.0<br>68.4<br>39.1<br>21.0<br>51.3<br>64.8<br>35.1<br>42.7<br>75.4<br>84.2<br>57.1|15.6<br>39.4<br>51.7<br>27.1<br>**19.1**<br>**48.1**<br>**61.2**<br>**32.5**<br>9.9<br>28.3<br>39.7<br>19.6<br>17.6<br>41.6<br>53.8<br>29.2|22.6<br>49.8<br>63.2<br>35.6<br>15.1<br>37.1<br>50.4<br>26.4<br>29.1<br>64.2<br>**77.9**<br>44.8<br>**33.1**<br>**64.8**<br>77.4<br>**47.6**|13.1<br>33.0<br>45.8<br>23.5<br>9.8<br>28.7<br>40.6<br>19.7<br>14.6<br>39.3<br>53.4<br>26.9<br>**38.4**<br>**68.5**<br>**78.7**<br>**52.1**|24.3<br>50.5<br>62.1<br>36.7<br>17.3<br>42.5<br>55.2<br>29.4<br>18.7<br>45.8<br>59.0<br>31.6<br>**33.0**<br>**62.6**<br>**73.5**<br>**46.5**|



Table 4: **Cross Dataset Evaluation on Video/Caption Retrieval:** Here we compare the generalization performance of models trained
on four different datasets for video/caption retrieval. Each model is trained on a single dataset and we average the evaluation on five 1k
video-caption samples from the test set of each other dataset. We additionally show the mean performance accross datasets. The S-MiT
model shows it generalizes very strongly to the other datasets even beating the MSR-VTT model on its own test set.



where _α_ is a dampening parameter to weight the strength
of the margin. When _M_ _xy_ in Equation 3 is applied to Equation 2 with _α_ = 1, the margin removes the positive pair similarity from the optimization. Ablation studies on different
alpha values can be found in the supplementary material. In
practice we use _α_ = 0 _._ 5 in our experiments.


This has the effect of increasing the margin as the dif


ference between the true pair similarity and the similarity
of the negative pairs increases. As the training progresses,
and the learning approaches convergence, the margin generally increases with the increased separation between positive and negative pair-wise similarities. This also removes
the need to tune the margin and growth rate which may
have different optimal values for different similarity met


7


rics, batch sizes and datasets.
We refer to this as an **Adaptive Mean Margin** (AMM)
for contrastive learning and show in Section 5 the effect of
applying this adaptive margin.


**5. Results**


**5.1. Video/Caption Retrieval**


In Tables 1, 2 and 3 we show results of R@k recall scores
(for _k_ = 1 _,_ 5 _,_ 10) and mean average precision (mAP) on
both caption to video and video to caption retrieval. Results
are averaged over five random sets of 1k video-caption pairs
from the test set. Each model in Tables 1 and 2 uses the

output of a pretrained ASR model, the Google Cloud ASR
engine, as input into a trained language model to generate
a feature representation for each caption. Alternatively, the
spoken caption models align visual representations directly
from the audio signal without pretrained modules.
Table 1 shows the result of using different language models to generate our caption representations from ASR text
transcriptions. Each of these models was trained using the
proposed AMM loss function described in Section 4.3. We
evaluate the AMM loss in Table 2 where we compare the results on the NCE, SHN, MMS and AMM loss functions de
scribed in Sections 2.3.1 and 4.3 on four different datasets

(the proposed Spoken Moments in Time dataset (S-MiT) as
well as Vatex-en [69], MSR-VTT [71] and ActivityNet Captions [1] [33]). The proposed AMM loss function consistently
achieves the best results across each dataset in Table 2 and

the BART language model provides the strongest representations for the retrieval task in Table 1.

Table 2 shows a comparison of our AMM approach to
other methods for cross-modal contrastive learning. We
use the BART language model [37] to generate representations of words transcribed from the audio captions via a pretrained ASR model. Replacing the monotonically increasing margin used in MMS [29] with an adaptive margin that
scales with the samples in a batch achieves the strongest results. We observed that as training continues and the margin
in MMS continues to grow the training performance begins
to degrade. This is likely due to the margin becoming too
large for stable training as described in prior work [68].
In Table 3, we show a comparison of different spoken
caption models with different loss functions. The proposed
AMM approach beats the other loss functions consistently.


**5.2. Cross Dataset Evaluation**


To further examine the strength of our proposed Spoken
Moments in Time (S-MiT) dataset, we compare the generalization performance of models trained on four different
datasets (S-MiT as well as Vatex-en [69], MSR-VTT [71]
and ActivityNet Captions [33]) for video/caption retrieval


1 We used the groundtruth timestamps to get corresponding video clips.



(see Table 2 (right) for comparisons of these datasets). We
train each model on a single dataset using the approach described in Section 4.3 and evaluate on the test set from each

other dataset. For example, a model trained on Vatex is
evaluated on, in addition to its own, the test sets of ActivityNet Captions, MSR-VTT and S-MiT. We sample five
sets of 1k video-caption pairs from each test set. This allows us to fairly compare results across test sets of different
sizes (see supplementary material for full test set results).
Each model in this evaluation was trained using the BART

[37] language model and the proposed AMM loss function
which was found to give the best results (see Tables 1, 2).
We evaluate the models using the mean between the videoto-caption and caption-to-video retrieval tasks. We are not
able to compare the spoken caption models from Table 3
here as the other datasets only include text captions.
In Table 4, we can see that the S-MiT model generalizes
better than the other models in spite of the additional noise
introduced by the ASR model. Additionally, the restriction
to 3-second videos in S-MiT does not hinder it’s ability to
generalize to the much longer videos of the other datasets.


**6. Conclusions**


In this paper, we have introduced the Spoken Moments in
Time dataset which includes 500k pairs of video clips and
corresponding spoken descriptions. This new dataset represents the largest video caption dataset available and will
serve as a new benchmark for the community. We compared various benchmark models for learning joint representations between captions and videos, and evaluated our
approaches on multiple datasets to highlight the strength
of the models as well as the ability of models trained on
our proposed dataset to generalize to tasks in other datasets.
With these results we are confident that the presented Spoken Moments dataset will have a positive impact on the
fields of video understanding and cross-modal learning.


**7. Acknowledgment**


This work was supported by the MIT-IBM Watson AI
Lab as well as the Intelligence Advanced Research Projects
Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) contract number D17PC00341.


**8. Disclaimer**


The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding
any copyright annotation thereon. The views and conclusions contained herein are those of the authors and should

not be interpreted as necessarily representing the official
policies or endorsements, either expressed or implied, of
IARPA, DOI/IBC, or the U.S. Government.



8


**References**


[1] Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Apostol (Paul) Natsev, George Toderici, Balakrishnan Varadarajan, and Sudheendra Vijayanarasimhan. Youtube-8m: A
large-scale video classification benchmark. _arXiv preprint_
_arXiv:1609.08675_, 2016. 2

[2] Jean-Baptiste Alayrac, Piotr Bojanowski, Nishant Agrawal,
Josef Sivic, Ivan Laptev, and Simon Lacoste-Julien. Unsupervised learning from narrated instruction videos. In _IEEE_
_Conf. Comput. Vis. Pattern Recog._, June 2016. 3

[3] Yusuf Aytar, Carl Vondrick, and Antonio Torralba. Soundnet: Learning sound representations from unlabeled video.
In _Adv. Neural Inform. Process. Syst._, 2016. 3

[4] Angie Boggust, Kartik Audhkhasi, Dhiraj Joshi, David Harwath, Samuel Thomas, Rogerio Feris, Dan Gutfreund, Yang
Zhang, Antonio Torralba, Michael Picheny, and James Glass.
Grounding spoken words in unlabeled video. In _CVPR Sight_
_and Sound Workshop_, 2019. 2

[5] Piotr Bojanowski, Edouard Grave, Armand Joulin, and
Tomas Mikolov. Enriching word vectors with subword information. _Transactions of the Association for Computational_
_Linguistics_, 5:135–146, 2017. 6, 7

[6] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem,
and Juan Carlos Niebles. Activitynet: A large-scale video
benchmark for human activity understanding. In _IEEE Conf._
_Comput. Vis. Pattern Recog._, 2015. 2

[7] Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe
Hillier, and Andrew Zisserman. A short note about kinetics600. _arXiv preprint arXiv:1808.01340_, 2018. 3

[8] Joao Carreira and Andrew Zisserman. Quo vadis, action
recognition? a new model and the kinetics dataset. In _Int._
_Conf. Comput. Vis._, 2017. 2

[9] David Chen and William B Dolan. Collecting highly parallel
data for paraphrase evaluation. In _Proceedings of the 49th_
_Annual Meeting of the Association for Computational Lin-_
_guistics: Human Language Technologies_, pages 190–200,
2011. 3, 5

[10] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning
of visual representations. _arXiv preprint arXiv:2002.05709_,
2020. 3, 4

[11] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick.
Microsoft coco captions: Data collection and evaluation
server. _arXiv preprint arXiv:1504.00325_, 2015. 2, 3

[12] Dima Damen, Hazel Doughty, Giovanni Maria Farinella,
, Antonino Furnari, Jian Ma, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price,
and Michael Wray. Rescaling egocentric vision. _CoRR_,
abs/2006.13256, 2020. 3, 5

[13] Dima Damen, Hazel Doughty, Giovanni Maria Farinella,
Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide
Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and
Michael Wray. Scaling egocentric vision: The epic-kitchens
dataset. In _Eur. Conf. Comput. Vis._, 2018. 3, 5

[14] Pradipto Das, Chenliang Xu, Richard F Doell, and Jason J
Corso. A thousand frames in just a few words: Lingual de


scription of videos through latent topics and sparse object
stitching. In _IEEE Conf. Comput. Vis. Pattern Recog._, pages
2634–2641, 2013. 3

[15] Yann N Dauphin, Angela Fan, Michael Auli, and David
Grangier. Language modeling with gated convolutional networks. In _International conference on machine learning_,
pages 933–941, 2017. 5

[16] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In _IEEE Conf. Comput. Vis. Pattern Recog._, 2009.
2, 5

[17] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In _North American_
_Chapter of the Association for Computational Linguistics_,
pages 4171–4186, 2019. 4, 6, 7

[18] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In
_Int. Conf. Comput. Vis._, pages 1422–1430, 2015. 3

[19] Fartash Faghri, David J Fleet, Jamie Ryan Kiros, and Sanja
Fidler. VSE++: Improving visual-semantic embeddings with
hard negatives. _arXiv preprint arXiv:1707.05612_, 2017. 4

[20] Spandana Gella, Mike Lewis, and Marcus Rohrbach. A
dataset for telling the stories of social media videos. In
_Empirical Methods in Natural Language Processing_, pages
968–974, Oct.-Nov. 2018. 2, 3, 5

[21] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim,
Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz
Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax,
and Roland Memisevic. The “something something” video
database for learning and evaluating visual common sense.
In _Int. Conf. Comput. Vis._, Oct 2017. 1, 2

[22] Chunhui Gu, Chen Sun, David A. Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, Cordelia Schmid, and Jitendra Malik. Ava: A video
dataset of spatio-temporally localized atomic visual actions.
In _IEEE Conf. Comput. Vis. Pattern Recog._, June 2018. 1

[23] Michael Gutmann and Aapo Hyv¨arinen. Noise-contrastive
estimation: A new estimation principle for unnormalized
statistical models. In _Proceedings of the Thirteenth Inter-_
_national Conference on Artificial Intelligence and Statistics_,
pages 297–304. JMLR Workshop and Conference Proceedings, 2010. 4

[24] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In _IEEE_
_Conf. Comput. Vis. Pattern Recog._, volume 2, pages 1735–
1742, 2006. 3

[25] David Harwath, Adria Recasens, Didac Suris, Galen
Chuang, Antonio Torralba, and James Glass. Jointly discovering visual objects and spoken words from raw sensory
input. _Int. J. Comput. Vis._, (128):620–641, 2020. 2, 3, 4, 5,
6, 7, 12

[26] David Harwath, Antonio Torralba, and James Glass. Unsupervised learning of spoken language with visual context. In
_Adv. Neural Inform. Process. Syst._, 2016. 2, 3, 12



9


[27] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In _IEEE Conf._
_Comput. Vis. Pattern Recog._, pages 770–778, 2016. 6

[28] Micah Hodosh, Peter Young, and Julia Hockenmaier. Framing image description as a ranking task: Data, models and
evaluation metrics. _Journal of Artificial Intelligence Re-_
_search_, 47:853–899, 2013. 2, 3

[29] Gabriel Ilharco, Yuan Zhang, and Jason Baldridge. Largescale representation learning from visually grounded untranscribed speech. In _Conference on Computational Natural_
_Language Learning_, pages 55–65, Nov. 2019. 2, 3, 4, 8

[30] Aren Jansen, Manoj Plakal, Ratheet Pandya, Daniel PW Ellis, Shawn Hershey, Jiayang Liu, R Channing Moore, and
Rif A Saurous. Unsupervised learning of semantic audio representations. In _IEEE international conference on acoustics,_
_speech and signal processing_, pages 126–130, 2018. 4

[31] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang,
Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola,
Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman,
and Andrew Zisserman. The kinetics human action video

dataset. _arXiv preprint arXiv:1705.06950_, 2017. 1, 2, 3, 5,

12

[32] Diederik P Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. _arXiv preprint arXiv:1412.6980_,
2014. 12

[33] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and
Juan Carlos Niebles. Dense-captioning events in videos. In
_Int. Conf. Comput. Vis._, 2017. 2, 3, 5, 7, 8, 12, 14

[34] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson,
Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, Michael Bernstein, and Li
Fei-Fei. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _Int. J. Comput._
_Vis._, 123(1):32–73, 2017. 2, 3

[35] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.
Imagenet classification with deep convolutional neural networks. _Adv. Neural Inform. Process. Syst._, 25:1097–1105,
2012. 1, 6, 12

[36] Hilde Kuehne, Hueihan Jhuang, Rainer Stiefelhagen, and
Thomas Serre. Hmdb51: A large video database for human motion recognition. In _High Performance Computing_
_in Science and Engineering ‘12_, pages 571–582, 2013. 2

[37] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and
Luke Zettlemoyer. Bart: Denoising sequence-to-sequence
pre-training for natural language generation, translation, and
comprehension. _arXiv preprint arXiv:1910.13461_, 2019. 6,
7, 8, 12, 13

[38] Aoxue Li, Weiran Huang, Xu Lan, Jiashi Feng, Zhenguo Li,
and Liwei Wang. Boosting few-shot learning with adaptive
margin loss. In _IEEE Conf. Comput. Vis. Pattern Recog._,
2020. 6

[39] Sijin Li, Weichen Zhang, and Antoni B. Chan. Maximummargin structured learning with deep networks for 3d human
pose estimation. In _Int. Conf. Comput. Vis._, December 2015.
6

[40] Ji Lin, Chuang Gan, and Song Han. Tsm: Temporal shift
module for efficient video understanding. In _Int. Conf. Com-_
_put. Vis._, October 2019. 2, 6, 12




[41] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence
Zitnick. Microsoft coco: Common objects in context. In _Eur._
_Conf. Comput. Vis._, pages 740–755. 2014. 5

[42] Hao Liu, Xiangyu Zhu, Zhen Lei, and Stan Z. Li. Adaptiveface: Adaptive margin and sampling for face recognition. In
_IEEE Conf. Comput. Vis. Pattern Recog._, June 2019. 6

[43] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert:
Pretraining task-agnostic visiolinguistic representations for
vision-and-language tasks. In _Adv. Neural Inform. Process._
_Syst._, pages 13–23, 2019. 3

[44] Danny Merkx, Stefan L. Frank, and Mirjam Ernestus. Language learning using speech to image retrieval. In _Interna-_
_tional Speech Communication Association_, September 2019.
2

[45] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac,
Makarand Tapaswi, Ivan Laptev, and Josef Sivic.
Howto100m: Learning a text-video embedding by watching
hundred million narrated video clips. In _Int. Conf. Comput._
_Vis._, pages 2630–2640, 2019. 3, 5, 6

[46] Mathew Monfort, Alex Andonian, Bolei Zhou, Kandan Ramakrishnan, Sarah Adel Bargal, Tom Yan, Lisa Brown,
Quanfu Fan, Dan Gutfreund, Carl Vondrick, and Aude Oliva.

Moments in time dataset: one million videos for event

understanding. _IEEE Trans. Pattern Anal. Mach. Intell._,
42(2):502–508, 2019. 1, 2, 3, 4

[47] Mathew Monfort, Kandan Ramakrishnan, Alex Andonian,
Barry A McNamara, Alex Lascelles, Bowen Pan, Quanfu
Fan, Dan Gutfreund, Rogerio Feris, and Aude Oliva.
Multi-moments in time: Learning and interpreting models for multi-action video understanding. _arXiv preprint_
_arXiv:1911.00232_, 2019. 1, 2, 3, 5, 6, 12

[48] Andrew Owens, Jiajun Wu, Josh McDermott, William Freeman, and Antonio Torralba. Ambient sound provides supervision for visual learning. In _Eur. Conf. Comput. Vis._, 2016.
3

[49] Bryan A. Plummer, Liwei Wang, Chris M. Cervantes,
Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik.
Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In _Int. Conf._
_Comput. Vis._, December 2015. 2, 3

[50] Esteban Real, Jonathon Shlens, Stefano Mazzocchi, Xin Pan,
and Vincent Vanhoucke. Youtube-boundingboxes: A large
high-precision human-annotated data set for object detection
in video. In _IEEE Conf. Comput. Vis. Pattern Recog._, pages
5296–5305, 2017. 1

[51] Michaela Regneri, Marcus Rohrbach, Dominikus Wetzel,
Stefan Thater, Bernt Schiele, and Manfred Pinkal. Grounding action descriptions in videos. _Transactions of the Asso-_
_ciation for Computational Linguistics_, 1:25–36, 2013. 3, 5

[52] Anna Rohrbach, Marcus Rohrbach, Wei Qiu, Annemarie
Friedrich, Manfred Pinkal, and Bernt Schiele. Coherent
multi-sentence video description with variable level of detail. In _German conference on pattern recognition_, pages
184–195, 2014. 3



10


[53] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt
Schiele. A dataset for movie description. In _IEEE Conf._
_Comput. Vis. Pattern Recog._, pages 3202–3212, 2015. 3, 5

[54] Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket
Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville,
and Bernt Schiele. Movie description. _Int. J. Comput. Vis._,
123(1):94–120, 2017. 3

[55] Marcus Rohrbach, Anna Rohrbach, Michaela Regneri,
Sikandar Amin, Mykhaylo Andriluka, Manfred Pinkal, and
Bernt Schiele. Recognizing fine-grained and composite activities using hand-centric features and script data. _Int. J._
_Comput. Vis._, 119(3):346–373, Sept. 2016. 3

[56] Andrew Rouditchenko, Angie Boggust, David Harwath,
Dhiraj Joshi, Samuel Thomas, Kartik Audhkhasi, Rogerio Feris, Brian Kingsbury, Michael Picheny, Antonio Torralba, and James Glass. Avlnet: Learning audio-visual
language representations from instructional videos. In
_arXiv:2006.09199_, 2020. 5

[57] Steffen Schneider, Alexei Baevski, Ronan Collobert, and
Michael Auli. wav2vec: Unsupervised pre-training for
speech recognition. In _arXiv:1904.05862_, 2019. 6, 7

[58] Florian Schroff, Dmitry Kalenichenko, and James Philbin.
Facenet: A unified embedding for face recognition and clustering. In _IEEE Conf. Comput. Vis. Pattern Recog._, June
2015. 3, 4

[59] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu
Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _An-_
_nual Meeting of the Association for Computational Linguis-_
_tics_, pages 2556–2565, 2018. 2, 3

[60] Gunnar A Sigurdsson, G¨ul Varol, Xiaolong Wang, Ali
Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in
homes: Crowdsourcing data collection for activity understanding. In _Eur. Conf. Comput. Vis._, pages 510–526, 2016.
2, 5

[61] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. In _Adv._
_Neural Inform. Process. Syst._, pages 568–576, 2014. 2

[62] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah.

Ucf101: A dataset of 101 human actions classes from videos

in the wild. _arXiv preprint arXiv:1212.0402_, 2012. 2

[63] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and
Cordelia Schmid. Videobert: A joint model for video and
language representation learning. In _Int. Conf. Comput. Vis._,
October 2019. 4

[64] Didac Suris, Adria Recasens, David Bau, David Harwath,
James Glass, and Antonio Torralba. Learning words by
drawing images. In _IEEE Conf. Comput. Vis. Pattern Recog._,
June 2019. 2

[65] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani,
and Manohar Paluri. Learning spatiotemporal features with
3d convolutional networks. In _IEEE Conf. Comput. Vis. Pat-_
_tern Recog._, 2015. 2




[66] A¨aron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. _arXiv_
_preprint arXiv:1807.03748_, 2018. 3, 4

[67] Arun Balajee Vasudevan, Dengxin Dai, and Luc Van Gool.
Object referring in visual scene with spoken language. In
_IEEE Winter Conference on Applications of Computer Vi-_
_sion_, pages 1861–1870, 2018. 2

[68] Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong
Gong, Jingchao Zhou, Zhifeng Li, and Wei Liu. Cosface:
Large margin cosine loss for deep face recognition. In _IEEE_
_Conf. Comput. Vis. Pattern Recog._, June 2018. 4, 8

[69] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang
Wang, and William Yang Wang. Vatex: A large-scale, highquality multilingual dataset for video-and-language research.
In _Int. Conf. Comput. Vis._, October 2019. 3, 5, 7, 8, 12, 14

[70] Yufei Wang, Zhe Lin, Xiaohui Shen, Scott Cohen, and Garrison W. Cottrell. Skeleton key: Image captioning by skeletonattribute decomposition. In _IEEE Conf. Comput. Vis. Pattern_
_Recog._, July 2017. 2, 3

[71] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large
video description dataset for bridging video and language.
IEEE Conf. Comput. Vis. Pattern Recog., June 2016. 2, 3, 5,
7, 8, 12, 14

[72] Serena Yeung, Olga Russakovsky, Ning Jin, Mykhaylo Andriluka, Greg Mori, and Li Fei-Fei. Every moment counts:
Dense detailed labeling of actions in complex videos. _Int. J._
_Comput. Vis._, 126(2-4):375–389, 2018. 1

[73] Junjie Zhang, Qi Wu, Chunhua Shen, Jian Zhang, and Jianfeng Lu. Multi-label image classification with regional latent
semantic dependencies. _IEEE Trans. Multimedia_, 2018. 1

[74] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful
image colorization. In _Eur. Conf. Comput. Vis._, 2016. 3

[75] Richard Zhang, Phillip Isola, and Alexei A Efros. Split-brain
autoencoders: Unsupervised learning by cross-channel prediction. In _IEEE Conf. Comput. Vis. Pattern Recog._, pages
645–654, 2017. 3

[76] Hang Zhao, Chuang Gan, Andrew Rouditchenko, Carl Vondrick, Josh McDermott, and Antonio Torralba. The sound of
pixels. In _Eur. Conf. Comput. Vis._, September 2018. 3

[77] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva,
and Antonio Torralba. Places: A 10 million image database
for scene recognition. _IEEE Trans. Pattern Anal. Mach. In-_
_tell._, 2017. 2

[78] Bolei Zhou, Agata Lapedriza, Jianxiong Xiao, Antonio Torralba, and Aude Oliva. Learning deep features for scene
recognition using places database. In _Adv. Neural Inform._
_Process. Syst._, pages 487–495. 2014. 2, 5

[79] Luowei Zhou, Nathan Louis, and Jason J Corso. Weaklysupervised video object grounding from text by loss weighting and object interaction. In _Brit. Mach. Vis. Conf._, 2018.
3

[80] Luowei Zhou, Chenliang Xu, and Jason J. Corso. Towards automatic learning of procedures from web instructional videos. In _AAAI_, 2018. 2, 3, 5



11


**A. Annotation**


We follow the approach used to collect the Places Audio
Caption dataset [26, 25] and collect audio descriptions of
each video in the dataset using Amazon Mechanical Turk
(AMT). In order to ensure that we have a large and diverse
dataset, we collect an audio description using AMT for each
video in a set of 500k randomly selected videos from the
training set and at least two unique descriptions for each
video in the 10k videos used for both the validation and test

sets. Each AMT worker is presented with a task of recording themselves describing 10 different videos. Each video
is shown on the left of the screen while a video with an ex
ample text description is shown on the right. This example
helps to show the workers the types of descriptions we are
looking for and the amount of detail we expect from them.
This example stays on the right side of the screen throughout the task while the target videos on the left cycle as the
worker completes each description. Figure 4 shows an example of this interface with an example video and caption
on the right and a target video on the left. Below each target description is a button that allows the worker to start
recording their voice as they describe the video. Once they
press this button, the video is removed from the screen and
the recording is started. We block the worker from seeing the video while recording the description to ensure that
the recordings are concise and pertain only to the important events highlighted in their memory. We use the Google
Cloud ASR engine to verify the quality of each recorded description and flag AMT workers for poor performance. This
is done by checking that the generated text has at least five
words, is unique (some bots repeat pre-recorded audio to
trick the system) and that the audio is at least three seconds
long. If any of these checks fail we don’t let the worker continue to the next video until they record a new description
that passes our checks. Once the descriptions are recorded,
we periodically sample videos to check the quality of the
audio paired with the ASR to ensure they match the videos
and have an appropriate level of detail. If these checks fail,
we flag the workers that recorded the descriptions, don’t allow them to record in the future and recheck all of their

recorded data. This process allows us to ensure a strong
level of quality in our collected spoken captions. Examples
of some of the videos and corresponding text transcriptions
of the descriptions we collected can be seen in Figure 1.


**B. Implementation Details**


We train each model on a server with 8 24GB Titan RTX

cards using a mini-batch size of 2048 for 100 epochs. We
examine the effect of the mini-batch size on learning in the
next section. We take the best parameters as evaluated on
the evaluation set of the training dataset after each epoch.
We repeat this process for two phases of training. First
we freeze the visual backbone models and train only the



projection heads (including the full caption model for the
spoken models) and then, in a second round, allow the full
visual model to train as well. We keep the language and
ASR components frozen for the language caption models
and reserve fine-tuning these components for future work.
For model training, we use an Adam [32] optimizer where a
fixed learning rate of 0 _._ 001 and 0 _._ 00001 are set for the first
and the second round model training, respectively.


**C. Ablation Studies**


In Tables 5, 6, 7, 8, and 9, we show several ablation studies. Unless otherwise listed in the table we use the proposed
AMM loss function with the BART [37] language model
as part of the language caption model described in Section
4.2.1 for each experiment. Results are averaged over five
rounds with a single random batch of 1k caption-video pairs
from the test set. Due to the increased computation demand
of these studies we freeze the base models and train the projection heads for alignment. We use the best model settings
found in this analysis to train the full models with results
reported in Section 5.
Table 5 shows the effect of using two different pretrained
temporal shift [40] video models on four different datasets
in order to choose the most appropriate base models (MultiMoments in Time (M-MiT) [47] or Kinetics [31]). Here we
use the BART language model and the proposed AMM loss
function as described in Section 4 as this combination gave
us the best results on each dataset.

Table 6 compares the effect of the video model (TSM)
trained for action recognition and the 2D model trained for
object recognition. Most captions reference both objects
and actions in a video with an average of 4.37 nouns used
per caption compared to 1.58 verbs. The strength of the
2D obect model makes sense when we consider this prevalence of nouns in the captions. The combination of the TSM
model trained on M-MiT [47] and the 2D models trained on
ImageNet [35] provided the best performance when used
with the model described in Section 4.

In Tables 7 and 8 we compare the the effect of the batch
size and projection size on the performance of the S-MiT
model described in Section 4 in order to validate our choice

of a 2048 batch and a 4096 projection. Similarly, Table 9
shows the effect of using the caption sampling approach for
the transcription model as described in Section 4.2.1. In
Table 10, we explore different dampening parameters.


**D. Cross Dataset Generalization**


In Table 11, we expand on Table 4 and compare the generalization performance of models trained on four different
datasets (S-MiT as well as Vatex-en [69], MSR-VTT [71]
and ActivityNet Captions [33]) for video/caption retrieval
on their full test sets. In Table 4 we ran the comparison on



12


Figure 4: **Spoken Caption Collection:** Target videos for which descriptions are collected on the left and a video with an example text
description is always visible on the right.



five samples of 1k video-caption pairs to be consistent on
evaluating across different size test sets. Here we evaluate
on the full test set of each dataset to provide a baseline for
each test set. The strength of the model trained on S-MiT is
even more evident here as it achieves higher results on the
test sets of both ActivityNet and MSR-VTT than the models trained on those datasets. It even comes very close to the
performance of the Vatex model on the Vatex test set. This
shows that the scale and diversity of the S-MiT dataset is
highly beneficial to training robust models.


**E. Qualitative Results**


In Tables 12 and 13, we show the top five retrieval results
for some examples from the Spoken Moments dataset. For
this analysis, we use the language caption model described
in Section 4.2.1 with the BART [37] language model and
the proposed AMM loss function. Table 12 shows the top
five retrieved captions given a query video, while Table 13
shows the top five retrieved videos given a query caption.
Blue boxes indicate the ground-truth results.



Our model retrieves results by recognizing key objects
or environments in the videos. For example, in Table 12
(c), _lettuce_ is distinguished from the other vegetables. In
Table 13 (f), the model not only _recognizes the planets_ in
space but also _understands that they are crashing into each_
_other_ . Some of the examples show that the top retrieval result is not the ground-truth. However, as we can see, the top
predictions are typically still a strong match for the queries,
as in (e), (i) in Table 12 and (a), (b) in Table 13.
For this demonstration, we use transcribed words from
the audio captions using a pretrained ASR model. Noise in
these transcriptions may contribute to some errors. In the
future, we plan to investigate jointly training a pre-trained
ASR model, and language model, with the video model to
improve our performance.


**F. Captions in the Spoken Moments Dataset**


Table 14 shows some captions in the Spoken Moments
dataset that capture motion and sequential events which
would be difficult to represent with a single image.



13


|Dataset|Pretrained TSM<br>Dataset|Caption to Video<br>R@1 R@5 R@10 mAP|Video to Caption<br>R@1 R@5 R@10 mAP|Mean<br>R@1 R@5 R@10 mAP|
|---|---|---|---|---|
|**Vatex** [69]|**Kinetics**<br>**S-MiT**|39.6_±_1.0<br>77.5_±_1.5<br>87.2_±_1.0<br>55.9_±_0.8<br>**47.4**_±_1.1<br>**81.5**_±_0.7<br>**89.0**_±_1.1<br>**62.3**_±_0.6|46.4_±_0.6<br>**82.1**_±_1.0<br>**90.2**_±_1.2<br>**61.9**_±_0.6<br>43.1_±_0.9<br>78.3_±_0.6<br>86.2_±_0.3<br>58.5_±_0.5|43.0_±_0.7<br>79.8_±_1.1<br>**88.7**_±_1.0<br>58.9_±_0.7<br>**45.3**_±_0.8<br>**79.9**_±_0.4<br>87.6_±_0.6<br>**60.4**_±_0.5|
|**ActivityNet** [33]|**Kinetics**<br>**M-MiT**|**18.7**_±_1.0<br>**45.6**_±_0.9<br>57.2_±_1.4<br>**31.0**_±_0.7<br>16.1_±_1.7<br>44.0_±_1.0<br>**57.5**_±_1.7<br>29.3_±_1.0|**20.8**_±_0.8<br>**50.1**_±_1.4<br>**61.8**_±_1.3<br>**34.1**_±_0.4<br>19.0_±_1.3<br>48.2_±_0.9<br>61.0_±_1.1<br>32.5_±_0.8|**19.8**_±_0.8<br>**47.8**_±_0.9<br>**59.5**_±_1.0<br>**32.5**_±_0.4<br>17.6_±_1.5<br>46.1_±_0.7<br>59.2_±_1.4<br>30.9_±_0.8|
|**MSR-VTT** [71]|**Kinetics**<br>**M-MiT**|17.6_±_1.3<br>48.9_±_1.8<br>65.6_±_1.2<br>**31.6**_±_1.3<br>**20.7**_±_0.5<br>**54.2**_±_0.9<br>**70.6**_±_1.0<br>30.5_±_0.4|25.5_±_0.7<br>59.7_±_1.8<br>74.1_±_1.6<br>40.6_±_0.9<br>**31.3**_±_1.1<br>**61.0**_±_1.0<br>**75.0**_±_0.9<br>**40.9**_±_0.8|21.6_±_0.8<br>54.3_±_1.4<br>69.8_±_1.4<br>36.1_±_0.9<br>**24.0**_±_0.6<br>**57.6**_±_0.6<br>**72.8**_±_0.8<br>**37.7**_±_0.4|
|**S-MiT**|**Kinetics**<br>**M-MiT**|27.6_±_1.4<br>57.5_±_2.4<br>70.4_±_1.9<br>41.3_±_1.7<br>**29.8**_±_2.5<br>**60.6**_±_2.4<br>**72.2**_±_1.9<br>**44.0**_±_2.2|37.2_±_2.3<br>65.0_±_1.7<br>75.2_±_1.5<br>50.0_±_1.7<br>**39.4**_±_2.1<br>**68.0**_±_2.0<br>**77.5**_±_1.8<br>**52.3**_±_2.0|32.4_±_1.8<br>61.3_±_2.0<br>72.8_±_1.6<br>45.7_±_1.7<br>**34.6**_±_2.1<br>**64.3**_±_2.2<br>**74.9**_±_1.8<br>**48.2**_±_2.0|


Table 5: **Comparison of different Pretrained TSM models on multiple datasets using AMM and Bart**

|Visual Base Model|Caption to Video<br>R@1 R@5 R@10 mAP|Video to Caption<br>R@1 R@5 R@10 mAP|Mean<br>R@1 R@5 R@10 mAP|
|---|---|---|---|
|**TSM Kinetics**<br>**TSM M-MiT**<br>**ResNet-152 ImageNet (2D)**<br>**TSM Kinetics + 2D**<br>**TSM M-MiT + 2D**|20.2_±_1.1<br>47.9_±_2.3<br>61.0_±_0.8<br>33.2_±_1.1<br>19.7_±_1.1<br>48.6_±_2.0<br>61.9_±_1.6<br>33.5_±_1.3<br>24.2_±_2.4<br>53.6_±_1.8<br>66.5_±_2.1<br>37.9_±_2.0<br>27.6_±_1.4<br>57.5_±_2.4<br>70.4_±_1.9<br>41.3_±_1.7<br>**29.8**_±_2.5<br>**60.6**_±_2.4<br>**72.2**_±_1.9<br>**44.0**_±_2.2|28.2_±_1.5<br>54.9_±_1.5<br>67.1_±_1.6<br>40.8_±_1.6<br>28.4_±_1.4<br>58.0_±_2.5<br>69.2_±_1.9<br>41.9_±_1.4<br>32.9_±_2.1<br>61.7_±_1.6<br>71.6_±_1.0<br>45.9_±_1.8<br>37.2_±_2.3<br>65.0_±_1.7<br>75.2_±_1.5<br>50.0_±_1.7<br>**39.4**_±_2.1<br>**68.0**_±_2.0<br>**77.5**_±_1.8<br>**52.3**_±_2.0|24.2_±_1.1<br>51.4_±_1.9<br>64.0_±_1.0<br>37.0_±_1.3<br>24.1_±_1.2<br>53.3_±_2.1<br>65.6_±_1.7<br>37.7_±_1.4<br>28.5_±_2.2<br>57.7_±_1.7<br>69.1_±_1.5<br>41.9_±_1.9<br>32.4_±_1.8<br>61.3_±_2.0<br>72.8_±_1.6<br>45.7_±_1.7<br>**34.6**_±_2.1<br>**64.3**_±_2.2<br>**74.9**_±_1.8<br>**48.2**_±_2.0|



Table 6: **Comparison of different visual base model combinations on S-MiT using AMM and Bart**

|Batch Size|Caption to Video<br>R@1 R@5 R@10 mAP|Video to Caption<br>R@1 R@5 R@10 mAP|Mean<br>R@1 R@5 R@10 mAP|
|---|---|---|---|
|**512**<br>**1024**<br>**2048**<br>**4096**|27.2_±_1.6<br>57.4_±_1.3<br>69.4_±_1.0<br>41.0_±_1.5<br>27.8_±_2.0<br>57.7_±_1.4<br>69.8_±_1.2<br>41.5_±_1.9<br>**29.8**_±_2.5<br>**60.6**_±_2.4<br>**72.2**_±_1.9<br>**44.0**_±_2.2<br>29.2_±_2.7<br>58.4_±_1.6<br>70.8_±_1.9<br>42.8_±_2.3|35.5_±_2.5<br>64.0_±_1.4<br>74.4_±_1.1<br>48.4_±_2.1<br>36.5_±_2.8<br>65.6_±_1.4<br>75.2_±_1.7<br>49.7_±_2.0<br>**39.4**_±_2.1<br>**68.0**_±_2.0<br>**77.5**_±_1.8<br>**52.3**_±_2.0<br>**39.4**_±_2.3<br>66.6_±_1.8<br>75.7_±_1.4<br>51.8_±_1.9|31.3_±_1.9<br>60.7_±_1.3<br>71.9_±_1.0<br>44.7_±_1.7<br>32.2_±_2.3<br>61.7_±_1.4<br>72.5_±_1.3<br>45.6_±_1.9<br>**34.6**_±_2.1<br>**64.3**_±_2.2<br>**74.9**_±_1.8<br>**48.2**_±_2.0<br>34.3_±_2.3<br>62.5_±_1.6<br>73.3_±_1.6<br>47.3_±_2.0|



Table 7: **Comparison of different batch sizes on S-MiT using AMM and Bart**

|Projection Size|Caption to Video<br>R@1 R@5 R@10 mAP|Video to Caption<br>R@1 R@5 R@10 mAP|Mean<br>R@1 R@5 R@10 mAP|
|---|---|---|---|
|**1024**<br>**2048**<br>**4096**<br>**8192**|27.4_±_1.8<br>56.6_±_1.6<br>69.5_±_0.9<br>41.1_±_1.5<br>27.8_±_1.8<br>57.4_±_2.0<br>69.2_±_1.5<br>41.5_±_1.8<br>**29.8**_±_2.5<br>**60.6**_±_2.4<br>**72.2**_±_1.9<br>**44.0**_±_2.2<br>29.4_±_2.0<br>58.0_±_2.3<br>70.3_±_1.2<br>42.6_±_1.8|38.6_±_1.6<br>66.6_±_1.1<br>76.3_±_1.3<br>51.3_±_1.2<br>38.4_±_2.1<br>65.9_±_1.4<br>75.6_±_1.5<br>51.1_±_1.6<br>**39.4**_±_2.1<br>**68.0**_±_2.0<br>**77.5**_±_1.8<br>**52.3**_±_2.0<br>38.5_±_2.4<br>66.1_±_2.1<br>76.1_±_1.5<br>51.2_±_2.1|33.0_±_1.6<br>61.6_±_1.3<br>72.9_±_1.0<br>46.2_±_1.3<br>33.1_±_1.9<br>61.6_±_1.6<br>72.4_±_1.4<br>46.3_±_1.7<br>**34.6**_±_2.1<br>**64.3**_±_2.2<br>**74.9**_±_1.8<br>**48.2**_±_2.0<br>33.9_±_2.2<br>62.0_±_2.2<br>73.2_±_1.3<br>46.9_±_1.9|



Table 8: **Comparison of different projection sizes on S-MiT using AMM and Bart**

|Sampling|Caption to Video<br>R@1 R@5 R@10 mAP|Video to Caption<br>R@1 R@5 R@10 mAP|Mean<br>R@1 R@5 R@10 mAP|
|---|---|---|---|
|**N**<br>**Y**|28.1_±_1.1<br>57.5_±_2.0<br>69.8_±_1.4<br>41.8_±_1.3<br>**29.8**_±_2.5<br>**60.6**_±_2.4<br>**72.2**_±_1.9<br>**44.0**_±_2.2|39.1_±_1.3<br>66.5_±_2.0<br>76.3_±_1.8<br>51.5_±_1.4<br>**39.4**_±_2.1<br>**68.0**_±_2.0<br>**77.5**_±_1.8<br>**52.3**_±_2.0|33.6_±_1.1<br>62.0_±_1.9<br>73.0_±_1.5<br>46.7_±_1.3<br>**34.6**_±_2.1<br>**64.3**_±_2.2<br>**74.9**_±_1.8<br>**48.2**_±_2.0|



Table 9: **Comparison of sampling approach on S-MiT using AMM and Bart**

|Col1|Table 10: Comparison of different dampening multipliers, α, in AMM on S-MiT using Bart|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|**Trained On**|**Evaluated On**|**Evaluated On**|**Evaluated On**|**Evaluated On**|**Evaluated On**|
|**Trained On**|**Vatex**<br>R@1<br>R@5<br>R@10<br>mAP|**ActivityNet**<br>R@1<br>R@5<br>R@10<br>mAP|**MSR-VTT**<br>R@1<br>R@5<br>R@10<br>mAP|**S-MiT**<br>R@1<br>R@5<br>R@10<br>mAP|**Mean**<br>R@1<br>R@5<br>R@10<br>mAP|
|**Vatex**<br>**ActivityNet**<br>**MSR-VTT**<br>**S-MiT**|**19.8**<br>**48.4**<br>**63.7**<br>**33.4**<br>12.1<br>33.3<br>46.8<br>23.0<br>6.5<br>19.2<br>28.8<br>13.8<br>19.4<br>44.6<br>57.7<br>31.7|1.5<br>5.2<br>8.6<br>4.2<br>2.0<br>7.3<br>12.0<br>5.6<br>1.3<br>4.6<br>7.8<br>3.7<br>**2.7**<br>**8.4**<br>**13.6**<br>**6.5**|10.3<br>28.7<br>39.3<br>19.8<br>7.5<br>22.1<br>31.2<br>15.4<br>11.8<br>33.9<br>48.2<br>23.2<br>**17.3**<br>**39.8**<br>**51.8**<br>**28.4**|7.1<br>20.2<br>28.6<br>14.4<br>4.9<br>15.6<br>24.1<br>11.4<br>8.0<br>23.6<br>34.3<br>16.4<br>**25.8**<br>**52.8**<br>**64.7**<br>**38.5**|9.7<br>25.6<br>35.1<br>18.0<br>6.6<br>19.6<br>28.5<br>13.9<br>6.9<br>20.3<br>29.8<br>14.3<br>**16.3**<br>**36.4**<br>**47.0**<br>**26.3**|


|α|Caption to Video<br>R@1 R@5 R@10 mAP|Video to Caption<br>R@1 R@5 R@10 mAP|Mean<br>R@1 R@5 R@10 mAP|
|---|---|---|---|
|**0.1**<br>**0.2**<br>**0.3**<br>**0.4**<br>**0.5**<br>**0.6**<br>**0.7**<br>**0.8**<br>**0.9**|29.3_±_1.4<br>60.0_±_1.2<br>**72.7**_±_1.4<br>43.4_±_1.2<br>28.4_±_1.2<br>58.1_±_1.9<br>70.9_±_1.5<br>42.3_±_1.4<br>27.1_±_2.5<br>58.9_±_2.9<br>71.5_±_2.2<br>41.6_±_2.3<br>28.1_±_1.1<br>58.1_±_2.1<br>69.8_±_2.3<br>41.9_±_1.5<br>**29.8**_±_2.5<br>**60.6**_±_2.4<br>72.2_±_1.9<br>**44.0**_±_2.2<br>28.1_±_2.1<br>59.1_±_2.3<br>71.3_±_2.1<br>42.3_±_1.9<br>28.9_±_1.5<br>59.2_±_1.3<br>70.8_±_1.3<br>42.8_±_1.4<br>29.0_±_1.9<br>59.2_±_2.4<br>70.7_±_1.4<br>42.8_±_1.9<br>27.7_±_2.1<br>57.0_±_2.5<br>68.2_±_2.0<br>41.1_±_2.2|39.2_±_1.8<br>66.2_±_1.5<br>77.0_±_1.3<br>51.7_±_1.6<br>39.3_±_1.6<br>67.4_±_1.5<br>77.0_±_1.8<br>52.0_±_1.4<br>38.5_±_2.4<br>67.1_±_1.0<br>76.6_±_1.9<br>51.5_±_1.9<br>38.8_±_2.4<br>66.9_±_1.2<br>75.8_±_1.5<br>51.5_±_1.8<br>**39.4**_±_2.1<br>**68.0**_±_2.0<br>**77.5**_±_1.8<br>**52.3**_±_2.0<br>38.3_±_1.9<br>67.1_±_1.6<br>76.6_±_1.7<br>51.4_±_1.6<br>38.9_±_1.7<br>66.3_±_1.4<br>76.0_±_1.5<br>51.3_±_1.5<br>38.3_±_2.1<br>66.3_±_1.6<br>75.9_±_1.5<br>51.1_±_1.7<br>37.5_±_2.6<br>64.6_±_2.4<br>74.1_±_1.5<br>49.8_±_2.2|34.2_±_1.5<br>63.1_±_1.3<br>74.8_±_1.1<br>47.5_±_1.4<br>33.9_±_1.4<br>62.7_±_1.7<br>74.0_±_1.5<br>47.2_±_1.4<br>32.8_±_2.3<br>63.0_±_1.9<br>74.0_±_1.9<br>46.5_±_2.1<br>33.5_±_1.8<br>62.5_±_1.6<br>72.8_±_1.7<br>46.7_±_1.6<br>**34.6**_±_2.1<br>**64.3**_±_2.2<br>**74.9**_±_1.8<br>**48.2**_±_2.0<br>33.2_±_1.9<br>63.1_±_1.8<br>73.9_±_1.8<br>46.9_±_1.7<br>33.9_±_1.6<br>62.7_±_1.4<br>73.4_±_1.3<br>47.1_±_1.4<br>33.6_±_1.9<br>62.8_±_1.9<br>73.3_±_1.3<br>46.9_±_1.7<br>32.6_±_2.4<br>60.8_±_2.4<br>71.2_±_1.7<br>45.5_±_2.2|



Table 11: **Cross Dataset Evaluation on Video/Caption Retrieval on Full Test Set**


14


|Col1|Retrieval Results<br>Query<br>R@1 R@2 R@3 R@4 R@5|
|---|---|
|(a)||
|(b)||
|(c)||
|(d)||
|(e)||
|(f)||
|(g)||
|(h)||
|(i)||


Table 12: **Spoken Moments Examples of Caption to Video Retrieval Results:** Given a query caption, we show five top retrieved captions
where words transcribed from the audio captions using a pretrained ASR model are used as a caption. We use a BART model trained with
the AMM loss function on the S-MiT dataset. Blue indicates the ground-truth results.


15


|Col1|Retrieval Results<br>Query<br>R@1 R@2 R@3 R@4 R@5|
|---|---|
|(a)||
|(b)||
|(c)||
|(d)||
|(e)||
|(f)||
|(g)||
|(h)||
|(i)||


Table 13: **Spoken Moments Examples of Video to Caption Retrieval Results:** Given a query video, we show five top retrieval captions
where words transcribed from the audio captions using a pretrained ASR model are used as a caption. We use a BART model trained with
the AMM loss function on the S-MiT dataset. Blue indicates the ground-truth results.


16


|Col1|Frames<br>Caption<br>−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−→time|
|---|---|
|(a)|a boy and a red white and blue shirt<br>is sitting on a couch he is** holding**<br>an infant life vest and** picks it up**<br>**to blow** through the two|
|(b)|there’s a gauge or a lock thing turns<br>from rides and then** being turned**<br>**to the left**|
|(c)|a picture of a man** drinking coffee**<br>and** play with a cell phone** in** fast**<br>**motion**|
|(d)|in** slow motion** we see a collie<br>**jump into the air** and** catch** a<br>white frisbee in ﬂight|
|(e)|these are track and ﬁeld runners<br>and it’s a relay race and** they take**<br>**off when they are handed the**<br>**batons**|
|(f)|there is water** dripping off** the<br>edge of something all you can hear<br>is the water dripping|



Table 14: **Spoken Moments Captions:** We show some examples of captions, and associated video frames, from the Spoken Moments
dataset, where the captions describe a sequence of actions or motion.


17


