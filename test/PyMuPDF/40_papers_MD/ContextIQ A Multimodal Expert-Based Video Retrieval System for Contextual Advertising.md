## **ContextIQ: A Multimodal Expert-Based Video Retrieval System for Contextual** **Advertising**

Ashutosh Chaubey [2] _[∗†]_, Anoubhav Agarwaal [1] _[∗]_, Sartaki Sinha Roy [1] _[∗]_, Aayush Agrawal [1] _[∗]_, Susmita Ghose [1] _[∗]_

1 Anoki Inc. 2 University of Southern California


achaubey@usc.edu _{_ anoubhav,sartaki,aayush,susmita _}_ @anoki.tv



**Abstract**


_Contextual advertising serves ads that are aligned to the_
_content that the user is viewing. The rapid growth of video_
_content on social platforms and streaming services, along_
_with privacy concerns, has increased the need for contex-_
_tual advertising. Placing the right ad in the right context_
_creates a seamless and pleasant ad viewing experience, re-_
_sulting in higher audience engagement and, ultimately, bet-_
_ter ad monetization. From a technology standpoint, effec-_
_tive contextual advertising requires a video retrieval sys-_
_tem capable of understanding complex video content at a_
_very granular level. Current text-to-video retrieval models_
_based on joint multimodal training demand large datasets_
_and computational resources, limiting their practicality and_
_lacking the key functionalities required for ad ecosystem in-_
_tegration. We introduce_ _**ContextIQ**_ _, a multimodal expert-_
_based video retrieval system designed specifically for con-_
_textual advertising. ContextIQ utilizes modality-specific ex-_
_perts—video, audio, transcript (captions), and metadata_
_such as objects, actions, emotion, etc.—to create seman-_
_tically rich video representations. We show that our system,_
_without joint training, achieves better or comparable results_
_to state-of-the-art models and commercial solutions on mul-_
_tiple text-to-video retrieval benchmarks. Our ablation stud-_
_ies highlight the benefits of leveraging multiple modalities_
_for enhanced video retrieval accuracy instead of using a_
_vision-language model alone. Furthermore, we show how_
_video retrieval systems such as ContextIQ can be used for_
_contextual advertising in an ad ecosystem while also ad-_
_dressing concerns related to brand safety and filtering in-_
_appropriate content._


**1. Introduction**


Contextual advertising involves placing ads based on
the content a consumer is viewing, improving user experience and leading to higher engagement and ad monetization [74]. This method mitigates the need for personal data


_†_ Work done while employed at Anoki Inc.
  - These authors contributed equally to this work



in advertising, which raises notable legal and ethical concerns [20, 59]. Contextual advertising is already widely integrated into ads served on websites powered by platforms
like Google AdSense [27]. Recent AI advancements have
taken this a step further by enabling a deeper semantic understanding of multimedia content [34], allowing for more
precise ad placements beyond traditional targeting [41].
Building on these developments, this study expands the use
of contextual advertising to video content, aiming to improve ad targeting on social platforms such as YouTube and
streaming services like Free Ad-Supported Streaming Television (FAST) and Video on Demand(VoD).
Modern video streaming platforms, with their vast content libraries, require advanced methods for analyzing video
content to retrieve the most suitable content for ad placements for contextual advertising. We propose that this task
of identifying the most relevant video can be achieved with
text prompts that align with an advertisement campaign and
hence can be formulated as the text-to-video (T2V) retrieval
problem in multimodal learning [45, 46, 51, 78]. Therefore,
improvements in T2V retrieval models can enhance contextual advertising by better-aligning ads with the semantic
context of video content.

With the exponential growth of video content, T2V
retrieval has become more essential, driving progress in
search algorithms [11,21,48] and video representation techniques such as masked reconstruction [31, 62, 66], multimodal contrastive learning [18, 72], and next-token prediction from video data [43, 60].

A recent trend in T2V retrieval is large-scale pretraining to learn a joint multimodal representation of a video

[17, 18, 69]. While these joint models have demonstrated
strong performance on public benchmarks, their reliance on
massive multimodal datasets and substantial computational
resources limits their practical use. In contrast, there are
expert-based approaches that use specialized models to extract features from individual video modalities such as vi
suals, motion, speech, audio, OCR, etc. [45, 50]. Though
this method is well-established in T2V retrieval, its prominence has declined with the rise of joint multimodal models.


However, expert models present unique benefits for contextual advertising, which we explore in this work.
This paper introduces ContextIQ, a multimodal expertbased video retrieval system for contextual advertising. ContextIQ leverages experts across multiple modalities—video, audio, captions (transcripts), and metadata
such as objects, actions, emotions, etc.—to create semantically rich video representations (ref. Sec. 3 & 4). Its
modular design makes it highly flexible to various brand
targeting needs. For instance, for a beauty brand, we could
integrate advanced object detection to spot niche beauty accessories. Moreover, the proposed system can selectively
use only a relevant expert model, focusing on the key metadata to enable efficient targeting, making it well-suited for a
fast ad-serving system. Furthermore, our system enhances
interpretability by allowing individual experts to be analyzed for error tracing and model improvements, which is
vital for explaining business outcomes in ad targeting. Beyond conventional video retrieval, we show how ContextIQ
can be integrated seamlessly into the ad ecosystem, processing long-form streaming content and implementing brand
safety filters to ensure ads are placed within contextually
appropriate and safe content (ref. Sec. 6).
We evaluate the effectiveness of our ContextIQ system
across several video retrieval benchmarks, such as MSRVTT [73], Condensed Movies [9], and a newly curated
dataset ( _Val-1_ ) of high-production content that we make
public. We compare it to state-of-the-art joint multimodal
models [54, 68, 77] and commercial solutions like Google’s
multimodal embedding API [28] and Twelvelabs Marengo

[63]. The results show that our expert-based approach is
better or comparable to these solutions (ref. Sec. 5.1).
We also address key evaluation challenges, particularly
the differences between video domains in existing public
datasets—such as shorter, amateur-produced content—and
the complex, high-production content typical of contextual
advertising by validating the proposed technique on the curated dataset ( _Val-1_ ) of movie contents and evaluating different approaches by manual validators. To further concretize our design choice of having multiple experts, we
perform ablation studies on the impact of incorporating
multiple modalities on an internal dataset ( _Val-2_ ), demonstrating how our system effectively aggregates modalityspecific experts. By leveraging the complementarity of
different modalities, ContextIQ achieves improvements in
both performance and coverage (ref Sec. 5.2). We release
our datasets, retrieval queries and annotations publicly at
[https://github.com/AnokiAI/ContextIQ-Paper. In summary,](https://github.com/AnokiAI/ContextIQ-Paper)
our contributions are as follows,


(i) We present ContextIQ, a multimodal expert-based
video retrieval system specifically designed for contextual advertising. With capabilities for long-form
content processing, modularity for real-time ad serv


ing, and robust brand safety measures, it integrates
seamlessly into the advertising ecosystem, effectively
bridging the gap between video retrieval research and
contextual video advertising.


(ii) On existing benchmarks, we show that combining
modality-specific experts can yield better or comparable results with state-of-the-art joint multimodal
models without the need for joint training, large-scale
multimodal datasets, or significant computational re
sources.


(iii) We release a validation dataset ( _Val-1_ ) of highproduction content for text-to-video retrieval for contextual advertising and show the superior performance of the proposed approach on that dataset.


(iv) We perform ablation studies to show that additional
modalities via expert-based models increase the coverage and accuracy of text-to-video retrieval for contextual advertising by showing results on an internal
dataset of video contents.


**2. Related Works**


**AI in advertising.** Liu _et al_ . [44] employed text-based topic
modeling to tag content, enabling buyers to make more informed ad bids. Wen _et al_ . [61] utilized decision trees to
identify key attributes that enhance ad persuasiveness, facilitating the selection of suitable ads and insertion points
within videos. Bushi _et al_ . [12] analyzed sentiment to prevent negative ad placements, while Vedula _et al_ . [65] predicted ad effectiveness by training separate neural network
models on video, audio, and text modalities.
**Text-to-Video Retrieval.** Earlier approaches relied on
convolutional neural networks (CNNs) and recurrent neural networks (RNNs) [30, 51] to extract spatio-temporal
features from video frames. However, the advent of
Transformer-based architectures [24, 26, 57] has led to a
paradigm shift, with Transformers outperforming traditional CNNs and RNNs in many public benchmark datasets.
**Multimodal Pre-training and Joint Learning.** The impact of large-scale pre-training, especially using models like
CLIP [54] that align images with text, has been groundbreaking in the video retrieval domain [23, 38, 46, 47,
75]. For instance, CLIP4Clip [46] builds upon CLIP to
enable end-to-end video-text retrieval via video-sentence

contrastive learning. Similarly, CLAP [71] aligns audio
with text using contrastive learning. Incorporating additional modalities contrastively can significantly enhance the
model’s performance and robustness [18, 69]. LanguageBind contrastively binds language across multiple modalities within a common embedding space [77]. VALOR utilized separate encoders for video, audio, and text to develop a joint visual-audio-text representation [17]. We show


|Col1|Col2|Encoder Image<br>Frame-level<br>Aggregation<br>V|
|---|---|---|
||||
















|Col1|Object Detector|
|---|---|
||Place Detector|
|Metadata<br>Extraction<br>Module<br>Plac<br>Video A<br>Emotio<br>|Plac|



Metadata Extraction Module

Figure 1. Multimodal embedding generation pipeline for ContextIQ. Input videos are processed by the metadata extracting module, which
uses expert models for extracting objects, actions, places, etc., and converts it into a metadata sentence ( _m_ _i_ ). Four modality encoders then
encode the video frames, audio, caption (transcripts) and metadata in parallel and store the embeddings in a multimodal embeddings DB.



that compared to these jointly trained approaches, ContextIQ performs on par on video-retrieval benchmarks such as
MSR-VTT [73] without any joint training.
**Multimodal Experts.** There have been several works in
video retrieval that employ multiple experts to extract features from different video modalities, including scene visuals, motion, speech, audio, OCR, and facial features

[19, 45, 50]. This approach introduces the challenge of effectively aggregating these expert-derived features. For example, Miech _et al_ . [50] utilizes precomputed features from
experts in text-to-video retrieval, where the overall similarity is calculated as a weighted sum of each expert’s similarity score. Liu _et al_ . [45] extends the mixture of experts model by employing a collaborative gating mechanism, which modulates each expert feature in relation to the
others. [25] integrates a multimodal transformer to encode
visual, audio, and speech features from different experts,
with BERT handling the text query. In contrast to these
techniques, we show how our expert-based retrieval system
is specifically designed for contextual advertising by providing modularity for targeting flexibility and interpretability, brand safety filters, and ad ecosystem integration.


**3. Approach**


For a video dataset _V_ = _{v_ 1 _, ..., v_ _N_ _}_, our goal is to
develop a text-to-video retrieval system. Conceptually, we
aim to learn a similarity function _S_ ( _v_ _i_ _, t_ ) that assigns higher
scores to videos that are more relevant to the given text
query _t_ .
We extract embeddings from multiple modalities, including raw video, audio, captions (speech transcripts), and
combined visual (objects, places, actions, emotion) and textual (NER, emotion, profanity, hate speech) metadata, as
shown in the Metadata Extraction Module in Fig. 1. Since
the expert encoders for each modality are not jointly trained,



their embeddings reside in separate semantic spaces. These
embeddings are then stored in a multimodal embeddings
database. We propose a retrieval approach (Fig. 2) that
compares, merges, and re-ranks these modality-specific embeddings to deliver the most contextually relevant videos.


**3.1. Multimodal Embedding Generation**


**3.1.1** **Video**


As illustrated in Fig. 1, we consider each video _v_ _i_ to be
composed of _M_ frames _v_ _i_ _[j]_ [, which are encoded using an]
image-encoder _f_ _θ_ 1 from a vision-text model [42]. This
encoding captures intrinsic video features learned by the
vision-text model.

We found that splitting the video into fixed nonoverlapping time segments, each containing T frames, enhances both retrieval accuracy and localization. For each
temporal segment _C_ _i_ _[k]_ [=] _[ {][v]_ _i_ _[z]_ _[|][ z][ ∈{][kT]_ [+1] _[, . . .,]_ [ (] _[k]_ [+1)] _[T]_ _[}}]_ [,]
segment-level embeddings are obtained by applying an aggregation function _A_ _v_ to the frame-level features,


**e** _[k]_ _i_ [=] _[ A]_ _[v]_ [(] _[{][f]_ _[θ]_ 1 [(] _[v]_ _i_ _[z]_ [)] _[ |][ z][ ∈{][kT]_ [ + 1] _[, . . .,]_ [ (] _[k]_ [ + 1)] _[T]_ _[}}]_ [)][ (1)]


The final set of video-level embeddings for the video _v_ _i_
that we store is,

_F_ _i_ _[v]_ [=] _[ {]_ **[e]** [1] _i_ _[, . . .,]_ **[ e]** [(] _i_ _[M/T]_ [ )] _}_ (2)


Note that for a single video, we store ( _M/T_ ) embeddings in the multimodal database corresponding to intrinsic
video features. This ensures that for long-form video content, the features have high temporal resolution and do not
average out, resulting in better retrieval.

**3.1.2** **Audio**


To capture the non-verbal audio elements in a video, such
as sound effects, music, and ambient noise, we utilize the


Figure 2. Multimodal search pipeline for ContextIQ. Text query _t_ is first encoded by different text encoders and the multimodal embedding
DB is searched to find similar videos. The aggregation module combines the results obtained from different modalities, and the final results
are obtained after applying brand-safety filters utilizing emotion, profanity, and hate speech.



audio encoder _g_ _θ_ 2 from an audio-text model [71]. While
speech and textual content are encoded through captions
(ref. Sec. 3.1.3), this approach specifically targets the rich
auditory features beyond spoken language.
The audio track _a_ _i_ for a particular video _v_ _i_ is divided
temporally into equal-sized segments _a_ _[k]_ _i_ [, each of which is]
encoded using _g_ _θ_ 2 . The final audio embedding _F_ _i_ _[a]_ [for the]
video is obtained by applying an aggregation function _A_ _a_
to the features across all the segments:

_F_ _i_ _[a]_ [=] _[ A]_ _[a]_ � _g_ _θ_ 2 ( _a_ [1] _i_ [)] _[, g]_ _[θ]_ 2 [(] _[a]_ [2] _i_ [)] _[, . . ., g]_ _[θ]_ 2 [(] _[a]_ _[K]_ _i_ [)] � (3)


**3.1.3** **Caption**


To incorporate textual and speech information from a video,
we encode the caption (transcript) _c_ _i_ for a given video using
a text encoder _h_ _θ_ 3, resulting in the caption feature embedding of the video:
_F_ _i_ _[c]_ [=] _[ h]_ _[θ]_ 3 [(] _[c]_ _[i]_ [)] (4)


**3.1.4** **Metadata**


Foundation vision models [42,54] excel at learning features
for vision-specific tasks with minimal fine-tuning. To enhance the performance of our text-to-video retrieval system,
we incorporate models specifically trained for vision and
textual tasks, effectively augmenting foundational models
(ref. Sec. 5.2).
After inference with the task-specific models, we store
the extracted information into a metadata sentence _m_ _i_ . This
sentence captures objects, places, actions, emotions, named
entities and flags any profanity, hate speech, or offensive
language. This metadata is vital for contextual advertising. For example, object detection can allow fashion brands
to target content featuring clothing or accessories, while
place detection can benefit outdoor gear brands by identifying natural landscapes. Video action recognition can enable fitness brands to reach workout-related content, and
named entity recognition might help travel agencies focus
on videos mentioning tourist destinations. _m_ _i_ provides a



unique and flexible method for integrating outputs from the
different expert models. _m_ _i_ follows the template, as illustrated in Fig. 1.
Finally, we encode _m_ _i_ by the same text encoder model
_h_ _θ_ 3 from Sec. 3.1.3, to obtain the metadata feature embedding for the video:


_F_ _i_ _[m]_ = _h_ _θ_ 3 ( _m_ _i_ ) (5)


Additionally, the emotion, profanity, and hate speech
metadata are used as a filtering mechanism during the
search process to ensure the retrieval of brand-safe videos.
This capability extends beyond the conventional scope of
video-text retrieval research, with its application further
elaborated in Sec. 6 on contextual advertising.


**3.2. Multimodal Search**


Our system employs multimodal embeddings from audio, video, text and metadata to enable efficient and precise
content retrieval. As shown in the Fig. 2, a text query _t_
is first encoded using the text encoders of the vision-text
model ( _f_ _θ_ _[T]_ 1 [), audio-text model (] _[g]_ _θ_ _[T]_ 2 [) and text-text model]
( _h_ _θ_ 3 ) resulting in the embeddings _F_ _t_ _[v]_ [,] _[ F]_ _t_ _[a]_ [and] _[ F]_ _t_ _[T]_ [respec-]
tively.
The text embeddings are then compared against their
corresponding embedding databases using cosine similarity. Specifically, _F_ _t_ _[a]_ [is compared with audio embeddings]
_{F_ _i_ _[a]_ _[}]_ [, producing similarity scores] _[ {S]_ _[a]_ [(] _[v]_ _[i]_ _[, t]_ [)] _[}]_ [;] _[ F]_ _t_ _[v]_ [is com-]
pared with visual embeddings _{F_ _i_ _[v]_ _[}]_ [, yielding similarity]
scores _{S_ _[v]_ ( _v_ _i_ _, t_ ) _}_ ; and _F_ _t_ _[T]_ [is compared with metadata em-]
beddings _{F_ _i_ _[m]_ _[}]_ [ and caption embeddings] _[ {F]_ _i_ _[c]_ _[}]_ [, generating]
similarity scores _{S_ _[m]_ ( _v_ _i_ _, t_ ) _}_ and _{S_ _[c]_ ( _v_ _i_ _, t_ ) _}_, respectively.
Note that _{S_ _[v]_ ( _v_ _i_ _, t_ ) _}_ is the max-similarity score of _t_ with
the segments ( _{C_ _i_ [1] _[, . . ., C]_ _i_ [(] _[M/T]_ [ )] ) of video _v_ _i_ .
These similarity scores _{_ ( _S_ _[c]_ _, S_ _[m]_ _, S_ _[a]_ _, S_ _[v]_ )( _v_ _i_ _, t_ ) _}_ are
subsequently aggregated into a combined score _S_ ( _v_ _i_ _, t_ ) using an aggregation module _A_ _S_ (ref. Fig. 2) which involves
normalization, thresholding, and weighted merging,


- **Normalization.** Scores are standardized within their re
spective modality space and weighted to produce normalized score:
_N_ _[k]_ ( _v_ _i_ _, t_ ) = _λ_ _[k]_ _·_ _[S]_ _[k]_ [(] _[v]_ _[i]_ _[,][ t]_ [)] _[ −]_ _[µ]_ _[k]_ (6)

_σ_ _[k]_

where _µ_ _[k]_ = _n_ 1 � _ni_ _[S]_ _[k]_ [(] _[v]_ _[i]_ _[, t]_ [)][ is the mean,] _[ σ]_ _[k]_ =
� _n_ 1 ~~�~~ _n_ 1 [(] _[S]_ _[k]_ [(] _[v]_ _[i]_ _[, t]_ [)] _[ −]_ _[µ]_ _[k]_ [)] [2] [ is the standard deviation and]

_λ_ _[k]_ is the weight of modality _k_ (metadata, caption, video
or audio).

- **Thresholding.** Scores are thresholded to obtain
dictionaries for each modality _X_ _[k]_ = _{_ ( _i_ :
_N_ _[k]_ ( _v_ _i_ _, t_ )) _| S_ _[k]_ ( _v_ _i_ _, t_ ) _> α_ _[k]_ _}_ where _α_ _[k]_ is the modality
specific threshold.

- **Merging.** The thresholded dictionaries for each modality
are merged using a max-aggregation approach, with the
final scores _S_ ( _v_ _i_ _, t_ ) determined by the maximum value
for each key.


The thresholds _α_ _[k]_ and weights _λ_ _[k]_ can be tuned for optimal performance on a representative set of queries. The resulting video contents are finally filtered through the brand
safety mechanism (ref. Sec. 6).


**4. Implementation Details**


**4.1. Multimodal encoders**


We use PyTorch [8] for all our model implementations,
and we use four NVIDIA RTX 4090 GPUs to run all our

experiments. The text encoder _h_ _θ_ 3 is a pre-trained MPNet
model [58] fine-tuned on a set of 1 billion text-text pairs as
described by Reimers et al [55]. The vision-text model _f_ _θ_ 1
is a pre-trained BLIP2 Qformer [42], and we use the implementation provided by LAVIS [40]. We split the video
into equal segments of 15 seconds each and sample one out
of ten frames for embedding generation (ref. Sec. 3.1.1).
The audio encoder _g_ _θ_ 2 is the CLAP [71] model as implemented in [2]. Since CLAP is trained on 5-second audio
segments, we divide the audio into 5-second chunks ( _a_ _[k]_ _i_ [)]
for inference. The aggregation functions _A_ _a_ and _A_ _v_ used
during audio and video embedding generation (Sec. 3.1)
are temporal mean pooling functions, shown to be effective
over other temporal aggregation techniques [10, 46].


**4.2. Metadata Extraction**


We use the YOLOv5 model [64], trained on the Objects365 dataset [56], to detect objects within videos with an
IOU threshold of 0.45 and a confidence threshold of 0.35.
An object is considered present in the video only if it appears in at least 20% of the frames, to filter false positives.
For place detection, we finetune a ResNet50 model [32]
on the Places365 dataset [76]. Only frames where object
detection for the _Person_ class covers less than 10% of the

area are used to ensure a clear background. Top predictions



from these frames with softmax scores above 0.3 are con
sidered, and the most frequent place prediction is tagged as
the video’s location. Since video segments are short, we
assume a single location per segment. For both place and
object detection, predictions are sampled from every 10th
frame.

We use a fine-tuned video masked autoencoder model
(VideoMAE2) [67] on the Kinetics 710 dataset [14, 15, 35]
for video action recognition. The Kinetics dataset comprises shorter, simpler YouTube clips featuring single actions, whereas our video retrieval algorithm is applied to
videos with longer, more complex scenes involving multiple actions. To bridge this gap, we reduced the Kinetics
710 classes to 185 by eliminating less relevant or overly
specific classes for advertising, merging correlated classes,
and discarding those with low Kinetics validation accuracy.
We also refined the majority voting method by incorporating prediction probability scores, improving the handling
of multiple actions in a single clip. More implementation
details are present in the supplementary material, and we
share the list of reduced classes on our github repository [https://github.com/AnokiAI/ContextIQ-Paper.](https://github.com/AnokiAI/ContextIQ-Paper)
For named entity recognition, we utilize the text-based
RoBERTa model, en ~~c~~ ore ~~w~~ eb ~~t~~ rf from Spacy [33] to extract the named entities from the captions of each video
scene. For profanity detection, we used the _alt-profanity-_
_check_ [1] python package on the video transcript (caption).
Additionally, we use a predefined list of words given in [3]
to further filter out profane videos.
For hate speech detection, we use a weighted ensemble of two models. First, we use a pre-trained LLAMA 3
8B Instruct model [7] with a temperature of 0.6, leveraging advanced prompting strategies such as JSON-parseable
responses and chain-of-thought reasoning to flag content
as hateful. Secondly, we use a pre-trained BERT classifier [36] trained on the HateXplain dataset [49] to categorize content into three classes: hate speech, offensive, and
normal. For text emotion recognition, first, we use a pretrained Emoberta-Large model [37] from huggingface [6]
which is trained on the MELD [53] and IEMOCAP [13]
datasets. Furthermore, we also leverage the computed video
and audio embeddings from Sec. 3.1 for tagging emotions
by associating text concepts with different emotions. For
example, we tagged the emotion _joy_ with text queries like
_people smiling_ and _people dancing_, and assigned the emotion _joy_ to all videos retrieved through the video (vision)
modality using these queries. More implementation details
of emotion, profanity, and hate speech detection are present
in the supplementary material.


**4.3. Validation datasets and Metrics**


**Validation datasets.** As mentioned before, we show the
efficacy of the proposed approach in comparison to state

**Model** **P@1** **P@5** **R@5** **MAP@5**


Vertex API [28] 81.9 57.0 93.2 83.1
LanguageBind [77] **85.5** **66.6** **97.7** **86.6**
**ContextIQ** **(Ours)** 81.7 59.1 93.7 83.2


Table 1. Performance comparison on MSR-VTT for ContextIQ,
Google Vertex [28] and LanguageBind [77].


**Model** **P@1** **P@5** **R@5** **MAP@5**


TwelveLabs [63] 96.6 **90.3** 100 **95.6**
LanguageBind [77] 89.7 83.5 100 91.5
**ContextIQ** **(Ours)** **96.6** 88.3 **100** 94.4


Table 2. Performance comparison on Condensed Movies [9] for
ContextIQ, TwelveLabs [63] and LanguageBind [77].


of-the-art video retrieval methods on two public datasets,
MSR-VTT [73] and Condensed Movies [9]. We use the
1kA subset of the MSR-VTT test set for evaluation, which
consists of 1k videos and 20 text descriptions for each of
them. During our analysis, we observed duplicate text descriptions both within individual video clips and across different clips. Hence to assess performance, we randomly
sample one caption per video clip.
Since the MSR-VTT dataset includes a variety of video
rather than entertainment-focused content, we also utilize
the Condensed Movies dataset [9], which consists of scene
clips from 3K+ movies. For evaluation, we randomly sample 600 scene clips and extract the first minute of each.
Based on the movie and scene descriptions, we use ChatGPT [52] to generate a set of 29 text queries, focusing on
concepts such as objects, locations, emotion, and other contextual elements to search across the 600 video clips. Because the condensed movies dataset is not tagged with the
set of queries we obtained, we manually validated the results on this dataset. (ref. Sec. 5.1).
Furthermore, to show that our system performs well
for contextual advertisement targeting, we manually collected a set of 500 movie clips of different genres from
YouTube corresponding to different potential advertisement
categories. We call this dataset _Val-1_ . The selected clips
represent one or more of the following ‘concepts’ - _burger_,
_concert_, _cooking_, _cowboys and western_, _dog_, _space shuttle_,
_sports_, and _army_ . Each of the collected movie clips is then
annotated with one or more of these concepts by at least 2
annotators, and we take the union of annotated concepts as
ground truth.
We release all the details about the datasets, including
text queries used for validation and annotations on GitHub

[- https://github.com/AnokiAI/ContextIQ-Paper.](https://github.com/AnokiAI/ContextIQ-Paper)
**Metrics.** For all our experiments, we report one or more
of the following metrics (i) Precision@K ( _P_ @ _K_ ), which
is the proportion of retrieved videos marked as correct out
of the top _K_ retrieved videos, averaged across all text
queries, (ii) Recall@K ( _R_ @ _K_ ), which is the average number of queries for which at least one of the top K retrieved results is marked as correct, and (iii) Mean Average



Correct Incorrect Not Marked


If marked WRONG on Video #1 - Enter alternate queries:


Figure 3. Validation tool built with Streamlit [5]. Note that the
different methods are kept anonymous to remove any bias.


Precision@K( _MAP_ @ _K_ ): The mean of the average precision scores from 1 to _K_ ( _{_ 1 _, . . ., K}_ ), computed across all
queries.


**5. Results**


**5.1. Zero-shot text-to-video retrieval**


We evaluate the performance of ContextIQ for text-tovideo retrieval using the validation datasets mentioned in
Sec. 4.3. Tab. 1 shows the performance of the proposed approach on the MSR-VTT dataset [73] compared to a stateof-the-art jointly trained multimodal model (LanguageBind

[77]) and a popular industry solution for video retrieval
(Google’s Vertex API [28]). For LanguageBind we use
the huggingface implementation - _LanguageBind_ ~~_V_~~ _ideo_ ~~_F_~~ _T_

[4]. Although ContextIQ is not jointly trained on multiple
modalities, it performs slightly better than Google’s Vertex. We hypothesize that LanguageBind’s superior performance on MSR-VTT can be attributed to its joint training
on the large-scale 10M VIDAL dataset, which includes a
diverse range of videos similar to the general content found
in MSR-VTT, rather than being focused on entertainmentspecific content. Additionally, LanguageBind incorporates
modalities such as depth, which are absent in ContextIQ.

As shown in Tab. 2, for the Condensed Movies dataset

[9], we compare the results of our proposed technique with
_TwelveLabs_, utilizing their _Marengo_ API [63], a jointly
trained multimodal model for text-to-video retrieval. For

all the approaches listed in Tab. 2, we first retrieve videos
for each of the text queries generated by ChatGPT (ref. Sec.
4.3) and then ask three manual validators to validate the top
5 results using the validation tool built using _Streamlit_ [5]
as shown in Fig. 3. We use a voting-based system among
the annotators to compile the results of our validation. We
can see that ContextIQ performs better than LanguageBind
on all the metrics, while it performs comparable to TwelveLabs. These findings further show that ContextIQ, a mix


**Accuracy of Video Retrieval**


Annotator name:


Select a query:







Method 1 Method 2


#1; 71806.mp4


Rate Video #1



Method 3


**Model** **Jointly** **Modalities** **P@5** **P@10** **P@15** **P@20** **P@25** **P@30** **P@35** **P@40** **P@45** **P@50**
**trained**


CLIP (Large) [46] ✗ _V, L_ 100 100 99.2 98.8 98.5 98.8 96.8 95 93.3 90.3
LanguageBind [77] ✓ _L, V, A, δ, I_ _r_ 100 98.8 98.3 98.1 98.5 98.8 97.1 95.9 94.7 92
One-Peace [68] ✓ _L, V, A_ 92.5 91.3 92.5 94.4 93.5 93.3 93.2 92.5 90.8 90.3
**ContextIQ** **(Ours)** ✗ L, V, A **100** **100** **100** **99.4** **99** **98.8** **98.2** **98.1** **97.2** **97.3**


Table 3. Performance comparison on the curated dataset ( _Val-1_ ) for ContextIQ, LanguageBind [77], One-Peace [68] and CLIP-Large [54].
_V_ : vision, _L_ : language, _A_ : audio, _δ_ : depth, _I_ _r_ : infrared.



ture of expert-based models can perform comparable to or
even better than the jointly trained multimodal models.
Tab. 3 shows the performance of the proposed approach
as compared to different approaches on our curated dataset
( _Val-1_ as described in Sec. 4.3). We use the large variant of
CLIP [54] for our comparison. Note that One-Peace [68]
and LanguageBind [77] are multimodal models that are
trained jointly, where embeddings from all modalities reside in the same space. ContextIQ performs significantly
better than the baselines, especially when we check the precision at higher k values. It is also important to highlight
that CLIP, which is just a vision-language model, performs
comparably to the multimodal approaches LanguageBind
and One-Peace, highlighting that for most of the queries,
only visual understanding results in good retrieval results.


**5.2. Ablation: Impact of additional experts and**
**modalities**


We saw in the previous paragraph that only a visionlanguage model achieves comparable results on the task of
video retrieval for advertising on our advertising-focused
curated dataset ( _Val-1_ ). In this section, we perform an ablation study to see the efficacy of different modalities when
combined with the vision-text model encoder.

Since we want to simulate the effect of using our system in a large database of multimedia content, we utilize
an internal dataset ( _Val-2_ ) comprising over 2,000 long-form
videos (movies, TV, and OTT contents) processed through
our ContextIQ system (ref. Sec. 6) to generate over 100,000
scenes (videos), averaging 30 seconds in duration. We curate retrieval query sets for each additional modality, focusing on queries that highlight their individual strengths and
are relevant to ad targeting. Details about the dataset are
included in our github repository.
We compute the audio, video, caption and metadata embeddings for the entire dataset and store them separately.
We then compare the performance of vision-only (video)
embeddings with vision combined with different modalities
as shown in Tab. 4. Since we do not have labeled ground
truth for this internal dataset as well, we employ a similar
manual validation technique which was used for validating
Condensed Movies (ref. Tab. 2). For each query, we search
and retrieve the top 30 videos from the dataset. Vision-only
results are based on similarity scores between the text query
and vision embeddings, while vision + additional modality



Where, _K_ is the set of _K_ values _{_ 5 _,_ 10 _, . . .,_ 30 _}_ and
_P_ _V,K_ is the precision for vision-only variant at _K_, and
_P_ _V_ + _X,K_ is the precision for vision+additional modality.
We observe that the average precision delta is highest
for audio, followed by caption and then metadata. This is
due to the fact that the vision modality doesn’t capture raw
audio or captions (transcripts) but is better at representing
metadata elements like objects, actions, places, etc.
Tab. 5 further shows that without our aggregation module, the scenes retrieved by each modality independently for
the same query differ significantly from those retrieved by
the vision-only variant, resulting in very low overlap. This
confirms that each modality captures distinct aspects of the
video. The overlap with vision follows the order: audio _<_
caption _<_ metadata, which aligns with expectations. These
findings show that utilizing the complementarity of various
modalities improves both performance and coverage.



**Query set** **Modalities** **P@5** **P@10** **P@15** **P@20** **P@25** **P@30**

Metadata set _V_ 85.7 84.3 80.5 79.5 77.6 76.2
(∆ avg = 4 _._ 08) _V_ + _M_ **87.9** **86.4** **85.0** **83.6** **83.0** **82.4**
Caption set _V_ 84.2 79.5 75.4 76.6 75.4 74.6
(∆ avg = 5 _._ 42) _V_ + _L_ **84.2** **82.1** **83.5** **83.4** **82.5** **82.5**

Audio set _V_ 85.7 82.9 79.0 83.6 83.4 84.8
(∆ avg = 5 _._ 67) _V_ + _A_ **88.6** **87.1** **87.6** **89.3** **90.3** **90.5**


Table 4. Performance gain on Adding Modalities to Vision-Only
System. _V_ : vision, _L_ : language (captions), _A_ : audio, _M_ : metadata


**Modality** **Overlap % in Top-K with Vision Modality**


@5 @10 @15 @20 @25 @30


Metadata (M) 2.96 5.93 6.17 6.48 7.70 7.78
Caption (L) 1.11 0.56 0.74 1.39 2.22 2.22
Audio (A) 0.00 0.00 0.00 0.00 0.00 0.95


Table 5. Overlap percentage in Top-K results for vision only and
vision + additional modality.


results combine scores from both modalities using the aggregation module (ref. Sec. 3.2). The top 30 videos from
both methods are then annotated for correctness by 3 anno
tators.

Tab. 4 shows that adding an additional modality consistently improves precision across all _K_ values compared
to using a vision-only model. The precision gap between
vision and vision+modality widens as _K_ increases. ∆ _avg_
represents the average difference in precision between vision and vision+additional modality:



1
∆ avg = _|K|_



� _|P_ _V,K_ _−_ _P_ _V_ + _X,K_ _|_ (7)

_K∈K_


|Col1|Col2|Brand Safety Filters|Col4|
|---|---|---|---|
|||Relevant<br>Scenes for<br>serving Ad||
|||||



Figure 4. End-to-End ContextIQ video retrieval system for contextual advertising (ref. Sec. 6)



**6. Contextual Advertising**


Contextual Advertising targets ads based on the content
a user is viewing, improving the experience, boosting engagement, and increasing conversion rates, all without using personal information, making it privacy-friendly. Fig. 4
shows how ContextIQ is integrated into the Connected TV
advertising ecosystem that enables advertisers to perform
contextual advertising.
**Processing Long-Form content.** Long-form content, such
as movies and shows, is processed by ContextIQ by breaking them into shorter videos using scene detection. We use
PySceneDetect [16] for scene detection with default parameters. Each scene is subsequently processed through ContextIQ’s multimodal embedding generation module (ref.
Sec. 3.1) to generate the reference multimodal scene embeddings.
**Integration into Ad Serving system.** Depending on the
brand campaign and the advertisements to be served, an
advertiser defines a set of relevant text queries. For example, a pet food brand might have queries such as _dogs_,
_cats_, _pet food_, etc. Using these brand-specific text queries
and the multimodal embeddings, ContextIQ’s multimodal
search (ref. Sec. 3.2) identifies scenes where creatives can
be contextually served. Additionally, these scenes can be
passed through brand safety filters to ensure that brands
don’t get associated with sensitive/profane scenes. The selected scenes are stored in the scene-to-context lookup DB.
When a viewer is watching the TV, the ad gateway looks
up the scene to context lookup DB to find out the relevant
context for showing advertisements; the ad gateway serves
the brand’s ad as shown in Fig. 4. ContextIQ can easily be
extended to retrieve relevant scenes for showing an ad based
on image, video or even audios. For example, an advertiser
might directly use their brand advertisement as a query for
video retrieval and get the relevant search results (more details in the Supplementary material).
**Brand Safety.** Ensuring brand safety is crucial in video retrieval systems for contextual advertising. Advertisers are
increasingly vigilant about the environments in which their
brands appear, as association with inappropriate content,
such as offensive language, hate speech, negative emotions,
adult material, or references to crime and terrorism, can
cause significant reputational damage and erode consumer



trust. To address these concerns, our ContextIQ system integrates a safety mechanism comprising two key filters: (i)
**Emotion Recognition** filter, which evaluates the emotional
quality of content, ensuring alignment with the brand’s messaging. (ii) **Hate Speech and Profanity Detection** filter,
which blocks content containing hate speech, explicit language, or other inappropriate communication, preventing
ads from appearing alongside harmful content. Since we
already extract emotional and profanity information during
metadata extraction (ref. Sec. 3.1.4 & 4.2), we use the same
information during brand safety filtering with no additional

compute.


**Modularity.** ContextIQ, with its diverse set of expert models, offers flexibility by allowing the use of a specific subset
of models tailored to particular use cases. For example, the
textual modality can be used for real-time content ingestion
to serve ads during live news segments, filtering out violent content that many brands prefer to avoid. Additionally,
each expert model can be fine-tuned to meet specific brand
requirements. For instance, place and object detection models can be fine-tuned accordingly to support a casino brand
looking to detect both a casino location and a roulette wheel
within the content.


**7. Conclusion**


This paper introduces ContextIQ, an end-to-end video
retrieval system designed for contextual advertising. By
leveraging multimodal experts across video, audio, captions, and metadata, ContextIQ effectively aggregates these
diverse modalities to create semantically rich video representations. We demonstrate strong performance on multiple video retrieval benchmarks, achieving results better
or comparable to jointly trained multimodal models without requiring extensive multimodal datasets and computational resources. Our ablation study shows the advantage of
incorporating multiple modalities over a vision-only baseline. We further examine how ContextIQ extends beyond
the conventional video retrieval task by integrating seamlessly into the ad ecosystem, processing streamed long-form
content, offering modularity for efficient real-time ad serving, and implementing brand safety filters to ensure ads are
placed within contextually appropriate and safe content.


**References**


[1] GitHub - dimitrismistriotis/alt-profanity-check: A fast, robust library to check for offensive language in strings,
dropdown replacement of ”profanity-check”. — github.com.
[https://github.com/dimitrismistriotis/](https://github.com/dimitrismistriotis/alt-profanity-check)
[alt-profanity-check. [Accessed 08-09-2024]. 5](https://github.com/dimitrismistriotis/alt-profanity-check)

[2] GitHub - LAION-AI/CLAP: Contrastive Language-Audio
Pretraining — github.com. [https://github.com/](https://github.com/LAION-AI/CLAP)
[LAION-AI/CLAP. [Accessed 08-09-2024]. 5](https://github.com/LAION-AI/CLAP)

[3] GitHub - surge-ai/profanity: The world’s largest profanity
[list. — github.com. https://github.com/surge-](https://github.com/surge-ai/profanity)
[ai/profanity. [Accessed 08-09-2024]. 5](https://github.com/surge-ai/profanity)

[4] LanguageBind/LanguageBind ~~V~~ ideo ~~F~~ T · Hugging Face
— huggingface.co. [https://huggingface.co/](https://huggingface.co/LanguageBind/LanguageBind_Video_FT)
[LanguageBind/LanguageBind_Video_FT.](https://huggingface.co/LanguageBind/LanguageBind_Video_FT) [Accessed 08-09-2024]. 6

[5] Streamlit • A faster way to build and share data apps —
[streamlit.io. https://streamlit.io/. [Accessed 08-](https://streamlit.io/)
09-2024]. 6

[6] tae898/emoberta-large · Hugging Face — huggingface.co.

[https://huggingface.co/tae898/emoberta-](https://huggingface.co/tae898/emoberta-large)
[large. [Accessed 08-09-2024]. 5](https://huggingface.co/tae898/emoberta-large)

[7] AI@Meta. Llama 3 model card. 2024. 5

[8] Jason Ansel et al. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and
Graph Compilation. In _29th ACM International Conference_
_on Architectural Support for Programming Languages and_
_Operating Systems, Volume 2 (ASPLOS ’24)_ . ACM, Apr.
2024. 5

[9] Max Bain, Arsha Nagrani, Andrew Brown, and Andrew Zisserman. Condensed movies: Story based retrieval with contextual embeddings, 2020. 2, 6

[10] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. A clip-hitchhiker’s guide to long video retrieval, 2022.
5

[11] Dmitry Baranchuk, Artem Babenko, and Yury Malkov. Revisiting the inverted indices for billion-scale approximate
nearest neighbors, 2018. 1

[12] Samuel Suraj Bushi and Osmar Zaiane. Apnea: Intelligent
ad-bidding using sentiment analysis. WI ’19, page 76–83,
New York, NY, USA, 2019. Association for Computing Machinery. 2

[13] Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe
Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N.
Chang, Sungbok Lee, and Shrikanth S. Narayanan. Iemocap: interactive emotional dyadic motion capture database.
_Language Resources and Evaluation_, 42(4):335–359, Nov.
2008. 5, 13

[14] Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe
Hillier, and Andrew Zisserman. A short note about kinetics600. _arXiv preprint arXiv:1808.01340_, 2018. Submitted on
3 Aug 2018. 5, 12

[15] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zis
serman. A short note on the kinetics-700 human action

dataset. _arXiv preprint arXiv:1907.06987_, 2019. Submitted
on 15 Jul 2019 (v1), last revised 17 Oct 2022 (this version,
v2). 5, 12




[16] Brandon Castellano. Home - PySceneDetect — scenede[tect.com. https://www.scenedetect.com/. [Ac-](https://www.scenedetect.com/)
cessed 08-09-2024]. 8

[17] Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, Jinhui Tang, and Jing Liu. Valor: Vision-audiolanguage omni-perception pretraining model and dataset,
2023. 1, 2

[18] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao,
Mingzhen Sun, Xinxin Zhu, and Jing Liu. VAST: A
vision-audio-subtitle-text omni-modality foundation model
and dataset. In _Thirty-seventh Conference on Neural Infor-_
_mation Processing Systems_, 2023. 1, 2

[19] Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, and
Dong Shen. Improving video-text retrieval by multi-stream
corpus alignment and dual softmax loss, 2021. 3

[20] Adrian Dabrowski, Georg Merzdovnik, Johanna Ullrich,
Gerald Sendera, and Edgar Weippl. Measuring cookies and
web privacy in a post-gdpr world. In David Choffnes and
Marinho Barcellos, editors, _Passive and Active Measure-_
_ment_, pages 258–270, Cham, 2019. Springer International
Publishing. 1

[21] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazar´e, Maria
Lomeli, Lucas Hosseini, and Herv´e J´egou. The faiss library,
2024. 1

[22] Mai ElSherief, Caleb Ziems, David Muchlinski, Vaishnavi
Anupindi, Jordyn Seybolt, Munmun De Choudhury, and
Diyi Yang. Latent hatred: A benchmark for understanding
implicit hate speech. In _Proceedings of the 2021 Confer-_
_ence on Empirical Methods in Natural Language Processing_,
pages 345–363, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics.
14

[23] Han Fang, Pengfei Xiong, Luhui Xu, and Yu Chen.
Clip2video: Mastering video-text retrieval via image clip,
2021. 2

[24] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and
Kaiming He. Slowfast networks for video recognition, 2019.
2

[25] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia
Schmid. Multi-modal transformer for video retrieval, 2020.

3

[26] Yuying Ge, Yixiao Ge, Xihui Liu, Alex Jinpeng Wang, Jianping Wu, Ying Shan, Xiaohu Qie, and Ping Luo. Miles: Visual bert pre-training with injected language semantics for
video-text retrieval, 2022. 2

[[27] Google AdSense. Google adsense. https://adsense.](https://adsense.google.com/start/)
[google.com/start/, 2024. Accessed: 2024-08-27. 1](https://adsense.google.com/start/)

[28] Google Cloud. Vertex ai: Multimodal embeddings
api. [https : / / cloud . google . com / vertex -](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api)
[ai/generative- ai/docs/model- reference/](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api)
[multimodal- embeddings- api, 2024.](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api) Accessed:
2024-08-27. 2, 6

[29] Keyan Guo, Alexander Hu, Jaden Mu, Ziheng Shi, Ziming
Zhao, Nishant Vishwamitra, and Hongxin Hu. An investigation of large language models for real-world hate speech
detection, 2024. 14


[30] Xudong Guo, Xun Guo, and Yan Lu. Ssan: Separable selfattention network for video representation learning, 2021. 2

[31] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr
Doll´ar, and Ross Girshick. Masked autoencoders are scalable
vision learners, 2021. 1

[32] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep
residual learning for image recognition. _2016 IEEE Confer-_
_ence on Computer Vision and Pattern Recognition (CVPR)_,
pages 770–778, 2015. 5

[33] Matthew Honnibal, Ines Montani, Sofie Van Landeghem,
and Adriane Boyd. spaCy: Industrial-strength Natural Language Processing in Python. 2020. 5

[34] E. H¨aglund and J. Bj¨orklund. Ai-driven contextual advertising: Toward relevant messaging without personal data. _Jour-_
_nal of Current Issues & Research in Advertising_, 45(3):301–
319, 2024. 1

[35] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang,
Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola,
Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman,
and Andrew Zisserman. The kinetics human action video

dataset. _arXiv preprint arXiv:1705.06950_, 2017. Submitted
on 19 May 2017. 5, 12

[36] Jiyun Kim, Byounghan Lee, and Kyung-Ah Sohn. Why
is it hate speech? masked rationale prediction for explainable hate speech detection. In Nicoletta Calzolari, ChuRen Huang, Hansaem Kim, James Pustejovsky, Leo Wanner, Key-Sun Choi, Pum-Mo Ryu, Hsin-Hsi Chen, Lucia Donatelli, Heng Ji, Sadao Kurohashi, Patrizia Paggio,
Nianwen Xue, Seokhwan Kim, Younggyun Hahm, Zhong
He, Tony Kyungil Lee, Enrico Santus, Francis Bond, and
Seung-Hoon Na, editors, _Proceedings of the 29th Inter-_
_national Conference on Computational Linguistics_, pages
6644–6655, Gyeongju, Republic of Korea, Oct. 2022. International Committee on Computational Linguistics. 5

[37] Taewoon Kim and Piek Vossen. Emoberta: Speaker-aware
emotion recognition in conversation with roberta, 2021. 5,
13

[38] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg,
Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for
video-and-language learning via sparse sampling, 2021. 2

[39] Nicolas Lemieux and Rita Noumeir. A hierarchical learning approach for human action recognition. _Sensors_, 20(17),
2020. 12

[40] Dongxu Li, Junnan Li, Hung Le, Guangsen Wang, Silvio
Savarese, and Steven C.H. Hoi. LAVIS: A one-stop library
for language-vision intelligence. In _Proceedings of the 61st_
_Annual Meeting of the Association for Computational Lin-_
_guistics (Volume 3: System Demonstrations)_, pages 31–41,
Toronto, Canada, July 2023. Association for Computational
Linguistics. 5

[41] Huiran Li and Yanwu Yang. Keyword targeting optimization
in sponsored search advertising: Combining selection and
matching. _Electronic Commerce Research and Applications_,
56:101209, 2022. 1

[42] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
Blip-2: bootstrapping language-image pre-training with
frozen image encoders and large language models. In _Pro-_



_ceedings of the 40th International Conference on Machine_
_Learning_, ICML’23. JMLR.org, 2023. 3, 4, 5, 13

[43] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang,
Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin
Wang, and Yu Qiao. Mvbench: A comprehensive multimodal video understanding benchmark, 2024. 1

[44] Jingang Liu, Chunhe Xia, Xiaojian Li, Haihua Yan, and
Tengteng Liu. A bert-based ensemble model for chinese
news topic prediction. BDE ’20, page 18–23, New York,
NY, USA, 2020. Association for Computing Machinery. 2

[45] Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. Use what you have: Video retrieval using representations from collaborative experts, 2020. 1, 3

[46] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei,
Nan Duan, and Tianrui Li. Clip4clip: An empirical study of
clip for end to end video clip retrieval, 2021. 1, 2, 5, 7

[47] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang,
and Rongrong Ji. X-clip: End-to-end multi-grained contrastive learning for video-text retrieval, 2022. 2

[48] Yu. A. Malkov and D. A. Yashunin. Efficient and robust
approximate nearest neighbor search using hierarchical navigable small world graphs, 2018. 1

[49] Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris
Biemann, Pawan Goyal, and Animesh Mukherjee. Hatexplain: A benchmark dataset for explainable hate speech detection. In _Proceedings of the AAAI Conference on Artificial_
_Intelligence_, volume 35, pages 14867–14875, 2021. 5

[50] Antoine Miech, Ivan Laptev, and Josef Sivic. Learning a
text-video embedding from incomplete and heterogeneous
data, 2020. 1, 3

[51] Niluthpol Chowdhury Mithun, Juncheng Li, Florian Metze,
and Amit K. Roy-Chowdhury. Learning joint embedding
with multimodal cues for cross-modal video-text retrieval. In

_Proceedings of the 2018 ACM on International Conference_
_on Multimedia Retrieval_, ICMR ’18, page 19–27, New York,
NY, USA, 2018. Association for Computing Machinery. 1, 2

[52] OpenAI. Chatgpt (gpt-4). [https://chat.openai.](https://chat.openai.com)
[com, 2024. Accessed: 2024-09-08. 6](https://chat.openai.com)

[53] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder,
Gautam Naik, Erik Cambria, and Rada Mihalcea. MELD:
A multimodal multi-party dataset for emotion recognition in
conversations. In Anna Korhonen, David Traum, and Llu´ıs
M`arquez, editors, _Proceedings of the 57th Annual Meeting of_
_the Association for Computational Linguistics_, pages 527–
536, Florence, Italy, July 2019. Association for Computational Linguistics. 5, 13

[54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. _CoRR_,
abs/2103.00020, 2021. 2, 4, 7

[55] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence
embeddings using siamese bert-networks. In _Proceedings of_
_the 2019 Conference on Empirical Methods in Natural Lan-_
_guage Processing_ . Association for Computational Linguistics, 11 2019. 5


[56] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang
Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365:
A large-scale, high-quality dataset for object detection. In
_Proceedings of the IEEE/CVF International Conference on_
_Computer Vision (ICCV)_, October 2019. 5

[57] Nina Shvetsova, Brian Chen, Andrew Rouditchenko, Samuel
Thomas, Brian Kingsbury, Rogerio Feris, David Harwath,
James Glass, and Hilde Kuehne. Everything at once – multimodal fusion transformer for video retrieval, 2022. 2

[58] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan
Liu. Mpnet: Masked and permuted pre-training for language
understanding. _arXiv preprint arXiv:2004.09297_, 2020. 5

[59] Joanna Strycharz, Edith Smit, Natali Helberger, and Guda
van Noort. No to cookies: Empowering impact of technical
and legal knowledge on rejecting tracking cookies. _Comput-_
_ers in Human Behavior_, 120:106750, 2021. 1

[60] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong
Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun
Huang, and Xinlong Wang. Emu: Generative pretraining in
multimodality, 2024. 1

[61] Jing Yang Taylor Jing Wen, Ching-Hua Chuan and
Wanhsiu Sunny Tsai. Predicting advertising persuasiveness: A decision tree method for understanding emotional
(in)congruence of ad placement on youtube. _Journal of Cur-_
_rent Issues & Research in Advertising_, 43(2):200–218, 2022.

2

[62] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang.
Videomae: Masked autoencoders are data-efficient learners
for self-supervised video pre-training, 2022. 1

[[63] Twelve Labs. Introducing marengo 2.6. https://www.](https://www.twelvelabs.io/blog/introducing-marengo-2-6)
[twelvelabs.io/blog/introducing-marengo-](https://www.twelvelabs.io/blog/introducing-marengo-2-6)
[2-6, 2024. Accessed: 2024-08-27. 2, 6](https://www.twelvelabs.io/blog/introducing-marengo-2-6)

[64] Ultralytics. YOLOv5: A state-of-the-art real-time object de[tection system. https://docs.ultralytics.com,](https://docs.ultralytics.com)
2021. 5

[65] Nikhita Vedula, Wei Sun, Hyunhwan Lee, Harsh Gupta, Mitsunori Ogihara, Joseph Johnson, Gang Ren, and Srinivasan
Parthasarathy. Multimodal content analysis for effective advertisements on youtube, 2017. 2

[66] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae
v2: Scaling video masked autoencoders with dual masking,
2023. 1, 12

[67] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae
v2: Scaling video masked autoencoders with dual masking.
_arXiv preprint arXiv:2303.16727_, 2023. Submitted on 29
Mar 2023 (v1), last revised 18 Apr 2023 (this version, v2). 5

[68] Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou.
One-peace: Exploring one general representation model toward unlimited modalities, 2023. 2, 7

[69] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He,
Chenting Wang, Guo Chen, Baoqi Pei, Ziang Yan, Rongkun
Zheng, Jilan Xu, Zun Wang, Yansong Shi, Tianxiang Jiang,
Songze Li, Hongjie Zhang, Yifei Huang, Yu Qiao, Yali
Wang, and Limin Wang. Internvideo2: Scaling foundation
models for multimodal video understanding, 2024. 1, 2




[70] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and
Denny Zhou. Chain-of-thought prompting elicits reasoning
in large language models. In _Proceedings of the 36th Inter-_
_national Conference on Neural Information Processing Sys-_
_tems_, NIPS ’22, Red Hook, NY, USA, 2024. Curran Asso
ciates Inc. 14

[71] Yusong Wu*, Ke Chen*, Tianyu Zhang*, Yuchen Hui*, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and
keyword-to-caption augmentation. In _IEEE International_
_Conference on Acoustics, Speech and Signal Processing,_
_ICASSP_, 2023. 2, 4, 5, 14

[72] Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye,
Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang,
Guohai Xu, Ji Zhang, Songfang Huang, Fei Huang, and Jingren Zhou. mplug-2: A modularized multi-modal foundation
model across text, image and video, 2023. 1

[73] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large
video description dataset for bridging video and language.
In _2016 IEEE Conference on Computer Vision and Pattern_
_Recognition (CVPR)_, pages 5288–5296, 2016. 2, 3, 6

[74] Kaifu Zhang and Zsolt Katona. Contextual advertising. _Mar-_
_keting Science_, 31(6):980–994, 2012. 1

[75] Shuai Zhao, Linchao Zhu, Xiaohan Wang, and Yi Yang. Centerclip: Token clustering for efficient text-video retrieval. In
_Proceedings of the 45th International ACM SIGIR Confer-_
_ence on Research and Development in Information Retrieval_,
SIGIR ’22. ACM, July 2022. 2

[76] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva,
and Antonio Torralba. Places: A 10 million image database
for scene recognition. _IEEE Transactions on Pattern Analy-_
_sis and Machine Intelligence_, 2017. 5

[77] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa
Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, and Li Yuan.
Languagebind: Extending video-language pretraining to nmodality by language-based semantic alignment, 2024. 2, 6,

7

[78] Cunjuan Zhu, Qi Jia, Wei Chen, Yanming Guo, and Yu Liu.
Deep learning for video-text retrieval: a review, 2023. 1


**A. Alternative Modality Queries to ContextIQ**


The flexibility of our system allows us to effortlessly perform queries across different modalities, including video,
audio, and image, ensuring any-to-any search capabilities.
**Image Query:** The process begins by encoding the input query image using the vision encoder of the vision-text
model _f_ _θ_ 1, resulting in an image embedding. This embedding is then compared against the vision embeddings of all
available content _{F_ _i_ _[v]_ [:] _[ i]_ [ = 1] _[,]_ [ 2] _[, ..., N]_ _[}]_ [ using cosine sim-]
ilarity. The system retrieves and ranks content based on
these similarity scores.
**Video Query:** For video queries, the system first extracts frame-level embeddings from the sampled frames of
the query video using the vision encoder of the vision-text
model _f_ _θ_ 1 . These frame-level embeddings are then aggregated using the previously defined aggregation function
_A_ _v_ to generate a single video embedding. This video embedding is compared directly with the vision embedding
database _{F_ _i_ _[v]_ : _i_ = 1 _,_ 2 _, ..., N_ _}_ using cosine similarity.
The content is then ranked according to similarity, with the
most relevant videos appearing at the top of the results.
**Audio Query:** The process for audio queries begins by
segmenting the query audio into a fixed number of segments. Each segment is encoded using the audio encoder of
the audio-text model _g_ _θ_ 2 . These segment-level encodings
are then aggregated using the previously defined aggregation function _A_ _a_ to form a single audio embedding. This
aggregated embedding is compared against the audio embeddings database _{F_ _i_ _[a]_ [:] _[ i]_ [ = 1] _[,]_ [ 2] _[, ..., N]_ _[}]_ [ using cosine sim-]
ilarity. The results are then ranked based on these similarity

scores.


**B. Video Action Recognition**


**B.1. Simplifying Kinetics 710 classes**


Reducing Kinetics 710 [14, 15, 35] classes to minimize
inter-class confusion can be done by either discarding irrelevant classes or combining similar ones. A hierarchical approach to combining Kinetics classes was explored
in [39] using a clustering method. However, this approach
only provides examples rather than hierarchical clustering
for the entire Kinetics dataset. In our ContextIQ system, we
reduced the number of classes by collecting various signals
and manually determining which classes to discard or combine. As a result, the number of classes was reduced from
[710 to 185. The result is captured in this sheet (as referred in](https://github.com/AnokiAI/ContextIQ-Paper/blob/master/supplementary/video_action_recognition/simply_kinetics710_to_185_classes.xlsx)
the following paragraphs) present in our GitHub repository
[https://github.com/AnokiAI/ContextIQ-Paper/. The signals](https://github.com/AnokiAI/ContextIQ-Paper)
used were:


1. **Relevance to contextual advertiser** : Some classes,
like ”stretching arm” or ”shuffling feet” may be too
mundane, while others, like ”playing oboe” or ”clam



digging,” are too niche for a broad audience targeting.
Using GPT-4, we identified and marked about 50%
of classes as irrelevant for audience targeting (highlighted in red in the attached sheet). Examples of
discarded classes include ”Playing oboe” (niche instrument with limited audience), ”Pole vault” (niche
sport), and ”Stretching leg” (too general for segmentation).


GPT4 Prompt:
_I have a list of 710 actions. Create a downloadable_
_sheet with three columns, 710 actions, Discard, Rea-_
_son. The discard should be Yes, but only if it seems less_
_useful for detecting an action. If Discard is yes, also_
_mention the reason (1-2 lines). I want to discard about_
_50% of the actions to keep the most useful half. An ac-_
_tion is less useful if it does not seem helpful for creating_
_audience segments for ad targeting. E.g, pinching ac-_
_tion does not seem useful to target. Do not make any_
_action class. Use the 710 as it is_ .

_[[Paste the list of 710 actions]]_


2. **Groupings and correlated classes** : Many classes in
the Kinetics 710 set have high overlap, and their occurrence is highly correlated. For example, there are three
separate classes for playing guitar, strumming guitar,
and tapping guitar, which the model finds difficult to
differentiate, leading to higher inter-class confusion.
To address this, we used two approaches: one extends
the existing Kinetics 400 [35] groupings to 710 classes,
and the other examines the top correlated classes during prediction.


**K400 Groupings** : The K400 set [35] provided groupings of the 400 classes into 37 groups. However,
these groupings were not extended to the additional
300 classes in the K710 set. For these additional 300

classes, we inferred their groupings by finding their
text similarity with the 37 groups using the text encoder _h_ _θ_ 3 used in the main paper and tagging the class
to the most similar group. These classes are marked
with a double asterisk in the K400 grouping column in
the sheet.


**Top-3 correlated classes** : The VideoMAE2 model

[66] generates logits for 710 classes during inference.
We build a co-occurrence matrix by counting every
pair of classes in the Top-10 logit scores, then compute
the correlation matrix. This process is applied to both
the Kinetics validation set (50,000 videos) and our internal movie/TV clip set (Sec. 5.2 in the main paper).
For instance, classes like dunking, dribbling, shooting,
and playing basketball are highly correlated, allowing
us to merge them into a single class, such as playing
basketball.


3. **Accuracy on the K710 validation set** : Some classes
perform poorly on the simpler Kinetics validation set
(likely the same data distribution they were trained on),
making them less likely to perform well on our movie
clip dataset. We calculate the Top-1 and Top-3 accuracy for each class on the Kinetics set and highlight
those in the bottom 25th percentile in the sheet. For
instance, photobombing has a 38.8% Top-1 and 51%
Top-3 accuracy, making it a candidate for discarding
to reduce false positives and inter-class confusion.


4. **Class occurrence ranking** : Kinetics includes many
classes that rarely occur in the wild, such as wood
burning (art), stacking die, and wrestling alligator. In
our large, diverse internal dataset, we found that 90%
of the Top-3 search results come from only 218 classes
5. In the attached sheet, we list the occurrence count
rank of each class (from 1 to 710) and highlight those
beyond the top 218.


Figure 5. Cummulative Percentage Plot of Top-3 predicted actions
on the internal set


Using the above four signals, the 710 classes were manually screened and reduced to **185 classes** :


  - 182 classes were discarded.


  - 418 classes were combined into 89 classes (see Tab. 6

for some of the obtained combined classes).


  - 96 classes were retained.


Combining classes involves a trade-off between losing
specificity and improving average precision @ K and prediction confidence. For instance, while predicting a broader
class such as “drinking alcohol” (refer to row 2 of Table 6)
can yield higher precision, it sacrifices the ability to differentiate between specific types like wine and beer.


**C. Emotion Recognition**


**Text-based Emotion Recognition.** We use a pre-trained
Emoberta-Large [37] model, which is trained on the MELD



Table 6. Few examples of obtained combined classes

|Combined class|Actions belonging to the class|
|---|---|
|playing cards<br>drinking alcohol<br>riding animal<br>playing board game<br>cleaning foor|playing poker, shuffing cards, card<br>stacking, card throwing, dealing<br>cards, playing blackjack<br>uncorking champagne, bartending,<br>drinking beer, drinking shots, tast-<br>ing beer, playing beer pong, pour-<br>ing beer,<br>tasting wine,<br>pouring<br>wine, opening bottle (not wine),<br>opening wine bottle<br>riding camel, riding elephant, rid-<br>ing mule, riding or walking with<br>horse<br>playing monopoly, playing check-<br>ers,<br>playing dominoes,<br>playing<br>mahjong, playing scrabble<br>cleaning<br>foor,<br>mopping<br>foor,<br>sweeping<br>foor,<br>brushing<br>foor,<br>vacuuming foor, sanding foor|



[53] and IEMOCAP [13] datasets, for text-based emotion
recognition as it is a speaker-aware model and shows better
performance empirically on movie scene subtitles.
**Leveraging Visual and Audio Cues for Emotion Recog-**
**nition** . The text-based models work only when there is
enough text for the model to make a prediction. Moreover, it is difficult to find subtitles for some content, but
still, we need to predict emotions in them for better retrieval
and brand safe filtering. Since we already use the visiontext model and audio-text models for different parts of the
ContextIQ system, we use these models to get some extra
signals for predicting emotion. For example, we tagged
the emotion _joy_ with text queries like _people smiling_ and
_people dancing_, and assigned the emotion _joy_ to all videos
retrieved through the video (vision) modality using these
queries. Concretely, we associate textual concepts that can
be linked to different emotions and then find the scenes that
have high video embedding similarity with the emotional
text concept. Assume _Q_ _t_ = _{t_ : _e}_ to be the text concept
dictionary which contains strings _t_ associated with different
emotions _e_ . Then, for a particular video scene _x_, we say
that it is associated to an emotion _e_ if,


_f_ _θ_ 1 ( _x_ ) _· f_ _θ_ _[T]_ 1 [(] _[t]_ [)] _[ > τ]_ _[e]_ (8)


where _f_ _θ_ 1 and _f_ _θ_ _[T]_ 1 [are the video and text encoders, re-]
spectively, of the vision-text model [42], and _τ_ _e_ is a predefined threshold for the concept-emotion pair _t_ : _e_ .
Empirical results show that textual emotion concepts
work well only for _joy_ emotion. For other emotions, either
it is difficult to find emotional text concepts which are rel

Table 7. Classification metrics for LLM, BERT and Ensemble Model


Explicit Hate vs Normal Speech Implicit Hate vs Normal Speech

Ensemble Ensemble Ensemble Ensemble
Metric LLM BERT LLM BERT
(OR, _θ_ = 0.7) (AND, _θ_ = 0.2) (OR, _θ_ = 0.7) (OR, _θ_ = 0.2)
Accuracy 83.9 77.7 81.5 85.3 75.3 63.4 73 73.2
Precision 78.9 75.9 74.5 82 75.2 66.8 70.7 76.9

Recall 92.3 81.1 95.1 90.2 74.9 52.4 78.1 66

F1 Score 85.1 78.4 83.5 85.9 75.1 58.8 74.2 71


Table 8. Differential Analysis for different prompting strategies


Explicit Hate vs Normal Speech Implicit Hate vs Normal Speech
Reasoning Yes No Yes Yes Yes Yes No Yes Yes Yes
Definition of Hate Speech Yes Yes Yes Yes No Yes Yes Yes Yes No
Number of Examples 3 3 1 0 3 3 3 1 0 3
Recall 94.6 **97.2** 95.2 93.5 94.9 76.5 **85.6** 74.8 77.8 75.5

Precision **73.9** 65.8 72.3 70.2 71.0 **70.3** 62.9 67.3 66.3 67.4

Accuracy **80.8** 73.4 79.4 77.1 78.7 **71.9** 67.6 69.2 69.3 69.5
F1 Score **83.0** 78.4 82.2 80.2 81.2 **73.3** 72.5 70.8 71.6 71.2



evant to that emotion, or the text concept associated to the
emotion is not well represented by the vision-text model.
Similar to visual concepts, we associate audio concepts
to different emotions given by _Q_ _a_ = _a_ : _e_, which contains
audio files _a_ and corresponding emotion _e_ associated with
that audio file. Then for a particular video scene _x_, we say
that it is associated to an emotion _e_ if,


_g_ _θ_ 2 ( _x_ _a_ ) _· g_ _θ_ 2 ( _a_ ) _> τ_ _e_ (9)


where _g_ _θ_ 2 is the audio encoder of CLAP [71], _x_ _a_ is the
audio for the given video and _τ_ _e_ is a predefined threshold
for the concept-emotion pair _a_ : _e_ . Note that we do not
use the text encoder of CLAP because text-audio matching
did not result into as good results as audio-audio matching.
We have only linked audio emotion concepts to _sad_ emotion
because the rest of the emotions do not show good results
empirically.


**D. Hate Speech Detection**


**Aggregation Strategy** : To combine predictions from the
BERT model, the scores for the Hate Speech and Offensive
classes are summed. This aggregated score is then compared against a threshold of _θ_ = 0 _._ 7. The final prediction
is obtained by applying a logical OR operation between the
thresholded BERT prediction and the predictions from the
LLM to boost recall.

**Prompting Strategies** : We implement various prompting techniques to enhance the predictive performance of the
LLM [29].

1. **Few-Shot Learning** : A few examples are provided to
the model to establish task context, improving its ability



to accurately identify hate speech. Specifically we use
three examples for the same.


2. **Definition of Hate Speech** : A precise definition of hate
speech is included in the prompt to ensure consistent detection aligned with the dataset annotations. We use the
following definition of hate speech : _Language that dis-_
_parages a person or group on the basis of protected char-_
_acteristics like race, gender, and cultural identity._


3. **Structured JSON Output** : The model is instructed to
return its response in JSON format, enabling easy parsing and seamless integration with the contextIQ system.


4. **Chain of Thought Reasoning** : The model is prompted
to generate intermediate reasoning steps before determining whether content qualifies as hate speech, enhancing prediction accuracy. [70]


Various analyses were performed to evaluate the effectiveness of these strategies by using a combination of
them for detection. Table 8 presents the results of these
analyses. The results demonstrate that incorporating
all the prompting strategies enhances detection performance, leading to improvements in accuracy, precision,
and F1 score.


**Validation Data and Results** : We conducted validation

using two datasets: an internal dataset and the implicit-hate
dataset [22]. For implicit-hate, we sampled 250 examples
each of Explicit Hate Speech, Implicit Hate Speech, and
Normal Speech to ensure a balanced evaluation across different types of speech. In contrast, the internal dataset consisted of 11,645 examples, which, after applying a profanity


filter, was reduced to 10,645. Given the unbalanced distribution of hate speech versus normal speech on internal
dataset, calculating recall was challenging. As a result, we
only focused on the positive predictions generated by each
model.

On the internal dataset, the BERT model identified 397
out of 10,645 examples (3.7%) as positive, while the LLM
predicted 509 examples (4.8%) as positive. To assess these
predictions, we randomly sampled 40 examples from each
set of positive predictions, which were reviewed by two independent curators, given the subjective nature of the task.
While precision varied significantly between curators owing to the subjective nature of the task, the LLM consistently outperformed the BERT model, with an average delta
of 7.5%.

For the implicit-hate dataset, we evaluated various
prompt templates and temperature values to enhance the
performance of the LLM. A temperature value of 0.6, com[bined with the prompt template described here, yielded the](https://github.com/AnokiAI/ContextIQ-Paper/blob/master/supplementary/hatespeech_detection/default_prompt_template.yaml)
optimal results. Table 7 presents the results for the best parameter combinations for both the LLM-based and BERT

models, along with the outcomes for the ensemble models.
The ensemble model outperformed the individual models,
offering the flexibility to fine-tune precision and recall according to specific requirements. Additionally, the table
also provides results for the ensemble model using both
AND and OR operations across two different threshold values. The selection of these parameters can be guided by
the desired balance between precision and recall in different scenarios.


