## **TL;DW? Summarizing Instructional Videos with** **Task Relevance & Cross-Modal Saliency**

Medhini Narasimhan [1] _[,]_ [2] _[∗]_, Arsha Nagrani [2], Chen Sun [2] _[,]_ [3], Michael Rubinstein [2],
Trevor Darrell [1] _[†]_, Anna Rohrbach [1] _[†]_, and Cordelia Schmid [2] _[†]_


1 UC Berkeley 2 Google Research 2 Brown University
```
           https://medhini.github.io/ivsum

```




Summary comprising of task relevant and salient steps in input video


Fig. 1: **Summarizing Instructional Videos** We introduce an approach for creating
short visual summaries comprising steps that are most relevant to the task, as well as
salient in the video, i.e. referenced in the speech. For example, given a long video on
_“How to make a veggie burger”_ shown above, the summary comprises key steps such as
_fry ingredients, blend beans_, and _fry patty_ .


**Abstract.** YouTube users looking for instructions for a specific task may
spend a long time browsing content trying to find the right video that
matches their needs. Creating a visual summary (abridged version of a
video) provides viewers with a quick overview and massively reduces search
time. In this work, we focus on summarizing _instructional_ videos, an underexplored area of video summarization. In comparison to generic videos,
instructional videos can be parsed into semantically meaningful segments
that correspond to important steps of the demonstrated task. Existing
video summarization datasets rely on manual frame-level annotations,
making them subjective and limited in size. To overcome this, we first
automatically generate _pseudo summaries_ for a corpus of instructional
videos by exploiting two key assumptions: (i) relevant steps are likely to
appear in multiple videos of the same task ( _Task Relevance_ ), and (ii) they
are more likely to be described by the demonstrator verbally ( _Cross-Modal_
_Saliency_ ). We propose an instructional video summarization network that
combines a context-aware temporal video encoder and a segment scoring


TL;DW? - Too Long; Didn’t Watch?
_∗_ Work done while an intern at Google Research. Correspondence to
```
 medhini@berkeley.edu
```

_†_ Equal contribution.


2 M. Narasimhan et al.


transformer. Using pseudo summaries as weak supervision, our network
constructs a visual summary for an instructional video given only video
and transcribed speech. To evaluate our model, we collect a high-quality
test set, _WikiHow Summaries_, by scraping WikiHow articles that contain
video demonstrations and visual depictions of steps allowing us to obtain
the ground-truth summaries. We outperform several baselines and a
state-of-the-art video summarization model on this new benchmark.


**1** **Introduction**


The search query _“How to make a veggie burger?”_ on YouTube yields thousands
of videos, each showing a slightly different technique for the same task. It is often
time-consuming for a first-time burger maker to sift through this plethora of
video content. Imagine instead, if they could watch a compact visual summary of
each video which encapsulates all semantically meaningful steps relevant to the
task. Such a summary could provide a quick overview of what the longer video
has to offer, and may even answer some questions about the task without the
viewer having to watch the whole video. In this work, we propose a method to
create such succinct visual summaries from long instructional videos.
Since our goal is to summarize videos, we consider prior work on generic [ 9, 35 ]
and query-focused [ 34 ] video summarization. Generic video summarization datasets

[ 9, 35 ] tend to contain videos from _unrestricted domains_ such as sports, news and
day-to-day events. Given that annotations are obtained manually, the notion
of what constitutes a good summary is subjective, and might differ from one
annotator to the next. Query-focused video summarization partially overcomes
this subjectivity by allowing users to customize a summary by specifying a natural
language query [ 34, 22 ]. However, both generic and query-focused approaches
require datasets to be annotated manually at a per-frame level. This is very
expensive, resulting in very small-scale datasets (25-50 videos) with limited utility
and generalization.
Here, we focus on a specific domain – that of instructional videos [ 36, 45, 21 ].
We argue that a unique characteristic of these videos is that a summary can
be clearly defined as a minimally sufficient _procedural_ one, i.e., it must include
the steps necessary to complete the task (see Fig. 1). To circumvent having
to manually annotate our training data, we use an unsupervised algorithm to
obtain weak supervision in the form of pseudo ground-truth summaries for a
large corpus of instructional videos. We design our unsupervised objectives based
on two hypotheses: (i) steps that are relevant to the task will appear across
multiple videos of the same task, and (ii) salient steps are more likely to be
described by the demonstrator verbally. In practice, we segment the video and
group individual segments into steps based on their visual similarity. Then we
compare the steps across videos of the same task to obtain _task relevance scores_ .
We also transcribe the videos using Automatic Speech Recognition (ASR) and
compare the video segments to the transcript. We aggregate these _task relevance_
and _cross-modal scores_ to obtain the _importance scores_ for all segments, i.e., our
pseudo ground-truth summary.


Summarizing Instructional Videos 3


Next, given an input video and transcribed speech, we train an instructional
video summarization network ( _IV-Sum_ ). IV-Sum learns to assign scores to short
_video segments_ using 3D video features which capture temporal context. Our
network consists of a video encoder that learns context-aware temporal representations for each segment and a segment scoring transformer (SST) that then
assigns importance scores to each segment. Our model is trained end-to-end using
the importance scores from the pseudo summaries. Finally, we concatenate the
highest scoring segments to form the final video summary.
While we can rely on pseudo ground-truth for training, we collect a clean,
manually verified test set to evaluate our method. Since manually creating a
labeled test set from scratch would be extremely expensive, we find a solution in
the form of the WikiHow resource [1] . WikiHow articles often contain a link to an

instructional video and a set of human-annotated steps present in the task along
with corresponding images or short clips. To construct our test set (referred to
as _WikiHow Summaries_ ), we automatically localize these images/clips in the
video. We obtain localized segments for the images (using a window around the
localized frame) and clips, and stitch the segments together to create a summary.
This provides us with binary labels for each frame which serve as ground-truth
annotations. We evaluate our model on _WikiHow Summaries_ and compare it to
several baselines and the state-of-the-art video summarization model CLIP-It [ 22 ].
Our model surpasses prior work and several baselines on three standard metrics
(F-Score, Kendall [15], and Spearman [46] coefficients).
To summarize (pun intended), we introduce an approach for summarizing instructional videos that involves training our _IV-Sum_ model on pseudo summaries
created from a large corpus of instructional videos. _IV-Sum_ learns to rank different segments in the video by learning context-aware temporal representations for
each segment and a segment scoring transformer that assigns scores to segments
based on their task relevance and cross-modal saliency. Our method is weaklysupervised (it only requires the task labels for videos), multimodal – uses both
video and speech transcripts, and is scalable to large online corpora of instructional
videos. We collect a high-quality test set, _WikiHow Summaries_ for benchmarking
instructional video summarization, which will be publicly released. Our model outperforms state-of-the-art video summarization methods on all metrics. Compared
to the baselines, our method is especially good at capturing task relevant steps
and assigning higher scores to salient frames, as seen through qualitative analysis.


**2** **Related Work**


We review several lines of work related to summarization of instructional videos.

**Generic Video Summarization.** This task involves creating abridged versions
of generic videos by stitching together short important clips from the original
video [ 10, 19, 22, 25, 29, 40, 42, 43, 44 ]. Some of the more recent methods attempt to
learn contextual representations to perform video summarization, via attention


1 `[https://www.wikihow.com/](https://www.wikihow.com/)`


4 M. Narasimhan et al.


mechanism [ 7 ], graph based [ 25 ] or transformer-based [ 22 ] methods. Representative datasets include SumMe [ 9 ] and TVSum [ 35 ], where the ground-truth
summaries were created by annotators assigning scores to each frame in the video,
which is highly time consuming and expensive. As a consequence, the generic
video summarization datasets are small and the quality of the summaries is often
very subjective. Here, we focus on instructional videos which contain structure in
the form of task steps, thus we have a clear definition of what a good summary
should contain - a set of necessary steps for performing that specific task.
**Query Focused Video Summarization.** To address the subjectivity issues
with Generic Summarization, Query Focused Video Summarization allowed for
having user defined natural language queries to customize the summaries [ 14, 34, 38 ].
A representative dataset is Query Focused Video Summarization [ 33 ]; it is very
small and the queries correspond to a very narrow set of objects. In contrast, our
task is large and we do not rely on any additional user input.
**Step Localization.** Step localization (also known as temporal action segmentation) is a related albeit distinct task. It typically implies predicting
temporal boundaries of steps when the step labels [ 28, 36, 45 ] and even their
ordering [ 2, 4, 6, 12, 17, 27 ] are given. Representative datasets, COIN [ 36 ] and
CrossTask [ 45 ] consist of instructional videos and a fixed set of steps for each task
(from the WikiHow resource), and the task is to localize these steps in the video.
Our task is different in that we are only given a video without corresponding
input steps. Our model learns to pick out segments that correspond to relevant
and salient steps in order to construct a video summary. We discuss and illustrate
the shortcomings of the step localization annotations in Sec. 5 and Fig. 6.
**Unsupervised Parsing of Instructional Videos.** Closest to ours is the line
of work on unsupervised video parsing and segmentation that discovers steps in
instructional videos in an unsupervised manner [ 1, 8, 18, 32, 31 ]. However, these
works - (1) do not focus on video summarization, thus they might miss some salient
steps in video, (2) often use very small datasets for training and evaluation that
do not capture the broad range of instructional videos found in, e.g., COIN [ 36 ]
and CrossTask [45].


**3** **Summarizing Instructional Videos**


**Overview.** We propose a novel approach for constructing visual summaries
of instructional videos. An instructional video typically consists of a visual
demonstration of a specific task, e.g. _“How to make a pancake?”_ . Our goal is to
construct a visual summary of the input video containing only the steps that are
crucial to the task and salient in the video, i.e. referenced in the speech. Fig. 2
illustrates an outline of our approach. Our instructional video summarization
pipeline consists of two stages - (i) first, we use a weakly supervised algorithm to
generate pseudo summaries and frame-wise importance scores for a large corpus
of instructional videos, relying only on the task label for each video (ii) next,
using the pseudo summaries as supervision, we train an instructional video
summarization network which takes as input the video and the corresponding


Summarizing Instructional Videos 5







Make Strawberry Cake


Set up a campsite



Grow a lemon tree


Obtain scores for all videos









Fig. 2: **Summarizing Instructional Videos.** We first obtain pseudo summaries for a
large collection of videos using our weakly supervised algorithm (more details in Fig. 3).
Next, using the pseudo summaries as weak-supervision, we train our Instructional Video
Summarizer ( _IV-Sum_ ). It takes an input video along with the corresponding ASR
transcript and learns to assign importance scores to each segment in the video. The
final summary is a compilation of the high scoring video segments.


transcribed speech and learns to assign scores to different segments in the input
video. The network consists of a video encoder and a segment scoring transformer
(SST) and is trained using the importance scores of the pseudo summaries.
The final summary is constructed by selecting and concatenating the segments
with high importance scores. We first describe our pseudo summary generation
algorithm, followed by details on our instructional video summarizer ( _IV-Sum_ ),
and the inference procedure.


**3.1** **Generating Pseudo Summaries**


Since manually collecting annotations for summarization is expensive and time
consuming, we propose an automatic weakly supervised approach for generating
summaries that may contain noise but have enough valuable signal for training
a summarization network. The main intuition behind our pseudo summary
generation pipeline is that given many videos of a task, steps that are crucial to
the task are likely to appear across multiple videos (task relevance). Additionally,
if a step is important, it is typical for the demonstrator to speak about this step
either before, during, or after performing it. Therefore, the subtitles for the video
obtained using Automatic Speech Recognition (ASR) will likely reference these
key steps (cross-modal saliency). These two hypotheses shape our objectives for
generating pseudo summaries.
**Task Relevance.** We first group videos based on the task. Say videos _V_ _i_ _, i ∈_

[1 _, . . . K_ ] are _K_ videos from the same task, as shown in Fig. 3. For a given video,
we divide it into _N_ equally sized non-overlapping segments _s_ _i_ _, i ∈_ [1 _, . . . N_ ] and
embed each segment using a pre-trained 3D CNN video encoder _g_ _vid_ [ 20 ]. We


6 M. Narasimhan et al.


n



Task Relevance


Compare to other videos of the same task



Video K





10





3


2


1

















Combine steps to obtain

Pseudo summaries



Video 1





Group video segments along
Extract Features for Video Segments Assign importance scores to steps

time into steps



Fig. 3: **Pseudo Summary Generation.** To generate the pseudo summary, we first
uniformly partition the video into segments, then group the segments based on visual
similarity into steps (shown in different colors), assign _importance scores_ to steps based
on _Task Relevance_ and _Cross-Modal Saliency_, and then pick high scoring steps to obtain
pseudo summaries.


merge segments along the time axis based on their dot-product similarity, i.e.
if similarity of a segment to the one prior to it is greater than a threshold, the
two are grouped together and the joint feature representation is an average of
the feature representation of the two segments. The threshold for similarity is
heuristically set to be 90% of the maximum similarity between any two segments
in the video. We call these merged segments _steps_, as they typically correspond
to semantic steps as we show through qualitative results in supplemental. We do
this for all _K_ videos in the task, and then compare each step to all the _S_ steps
across all _K_ videos of the task. We assign _task relevance scores_ trs _S_ _i_, to each step
_S_ _i_ _, i ∈S_ based on its visual similarity to all the _S_ steps from all _K_ videos of this
task, as shown below:



trs _S_ _i_ = _|S|_ [1]



� _g_ _vid_ ( _S_ _i_ ) _· g_ _vid_ ( _S_ _j_ )

_j∈S_



**Cross-Modal Saliency.** We also compare each video step to each sentence in
the transcript of the same video. This enforces our idea that if a step is important,
it will likely be referenced in the speech. To do this, we encode both, the input
segments and the transcript sentences, using a pre-trained video-text model
where the video and text streams are trained jointly using MIL-NCE loss [ 20 ].
Each visual step is assigned a _cross-modal score_ by averaging its similarity over
all the sentences.


Each step (and all the segments in it) is then assigned an importance score that
is an average of the _task relevance_ and the _cross-modal scores_ . This constitutes
our pseudo summary scores. For any given video, the top _t_ % highest scoring
steps are retained to be a part of the summary.


Summarizing Instructional Videos 7


**3.2** **Instructional Video Summarizer (** _**IV-Sum**_ **)**


Recall that our goal is to construct a visual summary of any instructional video
by picking out the important steps in it, without having to rely on other videos
of the same task or the task label. To do this, we use the pseudo summaries
generated above as weak supervision to train _IV-Sum_, which learns to assign
importance scores to individual segments in the video using only the information
in the video and the corresponding transcripts as seen in Fig. 2. While some
prior summarization methods operate on independent frames [ 22, 25 ], _IV-Sum_
operates on non-overlapping segments _s_ _i_ _, i ∈_ [1 _, . . . N_ ], and learns _context-aware_
_temporal representations_ using a 3D CNN video encoder _f_ vid . The transcript is
projected onto the same embedding space using a text encoder _f_ text, and the
text representations are concatenated individually to each of the segments. To
contextualize information across several segments, we use a segment scoring
encoder-only transformer [ 37 ] _f_ trans with positional embeddings, that assigns
importance scores _Y_ _s_ _[′]_ _i_ [to each segment as shown in Eq.][ 1][. The network is trained]
using supervision from the importance scores of the pseudo summaries _Y_ _s_ _i_, using
Mean-Squared Error Loss as shown in Eq. 2.


_\_ _l_ [a] _[ b]_ [el {e] [q:ivsum }] [Y'_{] [s_i} &= f_{\] _[t e]_ [xt ] [{] _[t]_ _[r]_ [ans] _[ }][ }][ ( \]_ tex

_t_ {conc a t : \ l _e_ _t_ _[f]_ _[( f]_ _[_]_ _{_ [\text {text}}(\text {transcript}),f_{\text {vid}}(s_i))\right ) \; \forall \; i \in \mathcal {N} \\ \mathcal {L}_{\text {IV-Sum}} &= \sum _{i \in \mathcal {N}} \text {MSE}\: (Y'_{s_i}, Y_{s_i}) \label {eq:mse}] (2)

_} \_


During inference, we sort the segments based on the predicted scores and
assign the label 1 to the top _t_ % of the segments, and the label 0 to the remaining
ones. When a segment is assigned a label, all the frames in the segment also get
assigned the same label. The summary is constructed by stitching together all
the frames with label 1.


**4** **Instructional Video Summarization Datasets**


We describe the details of the data collection process for the annotations used
in our work — _Pseudo Summaries_ annotations for training and the _WikiHow_
_Summaries_ annotations for evaluation.

**Pseudo Summaries Training Dataset.** As described in Sec. 3.1, we use the
pseudo summary generation process for creating our training set. We use the
videos and task annotations from COIN [ 36 ] and CrossTask [ 45 ] datasets for
creating our training datasets.
**COIN:** COIN consists of 11K videos related to 180 tasks. As this is a dynamic
YouTube dataset, we were able to obtain 8,521 videos at the time of this work.
**Cross-Task:** CrossTask consists of 4,700 instructional videos (of which we were
able to access 3,675 videos) across 83 different tasks.
**Pseudo Summaries:** We combined the two datasets to create pseudo summaries
comprising of 12,160 videos, whilst using the videos that were common to both
datasets only once. They span 263 different tasks, have an average length of


8 M. Narasimhan et al.


Table 1: **Instructional Video Summarization Datasets Statistics.** _†_ Our _Wik-_
_iHow Summaries_ dataset was created automatically using a scalable pipeline, but
manually verified for correctness.


TVSum SumMe Pseudo Summaries WikiHow Summaries


Number of videos 50 25 12160 2106

Annotation Manual Manual Automatic Manually verified _†_
Number of Tasks/Categories 10 25 185 20
Total Input Duration (Hours) 3.5 1.0 628.53 42.94


3.09 minutes, and in total comprise of 628.53 hours of content. The summary
videos that were constructed using our pseudo ground-truth generation pipeline
are 1.71 minutes long on an average, with each summary being 60% of the
original video. While it is possible to construct pseudo summaries using the
step-localization annotations, we show in Sec. 5 that such summaries may miss
important steps or do not pick up on steps that are salient in the video. Moreover,
our pseudo summary generation mechanism is weakly-supervised, requiring only
task annotations and no step-localization annotations.

**WikiHow Summaries Dataset.** To provide a test bed for instructional video
summarization, we automatically create and manually verify _WikiHow Summaries_,
a video summarization dataset consisting of 2,106 input videos and summaries,
[where each video describes a unique task. Each article on the WikiHow Videos](https://www.wikihow.com/Video)
website consists of a main instructional video demonstrating a task that often
includes promotional content, clips of the instructor speaking to the camera with
no visual information of the task, and steps that are not crucial for performing
the task. Viewers who want an overview of the task would prefer a shorter
video without all of the aforementioned irrelevant information. The WikiHow
[articles (e.g., see How to Make Sushi Rice) contain exactly this: corresponding](https://www.wikihow.com/Make-Sushi-Rice)
text that contains all the important steps in the video listed with accompanying
images/clips illustrating the various steps in the task. These manually annotated
articles are a good source for automatically creating ground-truth summaries
for the main videos. We obtain the summaries and the corresponding labels and
importance scores using the following process (see supp. for an overview figure):
**1. Scraping WikiHow videos.** [We scrape the WikiHow Videos website for all](https://www.wikihow.com/Video)
the long instructional videos along with each step and the images/video clips
(GIFs) associated with the step.
**2. Localizing images/clips.** We automatically localize these images/clips in
the main video by finding the closest match in the video. To localize an image,
we compare ResNet50 [ 11 ] features of the image and to that of all the frames
in the video. The most similar frame is selected and this step is localized in the
input video to a 5 second window centered around the frame. If the step contains
a video clip/GIF, we localize the first frame of the video clip/GIF in the input
video by similarly comparing ResNet features, as above, and the localization is
set to be the length of the step video clip.


Summarizing Instructional Videos 9


**3. Ground-truth summary from localized clips.** We stitch the shorter
localized clips together to create the ground truth summary video. Consequently,
we assign labels to each frame in the input video, depending on whether it belongs
to the input summary (label 1) or not (label 0). To obtain importance scores, we
partition each input video into equally sized segments (same as in Sec. 3.2) and
compute the importance score for each segment to be the average of the labels
assigned to the individual frames in the segment.
**4. Manual verification.** We verified that the summaries are at least 30% of
the original video and manually fixed summaries that were extremely short/long.
_Online Longevity and Scalability._ We note that a common problem plaguing
YouTube datasets today is shrinkage of datasets as user uploaded videos are
taken down by users (eg. Kinetics [ 3 ]). WikiHow articles are less likely to be
taken down, and this is an actively growing resource as new How-To videos are
released and added (25% growth since we collected the data). Hence there is a
potential to continually increase the size of the dataset.
For each video, we provide the following: (i) frame-level binary labels (ii)
the summary formed by combining the frames with label 1 (iii) segment-level
importance scores between 0 and 1, which are computed as an average of the
importance scores for all the frames in the segment (iv) the localization of the
visual steps in the video (i.e. the frames associated with each step). We also
scrape natural language descriptions of each step as a bonus that could be useful
for future work. We divide our WikiHow dataset into 768 validation and 1,339
test videos. Tab. 1 shows the statistics of both our datasets. Both datasets are

much larger in size compared to existing generic video summarization datasets,
contain a broader range of tasks, and are scalable.


**5** **Experiments**


Next, we describe the experimental setup and evaluation for instructional video
summarization. We compare our method to several baselines, including CLIPIt [22], the state-of-the-art on generic and query-focused video summarization.
**Implementation Details.** For the video and text encoders, we use an S3D [ 39 ]
network, initialized with weights from pre-training on HowTo100M [ 21 ] using the
MIL-NCE loss [ 20 ]. We fine-tune the _mixed_ ~~_5_~~ _*_ layers and freeze the rest. The
segment scoring transformer is an encoder consisting of 24 layers and 8 heads and
is initialized randomly. The network is trained using the Adam optimizer [ 16 ],
with learning rate of 0.01, and a batch size of 24. We use Distributed Data
Parallel to train for 300 epochs across 8 NVIDIA RTX 2080 GPUs. Additional
implementation details are mentioned in supplemental.
**Metrics.** To evaluate instructional video summaries, we follow the evaluation
protocol used in past video summarization works [ 41, 25, 22 ] and report Precision,
Recall and F-Score values. As described in Sec. 4, each video in the _WikiHow_
_Summaries_ dataset contains the ground-truth labels _Y_ _l_ (binary labels for each
frame in the video) and the ground-truth scores _Y_ _s_ (importance scores in the
range [0-1] for each segment in the video). We compare the binary labels predicted


10 M. Narasimhan et al.


Table 2: **Instructional Video Summarization results on** _**WikiHow Summaries.**_

We compare F-Score, Kendall and Spearman correlation metrics of our method IV-Sum,
to all the baselines. Our method achieves state-of-the-art on all three metrics.


F-Score _τ_ [ 15 ] _ρ_ [ 46 ]
Method


Val Test Test Test


ASR RGB Pseudo


Frame Cross-Modal Similarity ✓ ✓ - 52.8 53.1 0.022 0.051
Segment Cross-Modal Similarity ✓ ✓ - 55.1 55.5 0.034 0.060
Step Cross-Modal Similarity ✓ ✓ - 57.9 58.3 0.037 0.061
CLIP-It with captions [22] - ✓ - 22.5 22.1 0.036 0.064
CLIP-It with ASR [22] ✓ ✓ - 27.9 27.2 0.055 0.088


CLIP-It with ASR ✓ ✓ ✓ 62.5 61.8 0.093 0.191

IV-Sum without ASR - ✓ ✓ 65.8 65.2 0.095 0.202

**IV-Sum** ✓ ✓ ✓ **67.9** **67.3** **0.101 0.212**


for the frames in the video _Y_ _l_ _[′]_ [, to the ground truth labels] _[ Y]_ _[l]_ [, and measure F-Score,]
Precision and Recall, as defined in prior summarization works [30,29].
While these scores assess the quality of the predicted frame-wise binary labels,
to assess the quality of the predicted segment-wise importance scores _Y_ _s_ _[′]_ [, we]
follow Otani _et al_ . [ 24 ], and report results on the rank-based metrics Kendall’s
_τ_ [ 15 ] and Spearman’s _ρ_ [ 46 ] correlation coefficients. We first rank the video
frames according to the generated importance scores _Y_ _s_ _[′]_ [and the ground-truth]
importance scores _Y_ _s_ . We then compare the generated ranking to each groundtruth ranking of video segments for each video obtained from the frame-wise
binary labels as described in Sec. 4. The final correlation score is computed by
averaging over the individual scores for each video.
**Baselines.** We compare our method to the state-of-the-art video summarization
model CLIP-It [ 22 ]. To validate the need for pseudo summaries, we construct
three unsupervised baselines as alternatives to our pseudo summary generation
algorithm. We first describe the three unsupervised baselines.
**Frame Cross-Modal Similarity.** We sample frames (at the same FPS used by
our method) from an input video and compute the similarity between CLIP (ViTB/32) [ 26 ] frame embeddings and CLIP text embeddings of each sentence in the
transcript. The embeddings do not encode temporal information but leverage the
priors learned by the CLIP model. Based on the scores assigned to each frame,
we threshold _t_ % of the higher scoring frames to be part of the summary. Frame
scores are propagated to the segments they belong to, and the summary is a
compilation of the chosen segments.
**Segment Cross-Modal Similarity.** We uniformly divide the video into segments and compute MIL-NCE [ 20 ] video features for each segment. We embed
each sentence in the transcript to the same feature space using the MIL-NCE
text encoder. We compute the pairwise similarity between all video segments
and the sentences, and average over sentences to obtain a score for each segment.


Summarizing Instructional Videos 11



Fold into a triangle Fold triangle again Make valley fold and



Ground Truth Get paper square Fold into a triangle Fold triangle again Paper Claw



tuck tip into picket



IV-Sum


CLIP-It with ASR


(1) Make an Origami Paper Claw


Ground Truth Add salt and stir egg yolks Add egg and cook Add cheese Fold Omelette


IV-Sum


Step Cross-Modal

Similarity


(2) Make a Fluffy Omelette


Fig. 4: **Qualitative comparisons to baselines.** We show the steps in the groundtruth as text (note we never train with step descriptions, these are shown here simply
for illustrative purposes) and compare frames selected in summaries generated by our
method IV-Sum, CLIP-It with ASR, and Step Cross-Modal Similarity. In (1), CLIP-It
misses steps which are deemed important by our method ( _“Fold into a triangle”_ ) and
assigns higher scores to less salient frames for the step ( _“Make valley fold and tuck tip_
_into picket”_ ) where neither the valley fold nor the picket are clearly visible. In (2), Step
Cross-Modal Similarity misses ( _“Add egg and cook”_ ) and selects too many redundant
frames for the step ( _“Add salt and stir egg yolks”_ ).


Our intuition is that since demonstrators typically describe the important steps
shortly before, after or while performing them, a high similarity between the
visuals and transcripts would directly correlate with the significance of the step.
We filter _t_ % of the highest scoring segments, where _t_ is determined heuristically
using the _WikiHow Summaries_ validation set and is consistent across all baselines
and our model. The filtered segments are stitched together to form the summary.

**Step Cross-Modal Similarity.** We first group segments into steps and then
compare them to the ASR transcripts. For this we employ the technique described
in Sec. 3.1, i.e. we extract MIL-NCE features for the video segments and group
them together based on their similarity to form steps. [2] The embedding for a step
is set to be the average of all the segment embeddings in it. If a step is similar to
the transcript, all the segments in that step are chosen to be part of the summary.
This baseline is the closest to our pseudo summary generation algorithm.

Next, we describe the CLIP-It baseline and ablations, trained with supervision.
**CLIP-It with captions.** We evaluate CLIP-It [ 22 ] trained on TVSum [ 35 ],
SumMe [ 9 ], OVP [ 23 ], and YouTube [ 5 ] against our _WikiHow Summaries_ . We
use the same protocol as in CLIP-It for evaluation and describe further details
in supplemental. For language-conditioning, we follow CLIP-It and generate
captions for the _WikiHow Summaries_ dataset using BMT [ 13 ]; we feed these as
input to the CLIP-It model.


2 Since we process a single input video (not multiple videos per task), we can not use
the Task Relevance component.


12 M. Narasimhan et al.

|(1) Draw a Cow|Col2|Col3|Col4|
|---|---|---|---|
|(1) Draw a Cow||(1) Draw a Cow||



(2) Dutch braid your hair


Fig. 5: **Qualitative results.** We show summaries from our method IV-Sum along with
the predicted importance scores. The green and red arrows point to frames that were
assigned a high and low scores, respectively. Our model correctly assigns higher scores
to frames from all the steps that are relevant and lower scores to frames which aren’t
crucial to the task (as in (1)) and frames which don’t belong to a step (as in (2)).


**CLIP-It with ASR transcripts.** We evaluate the same CLIP-It model above
by replacing captions with ASR transcripts, so as to allow for a fair comparison
with the baselines and our method, IV-Sum which use ASR transcripts.

**CLIP-It with ASR transcripts trained on Pseudo Summaries.** We train
CLIP-It from scratch on our Pseudo-GT Summaries dataset using ASR transcripts
from the videos in place of captions.

**Quantitative Results.** We compare the baselines to the two versions of IVSum, one with ASR transcripts and another without. To train IV-Sum without
transcripts, we simply eliminate the text encoder ( _f_ _text_ ) in Eq. 1 and pass only
the visual embeddings of the individual segments to the transformer. We report
F-Score, Kendall’s _τ_ and Spearman’s _ρ_ coefficients in Tab. 9. As seen, IV-Sum
(both with and without ASR transcripts), outperforms all the baselines on all
metrics. Particularly, we achieve notable improvements on the correlation metrics
that compare the saliency scores, attesting to our model’s capabilities to assign
higher scores to segments that are more relevant. We also observe that CLIP-It
trained using the pseudo summaries generated by our method has a strong boost
in performance compared to CLIP-It trained on generic video summarization
datasets, reinforcing the effectiveness of our pseudo summaries for training. The
best method among the unsupervised ones is Step Cross-Modal Similarity, a
“reduced” version of our pseudo summary generation method.

**Qualitative Results.** We present qualitative results in Fig. 4. We show frames
in the summaries generated by our method IV-Sum, CLIP-It with ASR tran

Step Localization Summary


Our Pseudo Summary


Step Localization Summary


Our Pseudo Summary



Summarizing Instructional Videos 13


(1) Make Latte


(2) Make a Taco Salad



Fig. 6: **Pseudo summaries vs step-localization annotations.** We compare frames
in our automated pseudo summary to the step localization manual annotations, aligned
temporally. Frames corresponding to steps that are identified by our method but missed
by step localization are highlighted in yellow.


scripts (trained on generic video summarization datasets), and Step Cross-Modal
Similarity. We also list the steps in the ground-truth as text (for illustrative
purposes). In Fig. 4 (1), CLIP-It misses the step _“Fold into a triangle”_, as it
optimizes for diversity among the frames and was trained on a small dataset that
does not generalize well to our domain. It also picks the less salient frames for
the step _“Make valley fold and tuck tip into picket”_, whereas our model correctly
identifies all the steps and assigns higher scores to the more salient frames. The
summary from the Step Cross-Modal Similarity baseline, shown in Fig. 4 (2),
assigns high scores to several redundant frames ( _“Add salt and stir egg yolks”_ ),
but misses “Add egg and cook”.
Fig. 5 shows results from our method along with the predicted frame-wise
importance scores. The green and red arrows point to frames that are assigned
the highest and lowest scores by our method, respectively. As seen, our method
assigns high scores to frames in task relevant and salient steps and low scores to
frames which aren’t crucial to the step, like in Fig. 5 (1), or do not belong to a
step, like in Fig. 5 (2) where the person is talking to the camera.
**Ablations.** We compare different approaches to generate pseudo summaries
for training our instructional video summarizer network – (i) First, we ablate
the two objectives, Task Relevance and Cross-Modal Saliency, used to generate
the pseudo summaries. (ii) Next, we replace the annotations from our pseudo
summary generation pipeline with step localization annotations. We include
model and loss ablations in the supplemental.
_(i) Ablating Objectives._ We ablate the two objectives, Task Relevance and CrossModal Saliency, used for generating pseudo summaries, in Tab. 3a. We train
IV-Sum on different versions of pseudo summaries and report F-Scores on the
_WikiHow Summaries_ validation set. Combining both objectives is more effective
than using each objective individually.
_(ii) Using Step Localization Annotations._ COIN and CrossTask datasets contain
temporal localization annotations of a generic set of steps pertaining to the task in


14 M. Narasimhan et al.


Table 3: **Pseudo Summary Variations.** We report results on two variations of
generating the pseudo summaries: (i) ablating the objectives (ii) using step localization
annotations to generate pseudo summaries.



(a) **Ablating objectives.** We ablate the
two objectives in our pseudo summary generation pipeline.


**Method** **F-Score**


Task-Consistency only 64.1
Cross-Modal Similarity only 61.0
Both 67.9



(b) **Using Step-Localization Annota-**
**tions.** We compare pseudo summaries
from step-localization annotations with
our approach.


Method F-Score
IV-Sum (Step Localization) 57.6
IV-Sum (Ours) 66.8



the videos. We use these annotations to extract the visual segments corresponding
to the steps and concatenate them to form a summary. We assign binary labels to
each frame, depending on whether they belong in the summary or not. We then
use these step-localization summaries as supervision to train our model, IV-Sum
with a weighted-CE loss [ 22 ] as this works best for binary labels. In Tab. 3b,
we compare this to IV-Sum trained on pseudo summaries generated using our
pipeline and report F-Scores on our _WikiHow Summaries_ validation set. As seen,
IV-Sum trained on our generated summaries outperforms IV-Sum trained using
step-localization summaries. We qualitatively compare our automatic pseudo
summaries to the manually labeled step localization annotations in Fig. 6. Often
the step annotations only cover a few steps and miss other crucial steps as shown
in yellow in (1). In (2), we observe that our pseudo summary retrieves steps that
are unique to the task which the step localization annotation doesn’t include.


**6** **Conclusion**


We introduce a novel approach for generating visual summaries of instructional
videos — a practical task with broad applications. Specifically, we overcome the
need to manually label data in two important ways. For training, we propose
a weakly-supervised method to create pseudo summaries for a large number
of instructional videos. For evaluation, we leverage WikiHow (its videos and
step illustrations) to automatically build a _WikiHow Summaries_ dataset. We
manually verify that the obtained summaries are of high quality. We also propose
an effective model to tackle instructional video summarization, IV-Sum, that
uses temporal 3D CNN representations, unlike most prior work that relies on
frame-level representations. We demonstrate that all components of the proposed
approach are effective in a comprehensive ablation study.
**Acknowledgements:** We thank Daniel Fried and Bryan Seybold for valuable
discussions and feedback on the draft. This work was supported in part by DoD
including DARPA’s LwLL, PTG and/or SemaFor programs, as well as BAIR’s
industrial alliance programs.


Summarizing Instructional Videos 15

## **Supplementary Materials**

**TL;DW? Summarizing Instructional Videos with Task Relevance &**
**Cross-Modal Saliency**


This section is organised as follows:


1. _WikiHow Summaries_ Data Collection

2. Implementation Details
3. Additional Results

(a) Results on instructional videos in generic video summarization datasets
(b) Step recall

(c) Model architecture ablations
4. Additional Qualitative Results

(a) Qualitative comparison of ground-truth, IV-Sum, CLIP-It, and Step AV
(b) Pseudo summary vs IV-Sum summary

(c) Pseudo summary vs step-localization annotations


Main Video Frames


Localize steps in video



Scrape WikiHow videos and steps



Stitch localized clips to form summary



Fig. 7: _**WikiHow Summaries**_ **Data Collection.** We first scrape all the main videos
in the WikiHow articles, along with the images or video clips assosciated with each
step. Next, the image/clip corresponding to each step is localized in the main video.
The images are localized to _±_ 2.5 seconds(i.e. a 5 seconds window centered around the
image). The localized clips are stitched together to form the summary.


**7** _**WikiHow Summaries**_ **Data Collection**


We provide more details on the WikiHow Summaries data collection process.
As described in Sec. 4.2 of the main paper, these are the main stages of the
dataset creation: (1) Scraping WikiHow videos (2) Localizing images/clips in
video (3) Ground-truth summary from localized clips (4) Manual verification.
Fig. 7 illustrates our data collection process. We show an example for the article
_[“Prepare Tofu”](https://www.wikihow.com/Prepare-Tofu)_ . We localize each of the individual steps (images/clip) in the main
video by comparing the ResNet features and obtain short localized clips. The
clips are stitched together to form the summary. A handful of summaries with
spurious lengths (too long or too short) are manually verified and corrected.


16 M. Narasimhan et al.


We describe how we handle some edge cases in the articles, and the reasoning
behind using ResNet features in stage (2).
**Multiple methods.** Sometimes the articles contain multiple methods of performing a task. If the video also contains multiple methods, as in this _[“Draw](https://www.wikihow.com/Draw-a-Cow)_
_[a cow”](https://www.wikihow.com/Draw-a-Cow)_ example, we localize each method in the video, and the summary is a
compilation of all methods. The reasoning behind doing this is that users looking
for a specific way of drawing a cow can take a quick glimpse of the summary and
decide if they want to watch the whole video. However, if the article contains
multiple methods but the video only contains one, as in _[this](https://www.wikihow.com/Make-an-Envelope)_ example, only the
method depicted in the video is added in the summary.
**Reason for using ResNet features instead of direct pixel comparison.**
As described in the main paper Sec 4.2, we compare ResNet features to localize
the images/clips of the steps in the main video. The reason we compare ResNet
features and not pixel values directly is because the images/clips associated with
the steps aren’t always extracted from the main video. For example, in this article
on _[“Making a pinwheel”](https://www.wikihow.com/Make-a-Pinwheel)_, the frames in the images/clips are from a different video
and don’t have exact matches in the _[main video](https://youtu.be/J4_Wuq_VmOY)_ for the article. Using ResNet
features in place of pixels makes the localization robust to color/background
changes, allowing us to localize steps despite an exact match of frames.


Table 4: Hyperparameters for training IV-Sum.


**Hyperparamter** **Value**


Batch size 24

Epochs 300
Learning rate IV-Sum 1e-3
Learning rate S3D fine-tuning 1e-4
Weight decay 1e-4
Dropout 0.1
Learning rate decay StepLR
_t_ % 55%

#frames per segment 32
#frames per video during training 768
# Training FPS 8


**8** **Implementation Details**


**Video processing.** For generating pseudo summaries and for training IV-Sum,
the videos are down-sampled to 8 FPS, and divided into non-overlapping segments
of size 32 frames, which is the recommended segment size for MIL-NCE [ 20 ] [3] .


3 We use the implementation of MIL-NCE available here `[https://github.com/](https://github.com/antoine77340/MIL-NCE_HowTo100M)`
```
 antoine77340/MIL-NCE_HowTo100M

```

Summarizing Instructional Videos 17


While training IV-Sum, we fix the number of segments sampled from a video
to be 28 (i.e. 896 frames) which are selected as a contiguous sequence from a
randomly chosen start location. If the video is shorter in duration, it is padded
with zeros. During inference, we retain the original fps of the video and all the
segments are passed to IV-Sum. For concatenating the text representations to
the visual representations, we follow the approach in MIL-NCE and map each
visual clip to the sentences a few seconds before, after, and during the clip. The
text embedding is an average of all the sentence embeddings.
**Hyperparameters.** Tab. 4 shows detailed list of hyperparameters. For all
baselines and our method, to ensure a fair comparison we generate the summary
from scores by selecting the top _t_ % of the highest scoring segments to be in the
summary. _t_ is set to be 55 based on the statistics in the validation set of WikiHow
Summaries, where on average 55% of the original video appears in the summary.
**Dimensions.** We first describe the dimensions of each of the embeddings. The
image embeddings are in _f_ vid ( _s_ _i_ ) _∈_ R [512] . The text embeddings for _M_ transcript
sentences using _f_ text are in R _[M]_ _[×]_ [512] which are then fused using a 2 layer perceptron
to R [512] . _M_ is set to be the maximum number of sentences found in any ASR
transcript. The image and text embeddings are concatenated and passed to the
segment scoring transformer _f_ trans, the output dimension of this is in R [512] .
**Computation resources.** The training time is approximately 2 days using
Distributed Data Parallel to train for 300 epochs on 8 NVIDIA RTX 2080 GPUs.
The model inference time for a single video at its original fps is 1.5 minutes on

average.


**9** **Additional Results**


Table 5: **Evaluating on generic video summarization datasets.** We compare
F-Score of IV-Sum and CLIP-It on the instructional videos in TVSum.


**Method** **F-Score**


CLIP-It [22] 0.72
IV-Sum 0.73


**Evaluating on instructional videos in generic video summarization**
**datasets.** Here, we consider the existing generic video summarization datasets,
in particular, those videos that fall under “instructional” domain, in order to
validate our model further. Generic video summarization dataset TVSum [ 35 ] has
15 videos pertaining to the categories _changing a car tire_, _getting a car unstuck_,
and _making a sandwich_ while SumMe [ 9 ] has no instructional videos. We follow
the evaluation protocol described in CLIP-It [ 22 ]. For a fair comparison to CLIPIt [ 22 ], we curate a test set by randomly selecting 7 of these 15 videos, while the
remaining 8 are added to the training set, so as to ensure that the CLIP-It model


18 M. Narasimhan et al.


sees instructional videos during training. The augmented training set is curated
by combining the 8 videos with those in SumMe (25 videos), TVSum (45 videos),
OVP [ 23 ] (50 videos), and YouTube [ 5 ] (39 videos). CLIP-It is trained on this
augmented training set consisting of 168 videos (including 8 instructional videos)
and evaluated on the held out 7 instructional videos. Our method is trained on
pseudo summaries (built on top of CrossTask and COIN) and evaluated in a
_zero-shot_ way on the test set of 7 videos.
As seen in Tab. 5, our method IV-Sum, although trained with noisy / weakly
labeled pseudo summaries from a different data distribution, achieves an F-Score
comparable to the CLIP-It [22], trained on human annotated summaries.


Table 6: **Comparing step-recall.** We report step-recall on our method and 2 baselines.


**Method** **Step-recall**


Step Cross-Modal Similarity 0.68
CLIP-It with ASR 0.70

IV-Sum 0.94


**Step recall.** We define an additional metric, _step-recall_ to be the average
percentage of steps present in the ground-truth summary which were successfully
picked by the generated summary. Our _WikHow Summaries_ dataset contains
annotations of frames pertaining to each step, and if any of the frames from a
step are present in the summary, we assume the step is covered. Using this logic,
we generate a list of steps in the generated summary _Y_ step _[′]_ [, and a list of steps in]
the ground-truth _Y_ step . We compute step-recall as follows,


Step-recall = [overlap between] _[ Y]_ [step] [ and] _[ Y]_ ste _[ ′]_ p
total duration of _Y_ step


In Tab. 6 we report the step-recall for Step Cross-Modal Similarity, CLIP-It
with ASR (trained on generic video summarization datasets) and IV-Sum. Both
Step Cross-Modal Similarity and CLIP-It with ASR baselines miss 30% of steps
found in the ground truth summary while our method on average only misses
6% of the steps.
**Loss Ablations.** In Table 8, we explore additional loss functions as in prior
video summarization works [ 30, 29, 25, 22 ]. Diversity loss ensures diversity among
the summary segments and the reconstruction loss enforces similarity in representations of the reconstructed summary and the input video. Adding diversity
reduced the recall and we notice no improvement on adding reconstruction loss.
We believe this may be because frames corresponding to different steps are not
always diverse but are still important for the summary.
**Model ablations.** Table 7a shows the performance comparisons between freezing
the video and text encoding backbone (S3D) vs. fine-tuning part of the network.
In Table 7b, we ablate the segment scoring transformer (SST) in our model and


Summarizing Instructional Videos 19


Table 7: **Instructional Video Summarizer Ablations.** We perform ablations on
different components of the video summarizer network and report results on the
_WikiHowTo Summaries_ validation set.



(a) **IVSum S3D backbone Ablations.**
We compare fixing the pre-trained weights
of the S3D model to fine-tuning a part of

it.


**Method** **F-Score**


S3D fixed 65.8
S3D fine-tuned 67.9



(b) **IVSum Segment Scoring Trans-**
**former Ablations.** We compare different
architecture configurations of the segment
scoring transformer.


**Method** **F-Score**


#heads #layers
SST 8 16 63.1

SST 16 6 63.5

SST 8 12 66.7

SST 8 24 67.9

MLP - - 32.1



Table 8: **Loss Ablations.** We ablate different losses in the IV-Sum model and show
results on validation set of _WikiHow Summaries_


**Method** **F-Score Recall**


MSE 67.9 84.5

MSE + Diversity 61.2 63.4
MSE + Reconstruction 67.6. 85.8


change the number of encoder layers, heads, and also replace the transformer with
an MLP. We report the F-Score on the validation set of _WikiHowTo Summaries_ .

For a fair comparison of MIL-NCE vs CLIP features, we retrain our IV-Sum
model replacing video segments with frames and replacing MIL-NCE features
with CLIP image and text features (same as the ones used in the CLIP-It baseline).
We report results in Tab. 9. We see that IV-Sum with CLIP performs at par
with CLIP-It with ASR but falls short of IV-Sum, indicating the need to use
video segments and MIL-NCE features pre-trained on HowTo100M.


Table 9: **Instructional Video Summarization results on** _**WikiHow Summaries.**_

All models were trained on pseudo summaries.


F-Score _τ_ (Kendall) _ρ_ (Spearman)
Method


Val Test Test Test


CLIP-It with ASR 62.5 61.8 0.093 0.191

IV-Sum with CLIP 61.8 62.0 0.094 0.201

**IV-Sum** **67.9 67.3** **0.101** **0.212**


20 M. Narasimhan et al.


**10** **Additional Quantitative Results**


[Please watch the video on our website for qualitative results.](https://medhini.github.io/ivsum/)
**Comparison to baselines.** We show video results comparing the ground-truth
summary to that from IV-Sum (our method) and baselines Step Cross-Modal
Similarity and CLIP-It with ASR trained on generic video summarization datasets.
Our method picks all frames in the ground-truth and assigns high scores to
salient frames. Step Cross-Modal Similarity misses the crucial step _“fold and_
_tuck”_ at the end as it assigns higher scores to irrelevant frames at the start
of the video. This is because it has no knowledge of task-relevance. CLIPIt with ASR (trained on generic video summarization datasets) misses steps
(like _“fold into a triangle”_ ) and assigns lower scores to the key frames in a
step as it optimizes for diversity. **Evaluating Pseudo Summary generation**
**procedure for WikiHow Summaries.** We found that there are 15 tasks
which are shared between the Pseudo Summary training set and the WikiHow
Summaries test set. We applied the method used to construct pseudo summaries
to these 15 task videos in the WikiHow Summaries by fetching videos of the
same task from our training set. We compare this to IV-Sum and report results
in Tab 10. We notice a slight improvement on all three metrics, indicating that
our model is able to learn above the noise in the pseudo summaries.


Table 10: **Evaluating pseudo summary generation on subset of WikiHow Sum-**
**maries**


F-Score _τ_ _ρ_
Method


Test Test Test


Pseudo Summary Generation 38.0 0.03 0.36
**IV-Sum** **42.0** **0.04 0.38**


**Pseudo summary vs IV-Sum summary.** IV-Sum is trained on weakly labeled
pseudo summaries that may sometimes be noisy. However, since the training loss
is not 0, we check if our model learns despite the noise and produces summaries
of a better quality. In this example, we show summaries for _[“this”](https://www.youtube.com/watch?v=9krpJsOi3dE)_ video from the
Pseudo Summaries dataset. As seen the pseudo summary contains an irrelevant
segment where results from a web search are shown in Korean for nearly 10
seconds (9 [th] second to the 19 [th] second). The IV-Sum model trained on pseudo
summaries yields a resulting summary without this segment, as it is able to learn
_“task-relevance”_ and _“cross-modal saliency”_ .
**Pseudo summary vs step-localization annotation.** We compare pseudo
summaries generated using our method to the step-localization summary. In the
example _“Make a pumpkin spice latte”_, the input video can be found _[here](https://www.youtube.com/watch?v=9krpJsOi3dE)_ . Step
localization only localizes two main steps, _“boil milk”_ and _“add coffee and blend”_
whereas our pseudo summary contains all the main steps necessary to do the task.


Summarizing Instructional Videos 21


**Failure case.** Since we always select the top 55% of the segments to be in
the summary (i.e. t=55%), the summary chosen by our method is sometimes
much longer/shorter than the ground-truth summary. This is a failure case of
the baseline methods as well. For example, for this 38 second video on _[“How to](https://www.youtube.com/watch?v=HX1okLOSRz8)_
_[ripen a cantaloupe”](https://www.youtube.com/watch?v=HX1okLOSRz8)_, the ground-truth summary is a brief 15 seconds whereas our
summary covers the steps in more detail and is 21 seconds long.


22 M. Narasimhan et al.


**References**


1. Alayrac, J.B., Bojanowski, P., Agrawal, N., Sivic, J., Laptev, I., Lacoste-Julien, S.:
Unsupervised learning from narrated instruction videos. In: IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) (2016) 4
2. Bojanowski, P., Lajugie, R., Bach, F., Laptev, I., Ponce, J., Schmid, C., Sivic,
J.: Weakly supervised action labeling in videos under ordering constraints. In:
European Conference on Computer Vision (ECCV) (2014) 4
3. Carreira, J., Zisserman, A.: Quo vadis, action recognition? a new model and the
kinetics dataset. IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) (2017) 9
4. Chang, C.Y., Huang, D.A., Sui, Y., Fei-Fei, L., Niebles, J.C.: D3TW: Discriminative
differentiable dynamic time warping for weakly supervised action alignment and
segmentation. In: IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) (2019) 4
5. De Avila, S.E.F., Lopes, A.P.B., da Luz Jr, A., de Albuquerque Ara´ujo, A.: Vsumm:
A mechanism designed to produce static video summaries and a novel evaluation
method. Patt. Rec. Letters (2011) 11, 18
6. Ding, L., Xu, C.: Weakly-supervised action segmentation with iterative soft boundary assignment. In: IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) (2018) 4
7. Fajtl, J., Sokeh, H.S., Argyriou, V., Monekosso, D., Remagnino, P.: Summarizing
videos with attention. Asian Conference on Computer Vision (ACCV) (2018) 4
8. Fried, D., Alayrac, J.B., Blunsom, P., Dyer, C., Clark, S., Nematzadeh, A.: Learning
to segment actions from observation and narration. In: Association for Computational Linguistics (2020) 4
9. Gygli, M., Grabner, H., Riemenschneider, H., Gool, L.V.: Creating summaries from
user videos. European Conference on Computer Vision (ECCV) (2014) 2, 4, 11, 17
10. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016) 3
11. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition.
In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016)
8

12. Huang, D.A., Fei-Fei, L., Niebles, J.C.: Connectionist temporal modeling for weakly
supervised action labeling. In: European Conference on Computer Vision (ECCV)
(2016) 4
13. Iashin, V., Rahtu, E.: A better use of audio-visual cues: Dense video captioning
with bi-modal transformer. British Machine Vision Conference (BMVC) (2020) 11
14. Kanehira, A., Gool, L.V., Ushiku, Y., Harada, T.: Viewpoint-aware video summarization. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
(2018) 4
15. Kendall, M.G.: The treatment of ties in ranking problems. Biometrika **33** (3),
239–251 (1945) 3, 10
16. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. International
Conference on Learning Representations (ICLR) (2015) 9
17. Kuehne, H., Richard, A., Gall, J.: Weakly supervised learning of actions from
transcripts. In: CVIU (2017) 4
18. Kukleva, A., Kuehne, H., Sener, F., Gall, J.: Unsupervised learning of action classes
with continuous temporal embedding. In: IEEE Conference on Computer Vision
and Pattern Recognition (CVPR) (2019) 4


Summarizing Instructional Videos 23


19. Mahasseni, B., Lam, M., Todorovic, S.: Unsupervised video summarization with
adversarial lstm networks. IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) (2017) 3
20. Miech, A., Alayrac, J.B., Smaira, L., Laptev, I., Sivic, J., Zisserman, A.: End-to-end
learning of visual representations from uncurated instructional videos. In: IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) (2020) 5, 6, 9,
10, 16
21. Miech, A., Zhukov, D., Alayrac, J.B., Tapaswi, M., Laptev, I., Sivic, J.: Howto100m:
Learning a text-video embedding by watching hundred million narrated video clips.
In: IEEE International Conference on Computer Vision (ICCV) (2019) 2, 9
22. Narasimhan, M., Rohrbach, A., Darrell, T.: Clip-it! language-guided video summarization. Advances in Neural Information Processing Systems (NeurIPS) (2021) 2,
3, 4, 7, 9, 10, 11, 14, 17, 18
23. Open video project. `[https://open-video.org/](https://open-video.org/)` 11, 18
24. Otani, M., Nakashima, Y., Rahtu, E., Heikkil¨a, J.: Rethinking the evaluation of
video summaries. IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) (2019) 10
25. Park, J., Lee, J., Kim, I.J., Sohn, K.: Sumgraph: Video summarization via recursive
graph modeling. European Conference on Computer Vision (ECCV) (2020) 3, 4, 7,
9, 18
26. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from
natural language supervision. arXiv preprint arXiv:2103.00020 (2021) 10
27. Richard, A., Kuehne, H., Gall, J.: Weakly supervised action learning with RNN
based fine-to-coarse modeling. In: IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) (2017) 4
28. Richard, A., Kuehne, H., Gall, J.: Action sets: Weakly supervised action segmentation without ordering constraints. In: IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) (2018) 4
29. Rochan, M., Wang, Y.: Video summarization by learning from unpaired data. IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) (2019) 3, 10, 18
30. Rochan, M., Ye, L., Wang, Y.: Video summarization using fully convolutional
sequence networks. European Conference on Computer Vision (ECCV) (2018) 10,
18
31. Sener, F., Yao, A.: Unsupervised learning and segmentation of complex activities
from video. In: IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) (2018) 4
32. Sener, O., Zamir, A.R., Savarese, S., Saxena, A.: Unsupervised semantic parsing of
video collections. In: IEEE International Conference on Computer Vision (ICCV)
(2015) 4
33. Sharghi, A., Gong, B., Shah, M.: Query-focused extractive video summarization.
European Conference on Computer Vision (ECCV) (2016) 4
34. Sharghi, A., Laurel, J.S., Gong, B.: Query-focused video summarization: Dataset,
evaluation, and a memory network based approach. IEEE Conference on Computer
Vision and Pattern Recognition (CVPR) (2017) 2, 4
35. Song, Y., Vallmitjana, J., Stent, A., Jaimes, A.: Tvsum: Summarizing web videos
using titles. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
(2015) 2, 4, 11, 17
36. Tang, Y., Ding, D., Rao, Y., Zheng, Y., Zhang, D., Zhao, L., Lu, J., Zhou, J.:
Coin: A large-scale dataset for comprehensive instructional video analysis. In: IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) (2019) 2, 4, 7


24 M. Narasimhan et al.


37. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser,
�L ., Polosukhin, I.: Attention is all you need. In: Proceedings of the 31st International
Conference on Neural Information Processing Systems. pp. 6000–6010 (2017) 7
38. Wei, H., Ni, B., Yan, Y., Yu, H., Yang, X., Yao, C.: Video summarization via
semantic attended networks. The Association for the Advancement of Artificial
Intelligence Conference (AAAI) (2018) 4
39. Xie, S., Sun, C., Huang, J., Tu, Z., Murphy, K.: Rethinking spatiotemporal feature
learning: Speed-accuracy trade-offs in video classification. In: European Conference
on Computer Vision (ECCV) (2018) 9
40. Yuan, L., Tay, F.E., Li, P., Zhou, L., Feng, J.: Cycle-sum: Cycle-consistent adversarial lstm networks for unsupervised video summarization. The Association for
the Advancement of Artificial Intelligence Conference (AAAI) (2019) 3
41. Zhang, K., Chao, W.L., Sha, F., Grauman, K.: Summary transfer: Examplar-based
subset selection for video summarization. IEEE Conference on Computer Vision
and Pattern Recognition (CVPR) (2016) 9
42. Zhang, K., Chao, W.L., Sha, F., Grauman, K.: Video summarization with long
short-term memory. European Conference on Computer Vision (ECCV) (2016) 3
43. Zhang, K., Grauman, K., Sha, F.: Retrospective encoders for video summarization.
European Conference on Computer Vision (ECCV) (2018) 3
44. Zhao, B., Li, X., Lu, X.: Hsa-rnn: Hierarchical structure-adaptive rnn for video
summarization. In: IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) (2018) 3
45. Zhukov, D., Alayrac, J.B., Cinbis, R.G., Fouhey, D., Laptev, I., Sivic, J.: Cross-task
weakly supervised learning from instructional videos. In: IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) (2019) 2, 4, 7
46. Zwillinger, D., Kokoska, S.: Crc standard probability and statistics tables and
formulae. CRC Press (1999) 3, 10


