## **PreMind : Multi-Agent Video Understanding for Advanced Indexing of** **Pre sentation-style Videos**

**Kangda Wei** **[‡]** [*] **, Zhengyu Zhou, Bingqing Wang, Jun Araki,**
**Lukas Lange**, **Ruihong Huang** **[‡]**, **Zhe Feng**
Bosch Research North America & Bosch Center for Artificial Intelligence (BCAI)

    - Department of Computer Science and Engineering, Texas A&M University
kangda@tamu.edu, {Zhengyu.Zhou2, Bingqing.Wang, Jun.Araki}@us.bosch.com,

Lukas.Lange@de.bosch.com, huangrh@cse.tamu.edu, Zhe.Feng2@us.bosch.com



**Abstract**


In recent years, online lecture videos have
become an increasingly popular resource for
acquiring new knowledge. Systems capable
of effectively understanding/indexing lecture
videos are thus highly desirable, enabling downstream tasks like question answering to help
users efficiently locate specific information
within videos. This work proposes `PreMind`, a
novel multi-agent multimodal framework that
leverages various large models for advanced
understanding/indexing of presentation-style
videos. `PreMind` first segments videos into
slide-presentation segments using a VisionLanguage Model (VLM) to enhance modern
shot-detection techniques. Each segment is
then analyzed to generate multimodal indexes
through three key steps: (1) extracting slide
visual content, (2) transcribing speech narratives, and (3) consolidating these visual and
speech contents into an integrated understanding. Three innovative mechanisms are also
proposed to improve performance: leveraging
prior lecture knowledge to refine visual understanding, detecting/correcting speech transcription errors using a VLM, and utilizing a critic
agent for dynamic iterative self-reflection in vision analysis. Compared to traditional video
indexing methods, `PreMind` captures rich, reliable multimodal information, allowing users
to search for details like abbreviations shown

only on slides. Systematic evaluations on the
public LPM dataset and an internal enterprise
dataset are conducted to validate `PreMind` ’s

effectiveness, supported by detailed analyses.


**1** **Introduction**


Recent technological advancements have led to
the proliferation of online videos, which increasingly become an important source for learning new
knowledge (Soni and Dubey, 2019). Presentationstyle lecture videos, which mainly present slides
sequentially, are widely used for online courses and


  - Work done during an internship at Bosch Research North America.



trainings (Mishra et al., 2023). Systems that can effectively understand and index rich content of such
videos thus become desirable (Mishra et al., 2023),
which could lead to advanced downstream applications such as question answering (QA) based
on video details in various modalities. However,
state-of-the-art (SOTA) approaches for indexing
video content (Iyer et al., 2019; Saoudi and Jai Andaloussi, 2021) remain unsatisfactory, as they fail
to capture detailed multimodal information. With
the rapid advancements in large language model
(LLM) technology (Zhao et al., 2023; Chang et al.,
2024), Video Large Language Models (Vid-LLMs)
(Lin et al., 2023; Team et al., 2024) have emerged,
enabling users to directly ask questions about provided videos (Pan et al., 2023). However, Vid-LLM
cannot answer questions that require systems to
find the answers from a large number of videos due
to its design and computation limitation.
In this work, we propose `PreMind`, a novel multiagent multimodal framework that leverages various large models to capture detailed multimodal
information in presentation-style lecture videos,
leading to information-rich indexes that can benefit downstream tasks such as QA. For this work,
we adopt the broad understanding of agents, viewing agents as system components that each has its
own goals and work together to achieve a common
goal (Wang et al., 2024a). `PreMind` begins with
a video segmentation component that combines
a SOTA vision-based approach for segmentation
with VLM to efficiently and reliably segment a
video into many video segments, each covering
one presentation slide. Then, `PreMind` generates
textual description for each segment with an advanced video-segment understanding component.
For each segment, the component leverages appropriate agents to understand visual information,
capture speech narrative, and generate a consolidated description. The component also involves
innovative mechanisms to (i) improve vision under

standing by leveraging knowledge learned previously in the video lecture, (ii) automatically correct
speech recognition errors with VLM based on visual information and speech transcript, and (iii)
further improve vision understanding through dynamic self-refinement with a critic agent. Based on
the information-rich indexes created, downstream
tasks such as retrieval-based QA and summarization can be implemented for various applications.

We evaluate `PreMind` using public LPM dataset
(Lee et al., 2023) as well as an enterprise internal dataset. Both intrinsic evaluation and extrinsic

evaluation are conducted. The evaluation results

show the effectiveness of `PreMind`, demonstrating
the value of capturing detailed multi-modal information in indexes, as well as the benefits of the
proposed mechanisms for the video understanding/indexing task. To summarize, our contributions
include:

- We introduce `PreMind`, a multi-agent multimodal framework that uses various large models
to capture accurate and detailed multi-model information from presentation-style lecture videos,
which can further benefit the possible downstream applications such as QA.

- We demonstrate the effectiveness of `PreMind` on

the public LPM dataset (Lee et al., 2023) and an
enterprise internal dataset through intrinsic and
extrinsic evaluations.

- We conduct ablation studies to evaluate the efficacy of different mechanisms within `PreMind`,
and present comprehensive analyses of framework’s efficiency as well as case studies.


**2** **Related Work**


**Lecture Video Indexing** Indexing lecture videos
is an increasingly crucial task for enhancing access
to relevant content within educational materials,
which often involves video segmentation and information extraction from video segments. For
segmentation, Chand and O˘gul (2021) used Voice
Activity Detection and Gaussian Mixture Models
to segment videos based on speech. Shah et al.
(2015) aligns spoken content with Wikipedia text
for better segmentation. Jeong et al. (2012) applied SIFT for precise slide detection. For information extraction, Optical Character Recognition
(OCR) helps retrieve text from slides and Automatic Speech Recognition (ASR) helps obtaining
the speech in text format. The extracted text is
then used for indexing the video in a content-based



manner. Ip and Chan (1997) used OCR for hierarchical indexing, while Yang and Meinel (2014)
combined ASR and OCR for comprehensive video
search. Multimodal approaches, like those by Yamamoto et al. (2003) and Lin et al. (2004), integrate
ASR and OCR for improved indexing. However,
indexes generated from previous works (Yang et al.,
2011b; Ma et al., 2017; Yang et al., 2011a; Debnath et al., 2023; Arazzi et al., 2023; Medida and
KASARAPU, 2021) do not have rich multi-modal
information but rather simple text description obtained from OCR or ASR. Despite advancements,
challenges remain in achieving high accuracy in
indexing due to ASR errors, visual variations, and
lecture complexity. However, these issues can be
mitigated with the help of LLMs and VLMs. To the
best of our knowledge, our proposed framework is
the first to utilize Large Models to enhance video
indexing quality.


**Video Large Language Models** Vid-LLMs are
widely used for tasks involve video understanding
(Abdullah et al., 2024). For Vid-LLM, it typically
uniformly samples a certain numbers of frames
from videos and utilize a visual encoder (Dosovitskiy et al., 2020; Radford et al., 2021) to convert
each frame into vector representation. Then, an
adapter is used to map the video embeddings from
visual semantic space to text semantic space of
LLMs. Textual embeddings of instructions are then
added generating responses for downstream tasks
(Zhang et al., 2023; Maaz et al., 2024; Lin et al.,
2024; Song et al., 2024; Chen et al., 2023; Ma et al.,
2024), or a specially designed task-specific head
can be used to perform regression tasks (Yu et al.,
2023; Huang et al., 2024; Ren et al., 2024; Li et al.,
2024). For Vid-LLMs, although previous works
have demonstrated impressive video understanding
capabilities, they are not well-suited for contentbased lecture video indexing. One limitation lies in
their sampling strategy, which is suboptimal as it
can result in redundant information or the omission

of critical details (Wang et al., 2024b), thus not suitable for video indexing that requires segmentation.
Additionally, these systems are mostly designed to
process one query about one video or several at
a time in an online fashion, thus not suitable for
indexing videos.


**3** **Method**


The overall structure of `PreMind` is illustrated in

Figure 1. It consists of a video segmentation com

|Audio Understanding Agent<br>(speech-to-text)|Col2|Col3|
|---|---|---|
|Audio Understanding Agent<br>(speech-to-text)||ASR correction|













































Figure 1: Illustration of the proposed `PreMind` framework.



ponent and a video-segment understanding component, generating three understanding results based
on vision, speech, and consolidated information,
respectively, as the output indexes. Given a number
of lecture videos, `PreMind` processes the videos
one by one, and the resulting pool of indexes can
be used in downstream tasks.


**3.1** **Video Segmentation**


The video segmentation component attempts to segment a training video into multiple segments, each
covering the presentation of one slide. We adopt
the state-of-the-art PySceneDetect (Gruzman and
Kostenkova, 2014; Reddy and Jadhav, 2015) to conduct the first-round segmentation. PySceneDetect
faces two major challenges for this task, (1) often
missing slide with similar layout as the precedent
one and (2) splitting the presentation of a single
slide into multiple segments due to background
changes. Therefore, we innovatively use VLM
to refine the segmentation when needed. After
the first-round segmentation, for long segments
(>1 minute) detected by PySceneDetect, we apply
VLM to re-detect slides ( _Step_ _A_ ) in that segment
with the aim of catching the missed slides. The
time span for each newly detected slide is determined using vision and audio cues ( _Step_ _B_ ). For
other segments, VLM is leveraged to merge the
current segment with the previous one if the two
segments are deemed as presenting a same slide.
More details of our proposed video segmentation
algorithm is shown in Appendix A.1.



**3.2** **Video-Segment Understanding**


With the obtained video segments, denoted as
_{S_ 1 _, S_ 2 _, ..., S_ _n_ _}_, the video-segment understanding
component attempts to understand/index the content of each segment _S_ _i_ with a multi-agent solution,
as shown in Figure 1. We develop the understanding component in an incremental way. First, we design a baseline understanding system, which adopts
three separate agents to extract audio, vision, and
consolidated information from the segment. We
then leverage knowledge to enhance the vision understanding, extracting knowledge per slide, keeping a knowledge memory for the lecture in focus,
and leveraging the previously-learned knowledge
to understand the current slide. We further reduce

the impact of ASR errors on understanding results
by leveraging VLM to automatically correct ASR
errors based on both speech transcription and slide
visual content. Finally, we introduce a critic agent
to dynamically refine vision understanding result in
a self-reflection manner. Prompt templates for all
agents/algorithms can be found in Appendix A.5.


**3.2.1** **Baseline System**

- **Audio Understanding Agent:** This agent uses
a ASR model to generate a speech transcript, denoted as _transcript_ _i_, for each segment _S_ _i_ . In
practice, the ASR model is applied to transcribe
the whole video in focus. After the video segmentation process, the audio understanding agent
extracts _transcript_ _i_ for _S_ _i_ from the whole transcript based on the starting/ending times of _S_ _i_ .

- **Vision Understanding Agent:** Given _S_ _i_, this
agent generates a detailed description of the slide
presented in the segment using a SOTA VLM. It


samples a (representative) video frame from _S_ _i_
and asks the VLM to describe the slide shown in

that frame image in detail. The description generated is the vision understanding result, denoted
as _vision_ _ _understanding_ _i_ .

- **Vision-Audio Consolidation Agent:** Based
on the audio and vision understanding results,
the consolidation agent further generates a
consolidated understanding result, denoted as
_consolidated_ _ _information_ _i_, to provide a good
overall understanding about what is presented in
the video segment _S_ _i_ .


**3.2.2** **Knowledge-Related Enhancement**

- **Knowledge Memory** : Given a segment _S_ _i_,
knowledge presented in previous slides may be
helpful to understand the current slide. We
maintain a knowledge memory _KM_ for the
lecture that _S_ _i_ is part of. The knowledge
memory contains entries in the following format: _knowledge_ _m_ = { _embedding_ _m_, _name_ _m_,
_explanation_ _m_ }, where _m ∈_ _M_ and _M_ is the
number of entries in _KM_ . In each entry, _name_ _m_
stores a concept name, such as Product Lifecycle Management, _explanation_ _m_ stores the
explanation of that concept, and _embedding_ _m_
stores the embedding representation of the concept name, which is obtained by a SentenceTransformer model [1], for knowledge retrieval. For each
lecture, _KM_ is initially empty. It is then updated
when new knowledge is extracted, starting from
the first segment of the lecture. Notice that, unlike previous works described in (Hatalis et al.,
2024), where memory is used for tasks that only
involve text, we keep a knowledge memory that
contain multi-modal information from previous
video segments to help understand current seg
ment.

- **Knowledge Extraction Agent:** This agent extracts new knowledge from the consolidated understanding result for _S_ _i_, and updates _KM_ correspondingly. It asks a SOTA LLM to extract concepts, each including a concept name and an explanation, from _consolidated_ _ _information_ _i_ .
For each extracted concept, embedding representation of the concept name, _e_, is computed with
the SentenceTransformer, and entries in _KM_
are ranked by cosine similarity between _e_ and
_embedding_ _m_ . If the top-ranked entry has a similarity score less than 0.7, the extracted concept is
deemed new and we update the _KM_ by inserting


1 [https://www.sbert.net/](https://www.sbert.net/)



the concept as a new entry. Otherwise, the topranked entry is deemed to have similar concept
with the extracted concept. In this case, we update the top-ranked entry _knowledge_ _j_ by (1) appending the explanation of the extracted concept
to _explanation_ _j_ and (2) updating _embedding_ _j_
as the rolling average of _embedding_ _j_ and _e_ .

- **Keyword Extraction (Part of the Keyword Ex-**
**traction and ASR Error Correction Agent):**
Given _S_ _i_, we extract a set of keywords, denoted
as _keywords_ _i_, from the slide vision using a
VLM. These keywords are then used to retrieve
relevant knowledge from _KM_ . In this work, we
merge the tasks of keyword extraction and ASR
error correction into one agent, using one VLM
prompt to accomplish the two tasks at the same
time for improved efficiency and performance.

- **Keyword-based Knowledge Retrieval Agent:**
With _keywords_ _i_ extracted from _S_ _i_, this agent
retrieves relevant knowledge from _KM_ to facilitate vision understanding. For each keyword in
_keywords_ _i_, the agent computes its embedding
representation using the SentenceTransformer
model, and then ranks the _KM_ entries by cosine
similarity between the keyword’s embedding and
_embedding_ _m_ . Among the top 10 entries, those
entries with similarity score larger than 0.7 are
deemed relevant, and are provided to the vision
understanding agent as context.


**3.2.3** **ASR Correction**


Mistakes made by ASR model have a negative impact on the quality of generated understanding results. To reduce ASR errors, this work proposes an
innovative approach that leverages reliable visual
information, i.e., the keywords extracted from slide
vision, to correct ASR errors using a VLM, while
previous works on ASR correction mainly rely on
ASR results alone (Ma et al., 2023, 2025). For the
keyword extraction and ASR error correction agent,
given _S_ _i_, we use one prompt to ask the VLM to
(1) extract keywords shown in the slide, and (2)
check _transcript_ _i_ to identify possible ASR mistakes made on the keywords and make correction
suggestions. The combination of the two tasks not
only enhances efficiency but also benefit the performance of ASR correction, as constraining the
correction scope to keywords helps reduce hallucination of VLM on ASR correction.


**3.2.4** **Dynamic Critic**


LLM self-reflection has been found effective
for various text processing tasks(Madaan et al.,
2023; Liang et al., 2024). In this work, we extend this technique to multi-modal data, introducing a critic agent to enhance vision understanding through iterative reflection. Given the
slide of _S_ _i_, the retrieved knowledge, and the
_vision_ _ _understanding_ _i_ generated by the Vision
Understanding agent, the critic agent aims to identify defects of _vision_ _ _understanding_ _i_, such as
counting mistakes and missing figures in description. With the feedback from the critic agent, the
vision understanding agent further improves the understanding result and sends the updated result to
the critic agent to review. This process iterates until
the critic agent is satisfied with the understanding
result. We realize this dynamic reflection mechanism by grouping the critic agent and the vision
understanding agent into a AutoGen (Wu et al.,
2023) groupchat, which also includes an admin
agent to assist the chatting function. We configure
the groupchat to allow at maximum _N_ _max_ visionunderstanding/critic calls in the reflection iteration.
For early termination, we define the chat termination condition as the ’TERMINATE!!!’ command

issued by the Critic Agent.


**4** **Experiments and Analyses**


**4.1** **Dataset**

|Dataset|Total Video<br>Video # Video Segments #<br>Length(mins)|
|---|---|
|LPM<br>EI|6<br>54<br>60_._0<br>7<br>37<br>56_._3|



Table 1: Dataset statistics for video segmentation evaluation. Segments refer to ground-truth segments manually
labeled.

|Dataset|Total Video<br>Video # Video Segments #<br>Length(hours)|
|---|---|
|LPM<br>EI|188<br>1366<br>28_._17<br>66<br>264<br>5_._96|



Table 2: Dataset statistics for evalution on understanding
performance. Segments are manually labeled for LPM
data but automatically detected for EI data.


We evaluate `PreMind` on the public LPM dataset
(Lee et al., 2023) and an enterprise internal (EI)
dataset. The LPM dataset contains YouTube lec
tures across 10 different categories (e.g., biology, psychology), having more than 180 hours of



videos and providing manually- segmented slidepresentation segments for over 9,000 slides in total.
The internal dataset contains 66 videos (6 hours
in total) on various enterprise-training topics. The
videos in both datasets are presentation-style lectures, though the layout of the slide display in the
videos varies. In this work, we randomly sample
a subset of videos per dataset to evaluate video
segmentation performance, as shown in Table 1.
Note that for the LPM videos sampled (listed in
Appendix A.2), the first 10 minutes of each video
is used in video-segmentation evaluation. For the
evaluation of understanding performance, we select another subset of lectures from the LPM data,

which contains almost 30 hours of videos in to
tal, due to computational constraints. We use this
LPM subset and the whole EI dataset to evaluate

the video-understanding approaches as well as QA
performance in extrinsic evaluation, as listed in
Table 2. We construct the LPM subset for under
standing evaluation by (1) selecting the first three
lecture videos from each category except dental,
and (2) for dental videos, which contain 19 subcategories and each video is only around 5 minutes,
selecting all the videos of the first 3 subcategories.


**4.2** **Video Segmentation Evaluation**


**4.2.1** **Settings**


We evaluate our proposed video segmentation approach and compare it with PySceneDetect in Table 1. Details on the parameter tuning as well
as algorithm configurations are provided in Appendix A.4.1. GPT-4 Vision is used in our proposed
segmentation algorithm. For video-segmentation
evaluation, we report Precision, Recall, and F-1
score for detecting video segments. We also report
the Intersection over Union (IoU) score (ranging
from 0 to 1), which specifies the amount of overlap
on time span between the predicted and ground
truth segments. IoU is calculated as:


_IoU_ = _[|][A][ ∩]_ _[B][|]_ (1)

_|A ∪_ _B|_


where _A_ and _B_ are the time spans of the the
predicted and ground-truth segments, respectively.


**4.2.2** **Video Segmentation Results**

|Dataset|LPM|EI|
|---|---|---|
|Algorithm|PySceneDetect<br>Ours|PySceneDetect<br>Ours|
|Precision<br>Recall<br>F1|77.50<br>100.00<br>88.33<br>96.55<br>82.56<br>98.24|94.59<br>100.00<br>80.00<br>100.00<br>86.69<br>100.00|
|IoU|0.74<br>0.80|0.63<br>0.91|



Table 3: Video segmentation results.


The evaluation results on video segmentation are
shown in Table 3. We can see that our proposed
segmentation approach significantly outperforms
SOTA vision-based approach, demonstrating the
power of VLM. Our approach achieves almost perfect results on EI, successfully detecting all presented slides. The performance on LPM is somehow imperfect, as the LPM data contains occasional occurrences of animations/demonstrations,
making it more challenging. From efficiency aspect, as our proposed segmentation approach only
applies VLM when needed, the computational overhead introduced with VLM is minimized, and the
segmentation efficiency is largely maintained (as
will be further discussed in Section 4.4).


**4.3** **Video-Segment Understanding Evaluation**


**4.3.1** **Settings**

We conduct video-segment understanding experiments based on the datasets listed in Table 2. For

these experiments, we directly use the provided
manual segmentation for the LPM data and use
our proposed segmentation algorithm to segment
the EI data. Using video segments obtained in
this manner, four incrementally developed videosegment understanding approaches are applied to
each dataset: (1) the baseline system, (2) plus
knowledge-related enhancement, (3) plus ASR correction, and (4) plus dynamic critic. Each approach
is applied individually, generating a corresponding
set of understanding results, which are then evaluated and compared to assess their performance. In
this set of experiments, Whisper [2] is used to generate ASR result, and GPT-4 Turbo is used for all
agents that require a VLM/LLM in processing. Algorithm parameters for video-segment understanding are determined empirically, with details provided in Appendix A.4.2.


**4.3.2** **Intrinsic Evaluation**


**Evaluation Approach** We first evaluate the four
proposed video-segment understanding approaches


2 [https://openai.com/index/whisper/](https://openai.com/index/whisper/)


|Col1|Col2|Total #|Win # Tie # Lose #|Win %|(Win+Tie) %|
|---|---|---|---|---|---|
|LPM|B+K vs B<br>B+K+A vs B<br>B+K+A+D vs B|155<br>155<br>160|76<br>49<br>30<br>91<br>39<br>25<br>94<br>32<br>34|49_._03%<br>58_._71%<br>58_._75%|80_._65%<br>83_._9%<br>78_._75%|
|EI|B+K vs B<br>B+K+A vs B<br>B+K+A+D vs B|35<br>39<br>34|16<br>10<br>9<br>18<br>13<br>8<br>16<br>13<br>5|45_._71%<br>46_._15%<br>47_._06%|74_._29%<br>79_._49%<br>85_._29%|



Table 4: Comparison on vision understanding results
with human evaluation. B refers to the baseline system;
K refers to the knowledge-related enhancement mechanism; A refers to the ASR correction mechanism; D
refers to the dynamic critic self-reflection mechanism.


**Results** Table 4 reports the comparison results
on vision understanding performance for different
video-segment understanding settings against the
baseline system. For each pair of approaches in
comparison, we list (1) the total number of cases
where the final vision-understanding results generated by the two approaches are deemed different
by GPT-4 Turbo, and (2) among those cases, which
are sent to human annotators to label, the competition results according to the human annotation. If
the first approach is deemed better than the second
approach according to annotation results, it wins


3 [https://www.mturk.com/](https://www.mturk.com/)



on vision understanding performance. As this is a
challenging task by nature, we adopt human annotation to ensure the evaluation quality, and propose a
pairwise comparison schema for evaluation. Given
a pair of vision understanding results to compare
together with the corresponding slide image, to reduce the labeling workload for human, we first ask
GPT-4 Turbo to determine whether the two under
standing results are consistent in meaning ( prompt
listed in Appendix Table 13).
When an inconsistency is detected, the result
pair is sent to humans for manual quality comparison. For each pair requiring manual evaluation,
a questionnaire is prepared, asking annotators to
determine which result is better and explain their
reasoning based on the slide image and relevant
knowledge. Each questionnaire is completed by
three annotators, and the final judgment is determined by a majority vote. For EI data, due to
confidential issue, we recruit internal associates to
annotate related questionnaires. For the LPM data,
we leverage Amazon Mechanical Turk [3] for the
annotation, designing special measures to ensure
high quality of annotation (e.g., selecting workers
having trustful records, using dummy questions to
reject irresponsible answers). The details about
the measures and the questionnaire examples are
provided in Appendix A.3.






Question LPM **Accuracy on LPM** (given index type) EI **Accuracy on EI** (given index type)
Type Question # All Vision Speech Consolidation Question # All Vision Speech Consolidation


Vision 2716 78 _._ 57 78 _._ 76 33 _._ 59 74 _._ 96 392 76 _._ 27 75 _._ 77 23 _._ 21 68 _._ 62
Speech 2706 70 _._ 81 49 _._ 04 67 _._ 63 68 _._ 92 394 82 _._ 74 46 _._ 45 87 _._ 06 71 _._ 83
Consolidation 2722 86 _._ 41 79 _._ 21 66 _._ 13 87 _._ 88 394 90 _._ 36 79 _._ 19 79 _._ 44 88 _._ 83


**Overall** 8144 78 _._ 61 69 _._ 03 55 _._ 77 77 _._ 27 1180 83 _._ 13 67 _._ 11 63 _._ 31 76 _._ 44


Table 5: Question-answering performance on LPM and DC data.



the second approach. If it is deemed similar to
the second approach, a tie occurs. Otherwise, the
first approach loses the competition. From Table 4,
we can see that adding knowledge-related enhancement, ASR correction, and dynamic critic gradually
improves the quality of vision understanding for
both LPM and EI data. Note that the improvement
brought by ASR correction on vision understanding
is indirect (i.e., better speech transcript _→_ better
consolidated info _→_ better extracted knowledge _→_
better knowledge-assisted vision understanding).
For lose cases, we suspect these are caused by the
noises in the knowledge that is retrieved and fed
into the vision understanding agent. These noises
may distract the agent from important information
during the generation of slide description.


**4.3.3** **Extrinsic Evaluation with QA**


**Evaluation Approach** We conduct extrinsic evaluation by applying the generated indexes for QA
tasks to show the impact of the proposed videosegment understanding component on downstream
application. With the three indexes (i.e., vision
understanding result, speech transcript, and consolidation information) generated for each video segment in a dataset, we setup a retrieval-augmented
QA system. In this set of experiments, we adopt
the complete video-segment understanding component (i.e., the B+K+A+D setting) to generate the
understanding results as textual indexes. For evaluation, we ask GPT-4 Turbo to generate 6 questions
together with the ground-truth answers. Among
the 6 questions, 2 focus on speech, 2 focus on slide
vision, and 2 focus on the consolidated information. After using QA to generate an answer for
each question, we further leverage GPT-4 Turbo
to evaluate the answer based on the ground-truth.
The details about the QA settings, question/groundtruth generation, and the GPT based evaluation are
all provided in Appendix A.4.3.


**Results** We evaluate the QA performance as the
accuracy of the answers generated by QA. The results are reported in Table 5. The performance
of only using vision-understanding-result indexes,



**LPM** **EI**


Baseline + Knowledge 67 _._ 82 70 _._ 59
+ ASR correction 73 _._ 91 82 _._ 35


Table 6: QA performance (Accuracy) given the question
set for ASR-correction ablation study (870 questions for
LPM and 102 questions for EI).


only using speech-transcript indexes, only using
consolidated-information indexes, and using all
types of indexes in retrieval are evaluated per question type. The results demonstrate the value of capturing multimodal information in indexes for QA.
Each type of indexes (e.g., vision understanding
result) is advantageous for certain kind of questions
(e.g., those asking for vision details not presented
by speech). And adopting indexes of all types bring
the best overall performance.


**4.3.4** **Ablation Study**

Leveraging the QA system, we further conduct ablation study on the benefit of the proposed ASR
correction procedure by comparing the system performance with and without this procedure. For
this evaluation, we generate another set of questions together with their ground-truth in a similar
way as before. In this case, based on the predicted
ASR corrections, we ask GPT-4 Turbo to generate questions answerable with the corrected speech
transcript but not answerable without the corrections (See Table 12 for the prompt). With this set
of questions, using only speech-transcript indexes
in the QA system, we evaluate the answer accuracy
as before. We focus on the speech-transcript indexes here for the easiness of interpretation, as the
major impact of ASR Correction is on audio understanding result. The evaluation results are shown
in Table 6. We can see that ASR correction brings
substantial improvements on QA accuracy for both
datasets, demonstrating the benefit of correcting
ASR errors on video understanding.
In a similar manner, we also conduct another ablation study on vision-understanding aspect. In this
case, we focus on each pair of vision-understanding
results that has one result determined as better

than the other in human annotation, and ask GPT-4


**LPM** **EI**


Baseline 36 _._ 88 39 _._ 40
+ Knowledge 46 _._ 86 48 _._ 48
+ ASR Correction 46 _._ 20 51 _._ 52
+ Dynamic Critic 83 _._ 13 69 _._ 70


Table 7: QA performance (Accuracy) given the question set for vision-understanding ablation Study (320
questions for LPM and 66 questions for EI).


Turbo to generate questions answerable with the
better result but not with the other one (See Table 12 for the prompt). With these questions, we
evaluate the four proposed video-segment understanding settings with QA, using only the visionunderstanding indexes in the QA system in this
case. The results are reported in Table 7. We can
see that both the knowledge-related enhancement
and the dynamic critic mechanism bring substantial improvements on performance. For ASR correction, which only has indirect impact on vision
understanding, it unsurprisingly brings only mild
improvement on EI, and brings no improvement on
LPM probably due to generation noises.


**4.3.5** **Case Study**


|Video Segmentation|# API calls Video Length (s)|
|---|---|
|EI Data<br>LPM Data|_≈_4<br>483<br>_≈_3<br>600|
|**Video-Segment Understanding**|**# API calls**<br>**Time Lapse (s)**|
|Baseline<br>+ Knowledge<br>+ ASR Correction<br>+ Dynamic Critic|2<br>40<br>4<br>65<br>4<br>65<br>_≈_9<br>107|






|LPM The trends of the metabolic rate of endothermy<br>and ectorthermy are described in reverse. With<br>external knowledge of metabolic rate, the trends<br>are correctly described.<br>Knowledge EI The description of "ERP layer" wrongly includes<br>certain stages of "PLM layer" due to confusing<br>slide layout. With external knowledge of PLM<br>and ERP leveraged, this problem is fixed.|LPM|The trends of the metabolic rate of endothermy<br>and ectorthermy are described in reverse. With<br>external knowledge of metabolic rate, the trends<br>are correctly described.|
|---|---|---|
|Knowledge<br>LPM<br>The trends of the metabolic rate of endothermy<br>and ectorthermy are described in reverse. With<br>external knowledge of metabolic rate, the trends<br>are correctly described.<br>EI<br>The description of "ERP layer" wrongly includes<br>certain stages of "PLM layer" due to confusing<br>slide layout. With external knowledge of PLM<br>and ERP leveraged, this problem is fxed.|EI|The description of "ERP layer" wrongly includes<br>certain stages of "PLM layer" due to confusing<br>slide layout. With external knowledge of PLM<br>and ERP leveraged, this problem is fxed.|
|ASR Correction<br>LPM<br>A professor’s name "Bisque" is correctly<br>changed to "Bisk".<br>EI<br>A domain specifc term "Nexeed", previously<br>recognized as "NextSeat", is corrected.|LPM|A professor’s name "Bisque" is correctly<br>changed to "Bisk".|
|ASR Correction<br>LPM<br>A professor’s name "Bisque" is correctly<br>changed to "Bisk".<br>EI<br>A domain specifc term "Nexeed", previously<br>recognized as "NextSeat", is corrected.|EI|A domain specifc term "Nexeed", previously<br>recognized as "NextSeat", is corrected.|
|Dynamic Critic<br>LPM<br>The starting point for a shaded area in a fgure<br>previously recognized at 6.5, is correctly fxed<br>to 5.5.<br>EI<br>A Venn diagram previously missing from the<br>description is added.|LPM|The starting point for a shaded area in a fgure<br>previously recognized at 6.5, is correctly fxed<br>to 5.5.|
|Dynamic Critic<br>LPM<br>The starting point for a shaded area in a fgure<br>previously recognized at 6.5, is correctly fxed<br>to 5.5.<br>EI<br>A Venn diagram previously missing from the<br>description is added.|EI|A Venn diagram previously missing from the<br>description is added.|



Table 8: Case study for improvement brought by
knowledge-related enhancement, ASR correction, and
dynamic critic.


We also examine improvements achieved through
knowledge enhancement, ASR correction, and dynamic critic separately. Table 8 shows one example
per dataset for each mechanism. We observe that
knowledge is useful to help VLM understand ambiguous part of slides. For example, on a slide, several stages of PLM (product lifecycle management)
are listed right below the ERP (enterprise resource
planning) block. Although the PLM stages are not
connected to the ERP block in the slide, the VLM
wrongly describes that ERP includes those stages.



Table 9: Efficiency of the `PreMind` framework. For the
video segmentation procedure, the average length of
the videos and the average number of API calls needed
to segment one video are listed. For video-segment
understanding, the average number of API calls and
the average total time for generating indexes per video
segment are listed. ASR API call is **not included** as we
only need to call ASR model once per video.


This problem is fixed when definitions of PLM and
ERP are retrieved and provided to the VLM for
vision understanding. The ASR correction procedure excels at refining domain-specific terms and
resolving uncommon or ambiguous names. The dynamic critic self-reflection mechanism effectively
addresses specific errors made by the VLM in slide
descriptions, such as misinterpreting numbers in
figures, making counting errors, and missing visual
elements in the slide descriptions.


**4.4** **Framework Efficiency**


We evaluate the efficiency of our proposed framework `PreMind` in Table 9. Note that the framework

efficiency is largely determined by the API calls
used in various components. For each video, one
ASR API call is needed to generate a transcript for
the whole video. For the video segmentation procedure, except the VLM API calls, the additional processing time beyond the first-round PySceneDetect
is negligible. For the video-segment understanding, API calls occupy the majority of processing
time, while the average time for retrieving knowledge from the knowledge memory and the average
time for correcting detected ASR errors per video
segment are 364 ms and 10 ms, respectively.


**5** **Conclusion**


This work proposes `PreMind`, a novel framework
to understand/index rich multimodal information

for presentation-style lecture videos, with the aim
of enabling advanced downstream applications
such as QA. `PreMind` involves two components:
video segmentation and video-segment understanding. The video segmentation procedure combines
VLM with PySceneDetect to achieve the desired


performance with high efficiency. The videosegment understanding component not only captures visual/audio/consolidated information from

video using large models, but also introduces three
innovative mechanisms to improve the understanding performance. We evaluate `PreMind` on the
public LPM dataset and an internal dataset and
encouraging experimental results are achieved.


**Limitations**


`PreMind` relies heavily on proprietary LLMs and
VLMs for both video segmentation and understanding tasks. Open-sourced models may be used
to substitute proprietary models, but the performance may be effected. `PreMind` is optimized for
presentation-style lecture videos. Its generalizability to other video formats, such as freestyle videos
without any slides presented, has not been explored.
The reliance on human annotation for evaluation

introduces subjectivity, particularly in vision understanding tasks where judgment about description quality can vary across annotators. Future
work will address these limitations by exploring
lightweight and open-sourced model alternatives
and expanding the evaluation to include more diverse datasets and video formats.


**References**


Hasnat Md Abdullah, Tian Liu, Kangda Wei, Shu Kong,
[and Ruihong Huang. 2024. Ual-bench: The first com-](https://arxiv.org/abs/2410.01180)
[prehensive unusual activity localization benchmark.](https://arxiv.org/abs/2410.01180)
_Preprint_, arXiv:2410.01180.


Marco Arazzi, Marco Ferretti, and Antonino Nocera.
2023. [Semantic hierarchical indexing for online](https://api.semanticscholar.org/CorpusID:259044832)
[video lessons using natural language processing.](https://api.semanticscholar.org/CorpusID:259044832) _Big_
_Data Cogn. Comput._, 7:107.


[Dipesh Chand and Hasan O˘gul. 2021. A framework](https://doi.org/10.1109/SAMI50585.2021.9378632)
[for lecture video segmentation from extracted speech](https://doi.org/10.1109/SAMI50585.2021.9378632)
[content. In](https://doi.org/10.1109/SAMI50585.2021.9378632) _2021 IEEE 19th World Symposium on Ap-_
_plied Machine Intelligence and Informatics (SAMI)_,
pages 000299–000304.


Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu,
Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi,
Cunxiang Wang, Yidong Wang, et al. 2024. A survey on evaluation of large language models. _ACM_
_Transactions on Intelligent Systems and Technology_,
15(3):1–45.


Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu,
Yifei Huang, Junting Pan, Yi Wang, Yali Wang,
[Yu Qiao, Tong Lu, and Limin Wang. 2023. Vide-](https://arxiv.org/abs/2305.13292)
[ollm: Modeling video sequence with large language](https://arxiv.org/abs/2305.13292)
[models.](https://arxiv.org/abs/2305.13292) _Preprint_, arXiv:2305.13292.



Abhijit Debnath, K. Sreenivasa Rao, and Partha Pratim
[Das. 2023. A multi-modal lecture video indexing](https://api.semanticscholar.org/CorpusID:266533926)
[and retrieval framework with multi-scale residual](https://api.semanticscholar.org/CorpusID:266533926)
[attention network and multi-similarity computation.](https://api.semanticscholar.org/CorpusID:266533926)
_Signal Image Video Process._, 18:1993–2006.


Alexey Dosovitskiy, Lucas Beyer, Alexander
Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias
Minderer, Georg Heigold, Sylvain Gelly, Jakob
Uszkoreit, and Neil Houlsby. 2020. [An image](https://api.semanticscholar.org/CorpusID:225039882)
is worth 16x16 words: Transformers for image
[recognition at scale.](https://api.semanticscholar.org/CorpusID:225039882) _ArXiv_, abs/2010.11929.


Igor S Gruzman and Anna S Kostenkova. 2014. Algorithm of scene change detection in a video sequence
based on the threedimensional histogram of color
images. In _2014 12th International Conference on_
_Actual Problems of Electronics Instrument Engineer-_
_ing (APEIE)_, pages 1–1. IEEE.


Kostas Hatalis, Despina Christou, Joshua Myers, Steven
Jones, Keith Lambert, Adam Amos-Binks, Zohreh
[Dannenhauer, and Dustin Dannenhauer. 2024. Mem-](https://doi.org/10.1609/aaaiss.v2i1.27688)
[ory matters: The need to improve long-term memory](https://doi.org/10.1609/aaaiss.v2i1.27688)
[in llm-agents.](https://doi.org/10.1609/aaaiss.v2i1.27688) _Proceedings of the AAAI Symposium_
_Series_, 2(1):277–280.


Bin Huang, Xin Wang, Hong Chen, Zihan Song, and
Wenwu Zhu. 2024. Vtimellm: Empower llm to grasp
video moments. In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recog-_
_nition (CVPR)_, pages 14271–14280.


Horace Ho-Shing Ip and Siu-Lok Chan. 1997.

[Hypertext-assisted video indexing and content-based](https://doi.org/10.1145/267437.267478)
[retrieval. In](https://doi.org/10.1145/267437.267478) _Proceedings of the Eighth ACM Confer-_
_ence on Hypertext_, HYPERTEXT ’97, page 232–233,
New York, NY, USA. Association for Computing
Machinery.


Rahul Radhakrishnan Iyer, Sanjeel Parekh, Vikas Mohandoss, Anush Ramsurat, Bhiksha Raj, and Rita
[Singh. 2019. Content-based video indexing and re-](https://arxiv.org/abs/1602.08581)
[trieval using corr-lda.](https://arxiv.org/abs/1602.08581) _Preprint_, arXiv:1602.08581.


Hyun Ji Jeong, Tak-Eun Kim, and Myoung-Ho Kim.
2012. [An accurate lecture video segmentation](https://api.semanticscholar.org/CorpusID:14262991)
[method by using sift and adaptive threshold. In](https://api.semanticscholar.org/CorpusID:14262991) _Ad-_
_vances in Mobile Multimedia_ .


Dong Won Lee, Chaitanya Ahuja, Paul Pu Liang, Sanika
Natu, and Louis-Philippe Morency. 2023. Lecture
presentations multimodal dataset: Towards understanding multimodality in educational videos. In _Pro-_
_ceedings of the IEEE/CVF International Conference_
_on Computer Vision (ICCV)_, pages 20087–20098.


Zhaowei Li, Qi Xu, Dong Zhang, Hang Song, YiQing
Cai, Qi Qi, Ran Zhou, Junting Pan, Zefeng Li, Vu Tu,
[Zhida Huang, and Tao Wang. 2024. GroundingGPT:](https://doi.org/10.18653/v1/2024.acl-long.360)
[Language enhanced multi-modal grounding model.](https://doi.org/10.18653/v1/2024.acl-long.360)
In _Proceedings of the 62nd Annual Meeting of the_
_Association for Computational Linguistics (Volume 1:_
_Long Papers)_, pages 6657–6678, Bangkok, Thailand.
Association for Computational Linguistics.


Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang,
Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and
[Zhaopeng Tu. 2024. Encouraging divergent thinking](https://doi.org/10.18653/v1/2024.emnlp-main.992)
[in large language models through multi-agent debate.](https://doi.org/10.18653/v1/2024.emnlp-main.992)
In _Proceedings of the 2024 Conference on Empiri-_
_cal Methods in Natural Language Processing_, pages
17889–17904, Miami, Florida, USA. Association for
Computational Linguistics.


Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning,
[Peng Jin, and Li Yuan. 2023. Video-llava: Learn-](https://arxiv.org/abs/2311.10122)
[ing united visual representation by alignment before](https://arxiv.org/abs/2311.10122)
[projection.](https://arxiv.org/abs/2311.10122) _Preprint_, arXiv:2311.10122.


Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning,
[Peng Jin, and Li Yuan. 2024. Video-LLaVA: Learn-](https://doi.org/10.18653/v1/2024.emnlp-main.342)
[ing united visual representation by alignment before](https://doi.org/10.18653/v1/2024.emnlp-main.342)
[projection. In](https://doi.org/10.18653/v1/2024.emnlp-main.342) _Proceedings of the 2024 Conference on_
_Empirical Methods in Natural Language Processing_,
pages 5971–5984, Miami, Florida, USA. Association
for Computational Linguistics.


Ming Lin, J.F. Nunamaker, M. Chau, and Hsinchun
[Chen. 2004. Segmentation of lecture videos based on](https://doi.org/10.1109/HICSS.2004.1265045)
[text: a method combining multiple linguistic features.](https://doi.org/10.1109/HICSS.2004.1265045)
In _37th Annual Hawaii International Conference on_
_System Sciences, 2004. Proceedings of the_, pages 9

pp.–.


Di Ma, Xi Zhang, Xu Ouyang, and Gady Agam. 2017.

[Lecture vdeo indexing using boosted margin max-](https://doi.org/10.1109/ICMLA.2017.0-155)
[imizing neural networks.](https://doi.org/10.1109/ICMLA.2017.0-155) In _2017 16th IEEE In-_
_ternational Conference on Machine Learning and_
_Applications (ICMLA)_, pages 221–227.


Fan Ma, Xiaojie Jin, Heng Wang, Yuchen Xian, Jiashi
Feng, and Yi Yang. 2024. Vista-llama: Reducing
hallucination in video language models via equal
distance to visual tokens. In _Proceedings of the_
_IEEE/CVF Conference on Computer Vision and Pat-_
_tern Recognition (CVPR)_, pages 13151–13160.


Rao Ma, Mengjie Qian, Mark Gales, and Kate Knill.
[2025. Asr error correction using large language mod-](https://arxiv.org/abs/2409.09554)
[els.](https://arxiv.org/abs/2409.09554) _Preprint_, arXiv:2409.09554.


Rao Ma, Mengjie Qian, Potsawee Manakul, Mark Gales,
and Kate Knill. 2023. [Can generative large lan-](https://arxiv.org/abs/2307.04172)
[guage models perform asr error correction?](https://arxiv.org/abs/2307.04172) _Preprint_,
arXiv:2307.04172.


Muhammad Maaz, Hanoona Rasheed, Salman Khan,
[and Fahad Khan. 2024. Video-ChatGPT: Towards](https://doi.org/10.18653/v1/2024.acl-long.679)
[detailed video understanding via large vision and](https://doi.org/10.18653/v1/2024.acl-long.679)
[language models. In](https://doi.org/10.18653/v1/2024.acl-long.679) _Proceedings of the 62nd An-_
_nual Meeting of the Association for Computational_
_Linguistics (Volume 1: Long Papers)_, pages 12585–
12602, Bangkok, Thailand. Association for Computational Linguistics.


Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
Shashank Gupta, Bodhisattwa Prasad Majumder,



Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. [Self-refine: It-](https://arxiv.org/abs/2303.17651)
[erative refinement with self-feedback.](https://arxiv.org/abs/2303.17651) _Preprint_,
arXiv:2303.17651.


Lakshmi Medida and RAMANI KASARAPU. 2021.

[An optimized e-lecture video search and indexing](https://doi.org/10.22937/IJCSNS.2021.21.8.12)
[framework. 21:87–96.](https://doi.org/10.22937/IJCSNS.2021.21.8.12)


Gouri Shankar Mishra, Anand Raj, Amit Kumar,
Aman Kumar Kasaudhan, Pradeep Kumar Mishra,
[and Tarun Maini. 2023. Indexing and segmentation](https://doi.org/10.1109/INCET57972.2023.10170589)
[of video contents: A review. In](https://doi.org/10.1109/INCET57972.2023.10170589) _2023 4th Interna-_
_tional Conference for Emerging Technology (INCET)_,
pages 1–9.


Junting Pan, Ziyi Lin, Yuying Ge, Xiatian Zhu, Renrui Zhang, Yi Wang, Yu Qiao, and Hongsheng Li.
2023. Retrieving-to-answer: Zero-shot video question answering with frozen large language models.
In _Proceedings of the IEEE/CVF International Con-_
_ference on Computer Vision_, pages 272–283.


Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
[Gretchen Krueger, and Ilya Sutskever. 2021. Learn-](https://arxiv.org/abs/2103.00020)
[ing transferable visual models from natural language](https://arxiv.org/abs/2103.00020)
[supervision.](https://arxiv.org/abs/2103.00020) _Preprint_, arXiv:2103.00020.


Bindu Reddy and Anita Jadhav. 2015. Comparison of
scene change detection algorithms for videos. In
_2015 Fifth International Conference on Advanced_
_Computing & Communication Technologies_, pages
84–89. IEEE.


Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and
Lu Hou. 2024. Timechat: A time-sensitive multimodal large language model for long video understanding. In _Proceedings of the IEEE/CVF Confer-_
_ence on Computer Vision and Pattern Recognition_
_(CVPR)_, pages 14313–14323.


[ElMehdi Saoudi and Said Jai Andaloussi. 2021. A dis-](https://doi.org/10.21203/rs.3.rs-255106/v1)
[tributed content-based video retrieval system for large](https://doi.org/10.21203/rs.3.rs-255106/v1)
[data-sets.](https://doi.org/10.21203/rs.3.rs-255106/v1)


Rajiv Ratn Shah, Yi Yu, Anwar Dilawar Shaikh, and
[Roger Zimmermann. 2015. Trace: Linguistic-based](https://api.semanticscholar.org/CorpusID:38213200)
[approach for automatic lecture video segmentation](https://api.semanticscholar.org/CorpusID:38213200)
[leveraging wikipedia texts.](https://api.semanticscholar.org/CorpusID:38213200) _2015 IEEE International_
_Symposium on Multimedia (ISM)_, pages 217–220.


Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng
Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi,
Xun Guo, Tian Ye, Yanting Zhang, Yan Lu, Jenq[Neng Hwang, and Gaoang Wang. 2024. Moviechat:](https://arxiv.org/abs/2307.16449)
[From dense token to sparse memory for long video](https://arxiv.org/abs/2307.16449)
[understanding.](https://arxiv.org/abs/2307.16449) _Preprint_, arXiv:2307.16449.


[Shraddha Soni and Shubham Dubey. 2019. Towards](https://doi.org/10.32628/IJSRCSEIT)
[systematic literature review of e-learning.](https://doi.org/10.32628/IJSRCSEIT)


Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan
Schalkwyk, Andrew M. Dai, Anja Hauth, and et al.


[2024. Gemini: A family of highly capable multi-](https://arxiv.org/abs/2312.11805)
[modal models.](https://arxiv.org/abs/2312.11805) _Preprint_, arXiv:2312.11805.


Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao
Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang,
Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei,
[and Jirong Wen. 2024a. A survey on large language](https://doi.org/10.1007/s11704-024-40231-1)
[model based autonomous agents.](https://doi.org/10.1007/s11704-024-40231-1) _Frontiers of Com-_
_puter Science_, 18(6).


Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simon[celli. 2004. Image quality assessment: from error](https://doi.org/10.1109/TIP.2003.819861)
[visibility to structural similarity.](https://doi.org/10.1109/TIP.2003.819861) _IEEE Transactions_
_on Image Processing_, 13(4):600–612.


Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mo[hit Bansal. 2024b. Videotree: Adaptive tree-based](https://arxiv.org/abs/2405.19209)
[video representation for llm reasoning on long videos.](https://arxiv.org/abs/2405.19209)
_Preprint_, arXiv:2405.19209.


Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran
Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun
Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan
Awadallah, Ryen W White, Doug Burger, and Chi
[Wang. 2023. Autogen: Enabling next-gen llm ap-](https://arxiv.org/abs/2308.08155)
[plications via multi-agent conversation.](https://arxiv.org/abs/2308.08155) _Preprint_,
arXiv:2308.08155.


Natsuo Yamamoto, Jun Ogata, and Yasuo Ariki. 2003.

[Topic segmentation and retrieval system for lecture](https://doi.org/10.21437/EUROSPEECH.2003-333)
[videos based on spontaneous speech recognition. In](https://doi.org/10.21437/EUROSPEECH.2003-333)
_8th European Conference on Speech Communica-_
_tion and Technology, EUROSPEECH 2003 - INTER-_
_SPEECH 2003, Geneva, Switzerland, September 1-4,_
_2003_, pages 961–964. ISCA.


Haojin Yang and Christoph Meinel. 2014. [Content](https://doi.org/10.1109/TLT.2014.2307305)
[based lecture video retrieval using speech and video](https://doi.org/10.1109/TLT.2014.2307305)
[text information.](https://doi.org/10.1109/TLT.2014.2307305) _IEEE Transactions on Learning_
_Technologies_, 7(2):142–154.


Haojin Yang, Harald Sack, and Christoph Meinel. 2011a.

[Lecture video indexing and analysis using video ocr](https://api.semanticscholar.org/CorpusID:14263115)
[technology.](https://api.semanticscholar.org/CorpusID:14263115) _2011 Seventh International Conference_
_on Signal Image Technology & Internet-Based Sys-_
_tems_, pages 54–61.


Haojin Yang, Maria Siebert, Patrick Lühne, Harald Sack,
[and Christoph Meinel. 2011b. Lecture video index-](https://doi.org/10.1109/SITIS.2011.20)
[ing and analysis using video ocr technology.](https://doi.org/10.1109/SITIS.2011.20)


Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. 2023. [Self-chained image-language](https://arxiv.org/abs/2305.06988)
[model for video localization and question answer-](https://arxiv.org/abs/2305.06988)
[ing.](https://arxiv.org/abs/2305.06988) _Preprint_, arXiv:2305.06988.


[Hang Zhang, Xin Li, and Lidong Bing. 2023. Video-](https://doi.org/10.18653/v1/2023.emnlp-demo.49)
[LLaMA: An instruction-tuned audio-visual language](https://doi.org/10.18653/v1/2023.emnlp-demo.49)
[model for video understanding. In](https://doi.org/10.18653/v1/2023.emnlp-demo.49) _Proceedings of_
_the 2023 Conference on Empirical Methods in Nat-_
_ural Language Processing: System Demonstrations_,
pages 543–553, Singapore. Association for Computational Linguistics.



Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, et al. 2023. A
survey of large language models. _arXiv preprint_
_arXiv:2303.18223_ .


**A** **Appendix**


The Appendix consists of supplemental materials
in our research journey for the video understanding
topic.


**A.1** **Video Segmentation Details**


Figure 2 shows the details of video (scene) segmentation algorithm, which leverages off-the-shelf
tool PySceneDetect. Meanwhile, a more detailed
description of pseudo code is shown in Figure 3 for
reference.


In the proposed segmentation algorithm, we
merge short segments (e.g., less than 3s) with
the proceeding ones, assuming they are transitions between slides. For the remaining segments
with reasonable duration (e.g., one minute or less),
we re-check whether the current segment actually
presents the same slide as the previous segment,
first using efficient SSIM (structural similarity index) (Wang et al., 2004) to identify obvious cases
and then using VLM to verify the remaining tricky
ones. If the answer is yes, the two segments are
merged. For the relatively long segments (i.e.,
_duration >_ one minute), we suspect the segment
may actually contain multiple similar slides, and
thus re-detect slides ( _Step_ _A_ ) using VLM and determine the presentation time span for each detected
slide ( _Step_ _B_ ) based on vision/audio hints in this
segment. In _Step_ _A_, we sample a video frame every _N_ _ _sample_ seconds in the focused segment,
and use VLM to compare each frame with the previous one to check whether they contain a same
slide. Whenever the answer is negative, a new slide
is identified. In _Step_ _B_, we leverage Automatic
Speech Recognition (ASR) results (i.e., sentences
with time stamps) of the focused segment and sample a video frame from the middle point of the
time span for each sentence. By comparing the
extracted frame per sentence with its neighborhood
detected slides with image similarity, we can determine which slide the sentence explains, and thus
in this way, estimate the presentation time span of
each detected slide in the focused segment. The detailed approach of _Step_ _B_ is described in Figure 3.
Throughout the segmentation algorithm, we use
a same VLM and the same prompt (shown in Table 11) to determine whether two frames contain a

same slide.



Figure 2: Video segmentation algorithm. SSIM is the
Structure Similarity Index, a measure in image assessment, which gives values between 0 and 1. SSIM gives
high similarity, when two images have visually similar
look but have rather different pixel value (e.g. stretch,
mean-shift). While the approach is prone to give a low
measure value, if visual appearance of the two images
is much different.

|Video #|Video Link|
|---|---|
|V1<br>V2<br>V3<br>V4<br>V5<br>V6|https://www.youtube.com/watch?v=2NgUY8f1pa8<br>https://www.youtube.com/watch?v=2_dZ5GBlRgU<br>https://www.youtube.com/watch?v=_Awekr6-ilg<br>https://www.youtube.com/watch?v=BsXUWddl-as<br>https://www.youtube.com/watch?v=N75gvrZfO24<br>https://www.youtube.com/watch?v=_Jw3DQ7_pxg|



Table 10: LPM videos used for video segemntation
evaluation.


**A.2** **LPM videos for Video Segmentation**


Links for the LPM videos used for video segmentation evaluation can be found in Table 10. Note

only the first 10 minutes are used.


**A.3** **Benchmark datasets Creation from**

**Human Annotation.**


We take the following measures to ensure the quality of MTurk annotation.


  - **Workers qualification:** We recruit MTurk
workers who have “Master qualification”, a
Life-Time-Approval-Rate of at least 90%, and
at least 10 _,_ 000 tasks approved. In addition,
workers are preferred if English is their first
language, as some lecture videos might be
difficult to follow.


  - **Test Exercises and Dummy Questions:**


Figure 3: Detailed algorithm for _Step_ _B_ of the proposed Video Segmentation approach. In the given segment list, the
list of extraction times refer to the times of the sampled frames that are deemed to contain the same slide in _Step_ _A_ .



Workers are asked to complete two exercises,
which are similar to a typical questionnaire,
before working on a questionnaire set. Figure 4 is the starting page for the MTurkers,
and as shown in Figure 5 and 6 show the two
exercises for MTurkers, so that they can understand the assigned task better. If the workers answer the exercises correctly, they will
proceed to a page with explanations for the
exercises, as shown in Figure 7. After checking a box stating that they fully understand
the task, workers will proceed to the questionnaire set. Among the six questionnaires in
the set, one question is exactly the same with
one exercise. This dummy question is used
for quality control. We assume workers who
wrongly answered the dummy question either
rushed through the task or don’t understand
the task. We thus reject the corresponding
questionnaires for quality purpose.


- **Compensation:** In order to attract qualified
workers to work on our questionnaires, the



compensation for the annotators is set as an
hourly rate of $9, which is higher than the US
federal minimum wage of $7.25,


Please note, some of the figures in this section
are put back in the document due to its screenshot
size.


**A.4** **System Settings**


**A.4.1** **Experimental settings for the video**
**segmentation component**

We evaluate our proposed video segmentation approach and compare it with the SOTA PySceneDetect baseline on the datasets listed in Table 1.

Note that our approach also uses PySceneDetect
to conduct first-round video segmentation. In our
approach, we especially tune the PySceneDetect
to minimize the chance of missing slides in segmentation. For fair comparison, for the baseline PySceneDetect, we tune it again to achieve
its best overall performance. We tune all the algorithm parameters on a separate held-out video
set. In the proposed video-segmentation ap

proach, _Threshold_ 1, _Threshold_ 2, _Threshold_ 3
and _Threshold_ 4 are set as 3 seconds, 60 seconds, 0.9, 0.65, respectively, and _N_ _ _sample_ is
set as 60 seconds. For PySceneDetect, AdaptiveDetector is used with adaptive_threshold set
as 1 and min_content_val set as 10. For the baseline PySceneDetect, ContentDetector is adopted
with threshold set as 12. In this work, we adopt
GPT 4 Vision as the VLM used in our proposed
segmentation algorithm.
We set temperature as 0 and max_tokens as 800
for all GPT 4 models used in this work.


**A.4.2** **Experimental settings for the**
**video-segment understanding**
**component**


In this work, GPT-4 Turbo is used for all agents
that require a VLM or LLM in processing in the
Video-Segment Understanding component.
In ASR recognition, the Whisper model is
used to transcribe speech into text for each video.
To further reduce hallucination, among the corrections suggested by the VLM, we only modify _transcript_ _i_ accordingly if the suggestion is
" _term_ _A_ should be _term_ _B_ " and the acoustic difference level between _term_ _A_ and _term_ _B_ (evaluated
using PyPhonetics [4] ) is less than 5, which means
the two terms likely have similar pronunciations.
For ASR correction, the acoustic difference level
between two terms is evaluated using the RefinedSoundex.distance function of PyPhonetics.
For the dynamic critic, _N_ _max_ is set to 10.


**A.4.3** **Experimental Settings of the QA system**
**used in the extrinsic evaluation**


We setup a Retrieval Augmented Generation (RAG)
based QA system for extrinsic evaluation. The
RAG based QA system is composed of a retriever
based on FAISS [5] and a reader using LangChain [6] .
In the index-building process, the three multimodal agents generates the understanding result
respectively (including vision understanding result,
speech transcript, and consolidation information).
For each segment, the retriever builds up an embedding vector for that segmented scene(slide), which
is added to the FAISS index. The embedding model
is SentenceTransformer/all-MiniLM-L6-v2. In the

retrieval and QA phase, the reader wraps up the top
5 retrieval results as context, and sends the question


4 [https://pypi.org/project/pyphonetics/](https://pypi.org/project/pyphonetics/)
5 [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
6 [https://www.langchain.com/](https://www.langchain.com/)



together with the context to GPT 3.5 for answer
generation. For evaluation, we ask GPT-4 Turbo
to generate 2 questions answerable with vision information but not answerable with speech (denoted
as vision questions), 2 questions answerable with
speech but not answerable with vision information
(denoted as speech questions), and 2 questions that
can be best answered with the consolidated infor
mation (denoted as consolidation questions). We
also require the model to generate the ground-truth
answer at the same time for each generated question to facilitate evaluation (See Table 12 for the
prompt). We then run the retrieval-based QA system to generate an answer for each question. The
correctness of the answer generated by QA system
is also evaluated by GPT-4 Turbo using the prompt
shown in Table 13.


**A.5** **Prompts used in the Experiments**


We list the prompts used in the experiments in the
later part of Appendix, due to its table size. Table 11 lists the prompts used in all the agents of the
proposed framework. Table 12 shows the prompts
used to generate the questions together with the
corresponding ground-truth answers for the extrinsic evaluation with QA, the ablation study on ASR
correction, and the ablation study on vision understanding, respectively. Table 13 presents (1) the
prompt that is used to determine whether two vision
understanding results are inconsistent (i.e., containing conflicting information), and (2) the prompt
that is used to determine whether an answer generated by the QA system is correct based on the
question and the corresponding ground-truth. All
prompts are carefully designed.


|Agent/Algorithm|Prompt|
|---|---|
|Video<br>Segmentation|{image 1}<br>{image 2}<br>For the two images provided, if both images appear to be digitally corrupted or distorted, answer with "Yes. Both images are<br>corrupted." and terminate the response. Otherwise, do the following: For the two images provided, each should show a person or<br>several people presenting a slide. Is the slide shown in the frst image the same as the one shown in the second image? Please check<br>very carefully for different texts within the slides. Please start your answer with "Yes. " if the slides are the same and "No. " if the<br>slides are different. Then give an explanation for your answer.|
|Vision<br>Understanding<br>Agent (Baseline<br>System)|{image}<br>Given the image provided, please follow the following rules to generate a description:<br>(1) If this image contains a slide that occupies at least half of the image, please describe the content of that slide in detail. In this<br>case, when generating the description, please only focus on the slide’s content, and ignore the slide’s bottom part such as slide<br>number, footnotes, company logo, etc. as well as other parts of the image. If there are one or more humans presented in the image,<br>please also ignore the humans and don’t include them in the description in this case.<br>(2) Otherwise, if there is no signifcant slide in the image, please simply describe the image.|
|Vision<br>Understanding<br>Agent<br>(Knowledge-<br>Related<br>Enhancement)|{image}<br>Given the image provided, we have the following background knowledge that are likely to be relevant:<br>{retrieved knowledge}<br>Based on the given image and the background knowledge, please follow the following rules to generate a description:<br>(1) If this image contains a slide that occupies at least half of the image, please describe the content of that slide in detail. In this<br>case, when generating the description, please only focus on the slide’s content, and ignore the slide’s bottom part such as slide<br>number, footnotes, company logo, etc. as well as other parts of the image. If there are one or more humans presented in the image,<br>please also ignore the humans and don’t include them in the description in this case.<br>(2) Otherwise, if there is no signifcant slide in the image, please simple describe the image.|
|Vision-Audio<br>Consolidation<br>Agent|Given a video of someone presenting a slide, the text description of the slide (Part_1), and the speech narrative of the presentation<br>(Part_2) are provided below. Please consolidate the two parts into a nice overall description of the video content.<br>———————————————-<br>Part_1. The text description of slide: {vision understanding result}<br>———————————————-<br>Part_2. The speech narrative of the presentation: {speech transcript}|
|Vision-based<br>Keyword<br>Extraction and<br>ASR Error<br>Correction<br>Agent|{image}<br>Transcribed speech explanation for the image above: {speech transcript}<br>————————————<br>Given the provided image and its speech explanation transcript listed above, if the provided image contains a slide, extract the<br>keywords (i.e., important words or phrases) from the slide. Then, check the transcribed speech explanation to see whether any<br>keyword is misrecognized as other word or word sequence with similar pronunciation. Please generate the response following the<br>format below:<br>List of keywords:<br>- keyword1<br>- keyword2<br>- keyword3<br>...<br>(If a detected keyword contains ’,’ or ’;’ in middle, it should be split into multiple keywords. If no keyword is detected or the<br>slide/image is empty, just leave the list of keywords empty.)<br>Answer for whether certain keyword(s) is misrecognized: Yes or No (if the answer is Yes, provide the following expla-<br>nation:)<br>The term *** should be ****.<br>The term *** should be ****.<br>The term *** might be ****.<br>The term *** might be ****.<br>...|
|Knowledge<br>Extraction<br>Agent|Given the text description listed below, summarize the concepts presented in this text description. If the text description is not about<br>a slide presentation, reply with ’No concept extracted’. When generating the output, please follow the following format:<br>Concept: Concept name<br>Knowledge of Concept: explanation..<br>————-<br>Concept: Concept name<br>Knowledge of Concept: explanation..<br>————-<br>Concept: Concept name<br>Knowledge of Concept: explanation..<br>————-<br>....<br>————-<br>Text description:{consolidated understanding result}|
|Dynamic Critic|Critic Agent: Given an image that contains a slide presentation and a description about the slide presentation, decide whether<br>the description can be further improved. If the description is not comprehensive or containing potential mistakes, ask Vision<br>Understanding Agent to improve the description. Otherwise, if the description is comprehensive and accurate, DO NOT repeat the<br>description and just reply ’TERMINATE!!!’ to Admin.<br>***************************************************<br>Vision Understanding Agent: You can generate detailed description of slide presentation based on image provided with previous<br>knowledge. Start your response with ’Vision Understanding Result:’.<br>***************************************************<br>User Proxy: {image}<br>Given the image provided, we have the following background knowledge that might be relevant: {retrieved knowledge}<br>Based on the given image and the background knowledge, please use the following rules to generate a description:<br>(1) If this image contains a slide that occupies at least half of the image, please describe the content of that slide in detail. In this<br>case, when generating the description, please only focus on the slide’s content, and ignore the slide’s bottom part such as slide<br>number, footnotes, company logo, etc. as well as other parts of the image. If there are one or more humans presented in the image,<br>please also ignore the humans and don’t include them in the description in this case.<br>(2) Otherwise, if the image DOES NOT contain a slide that occupies at least half of the image, just reply ’TERMINATE!!!’ to<br>Admin.|


Table 11: Prompts for all agents used in this paper.


|Function|Prompt|
|---|---|
|Question<br>Generation<br>(extrinsic<br>evaluation with<br>QA)|We have a video segment in which a speaker is presenting a slide. The information about the video segement is summarizied into three<br>parts, listed below after the "—-" line. The frst part, denoted as "Part_1. The text description of slide:", provides a text description of the<br>vision information presented in the slide. The second part, denoted as "Part_2. The speech narrative of the presentation:", provides the<br>speech-to-text transcript about what the speaker said about the slide. The third part, denoted as "Part_3. Info-Consolidation Output:",<br>provides the overall description of the video segment, which consolidates the information from the frst part and second part.<br>Your task is to generate six questions satisfying following requirements respectively, and also generate the corresponding answer for each<br>generated question.<br>(1) Generate two questions (referred to as Question_1_vision and Question_2_vision) that are answerable with the slide vision information<br>(i.e., information provided in the frst part), but not answerable with the speech transcript (i.e., information provided in the second part).<br>(2) Generate two questions (referred to as Question_3_speech and Question_4_speech) that are answerable with the speech transcript (i.e.,<br>information provided in the second part), but not answerable with the slide vision information (i.e., information provided in the frst part).<br>(3) Generate two questions (referred to as Question_5_consolidated and Question_6_consolidated) that can be best answered with the<br>consolidated information, i.e., information provided in the third part.<br>In addition, when generting a question or a answer, please directly talk about the knowledge point and avoid the mentioning of slide,<br>speech/speaker, and video as information sources (e.g., avoid the use of "according to the speech", "according to the presentation",<br>"according to according to the consolidated information", "as discussed in the video", "according to the speaker", etc.). For example,<br>instead of asking a question "How does Business Intelligence aid in the preparation of decisions according to the presentation?", please<br>directly ask "How does Business Intelligence aid in the preparation of decisions?". Another example is that instead of answering "The slide<br>indicates that Business Intelligence helps interpret past data to inform future decisions.", please directly answer "Business Intelligence<br>helps interpret past data to inform future decisions.".<br>Please also note that if "slide" must be mentioned in a question, please always include slide title to show which slide it is about.<br>Please provide the generated questions and the answers in the following format:<br>Question_1_vision:<br>Answer_1_vision:<br>Question_2_vision:<br>Answer_2_vision:<br>Question_3_speech:<br>Answer_3_speech:<br>Question_4_speech:<br>Answer_4_speech:<br>Question_5_consolidated:<br>Answer_5_consolidated:<br>Question_6_consolidated:<br>Answer_6_consolidated:<br>———————————————-<br>Part 1. The text description of slide: {vision understanding result}<br>———————————————-<br>Part 2. The speech narrative of the presentation: {speech transcript}<br>———————————————-<br>Part 3. Info-Consolidation Output: {consolidated information}|
|Question<br>Generation<br>(ablation study<br>on vision<br>understanding)|We have a video segment in which a speaker is presenting a slide. Two visual descriptions about the video segment and their differences<br>are provided.<br>Your task is to generate two questions satisfying following requirements, and also generate the corresponding answer for each generated<br>question.<br>Generate two questions (referred to as Question_1_vision and Question_2_vision) that are answerable with Description 2, but not<br>answerable with Description 1.<br>In addition, when generating a question or a answer, please directly talk about the knowledge point and avoid the mentioning of slide,<br>speech/speaker, and video as information sources (e.g., avoid the use of "according to the speech", "according to the presentation",<br>"according to according to the consolidated information", "as discussed in the video", "according to the speaker", etc.). For example,<br>instead of asking a question "How does Business Intelligence aid in the preparation of decisions according to the presentation?", please<br>directly ask "How does Business Intelligence aid in the preparation of decisions?". Another example is that instead of answering "The slide<br>indicates that Business Intelligence helps interpret past data to inform future decisions.", please directly answer "Business Intelligence<br>helps interpret past data to inform future decisions.".<br>Please also note that if "slide" must be mentioned in a question, please always include slide title to show which slide it is about.<br>Please provide the generated questions and the answers in the following example format:<br>Question_1_vision:<br>Answer_1_vision:<br>Question_2_vision:<br>Answer_2_vision:<br>———————————————-<br>Description 1: {vision understanding result 1}<br>———————————————-<br>Description 2: {vision understanding result 2}<br>———————————————-<br>Difference detection result: {difference}|
|Question<br>Generation<br>(ablation study<br>on ASR<br>correction)|We have a video segment in which a speaker is presenting a slide. The transcript about the video segment is provided and the corrections, if<br>any, needed for the transcript.<br>Your task is to generate two questions satisfying following requirements, and also generate the corresponding answer for each generated<br>question.<br>Generate two questions (referred to as Question_3_speech and Question_4_speech) that are answerable with the speech transcript<br>correction, but not answerable without the speech transcript correction.<br>In addition, when generating a question or a answer, please directly talk about the knowledge point and avoid the mentioning of slide,<br>speech/speaker, and video as information sources (e.g., avoid the use of "according to the speech", "according to the presentation",<br>"according to according to the consolidated information", "as discussed in the video", "according to the speaker", etc.). For example,<br>instead of asking a question "How does Business Intelligence aid in the preparation of decisions according to the presentation?", please<br>directly ask "How does Business Intelligence aid in the preparation of decisions?". Another example is that instead of answering "The slide<br>indicates that Business Intelligence helps interpret past data to inform future decisions.", please directly answer "Business Intelligence<br>helps interpret past data to inform future decisions.".<br>Please also note that if "slide" must be mentioned in a question, please always include slide title to show which slide it is about.<br>Please provide the generated questions and the answers in the following example format:<br>Question_3_speech:<br>Answer_3_speech:<br>Question_4_speech:<br>Answer_4_speech:<br>———————————————-<br>Transcript: {original speech transcript}<br>———————————————-<br>Transcript correction needed:: {corrections}|


Table 12: Prompts for question generation.


|Function|Prompt|
|---|---|
|Inconsistency<br>Detection|Given two descriptions about a same slide (listed below), please determine whether there is any<br>confict in meaning between the two descriptions.<br>Please frst answer ’Yes’ or ’No’, and if the answer is ’Yes’, explain what the meaning confict(s)<br>is?<br>———————-<br>Description 1: {vision understanding 1}<br>———————-<br>Description 2: {vision understanding 2}|
|QA Evaluation<br>(Answer<br>Correctness)|Given a question and its ground-truth answer, check whether a automatically generated answer is<br>correct. The question, ground-truth answer, and the automatically generated answer are all listed<br>below. In your response, please simply say "correct" if you think the generated answer contains<br>consistent information as the ground-truth answer, simply say "wrong" if you think the generated<br>answer is wrong (i.e., conficting with the information in the ground-truth answer, or failing to<br>include the key messages in the ground-truth answer), and simply say "correct but with additional<br>information" if you think the generated answer contains the correct answer but includes additional<br>information not mentioned in the ground-truth answer.<br>Question: {question}<br>Ground-truth answer: {ground truth answer}<br>Automatically generated answer: {predicted answer}|



Table 13: Prompts for inconsistency detection given two vision understanding results and for answer correctness
evaluation.


Figure 4: MTurk questionnaire instructions page.


Figure 5: MTurk questionnaire exercise question 1.


Figure 6: MTurk questionnaire exercise question 2. Note that the questionnaires that need annotation share the same
format as this exercise question.


Figure 7: Answer explanation for MTurk questionnaire exercise question 2.


