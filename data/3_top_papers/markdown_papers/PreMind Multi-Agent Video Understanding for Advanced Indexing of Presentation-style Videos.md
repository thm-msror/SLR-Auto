# **PreMind**: Multi-Agent Video Understanding for Advanced Indexing of **Pre**sentation-style Videos

#### Kangda Wei‡\*, Zhengyu Zhou, Bingqing Wang, Jun Araki, Lukas Lange, Ruihong Huang‡ , Zhe Feng

Bosch Research North America & Bosch Center for Artificial Intelligence (BCAI) ‡Department of Computer Science and Engineering, Texas A&M University kangda@tamu.edu, {Zhengyu.Zhou2, Bingqing.Wang, Jun.Araki}@us.bosch.com, Lukas.Lange@de.bosch.com, huangrh@cse.tamu.edu, Zhe.Feng2@us.bosch.com

## Abstract

In recent years, online lecture videos have become an increasingly popular resource for acquiring new knowledge. Systems capable of effectively understanding/indexing lecture videos are thus highly desirable, enabling downstream tasks like question answering to help users efficiently locate specific information within videos. This work proposes PreMind, a novel multi-agent multimodal framework that leverages various large models for advanced understanding/indexing of presentation-style videos. PreMind first segments videos into slide-presentation segments using a Vision-Language Model (VLM) to enhance modern shot-detection techniques. Each segment is then analyzed to generate multimodal indexes through three key steps: (1) extracting slide visual content, (2) transcribing speech narratives, and (3) consolidating these visual and speech contents into an integrated understanding. Three innovative mechanisms are also proposed to improve performance: leveraging prior lecture knowledge to refine visual understanding, detecting/correcting speech transcription errors using a VLM, and utilizing a critic agent for dynamic iterative self-reflection in vision analysis. Compared to traditional video indexing methods, PreMind captures rich, reliable multimodal information, allowing users to search for details like abbreviations shown only on slides. Systematic evaluations on the public LPM dataset and an internal enterprise dataset are conducted to validate PreMind's effectiveness, supported by detailed analyses.

# 1 Introduction

Recent technological advancements have led to the proliferation of online videos, which increasingly become an important source for learning new knowledge [\(Soni and Dubey,](#page-9-0) [2019\)](#page-9-0). Presentationstyle lecture videos, which mainly present slides sequentially, are widely used for online courses and

trainings [\(Mishra et al.,](#page-9-1) [2023\)](#page-9-1). Systems that can effectively understand and index rich content of such videos thus become desirable [\(Mishra et al.,](#page-9-1) [2023\)](#page-9-1), which could lead to advanced downstream applications such as question answering (QA) based on video details in various modalities. However, state-of-the-art (SOTA) approaches for indexing video content [\(Iyer et al.,](#page-8-0) [2019;](#page-8-0) [Saoudi and Jai An](#page-9-2)[daloussi,](#page-9-2) [2021\)](#page-9-2) remain unsatisfactory, as they fail to capture detailed multimodal information. With the rapid advancements in large language model (LLM) technology [\(Zhao et al.,](#page-10-0) [2023;](#page-10-0) [Chang et al.,](#page-8-1) [2024\)](#page-8-1), Video Large Language Models (Vid-LLMs) [\(Lin et al.,](#page-9-3) [2023;](#page-9-3) [Team et al.,](#page-9-4) [2024\)](#page-9-4) have emerged, enabling users to directly ask questions about provided videos [\(Pan et al.,](#page-9-5) [2023\)](#page-9-5). However, Vid-LLM cannot answer questions that require systems to find the answers from a large number of videos due to its design and computation limitation.

In this work, we propose PreMind, a novel multiagent multimodal framework that leverages various large models to capture detailed multimodal information in presentation-style lecture videos, leading to information-rich indexes that can benefit downstream tasks such as QA. For this work, we adopt the broad understanding of agents, viewing agents as system components that each has its own goals and work together to achieve a common goal [\(Wang et al.,](#page-10-1) [2024a\)](#page-10-1). PreMind begins with a video segmentation component that combines a SOTA vision-based approach for segmentation with VLM to efficiently and reliably segment a video into many video segments, each covering one presentation slide. Then, PreMind generates textual description for each segment with an advanced video-segment understanding component. For each segment, the component leverages appropriate agents to understand visual information, capture speech narrative, and generate a consolidated description. The component also involves innovative mechanisms to (i) improve vision under-

<sup>\*</sup>Work done during an internship at Bosch Research North America.

standing by leveraging knowledge learned previously in the video lecture, (ii) automatically correct speech recognition errors with VLM based on visual information and speech transcript, and (iii) further improve vision understanding through dynamic self-refinement with a critic agent. Based on the information-rich indexes created, downstream tasks such as retrieval-based QA and summarization can be implemented for various applications.

We evaluate PreMind using public LPM dataset [\(Lee et al.,](#page-8-2) [2023\)](#page-8-2) as well as an enterprise internal dataset. Both intrinsic evaluation and extrinsic evaluation are conducted. The evaluation results show the effectiveness of PreMind, demonstrating the value of capturing detailed multi-modal information in indexes, as well as the benefits of the proposed mechanisms for the video understanding/indexing task. To summarize, our contributions include:

- We introduce PreMind, a multi-agent multimodal framework that uses various large models to capture accurate and detailed multi-model information from presentation-style lecture videos, which can further benefit the possible downstream applications such as QA.
- We demonstrate the effectiveness of PreMind on the public LPM dataset [\(Lee et al.,](#page-8-2) [2023\)](#page-8-2) and an enterprise internal dataset through intrinsic and extrinsic evaluations.
- We conduct ablation studies to evaluate the efficacy of different mechanisms within PreMind, and present comprehensive analyses of framework's efficiency as well as case studies.

# 2 Related Work

Lecture Video Indexing Indexing lecture videos is an increasingly crucial task for enhancing access to relevant content within educational materials, which often involves video segmentation and information extraction from video segments. For segmentation, [Chand and Ogul](#page-8-3) ˘ [\(2021\)](#page-8-3) used Voice Activity Detection and Gaussian Mixture Models to segment videos based on speech. [Shah et al.](#page-9-6) [\(2015\)](#page-9-6) aligns spoken content with Wikipedia text for better segmentation. [Jeong et al.](#page-8-4) [\(2012\)](#page-8-4) applied SIFT for precise slide detection. For information extraction, Optical Character Recognition (OCR) helps retrieve text from slides and Automatic Speech Recognition (ASR) helps obtaining the speech in text format. The extracted text is then used for indexing the video in a content-based

manner. [Ip and Chan](#page-8-5) [\(1997\)](#page-8-5) used OCR for hierarchical indexing, while [Yang and Meinel](#page-10-2) [\(2014\)](#page-10-2) combined ASR and OCR for comprehensive video search. Multimodal approaches, like those by [Ya](#page-10-3)[mamoto et al.](#page-10-3) [\(2003\)](#page-10-3) and [Lin et al.](#page-9-7) [\(2004\)](#page-9-7), integrate ASR and OCR for improved indexing. However, indexes generated from previous works [\(Yang et al.,](#page-10-4) [2011b;](#page-10-4) [Ma et al.,](#page-9-8) [2017;](#page-9-8) [Yang et al.,](#page-10-5) [2011a;](#page-10-5) [Deb](#page-8-6)[nath et al.,](#page-8-6) [2023;](#page-8-6) [Arazzi et al.,](#page-8-7) [2023;](#page-8-7) [Medida and](#page-9-9) [KASARAPU,](#page-9-9) [2021\)](#page-9-9) do not have rich multi-modal information but rather simple text description obtained from OCR or ASR. Despite advancements, challenges remain in achieving high accuracy in indexing due to ASR errors, visual variations, and lecture complexity. However, these issues can be mitigated with the help of LLMs and VLMs. To the best of our knowledge, our proposed framework is the first to utilize Large Models to enhance video indexing quality.

Video Large Language Models Vid-LLMs are widely used for tasks involve video understanding [\(Abdullah et al.,](#page-8-8) [2024\)](#page-8-8). For Vid-LLM, it typically uniformly samples a certain numbers of frames from videos and utilize a visual encoder [\(Dosovit](#page-8-9)[skiy et al.,](#page-8-9) [2020;](#page-8-9) [Radford et al.,](#page-9-10) [2021\)](#page-9-10) to convert each frame into vector representation. Then, an adapter is used to map the video embeddings from visual semantic space to text semantic space of LLMs. Textual embeddings of instructions are then added generating responses for downstream tasks [\(Zhang et al.,](#page-10-6) [2023;](#page-10-6) [Maaz et al.,](#page-9-11) [2024;](#page-9-11) [Lin et al.,](#page-9-12) [2024;](#page-9-12) [Song et al.,](#page-9-13) [2024;](#page-9-13) [Chen et al.,](#page-8-10) [2023;](#page-8-10) [Ma et al.,](#page-9-14) [2024\)](#page-9-14), or a specially designed task-specific head can be used to perform regression tasks [\(Yu et al.,](#page-10-7) [2023;](#page-10-7) [Huang et al.,](#page-8-11) [2024;](#page-8-11) [Ren et al.,](#page-9-15) [2024;](#page-9-15) [Li et al.,](#page-8-12) [2024\)](#page-8-12). For Vid-LLMs, although previous works have demonstrated impressive video understanding capabilities, they are not well-suited for contentbased lecture video indexing. One limitation lies in their sampling strategy, which is suboptimal as it can result in redundant information or the omission of critical details [\(Wang et al.,](#page-10-8) [2024b\)](#page-10-8), thus not suitable for video indexing that requires segmentation. Additionally, these systems are mostly designed to process one query about one video or several at a time in an online fashion, thus not suitable for indexing videos.

## 3 Method

The overall structure of PreMind is illustrated in Figure [1.](#page-2-0) It consists of a video segmentation com-

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 1: Illustration of the proposed PreMind framework.

ponent and a video-segment understanding component, generating three understanding results based on vision, speech, and consolidated information, respectively, as the output indexes. Given a number of lecture videos, PreMind processes the videos one by one, and the resulting pool of indexes can be used in downstream tasks.

### 3.1 Video Segmentation

The video segmentation component attempts to segment a training video into multiple segments, each covering the presentation of one slide. We adopt the state-of-the-art PySceneDetect (Gruzman and Kostenkova, 2014; Reddy and Jadhav, 2015) to conduct the first-round segmentation. PySceneDetect faces two major challenges for this task, (1) often missing slide with similar layout as the precedent one and (2) splitting the presentation of a single slide into multiple segments due to background changes. Therefore, we innovatively use VLM to refine the segmentation when needed. After the first-round segmentation, for long segments (>1 minute) detected by PySceneDetect, we apply VLM to re-detect slides  $(Step_A)$  in that segment with the aim of catching the missed slides. The time span for each newly detected slide is determined using vision and audio cues ( $Step_B$ ). For other segments, VLM is leveraged to merge the current segment with the previous one if the two segments are deemed as presenting a same slide. More details of our proposed video segmentation algorithm is shown in Appendix A.1.

### 3.2 Video-Segment Understanding

With the obtained video segments, denoted as  $\{S_1, S_2, ..., S_n\}$ , the video-segment understanding component attempts to understand/index the content of each segment  $S_i$  with a multi-agent solution, as shown in Figure 1. We develop the understanding component in an incremental way. First, we design a baseline understanding system, which adopts three separate agents to extract audio, vision, and consolidated information from the segment. We then leverage knowledge to enhance the vision understanding, extracting knowledge per slide, keeping a knowledge memory for the lecture in focus, and leveraging the previously-learned knowledge to understand the current slide. We further reduce the impact of ASR errors on understanding results by leveraging VLM to automatically correct ASR errors based on both speech transcription and slide visual content. Finally, we introduce a critic agent to dynamically refine vision understanding result in a self-reflection manner. Prompt templates for all agents/algorithms can be found in Appendix A.5.

### 3.2.1 Baseline System

- Audio Understanding Agent: This agent uses a ASR model to generate a speech transcript, denoted as  $transcript_i$ , for each segment  $S_i$ . In practice, the ASR model is applied to transcribe the whole video in focus. After the video segmentation process, the audio understanding agent extracts  $transcript_i$  for  $S_i$  from the whole transcript based on the starting/ending times of  $S_i$ .
- Vision Understanding Agent: Given S<sub>i</sub>, this
  agent generates a detailed description of the slide
  presented in the segment using a SOTA VLM. It

- samples a (representative) video frame from S<sup>i</sup> and asks the VLM to describe the slide shown in that frame image in detail. The description generated is the vision understanding result, denoted as vision\_understanding<sup>i</sup> .
- Vision-Audio Consolidation Agent: Based on the audio and vision understanding results, the consolidation agent further generates a consolidated understanding result, denoted as consolidated\_information<sup>i</sup> , to provide a good overall understanding about what is presented in the video segment S<sup>i</sup> .

## 3.2.2 Knowledge-Related Enhancement

- Knowledge Memory: Given a segment S<sup>i</sup> , knowledge presented in previous slides may be helpful to understand the current slide. We maintain a knowledge memory KM for the lecture that S<sup>i</sup> is part of. The knowledge memory contains entries in the following format: knowledge<sup>m</sup> = {embeddingm, namem, explanationm}, where m ∈ M and M is the number of entries in KM. In each entry, name<sup>m</sup> stores a concept name, such as Product Lifecycle Management, explanation<sup>m</sup> stores the explanation of that concept, and embedding<sup>m</sup> stores the embedding representation of the concept name, which is obtained by a SentenceTransformer model[1](#page-3-0) , for knowledge retrieval. For each lecture, KM is initially empty. It is then updated when new knowledge is extracted, starting from the first segment of the lecture. Notice that, unlike previous works described in [\(Hatalis et al.,](#page-8-14) [2024\)](#page-8-14), where memory is used for tasks that only involve text, we keep a knowledge memory that contain multi-modal information from previous video segments to help understand current segment.
- Knowledge Extraction Agent: This agent extracts new knowledge from the consolidated understanding result for S<sup>i</sup> , and updates KM correspondingly. It asks a SOTA LLM to extract concepts, each including a concept name and an explanation, from consolidated\_information<sup>i</sup> . For each extracted concept, embedding representation of the concept name, e, is computed with the SentenceTransformer, and entries in KM are ranked by cosine similarity between e and embeddingm. If the top-ranked entry has a similarity score less than 0.7, the extracted concept is deemed new and we update the KM by inserting

- the concept as a new entry. Otherwise, the topranked entry is deemed to have similar concept with the extracted concept. In this case, we update the top-ranked entry knowledge<sup>j</sup> by (1) appending the explanation of the extracted concept to explanation<sup>j</sup> and (2) updating embedding<sup>j</sup> as the rolling average of embedding<sup>j</sup> and e.
- Keyword Extraction (Part of the Keyword Extraction and ASR Error Correction Agent): Given S<sup>i</sup> , we extract a set of keywords, denoted as keywords<sup>i</sup> , from the slide vision using a VLM. These keywords are then used to retrieve relevant knowledge from KM. In this work, we merge the tasks of keyword extraction and ASR error correction into one agent, using one VLM prompt to accomplish the two tasks at the same time for improved efficiency and performance.
- Keyword-based Knowledge Retrieval Agent: With keywords<sup>i</sup> extracted from S<sup>i</sup> , this agent retrieves relevant knowledge from KM to facilitate vision understanding. For each keyword in keywords<sup>i</sup> , the agent computes its embedding representation using the SentenceTransformer model, and then ranks the KM entries by cosine similarity between the keyword's embedding and embeddingm. Among the top 10 entries, those entries with similarity score larger than 0.7 are deemed relevant, and are provided to the vision understanding agent as context.

### 3.2.3 ASR Correction

Mistakes made by ASR model have a negative impact on the quality of generated understanding results. To reduce ASR errors, this work proposes an innovative approach that leverages reliable visual information, i.e., the keywords extracted from slide vision, to correct ASR errors using a VLM, while previous works on ASR correction mainly rely on ASR results alone [\(Ma et al.,](#page-9-17) [2023,](#page-9-17) [2025\)](#page-9-18). For the keyword extraction and ASR error correction agent, given S<sup>i</sup> , we use one prompt to ask the VLM to (1) extract keywords shown in the slide, and (2) check transcript<sup>i</sup> to identify possible ASR mistakes made on the keywords and make correction suggestions. The combination of the two tasks not only enhances efficiency but also benefit the performance of ASR correction, as constraining the correction scope to keywords helps reduce hallucination of VLM on ASR correction.

<span id="page-3-0"></span><sup>1</sup> <https://www.sbert.net/>

## 3.2.4 Dynamic Critic

LLM self-reflection has been found effective for various text processing tasks[\(Madaan et al.,](#page-9-19) [2023;](#page-9-19) [Liang et al.,](#page-9-20) [2024\)](#page-9-20). In this work, we extend this technique to multi-modal data, introducing a critic agent to enhance vision understanding through iterative reflection. Given the slide of S<sup>i</sup> , the retrieved knowledge, and the vision\_understanding<sup>i</sup> generated by the Vision Understanding agent, the critic agent aims to identify defects of vision\_understanding<sup>i</sup> , such as counting mistakes and missing figures in description. With the feedback from the critic agent, the vision understanding agent further improves the understanding result and sends the updated result to the critic agent to review. This process iterates until the critic agent is satisfied with the understanding result. We realize this dynamic reflection mechanism by grouping the critic agent and the vision understanding agent into a AutoGen [\(Wu et al.,](#page-10-9) [2023\)](#page-10-9) groupchat, which also includes an admin agent to assist the chatting function. We configure the groupchat to allow at maximum Nmax visionunderstanding/critic calls in the reflection iteration. For early termination, we define the chat termination condition as the 'TERMINATE!!!' command issued by the Critic Agent.

## 4 Experiments and Analyses

### 4.1 Dataset

<span id="page-4-0"></span>

| Dataset | Video # | Video Segments # | Total Video<br>Length(mins) |
|---------|---------|------------------|-----------------------------|
| LPM     | 6       | 54               | 60.0                        |
| EI      | 7       | 37               | 56.3                        |

Table 1: Dataset statistics for video segmentation evaluation. Segments refer to ground-truth segments manually labeled.

<span id="page-4-1"></span>

| Dataset | Video # | Video Segments # | Total Video<br>Length(hours) |
|---------|---------|------------------|------------------------------|
| LPM     | 188     | 1366             | 28.17                        |
| EI      | 66      | 264              | 5.96                         |

Table 2: Dataset statistics for evalution on understanding performance. Segments are manually labeled for LPM data but automatically detected for EI data.

We evaluate PreMind on the public LPM dataset [\(Lee et al.,](#page-8-2) [2023\)](#page-8-2) and an enterprise internal (EI) dataset. The LPM dataset contains YouTube lectures across 10 different categories (e.g., biology, psychology), having more than 180 hours of

videos and providing manually- segmented slidepresentation segments for over 9,000 slides in total. The internal dataset contains 66 videos (6 hours in total) on various enterprise-training topics. The videos in both datasets are presentation-style lectures, though the layout of the slide display in the videos varies. In this work, we randomly sample a subset of videos per dataset to evaluate video segmentation performance, as shown in Table [1.](#page-4-0) Note that for the LPM videos sampled (listed in Appendix [A.2\)](#page-11-1), the first 10 minutes of each video is used in video-segmentation evaluation. For the evaluation of understanding performance, we select another subset of lectures from the LPM data, which contains almost 30 hours of videos in total, due to computational constraints. We use this LPM subset and the whole EI dataset to evaluate the video-understanding approaches as well as QA performance in extrinsic evaluation, as listed in Table [2.](#page-4-1) We construct the LPM subset for understanding evaluation by (1) selecting the first three lecture videos from each category except dental, and (2) for dental videos, which contain 19 subcategories and each video is only around 5 minutes, selecting all the videos of the first 3 subcategories.

### 4.2 Video Segmentation Evaluation

## 4.2.1 Settings

We evaluate our proposed video segmentation approach and compare it with PySceneDetect in Table [1.](#page-4-0) Details on the parameter tuning as well as algorithm configurations are provided in Appendix [A.4.1.](#page-12-0) GPT-4 Vision is used in our proposed segmentation algorithm. For video-segmentation evaluation, we report Precision, Recall, and F-1 score for detecting video segments. We also report the Intersection over Union (IoU) score (ranging from 0 to 1), which specifies the amount of overlap on time span between the predicted and ground truth segments. IoU is calculated as:

$$IoU = \frac{|A \cap B|}{|A \cup B|} \tag{1}$$

where A and B are the time spans of the the predicted and ground-truth segments, respectively.

#### 4.2.2 Video Segmentation Results

<span id="page-5-0"></span>

| Dataset   | LPM           |        | EI            |        |
|-----------|---------------|--------|---------------|--------|
| Algorithm | PySceneDetect | Ours   | PySceneDetect | Ours   |
| Precision | 77.50         | 100.00 | 94.59         | 100.00 |
| Recall    | 88.33         | 96.55  | 80.00         | 100.00 |
| F1        | 82.56         | 98.24  | 86.69         | 100.00 |
| IoU       | 0.74          | 0.80   | 0.63          | 0.91   |

Table 3: Video segmentation results.

The evaluation results on video segmentation are shown in Table 3. We can see that our proposed segmentation approach significantly outperforms SOTA vision-based approach, demonstrating the power of VLM. Our approach achieves almost perfect results on EI, successfully detecting all presented slides. The performance on LPM is somehow imperfect, as the LPM data contains occasional occurrences of animations/demonstrations, making it more challenging. From efficiency aspect, as our proposed segmentation approach only applies VLM when needed, the computational overhead introduced with VLM is minimized, and the segmentation efficiency is largely maintained (as will be further discussed in Section 4.4).

# 4.3 Video-Segment Understanding Evaluation

### 4.3.1 Settings

We conduct video-segment understanding experiments based on the datasets listed in Table 2. For these experiments, we directly use the provided manual segmentation for the LPM data and use our proposed segmentation algorithm to segment the EI data. Using video segments obtained in this manner, four incrementally developed videosegment understanding approaches are applied to each dataset: (1) the baseline system, (2) plus knowledge-related enhancement, (3) plus ASR correction, and (4) plus dynamic critic. Each approach is applied individually, generating a corresponding set of understanding results, which are then evaluated and compared to assess their performance. In this set of experiments, Whisper<sup>2</sup> is used to generate ASR result, and GPT-4 Turbo is used for all agents that require a VLM/LLM in processing. Algorithm parameters for video-segment understanding are determined empirically, with details provided in Appendix A.4.2.

#### 4.3.2 Intrinsic Evaluation

**Evaluation Approach** We first evaluate the four proposed video-segment understanding approaches

on vision understanding performance. As this is a challenging task by nature, we adopt human annotation to ensure the evaluation quality, and propose a pairwise comparison schema for evaluation. Given a pair of vision understanding results to compare together with the corresponding slide image, to reduce the labeling workload for human, we first ask GPT-4 Turbo to determine whether the two understanding results are consistent in meaning ( prompt listed in Appendix Table 13).

When an inconsistency is detected, the result pair is sent to humans for manual quality comparison. For each pair requiring manual evaluation, a questionnaire is prepared, asking annotators to determine which result is better and explain their reasoning based on the slide image and relevant knowledge. Each questionnaire is completed by three annotators, and the final judgment is determined by a majority vote. For EI data, due to confidential issue, we recruit internal associates to annotate related questionnaires. For the LPM data, we leverage Amazon Mechanical Turk <sup>3</sup> for the annotation, designing special measures to ensure high quality of annotation (e.g., selecting workers having trustful records, using dummy questions to reject irresponsible answers). The details about the measures and the questionnaire examples are provided in Appendix A.3.

<span id="page-5-3"></span>

|     |              | Total # | Win# | Tie # | Lose # | Win %  | (Win+Tie) % |
|-----|--------------|---------|------|-------|--------|--------|-------------|
| LPM | B+K vs B     | 155     | 76   | 49    | 30     | 49.03% | 80.65%      |
|     | B+K+A vs B   | 155     | 91   | 39    | 25     | 58.71% | 83.9%       |
|     | B+K+A+D vs B | 160     | 94   | 32    | 34     | 58.75% | 78.75%      |
| EI  | B+K vs B     | 35      | 16   | 10    | 9      | 45.71% | 74.29%      |
|     | B+K+A vs B   | 39      | 18   | 13    | 8      | 46.15% | 79.49%      |
|     | B+K+A+D vs B | 34      | 16   | 13    | 5      | 47.06% | 85.29%      |

Table 4: Comparison on vision understanding results with human evaluation. B refers to the baseline system; K refers to the knowledge-related enhancement mechanism; A refers to the ASR correction mechanism; D refers to the dynamic critic self-reflection mechanism.

**Results** Table 4 reports the comparison results on vision understanding performance for different video-segment understanding settings against the baseline system. For each pair of approaches in comparison, we list (1) the total number of cases where the final vision-understanding results generated by the two approaches are deemed different by GPT-4 Turbo, and (2) among those cases, which are sent to human annotators to label, the competition results according to the human annotation. If the first approach is deemed better than the second approach according to annotation results, it wins

<span id="page-5-1"></span><sup>2</sup>https://openai.com/index/whisper/

<span id="page-5-2"></span><sup>3</sup>https://www.mturk.com/

<span id="page-6-0"></span>

| Question      | LPM        |       |        |        | Accuracy on LPM (given index type) | EI         |       |        |        | Accuracy on EI (given index type) |
|---------------|------------|-------|--------|--------|------------------------------------|------------|-------|--------|--------|-----------------------------------|
| Type          | Question # | All   | Vision | Speech | Consolidation                      | Question # | All   | Vision | Speech | Consolidation                     |
| Vision        | 2716       | 78.57 | 78.76  | 33.59  | 74.96                              | 392        | 76.27 | 75.77  | 23.21  | 68.62                             |
| Speech        | 2706       | 70.81 | 49.04  | 67.63  | 68.92                              | 394        | 82.74 | 46.45  | 87.06  | 71.83                             |
| Consolidation | 2722       | 86.41 | 79.21  | 66.13  | 87.88                              | 394        | 90.36 | 79.19  | 79.44  | 88.83                             |
| Overall       | 8144       | 78.61 | 69.03  | 55.77  | 77.27                              | 1180       | 83.13 | 67.11  | 63.31  | 76.44                             |

Table 5: Question-answering performance on LPM and DC data.

the second approach. If it is deemed similar to the second approach, a tie occurs. Otherwise, the first approach loses the competition. From Table [4,](#page-5-3) we can see that adding knowledge-related enhancement, ASR correction, and dynamic critic gradually improves the quality of vision understanding for both LPM and EI data. Note that the improvement brought by ASR correction on vision understanding is indirect (i.e., better speech transcript → better consolidated info → better extracted knowledge → better knowledge-assisted vision understanding). For lose cases, we suspect these are caused by the noises in the knowledge that is retrieved and fed into the vision understanding agent. These noises may distract the agent from important information during the generation of slide description.

### 4.3.3 Extrinsic Evaluation with QA

Evaluation Approach We conduct extrinsic evaluation by applying the generated indexes for QA tasks to show the impact of the proposed videosegment understanding component on downstream application. With the three indexes (i.e., vision understanding result, speech transcript, and consolidation information) generated for each video segment in a dataset, we setup a retrieval-augmented QA system. In this set of experiments, we adopt the complete video-segment understanding component (i.e., the B+K+A+D setting) to generate the understanding results as textual indexes. For evaluation, we ask GPT-4 Turbo to generate 6 questions together with the ground-truth answers. Among the 6 questions, 2 focus on speech, 2 focus on slide vision, and 2 focus on the consolidated information. After using QA to generate an answer for each question, we further leverage GPT-4 Turbo to evaluate the answer based on the ground-truth. The details about the QA settings, question/groundtruth generation, and the GPT based evaluation are all provided in Appendix [A.4.3.](#page-13-2)

Results We evaluate the QA performance as the accuracy of the answers generated by QA. The results are reported in Table [5.](#page-6-0) The performance of only using vision-understanding-result indexes,

<span id="page-6-1"></span>

|                      | LPM   | EI    |
|----------------------|-------|-------|
| Baseline + Knowledge | 67.82 | 70.59 |
| + ASR correction     | 73.91 | 82.35 |

Table 6: QA performance (Accuracy) given the question set for ASR-correction ablation study (870 questions for LPM and 102 questions for EI).

only using speech-transcript indexes, only using consolidated-information indexes, and using all types of indexes in retrieval are evaluated per question type. The results demonstrate the value of capturing multimodal information in indexes for QA. Each type of indexes (e.g., vision understanding result) is advantageous for certain kind of questions (e.g., those asking for vision details not presented by speech). And adopting indexes of all types bring the best overall performance.

### 4.3.4 Ablation Study

Leveraging the QA system, we further conduct ablation study on the benefit of the proposed ASR correction procedure by comparing the system performance with and without this procedure. For this evaluation, we generate another set of questions together with their ground-truth in a similar way as before. In this case, based on the predicted ASR corrections, we ask GPT-4 Turbo to generate questions answerable with the corrected speech transcript but not answerable without the corrections (See Table [12](#page-15-0) for the prompt). With this set of questions, using only speech-transcript indexes in the QA system, we evaluate the answer accuracy as before. We focus on the speech-transcript indexes here for the easiness of interpretation, as the major impact of ASR Correction is on audio understanding result. The evaluation results are shown in Table [6.](#page-6-1) We can see that ASR correction brings substantial improvements on QA accuracy for both datasets, demonstrating the benefit of correcting ASR errors on video understanding.

In a similar manner, we also conduct another ablation study on vision-understanding aspect. In this case, we focus on each pair of vision-understanding results that has one result determined as better than the other in human annotation, and ask GPT-4

<span id="page-7-1"></span>

|                  | LPM   | EI    |
|------------------|-------|-------|
| Baseline         | 36.88 | 39.40 |
| + Knowledge      | 46.86 | 48.48 |
| + ASR Correction | 46.20 | 51.52 |
| + Dynamic Critic | 83.13 | 69.70 |

Table 7: QA performance (Accuracy) given the question set for vision-understanding ablation Study (320 questions for LPM and 66 questions for EI).

Turbo to generate questions answerable with the better result but not with the other one (See Table 12 for the prompt). With these questions, we evaluate the four proposed video-segment understanding settings with QA, using only the vision-understanding indexes in the QA system in this case. The results are reported in Table 7. We can see that both the knowledge-related enhancement and the dynamic critic mechanism bring substantial improvements on performance. For ASR correction, which only has indirect impact on vision understanding, it unsurprisingly brings only mild improvement on EI, and brings no improvement on LPM probably due to generation noises.

## 4.3.5 Case Study

<span id="page-7-2"></span>

|                | LPM | The trends of the metabolic rate of endothermy and ectorthermy are described in reverse. With external knowledge of metabolic rate, the trends are correctly described.               |
|----------------|-----|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Knowledge      | EI  | The description of "ERP layer" wrongly includes certain stages of "PLM layer" due to confusing slide layout. With external knowledge of PLM and ERP leveraged, this problem is fixed. |
|                | LPM | A professor's name "Bisque" is correctly changed to "Bisk".                                                                                                                           |
| ASR Correction | EI  | A domain specific term "Nexeed", previously recognized as "NextSeat", is corrected.                                                                                                   |
| Dynamic Critic | LPM | The starting point for a shaded area in a figure previously recognized at 6.5, is correctly fixed to 5.5.                                                                             |
|                | EI  | A Venn diagram previously missing from the description is added.                                                                                                                      |

Table 8: Case study for improvement brought by knowledge-related enhancement, ASR correction, and dynamic critic.

We also examine improvements achieved through knowledge enhancement, ASR correction, and dynamic critic separately. Table 8 shows one example per dataset for each mechanism. We observe that knowledge is useful to help VLM understand ambiguous part of slides. For example, on a slide, several stages of PLM (product lifecycle management) are listed right below the ERP (enterprise resource planning) block. Although the PLM stages are not connected to the ERP block in the slide, the VLM wrongly describes that ERP includes those stages.

<span id="page-7-3"></span>

| Video Segmentation          | # API calls             | Video Length (s) |
|-----------------------------|-------------------------|------------------|
| EI Data<br>LPM Data         | $\approx 4$ $\approx 3$ | 483<br>600       |
| Video-Segment Understanding | ≈ 3<br>  # API calls    | Time Lapse (s)   |
| Baseline                    | 2                       | 40               |
| + Knowledge                 | 4                       | 65               |
| + ASR Correction            | 4                       | 65               |
| + Dynamic Critic            | $\approx 9$             | 107              |

Table 9: Efficiency of the PreMind framework. For the video segmentation procedure, the average length of the videos and the average number of API calls needed to segment one video are listed. For video-segment understanding, the average number of API calls and the average total time for generating indexes per video segment are listed. ASR API call is **not included** as we only need to call ASR model once per video.

This problem is fixed when definitions of PLM and ERP are retrieved and provided to the VLM for vision understanding. The ASR correction procedure excels at refining domain-specific terms and resolving uncommon or ambiguous names. The dynamic critic self-reflection mechanism effectively addresses specific errors made by the VLM in slide descriptions, such as misinterpreting numbers in figures, making counting errors, and missing visual elements in the slide descriptions.

### <span id="page-7-0"></span>4.4 Framework Efficiency

We evaluate the efficiency of our proposed framework PreMind in Table 9. Note that the framework efficiency is largely determined by the API calls used in various components. For each video, one ASR API call is needed to generate a transcript for the whole video. For the video segmentation procedure, except the VLM API calls, the additional processing time beyond the first-round PySceneDetect is negligible. For the video-segment understanding, API calls occupy the majority of processing time, while the average time for retrieving knowledge from the knowledge memory and the average time for correcting detected ASR errors per video segment are 364 ms and 10 ms, respectively.

### 5 Conclusion

This work proposes PreMind, a novel framework to understand/index rich multimodal information for presentation-style lecture videos, with the aim of enabling advanced downstream applications such as QA. PreMind involves two components: video segmentation and video-segment understanding. The video segmentation procedure combines VLM with PySceneDetect to achieve the desired

performance with high efficiency. The videosegment understanding component not only captures visual/audio/consolidated information from video using large models, but also introduces three innovative mechanisms to improve the understanding performance. We evaluate PreMind on the public LPM dataset and an internal dataset and encouraging experimental results are achieved.

## Limitations

PreMind relies heavily on proprietary LLMs and VLMs for both video segmentation and understanding tasks. Open-sourced models may be used to substitute proprietary models, but the performance may be effected. PreMind is optimized for presentation-style lecture videos. Its generalizability to other video formats, such as freestyle videos without any slides presented, has not been explored. The reliance on human annotation for evaluation introduces subjectivity, particularly in vision understanding tasks where judgment about description quality can vary across annotators. Future work will address these limitations by exploring lightweight and open-sourced model alternatives and expanding the evaluation to include more diverse datasets and video formats.

# References

- <span id="page-8-8"></span>Hasnat Md Abdullah, Tian Liu, Kangda Wei, Shu Kong, and Ruihong Huang. 2024. [Ual-bench: The first com](https://arxiv.org/abs/2410.01180)[prehensive unusual activity localization benchmark.](https://arxiv.org/abs/2410.01180) *Preprint*, arXiv:2410.01180.
- <span id="page-8-7"></span>Marco Arazzi, Marco Ferretti, and Antonino Nocera. 2023. [Semantic hierarchical indexing for online](https://api.semanticscholar.org/CorpusID:259044832) [video lessons using natural language processing.](https://api.semanticscholar.org/CorpusID:259044832) *Big Data Cogn. Comput.*, 7:107.
- <span id="page-8-3"></span>Dipesh Chand and Hasan Ogul. 2021. ˘ [A framework](https://doi.org/10.1109/SAMI50585.2021.9378632) [for lecture video segmentation from extracted speech](https://doi.org/10.1109/SAMI50585.2021.9378632) [content.](https://doi.org/10.1109/SAMI50585.2021.9378632) In *2021 IEEE 19th World Symposium on Applied Machine Intelligence and Informatics (SAMI)*, pages 000299–000304.
- <span id="page-8-1"></span>Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. 2024. A survey on evaluation of large language models. *ACM Transactions on Intelligent Systems and Technology*, 15(3):1–45.
- <span id="page-8-10"></span>Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali Wang, Yu Qiao, Tong Lu, and Limin Wang. 2023. [Vide](https://arxiv.org/abs/2305.13292)[ollm: Modeling video sequence with large language](https://arxiv.org/abs/2305.13292) [models.](https://arxiv.org/abs/2305.13292) *Preprint*, arXiv:2305.13292.

- <span id="page-8-6"></span>Abhijit Debnath, K. Sreenivasa Rao, and Partha Pratim Das. 2023. [A multi-modal lecture video indexing](https://api.semanticscholar.org/CorpusID:266533926) [and retrieval framework with multi-scale residual](https://api.semanticscholar.org/CorpusID:266533926) [attention network and multi-similarity computation.](https://api.semanticscholar.org/CorpusID:266533926) *Signal Image Video Process.*, 18:1993–2006.
- <span id="page-8-9"></span>Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2020. [An image](https://api.semanticscholar.org/CorpusID:225039882) [is worth 16x16 words: Transformers for image](https://api.semanticscholar.org/CorpusID:225039882) [recognition at scale.](https://api.semanticscholar.org/CorpusID:225039882) *ArXiv*, abs/2010.11929.
- <span id="page-8-13"></span>Igor S Gruzman and Anna S Kostenkova. 2014. Algorithm of scene change detection in a video sequence based on the threedimensional histogram of color images. In *2014 12th International Conference on Actual Problems of Electronics Instrument Engineering (APEIE)*, pages 1–1. IEEE.
- <span id="page-8-14"></span>Kostas Hatalis, Despina Christou, Joshua Myers, Steven Jones, Keith Lambert, Adam Amos-Binks, Zohreh Dannenhauer, and Dustin Dannenhauer. 2024. [Mem](https://doi.org/10.1609/aaaiss.v2i1.27688)[ory matters: The need to improve long-term memory](https://doi.org/10.1609/aaaiss.v2i1.27688) [in llm-agents.](https://doi.org/10.1609/aaaiss.v2i1.27688) *Proceedings of the AAAI Symposium Series*, 2(1):277–280.
- <span id="page-8-11"></span>Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. 2024. Vtimellm: Empower llm to grasp video moments. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 14271–14280.
- <span id="page-8-5"></span>Horace Ho-Shing Ip and Siu-Lok Chan. 1997. [Hypertext-assisted video indexing and content-based](https://doi.org/10.1145/267437.267478) [retrieval.](https://doi.org/10.1145/267437.267478) In *Proceedings of the Eighth ACM Conference on Hypertext*, HYPERTEXT '97, page 232–233, New York, NY, USA. Association for Computing Machinery.
- <span id="page-8-0"></span>Rahul Radhakrishnan Iyer, Sanjeel Parekh, Vikas Mohandoss, Anush Ramsurat, Bhiksha Raj, and Rita Singh. 2019. [Content-based video indexing and re](https://arxiv.org/abs/1602.08581)[trieval using corr-lda.](https://arxiv.org/abs/1602.08581) *Preprint*, arXiv:1602.08581.
- <span id="page-8-4"></span>Hyun Ji Jeong, Tak-Eun Kim, and Myoung-Ho Kim. 2012. [An accurate lecture video segmentation](https://api.semanticscholar.org/CorpusID:14262991) [method by using sift and adaptive threshold.](https://api.semanticscholar.org/CorpusID:14262991) In *Advances in Mobile Multimedia*.
- <span id="page-8-2"></span>Dong Won Lee, Chaitanya Ahuja, Paul Pu Liang, Sanika Natu, and Louis-Philippe Morency. 2023. Lecture presentations multimodal dataset: Towards understanding multimodality in educational videos. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 20087–20098.
- <span id="page-8-12"></span>Zhaowei Li, Qi Xu, Dong Zhang, Hang Song, YiQing Cai, Qi Qi, Ran Zhou, Junting Pan, Zefeng Li, Vu Tu, Zhida Huang, and Tao Wang. 2024. [GroundingGPT:](https://doi.org/10.18653/v1/2024.acl-long.360) [Language enhanced multi-modal grounding model.](https://doi.org/10.18653/v1/2024.acl-long.360) In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 6657–6678, Bangkok, Thailand. Association for Computational Linguistics.

- <span id="page-9-20"></span>Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. 2024. [Encouraging divergent thinking](https://doi.org/10.18653/v1/2024.emnlp-main.992) [in large language models through multi-agent debate.](https://doi.org/10.18653/v1/2024.emnlp-main.992) In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 17889–17904, Miami, Florida, USA. Association for Computational Linguistics.
- <span id="page-9-3"></span>Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2023. [Video-llava: Learn](https://arxiv.org/abs/2311.10122)[ing united visual representation by alignment before](https://arxiv.org/abs/2311.10122) [projection.](https://arxiv.org/abs/2311.10122) *Preprint*, arXiv:2311.10122.
- <span id="page-9-12"></span>Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2024. [Video-LLaVA: Learn](https://doi.org/10.18653/v1/2024.emnlp-main.342)[ing united visual representation by alignment before](https://doi.org/10.18653/v1/2024.emnlp-main.342) [projection.](https://doi.org/10.18653/v1/2024.emnlp-main.342) In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 5971–5984, Miami, Florida, USA. Association for Computational Linguistics.
- <span id="page-9-7"></span>Ming Lin, J.F. Nunamaker, M. Chau, and Hsinchun Chen. 2004. [Segmentation of lecture videos based on](https://doi.org/10.1109/HICSS.2004.1265045) [text: a method combining multiple linguistic features.](https://doi.org/10.1109/HICSS.2004.1265045) In *37th Annual Hawaii International Conference on System Sciences, 2004. Proceedings of the*, pages 9 pp.–.
- <span id="page-9-8"></span>Di Ma, Xi Zhang, Xu Ouyang, and Gady Agam. 2017. [Lecture vdeo indexing using boosted margin max](https://doi.org/10.1109/ICMLA.2017.0-155)[imizing neural networks.](https://doi.org/10.1109/ICMLA.2017.0-155) In *2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA)*, pages 221–227.
- <span id="page-9-14"></span>Fan Ma, Xiaojie Jin, Heng Wang, Yuchen Xian, Jiashi Feng, and Yi Yang. 2024. Vista-llama: Reducing hallucination in video language models via equal distance to visual tokens. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 13151–13160.
- <span id="page-9-18"></span>Rao Ma, Mengjie Qian, Mark Gales, and Kate Knill. 2025. [Asr error correction using large language mod](https://arxiv.org/abs/2409.09554)[els.](https://arxiv.org/abs/2409.09554) *Preprint*, arXiv:2409.09554.
- <span id="page-9-17"></span>Rao Ma, Mengjie Qian, Potsawee Manakul, Mark Gales, and Kate Knill. 2023. [Can generative large lan](https://arxiv.org/abs/2307.04172)[guage models perform asr error correction?](https://arxiv.org/abs/2307.04172) *Preprint*, arXiv:2307.04172.
- <span id="page-9-11"></span>Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. 2024. [Video-ChatGPT: Towards](https://doi.org/10.18653/v1/2024.acl-long.679) [detailed video understanding via large vision and](https://doi.org/10.18653/v1/2024.acl-long.679) [language models.](https://doi.org/10.18653/v1/2024.acl-long.679) In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 12585– 12602, Bangkok, Thailand. Association for Computational Linguistics.
- <span id="page-9-19"></span>Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder,

- Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. [Self-refine: It](https://arxiv.org/abs/2303.17651)[erative refinement with self-feedback.](https://arxiv.org/abs/2303.17651) *Preprint*, arXiv:2303.17651.
- <span id="page-9-9"></span>Lakshmi Medida and RAMANI KASARAPU. 2021. [An optimized e-lecture video search and indexing](https://doi.org/10.22937/IJCSNS.2021.21.8.12) [framework.](https://doi.org/10.22937/IJCSNS.2021.21.8.12) 21:87–96.
- <span id="page-9-1"></span>Gouri Shankar Mishra, Anand Raj, Amit Kumar, Aman Kumar Kasaudhan, Pradeep Kumar Mishra, and Tarun Maini. 2023. [Indexing and segmentation](https://doi.org/10.1109/INCET57972.2023.10170589) [of video contents: A review.](https://doi.org/10.1109/INCET57972.2023.10170589) In *2023 4th International Conference for Emerging Technology (INCET)*, pages 1–9.
- <span id="page-9-5"></span>Junting Pan, Ziyi Lin, Yuying Ge, Xiatian Zhu, Renrui Zhang, Yi Wang, Yu Qiao, and Hongsheng Li. 2023. Retrieving-to-answer: Zero-shot video question answering with frozen large language models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 272–283.
- <span id="page-9-10"></span>Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. [Learn](https://arxiv.org/abs/2103.00020)[ing transferable visual models from natural language](https://arxiv.org/abs/2103.00020) [supervision.](https://arxiv.org/abs/2103.00020) *Preprint*, arXiv:2103.00020.
- <span id="page-9-16"></span>Bindu Reddy and Anita Jadhav. 2015. Comparison of scene change detection algorithms for videos. In *2015 Fifth International Conference on Advanced Computing & Communication Technologies*, pages 84–89. IEEE.
- <span id="page-9-15"></span>Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. 2024. Timechat: A time-sensitive multimodal large language model for long video understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 14313–14323.
- <span id="page-9-2"></span>ElMehdi Saoudi and Said Jai Andaloussi. 2021. [A dis](https://doi.org/10.21203/rs.3.rs-255106/v1)[tributed content-based video retrieval system for large](https://doi.org/10.21203/rs.3.rs-255106/v1) [data-sets.](https://doi.org/10.21203/rs.3.rs-255106/v1)
- <span id="page-9-6"></span>Rajiv Ratn Shah, Yi Yu, Anwar Dilawar Shaikh, and Roger Zimmermann. 2015. [Trace: Linguistic-based](https://api.semanticscholar.org/CorpusID:38213200) [approach for automatic lecture video segmentation](https://api.semanticscholar.org/CorpusID:38213200) [leveraging wikipedia texts.](https://api.semanticscholar.org/CorpusID:38213200) *2015 IEEE International Symposium on Multimedia (ISM)*, pages 217–220.
- <span id="page-9-13"></span>Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, Yan Lu, Jenq-Neng Hwang, and Gaoang Wang. 2024. [Moviechat:](https://arxiv.org/abs/2307.16449) [From dense token to sparse memory for long video](https://arxiv.org/abs/2307.16449) [understanding.](https://arxiv.org/abs/2307.16449) *Preprint*, arXiv:2307.16449.
- <span id="page-9-0"></span>Shraddha Soni and Shubham Dubey. 2019. [Towards](https://doi.org/10.32628/IJSRCSEIT) [systematic literature review of e-learning.](https://doi.org/10.32628/IJSRCSEIT)
- <span id="page-9-4"></span>Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, and et al.

- 2024. [Gemini: A family of highly capable multi](https://arxiv.org/abs/2312.11805)[modal models.](https://arxiv.org/abs/2312.11805) *Preprint*, arXiv:2312.11805.
- <span id="page-10-1"></span>Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. 2024a. [A survey on large language](https://doi.org/10.1007/s11704-024-40231-1) [model based autonomous agents.](https://doi.org/10.1007/s11704-024-40231-1) *Frontiers of Computer Science*, 18(6).
- <span id="page-10-10"></span>Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. 2004. [Image quality assessment: from error](https://doi.org/10.1109/TIP.2003.819861) [visibility to structural similarity.](https://doi.org/10.1109/TIP.2003.819861) *IEEE Transactions on Image Processing*, 13(4):600–612.
- <span id="page-10-8"></span>Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. 2024b. [Videotree: Adaptive tree-based](https://arxiv.org/abs/2405.19209) [video representation for llm reasoning on long videos.](https://arxiv.org/abs/2405.19209) *Preprint*, arXiv:2405.19209.
- <span id="page-10-9"></span>Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. 2023. [Autogen: Enabling next-gen llm ap](https://arxiv.org/abs/2308.08155)[plications via multi-agent conversation.](https://arxiv.org/abs/2308.08155) *Preprint*, arXiv:2308.08155.
- <span id="page-10-3"></span>Natsuo Yamamoto, Jun Ogata, and Yasuo Ariki. 2003. [Topic segmentation and retrieval system for lecture](https://doi.org/10.21437/EUROSPEECH.2003-333) [videos based on spontaneous speech recognition.](https://doi.org/10.21437/EUROSPEECH.2003-333) In *8th European Conference on Speech Communication and Technology, EUROSPEECH 2003 - INTER-SPEECH 2003, Geneva, Switzerland, September 1-4, 2003*, pages 961–964. ISCA.
- <span id="page-10-2"></span>Haojin Yang and Christoph Meinel. 2014. [Content](https://doi.org/10.1109/TLT.2014.2307305) [based lecture video retrieval using speech and video](https://doi.org/10.1109/TLT.2014.2307305) [text information.](https://doi.org/10.1109/TLT.2014.2307305) *IEEE Transactions on Learning Technologies*, 7(2):142–154.
- <span id="page-10-5"></span>Haojin Yang, Harald Sack, and Christoph Meinel. 2011a. [Lecture video indexing and analysis using video ocr](https://api.semanticscholar.org/CorpusID:14263115) [technology.](https://api.semanticscholar.org/CorpusID:14263115) *2011 Seventh International Conference on Signal Image Technology & Internet-Based Systems*, pages 54–61.
- <span id="page-10-4"></span>Haojin Yang, Maria Siebert, Patrick Lühne, Harald Sack, and Christoph Meinel. 2011b. [Lecture video index](https://doi.org/10.1109/SITIS.2011.20)[ing and analysis using video ocr technology.](https://doi.org/10.1109/SITIS.2011.20)
- <span id="page-10-7"></span>Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. 2023. [Self-chained image-language](https://arxiv.org/abs/2305.06988) [model for video localization and question answer](https://arxiv.org/abs/2305.06988)[ing.](https://arxiv.org/abs/2305.06988) *Preprint*, arXiv:2305.06988.
- <span id="page-10-6"></span>Hang Zhang, Xin Li, and Lidong Bing. 2023. [Video-](https://doi.org/10.18653/v1/2023.emnlp-demo.49)[LLaMA: An instruction-tuned audio-visual language](https://doi.org/10.18653/v1/2023.emnlp-demo.49) [model for video understanding.](https://doi.org/10.18653/v1/2023.emnlp-demo.49) In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 543–553, Singapore. Association for Computational Linguistics.

<span id="page-10-0"></span>Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. *arXiv preprint arXiv:2303.18223*.

## A Appendix

The Appendix consists of supplemental materials in our research journey for the video understanding topic.

### <span id="page-11-0"></span>A.1 Video Segmentation Details

Figure [2](#page-11-3) shows the details of video (scene) segmentation algorithm, which leverages off-the-shelf tool PySceneDetect. Meanwhile, a more detailed description of pseudo code is shown in Figure [3](#page-12-1) for reference.

In the proposed segmentation algorithm, we merge short segments (e.g., less than 3s) with the proceeding ones, assuming they are transitions between slides. For the remaining segments with reasonable duration (e.g., one minute or less), we re-check whether the current segment actually presents the same slide as the previous segment, first using efficient SSIM (structural similarity index) [\(Wang et al.,](#page-10-10) [2004\)](#page-10-10) to identify obvious cases and then using VLM to verify the remaining tricky ones. If the answer is yes, the two segments are merged. For the relatively long segments (i.e., duration > one minute), we suspect the segment may actually contain multiple similar slides, and thus re-detect slides (StepA) using VLM and determine the presentation time span for each detected slide (StepB) based on vision/audio hints in this segment. In StepA, we sample a video frame every N\_sample seconds in the focused segment, and use VLM to compare each frame with the previous one to check whether they contain a same slide. Whenever the answer is negative, a new slide is identified. In StepB, we leverage Automatic Speech Recognition (ASR) results (i.e., sentences with time stamps) of the focused segment and sample a video frame from the middle point of the time span for each sentence. By comparing the extracted frame per sentence with its neighborhood detected slides with image similarity, we can determine which slide the sentence explains, and thus in this way, estimate the presentation time span of each detected slide in the focused segment. The detailed approach of Step<sup>B</sup> is described in Figure [3.](#page-12-1) Throughout the segmentation algorithm, we use a same VLM and the same prompt (shown in Table [11\)](#page-14-0) to determine whether two frames contain a same slide.

Figure 2: Video segmentation algorithm. SSIM is the Structure Similarity Index, a measure in image assessment, which gives values between 0 and 1. SSIM gives high similarity, when two images have visually similar look but have rather different pixel value (e.g. stretch, mean-shift). While the approach is prone to give a low measure value, if visual appearance of the two images is much different.

<span id="page-11-4"></span>

| Video # | Video Link                                  |
|---------|---------------------------------------------|
| V1      | https://www.youtube.com/watch?v=2NgUY8f1pa8 |
| V2      | https://www.youtube.com/watch?v=2_dZ5GBlRgU |
| V3      | https://www.youtube.com/watch?v=_Awekr6-ilg |
| V4      | https://www.youtube.com/watch?v=BsXUWddl-as |
| V5      | https://www.youtube.com/watch?v=N75gvrZfO24 |
| V6      | https://www.youtube.com/watch?v=_Jw3DQ7_pxg |

Table 10: LPM videos used for video segemntation evaluation.

### <span id="page-11-1"></span>A.2 LPM videos for Video Segmentation

Links for the LPM videos used for video segmentation evaluation can be found in Table [10.](#page-11-4) Note only the first 10 minutes are used.

# <span id="page-11-2"></span>A.3 Benchmark datasets Creation from Human Annotation.

We take the following measures to ensure the quality of MTurk annotation.

- Workers qualification: We recruit MTurk workers who have "Master qualification", a Life-Time-Approval-Rate of at least 90%, and at least 10, 000 tasks approved. In addition, workers are preferred if English is their first language, as some lecture videos might be difficult to follow.
- Test Exercises and Dummy Questions:

Figure 3: Detailed algorithm for Step<sup>B</sup> of the proposed Video Segmentation approach. In the given segment list, the list of extraction times refer to the times of the sampled frames that are deemed to contain the same slide in StepA.

Workers are asked to complete two exercises, which are similar to a typical questionnaire, before working on a questionnaire set. Figure [4](#page-16-1) is the starting page for the MTurkers, and as shown in Figure [5](#page-17-0) and [6](#page-18-0) show the two exercises for MTurkers, so that they can understand the assigned task better. If the workers answer the exercises correctly, they will proceed to a page with explanations for the exercises, as shown in Figure [7.](#page-19-0) After checking a box stating that they fully understand the task, workers will proceed to the questionnaire set. Among the six questionnaires in the set, one question is exactly the same with one exercise. This dummy question is used for quality control. We assume workers who wrongly answered the dummy question either rushed through the task or don't understand the task. We thus reject the corresponding questionnaires for quality purpose.

• Compensation: In order to attract qualified workers to work on our questionnaires, the compensation for the annotators is set as an hourly rate of \$9, which is higher than the US federal minimum wage of \$7.25,

Please note, some of the figures in this section are put back in the document due to its screenshot size.

## A.4 System Settings

## <span id="page-12-0"></span>A.4.1 Experimental settings for the video segmentation component

We evaluate our proposed video segmentation approach and compare it with the SOTA PySceneDetect baseline on the datasets listed in Table [1.](#page-4-0) Note that our approach also uses PySceneDetect to conduct first-round video segmentation. In our approach, we especially tune the PySceneDetect to minimize the chance of missing slides in segmentation. For fair comparison, for the baseline PySceneDetect, we tune it again to achieve its best overall performance. We tune all the algorithm parameters on a separate held-out video set. In the proposed video-segmentation approach, T hreshold1, T hreshold2, T hreshold<sup>3</sup> and T hreshold<sup>4</sup> are set as 3 seconds, 60 seconds, 0.9, 0.65, respectively, and N\_sample is set as 60 seconds. For PySceneDetect, AdaptiveDetector is used with adaptive\_threshold set as 1 and min\_content\_val set as 10. For the baseline PySceneDetect, ContentDetector is adopted with threshold set as 12. In this work, we adopt GPT 4 Vision as the VLM used in our proposed segmentation algorithm.

We set temperature as 0 and max\_tokens as 800 for all GPT 4 models used in this work.

## <span id="page-13-1"></span>A.4.2 Experimental settings for the video-segment understanding component

In this work, GPT-4 Turbo is used for all agents that require a VLM or LLM in processing in the Video-Segment Understanding component.

In ASR recognition, the Whisper model is used to transcribe speech into text for each video. To further reduce hallucination, among the corrections suggested by the VLM, we only modify transcript<sup>i</sup> accordingly if the suggestion is "term<sup>A</sup> should be termB" and the acoustic difference level between term<sup>A</sup> and term<sup>B</sup> (evaluated using PyPhonetics[4](#page-13-3) ) is less than 5, which means the two terms likely have similar pronunciations. For ASR correction, the acoustic difference level between two terms is evaluated using the Refined-Soundex.distance function of PyPhonetics.

For the dynamic critic, Nmax is set to 10.

## <span id="page-13-2"></span>A.4.3 Experimental Settings of the QA system used in the extrinsic evaluation

We setup a Retrieval Augmented Generation (RAG) based QA system for extrinsic evaluation. The RAG based QA system is composed of a retriever based on FAISS[5](#page-13-4) and a reader using LangChain[6](#page-13-5) . In the index-building process, the three multimodal agents generates the understanding result respectively (including vision understanding result, speech transcript, and consolidation information). For each segment, the retriever builds up an embedding vector for that segmented scene(slide), which is added to the FAISS index. The embedding model is SentenceTransformer/all-MiniLM-L6-v2. In the retrieval and QA phase, the reader wraps up the top 5 retrieval results as context, and sends the question

together with the context to GPT 3.5 for answer generation. For evaluation, we ask GPT-4 Turbo to generate 2 questions answerable with vision information but not answerable with speech (denoted as vision questions), 2 questions answerable with speech but not answerable with vision information (denoted as speech questions), and 2 questions that can be best answered with the consolidated information (denoted as consolidation questions). We also require the model to generate the ground-truth answer at the same time for each generated question to facilitate evaluation (See Table [12](#page-15-0) for the prompt). We then run the retrieval-based QA system to generate an answer for each question. The correctness of the answer generated by QA system is also evaluated by GPT-4 Turbo using the prompt shown in Table [13.](#page-16-0)

## <span id="page-13-0"></span>A.5 Prompts used in the Experiments

We list the prompts used in the experiments in the later part of Appendix, due to its table size. Table [11](#page-14-0) lists the prompts used in all the agents of the proposed framework. Table [12](#page-15-0) shows the prompts used to generate the questions together with the corresponding ground-truth answers for the extrinsic evaluation with QA, the ablation study on ASR correction, and the ablation study on vision understanding, respectively. Table [13](#page-16-0) presents (1) the prompt that is used to determine whether two vision understanding results are inconsistent (i.e., containing conflicting information), and (2) the prompt that is used to determine whether an answer generated by the QA system is correct based on the question and the corresponding ground-truth. All prompts are carefully designed.

<span id="page-13-3"></span><sup>4</sup> <https://pypi.org/project/pyphonetics/>

<span id="page-13-4"></span><sup>5</sup> <https://github.com/facebookresearch/faiss>

<span id="page-13-5"></span><sup>6</sup> <https://www.langchain.com/>

<span id="page-14-0"></span>

| Agent/Algorithm                                                               | Prompt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Video<br>Segmentation                                                         | {image 1} {image 2} For the two images provided, if both images appear to be digitally corrupted or distorted, answer with "Yes. Both images are corrupted." and terminate the response. Otherwise, do the following: For the two images provided, each should show a person of several people presenting a slide. Is the slide shown in the first image the same as the one shown in the second image? Please check very carefully for different texts within the slides. Please start your answer with "Yes." if the slides are the same and "No." if the slides are different. Then give an explanation for your answer.                                                                                                                                                                                                                              |
| Vision<br>Understanding<br>Agent (Baseline<br>System)                         | {image} Given the image provided, please follow the following rules to generate a description: (1) If this image contains a slide that occupies at least half of the image, please describe the content of that slide in detail. In thi case, when generating the description, please only focus on the slide's content, and ignore the slide's bottom part such as slid number, footnotes, company logo, etc. as well as other parts of the image. If there are one or more humans presented in the image please also ignore the humans and don't include them in the description in this case.  (2) Otherwise, if there is no significant slide in the image, please simply describe the image.                                                                                                                                                        |
| Vision<br>Understanding<br>Agent<br>(Knowledge-<br>Related<br>Enhancement)    | [image] Given the image provided, we have the following background knowledge that are likely to be relevant: {retrieved knowledge} Based on the given image and the background knowledge, please follow the following rules to generate a description: (1) If this image contains a slide that occupies at least half of the image, please describe the content of that slide in detail. In thi case, when generating the description, please only focus on the slide's content, and ignore the slide's bottom part such as slid number, footnotes, company logo, etc. as well as other parts of the image. If there are one or more humans presented in the image please also ignore the humans and don't include them in the description in this case. (2) Otherwise, if there is no significant slide in the image, please simple describe the image. |
| Vision-Audio<br>Consolidation<br>Agent                                        | Given a video of someone presenting a slide, the text description of the slide (Part_1), and the speech narrative of the presentatio (Part_2) are provided below. Please consolidate the two parts into a nice overall description of the video content.  Part_1. The text description of slide: {vision understanding result}  Part_2. The speech narrative of the presentation: {speech transcript}                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Vision-based<br>Keyword<br>Extraction and<br>ASR Error<br>Correction<br>Agent | {image} Transcribed speech explanation for the image above: {speech transcript}  Given the provided image and its speech explanation transcript listed above, if the provided image contains a slide, extract th keywords (i.e., important words or phrases) from the slide. Then, check the transcribed speech explanation to see whether an keyword is misrecognized as other word or word sequence with similar pronunciation. Please generate the response following th format below:  List of keywords: - keyword1 - keyword2 - keyword3                                                                                                                                                                                                                                                                                                            |
|                                                                               | (If a detected keyword contains ',' or ',' in middle, it should be split into multiple keywords. If no keyword is detected or the slide/image is empty, just leave the list of keywords empty.)  Answer for whether certain keyword(s) is misrecognized: Yes or No (if the answer is Yes, provide the following explanation:) The term *** should be ****. The term *** should be ****. The term *** might be ****. The term *** might be ****.                                                                                                                                                                                                                                                                                                                                                                                                          |
| Knowledge<br>Extraction<br>Agent                                              | Given the text description listed below, summarize the concepts presented in this text description. If the text description is not about a slide presentation, reply with 'No concept extracted'. When generating the output, please follow the following format:  Concept: Concept name  Knowledge of Concept: explanation  Concept: Concept name  Knowledge of Concept: explanation  Concept: Concept name  Knowledge of Concept: explanation  Text description:{consolidated understanding result}                                                                                                                                                                                                                                                                                                                                                    |
| Dynamic Critic                                                                | Critic Agent: Given an image that contains a slide presentation and a description about the slide presentation, decide whethe the description can be further improved. If the description is not comprehensive or containing potential mistakes, ask Visiou Understanding Agent to improve the description. Otherwise, if the description is comprehensive and accurate, DO NOT repeat the description and just reply "TERMINATE!!!" to Admin.  ***********************************                                                                                                                                                                                                                                                                                                                                                                      |

Table 11: Prompts for all agents used in this paper.

<span id="page-15-0"></span>Function Prompt We have a video segment in which a speaker is presenting a slide. The information about the video segment is summarized into three Ouestion we have a video segment in winter a speaker is presenting a state. The internation about the vision information about the vision information presented in the slide. The second part, denoted as "Part\_1. The text description of slide," provides a text description of the vision information presented in the slide. The second part, denoted as "Part\_2. The speech narrative of the presentation:", provides the Generation (extrinsic evaluation with speech-to-text transcript about what the speaker said about the slide. The third part, denoted as "Part\_3. Info-Consolidation Output:", provides the overall description of the video segment, which consolidates the information from the first part and second part. Your task is to generate six questions satisfying following requirements respectively, and also generate the corresponding answer for each generated question.
(1) Generate two questions ons (referred to as Question\_1\_vision and Question\_2\_vision) that are answerable with the slide vision information (i.e., information provided in the first part), but not answerable with the speech transcript (i.e., information provided in the second part). (2) Generate two questions (referred to as Question\_3\_speech and Question\_4\_speech) that are answerable with the speech transcript (i.e., information provided in the second part), but not answerable with the slide vision information (i.e., information provided in the first part). (3) Generate two questions (referred to as Question\_5\_consolidated and Question\_6\_consolidated) that can be best answered with the consolidated information, i.e., information provided in the third part. In addition, when generting a question or a answer, please directly talk about the knowledge point and avoid the mentioning of slide, speech/speaker, and video as information sources (e.g., avoid the use of "according to the speech", "according to the presentation", "according to according to the consolidated information", "as discussed in the video", "according to the speaker", etc.). For example, instead of asking a question "How does Business Intelligence aid in the preparation of decisions according to the presentation?", please directly ask "How does Business Intelligence aid in the preparation of decisions?". Another example is that instead of answering "The slide indicates that Business Intelligence helps interpret past data to inform future decisions.", please directly answer "Business Intelligence helps interpret past data to inform future decisions.".

Please also note that if "slide" must be mentioned in a question, please always include slide title to show which slide it is about. Please provide the generated questions and the answers in the following format: Question 1 vision Answer\_1\_vision: Question\_2\_vision Answer 2 vision: Question\_3\_speech: Answer 3 speech: Question\_4\_speech: Answer 4 speech: Question 5 consolidated Answer\_5\_consolidated: Question 6 consolidated Answer\_6\_consolidated: Part 1. The text description of slide: {vision understanding result} Part 2. The speech narrative of the presentation: {speech transcript} Part 3. Info-Consolidation Output: {consolidated information} Question We have a video segment in which a speaker is presenting a slide. Two visual descriptions about the video segment and their differences are provided Generation Your task is to generate two questions satisfying following requirements, and also generate the corresponding answer for each generated on vision question. understanding Generate two questions (referred to as Question\_1\_vision and Question\_2\_vision) that are answerable with Description 2, but not answerable with Description 1 ddition, when generating a question or a answer, please directly talk about the knowledge point and avoid the mentioning of slide, speech/speaker, and video as information sources (e.g., avoid the use of "according to the speech", "according to the presentation", "according to according to the consolidated information", "as discussed in the video", "according to the speaker", etc.). For example, instead of asking a question "How does Business Intelligence aid in the preparation of decisions according to the presentation?", please directly ask "How does Business Intelligence aid in the preparation of decisions?". Another example is that instead of answering "The slide indicates that Business Intelligence helps interpret past data to inform future decisions.", please directly answer "Business Intelligence helps interpret past data to inform future decisions.".

Please also note that if "slide" must be mentioned in a question, please always include slide title to show which slide it is about Please provide the generated questions and the answers in the following example format: Question 1 vision Answer\_1\_vision: Question 2 vision Answer 2 vision Description 1: {vision understanding result 1} Description 2: {vision understanding result 2} Difference detection result: {difference} Question We have a video segment in which a speaker is presenting a slide. The transcript about the video segment is provided and the corrections, if Generation any, needed for the transcript. (ablation study Your task is to generate two questions satisfying following requirements, and also generate the corresponding answer for each generated on ASR correction) Generate two questions (referred to as Question\_3\_speech and Question\_4\_speech) that are answerable with the speech transcript correction, but not answerable without the speech transcript correction In addition, when generating a question or a answer, please directly talk about the knowledge point and avoid the mentioning of slide, specch/speaker, and video as information sources (e.g., avoid the use of "according to the speech," "according to the presentation",
"according to according to the consolidated information", "as discussed in the video", "according to the speaker", etc.). For example,
instead of asking a question "How does Business Intelligence aid in the preparation of decisions according to the presentation?", please directly ask "How does Business Intelligence aid in the preparation of decisions?". Another example is that instead of answering "The slide indicates that Business Intelligence helps interpret past data to inform future decisions.", please directly answer "Business Intelligence helps interpret past data to inform future decisions. Please also note that if "slide" must be mentioned in a question, please always include slide title to show which slide it is about. ease provide the generated questions and the answers in the following example format Answer 3 speech: Question\_4\_speech: Answer\_4\_speech: Transcript: {original speech transcript Transcript correction needed:: {corrections}

Table 12: Prompts for question generation.

<span id="page-16-0"></span>

| Function                                 | Prompt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Inconsistency<br>Detection               | Given two descriptions about a same slide (listed below), please determine whether there is any conflict in meaning between the two descriptions.  Please first answer 'Yes' or 'No', and if the answer is 'Yes', explain what the meaning conflict(s) is?  Description 1: {vision understanding 1}  Description 2: {vision understanding 2}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| QA Evaluation<br>(Answer<br>Correctness) | Given a question and its ground-truth answer, check whether a automatically generated answer is correct. The question, ground-truth answer, and the automatically generated answer are all listed below. In your response, please simply say "correct" if you think the generated answer contains consistent information as the ground-truth answer, simply say "wrong" if you think the generated answer is wrong (i.e., conflicting with the information in the ground-truth answer, or failing to include the key messages in the ground-truth answer), and simply say "correct but with additional information" if you think the generated answer contains the correct answer but includes additional information not mentioned in the ground-truth answer.  Question: {question}  Ground-truth answer: {ground truth answer}  Automatically generated answer: {predicted answer} |

Table 13: Prompts for inconsistency detection given two vision understanding results and for answer correctness evaluation.

<span id="page-16-1"></span>![](_page_16_Figure_2.jpeg)

Figure 4: MTurk questionnaire instructions page.

<span id="page-17-0"></span>![](_page_17_Picture_0.jpeg)

Figure 5: MTurk questionnaire exercise question 1.

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

Figure 6: MTurk questionnaire exercise question 2. Note that the questionnaires that need annotation share the same format as this exercise question.

<span id="page-19-0"></span>![](_page_19_Figure_0.jpeg)

Figure 7: Answer explanation for MTurk questionnaire exercise question 2.