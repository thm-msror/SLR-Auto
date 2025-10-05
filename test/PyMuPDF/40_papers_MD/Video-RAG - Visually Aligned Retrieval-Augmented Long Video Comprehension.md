## **Video-RAG : Visually-aligned Retrieval-Augmented Long Video Comprehension**

Yongdong Luo [1], Xiawu Zheng [1], Xiao Yang [1], Guilin Li [1], Haojia Lin [1],
Jinfa Huang [2], Jiayi Ji [1], Fei Chao [1], Jiebo Luo [2], Rongrong Ji [1]

1 Xiamen University 2 University of Rochester





**Abstract**


_Existing large video-language models (LVLMs) struggle_
_to comprehend long videos correctly due to limited context._
_To address this problem, fine-tuning long-context LVLMs_
_and employing GPT-based agents have emerged as promis-_
_ing solutions. However, fine-tuning LVLMs would require_
_extensive high-quality data and substantial GPU resources,_
_while GPT-based agents would rely on proprietary models_
_(e.g., GPT-4o). In this paper, we propose_ _**Video R**_ _etrieval-_
_**A**_ _ugmented_ _**G**_ _eneration (_ _**Video-RAG**_ _), a training-free and_
_cost-effective pipeline that employs visually-aligned auxil-_
_iary texts to help facilitate cross-modality alignment while_
_providing additional information beyond the visual con-_
_tent. Specifically, we leverage open-source external tools_
_to extract visually-aligned information from pure video data_
_(e.g., audio, optical character, and object detection), and in-_
_corporate the extracted information into an existing LVLM_
_as auxiliary texts, alongside video frames and queries, in_
_a plug-and-play manner._ _Our_ _**Video-RAG**_ _offers several_
_key advantages: (i) lightweight with low computing over-_
_head due to single-turn retrieval; (ii) easy implementa-_
_tion and compatibility with any LVLM; and (iii) signifi-_
_cant, consistent performance gains across long video un-_
_derstanding benchmarks, including Video-MME, MLVU,_
_and LongVideoBench. Notably, our model demonstrates su-_
_perior performance over proprietary models like Gemini-_
_1.5-Pro and GPT-4o when utilized with a 72B model._ [1]


**1. Introduction**


With the advancements in Large Language Models (LLMs),
numerous studies have been conducted to enhance their

ability to comprehend and process videos [2, 3, 10, 14–
17, 21–23, 45, 48, 50], collectively termed Large VideoLanguage Models (LVLMs). Although current LVLMs
have demonstrated promising performance in understanding short videos, effective comprehension of extremely long
videos continues to be a major challenge.


1 [Our code is available at https://github.com/Leon1207/](https://github.com/Leon1207/Video-RAG-master)

[Video-RAG-master.](https://github.com/Leon1207/Video-RAG-master)



Figure 1. Illustration of two common approaches for understanding long videos, alongside our Video-RAG. Video-RAG provides
a resource-efficient, training-free pipeline that is easily compatible
with any LVLM. By leveraging RAG, it retrieves auxiliary texts for
input, leading to notable performance enhancement.


To address this challenge, recent studies [33, 39, 43,
47, 53] have sought to extend the reasoning context length
of LVLMs, essentially finetuning long-context LVLMs for
long video understanding. LongVA [47] first introduces
increasing the token capacity of an LLM and transferring
its long-context comprehension capabilities to video data.
However, training such a model requires pre-training on an
extended corpus, and often there are distribution shifts between deployment videos and finetuning videos. As demonstrated in Video-MME [6], LongVA declines when increasing the video frame sampling rate from 128 to 384 (52.6%
→ 51.8%). This outcome suggests that simply increasing
the number of sampled frames not only leads to information
redundancy but also imposes additional challenges for the
model to handle complex reasoning. Retrieval-Augmented
Generation [12] (RAG) is a technique that enhances generative tasks by retrieving relevant documents from an external
corpus, thus improving response quality in LLMs. Recent
studies have begun exploring the integration of RAG with



**(a) Long-context LVLMs**



**Training-free:** **Resource:** **Flexbility:**









**(b) GPT-based Agent**



**Training-free:** **Resource:** **Flexbility:**













**(c) Our** _**Video-RAG**_



**Training-free:** **Resource:** **Flexbility:**















1


Figure 2. Comparison of the performance of Video-RAG with
LLaVA-Video-72B [49], Gemini-1.5-Pro [31], and GPT-4o [28]
across various benchmarks, including the sub-tasks from VideoMME [6] (here we focus only on those that outperform Gemini1.5-Pro), LongVideoBench [41], and MLVU [52] benchmarks.


video-based tasks [1, 25, 46], employing tools to process
videos in long contexts and sending them to a proprietary
model for generation, which is known as the GPT-based
Agent method. However, they come with serval limitations. First, most of them process long video content as
plain text, subsequently utilizing the RAG mechanisms to
retrieve relevant documents for LLMs. Therefore, they lack
alignment with the visual context of the video, resulting in
a loss of critical visual information. Second, they are often
resource-intensive in multi-turn interactions and typically
require powerful LLMs to function as the driving force, thus
limiting their flexibility and generative capabilities. Executing the whole Video-MME [6] using VideoAgent [4] requires approximately 20 days and incurs a substantial consumption of GPT-4o API tokens.

In this study, we propose Video-RAG, an effective RAG
pipeline that can be seamlessly integrated with any LVLM.
Specifically, instead of simply increasing the number of
sampled video frames, we propose to replace the corresponding extended visual tokens with auxiliary texts extracted from pure video data by invoking open-source foundation models, such as optical character recognition (OCR),
automatic speech recognition (ASR), and object detection.
These auxiliary texts are more aligned with the visual context while providing additional information beyond the visual data, as demonstrated in [4, 18]. Besides dealing with
the context windows limit of LVLMs, we employ RAG in
Video-RAG to filtering auxiliary texts, ensuring their rel


evance to the user’s query in the text embedding space.
As sampled visual context often lacks explicit alignment
with the instructions, the auxiliary texts can facilitate crossmodality alignment while reducing the modality divide. As
illustrated in Figure 6, with Video-RAG, the retrieved auxiliary texts help guide the LVLM to pay more attention to the
query-relevant keyframes, while simultaneously facilitating
cross-modality alignment between query and keyframes. In
this framework, an LVLM serves as the central component of Video-RAG, processing visual tokens to preserve
detailed visual context and minimize potential information
loss. Moreover, the retrieval process is parallelly executed
in a single operation, ensuring the efficiency of our pipeline.
We evaluate Video-RAG across several long video
benchmarks, including Video-MME [6], MLVU [52], and
LongVideoBench [41]. By applying the Video-RAG to six
distinctive open-source LVLMs, we achieve an average performance improvement of 8.0% on Video-MME with only
2.0K text tokens addition (equal to 14 frames in most configuration) per case, while beating the proprietary LVLM
Gemini-1.5-Pro [31] when integrated with the 72B model,
as shown in Figure 7. Applying Video-RAG to a 7B LVLM
only requires an additional 8GB of inference GPU memory
and approximately 5 seconds of inference time per case.
In summary, our contributions are as follows:

- **We integrate RAG into open-source LVLMs:** [2] VideoRAG incorporates three types of visually-aligned auxiliary texts (OCR, ASR, and object detection) processed
by external tools and retrieved via RAG, enhancing the
LVLM. It’s implemented using completely open-source
tools, without the need for any commercial APIs.

- **We design a versatile plug-and-play RAG-based**
**pipeline for any LVLM:** Video-RAG offers a trainingfree solution for a wide range of LVLMs, delivering
performance improvements with minimal additional resource requirements.

- **We achieve proprietary-level performance with open-**
**source models:** Applying Video-RAG to a 72B opensource model yields state-of-the-art performance in
Video-MME, surpassing models such as Gemini-1.5-Pro.


**2. Related Work**


**2.1. Large Video-Language Models**


With the rapid advancement of large language models
(LLMs), there has been increasing interest in developing
generalist video models capable of handling a wide range of
video-related tasks. Video-ChatGPT [26] extracts features
from individual frames and aggregates them through both


2 While some methods [1, 25, 46] use RAG system for video tasks, they
convert the video data fully into text while also relying on proprietary, nonopen-source models. These approaches may result in the loss of visual
information and lead to significant resource and time consumption.



2


Figure 3. The framework of our Video-RAG pipeline that contains three key phases. In the query decouple phase, the LVLM is prompted to
generate a retrieval request for auxiliary texts. Next, in the auxiliary text generation and retrieval phase, the video is processed **in parallel** to
extract three types of textual information (OCR, ASR, and object detection), and the relevant text is retrieved as the auxiliary text. Finally,
in the integration and generation phase, auxiliary texts are combined with the query and the video to generate the response.



spatial and temporal pooling operations. VideoChat [14]
encodes videos by generating both textual descriptions and
video appearance embeddings. Video-LLaVA [16] aligning
image and video encoders during a pre-processing phase,
using a shared projector to map the encoded representations into a common language space. LLaVA-NeXT-Video

[48] extends LLaVA-NeXT [20] by fine-tuning the model
specifically on video data. Despite their contributions, these
approaches face challenges when processing detailed and
long-length videos, primarily due to the limited number of
frames sampled for analysis.


**2.2. Long-context Large Video-Language Models**


Recent approaches have sought to expand the context window size to enhance detailed video understanding. LongVA

[47] and Long-LLaVA [43] address this by continuously
training LLMs on extended textual data, to transfer their
long-text comprehension capabilities to video processing.
INTP [33] introduces a video token rearrangement technique while proposing a training-free method for extending the LLM context window, allowing LVLMs to process
increased visual tokens. However, these methods face challenges in striking a balance between the high computational
costs associated with sampling video frames and the limited
performance improvements achieved. Due to the inherent
redundancy in video content and constraints on model ca


pacity, performance degradation may occur when the number of sampled frames surpasses a certain threshold.


**2.3. GPT-based Agent Video Understanding**


Initial efforts [7, 27, 36, 40, 44] have employed LLMs to interact with tools to process visual information as structured
long context for question-answering. MM-VID [18] enhances long video understanding by aligning video frames
with corresponding text descriptions. VLog [19] leverages
multimodel pre-trained models to capture and interpret visual and audio information, summarizing it into documents
for video comprehension. VideoAgent [4], DrVideo [25],
and OmAgent [46] integrate multimodal inputs and enable
dynamic querying of video segments to support long video
reasoning tasks. However, these methods take an extremely
long time to process videos while relying on proprietary
models (e.g., GPT-4o), thus limiting their efficiency and
adaptability to other open-source frameworks.


**3. Method**


We propose a novel, training-free pipeline for large videolanguage models (LVLMs), named Video-RAG, which can
be integrated into any LVLM. As illustrated in Figure 3, our
pipeline comprises three key phases: **(i) Query Decouple:**
In this phase, the user’s query is decomposed into a retrieval



3


request aimed at extracting auxiliary texts from the target
video. **(ii) Auxiliary Text Generation & Retrieval:** Multiple auxiliary texts are generated from the queried video in
parallel. Then, the retrieval request is used to obtain relevant external information. **(iii) Integration and Gener-**
**ation:** This phase integrates the retrieved auxiliary texts
with the user’s query, feeding this combined input into the
LVLMs to generate the final response.


**3.1. Large Video-Language Model**


Given a video **V**, a frame sampler first sample _N_ frames
**F** . Most existing methods uniformly sample frames from
a video for both effectiveness and simplicity. Then, video
features are extracted as **F** _**v**_ = VisualEnc( **F** ), where
VisualEnc is an image-based visual encoder, such as
CLIP-L [29]. Finally, the video features **F** _**v**_ and the user’s
query **Q** are fed into the LVLM to generate an output **O** :


**O** = LVLM( **F** _**v**_ _,_ **Q** ) (1)


**3.2. Query Decouple**


In this phase, upon receiving a user’s query about the video,
the LVLM begins by decoupling the query and generating
retrieval requests, denoted as **R**, for auxiliary texts. During
this phase, the LVLM processes only textual information,
without access to video frames, and the output requests are
formatted in JSON. We prompt the LVLM using a decoupling prompt **P** to generate the following retrieval requests
as necessary: (i) **R** _asr_ : Requests about automatic speech
recognition, to extract audio information from the video that
may pertain to the query. (ii) **R** _det_ : Requests for identifying
physical entities within the video that may assist in answering the query. (iii) **R** _type_ : Requests for details about the
location, quantity, and relationships of the identified physical entities. These requests, which may be NULL (indicating that the corresponding information is not required),
are then parsed and forwarded to the auxiliary text retrieval
phase. The entire process can be formally described as:


**R** = LVLM( **P** _,_ **Q** ) _,_ **R** = _{_ **R** _asr_ _,_ **R** _det_ _,_ **R** _type_ _}_ (2)


**3.3. Auxiliary Text Generation**


In this phase, we first generate the auxiliary texts from the
video and then retrieve them to assist the LVLMs accord
ing to the retrieval requests **R** . As the length of the video
increases, the number of tokens generated from the processed data also grows, leading to an increase in redundant information. Additionally, current open-source models are constrained by the limited length of their context
windows, which may prevent them from fully processing
all auxiliary texts. To address this issue, we draw inspiration from Retrieval-Augmented Generation (RAG) [12], retrieving only the auxiliary texts relevant to the user’s query.



Before retrieval, we construct the necessary databases from
the given video in parallel. Specifically, we implement
three distinct databases: the Optical Character Recognition
(OCR) database, denoted as _DB_ _ocr_ ; the Automatic Speech
Recognition (ASR) database, denoted as _DB_ _asr_ ; and the
Object Detection (DET) database, denoted as _DB_ _det_ .
**OCR database.** Current LVLM are still illusory in their
ability to accurately recognize characters, and their performance often falls short compared to proprietary models. To better leverage the information contained in video
frames and reduce hallucinations, we employ a proprietary
OCR model to extract text from each sampled video frame.
Specifically, we use EasyOCR [9] as our text recognition
model and segmented the recognized texts on a per-frame
basis, denoted as **T** _ocr_ . Subsequently, we implemented
RAG by utilizing the advanced text encoding model Contriever [8] to encode the fetched OCR texts into text embeddings **E** _ocr_ . These embeddings are then stored in a database
with the FAISS index [11], a library designed for efficient
similarity search and clustering of dense vectors. The entire
building process can be formally described as:


**T** _ocr_ = EasyOCR( **F** ) (3)


_DB_ _ocr_ _←−−−−−_ FAISS **E** _ocr_ = Contriever( **T** _ocr_ ) (4)


**ASR database.** Audio information (e.g., subtitles) plays a
crucial role in video comprehension, often providing additional context that may not be available through visual cues
alone. To incorporate them, we first extract the raw audio
**U** from the video and then transcribe them into texts **T** _asr_ .
Specifically, we use Whisper [30] as our audio transcription
model. Since the recognized texts can be quite extensive,
we chunk and encode them into a vector database, following the same procedure used to construct the OCR database.
The building process can be formally described as:


**T** _asr_ = Whisper( **U** ) (5)


_DB_ _asr_ _←−−−−−_ FAISS **E** _asr_ = Contriever( **T** _asr_ ) (6)


**DET database.** While LVLMs demonstrate strong performance in object recognition, they continue to face challenges such as object counting, precise object localization,
and understanding relative relationships between objects.
To mitigate the issue of hallucination, which can stem from
these challenges, we incorporate object detection information as auxiliary texts. We leverage a visual grounding
model to extract both the object categories and their corresponding positions from sampled video frames. This approach helps provide more accurate and context-aware object detection. To enhance processing efficiency, we limit
object detection to keyframes only. Specifically, we compute the CLIP similarity [29] between the object retrieval



4


request **R** _det_ and the sampled video frames **F** and select
relevant keyframes **F** _key_ based on a threshold _t_ :


**F** _key_ = CLIP ~~s~~ imilarity( **R** _det_ _,_ **F** ) _> t_ (7)


Once the keyframes are identified, we utilize APE [34],
an efficient open-vocabulary object detection model that
accepts object descriptions as prompts to detect relevant
objects within frames based on specific retrieval queries.
The capability of APE makes it particularly well-suited to
our requirements for on-demand object retrieval in video
frames. Finally, the detected objects’ categories and their
corresponding positional information are stored in the DET
database using natural language representations:


_DB_ _det_ _←−_ **T** _det_ = APE( **F** _key_ _,_ **R** _det_ ) (8)


**3.4. Auxiliary Text Retrieval**


During the retrieve phase, we employ the Contriever
framework to encode the user’s query and the parsed requests for OCR and ASR into text embeddings, then
concatenating to form the final query request **E** _req_ =
Contriever(Concat( **R** _,_ **Q** )) _,_ **R** _∈_ _{_ **R** _ocr_ _,_ **R** _asr_ _}_ .
Then we retrieve the auxiliary texts from _DB_ _∈_
_{DB_ _ocr_ _, DB_ _asr_ _}_ by the FAISS tool, which computes the
vector similarity between the query and text chunks stored
in the database. Text chunks with a FAISS similarity score
greater than threshold _t_ are indexed as the retrieval results
**A** _∈{_ **A** _ocr_ _,_ **A** _asr_ _}_ . The process can be formulated as:


Index
**A** _←−−−−_ FAISS ~~s~~ imilarity( _DB,_ **E** _req_ ) _> t_ (9)


The information stored in the DET database undergoes
an initial retrieval process. Since the text generated by
the detection model is in a raw format (“category: [x ~~m~~ in,
y ~~m~~ in, length, width]”), it challenges LVLMs to understand
the relative relationships between objects. To address this
issue, we preprocess the object information using a scene
graph, which helps to represent spatial and relational information more explicitly. This preprocessing allows us to
construct more coherent and semantically meaningful texts,
denoted as **A** _[p]_ _det_ [, which are more readily interpretable by]
LVLMs. We incorporate three types of object information
for each video keyframe: **(i) Object Location A** _loc_ **:** This
refines the positional information of the object, formatted
as: “Object _{_ node ID _}_ is a _{_ object category _}_ located at coordinates [x, y] with dimensions _{_ length _×_ width _}_ ” **(ii) Ob-**
**ject Counting A** _cnt_ **:** This counts the number of objects and
generates text in the following format: “Object counting:
- _{_ object category _}_ : _{_ number _}_ ” **(iii) Relative Positional**
**Relationships A** _rel_ **:** This captures the relative spatial relationships between objects using the format: “Object _{_ node



ID _}_ ( _{_ object category _}_ ) is _<_ positional description _>_ Object
_{_ node ID _}_ ( _{_ object category _}_ )”. By combining this information, we construct a detailed representation of the objects
in the frame, denoted as **A** _[p]_ _det_ [=] _[ {]_ **[A]** _[loc]_ _[,]_ **[ A]** _[cnt]_ _[,]_ **[ A]** _[rel]_ _[}]_ [:]


**A** _[p]_ _det_ [=][ SceneGraph][(] _[DB]_ _[det]_ [)] (10)


Finally, we acquire the object auxiliary texts based on
the object information type retrieval requests **R** _type_ of the
LVLMs in the first phase, which selects and finalizes the
object auxiliary information **A** _det_ . **A** _det_ is one of the elements of the power set _P_ of **A** _[p]_ _det_ [selected by] **[ R]** _[type]_ [, and]
the retrieve process can be formulated as:


**A** _det_ = **R** _type_ ( _P_ ( **A** _[p]_ _det_ [))] _[ ∈P]_ [(] **[A]** _[p]_ _det_ [)] (11)


**3.5. Integration and Generation**


After obtaining different types of auxiliary texts, we organize them chronologically using natural language to
create a unified auxiliary input, denoted as **A** _m_ =
Concat( **A** _ocr_ _,_ **A** _asr_ _,_ **A** _det_ ). These merged auxiliary inputs, along with the user’s query and the sampled video
frames, are then fed into the LVLM to produce the final
result. The overall process can be formulated as:


**O** = LVLM( **F** _**v**_ _,_ Concat( **A** _m_ _,_ **Q** )) (12)


**4. Experiments**


**4.1. Datasets**


**Video-MME** [6] is a widely used benchmark for assessing
the ability of LVLMs to handle detailed videos in real-world
scenarios. It is divided into three subsets based on video

length, with durations ranging from 11 seconds to 1 hour.
**MLVU** [52] is a long video understanding benchmark with
a large wide of 9 distinct tasks. It is created based on long
videos of diversified lengths, ranging from 3 minutes to 2
hours with about 12 minutes average video length.
**LongVideoBench** [41] is a benchmark designed to accurately retrieve and reason over detailed multimodal information from long videos, with 6,678 human-annotated
multiple-choice questions in 17 fine-grained categories.


**4.2. Implementation Details**


We performed all experiments on NVIDIA A100 80G
GPUs. During the auxiliary text generation phase, we first
filter the detection requests **R** _det_ generated by the LVLM
to ensure they correspond to CLIP-sensitive physical entities, avoiding the inclusion of abstract concepts. In the
auxiliary text retrieval phase, we set both the CLIP similarity threshold and the FAISS similarity threshold _t_ to
0.3. We employ the IndexFlatIP as the similarity calculating method of FAISS [11]. We utilize Long-LLaVA-7B



5


**Model** **#Text** **LLM Params** **Frames** **Short** **Medium** **Long** **Overall** **Gain**


_**Proprietary LVLMs**_


_**Open-Source LVLMs**_


Table 1. Performance on the Video-MME [6] benchmark. **#Text** donates the average token number of auxiliary texts when inferring a
single case. By applying our Video-RAG to six LVLMs, we observed an average performance improvement of 8.0% only with the addition
of token counts from approximately 14 video frames (144 tokens per frame). In particular, when applying Video-RAG with 72B LLaVAVideo [49], we perform better than the proprietary method Gemini-1.5-Pro [31]. All open-source results are our replication.




[43] for ablation studies. Since Long-LLaVA is easy to implement while supporting longer context windows, we can
investigate the impact of similarity threshold selection in
RAG and sampled frame rate on performance. Note that we
don’t include the GPT-based Agent methods for comparison
due to their resource-intensive nature (complete execution
of Video-MME [6] costs around $2000 for API purchasing when using VideoAgent [4]). Still, we include a miniexperiment of VideoAgent in the Appendix that compares
the overall performance, inference time, and GPU requirements with two long-context LVLMs and Video-RAG.


**4.3. Main Results**


**Video-MME** . We evaluate our Video-RAG in four

7B open-source LVLMs, including Video-LLaVA [16],
LLaVA-NeXT-Video [48], LongVA [47], Long-LLaVA

[43], and two 72B LVLM Qwen2-VL [38] and LLaVAVideo [49]. To ensure a fair assessment of the performance
improvements introduced by our pipeline, and given the resource constraints (especially 72B LVLMs), we reproduced
the performance of all open-source models under the 32frame setting. Results are shown in Table 1. Specifically,
after applying our Video-RAG in 72B LVLM, we perform
better than the SOTA proprietary model Gemini-1.5-Pro

[31] (75.7% vs. 75.0%). Across the six LVLMs used in our
experiments, we gained an average performance boost of
8.0%, especially a significant gain on long videos, demonstrating our pipeline’s effectiveness. This performance improvement is achieved by incorporating token counts from



approximately 14 additional video frames (equivalent to
2.0K tokens), each contributing around 144 tokens under
most LVLM configurations. We obtain such a large performance enhancement because most LVLMs are pre-trained
primarily within the text space and aligned with visual information, often lacking explicit alignment between embedding spaces. Auxiliary texts can serve as semantic supplements sensitive to LVLMs, facilitating model activation and
easing the understanding of complex videos.
**MLVU** . We evaluate Video-RAG when integrating into
the 7B and 72B LLaVA-Video [49] of MLVU [52] in a
multiple-choice task. As shown in Table 2, when applying Video-RAG in the 7B model, we achieve state-of-theart performance at scales smaller than 72B using only a
7B model, outperforming the 32B Qryx-1.5 [24] model by
0.1%. Additionally, the 72B LLaVA-Video also has a performance gain of 0.7%. The limited performance gain may
be due to having reached a bottleneck of model capacity.
**LongVideoBench** . We evaluate Video-RAG when applied
in the 7B and 72B LLaVA-Video [49] of LongVideoBench

[41]. We omit the interleaved input format introduced in
LongVideoBench when applying Video-RAG. The evaluation results in Table 3 demonstrate that 72B LLaVA-Video

with our Video-RAG achieves an overall performance of
65.4% on the validation set. This result surpasses the proprietary LVLM Gemini-1.5-Pro [31] by 1.4%, securing the
second place, just 1.3% behind GPT-4o [28]. Meanwhile,
the 7B LLaVA-Video also has a performance enhancement
of 2.1% when equipped with our Video-RAG.



6


**Model** **#Params Frames Overall**


_**Proprietary LVLMs**_


_**Open-Source LVLMs**_


Video-CCAM [5] 14B 96 63.1
Video-XL [35] 7B 256 64.9
Aria [13] 25.3B 256 70.6
LLaVA-Video* [49] 7B 64 70.8
Oryx-1.5 [24] 32B 128 72.3
LLaVA-Video* [49] 72B 64 73.1


Table 2. The overall performance in the multiple-choice task of the
MLVU [52] benchmark. * donates the results of our replication.


**Model** **#Params Frames Overall**


_**Proprietary LVLMs**_


_**Open-Source LVLMs**_


VideoChat2-Mistral [14] 7B 8 39.3
ShareGPT4Video [2] 7B 8 39.7
LLaVA-Next-Mistral [20] 7B 8 49.1
PLLaVA [42] 34B 16 53.2
LLaVA-Video* [49] 7B 64 56.6
LLaVA-Video* [49] 72B 64 61.9


Table 3. The overall performance on the validation set of
LongVideoBench [41]. * donates the results of our replication.


**4.4. Ablation Studies**


**Effect of different sampling frame number.** To explore
the effect of the number of sampling frames on Video-RAG,
we experience sampling frames number 8, 16, 32, and 64
frames in 7B model Long-LLaVA [43], results are shown
in Figure 4. As demonstrated, Video-RAG consistently delivers performance improvements across all frame rates especially in long videos, with these gains increasing as the
frame rate rises. Furthermore, the results indicate that the
highest accuracy without Video-RAG is achieved when 32
frames are sampled from the video. Therefore, we adopt
this configuration in the subsequent ablation experiments.
**Effect of different components of Video-RAG.** To explore
the effectiveness of auxiliary texts, we incrementally add
object detection, OCR, and ASR as auxiliary texts after retrieving by the RAG system to evaluate Long-LLaVA-7B















|Overall (Without Video-RAG) Overall (With Video-RAG) Short Video Performance Short Video Gain Medium Video Performance Medium Video Gain Long Video Performance Long Video Gain|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|Overall (With Video~~-~~RAG)<br>Short Video Gain<br>Medium Video Gain<br>Long Video Gain|
||9.6|9.9|||10|.2||1|.8||
||||||||||||
||||||||||||
||8 Frame|16 Fra|es||32 Fr|mes||64 F|ames||
|re<br>o-|4.<br>Perf<br>MME [6]|ormance wit<br> when using|h dif<br>Lon|fere<br>g-LL|nt sa<br>aVA|mplin<br>-7B [|g fr<br>43] a|ames<br>s the|rat<br> LV|e on<br>LM.|


**RAG DET OCR ASR Short Medium Long Overall**


Table 4. Results on combinations of different auxiliary texts in
Video-MME [6] when using Long-LLaVA-7B [43] as the LVLM.


[43] in the Video-MME [6] benchmark. As shown in Table 4, the performance of Long-LLaVA progressively improves as auxiliary texts are incrementally added (52.0%
→ 52.9% → 54.3% → 62.1%). Among these components,
ASR auxiliary texts contribute to a general improvement for
different video durations, underscoring the critical role of
audio transcription that can provide comprehensive information beyond visual cues in video understanding. Meanwhile, we randomly sampled an equivalent token number
of auxiliary texts to serve as inputs for assessing the effectiveness of RAG retrieval. The experiment demonstrated
a 3.0% improvement in performance after incorporating
RAG. We also evaluate the performance across sub-tasks
within Video-MME [6] and other video benchmarks like
VNBench [51], more details are shown in the Appendix.
**Effect of different threshold of RAG processing.** When
using the RAG tool for retrieval, we specify a similarity
threshold _t_ as a criterion for information selection. In the

retrieval for OCR and ASR texts, information is selected if
its FAISS similarity exceeds _t_ . For object detection, frames
are selected as keyframes based on their CLIP similarity
surpassing _t_, and the relevant information is then extracted.
Setting _t_ too high may hinder the retrieval of relevant information while setting it too low can result in information
redundancy and increased reasoning complexity. To investigate this trade-off, we conduct ablation experiments to evaluate the impact of different threshold values. The results are



7


_**Finetuning**_











Figure 5. Qualitative result shown in Video-MME [6] benchmark when applying Video-RAG with LLaVA-Video [49].



**w/o Auxiliary Texts**



**with Auxiliary Texts**



_**Finetuning**_

_**LVLM**_


_**Video-RAG**_




**[Query-irrelevant Frames with less attention]** **[Key Frame with more attention]**




**[More aligned cross-modality features]**



Figure 6. Grad-CAM visualizations of the last hidden state heatmap along with t-SNE visualizations of the user’s query and keyframe
features of the example shown in Figure 5. As demonstrated, the retrieved auxiliary texts help **cross-modality alignment** by assisting the
model to **pay more attention to query-relevant keyframes** and thus generate more robust and accurate answers to the user’s query.



shown in Table 5. Notably, _t_ = 0 and _t_ = 1 correspond to
all auxiliary texts input into the model and no auxiliary texts
input, respectively. To balance performance with information density and processing time (especially APE [34] detection in keyframes), we selected a threshold of 0.3 for our
implementation. More details about similarity scores are
shown in the Appendix. Under this configuration, the additional text length of approximately 1.9K tokens typically
remains within the context window limits of open-source
LVLMs. For models with more stringent context window
limitations, a threshold of 0.4 may also be a viable option.


**4.5. Qualitative Evaluation**


We present qualitative results in the case of Video-MME

[6] in Figure 5 and Figure 6. As illustrated, augmenting
LLaVA-Video with external tools to process and retrieve
auxiliary texts from videos significantly enhances its ability
to reduce visual hallucinations, thereby enabling more accurate responses to user queries. Grad-CAM [32] and t-SNE

[37] visualization results also show that applying VideoRAG helps the LVLM’s cross-modality alignment.


**5. Conclusion**


In this paper, we present Video-RAG for effective long
video understanding by integrating retrieved auxiliary texts



_t_ **#Token Time Short Medium Long Overall**


Table 5. Performance with different thresholds of retrieval on

Video-MME [6] when using Long-LLaVA-7B [43] as the LVLM.
**#Token** and **Time** denote the total token number of the auxiliary
texts and the average inference time per question, respectively.


with LVLMs, achieving proprietary-level performance with
72B open-source LVLM. Unlike traditional long-context
and GPT-based Agent methods that may have limited
performance gain and are resource-intensive, Video-RAG
offers a resource-efficient, plug-and-play solution for
any open-source LVLMs that only leverage open-source
external tools to extract the visually-aligned auxiliary texts
from pure video data for input. In the future, we will
explore how to more efficiently integrate auxiliary texts and
provide an adaptive frame selection strategy for LVLMs.



8


**References**


[1] Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sarwar Uddin, and Srimat Chakradhar. irag: An incremental retrieval
augmented generation system for videos. _arXiv preprint_
_arXiv:2404.12309_, 2024. 2

[2] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang,
Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu
Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. _arXiv preprint_
_arXiv:2406.04325_, 2024. 1, 7

[3] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen,
Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu,
Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In _Pro-_
_ceedings of the IEEE/CVF Conference on Computer Vision_
_and Pattern Recognition_, pages 24185–24198, 2024. 1

[4] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi
Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. In _European Confer-_
_ence on Computer Vision_, pages 75–92. Springer, 2025. 2, 3,
6, 1

[5] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu,
and Hui Wang. Video-ccam: Enhancing video-language understanding with causal cross-attention masks for short and
long videos. _arXiv preprint arXiv:2408.14023_, 2024. 7

[6] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren,
Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen,
Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video

analysis. _arXiv preprint arXiv:2405.21075_, 2024. 1, 2, 5, 6,
7, 8, 3

[7] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In
_Proceedings of the IEEE/CVF Conference on Computer Vi-_
_sion and Pattern Recognition_, pages 14953–14962, 2023. 3

[8] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard
Grave. Unsupervised dense information retrieval with contrastive learning. _arXiv preprint arXiv:2112.09118_, 2021. 4,
1

[9] JaidedAI. Easyocr. [https : / / github . com /](https://github.com/JaidedAI/EasyOCR)
[JaidedAI/EasyOCR, 2023. 4](https://github.com/JaidedAI/EasyOCR)

[10] Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao,
and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. _arXiv preprint arXiv:2311.08046_, 2023. 1

[11] Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billionscale similarity search with gpus. _IEEE Transactions on Big_
_Data_, 7(3):535–547, 2019. 4, 5, 1

[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp
tasks. _Advances in Neural Information Processing Systems_,
33:9459–9474, 2020. 1, 4

[13] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi
Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and



Junnan Li. Aria: An open multimodal native mixture-ofexperts model. _arXiv preprint arXiv:2410.05993_, 2024. 7

[14] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai
Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao.
Videochat: Chat-centric video understanding. _arXiv preprint_
_arXiv:2305.06355_, 2023. 1, 3, 7

[15] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid:
An image is worth 2 tokens in large language models. In
_European Conference on Computer Vision_, pages 323–340.
Springer, 2025.

[16] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and
Li Yuan. Video-llava: Learning united visual representation by alignment before projection. _arXiv preprint_
_arXiv:2311.10122_, 2023. 3, 6

[17] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In _Proceedings of the IEEE/CVF Con-_
_ference on Computer Vision and Pattern Recognition_, pages
26689–26699, 2024. 1

[18] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin,
Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin
Liang, Zicheng Liu, Yumao Lu, et al. Mm-vid: Advancing video understanding with gpt-4v (ision). _arXiv preprint_
_arXiv:2310.19773_, 2023. 2, 3

[19] Qinghong Lin. Vlog: Transform video as a document with
[chatgpt, clip, blip2, grit, whisper, langchain. https://](https://github.com/showlab/VLog)
[github.com/showlab/VLog, 2023. 3](https://github.com/showlab/VLog)

[20] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan
Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3, 7

[21] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu.
Kangaroo: A powerful video-language model supporting
long-context video input. _arXiv preprint arXiv:2408.15542_,
2024. 1

[22] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan,
and Ge Li. St-llm: Large language models are effective temporal learners. In _European Conference on Computer Vision_,
pages 1–18. Springer, 2025.

[23] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand
spatial-temporal understanding at arbitrary resolution. _arXiv_
_preprint arXiv:2409.12961_, 2024. 1

[24] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand
spatial-temporal understanding at arbitrary resolution. _arXiv_
_preprint arXiv:2409.12961_, 2024. 6, 7

[25] Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li,
Hamid Rezatofighi, and Jianfei Cai. Drvideo: Document
retrieval based long video understanding. _arXiv preprint_
_arXiv:2406.12846_, 2024. 2, 3

[26] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video
understanding via large vision and language models. _arXiv_
_preprint arXiv:2306.05424_, 2023. 2

[27] Juhong Min, Shyamal Buch, Arsha Nagrani, Minsu Cho,
and Cordelia Schmid. Morevqa: Exploring modular reasoning models for video question answering. In _Proceedings of_



9


_the IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition_, pages 13235–13245, 2024. 3

[[28] OpenAI. Gpt-4o system card. https://openai.com/](https://openai.com/index/gpt-4o-system-card/)
[index/gpt-4o-system-card/, 2024. 2, 6, 7](https://openai.com/index/gpt-4o-system-card/)

[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervision. In _International conference on machine learning_, pages
8748–8763. PMLR, 2021. 4, 1

[30] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman,
Christine McLeavey, and Ilya Sutskever. Robust speech
recognition via large-scale weak supervision. In _Interna-_
_tional conference on machine learning_, pages 28492–28518.
PMLR, 2023. 4

[31] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry
Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu
Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint_
_arXiv:2403.05530_, 2024. 2, 6, 7

[32] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das,
Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra.
Grad-cam: Visual explanations from deep networks via
gradient-based localization. In _Proceedings of the IEEE in-_
_ternational conference on computer vision_, pages 618–626,
2017. 8

[33] Yuzhang Shang, Bingxin Xu, Weitai Kang, Mu Cai, Yuheng
Li, Zehao Wen, Zhen Dong, Kurt Keutzer, Yong Jae Lee,
and Yan Yan. Interpolating video-llms: Toward longersequence lmms in a training-free manner. _arXiv preprint_
_arXiv:2409.12963_, 2024. 1, 3

[34] Yunhang Shen, Chaoyou Fu, Peixian Chen, Mengdan Zhang,
Ke Li, Xing Sun, Yunsheng Wu, Shaohui Lin, and Rongrong
Ji. Aligning and prompting everything all at once for universal visual perception. In _Proceedings of the IEEE/CVF Con-_
_ference on Computer Vision and Pattern Recognition_, pages
13193–13203, 2024. 5, 8

[35] Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie
Zhou, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long
vision language model for hour-scale video understanding.
_arXiv preprint arXiv:2409.14485_, 2024. 7

[36] D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt:
Visual inference via python execution for reasoning. In
_Proceedings of the IEEE/CVF International Conference on_
_Computer Vision_, pages 11888–11898, 2023. 3

[37] Laurens Van der Maaten and Geoffrey Hinton. Visualizing
data using t-sne. _Journal of machine learning research_, 9
(11), 2008. 8

[38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan,
Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, et al. Qwen2-vl: Enhancing vision-language model’s
perception of the world at any resolution. _arXiv preprint_
_arXiv:2409.12191_, 2024. 6

[39] Xidong Wang, Dingjie Song, Shunian Chen, Chen Zhang,
and Benyou Wang. Longllava: Scaling multi-modal llms
to 1000 images efficiently via hybrid architecture. _arXiv_
_preprint arXiv:2409.02889_, 2024. 1




[40] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena
Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. _arXiv preprint_
_arXiv:2403.10517_, 2024. 3

[41] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li.
Longvideobench: A benchmark for long-context interleaved video-language understanding. _arXiv preprint_
_arXiv:2407.15754_, 2024. 2, 5, 6, 7

[42] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng,
and Jiashi Feng. Pllava: Parameter-free llava extension from
images to videos for video dense captioning. _arXiv preprint_
_arXiv:2404.16994_, 2024. 7

[43] Yin Song and Chen Wu and Eden Duthie. awsprototyping/long-llava-qwen2-7b, 2024. 1, 3, 6, 7, 8, 2

[44] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang,
Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering.
_arXiv preprint arXiv:2312.17235_, 2023. 3

[45] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An
instruction-tuned audio-visual language model for video understanding. _arXiv preprint arXiv:2306.02858_, 2023. 1

[46] Lu Zhang, Tiancheng Zhao, Heting Ying, Yibo Ma, and Kyusong Lee. Omagent: A multi-modal agent framework for
complex video understanding with task divide-and-conquer.
_arXiv preprint arXiv:2406.16620_, 2024. 2, 3

[47] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng,
Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan,
Chunyuan Li, and Ziwei Liu. Long context transfer from
language to vision. _arXiv preprint arXiv:2406.16852_, 2024.
1, 3, 6

[48] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke
Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024.
1, 3, 6

[49] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with
synthetic data, 2024. 2, 6, 7, 8

[50] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang,
Zhang Zhang, Liang Wang, and Rong Jin. Beyond llava-hd:
Diving into high-resolution large multimodal models. _arXiv_
_preprint arXiv:2406.08487_, 2024. 1

[51] Zijia Zhao, Haoyu Lu, Yuqi Huo, Yifan Du, Tongtian Yue,
Longteng Guo, Bingning Wang, Weipeng Chen, and Jing
Liu. Needle in a video haystack: A scalable synthetic framework for benchmarking video mllms. _arXiv preprint_, 2024.
7, 2, 3

[52] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi
Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng
Liu. Mlvu: A comprehensive benchmark for multi-task
long video understanding. _arXiv preprint arXiv:2406.04264_,
2024. 2, 5, 6, 7

[53] Yongshuo Zong, Ismail Elezi, Yongxin Yang, Jiankang
Deng, and Timothy Hospedales. Long-context vision large
language models: Empirical insights and a baseline. In
_Workshop on Long Context Foundation Models_, 2024. 1



10


## **Video-RAG : Visually-aligned Retrieval-Augmented Long Video Comprehension**





**6. Decouple Query**


## Supplementary Material



In the initial phase of the proposed Video-RAG, we employ a decouple prompt, denoted as **P**, to guide the LVLM
in generating retrieval requests. In this section, we present
one example of a prompt designed for multiple-choice questions, as illustrated in Figure 9.


**7. Sub-set of Video-MME**


As outlined in the implementation details, we randomly
sampled a subset of the Video-MME [6] dataset to evaluate
a computationally resource-intensive, agent-based method
with long-context LVLMs. Specifically, we selected 10%
of the full dataset, comprising 30 short, 30 medium-length,
and 30 long videos. Each video contains three multiplechoice questions. Importantly, we ensured that the performance ranking of the methods on the subset mirrored that
of the full dataset. As shown in Tables 6 and 7, we evaluated four distinct 7B models Chat-Univi-v1.5 [10], LLaVANeXT-Video [48], LongVA [47], and Long-LLaVA [43] using a frame sampling rate of 16 for both the subset and the
full set. Our results indicate that the performance rankings
remained consistent across both evaluations.


**Method** **Short Medium Long Overall**


Chat-Univi-v1.5 [10] 50.0 33.3 17.8 33.7
LLaVA-NeXT-Video [48] 54.4 33.3 23.3 37.0
LongVA [47] 56.7 50.0 38.9 48.5
Long-LLaVA [43] 58.9 52.2 40.0 50.4


Table 6. Performance of Video-MME sub-set.


**Method** **Short Medium Long Overall**


Chat-Univi-v1.5 [10] 45.7 39.0 35.7 40.1
LLaVA-NeXT-Video [48] 51.1 41.8 36.8 43.2
LongVA [47] 60.8 45.2 41.4 49.1
Long-LLaVA [43] 59.3 49.3 44.4 51.0


Table 7. Performance of Video-MME full-set.


**8. Results on Video-MME Sub-Set**


We examine Video-RAG against two representative methods in terms of inference time, GPU resource requirements, and overall performance. Given that GPT-based


|0<br>1<br>0<br>0|Col2|Col3|Col4|LongV|A (16 f|rames)|Col8|+ Video|-RAG|Col11|Col12|Col13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**0**<br>**0**<br>**0**<br>**1**|**4% bet**|**ter**||**S**<br>**Long-**<br>** (16 fr**|**aving 1**<br>**LLaVA**<br>**ames)**|**L**<br>**50GB o**<br>|**L**<br>**50GB o**<br>|**ongVA**<br>**f GPU**|** (128 fr**<br>**Lon**<br>**Memo**|**ames)**<br>**gVA (38**<br>**ry**|**4 fram**|**es)**|
|**0**<br>**0**<br>**0**<br>**1**|||**Long**<br>|**VA (16**<br>|** frames**|**)**|**)**||||||
|**0**<br>**0**<br>**0**<br>**1**|||**Long**<br>|**VA (16**<br>|** frames**|**)**||**: GPT-**<br>|**Based Age**<br>**: L**<br>|**nt metho**<br>**ong-Cont**<br>|**d**<br>**ext LVLM**||
|**0**<br>**0**<br>**0**<br>**1**||~~**ideoA**~~|~~**ent (2**~~|~~**fps)**~~||||~~**: Our V**~~|~~**ideo-RA**~~||||
|**0**<br>**0**<br>**0**<br>**1**||~~**ideoA**~~|~~**ent (2**~~|~~**fps)**~~|||||||||



**20** **40** **60** **80** **100** **120** **140** **160** **180** **200**


**GPU Resource Requirements (GB)**


Figure 7. The comparison of our Video-RAG with two common
approaches. The size of the bubbles represents the total time consumed for completing inference on the Video-MME [6] sub-set.


Agent methods are resource-intensive, we randomly sampled a sub-set of the Video-MME [6] for evaluation, as described in Section 7. As demonstrated in Figure 7, VideoAgent [4], a typically GPT-based Agent method, requires
significant time to process video and deliver suboptimal
performance. Meanwhile, LongVA [47], a representative
long-context LVLM, shows limited improvement from increasing the frame rate and even experiences performance
degradation. Integrating our Video-RAG into the 16-frame
LongVA results in substantial performance improvements
while reducing GPU resource consumption. Specifically,
with only increasing 8GB GPU memory compared to the
base (16-frames LongVA), we achieve 11.5% overall performance improvement, while outperforming another longcontext LVLM Long-LLaVA-7B [43] in 16-frames setting
by 9.6% with less GPU memory requirements and compatible total inference time. These results demonstrated that our

Video-RAG is lightweight with lower computing overhead
than the other typical methods.


**9. Details of Similarity Score Calculation**


In the process of using the RAG system to retrieve auxiliary
texts extracted from videos, we define a similarity threshold _t_ to ensure the selection of relevant texts. Specifically,
we employ FAISS-based [11] similarity to select OCR and
ASR texts, while CLIP [29] similarity is used for keyframe
selection. In our implementation, the similarity threshold _t_
is set to 0.3. As for OCR and ASR selection, For any given
list of the retrieve request **R** and auxiliary texts **A**, the Contriever [8] framework maps the text to a text embedding as:











1


**E** _a_ _i_ = Contriever( **A** _i_ ) _,_ _i_ = 1 _,_ 2 _, . . ., n_


**E** _r_ _i_ = Contriever( **R** _i_ ) _,_ _i_ = 1 _,_ 2 _, . . ., n_


The average embedding of the retrieve request is then
computed as:



**E** _r_ = [1]

_n_



_n_
� **E** _r_ _i_


_i_ =1



After that, the embedding of the request and the list of
auxiliary texts is normalized:

**E** _a_ _i_ = _∥_ **EE** _aa_ _ii_ _∥_ _[,]_ **E** _r_ = _∥_ **EE** _rr_ _∥_


The similarity between the query embedding **E** _r_ and the
document vector **E** _a_ is computed using the inner product,
the FAISS library is employed to efficiently perform this
search and return the indices of the auxiliary texts meeting
the criterion:


_S_ ( **E** _r_ _,_ **E** _a_ _i_ ) = **E** _r_ _·_ **E** _a_ _i_ _> t_


As for object detection, we use CLIP to select the video
keyframe. During this process, we first filter the object detection request **R** _det_ to ensure they correspond to CLIPsensitive physical entities, avoiding the inclusion of abstract
concepts. Specifically, if it is a single word, direct part-ofspeech filtering is applied; if it is a compound word, certain
rules are followed to check for compliance, such as whether
it is an adjective plus a noun, or a noun plus a noun. We use
the Spacy library to achieve this. After this, we put the text
“A picture of” before each object detection request.
Then, we extracting embedding from both the video
frames **F** and the detection request **R** _det_ :


**E** **F** _j_ = CLIP( **F** _j_ ) _,_ _j_ = 1 _,_ 2 _, . . ., m_


**E** **R** _i_ = CLIP( **R** _det_ _i_ ) _,_ _i_ = 1 _,_ 2 _, . . ., n_


The similarity between each video frame and the detection retrieve requests is computed using the dot product between the image and text feature embeddings. For each
frame **F** _j_, and for each retrieve request **E** **R** _i_, the similarity score is given by:


_S_ _ij_ = **E** **F** _j_ _·_ **E** **R** _i_

where _·_ denotes the dot product. The final similarity score
for each frame is the average similarity across all requests:



_S_ _j_ = [1]

_n_



_n_
� _S_ _ij_


_i_ =1



This computes the mean similarity for each frame across
all text descriptions, resulting in a similarity vector **S** =




[ _S_ 1 _, S_ 2 _, . . ., S_ _m_ ]. The similarity scores are adjusted by a
scaling factor _α_, which is computed based on the number
of frames _m_ and a base frame number _b_ (which is set to 16
and 4.0, respectively) to adapted different video sampling
rate of LVLMs:


_α_ = _β ×_ _[m]_

_b_

where _β_ is a predefined scaling parameter.
Next, the similarity scores are scaled and normalized to
ensure that they sum to 1:

_S_ _j_ [norm] = ~~�~~ _α_ ~~_m_~~ _k_ _×_ =1 _S_ _[S]_ _j_ _[k]_


where _S_ _j_ [norm] represents the normalized similarity score for
frame **F** _j_ .
The final step is to select the keyframes based on the
normalized similarity scores. A threshold _t_ is applied to
the normalized similarities, such that frames with similarity
scores above the threshold are selected as keyframes:


Keyframe: **F** _j_ if _S_ _j_ [norm] _> t_


Thus, the set of selected keyframes is given by:


**F** _key_ = _{_ **F** _j_ _| S_ _j_ [norm] _> t, j_ = 1 _,_ 2 _, . . ., m}_


**10. More Ablation Studies**


**Effect of different components of Video-RAG.** We evaluate the performance across sub-tasks within Video-MME

[6], as shown in Figure 8. The results reveal that object detection auxiliary texts significantly enhance spatial perception and object counting, while OCR auxiliary texts specifically improve performance on text recognition tasks. Additionally, ASR auxiliary texts contribute to a general improvement in inference tasks, underscoring the critical role
of audio transcription in video understanding. Given that
audio transcription is considerably more time-consuming
than character recognition or object detection, these texts
should be selected based on the requirements of the application.

Besides studying the inference of different components
of Video-RAG in the Video-MME [6] benchmark, we also
experiment with a different type of video benchmark VNBench [51] with Long-LLaVA-7B [43]. VNBench is a synthetic benchmark designed to evaluate models’ long-context
abilities, covering tasks such as retrieval, ordering, and
counting. VNBench randomly inserts stickers or text into
the video that has nothing to do with the original content of
the video, thus typically challenging the model’s needle-inthe-haystack capability. As shown in Table 8, we find that
applying DET and OCR as auxiliary texts can significantly
improve the performance in retrieval, ordering, and counting tasks. However, the ASR component will decline the



2


Figure 8. Performance on 12 sub-tasks in Video-MME [6] benchmark after applying different components in Long-LLaVA.


performance due to the subtitles are not ancillary to this particular task. These results demonstrated that our proposed
distinct types of auxiliary texts can be selected according to
the application needs to meet the requirements better.


**RAG** **DET** **OCR** **ASR** **Ret** **Ord** **Cnt** **Overall**


Table 8. Results on combinations of different auxiliary texts in
VNBench [51] with 1-try setting when applying 7B Long-LLaVA

[43] as LVLM under the 32-frames setting. **Ret**, **Ord**, and **Cnt**
represent retrieval, ordering, and counting tasks, respectively.


**11. More Qualitative Results**


In this section, we show more results of LLaVA-Vdieo-7B
when applying Video-RAG in different examples in Figure 10. The figure highlights several representative cases
involving detailed video comprehension from Video-MME

[6]. As illustrated, augmenting LLaVA-Video with external tools to process and retrieve auxiliary texts from videos
significantly enhances its ability to reduce visual hallucinations, thereby enabling more accurate and confident responses to user queries.



3


Figure 9. Decouple prompt of the multiple-choice question for LVLMs.


4


_**Finetuning**_

_**LVLM**_




```
request = {

“ASR”: “The athlete first meet the student.”,

“DET”: [“athlete”, “student”],

“TYPE”: [“relation”]

```





```
}

```


**OCR / ASR**
**Databases**



_**Video-RAG**_ _**Video-RAG**_




**[OCR Frame x]**




**[OCR Frame y]**











_**Finetuning**_

_**LVLM**_






```
request = {

```

`“ASR”: “The number of the` attractions `.”,`



**OCR / ASR** **`Rock;`**
**Databases**



`“DET”: [“` attractions `”],`

```
“TYPE”: [“number”]

}

```




_**Video-RAG**_ _**Video-RAG**_




**[OCR Frame 1]**




**[OCR Frame 2]**



**......** **[OCR Frame 10]**





_**Finetuning**_





















Figure 10. Qualitative results of LLaVA-Vdieo when applying Video-RAG.


5


