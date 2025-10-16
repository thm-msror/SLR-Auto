# <span id="page-0-1"></span>Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension

Yongdong Luo<sup>1</sup>, Xiawu Zheng<sup>1</sup>, Xiao Yang<sup>1</sup>, Guilin Li<sup>1</sup>, Haojia Lin<sup>1</sup>, Jinfa Huang<sup>2</sup>, Jiayi Ji<sup>1</sup>, Fei Chao<sup>1</sup>, Jiebo Luo<sup>2</sup>, Rongrong Ji<sup>1</sup>

<sup>1</sup>Xiamen University

<sup>2</sup>University of Rochester

## **Abstract**

Existing large video-language models (LVLMs) struggle to comprehend long videos correctly due to limited context. To address this problem, fine-tuning long-context LVLMs and employing GPT-based agents have emerged as promising solutions. However, fine-tuning LVLMs would require extensive high-quality data and substantial GPU resources, while GPT-based agents would rely on proprietary models (e.g., GPT-40). In this paper, we propose Video Retrieval-Augmented Generation (Video-RAG), a training-free and cost-effective pipeline that employs visually-aligned auxiliary texts to help facilitate cross-modality alignment while providing additional information beyond the visual content. Specifically, we leverage open-source external tools to extract visually-aligned information from pure video data (e.g., audio, optical character, and object detection), and incorporate the extracted information into an existing LVLM as auxiliary texts, alongside video frames and queries, in a plug-and-play manner. Our Video-RAG offers several key advantages: (i) lightweight with low computing overhead due to single-turn retrieval; (ii) easy implementation and compatibility with any LVLM; and (iii) significant, consistent performance gains across long video understanding benchmarks, including Video-MME, MLVU, and Long Video Bench. Notably, our model demonstrates superior performance over proprietary models like Gemini-1.5-Pro and GPT-40 when utilized with a 72B model. 1

#### 1. Introduction

With the advancements in Large Language Models (LLMs), numerous studies have been conducted to enhance their ability to comprehend and process videos [2, 3, 10, 14–17, 21–23, 45, 48, 50], collectively termed Large Video-Language Models (LVLMs). Although current LVLMs have demonstrated promising performance in understanding short videos, effective comprehension of extremely long videos continues to be a major challenge.

![](_page_0_Figure_8.jpeg)

Figure 1. Illustration of two common approaches for understanding long videos, alongside our Video-RAG. Video-RAG provides a resource-efficient, training-free pipeline that is easily compatible with any LVLM. By leveraging RAG, it retrieves auxiliary texts for input, leading to notable performance enhancement.

To address this challenge, recent studies [33, 39, 43, 47, 53] have sought to extend the reasoning context length of LVLMs, essentially finetuning long-context LVLMs for long video understanding. LongVA [47] first introduces increasing the token capacity of an LLM and transferring its long-context comprehension capabilities to video data. However, training such a model requires pre-training on an extended corpus, and often there are distribution shifts between deployment videos and finetuning videos. As demonstrated in Video-MME [6], LongVA declines when increasing the video frame sampling rate from 128 to 384 (52.6%)  $\rightarrow$  51.8%). This outcome suggests that simply increasing the number of sampled frames not only leads to information redundancy but also imposes additional challenges for the model to handle complex reasoning. Retrieval-Augmented Generation [12] (RAG) is a technique that enhances generative tasks by retrieving relevant documents from an external corpus, thus improving response quality in LLMs. Recent studies have begun exploring the integration of RAG with

<span id="page-0-0"></span> $<sup>^{1}</sup>Our\ code$  is available at https://github.com/Leon1207/Video-RAG-master.

<span id="page-1-1"></span>![](_page_1_Figure_0.jpeg)

Figure 2. Comparison of the performance of Video-RAG with LLaVA-Video-72B [\[49\]](#page-9-8), Gemini-1.5-Pro [\[31\]](#page-9-9), and GPT-4o [\[28\]](#page-9-10) across various benchmarks, including the sub-tasks from Video-MME [\[6\]](#page-8-7) (here we focus only on those that outperform Gemini-1.5-Pro), LongVideoBench [\[41\]](#page-9-11), and MLVU [\[52\]](#page-9-12) benchmarks.

video-based tasks [\[1,](#page-8-9) [25,](#page-8-10) [46\]](#page-9-13), employing tools to process videos in long contexts and sending them to a proprietary model for generation, which is known as the GPT-based Agent method. However, they come with serval limitations. First, most of them process long video content as plain text, subsequently utilizing the RAG mechanisms to retrieve relevant documents for LLMs. Therefore, they lack alignment with the visual context of the video, resulting in a loss of critical visual information. Second, they are often resource-intensive in multi-turn interactions and typically require powerful LLMs to function as the driving force, thus limiting their flexibility and generative capabilities. Executing the whole Video-MME [\[6\]](#page-8-7) using VideoAgent [\[4\]](#page-8-11) requires approximately 20 days and incurs a substantial consumption of GPT-4o API tokens.

In this study, we propose Video-RAG, an effective RAG pipeline that can be seamlessly integrated with any LVLM. Specifically, instead of simply increasing the number of sampled video frames, we propose to replace the corresponding extended visual tokens with auxiliary texts extracted from pure video data by invoking open-source foundation models, such as optical character recognition (OCR), automatic speech recognition (ASR), and object detection. These auxiliary texts are more aligned with the visual context while providing additional information beyond the visual data, as demonstrated in [\[4,](#page-8-11) [18\]](#page-8-12). Besides dealing with the context windows limit of LVLMs, we employ RAG in Video-RAG to filtering auxiliary texts, ensuring their relevance to the user's query in the text embedding space. As sampled visual context often lacks explicit alignment with the instructions, the auxiliary texts can facilitate crossmodality alignment while reducing the modality divide. As illustrated in Figure [6,](#page-7-0) with Video-RAG, the retrieved auxiliary texts help guide the LVLM to pay more attention to the query-relevant keyframes, while simultaneously facilitating cross-modality alignment between query and keyframes. In this framework, an LVLM serves as the central component of Video-RAG, processing visual tokens to preserve detailed visual context and minimize potential information loss. Moreover, the retrieval process is parallelly executed in a single operation, ensuring the efficiency of our pipeline.

We evaluate Video-RAG across several long video benchmarks, including Video-MME [\[6\]](#page-8-7), MLVU [\[52\]](#page-9-12), and LongVideoBench [\[41\]](#page-9-11). By applying the Video-RAG to six distinctive open-source LVLMs, we achieve an average performance improvement of 8.0% on Video-MME with only 2.0K text tokens addition (equal to 14 frames in most configuration) per case, while beating the proprietary LVLM Gemini-1.5-Pro [\[31\]](#page-9-9) when integrated with the 72B model, as shown in Figure [7.](#page-10-0) Applying Video-RAG to a 7B LVLM only requires an additional 8GB of inference GPU memory and approximately 5 seconds of inference time per case.

In summary, our contributions are as follows:

- We integrate RAG into open-source LVLMs:[2](#page-1-0) Video-RAG incorporates three types of visually-aligned auxiliary texts (OCR, ASR, and object detection) processed by external tools and retrieved via RAG, enhancing the LVLM. It's implemented using completely open-source tools, without the need for any commercial APIs.
- We design a versatile plug-and-play RAG-based pipeline for any LVLM: Video-RAG offers a trainingfree solution for a wide range of LVLMs, delivering performance improvements with minimal additional resource requirements.
- We achieve proprietary-level performance with opensource models: Applying Video-RAG to a 72B opensource model yields state-of-the-art performance in Video-MME, surpassing models such as Gemini-1.5-Pro.

## 2. Related Work

## 2.1. Large Video-Language Models

With the rapid advancement of large language models (LLMs), there has been increasing interest in developing generalist video models capable of handling a wide range of video-related tasks. Video-ChatGPT [\[26\]](#page-8-13) extracts features from individual frames and aggregates them through both

<span id="page-1-0"></span><sup>2</sup>While some methods [\[1,](#page-8-9) [25,](#page-8-10) [46\]](#page-9-13) use RAG system for video tasks, they convert the video data fully into text while also relying on proprietary, nonopen-source models. These approaches may result in the loss of visual information and lead to significant resource and time consumption.

<span id="page-2-1"></span><span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 3. The framework of our Video-RAG pipeline that contains three key phases. In the query decouple phase, the LVLM is prompted to generate a retrieval request for auxiliary texts. Next, in the auxiliary text generation and retrieval phase, the video is processed in parallel to extract three types of textual information (OCR, ASR, and object detection), and the relevant text is retrieved as the auxiliary text. Finally, in the integration and generation phase, auxiliary texts are combined with the query and the video to generate the response.

spatial and temporal pooling operations. VideoChat [\[14\]](#page-8-3) encodes videos by generating both textual descriptions and video appearance embeddings. Video-LLaVA [\[16\]](#page-8-14) aligning image and video encoders during a pre-processing phase, using a shared projector to map the encoded representations into a common language space. LLaVA-NeXT-Video [\[48\]](#page-9-1) extends LLaVA-NeXT [\[20\]](#page-8-15) by fine-tuning the model specifically on video data. Despite their contributions, these approaches face challenges when processing detailed and long-length videos, primarily due to the limited number of frames sampled for analysis.

### 2.2. Long-context Large Video-Language Models

Recent approaches have sought to expand the context window size to enhance detailed video understanding. LongVA [\[47\]](#page-9-6) and Long-LLaVA [\[43\]](#page-9-5) address this by continuously training LLMs on extended textual data, to transfer their long-text comprehension capabilities to video processing. INTP [\[33\]](#page-9-3) introduces a video token rearrangement technique while proposing a training-free method for extending the LLM context window, allowing LVLMs to process increased visual tokens. However, these methods face challenges in striking a balance between the high computational costs associated with sampling video frames and the limited performance improvements achieved. Due to the inherent redundancy in video content and constraints on model capacity, performance degradation may occur when the number of sampled frames surpasses a certain threshold.

## 2.3. GPT-based Agent Video Understanding

Initial efforts [\[7,](#page-8-16) [27,](#page-8-17) [36,](#page-9-14) [40,](#page-9-15) [44\]](#page-9-16) have employed LLMs to interact with tools to process visual information as structured long context for question-answering. MM-VID [\[18\]](#page-8-12) enhances long video understanding by aligning video frames with corresponding text descriptions. VLog [\[19\]](#page-8-18) leverages multimodel pre-trained models to capture and interpret visual and audio information, summarizing it into documents for video comprehension. VideoAgent [\[4\]](#page-8-11), DrVideo [\[25\]](#page-8-10), and OmAgent [\[46\]](#page-9-13) integrate multimodal inputs and enable dynamic querying of video segments to support long video reasoning tasks. However, these methods take an extremely long time to process videos while relying on proprietary models (e.g., GPT-4o), thus limiting their efficiency and adaptability to other open-source frameworks.

# 3. Method

We propose a novel, training-free pipeline for large videolanguage models (LVLMs), named Video-RAG, which can be integrated into any LVLM. As illustrated in Figure [3,](#page-2-0) our pipeline comprises three key phases: (i) Query Decouple: In this phase, the user's query is decomposed into a retrieval <span id="page-3-0"></span>request aimed at extracting auxiliary texts from the target video. (ii) Auxiliary Text Generation & Retrieval: Multiple auxiliary texts are generated from the queried video in parallel. Then, the retrieval request is used to obtain relevant external information. (iii) Integration and Generation: This phase integrates the retrieved auxiliary texts with the user's query, feeding this combined input into the LVLMs to generate the final response.

### 3.1. Large Video-Language Model

Given a video V, a frame sampler first sample N frames F. Most existing methods uniformly sample frames from a video for both effectiveness and simplicity. Then, video features are extracted as F<sup>v</sup> = VisualEnc(F), where VisualEnc is an image-based visual encoder, such as CLIP-L [\[29\]](#page-9-17). Finally, the video features F<sup>v</sup> and the user's query Q are fed into the LVLM to generate an output O:

$$\mathbf{O} = \text{LVLM}(\mathbf{F}_{\boldsymbol{v}}, \mathbf{Q}) \tag{1}$$

## 3.2. Query Decouple

In this phase, upon receiving a user's query about the video, the LVLM begins by decoupling the query and generating retrieval requests, denoted as R, for auxiliary texts. During this phase, the LVLM processes only textual information, without access to video frames, and the output requests are formatted in JSON. We prompt the LVLM using a decoupling prompt P to generate the following retrieval requests as necessary: (i) Rasr: Requests about automatic speech recognition, to extract audio information from the video that may pertain to the query. (ii) Rdet: Requests for identifying physical entities within the video that may assist in answering the query. (iii) Rtype: Requests for details about the location, quantity, and relationships of the identified physical entities. These requests, which may be NULL (indicating that the corresponding information is not required), are then parsed and forwarded to the auxiliary text retrieval phase. The entire process can be formally described as:

$$\mathbf{R} = \text{LVLM}(\mathbf{P}, \mathbf{Q}), \ \mathbf{R} = \{\mathbf{R}_{asr}, \mathbf{R}_{det}, \mathbf{R}_{type}\}$$
 (2)

### 3.3. Auxiliary Text Generation

In this phase, we first generate the auxiliary texts from the video and then retrieve them to assist the LVLMs according to the retrieval requests R. As the length of the video increases, the number of tokens generated from the processed data also grows, leading to an increase in redundant information. Additionally, current open-source models are constrained by the limited length of their context windows, which may prevent them from fully processing all auxiliary texts. To address this issue, we draw inspiration from Retrieval-Augmented Generation (RAG) [\[12\]](#page-8-8), retrieving only the auxiliary texts relevant to the user's query. Before retrieval, we construct the necessary databases from the given video in parallel. Specifically, we implement three distinct databases: the Optical Character Recognition (OCR) database, denoted as DBocr; the Automatic Speech Recognition (ASR) database, denoted as DBasr; and the Object Detection (DET) database, denoted as DBdet.

OCR database. Current LVLM are still illusory in their ability to accurately recognize characters, and their performance often falls short compared to proprietary models. To better leverage the information contained in video frames and reduce hallucinations, we employ a proprietary OCR model to extract text from each sampled video frame. Specifically, we use EasyOCR [\[9\]](#page-8-19) as our text recognition model and segmented the recognized texts on a per-frame basis, denoted as Tocr. Subsequently, we implemented RAG by utilizing the advanced text encoding model Contriever [\[8\]](#page-8-20) to encode the fetched OCR texts into text embeddings Eocr. These embeddings are then stored in a database with the FAISS index [\[11\]](#page-8-21), a library designed for efficient similarity search and clustering of dense vectors. The entire building process can be formally described as:

$$\mathbf{T}_{ocr} = \mathtt{EasyOCR}(\mathbf{F}) \tag{3}$$

$$DB_{ocr} \xleftarrow{\text{FAISS}} \mathbf{E}_{ocr} = \text{Contriever}(\mathbf{T}_{ocr})$$
 (4)

ASR database. Audio information (e.g., subtitles) plays a crucial role in video comprehension, often providing additional context that may not be available through visual cues alone. To incorporate them, we first extract the raw audio U from the video and then transcribe them into texts Tasr. Specifically, we use Whisper [\[30\]](#page-9-18) as our audio transcription model. Since the recognized texts can be quite extensive, we chunk and encode them into a vector database, following the same procedure used to construct the OCR database. The building process can be formally described as:

$$\mathbf{T}_{asr} = \mathtt{Whisper}(\mathbf{U})$$
 (5)

$$DB_{asr} \xleftarrow{\text{FAISS}} \mathbf{E}_{asr} = \text{Contriever}(\mathbf{T}_{asr})$$
 (6)

DET database. While LVLMs demonstrate strong performance in object recognition, they continue to face challenges such as object counting, precise object localization, and understanding relative relationships between objects. To mitigate the issue of hallucination, which can stem from these challenges, we incorporate object detection information as auxiliary texts. We leverage a visual grounding model to extract both the object categories and their corresponding positions from sampled video frames. This approach helps provide more accurate and context-aware object detection. To enhance processing efficiency, we limit object detection to keyframes only. Specifically, we compute the CLIP similarity [\[29\]](#page-9-17) between the object retrieval <span id="page-4-0"></span>request  $\mathbf{R}_{det}$  and the sampled video frames  $\mathbf{F}$  and select relevant keyframes  $\mathbf{F}_{key}$  based on a threshold t:

$$\mathbf{F}_{key} = \text{CLIP\_similarity}(\mathbf{R}_{det}, \mathbf{F}) > t$$
 (7)

Once the keyframes are identified, we utilize APE [34], an efficient open-vocabulary object detection model that accepts object descriptions as prompts to detect relevant objects within frames based on specific retrieval queries. The capability of APE makes it particularly well-suited to our requirements for on-demand object retrieval in video frames. Finally, the detected objects' categories and their corresponding positional information are stored in the DET database using natural language representations:

$$DB_{det} \leftarrow \mathbf{T}_{det} = APE(\mathbf{F}_{key}, \mathbf{R}_{det})$$
 (8)

#### 3.4. Auxiliary Text Retrieval

During the retrieve phase, we employ the Contriever framework to encode the user's query and the parsed requests for OCR and ASR into text embeddings, then concatenating to form the final query request  $\mathbf{E}_{req} = \texttt{Contriever}(\texttt{Concat}(\mathbf{R},\mathbf{Q})), \ \mathbf{R} \in \{\mathbf{R}_{ocr},\mathbf{R}_{asr}\}.$  Then we retrieve the auxiliary texts from  $DB \in \{DB_{ocr},DB_{asr}\}$  by the FAISS tool, which computes the vector similarity between the query and text chunks stored in the database. Text chunks with a FAISS similarity score greater than threshold t are indexed as the retrieval results  $\mathbf{A} \in \{\mathbf{A}_{ocr},\mathbf{A}_{asr}\}$ . The process can be formulated as:

$$\mathbf{A} \xleftarrow{\text{Index}} \text{FAISS\_similarity}(DB, \mathbf{E}_{req}) > t$$
 (9)

The information stored in the DET database undergoes an initial retrieval process. Since the text generated by the detection model is in a raw format ("category: [x\_min, y\_min, length, width]"), it challenges LVLMs to understand the relative relationships between objects. To address this issue, we preprocess the object information using a scene graph, which helps to represent spatial and relational information more explicitly. This preprocessing allows us to construct more coherent and semantically meaningful texts, denoted as  $\mathbf{A}_{det}^p$ , which are more readily interpretable by LVLMs. We incorporate three types of object information for each video keyframe: (i) Object Location  $A_{loc}$ : This refines the positional information of the object, formatted as: "Object {node ID} is a {object category} located at coordinates [x, y] with dimensions  $\{length \times width\}$ " (ii) Ob**ject Counting A** $_{cnt}$ : This counts the number of objects and generates text in the following format: "Object counting: - {object category}: {number}" (iii) Relative Positional **Relationships**  $A_{rel}$ : This captures the relative spatial relationships between objects using the format: "Object {node ID} ({object category}) is <positional description> Object {node ID} ({object category})". By combining this information, we construct a detailed representation of the objects in the frame, denoted as  $\mathbf{A}_{det}^p = \{\mathbf{A}_{loc}, \mathbf{A}_{cnt}, \mathbf{A}_{rel}\}$ :

$$\mathbf{A}_{det}^{p} = \text{SceneGraph}(DB_{det}) \tag{10}$$

Finally, we acquire the object auxiliary texts based on the object information type retrieval requests  $\mathbf{R}_{type}$  of the LVLMs in the first phase, which selects and finalizes the object auxiliary information  $\mathbf{A}_{det}$ .  $\mathbf{A}_{det}$  is one of the elements of the power set  $\mathcal{P}$  of  $\mathbf{A}_{det}^{\mathcal{P}}$  selected by  $\mathbf{R}_{type}$ , and the retrieve process can be formulated as:

$$\mathbf{A}_{det} = \mathbf{R}_{tupe}(\mathcal{P}(\mathbf{A}_{det}^p)) \in \mathcal{P}(\mathbf{A}_{det}^p)$$
(11)

#### 3.5. Integration and Generation

After obtaining different types of auxiliary texts, we organize them chronologically using natural language to create a unified auxiliary input, denoted as  $\mathbf{A}_m = \mathtt{Concat}(\mathbf{A}_{ocr}, \mathbf{A}_{asr}, \mathbf{A}_{det})$ . These merged auxiliary inputs, along with the user's query and the sampled video frames, are then fed into the LVLM to produce the final result. The overall process can be formulated as:

$$\mathbf{O} = \text{LVLM}(\mathbf{F}_{v}, \text{Concat}(\mathbf{A}_{m}, \mathbf{Q})) \tag{12}$$

## 4. Experiments

#### 4.1. Datasets

**Video-MME** [6] is a widely used benchmark for assessing the ability of LVLMs to handle detailed videos in real-world scenarios. It is divided into three subsets based on video length, with durations ranging from 11 seconds to 1 hour. **MLVU** [52] is a long video understanding benchmark with a large wide of 9 distinct tasks. It is created based on long videos of diversified lengths, ranging from 3 minutes to 2

**LongVideoBench** [41] is a benchmark designed to accurately retrieve and reason over detailed multimodal information from long videos, with 6,678 human-annotated multiple-choice questions in 17 fine-grained categories.

hours with about 12 minutes average video length.

#### 4.2. Implementation Details

We performed all experiments on NVIDIA A100 80G GPUs. During the auxiliary text generation phase, we first filter the detection requests  $\mathbf{R}_{det}$  generated by the LVLM to ensure they correspond to CLIP-sensitive physical entities, avoiding the inclusion of abstract concepts. In the auxiliary text retrieval phase, we set both the CLIP similarity threshold and the FAISS similarity threshold t to 0.3. We employ the IndexFlatIP as the similarity calculating method of FAISS [11]. We utilize Long-LLaVA-7B

<span id="page-5-1"></span><span id="page-5-0"></span>

| Model                        | #Text             | LLM Params | Frames  | Short | Medium | Long | Overall | Gain  |  |
|------------------------------|-------------------|------------|---------|-------|--------|------|---------|-------|--|
| Proprietary LVLMs            |                   |            |         |       |        |      |         |       |  |
| GPT-4o [28]                  | -                 | -          | 384     | 80.0  | 70.3   | 65.3 | 71.9    | -     |  |
| Gemini-1.5-Pro [31]          | -                 | -          | 0.5 fps | 81.7  | 74.3   | 67.4 | 75.0    | -     |  |
|                              | Open-Source LVLMs |            |         |       |        |      |         |       |  |
| Video-LLaVA [16]             | -                 | 7B         | 8       | 44.6  | 38.3   | 35.8 | 39.6    | -     |  |
| Video-LLaVA + Video-RAG      | 2.0K              | 7B         | 8       | 49.5  | 43.0   | 42.5 | 45.0    | +5.4  |  |
| LLaVA-NeXT-Video [48]        | -                 | 7B         | 16      | 49.4  | 43.0   | 36.7 | 43.0    | -     |  |
| LLaVA-NeXT-Video + Video-RAG | 2.0K              | 7B         | 16      | 56.6  | 47.4   | 46.0 | 50.0    | +7.0  |  |
| LongVA [47]                  | -                 | 7B         | 32      | 60.9  | 49.3   | 44.0 | 51.4    | -     |  |
| LongVA + Video-RAG           | 1.8K              | 7B         | 32      | 65.4  | 59.1   | 55.7 | 60.1    | +8.7  |  |
| Long-LLaVA [43]              | -                 | 7B         | 32      | 60.3  | 51.4   | 44.1 | 52.0    | -     |  |
| Long-LLaVA + Video-RAG       | 1.9K              | 7B         | 32      | 66.4  | 60.2   | 59.8 | 62.1    | +10.1 |  |
| Qwen2-VL [38]                | -                 | 72B        | 32      | 75.0  | 63.3   | 56.3 | 64.9    | -     |  |
| Qwen2-VL + Video-RAG         | 2.1K              | 72B        | 32      | 77.4  | 70.2   | 71.0 | 72.9    | +8.0  |  |
| LLaVA-Video [49]             | -                 | 72B        | 32      | 78.0  | 63.7   | 59.6 | 67.1    | -     |  |
| LLaVA-Video + Video-RAG      | 2.1K              | 72B        | 32      | 81.1  | 72.9   | 73.1 | 75.7    | +8.6  |  |

Table 1. Performance on the Video-MME [\[6\]](#page-8-7) benchmark. #Text donates the average token number of auxiliary texts when inferring a single case. By applying our Video-RAG to six LVLMs, we observed an average performance improvement of 8.0% only with the addition of token counts from approximately 14 video frames (144 tokens per frame). In particular, when applying Video-RAG with 72B LLaVA-Video [\[49\]](#page-9-8), we perform better than the proprietary method Gemini-1.5-Pro [\[31\]](#page-9-9). All open-source results are our replication.

[\[43\]](#page-9-5) for ablation studies. Since Long-LLaVA is easy to implement while supporting longer context windows, we can investigate the impact of similarity threshold selection in RAG and sampled frame rate on performance. Note that we don't include the GPT-based Agent methods for comparison due to their resource-intensive nature (complete execution of Video-MME [\[6\]](#page-8-7) costs around \$2000 for API purchasing when using VideoAgent [\[4\]](#page-8-11)). Still, we include a miniexperiment of VideoAgent in the Appendix that compares the overall performance, inference time, and GPU requirements with two long-context LVLMs and Video-RAG.

### 4.3. Main Results

Video-MME. We evaluate our Video-RAG in four 7B open-source LVLMs, including Video-LLaVA [\[16\]](#page-8-14), LLaVA-NeXT-Video [\[48\]](#page-9-1), LongVA [\[47\]](#page-9-6), Long-LLaVA [\[43\]](#page-9-5), and two 72B LVLM Qwen2-VL [\[38\]](#page-9-20) and LLaVA-Video [\[49\]](#page-9-8). To ensure a fair assessment of the performance improvements introduced by our pipeline, and given the resource constraints (especially 72B LVLMs), we reproduced the performance of all open-source models under the 32 frame setting. Results are shown in Table [1.](#page-5-0) Specifically, after applying our Video-RAG in 72B LVLM, we perform better than the SOTA proprietary model Gemini-1.5-Pro [\[31\]](#page-9-9) (75.7% vs. 75.0%). Across the six LVLMs used in our experiments, we gained an average performance boost of 8.0%, especially a significant gain on long videos, demonstrating our pipeline's effectiveness. This performance improvement is achieved by incorporating token counts from approximately 14 additional video frames (equivalent to 2.0K tokens), each contributing around 144 tokens under most LVLM configurations. We obtain such a large performance enhancement because most LVLMs are pre-trained primarily within the text space and aligned with visual information, often lacking explicit alignment between embedding spaces. Auxiliary texts can serve as semantic supplements sensitive to LVLMs, facilitating model activation and easing the understanding of complex videos.

MLVU. We evaluate Video-RAG when integrating into the 7B and 72B LLaVA-Video [\[49\]](#page-9-8) of MLVU [\[52\]](#page-9-12) in a multiple-choice task. As shown in Table [2,](#page-6-0) when applying Video-RAG in the 7B model, we achieve state-of-theart performance at scales smaller than 72B using only a 7B model, outperforming the 32B Qryx-1.5 [\[24\]](#page-8-22) model by 0.1%. Additionally, the 72B LLaVA-Video also has a performance gain of 0.7%. The limited performance gain may be due to having reached a bottleneck of model capacity.

LongVideoBench. We evaluate Video-RAG when applied in the 7B and 72B LLaVA-Video [\[49\]](#page-9-8) of LongVideoBench [\[41\]](#page-9-11). We omit the interleaved input format introduced in LongVideoBench when applying Video-RAG. The evaluation results in Table [3](#page-6-1) demonstrate that 72B LLaVA-Video with our Video-RAG achieves an overall performance of 65.4% on the validation set. This result surpasses the proprietary LVLM Gemini-1.5-Pro [\[31\]](#page-9-9) by 1.4%, securing the second place, just 1.3% behind GPT-4o [\[28\]](#page-9-10). Meanwhile, the 7B LLaVA-Video also has a performance enhancement of 2.1% when equipped with our Video-RAG.

<span id="page-6-4"></span><span id="page-6-0"></span>

| Model                   | #Params | Frames  | Overall     |  |  |  |  |  |
|-------------------------|---------|---------|-------------|--|--|--|--|--|
| Proprietary LVLMs       |         |         |             |  |  |  |  |  |
| GPT-4o [28]             | -       | 0.5 fps | 64.6        |  |  |  |  |  |
| Open-Source LVLMs       |         |         |             |  |  |  |  |  |
| Video-CCAM [5]          | 14B     | 96      | 63.1        |  |  |  |  |  |
| Video-XL [35]           | 7B      | 256     | 64.9        |  |  |  |  |  |
| Aria [13]               | 25.3B   | 256     | 70.6        |  |  |  |  |  |
| LLaVA-Video* [49]       | 7B      | 64      | 70.8        |  |  |  |  |  |
| Oryx-1.5 [24]           | 32B     | 128     | 72.3        |  |  |  |  |  |
| LLaVA-Video* [49]       | 72B     | 64      | <u>73.1</u> |  |  |  |  |  |
| LLaVA-Video + Video-RAG | 7B      | 64      | 72.4        |  |  |  |  |  |
| LLaVA-Video + Video-RAG | 72B     | 64      | 73.8        |  |  |  |  |  |

Table 2. The overall performance in the multiple-choice task of the MLVU [52] benchmark. \* donates the results of our replication.

<span id="page-6-1"></span>

| Model                   | #Params | Frames | Overall     |  |  |  |  |  |
|-------------------------|---------|--------|-------------|--|--|--|--|--|
| Proprietary LVLMs       |         |        |             |  |  |  |  |  |
| Gemini-1.5-Pro [31]     | -       | 256    | 64.0        |  |  |  |  |  |
| GPT-4o [28]             | -       | 256    | 66.7        |  |  |  |  |  |
| Open-Source LVLMs       |         |        |             |  |  |  |  |  |
| VideoChat2-Mistral [14] | 7B      | 8      | 39.3        |  |  |  |  |  |
| ShareGPT4Video [2]      | 7B      | 8      | 39.7        |  |  |  |  |  |
| LLaVA-Next-Mistral [20] | 7B      | 8      | 49.1        |  |  |  |  |  |
| PLLaVA [42]             | 34B     | 16     | 53.2        |  |  |  |  |  |
| LLaVA-Video* [49]       | 7B      | 64     | 56.6        |  |  |  |  |  |
| LLaVA-Video* [49]       | 72B     | 64     | 61.9        |  |  |  |  |  |
| LLaVA-Video + Video-RAG | 7B      | 64     | 58.7        |  |  |  |  |  |
| LLaVA-Video + Video-RAG | 72B     | 64     | <u>65.4</u> |  |  |  |  |  |

Table 3. The overall performance on the validation set of LongVideoBench [41]. \* donates the results of our replication.

#### 4.4. Ablation Studies

Effect of different sampling frame number. To explore the effect of the number of sampling frames on Video-RAG, we experience sampling frames number 8, 16, 32, and 64 frames in 7B model Long-LLaVA [43], results are shown in Figure 4. As demonstrated, Video-RAG consistently delivers performance improvements across all frame rates especially in long videos, with these gains increasing as the frame rate rises. Furthermore, the results indicate that the highest accuracy without Video-RAG is achieved when 32 frames are sampled from the video. Therefore, we adopt this configuration in the subsequent ablation experiments.

**Effect of different components of Video-RAG.** To explore the effectiveness of auxiliary texts, we incrementally add object detection, OCR, and ASR as auxiliary texts after retrieving by the RAG system to evaluate Long-LLaVA-7B

<span id="page-6-2"></span>![](_page_6_Figure_7.jpeg)

Figure 4. Performance with different sampling frames rate on Video-MME [6] when using Long-LLaVA-7B [43] as the LVLM.

<span id="page-6-3"></span>

| RAG          | DET | OCR          | ASR          | Short | Medium | Long | Overall |
|--------------|-----|--------------|--------------|-------|--------|------|---------|
|              |     |              |              | 60.3  | 51.4   | 44.1 | 52.0    |
|              |     |              | $\checkmark$ | 66.2  | 54.7   | 50.3 | 57.1    |
| $\checkmark$ | ✓   |              |              | 61.4  | 51.9   | 45.2 | 52.9    |
| $\checkmark$ | ✓   | $\checkmark$ |              | 63.2  | 53.2   | 46.3 | 54.3    |
|              | ✓   | $\checkmark$ | $\checkmark$ | 65.7  | 55.8   | 56.0 | 59.1    |
| $\checkmark$ | ✓   | $\checkmark$ | $\checkmark$ | 66.4  | 60.2   | 59.8 | 62.1    |

Table 4. Results on combinations of different auxiliary texts in Video-MME [6] when using Long-LLaVA-7B [43] as the LVLM.

[43] in the Video-MME [6] benchmark. As shown in Table 4, the performance of Long-LLaVA progressively improves as auxiliary texts are incrementally added (52.0%  $\rightarrow$  52.9%  $\rightarrow$  54.3%  $\rightarrow$  62.1%). Among these components, ASR auxiliary texts contribute to a general improvement for different video durations, underscoring the critical role of audio transcription that can provide comprehensive information beyond visual cues in video understanding. Meanwhile, we randomly sampled an equivalent token number of auxiliary texts to serve as inputs for assessing the effectiveness of RAG retrieval. The experiment demonstrated a 3.0% improvement in performance after incorporating RAG. We also evaluate the performance across sub-tasks within Video-MME [6] and other video benchmarks like VNBench [51], more details are shown in the Appendix.

Effect of different threshold of RAG processing. When using the RAG tool for retrieval, we specify a similarity threshold t as a criterion for information selection. In the retrieval for OCR and ASR texts, information is selected if its FAISS similarity exceeds t. For object detection, frames are selected as keyframes based on their CLIP similarity surpassing t, and the relevant information is then extracted. Setting t too high may hinder the retrieval of relevant information while setting it too low can result in information redundancy and increased reasoning complexity. To investigate this trade-off, we conduct ablation experiments to evaluate the impact of different threshold values. The results are

<span id="page-7-3"></span><span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

Figure 5. Qualitative result shown in Video-MME [\[6\]](#page-8-7) benchmark when applying Video-RAG with LLaVA-Video [\[49\]](#page-9-8).

<span id="page-7-0"></span>![](_page_7_Figure_2.jpeg)

Figure 6. Grad-CAM visualizations of the last hidden state heatmap along with t-SNE visualizations of the user's query and keyframe features of the example shown in Figure [5.](#page-7-1) As demonstrated, the retrieved auxiliary texts help cross-modality alignment by assisting the model to pay more attention to query-relevant keyframes and thus generate more robust and accurate answers to the user's query.

shown in Table [5.](#page-7-2) Notably, t = 0 and t = 1 correspond to all auxiliary texts input into the model and no auxiliary texts input, respectively. To balance performance with information density and processing time (especially APE [\[34\]](#page-9-19) detection in keyframes), we selected a threshold of 0.3 for our implementation. More details about similarity scores are shown in the Appendix. Under this configuration, the additional text length of approximately 1.9K tokens typically remains within the context window limits of open-source LVLMs. For models with more stringent context window limitations, a threshold of 0.4 may also be a viable option.

### 4.5. Qualitative Evaluation

We present qualitative results in the case of Video-MME [\[6\]](#page-8-7) in Figure [5](#page-7-1) and Figure [6.](#page-7-0) As illustrated, augmenting LLaVA-Video with external tools to process and retrieve auxiliary texts from videos significantly enhances its ability to reduce visual hallucinations, thereby enabling more accurate responses to user queries. Grad-CAM [\[32\]](#page-9-24) and t-SNE [\[37\]](#page-9-25) visualization results also show that applying Video-RAG helps the LVLM's cross-modality alignment.

## 5. Conclusion

In this paper, we present Video-RAG for effective long video understanding by integrating retrieved auxiliary texts

<span id="page-7-2"></span>

| t   | #Token | Time | Short | Medium | Long | Overall |
|-----|--------|------|-------|--------|------|---------|
| 0.0 | 3.6K   | 36s  | 67.6  | 59.4   | 59.1 | 62.0    |
| 0.1 | 3.4K   | 30s  | 67.0  | 59.7   | 59.1 | 61.9    |
| 0.2 | 2.7K   | 18s  | 66.0  | 60.2   | 59.2 | 61.8    |
| 0.3 | 1.9K   | 11s  | 66.4  | 60.2   | 59.8 | 62.1    |
| 0.4 | 0.8K   | 8s   | 65.6  | 58.0   | 58.3 | 60.6    |
| 0.5 | 0.3K   | 7s   | 63.1  | 54.9   | 50.2 | 56.1    |
| 1.0 | 0.0K   | 6s   | 60.3  | 51.4   | 44.1 | 52.0    |

Table 5. Performance with different thresholds of retrieval on Video-MME [\[6\]](#page-8-7) when using Long-LLaVA-7B [\[43\]](#page-9-5) as the LVLM. #Token and Time denote the total token number of the auxiliary texts and the average inference time per question, respectively.

with LVLMs, achieving proprietary-level performance with 72B open-source LVLM. Unlike traditional long-context and GPT-based Agent methods that may have limited performance gain and are resource-intensive, Video-RAG offers a resource-efficient, plug-and-play solution for any open-source LVLMs that only leverage open-source external tools to extract the visually-aligned auxiliary texts from pure video data for input. In the future, we will explore how to more efficiently integrate auxiliary texts and provide an adaptive frame selection strategy for LVLMs.

## References

- <span id="page-8-9"></span>[1] Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sarwar Uddin, and Srimat Chakradhar. irag: An incremental retrieval augmented generation system for videos. *arXiv preprint arXiv:2404.12309*, 2024. [2](#page-1-1)
- <span id="page-8-0"></span>[2] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. *arXiv preprint arXiv:2406.04325*, 2024. [1,](#page-0-1) [7](#page-6-4)
- <span id="page-8-1"></span>[3] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 24185–24198, 2024. [1](#page-0-1)
- <span id="page-8-11"></span>[4] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. In *European Conference on Computer Vision*, pages 75–92. Springer, 2025. [2,](#page-1-1) [3,](#page-2-1) [6,](#page-5-1) [1](#page-0-1)
- <span id="page-8-23"></span>[5] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang. Video-ccam: Enhancing video-language understanding with causal cross-attention masks for short and long videos. *arXiv preprint arXiv:2408.14023*, 2024. [7](#page-6-4)
- <span id="page-8-7"></span>[6] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. *arXiv preprint arXiv:2405.21075*, 2024. [1,](#page-0-1) [2,](#page-1-1) [5,](#page-4-0) [6,](#page-5-1) [7,](#page-6-4) [8,](#page-7-3) [3](#page-2-1)
- <span id="page-8-16"></span>[7] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14953–14962, 2023. [3](#page-2-1)
- <span id="page-8-20"></span>[8] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. *arXiv preprint arXiv:2112.09118*, 2021. [4,](#page-3-0) [1](#page-0-1)
- <span id="page-8-19"></span>[9] JaidedAI. Easyocr. [https : / / github . com /](https://github.com/JaidedAI/EasyOCR) [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR), 2023. [4](#page-3-0)
- <span id="page-8-2"></span>[10] Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. *arXiv preprint arXiv:2311.08046*, 2023. [1](#page-0-1)
- <span id="page-8-21"></span>[11] Jeff Johnson, Matthijs Douze, and Herve J ´ egou. Billion- ´ scale similarity search with gpus. *IEEE Transactions on Big Data*, 7(3):535–547, 2019. [4,](#page-3-0) [5,](#page-4-0) [1](#page-0-1)
- <span id="page-8-8"></span>[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨ aschel, et al. ¨ Retrieval-augmented generation for knowledge-intensive nlp tasks. *Advances in Neural Information Processing Systems*, 33:9459–9474, 2020. [1,](#page-0-1) [4](#page-3-0)
- <span id="page-8-24"></span>[13] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and

- Junnan Li. Aria: An open multimodal native mixture-ofexperts model. *arXiv preprint arXiv:2410.05993*, 2024. [7](#page-6-4)
- <span id="page-8-3"></span>[14] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. *arXiv preprint arXiv:2305.06355*, 2023. [1,](#page-0-1) [3,](#page-2-1) [7](#page-6-4)
- [15] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In *European Conference on Computer Vision*, pages 323–340. Springer, 2025.
- <span id="page-8-14"></span>[16] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. *arXiv preprint arXiv:2311.10122*, 2023. [3,](#page-2-1) [6](#page-5-1)
- <span id="page-8-4"></span>[17] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 26689–26699, 2024. [1](#page-0-1)
- <span id="page-8-12"></span>[18] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. Mm-vid: Advancing video understanding with gpt-4v (ision). *arXiv preprint arXiv:2310.19773*, 2023. [2,](#page-1-1) [3](#page-2-1)
- <span id="page-8-18"></span>[19] Qinghong Lin. Vlog: Transform video as a document with chatgpt, clip, blip2, grit, whisper, langchain. [https://](https://github.com/showlab/VLog) [github.com/showlab/VLog](https://github.com/showlab/VLog), 2023. [3](#page-2-1)
- <span id="page-8-15"></span>[20] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. [3,](#page-2-1) [7](#page-6-4)
- <span id="page-8-5"></span>[21] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. *arXiv preprint arXiv:2408.15542*, 2024. [1](#page-0-1)
- [22] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan, and Ge Li. St-llm: Large language models are effective temporal learners. In *European Conference on Computer Vision*, pages 1–18. Springer, 2025.
- <span id="page-8-6"></span>[23] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. *arXiv preprint arXiv:2409.12961*, 2024. [1](#page-0-1)
- <span id="page-8-22"></span>[24] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. *arXiv preprint arXiv:2409.12961*, 2024. [6,](#page-5-1) [7](#page-6-4)
- <span id="page-8-10"></span>[25] Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi, and Jianfei Cai. Drvideo: Document retrieval based long video understanding. *arXiv preprint arXiv:2406.12846*, 2024. [2,](#page-1-1) [3](#page-2-1)
- <span id="page-8-13"></span>[26] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. *arXiv preprint arXiv:2306.05424*, 2023. [2](#page-1-1)
- <span id="page-8-17"></span>[27] Juhong Min, Shyamal Buch, Arsha Nagrani, Minsu Cho, and Cordelia Schmid. Morevqa: Exploring modular reasoning models for video question answering. In *Proceedings of*

- *the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13235–13245, 2024. [3](#page-2-1)
- <span id="page-9-10"></span>[28] OpenAI. Gpt-4o system card. [https://openai.com/](https://openai.com/index/gpt-4o-system-card/) [index/gpt-4o-system-card/](https://openai.com/index/gpt-4o-system-card/), 2024. [2,](#page-1-1) [6,](#page-5-1) [7](#page-6-4)
- <span id="page-9-17"></span>[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021. [4,](#page-3-0) [1](#page-0-1)
- <span id="page-9-18"></span>[30] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In *International conference on machine learning*, pages 28492–28518. PMLR, 2023. [4](#page-3-0)
- <span id="page-9-9"></span>[31] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv preprint arXiv:2403.05530*, 2024. [2,](#page-1-1) [6,](#page-5-1) [7](#page-6-4)
- <span id="page-9-24"></span>[32] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE international conference on computer vision*, pages 618–626, 2017. [8](#page-7-3)
- <span id="page-9-3"></span>[33] Yuzhang Shang, Bingxin Xu, Weitai Kang, Mu Cai, Yuheng Li, Zehao Wen, Zhen Dong, Kurt Keutzer, Yong Jae Lee, and Yan Yan. Interpolating video-llms: Toward longersequence lmms in a training-free manner. *arXiv preprint arXiv:2409.12963*, 2024. [1,](#page-0-1) [3](#page-2-1)
- <span id="page-9-19"></span>[34] Yunhang Shen, Chaoyou Fu, Peixian Chen, Mengdan Zhang, Ke Li, Xing Sun, Yunsheng Wu, Shaohui Lin, and Rongrong Ji. Aligning and prompting everything all at once for universal visual perception. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13193–13203, 2024. [5,](#page-4-0) [8](#page-7-3)
- <span id="page-9-21"></span>[35] Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie Zhou, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. *arXiv preprint arXiv:2409.14485*, 2024. [7](#page-6-4)
- <span id="page-9-14"></span>[36] D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 11888–11898, 2023. [3](#page-2-1)
- <span id="page-9-25"></span>[37] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. *Journal of machine learning research*, 9 (11), 2008. [8](#page-7-3)
- <span id="page-9-20"></span>[38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024. [6](#page-5-1)
- <span id="page-9-4"></span>[39] Xidong Wang, Dingjie Song, Shunian Chen, Chen Zhang, and Benyou Wang. Longllava: Scaling multi-modal llms to 1000 images efficiently via hybrid architecture. *arXiv preprint arXiv:2409.02889*, 2024. [1](#page-0-1)

- <span id="page-9-15"></span>[40] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. *arXiv preprint arXiv:2403.10517*, 2024. [3](#page-2-1)
- <span id="page-9-11"></span>[41] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. *arXiv preprint arXiv:2407.15754*, 2024. [2,](#page-1-1) [5,](#page-4-0) [6,](#page-5-1) [7](#page-6-4)
- <span id="page-9-22"></span>[42] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. *arXiv preprint arXiv:2404.16994*, 2024. [7](#page-6-4)
- <span id="page-9-5"></span>[43] Yin Song and Chen Wu and Eden Duthie. awsprototyping/long-llava-qwen2-7b, 2024. [1,](#page-0-1) [3,](#page-2-1) [6,](#page-5-1) [7,](#page-6-4) [8,](#page-7-3) [2](#page-1-1)
- <span id="page-9-16"></span>[44] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering. *arXiv preprint arXiv:2312.17235*, 2023. [3](#page-2-1)
- <span id="page-9-0"></span>[45] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. *arXiv preprint arXiv:2306.02858*, 2023. [1](#page-0-1)
- <span id="page-9-13"></span>[46] Lu Zhang, Tiancheng Zhao, Heting Ying, Yibo Ma, and Kyusong Lee. Omagent: A multi-modal agent framework for complex video understanding with task divide-and-conquer. *arXiv preprint arXiv:2406.16620*, 2024. [2,](#page-1-1) [3](#page-2-1)
- <span id="page-9-6"></span>[47] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. *arXiv preprint arXiv:2406.16852*, 2024. [1,](#page-0-1) [3,](#page-2-1) [6](#page-5-1)
- <span id="page-9-1"></span>[48] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. [1,](#page-0-1) [3,](#page-2-1) [6](#page-5-1)
- <span id="page-9-8"></span>[49] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. [2,](#page-1-1) [6,](#page-5-1) [7,](#page-6-4) [8](#page-7-3)
- <span id="page-9-2"></span>[50] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang, Zhang Zhang, Liang Wang, and Rong Jin. Beyond llava-hd: Diving into high-resolution large multimodal models. *arXiv preprint arXiv:2406.08487*, 2024. [1](#page-0-1)
- <span id="page-9-23"></span>[51] Zijia Zhao, Haoyu Lu, Yuqi Huo, Yifan Du, Tongtian Yue, Longteng Guo, Bingning Wang, Weipeng Chen, and Jing Liu. Needle in a video haystack: A scalable synthetic framework for benchmarking video mllms. *arXiv preprint*, 2024. [7,](#page-6-4) [2,](#page-1-1) [3](#page-2-1)
- <span id="page-9-12"></span>[52] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. *arXiv preprint arXiv:2406.04264*, 2024. [2,](#page-1-1) [5,](#page-4-0) [6,](#page-5-1) [7](#page-6-4)
- <span id="page-9-7"></span>[53] Yongshuo Zong, Ismail Elezi, Yongxin Yang, Jiankang Deng, and Timothy Hospedales. Long-context vision large language models: Empirical insights and a baseline. In *Workshop on Long Context Foundation Models*, 2024. [1](#page-0-1)

# Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension

# Supplementary Material

## 6. Decouple Query

In the initial phase of the proposed Video-RAG, we employ a decouple prompt, denoted as **P**, to guide the LVLM in generating retrieval requests. In this section, we present one example of a prompt designed for multiple-choice questions, as illustrated in Figure 9.

#### <span id="page-10-3"></span>7. Sub-set of Video-MME

As outlined in the implementation details, we randomly sampled a subset of the Video-MME [6] dataset to evaluate a computationally resource-intensive, agent-based method with long-context LVLMs. Specifically, we selected 10% of the full dataset, comprising 30 short, 30 medium-length, and 30 long videos. Each video contains three multiple-choice questions. Importantly, we ensured that the performance ranking of the methods on the subset mirrored that of the full dataset. As shown in Tables 6 and 7, we evaluated four distinct 7B models Chat-Univi-v1.5 [10], LLaVA-NeXT-Video [48], LongVA [47], and Long-LLaVA [43] using a frame sampling rate of 16 for both the subset and the full set. Our results indicate that the performance rankings remained consistent across both evaluations.

<span id="page-10-1"></span>

| Method                | Short | Medium | Long | Overall |
|-----------------------|-------|--------|------|---------|
| Chat-Univi-v1.5 [10]  | 50.0  | 33.3   | 17.8 | 33.7    |
| LLaVA-NeXT-Video [48] | 54.4  | 33.3   | 23.3 | 37.0    |
| LongVA [47]           | 56.7  | 50.0   | 38.9 | 48.5    |
| Long-LLaVA [43]       | 58.9  | 52.2   | 40.0 | 50.4    |

Table 6. Performance of Video-MME sub-set.

<span id="page-10-2"></span>

| Method                | Short | Medium | Long | Overall |
|-----------------------|-------|--------|------|---------|
| Chat-Univi-v1.5 [10]  | 45.7  | 39.0   | 35.7 | 40.1    |
| LLaVA-NeXT-Video [48] | 51.1  | 41.8   | 36.8 | 43.2    |
| LongVA [47]           | 60.8  | 45.2   | 41.4 | 49.1    |
| Long-LLaVA [43]       | 59.3  | 49.3   | 44.4 | 51.0    |

Table 7. Performance of Video-MME full-set.

#### 8. Results on Video-MME Sub-Set

We examine Video-RAG against two representative methods in terms of inference time, GPU resource requirements, and overall performance. Given that GPT-based

<span id="page-10-0"></span>![](_page_10_Figure_12.jpeg)

Figure 7. The comparison of our Video-RAG with two common approaches. The size of the bubbles represents the total time consumed for completing inference on the Video-MME [6] sub-set.

Agent methods are resource-intensive, we randomly sampled a sub-set of the Video-MME [6] for evaluation, as described in Section 7. As demonstrated in Figure 7, VideoAgent [4], a typically GPT-based Agent method, requires significant time to process video and deliver suboptimal performance. Meanwhile, LongVA [47], a representative long-context LVLM, shows limited improvement from increasing the frame rate and even experiences performance degradation. Integrating our Video-RAG into the 16-frame LongVA results in substantial performance improvements while reducing GPU resource consumption. Specifically, with only increasing 8GB GPU memory compared to the base (16-frames LongVA), we achieve 11.5% overall performance improvement, while outperforming another longcontext LVLM Long-LLaVA-7B [43] in 16-frames setting by 9.6% with less GPU memory requirements and compatible total inference time. These results demonstrated that our Video-RAG is lightweight with lower computing overhead than the other typical methods.

#### 9. Details of Similarity Score Calculation

In the process of using the RAG system to retrieve auxiliary texts extracted from videos, we define a similarity threshold t to ensure the selection of relevant texts. Specifically, we employ FAISS-based [11] similarity to select OCR and ASR texts, while CLIP [29] similarity is used for keyframe selection. In our implementation, the similarity threshold t is set to 0.3. As for OCR and ASR selection, For any given list of the retrieve request  $\mathbf R$  and auxiliary texts  $\mathbf A$ , the Contriever [8] framework maps the text to a text embedding as:

$$\mathbf{E}_{a_i} = \text{Contriever}(\mathbf{A}_i), \quad i = 1, 2, \dots, n$$

$$\mathbf{E}_{r_i} = \text{Contriever}(\mathbf{R}_i), \quad i = 1, 2, \dots, n$$

The average embedding of the retrieve request is then computed as:

$$\mathbf{E}_r = \frac{1}{n} \sum_{i=1}^n \mathbf{E}_{r_i}$$

After that, the embedding of the request and the list of auxiliary texts is normalized:

$$\mathbf{E}_{a_i} = \frac{\mathbf{E}_{a_i}}{\parallel \mathbf{E}_{a_i} \parallel}, \quad \mathbf{E}_r = \frac{\mathbf{E}_r}{\parallel \mathbf{E}_r \parallel}$$

The similarity between the query embedding  $\mathbf{E}_r$  and the document vector  $\mathbf{E}_a$  is computed using the inner product, the FAISS library is employed to efficiently perform this search and return the indices of the auxiliary texts meeting the criterion:

$$S(\mathbf{E}_r, \mathbf{E}_{a_i}) = \mathbf{E}_r \cdot \mathbf{E}_{a_i} > t$$

As for object detection, we use CLIP to select the video keyframe. During this process, we first filter the object detection request  $\mathbf{R}_{det}$  to ensure they correspond to CLIP-sensitive physical entities, avoiding the inclusion of abstract concepts. Specifically, if it is a single word, direct part-of-speech filtering is applied; if it is a compound word, certain rules are followed to check for compliance, such as whether it is an adjective plus a noun, or a noun plus a noun. We use the Spacy library to achieve this. After this, we put the text "A picture of" before each object detection request.

Then, we extracting embedding from both the video frames F and the detection request  $R_{det}$ :

$$\begin{split} \mathbf{E}_{\mathbf{F}_j} &= \texttt{CLIP}(\mathbf{F}_j), \quad j = 1, 2, \dots, m \\ \mathbf{E}_{\mathbf{R}_i} &= \texttt{CLIP}(\mathbf{R}_{det_i}), \quad i = 1, 2, \dots, n \end{split}$$

The similarity between each video frame and the detection retrieve requests is computed using the dot product between the image and text feature embeddings. For each frame  $\mathbf{F}_j$ , and for each retrieve request  $\mathbf{E}_{\mathbf{R}_i}$ , the similarity score is given by:

$$S_{ij} = \mathbf{E}_{\mathbf{F}_i} \cdot \mathbf{E}_{\mathbf{R}_i}$$

where  $\cdot$  denotes the dot product. The final similarity score for each frame is the average similarity across all requests:

$$S_j = \frac{1}{n} \sum_{i=1}^n S_{ij}$$

This computes the mean similarity for each frame across all text descriptions, resulting in a similarity vector S =

 $[S_1, S_2, \ldots, S_m]$ . The similarity scores are adjusted by a scaling factor  $\alpha$ , which is computed based on the number of frames m and a base frame number b (which is set to 16 and 4.0, respectively) to adapted different video sampling rate of LVLMs:

$$\alpha = \beta \times \frac{m}{h}$$

where  $\beta$  is a predefined scaling parameter.

Next, the similarity scores are scaled and normalized to ensure that they sum to 1:

$$S_j^{\text{norm}} = \frac{\alpha \times S_j}{\sum_{k=1}^m S_k}$$

where  $S_j^{\text{norm}}$  represents the normalized similarity score for frame  $\mathbf{F}_j$ .

The final step is to select the keyframes based on the normalized similarity scores. A threshold t is applied to the normalized similarities, such that frames with similarity scores above the threshold are selected as keyframes:

Keyframe: 
$$\mathbf{F}_j$$
 if  $S_j^{\text{norm}} > t$ 

Thus, the set of selected keyframes is given by:

$$\mathbf{F}_{key} = \{ \mathbf{F}_j \mid S_j^{\text{norm}} > t, j = 1, 2, \dots, m \}$$

#### 10. More Ablation Studies

Effect of different components of Video-RAG. We evaluate the performance across sub-tasks within Video-MME [6], as shown in Figure 8. The results reveal that object detection auxiliary texts significantly enhance spatial perception and object counting, while OCR auxiliary texts specifically improve performance on text recognition tasks. Additionally, ASR auxiliary texts contribute to a general improvement in inference tasks, underscoring the critical role of audio transcription in video understanding. Given that audio transcription is considerably more time-consuming than character recognition or object detection, these texts should be selected based on the requirements of the application.

Besides studying the inference of different components of Video-RAG in the Video-MME [6] benchmark, we also experiment with a different type of video benchmark VN-Bench [51] with Long-LLaVA-7B [43]. VNBench is a synthetic benchmark designed to evaluate models' long-context abilities, covering tasks such as retrieval, ordering, and counting. VNBench randomly inserts stickers or text into the video that has nothing to do with the original content of the video, thus typically challenging the model's needle-inthe-haystack capability. As shown in Table 8, we find that applying DET and OCR as auxiliary texts can significantly improve the performance in retrieval, ordering, and counting tasks. However, the ASR component will decline the

<span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure 8. Performance on 12 sub-tasks in Video-MME [6] benchmark after applying different components in Long-LLaVA.

performance due to the subtitles are not ancillary to this particular task. These results demonstrated that our proposed distinct types of auxiliary texts can be selected according to the application needs to meet the requirements better.

<span id="page-12-1"></span>

| RAG          | DET | OCR          | ASR          | Ret  | Ord  | Cnt  | Overall      |
|--------------|-----|--------------|--------------|------|------|------|--------------|
|              |     |              |              | 65.1 | 25.6 | 24.2 | 38.3<br>39.7 |
| $\checkmark$ | ✓   |              |              | 66.9 | 28.4 | 23.8 | 39.7         |
| $\checkmark$ | ✓   | $\checkmark$ |              |      |      |      | 42.8         |
| $\checkmark$ | ✓   | $\checkmark$ | $\checkmark$ | 66.7 | 31.3 | 29.6 | 42.5         |

Table 8. Results on combinations of different auxiliary texts in VNBench [51] with 1-try setting when applying 7B Long-LLaVA [43] as LVLM under the 32-frames setting. **Ret**, **Ord**, and **Cnt** represent retrieval, ordering, and counting tasks, respectively.

## 11. More Qualitative Results

In this section, we show more results of LLaVA-Vdieo-7B when applying Video-RAG in different examples in Figure 10. The figure highlights several representative cases involving detailed video comprehension from Video-MME [6]. As illustrated, augmenting LLaVA-Video with external tools to process and retrieve auxiliary texts from videos significantly enhances its ability to reduce visual hallucinations, thereby enabling more accurate and confident responses to user queries.

## **Decouple Prompt of the Multiple-choice Question**

```
To answer the question step by step, list all the physical entities related to
the question you want to retrieve, you can provide your retrieve request to
assist you by the following JSON format:
{
 "ASR": Optional[str]. The subtitles of the video that may relavent to the
question you want to retrieve, in two sentences. If you no need for this
information, please return null.
 "DET": Optional[list]. (The output must include only physical entities, not
abstract concepts, less than five entities) All the physical entities and their
location related to the question you want to retrieve, not abstract concepts. If
you no need for this information, please return null.
 "TYPE": Optional[list]. (The output must be specified as null or a list
containing only one or more of the following strings: 'location', 'number',
'relation'. No other values are valid for this field) The information you want
to obtain about the detected objects. If you need the object location in the
video frame, output "location"; if you need the number of specific object,
output "number"; if you need the positional relationship between objects, output
"relation".
}
## Example 1:
Question: How many blue balloons are over the long table in the middle of the
room at the end of this video? A. 1. B. 2. C. 3. D. 4.
Your retrieve can be:
{
 "ASR": "The location and the color of balloons, the number of the blue
balloons.",
 "DET": ["blue ballons", "long table"],
 "TYPE": ["relation", "number"]
}
## Example 2:
Question: In the lower left corner of the video, what color is the woman wearing
on the right side of the man in black clothes? A. Blue. B. White. C. Red. D.
Yellow.
Your retrieve can be:
{
 "ASR": null,
 "DET": ["the man in black", "woman"],
 "TYPE": ["location", "relation"]
}
## Example 3:
Question: In which country is the comedy featured in the video recognized
worldwide? A. China. B. UK. C. Germany. D. United States.
Your retrieve can be:
{
 "ASR": "The country recognized worldwide for its comedy.",
 "DET": null,
 "TYPE": null
}
Note that you don't need to answer the question in this step, so you don't need
any infomation about the video of image. You only need to provide your retrieve
request (it's optional), and I will help you retrieve the infomation you want.
Please provide the json format.
```

Figure 9. Decouple prompt of the multiple-choice question for LVLMs.

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

 $Figure\ 10.\ Qualitative\ results\ of\ LLaVA-V dieo\ when\ applying\ Video-RAG.$