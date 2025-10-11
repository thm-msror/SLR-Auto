## MMMORRF:

# Multimodal Multilingual MOdularized Reciprocal Rank Fusion

Saron Samuel Stanford University Stanford, CA, USA sdsam@stanford.edu

Oluwaseun Eisape UC Berkeley Berkeley, CA, USA eisape@berkeley.edu

Andrew Yates Johns Hopkins University Baltimore, MD, USA andrew.yates@jhu.edu

Efsun Kayi Johns Hopkins University Laurel, MD, USA ekay1@jhu.edu

Dan DeGenaro Georgetown University Washington, DC, USA drd92@georgetown.edu

Tanner Spendlove BYU Provo, UT, USA ths29@byu.edu

Eugene Yang Johns Hopkins University Baltimore, MD, USA eugene.yang@jhu.edu

Matthew Wiesner Johns Hopkins University Baltimore, MD, USA mwiesne2@jhu.edu

Jimena Guallar-Blasco Johns Hopkins University Baltimore, MD, USA jgualla1@jhu.edu

Arun Reddy Johns Hopkins University Laurel, MD, USA areddy24@jhu.edu

Cameron Carpenter Johns Hopkins University Baltimore, MD, USA ccarpe18@jhu.edu

Kenton Murray Johns Hopkins University Baltimore, MD, USA kenton@jhu.edu

Kate Sanders Johns Hopkins University Baltimore, MD, USA ksande25@jhu.edu

Alexander Martin Johns Hopkins University Baltimore, MD, USA amart233@jhu.edu

David Etter Johns Hopkins University Baltimore, MD, USA detter2@jhu.edu

Reno Kriz Johns Hopkins University Baltimore, MD, USA rkriz1@jhu.edu

#### ABSTRACT

Videos inherently contain multiple modalities, including visual events, text overlays, sounds, and speech, all of which are important for retrieval. However, state-of-the-art multimodal language models like VAST and LanguageBind are built on vision-language models (VLMs), and thus overly prioritize visual signals. Retrieval benchmarks further reinforce this bias by focusing on visual queries and neglecting other modalities. We create a search system MM-MORRF that extracts text and features from both visual and audio modalities and integrates them with a novel modality-aware weighted reciprocal rank fusion. MMMORRF is both effective and efficient, demonstrating practicality in searching videos based on users' information needs instead of visual descriptive queries. We evaluate MMMORRF on MultiVENT 2.0 and TVR, two multimodal benchmarks designed for more targeted information needs, and find that it improves nDCG@20 by 81% over leading multimodal encoders and 37% over single-modality retrieval.

#### CCS CONCEPTS

• Information systems → Video search; Combination, fusion and federated search; Multilingual and cross-lingual retrieval.

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the owner/author(s).

SIGIR '25, July 13–18, 2025, Padua, Italy © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1592-1/2025/07. <https://doi.org/10.1145/3726302.3730157>

## KEYWORDS

Video Retrieval, Fusion, Multimodal, Multilingual

#### ACM Reference Format:

Saron Samuel, Dan DeGenaro, Jimena Guallar-Blasco, Kate Sanders, Oluwaseun Eisape, Tanner Spendlove, Arun Reddy, Alexander Martin, Andrew Yates, Eugene Yang, Cameron Carpenter, David Etter, Efsun Kayi, Matthew Wiesner, Kenton Murray, and Reno Kriz. 2025. MMMORRF: Multimodal Multilingual MOdularized Reciprocal Rank Fusion. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25), July 13–18, 2025, Padua, Italy. ACM, New York, NY, USA, [6](#page-5-0) pages.<https://doi.org/10.1145/3726302.3730157>

#### 1 INTRODUCTION

Online content has increasingly shifted toward video, many of which are multilingual and span multiple modalities, including visual events, spoken audio, embedded text, and non-speech sounds, such as music. As a result, users now require retrieval systems that handle both linguistic and modality diversity. Current commercial systems, such as YouTube search, rely heavily on non-visual metadata like titles, descriptions, and user engagement signals. While useful, these features are often incomplete or missing. To address this, we propose a multilingual video retrieval system that relies solely on visual, audio, and embedded textual content.

Prior research has shown strong results using vision-language models [\[7,](#page-4-0) [33,](#page-4-1) [43\]](#page-5-1) on academic benchmarks like VALOR-32k [\[6\]](#page-4-2) and MSR-VTT [\[36\]](#page-4-3). However, these datasets are small, English-centric, and rely on descriptive captions that do not reflect real-world search behaviors. Rather than relying on large vision and audio-language models, we introduce a practical and efficient fusion search engine, MMMORRF, that leverages information extracted using mature modality-specific technologies. Figure [1](#page-1-0) illustrates our approach,

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 1: Diagram of MMMORRF pipeline and fusion system for event-centric video retrieval. Components to the left of the blue dotted line are processed at indexing time; components to the right are processed at search time.

which achieves state-of-the-art results on MultiVENT 2.0, with more than 80% improvement over the best vision-language system.

Our contributions are three-fold: (1) we present, MMMORRF, a system that combines state-of-the-art components for multilingual, event-centric video retrieval; (2) we introduce a novel modalityaware fusion method for integrating multiple modalities; and (3) we provide a comprehensive comparison with vision-language models, demonstrating the effectiveness of our approach on a more realistic video retrieval task. A video demonstrating MMMORRF is available on YouTube[1](#page-1-1) , and the implementation is available on GitHub. [2](#page-1-2)

#### 2 BACKGROUND

Multilingual and cross-language retrieval. Multilingual information retrieval (MLIR) involves retrieving documents in multiple languages using queries in a different language [\[17,](#page-4-4) [25\]](#page-4-5). Crosslanguage information retrieval (CLIR) is a specialized case where the corpus is monolingual but queries are in another language [\[16\]](#page-4-6). Recent advancements focus on end-to-end retrieval systems that eliminate translation dependencies [\[20,](#page-4-7) [23\]](#page-4-8), improving scalability in multilingual settings [\[18\]](#page-4-9). Techniques such as language model pretraining [\[40\]](#page-4-10), data curation and translation [\[23,](#page-4-8) [24\]](#page-4-11), and knowledge distillation [\[20,](#page-4-7) [38,](#page-4-12) [39\]](#page-4-13) have enhanced the effectiveness of neural retrieval models [\[17,](#page-4-4) [38\]](#page-4-12).

Cross-modal retrieval. Cross-modal retrieval retrieves data in one modality using queries from another, such as retrieving videos using textual queries. Traditional methods treat each modality independently [\[32\]](#page-4-14), while modern deep learning enables joint multimodal representations, allowing direct comparisons across modalities. Models based on Contrastive Language–Image Pretraining (CLIP) [\[26\]](#page-4-15) have advanced this field and are further adapted for video retrieval [\[12,](#page-4-16) [14,](#page-4-17) [37\]](#page-4-18). However, most existing retrieval models remain limited to single-language and single-modality settings, whereas real-world online content spans multiple languages and integrates textual, auditory, and visual information, posing new challenges for retrieval systems.

Video retrieval. Video retrieval involves identifying the most relevant videos for a given query [\[4,](#page-4-19) [30\]](#page-4-20). Early systems relied on metadata such as titles and descriptions [\[31\]](#page-4-21), but deep learning has enabled retrieval using visual, audio, and embedded text features [\[2,](#page-4-22) [13\]](#page-4-23). Recent multimodal systems integrate these signals via fusion

transformers [\[6,](#page-4-2) [33\]](#page-4-1). However, most existing benchmarks focus on monolingual collections and simple queries [\[29\]](#page-4-24), making the task of true multilingual, multimodal retrieval particularly challenging to evaluate. Extracted video text has been explored in monolingual retrieval [\[35\]](#page-4-25), video question-answering [\[42\]](#page-4-26), and captioning [\[34\]](#page-4-27), but robust multilingual text extraction for video retrieval remains underexplored [\[28\]](#page-4-28).

#### 3 VIDEO RETRIEVAL PIPELINE

In this section, we present the components of our MMMORRF retrieval system for multilingual video retrieval.

Visual Content. To capture signals from the visual content, we develop a sub-pipeline that selects, encodes, and searches video frames, illustrated at the top of Figure [1.](#page-1-0) To balance efficiency and effectiveness, we uniformly sample 16 frames per video. Through pilot testing on MultiVent 2.0 training set [\[15\]](#page-4-29), we found that the benefit of sampling more frames saturates dramatically after 16. We also investigated key-frame detection tools, such as PySceneDetect [\[5\]](#page-4-30), but found no substantial difference between uniform sampling.

To bridge the modality gap between textual queries and video frames, we use SigLIP [\[41\]](#page-4-31), a variant of CLIP that replaces the standard contrastive learning softmax normalization with a sigmoid loss function. For efficient retrieval, frame embeddings are indexed using FAISS, a scalable vector search engine [\[10\]](#page-4-32). At search time, queries are encoded using SigLIP's text encoder and matched against the indexed frame embeddings by computing the maximum query-frame score (MaxFrame) for each video/query pair.

Video Optical Character Recognition (OCR). To accurately search embedded text in videos, such as overlay text from news broadcasts or street signs, we utilize an OCR sub-pipeline consisting of three steps: frame selection, text localization (or detection) and text recognition. For frame selection, we again select 16 uniformly sampled frames. Text localization is performed using the opensource PaddleOCR toolkit [\[21,](#page-4-33) [22\]](#page-4-34). The localized line images are then cropped and recognized using the multilingual OCR system described in Etter et al. [\[11\]](#page-4-35), which combines a vision transformer encoder trained with a Connectionist Temporal Classification (CTC) objective and an auto-regressive, character-level decoder.

Audio Transcription. We use OpenAI's Whisper Large-v2, a state-of-the-art multilingual automatic speech recognition (ASR) model, to transcribe video audio [\[27\]](#page-4-36). Whisper is a transformerbased sequence-to-sequence model pre-trained on 680,000 hours of

<span id="page-1-1"></span><sup>1</sup><https://youtu.be/7jLisTeSjKM>

<span id="page-1-2"></span><sup>2</sup><https://github.com/hltcoe/video-retrieval-demo>

multilingual audio for several general speech tasks, including language identification, time-aligned transcription, and time-aligned any-to-English translation. Its multilingual capabilities and ability to handle non-speech audio make it particularly well-suited for the diverse audio content found in video collections.

Extracted text Retrieval. To search the text extracted from visual and audio content, we employ multilingual ColBERT-X [18, 23], a multilingual cross-language variant of ColBERT with PLAID-X [39]<sup>3</sup> implementation. During indexing, each piece of extracted text, i.e., ASR and OCR outputs, is encoded into multiple vectors and preprocessed with K-means clustering to enable fast approximate search. At search time, PLAID-X encodes the queries and performs contextual late interaction between the query and text representations using product quantization. Both the extracted text and queries are encoded using a multilingual ColBERT-X [38] model to ensure robust processing across languages.<sup>4</sup> As shown in Figure 1, the OCR and ASR outputs are jointly fed into ColBERT-X at index time, referred to as Joint Index (JI) in Table 1.

Multimodal Retrieval Late Fusion. To leverage the strengths of both text-based and vision-based retrieval methods, we implement a modality-aware Weighted Reciprocal Rank Fusion (WRRF) approach. This fusion method combines the rankings from the text retrieval system (PLAID-X) and the vision retrieval system (SigLIP), assigning importance weights based on each video's characteristics. WRRF extends traditional Reciprocal Rank Fusion (RRF) [9]. Instead of summing the reciprocal ranks from both systems, WRRF computes a *video-dependent* linear combination of the two ranks to produce the final ranking score, defined as follows:

$$\mathrm{WRRF}(q,d) = \frac{\alpha_d}{r_{\mathrm{text}}(q,d) + k} + \frac{1 - \alpha_d}{r_{\mathrm{vision}}(q,d) + k}$$

where  $\alpha_d$  is the coefficient for the combination. Given the query q and video d, the ranks from the two systems are denoted as  $r_{text}(q,d)$  and  $r_{vision}(q,d)$ , respectively, with a smoothing hyperparameter k. In our implementation, we set k=0, diverging from the typical value of 60, to emphasize highly ranked results from either modality. This adaptive weighting strategy enables our system to capitalize on the strengths of each modality, enhancing retrieval performance over a diverse video collection.

We estimate the weighting for each video based on the retrieval score of a predefined query using the SigLIP model, which reflects the relative reliability of text-based and vision-based retrieval. We operationalize the weightings based on how closely a video resembles professional news content, hypothesizing that news-like videos yield more informative ASR and OCR text. Specifically, we employ the text query, news anchor live coverage broadcast microphone breaking news, and follow our video retrieval pipeline, which retrieves at the frame level and aggregates results using MaxFrame. These scores are independent of any specific query and simply assess the characteristics of the videos, which means scoring occurs during the indexing phase. We argue that this preprocessing step, while it may appear ad hoc, is analogous to tokenization decisions in text retrieval pipelines. Different retrieval collections, particularly those from varying genres, necessitate distinct tokenization and text-cleaning processes to achieve optimal retrieval results.

#### 4 EXPERIMENT SETUP

We evaluate our proposed video retrieval pipeline on two datasets: MultiVENT 2.0 and TVR. These datasets differ significantly in retrieval scenarios, providing a broad evaluation of our approach.

Multivent 2.0 [15] is a large-scale multilingual event-centric video retrieval dataset consisting of 218,000 videos and 3,906 manually curated queries covering diverse world events. The dataset is split into Multivent-Train (108,600 videos, 1,361 queries) and Multivent-Test (109,800 videos, 2,545 queries). Videos span professionally edited news reports to raw mobile footage, covering six primary languages: Arabic, Chinese, English, Korean, Russian, and Spanish. Unlike conventional video retrieval benchmarks, which often use captions as queries, Multivent 2.0 emphasizes real-world information-seeking behavior, making it more analogous to ad hoc text retrieval tasks like MS MARCO [3].

**TVR** (TV show Retrieval) is a multimodal video retrieval dataset consisting of 109K queries on 21.8K videos from six TV shows [19]. The dataset requires systems to understand both videos and subtitles. The dataset is split into 80% train, 10% val, 5% test-public and 5% test-private splits. We use the validation split in our analysis. Each query is written using either the subtitles, video, or both, and is linked to a specific temporal window, requiring precise moment localization within videos.

All extracted text from MultiVENT 2.0 is concatenated with their machine translation obtained through Google Translate before PLAID-X indexing, which is a common practice in CLIR [16, 17]. For TVR, since the videos are already in English, we do not obtain additional text before indexing. Following common video retrieval evaluation, we report Recall at 1 and 10 (R@1 and R@10) for both datasets, as well as normalized Discounted Cumulative Gain (nDCG@10) for MultiVENT 2.0 to assess ranked retrieval performance.

In addition to pipeline approaches, we report results using three end-to-end multimodal retrieval models that leverage pretrained vision-language models (VLMs): InternVideo2 [33], VAST [8], and Language-Bind [43]. These models have recently achieved state-of-the-art results on various vision tasks such as visual question answering and text-video retrieval, though often on small, unrealistic datasets. Most VLMs feature separate encoders for different modalities, while others attempt to use a unified encoder across modalities.

We use eight NVIDIA V100 GPUs for PLAID-X indexing and SigLIP encoding for faster parallel encoding. We ran FAISS without any compression and approximation. Serving queries can be done without a GPU for PLAID-X, but it is necessary for SigLIP. Therefore, at query serving time, we also use a V100 GPU for encoding the queries.

#### 5 RESULTS AND ANALYSIS

Table 1 summarizes the evaluation results of both the baselines and variants of MMMORRF on MultiVENT 2.0 and TVR. While InternVideo2, VAST, LanguageBind are strong vision retrieval models with some decent effectiveness on TVR, their performance on MultiVENT 2.0 is notably poor. InternVideo2's results are particularly surprising, given that it was trained on much of the MultiVENT 2.0 collection. This underperformance likely stems from its reliance on automatically generated captions, reinforcing

<span id="page-2-0"></span><sup>&</sup>lt;sup>3</sup>https://github.com/hltcoe/ColBERT-X

<span id="page-2-1"></span><sup>&</sup>lt;sup>4</sup>https://huggingface.co/hltcoe/plaidx-large-eng-tdist-mt5xxl-engeng

<span id="page-3-0"></span>Table 1: Retrieval effectiveness on **MultiVENT 2.0** and **TVR**. "JI" in the fusion method column indicates joint index. Only values in **MultiVENT 2.0** are tested for statistical significance, employing a paired t-test with 95% confidence under Bonferroni correction of 110 tests (all pairs of 11 systems and 2 metrics). Almost all differences are significant, with exceptions marked in superscript using a red strike line.

|    | Vision       |         | Audio    |         | Fusion  | MultiVENT 2.0 |         | TVR   |       |
|----|--------------|---------|----------|---------|---------|---------------|---------|-------|-------|
|    | Frames       | OCR     | Acoustic | ASR     | Method  | nDCG@10       | R@10    | R@1   | R@10  |
| a. | InternVideo2 | —       | —        | —       | Embed.  | 0.005         | 0.004   | 0.053 | 0.158 |
| b. | VAST         | —       | VAST     | —       | Embed.  | 0.035         | 0.042   | 0.100 | 0.268 |
| c. | LangBind     | —       | —        | —       | Embed.  | 0.324g        | 0.355eg | 0.087 | 0.258 |
| d. | SigLIP       |         |          |         | —       | 0.375eg       | 0.409fh | 0.156 | 0.375 |
| e. |              | BM25    |          |         | —       | 0.370d        | 0.367cg | 0.005 | 0.020 |
| f. |              | PLAID-X |          |         | —       | 0.415h        | 0.414dh | 0.004 | 0.015 |
| g. |              |         |          | BM25    | —       | 0.347cd       | 0.313ce | 0.200 | 0.381 |
| h. |              |         |          | PLAID-X | —       | 0.427f        | 0.425df | 0.192 | 0.400 |
| i. |              | PLAID-X |          | PLAID-X | JI      | 0.551j        | 0.556   | 0.192 | 0.400 |
| j. | SigLIP       | PLAID-X |          | PLAID-X | JI+RRF  | 0.562i        | 0.600   | 0.232 | 0.537 |
| k. | SigLIP       | PLAID-X |          | PLAID-X | JI+WRRF | 0.586         | 0.611   | 0.201 | 0.540 |

our argument that MultiVENT 2.0 presents a more realistic and challenging video retrieval task.

Pipelines that independently pass OCR and ASR outputs through mature text retrieval systems yield nDCG@10 scores of 0.415 and 0.427 on MultiVENT 2.0, respectively, both outperforming the two vision-language models. In TVR, where the videos are already in English, searching with extracted text still outperforms VAST. Here, PLAID-X does not provide much improvement, indicating that serving these queries can merely rely on lexical matches.

Jointly indexing both OCR and ASR outputs achieves an nDCG@10 of 0.551, outperforming all non-translated single-modality systems. Interestingly, combining the text sub-pipeline with the visual retrieval system (SigLIP) using ordinary reciprocal rank fusion (RRF) yields only a marginal improvement without statistical significance. However, applying the video-dependent weighted RRF (WRRF) we developed yields a statistically significant 4.2% improvement (from 0.562 to 0.586 nDCG@10). Notably, incorporating SigLIP into the pipeline results in a statistically significant 6.4% improvement (from 0.551 to 0.586) while introducing minimal additional latency, as discussed in Section [5.1.](#page-3-1)

While end-to-end models have been the focus of video retrieval research, our results demonstrate that a pipeline approach composed of mature components remains both more effective and practical for real-world applications—especially when evaluated on a large-scale, realistic benchmark like MultiVENT 2.0.

## <span id="page-3-1"></span>5.1 Encoding Throughput and Query Latency

We report both preprocessing (encoding and indexing) and search times in Table [2.](#page-3-2) Since the OCR and ASR processes can be run in parallel, the overall text retrieval pipeline takes approximately 37 seconds per video. SigLIP video frame encoding and indexing, the other major component, requires only 1 second per video and runs concurrently with the text pipeline. At search time, the combination of the FAISS index and PLAID-X retrieval system results in an overall per-query latency of approximately 300 ms, as both the

<span id="page-3-2"></span>Table 2: Preprocessing time (per video) and query latency (per query). The overall timing assumes parallelism when possible.

|                      | Vision OCR ASR |     |     | MT     | PLAID-X Overall |       |
|----------------------|----------------|-----|-----|--------|-----------------|-------|
| Preprocess / video   | 1s             | 30s | 36s | < 0.5s | < 0.5s          | 37s   |
| Search / query 200ms |                | –   | –   | –      | 300ms           | 300ms |

vision and text pipelines operate in parallel. This demonstrates that our system is both effective and efficient. A query latency of 300 ms is unlikely to be noticeable to users, making our system highly practical for real-world search applications [\[1\]](#page-4-41).

## 6 CONCLUSION

In this work, we present MMMORRF – a pipeline and fusion system for event-centric multimodal retrieval, which we argue is a more practical framing for video retrieval tasks. Unlike prior approaches that focus primarily on descriptive visual information, our system extracts specific text from videos via Automatic Speech Recognition and Optical Character Recognition and processes this information with a mature multilingual neural retrieval system. Evaluating on MultiVENT 2.0, a large-scale multilingual collection, we demonstrate that our pipeline and fusion system achieves a new state-of-the-art, outperforming the best vision-language model by 81%, and the top-performing single-modality system by 37%. Notably, our approach also achieves a query serving time of under 300 ms, making it both high-performing and efficient. To the best of our knowledge, this is the first scholarly work to integrate modern neural video encoding and neural retrieval models into a practical and deployable system for ad hoc video retrieval. We believe our approach lays the groundwork for building scalable, real-time video retrieval systems capable of addressing the complexities of multilingual, multimodal event-centric search tasks.

#### REFERENCES

- <span id="page-4-41"></span>[1] Ioannis Arapakis, Xiao Bai, and B Barla Cambazoglu. 2014. Impact of response latency on user behavior in web search. In Proceedings of the 37th international ACM SIGIR conference on Research & development in information retrieval. 103– 112.
- <span id="page-4-22"></span>[2] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. 2021. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision. 1728–1738.
- <span id="page-4-38"></span>[3] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. arXiv[:1611.09268](https://arxiv.org/abs/1611.09268) [cs.CL]<https://arxiv.org/abs/1611.09268>
- <span id="page-4-19"></span>[4] Meng Cao, Haoran Tang, Jinfa Huang, Peng Jin, Can Zhang, Ruyang Liu, Long Chen, Xiaodan Liang, Li Yuan, and Ge Li. 2024. RAP: Efficient Text-Video Retrieval with Sparse-and-Correlated Adapter. arXiv[:2405.19465](https://arxiv.org/abs/2405.19465) [cs.CV] [https://arxiv.org/](https://arxiv.org/abs/2405.19465) [abs/2405.19465](https://arxiv.org/abs/2405.19465)
- <span id="page-4-30"></span>[5] Brandon Castellano. [n.d.]. PySceneDetect. [https://github.com/Breakthrough/](https://github.com/Breakthrough/PySceneDetect) [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- <span id="page-4-2"></span>[6] Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, Jinhui Tang, and Jing Liu. 2023. VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset. arXiv[:2304.08345](https://arxiv.org/abs/2304.08345) [cs.LG]<https://arxiv.org/abs/2304.08345>
- <span id="page-4-0"></span>[7] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. 2023. Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset. Advances in Neural Information Processing Systems 36 (2023), 72842–72866.
- <span id="page-4-40"></span>[8] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. 2023. VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset. arXiv[:2305.18500](https://arxiv.org/abs/2305.18500) [cs.CV] [https://arxiv.org/abs/2305.](https://arxiv.org/abs/2305.18500) [18500](https://arxiv.org/abs/2305.18500)
- <span id="page-4-37"></span>[9] Gordon V. Cormack, Charles L A Clarke, and Stefan Buettcher. 2009. Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (Boston, MA, USA) (SIGIR '09). Association for Computing Machinery, New York, NY, USA, 758–759. [https://doi.org/10.](https://doi.org/10.1145/1571941.1572114) [1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)
- <span id="page-4-32"></span>[10] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024. The faiss library. arXiv preprint arXiv:2401.08281 (2024).
- <span id="page-4-35"></span>[11] David Etter, Cameron Carpenter, and Nolan King. 2023. A Hybrid Model for Multilingual OCR. In Document Analysis and Recognition - ICDAR 2023, Gernot A. Fink, Rajiv Jain, Koichi Kise, and Richard Zanibbi (Eds.). Springer Nature Switzerland, Cham, 467–483.
- <span id="page-4-16"></span>[12] Han Fang, Pengfei Xiong, Luhui Xu, and Yu Chen. 2021. CLIP2Video: Mastering Video-Text Retrieval via Image CLIP. arXiv[:2106.11097](https://arxiv.org/abs/2106.11097) [cs.CV] [https://arxiv.org/](https://arxiv.org/abs/2106.11097) [abs/2106.11097](https://arxiv.org/abs/2106.11097)
- <span id="page-4-23"></span>[13] Han Fang, Pengfei Xiong, Luhui Xu, and Yu Chen. 2021. Clip2video: Mastering video-text retrieval via image clip. arXiv preprint arXiv:2106.11097 (2021).
- <span id="page-4-17"></span>[14] Peng Jin, JinFa Huang, Fenglin Liu, Xian Wu, Shen Ge, Guoli Song, David A. Clifton, and Jie Chen. 2022. Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations. In Advances in Neural Information Processing Systems, Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (Eds.), Vol. 35. 30291–30306.
- <span id="page-4-29"></span>[15] Reno Kriz, Kate Sanders, David Etter, Kenton Murray, Cameron Carpenter, Kelly Van Ochten, Hannah Recknor, Jimena Guallar-Blasco, Alexander Martin, Ronald Colaianni, Nolan King, Eugene Yang, and Benjamin Van Durme. 2024. MultiVENT 2.0: A Massive Multilingual Benchmark for Event-Centric Video Retrieval. arXiv[:2410.11619](https://arxiv.org/abs/2410.11619) [cs.CV]<https://arxiv.org/abs/2410.11619>
- <span id="page-4-6"></span>[16] Dawn Lawrie, Sean MacAvaney, James Mayfield, Paul McNamee, Douglas W Oard, Luca Soldaini, and Eugene Yang. 2023. Overview of the TREC 2022 NeuCLIR Track. arXiv preprint arXiv:2304.12367 (2023).
- <span id="page-4-4"></span>[17] Dawn Lawrie, Sean MacAvaney, James Mayfield, Paul McNamee, Douglas W Oard, Luca Soldaini, and Eugene Yang. 2024. Overview of the TREC 2023 NeuCLIR Track. arXiv preprint arXiv:2404.08071 (2024).
- <span id="page-4-9"></span>[18] Dawn Lawrie, Eugene Yang, Douglas W Oard, and James Mayfield. 2023. Neural approaches to multilingual information retrieval. In European Conference on Information Retrieval. Springer, 521–536.
- <span id="page-4-39"></span>[19] Jie Lei, Licheng Yu, Tamara L Berg, and Mohit Bansal. 2020. TVR: A large-scale dataset for video-subtitle moment retrieval. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXI 16. Springer, 447–463.
- <span id="page-4-7"></span>[20] Yulong Li, Martin Franz, Md Arafat Sultan, Bhavani Iyer, Young-Suk Lee, and Avirup Sil. 2022. Learning Cross-Lingual IR from an English Retriever. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz (Eds.). Association for Computational Linguistics, Seattle, United States, 4428–4436.

- <https://doi.org/10.18653/v1/2022.naacl-main.329>
- <span id="page-4-33"></span>[21] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, and Xiang Bai. 2020. Real-time Scene Text Detection with Differentiable Binarization. In Proc. AAAI.
- <span id="page-4-34"></span>[22] Minghui Liao, Zhisheng Zou, Zhaoyi Wan, Cong Yao, and Xiang Bai. 2022. Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence (2022).
- <span id="page-4-8"></span>[23] Suraj Nair, Eugene Yang, Dawn Lawrie, Kevin Duh, Paul McNamee, Kenton Murray, James Mayfield, and Douglas W Oard. 2022. Transfer learning approaches for building cross-language dense retrieval models. In European Conference on Information Retrieval. Springer, 382–396.
- <span id="page-4-11"></span>[24] Suraj Nair, Eugene Yang, Dawn Lawrie, James Mayfield, and Douglas W Oard. 2023. BLADE: combining vocabulary pruning and intermediate pretraining for scaleable neural CLIR. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1219–1229.
- <span id="page-4-5"></span>[25] Douglas W Oard and Bonnie Jean Dorr. 1998. A survey of multilingual text retrieval. Citeseer.
- <span id="page-4-15"></span>[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. arXiv[:2103.00020](https://arxiv.org/abs/2103.00020) [cs.CV] [https://arxiv.org/](https://arxiv.org/abs/2103.00020) [abs/2103.00020](https://arxiv.org/abs/2103.00020)
- <span id="page-4-36"></span>[27] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. 2023. Robust Speech Recognition via Large-Scale Weak Supervision. In Proceedings of the 40th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 202), Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 28492–28518.<https://proceedings.mlr.press/v202/radford23a.html>
- <span id="page-4-28"></span>[28] Kate Sanders, David Etter, Reno Kriz, and Benjamin Van Durme. 2023. MultiVENT: Multilingual Videos of Events with Aligned Natural Text. arXiv[:2307.03153](https://arxiv.org/abs/2307.03153) [cs.IR] <https://arxiv.org/abs/2307.03153>
- <span id="page-4-24"></span>[29] Kate Sanders and Benjamin Van Durme. 2024. A Survey of Video Datasets for Grounded Event Understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7314–7327.
- <span id="page-4-20"></span>[30] Haoran Tang, Meng Cao, Jinfa Huang, Ruyang Liu, Peng Jin, Ge Li, and Xiaodan Liang. 2024. MUSE: Mamba is Efficient Multi-scale Learner for Text-video Retrieval. arXiv[:2408.10575](https://arxiv.org/abs/2408.10575) [cs.CV]<https://arxiv.org/abs/2408.10575>
- <span id="page-4-21"></span>[31] Howard D Wactlar and Michael G Christel. 2002. Digital video archives: Managing through metadata. Building a national strategy for digital preservation: Issues in digital media archiving 84 (2002), 99.
- <span id="page-4-14"></span>[32] Kaiye Wang, Qiyue Yin, Wei Wang, Shu Wu, and Liang Wang. 2016. A comprehensive survey on cross-modal retrieval. arXiv preprint arXiv:1607.06215 (2016).
- <span id="page-4-1"></span>[33] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. 2024. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377 (2024).
- <span id="page-4-27"></span>[34] Wenhao Wu, Haipeng Luo, Bo Fang, Jingdong Wang, and Wanli Ouyang. 2023. Cap4Video: What Can Auxiliary Captions Do for Text-Video Retrieval?. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 10704–10713.
- <span id="page-4-25"></span>[35] Weijia Wu, Yuzhong Zhao, Zhuang Li, Jiahong Li, Hong Zhou, Mike Zheng Shou, and Xiang Bai. 2025. A large cross-modal video retrieval dataset with reading comprehension. Pattern Recognition 157 (2025), 110818.
- <span id="page-4-3"></span>[36] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. 2016. MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 5288–5296. [https://doi.org/](https://doi.org/10.1109/CVPR.2016.571) [10.1109/CVPR.2016.571](https://doi.org/10.1109/CVPR.2016.571)
- <span id="page-4-18"></span>[37] Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, and Jiebo Luo. 2023. CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment. arXiv[:2209.06430](https://arxiv.org/abs/2209.06430) [cs.CV] [https://arxiv.](https://arxiv.org/abs/2209.06430) [org/abs/2209.06430](https://arxiv.org/abs/2209.06430)
- <span id="page-4-12"></span>[38] Eugene Yang, Dawn Lawrie, and James Mayfield. 2024. Distillation for Multilingual Information Retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2368–2373.
- <span id="page-4-13"></span>[39] Eugene Yang, Dawn Lawrie, James Mayfield, Douglas W Oard, and Scott Miller. 2024. Translate-Distill: Learning Cross-Language Dense Retrieval by Translation and Distillation. In European Conference on Information Retrieval. Springer, 50–65.
- <span id="page-4-10"></span>[40] Eugene Yang, Suraj Nair, Ramraj Chandradevan, Rebecca Iglesias-Flores, and Douglas W Oard. 2022. C3: Continued pretraining with contrastive weak supervision for cross language ad-hoc retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2507–2512.
- <span id="page-4-31"></span>[41] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid Loss for Language Image Pre-Training. arXiv[:2303.15343](https://arxiv.org/abs/2303.15343) [cs.CV] <https://arxiv.org/abs/2303.15343>
- <span id="page-4-26"></span>[42] Minyi Zhao, Bingjia Li, Jie Wang, Wanqing Li, Wenjing Zhou, Lan Zhang, Shijie Xuyang, Zhihang Yu, Xinkun Yu, Guangze Li, et al. 2022. Towards video text visual question answering: Benchmark and baseline. Advances in Neural Information

<span id="page-5-0"></span>Processing Systems 35 (2022), 35549–35562.

<span id="page-5-1"></span>[43] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, and Li Yuan. 2024. LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment. arXiv[:2310.01852](https://arxiv.org/abs/2310.01852) [cs.CV] <https://arxiv.org/abs/2310.01852>