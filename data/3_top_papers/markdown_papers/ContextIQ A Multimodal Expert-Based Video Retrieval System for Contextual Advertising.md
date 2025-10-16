# <span id="page-0-0"></span>ContextIQ: A Multimodal Expert-Based Video Retrieval System for Contextual Advertising

Ashutosh Chaubey<sup>2</sup>∗†, Anoubhav Agarwaal<sup>1</sup><sup>∗</sup> , Sartaki Sinha Roy<sup>1</sup><sup>∗</sup> , Aayush Agrawal<sup>1</sup><sup>∗</sup> , Susmita Ghose<sup>1</sup><sup>∗</sup> <sup>1</sup>Anoki Inc. <sup>2</sup>University of Southern California

achaubey@usc.edu {anoubhav,sartaki,aayush,susmita}@anoki.tv

# Abstract

*Contextual advertising serves ads that are aligned to the content that the user is viewing. The rapid growth of video content on social platforms and streaming services, along with privacy concerns, has increased the need for contextual advertising. Placing the right ad in the right context creates a seamless and pleasant ad viewing experience, resulting in higher audience engagement and, ultimately, better ad monetization. From a technology standpoint, effective contextual advertising requires a video retrieval system capable of understanding complex video content at a very granular level. Current text-to-video retrieval models based on joint multimodal training demand large datasets and computational resources, limiting their practicality and lacking the key functionalities required for ad ecosystem integration. We introduce ContextIQ, a multimodal expertbased video retrieval system designed specifically for contextual advertising. ContextIQ utilizes modality-specific experts—video, audio, transcript (captions), and metadata such as objects, actions, emotion, etc.—to create semantically rich video representations. We show that our system, without joint training, achieves better or comparable results to state-of-the-art models and commercial solutions on multiple text-to-video retrieval benchmarks. Our ablation studies highlight the benefits of leveraging multiple modalities for enhanced video retrieval accuracy instead of using a vision-language model alone. Furthermore, we show how video retrieval systems such as ContextIQ can be used for contextual advertising in an ad ecosystem while also addressing concerns related to brand safety and filtering inappropriate content.*

# 1. Introduction

Contextual advertising involves placing ads based on the content a consumer is viewing, improving user experience and leading to higher engagement and ad monetization [\[74\]](#page-10-0). This method mitigates the need for personal data in advertising, which raises notable legal and ethical concerns [\[20,](#page-8-0) [59\]](#page-10-1). Contextual advertising is already widely integrated into ads served on websites powered by platforms like Google AdSense [\[27\]](#page-8-1). Recent AI advancements have taken this a step further by enabling a deeper semantic understanding of multimedia content [\[34\]](#page-9-0), allowing for more precise ad placements beyond traditional targeting [\[41\]](#page-9-1). Building on these developments, this study expands the use of contextual advertising to video content, aiming to improve ad targeting on social platforms such as YouTube and streaming services like Free Ad-Supported Streaming Television (FAST) and Video on Demand(VoD).

Modern video streaming platforms, with their vast content libraries, require advanced methods for analyzing video content to retrieve the most suitable content for ad placements for contextual advertising. We propose that this task of identifying the most relevant video can be achieved with text prompts that align with an advertisement campaign and hence can be formulated as the text-to-video (T2V) retrieval problem in multimodal learning [\[45,](#page-9-2) [46,](#page-9-3) [51,](#page-9-4) [78\]](#page-10-2). Therefore, improvements in T2V retrieval models can enhance contextual advertising by better-aligning ads with the semantic context of video content.

With the exponential growth of video content, T2V retrieval has become more essential, driving progress in search algorithms [\[11,](#page-8-2)[21,](#page-8-3)[48\]](#page-9-5) and video representation techniques such as masked reconstruction [\[31,](#page-9-6) [62,](#page-10-3) [66\]](#page-10-4), multimodal contrastive learning [\[18,](#page-8-4) [72\]](#page-10-5), and next-token prediction from video data [\[43,](#page-9-7) [60\]](#page-10-6).

A recent trend in T2V retrieval is large-scale pretraining to learn a joint multimodal representation of a video [\[17,](#page-8-5) [18,](#page-8-4) [69\]](#page-10-7). While these joint models have demonstrated strong performance on public benchmarks, their reliance on massive multimodal datasets and substantial computational resources limits their practical use. In contrast, there are expert-based approaches that use specialized models to extract features from individual video modalities such as visuals, motion, speech, audio, OCR, etc. [\[45,](#page-9-2) [50\]](#page-9-8). Though this method is well-established in T2V retrieval, its prominence has declined with the rise of joint multimodal models.

<sup>†</sup>Work done while employed at Anoki Inc.

<sup>\*</sup>These authors contributed equally to this work

<span id="page-1-0"></span>However, expert models present unique benefits for contextual advertising, which we explore in this work.

This paper introduces ContextIQ, a multimodal expertbased video retrieval system for contextual advertising. ContextIQ leverages experts across multiple modalities—video, audio, captions (transcripts), and metadata such as objects, actions, emotions, etc.—to create semantically rich video representations (ref. Sec. [3](#page-2-0) & [4\)](#page-4-0). Its modular design makes it highly flexible to various brand targeting needs. For instance, for a beauty brand, we could integrate advanced object detection to spot niche beauty accessories. Moreover, the proposed system can selectively use only a relevant expert model, focusing on the key metadata to enable efficient targeting, making it well-suited for a fast ad-serving system. Furthermore, our system enhances interpretability by allowing individual experts to be analyzed for error tracing and model improvements, which is vital for explaining business outcomes in ad targeting. Beyond conventional video retrieval, we show how ContextIQ can be integrated seamlessly into the ad ecosystem, processing long-form streaming content and implementing brand safety filters to ensure ads are placed within contextually appropriate and safe content (ref. Sec. [6\)](#page-7-0).

We evaluate the effectiveness of our ContextIQ system across several video retrieval benchmarks, such as MSR-VTT [\[73\]](#page-10-8), Condensed Movies [\[9\]](#page-8-6), and a newly curated dataset (*Val-1*) of high-production content that we make public. We compare it to state-of-the-art joint multimodal models [\[54,](#page-9-9) [68,](#page-10-9) [77\]](#page-10-10) and commercial solutions like Google's multimodal embedding API [\[28\]](#page-8-7) and Twelvelabs Marengo [\[63\]](#page-10-11). The results show that our expert-based approach is better or comparable to these solutions (ref. Sec. [5.1\)](#page-5-0).

We also address key evaluation challenges, particularly the differences between video domains in existing public datasets—such as shorter, amateur-produced content—and the complex, high-production content typical of contextual advertising by validating the proposed technique on the curated dataset (*Val-1*) of movie contents and evaluating different approaches by manual validators. To further concretize our design choice of having multiple experts, we perform ablation studies on the impact of incorporating multiple modalities on an internal dataset (*Val-2*), demonstrating how our system effectively aggregates modalityspecific experts. By leveraging the complementarity of different modalities, ContextIQ achieves improvements in both performance and coverage (ref Sec. [5.2\)](#page-6-0). We release our datasets, retrieval queries and annotations publicly at [https://github.com/AnokiAI/ContextIQ-Paper.](https://github.com/AnokiAI/ContextIQ-Paper) In summary, our contributions are as follows,

(i) We present ContextIQ, a multimodal expert-based video retrieval system specifically designed for contextual advertising. With capabilities for long-form content processing, modularity for real-time ad serv-

- ing, and robust brand safety measures, it integrates seamlessly into the advertising ecosystem, effectively bridging the gap between video retrieval research and contextual video advertising.
- (ii) On existing benchmarks, we show that combining modality-specific experts can yield better or comparable results with state-of-the-art joint multimodal models without the need for joint training, large-scale multimodal datasets, or significant computational resources.
- (iii) We release a validation dataset (*Val-1*) of highproduction content for text-to-video retrieval for contextual advertising and show the superior performance of the proposed approach on that dataset.
- (iv) We perform ablation studies to show that additional modalities via expert-based models increase the coverage and accuracy of text-to-video retrieval for contextual advertising by showing results on an internal dataset of video contents.

# 2. Related Works

AI in advertising. Liu *et al*. [\[44\]](#page-9-10) employed text-based topic modeling to tag content, enabling buyers to make more informed ad bids. Wen *et al*. [\[61\]](#page-10-12) utilized decision trees to identify key attributes that enhance ad persuasiveness, facilitating the selection of suitable ads and insertion points within videos. Bushi *et al*. [\[12\]](#page-8-8) analyzed sentiment to prevent negative ad placements, while Vedula *et al*. [\[65\]](#page-10-13) predicted ad effectiveness by training separate neural network models on video, audio, and text modalities.

Text-to-Video Retrieval. Earlier approaches relied on convolutional neural networks (CNNs) and recurrent neural networks (RNNs) [\[30,](#page-9-11) [51\]](#page-9-4) to extract spatio-temporal features from video frames. However, the advent of Transformer-based architectures [\[24,](#page-8-9) [26,](#page-8-10) [57\]](#page-10-14) has led to a paradigm shift, with Transformers outperforming traditional CNNs and RNNs in many public benchmark datasets. Multimodal Pre-training and Joint Learning. The impact of large-scale pre-training, especially using models like CLIP [\[54\]](#page-9-9) that align images with text, has been groundbreaking in the video retrieval domain [\[23,](#page-8-11) [38,](#page-9-12) [46,](#page-9-3) [47,](#page-9-13) [75\]](#page-10-15). For instance, CLIP4Clip [\[46\]](#page-9-3) builds upon CLIP to enable end-to-end video-text retrieval via video-sentence contrastive learning. Similarly, CLAP [\[71\]](#page-10-16) aligns audio with text using contrastive learning. Incorporating additional modalities contrastively can significantly enhance the model's performance and robustness [\[18,](#page-8-4) [69\]](#page-10-7). Language-Bind contrastively binds language across multiple modalities within a common embedding space [\[77\]](#page-10-10). VALOR utilized separate encoders for video, audio, and text to develop a joint visual-audio-text representation [\[17\]](#page-8-5). We show

<span id="page-2-4"></span><span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Figure 1. Multimodal embedding generation pipeline for ContextIQ. Input videos are processed by the metadata extracting module, which uses expert models for extracting objects, actions, places, etc., and converts it into a metadata sentence  $(m_i)$ . Four modality encoders then encode the video frames, audio, caption (transcripts) and metadata in parallel and store the embeddings in a multimodal embeddings DB.

that compared to these jointly trained approaches, ContextIQ performs on par on video-retrieval benchmarks such as MSR-VTT [73] without any joint training.

**Multimodal Experts.** There have been several works in video retrieval that employ multiple experts to extract features from different video modalities, including scene visuals, motion, speech, audio, OCR, and facial features [19, 45, 50]. This approach introduces the challenge of effectively aggregating these expert-derived features. For example, Miech et al. [50] utilizes precomputed features from experts in text-to-video retrieval, where the overall similarity is calculated as a weighted sum of each expert's similarity score. Liu et al. [45] extends the mixture of experts model by employing a collaborative gating mechanism, which modulates each expert feature in relation to the others. [25] integrates a multimodal transformer to encode visual, audio, and speech features from different experts, with BERT handling the text query. In contrast to these techniques, we show how our expert-based retrieval system is specifically designed for contextual advertising by providing modularity for targeting flexibility and interpretability, brand safety filters, and ad ecosystem integration.

#### <span id="page-2-0"></span>3. Approach

For a video dataset  $V = \{v_1, ..., v_N\}$ , our goal is to develop a text-to-video retrieval system. Conceptually, we aim to learn a similarity function  $S(v_i, t)$  that assigns higher scores to videos that are more relevant to the given text query t.

We extract embeddings from multiple modalities, including raw video, audio, captions (speech transcripts), and combined visual (objects, places, actions, emotion) and textual (NER, emotion, profanity, hate speech) metadata, as shown in the Metadata Extraction Module in Fig. 1. Since the expert encoders for each modality are not jointly trained,

their embeddings reside in separate semantic spaces. These embeddings are then stored in a multimodal embeddings database. We propose a retrieval approach (Fig. 2) that compares, merges, and re-ranks these modality-specific embeddings to deliver the most contextually relevant videos.

## <span id="page-2-3"></span>3.1. Multimodal Embedding Generation

#### <span id="page-2-2"></span>3.1.1 Video

As illustrated in Fig. 1, we consider each video  $v_i$  to be composed of M frames  $v_i^j$ , which are encoded using an image-encoder  $f_{\theta_1}$  from a vision-text model [42]. This encoding captures intrinsic video features learned by the vision-text model.

We found that splitting the video into fixed non-overlapping time segments, each containing T frames, enhances both retrieval accuracy and localization. For each temporal segment  $C_i^k = \{v_i^z \mid z \in \{kT+1,\dots,(k+1)T\}\}$ , segment-level embeddings are obtained by applying an aggregation function  $\mathcal{A}_v$  to the frame-level features,

$$\mathbf{e}_{i}^{k} = \mathcal{A}_{v}(\{f_{\theta_{1}}(v_{i}^{z}) \mid z \in \{kT+1,\dots,(k+1)T\}\})$$
 (1)

The final set of video-level embeddings for the video  $v_i$  that we store is,

$$\mathcal{F}_i^v = \{ \mathbf{e}_i^1, \dots, \mathbf{e}_i^{(M/T)} \}$$
 (2)

Note that for a single video, we store (M/T) embeddings in the multimodal database corresponding to intrinsic video features. This ensures that for long-form video content, the features have high temporal resolution and do not average out, resulting in better retrieval.

## 3.1.2 **Audio**

To capture the non-verbal audio elements in a video, such as sound effects, music, and ambient noise, we utilize the

<span id="page-3-4"></span><span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2. Multimodal search pipeline for ContextIQ. Text query t is first encoded by different text encoders and the multimodal embedding DB is searched to find similar videos. The aggregation module combines the results obtained from different modalities, and the final results are obtained after applying brand-safety filters utilizing emotion, profanity, and hate speech.

audio encoder  $g_{\theta_2}$  from an audio-text model [71]. While speech and textual content are encoded through captions (ref. Sec. 3.1.3), this approach specifically targets the rich auditory features beyond spoken language.

The audio track  $a_i$  for a particular video  $v_i$  is divided temporally into equal-sized segments  $a_i^k$ , each of which is encoded using  $g_{\theta_2}$ . The final audio embedding  $\mathcal{F}_i^a$  for the video is obtained by applying an aggregation function  $\mathcal{A}_a$  to the features across all the segments:

$$\mathcal{F}_{i}^{a} = \mathcal{A}_{a} \left( g_{\theta_{2}}(a_{i}^{1}), g_{\theta_{2}}(a_{i}^{2}), \dots, g_{\theta_{2}}(a_{i}^{K}) \right)$$
 (3)

#### <span id="page-3-1"></span>3.1.3 Caption

To incorporate textual and speech information from a video, we encode the caption (transcript)  $c_i$  for a given video using a text encoder  $h_{\theta_3}$ , resulting in the caption feature embedding of the video:  $\mathcal{F}_i^c = h_{\theta_3}(c_i) \tag{4}$ 

#### <span id="page-3-3"></span>3.1.4 Metadata

Foundation vision models [42,54] excel at learning features for vision-specific tasks with minimal fine-tuning. To enhance the performance of our text-to-video retrieval system, we incorporate models specifically trained for vision and textual tasks, effectively augmenting foundational models (ref. Sec. 5.2).

After inference with the task-specific models, we store the extracted information into a metadata sentence  $m_i$ . This sentence captures objects, places, actions, emotions, named entities and flags any profanity, hate speech, or offensive language. This metadata is vital for contextual advertising. For example, object detection can allow fashion brands to target content featuring clothing or accessories, while place detection can benefit outdoor gear brands by identifying natural landscapes. Video action recognition can enable fitness brands to reach workout-related content, and named entity recognition might help travel agencies focus on videos mentioning tourist destinations.  $m_i$  provides a

unique and flexible method for integrating outputs from the different expert models.  $m_i$  follows the template, as illustrated in Fig. 1.

Finally, we encode  $m_i$  by the same text encoder model  $h_{\theta_3}$  from Sec. 3.1.3, to obtain the metadata feature embedding for the video:

$$\mathcal{F}_i^m = h_{\theta_3}(m_i) \tag{5}$$

Additionally, the emotion, profanity, and hate speech metadata are used as a filtering mechanism during the search process to ensure the retrieval of brand-safe videos. This capability extends beyond the conventional scope of video-text retrieval research, with its application further elaborated in Sec. 6 on contextual advertising.

#### <span id="page-3-2"></span>3.2. Multimodal Search

Our system employs multimodal embeddings from audio, video, text and metadata to enable efficient and precise content retrieval. As shown in the Fig. 2, a text query t is first encoded using the text encoders of the vision-text model  $(f_{\theta_1}^T)$ , audio-text model  $(g_{\theta_2}^T)$  and text-text model  $(h_{\theta_3})$  resulting in the embeddings  $\mathcal{F}_t^v$ ,  $\mathcal{F}_t^a$  and  $\mathcal{F}_t^T$  respectively.

The text embeddings are then compared against their corresponding embedding databases using cosine similarity. Specifically,  $\mathcal{F}^a_t$  is compared with audio embeddings  $\{\mathcal{F}^a_i\}$ , producing similarity scores  $\{\mathcal{S}^a(v_i,t)\}$ ;  $\mathcal{F}^v_t$  is compared with visual embeddings  $\{\mathcal{F}^v_i\}$ , yielding similarity scores  $\{\mathcal{S}^v(v_i,t)\}$ ; and  $\mathcal{F}^T_t$  is compared with metadata embeddings  $\{\mathcal{F}^m_i\}$  and caption embeddings  $\{\mathcal{F}^c_i\}$ , generating similarity scores  $\{\mathcal{S}^m(v_i,t)\}$  and  $\{\mathcal{S}^c(v_i,t)\}$ , respectively. Note that  $\{\mathcal{S}^v(v_i,t)\}$  is the max-similarity score of t with the segments  $(\{C^1_i,\ldots,C^{(M/T)}_i)$  of video  $v_i$ .

These similarity scores  $\{(S^c, S^m, S^a, S^v)(v_i, t)\}$  are subsequently aggregated into a combined score  $S(v_i, t)$  using an aggregation module  $A_S$  (ref. Fig. 2) which involves normalization, thresholding, and weighted merging,

<span id="page-4-3"></span>• **Normalization.** Scores are standardized within their respective modality space and weighted to produce normalized score:

where  $\mu^k = \frac{1}{n} \sum_i^n S^k(v_i, t) - \mu^k$  (6) where  $\mu^k = \frac{1}{n} \sum_i^n S^k(v_i, t)$  is the mean,  $\sigma^k = \sqrt{\frac{1}{n} \sum_i^n (S^k(v_i, t) - \mu^k)^2}$  is the standard deviation and  $\lambda^k$  is the weight of modality k (metadata, caption, video or audio).

- Thresholding. Scores are thresholded to obtain dictionaries for each modality  $\mathcal{X}^k = \{(i : \mathcal{N}^k(v_i,t)) \mid \mathcal{S}^k(v_i,t) > \alpha^k\}$  where  $\alpha^k$  is the modality specific threshold.
- Merging. The thresholded dictionaries for each modality are merged using a max-aggregation approach, with the final scores  $S(v_i,t)$  determined by the maximum value for each key.

The thresholds  $\alpha^k$  and weights  $\lambda^k$  can be tuned for optimal performance on a representative set of queries. The resulting video contents are finally filtered through the brand safety mechanism (ref. Sec. 6).

## <span id="page-4-0"></span>4. Implementation Details

#### 4.1. Multimodal encoders

We use PyTorch [8] for all our model implementations, and we use four NVIDIA RTX 4090 GPUs to run all our experiments. The text encoder  $h_{\theta_3}$  is a pre-trained MPNet model [58] fine-tuned on a set of 1 billion text-text pairs as described by Reimers et al [55]. The vision-text model  $f_{\theta_1}$ is a pre-trained BLIP2 Qformer [42], and we use the implementation provided by LAVIS [40]. We split the video into equal segments of 15 seconds each and sample one out of ten frames for embedding generation (ref. Sec. 3.1.1). The audio encoder  $g_{\theta_2}$  is the CLAP [71] model as implemented in [2]. Since CLAP is trained on 5-second audio segments, we divide the audio into 5-second chunks  $(a_i^k)$ for inference. The aggregation functions  $\mathcal{A}_a$  and  $\mathcal{A}_v$  used during audio and video embedding generation (Sec. 3.1) are temporal mean pooling functions, shown to be effective over other temporal aggregation techniques [10,46].

#### <span id="page-4-2"></span>4.2. Metadata Extraction

We use the YOLOv5 model [64], trained on the Objects 365 dataset [56], to detect objects within videos with an IOU threshold of 0.45 and a confidence threshold of 0.35. An object is considered present in the video only if it appears in at least 20% of the frames, to filter false positives.

For place detection, we finetune a ResNet50 model [32] on the Places365 dataset [76]. Only frames where object detection for the *Person* class covers less than 10% of the area are used to ensure a clear background. Top predictions

from these frames with softmax scores above 0.3 are considered, and the most frequent place prediction is tagged as the video's location. Since video segments are short, we assume a single location per segment. For both place and object detection, predictions are sampled from every 10th frame.

We use a fine-tuned video masked autoencoder model (VideoMAE2) [67] on the Kinetics 710 dataset [14, 15, 35] for video action recognition. The Kinetics dataset comprises shorter, simpler YouTube clips featuring single actions, whereas our video retrieval algorithm is applied to videos with longer, more complex scenes involving multiple actions. To bridge this gap, we reduced the Kinetics 710 classes to 185 by eliminating less relevant or overly specific classes for advertising, merging correlated classes, and discarding those with low Kinetics validation accuracy. We also refined the majority voting method by incorporating prediction probability scores, improving the handling of multiple actions in a single clip. More implementation details are present in the supplementary material, and we share the list of reduced classes on our github repository https://github.com/AnokiAI/ContextIQ-Paper.

For named entity recognition, we utilize the text-based RoBERTa model, en\_core\_web\_trf from Spacy [33] to extract the named entities from the captions of each video scene. For profanity detection, we used the *alt-profanity-check* [1] python package on the video transcript (caption). Additionally, we use a predefined list of words given in [3] to further filter out profane videos.

For hate speech detection, we use a weighted ensemble of two models. First, we use a pre-trained LLAMA 3 8B Instruct model [7] with a temperature of 0.6, leveraging advanced prompting strategies such as JSON-parseable responses and chain-of-thought reasoning to flag content as hateful. Secondly, we use a pre-trained BERT classifier [36] trained on the HateXplain dataset [49] to categorize content into three classes: hate speech, offensive, and normal. For text emotion recognition, first, we use a pretrained Emoberta-Large model [37] from huggingface [6] which is trained on the MELD [53] and IEMOCAP [13] datasets. Furthermore, we also leverage the computed video and audio embeddings from Sec. 3.1 for tagging emotions by associating text concepts with different emotions. For example, we tagged the emotion joy with text queries like people smiling and people dancing, and assigned the emotion joy to all videos retrieved through the video (vision) modality using these queries. More implementation details of emotion, profanity, and hate speech detection are present in the supplementary material.

## <span id="page-4-1"></span>4.3. Validation datasets and Metrics

**Validation datasets.** As mentioned before, we show the efficacy of the proposed approach in comparison to state-

<span id="page-5-4"></span><span id="page-5-1"></span>

| Model             | P@1  | P@5         | R@5         | MAP@5       |
|-------------------|------|-------------|-------------|-------------|
| Vertex API [28]   | 81.9 | 57.0        | 93.2        | 83.1        |
| LanguageBind [77] | 85.5 | 66.6        | 97.7        | 86.6        |
| ContextIQ (Ours)  | 81.7 | <u>59.1</u> | <u>93.7</u> | <u>83.2</u> |

<span id="page-5-2"></span>Table 1. Performance comparison on MSR-VTT for ContextIQ, Google Vertex [28] and LanguageBind [77].

| Model             | P@1  | P@5         | R@5 | MAP@5       |
|-------------------|------|-------------|-----|-------------|
| TwelveLabs [63]   | 96.6 | 90.3        | 100 | 95.6        |
| LanguageBind [77] | 89.7 | 83.5        | 100 | 91.5        |
| ContextIQ (Ours)  | 96.6 | <u>88.3</u> | 100 | <u>94.4</u> |

Table 2. Performance comparison on Condensed Movies [9] for ContextIQ, TwelveLabs [63] and LanguageBind [77].

of-the-art video retrieval methods on two public datasets, MSR-VTT [73] and Condensed Movies [9]. We use the 1kA subset of the MSR-VTT test set for evaluation, which consists of 1k videos and 20 text descriptions for each of them. During our analysis, we observed duplicate text descriptions both within individual video clips and across different clips. Hence to assess performance, we randomly sample one caption per video clip.

Since the MSR-VTT dataset includes a variety of video rather than entertainment-focused content, we also utilize the Condensed Movies dataset [9], which consists of scene clips from 3K+ movies. For evaluation, we randomly sample 600 scene clips and extract the first minute of each. Based on the movie and scene descriptions, we use Chat-GPT [52] to generate a set of 29 text queries, focusing on concepts such as objects, locations, emotion, and other contextual elements to search across the 600 video clips. Because the condensed movies dataset is not tagged with the set of queries we obtained, we manually validated the results on this dataset. (ref. Sec. 5.1).

Furthermore, to show that our system performs well for contextual advertisement targeting, we manually collected a set of 500 movie clips of different genres from YouTube corresponding to different potential advertisement categories. We call this dataset *Val-1*. The selected clips represent one or more of the following 'concepts' - *burger*, *concert*, *cooking*, *cowboys and western*, *dog*, *space shuttle*, *sports*, and *army*. Each of the collected movie clips is then annotated with one or more of these concepts by at least 2 annotators, and we take the union of annotated concepts as ground truth.

We release all the details about the datasets, including text queries used for validation and annotations on GitHub - https://github.com/AnokiAI/ContextIQ-Paper.

**Metrics.** For all our experiments, we report one or more of the following metrics (i) Precision@K (P@K), which is the proportion of retrieved videos marked as correct out of the top K retrieved videos, averaged across all text queries, (ii) Recall@K (R@K), which is the average number of queries for which at least one of the top K retrieved results is marked as correct, and (iii) Mean Average

<span id="page-5-3"></span>![](_page_5_Picture_9.jpeg)

Figure 3. Validation tool built with Streamlit [5]. Note that the different methods are kept anonymous to remove any bias.

Precision@K(MAP@K): The mean of the average precision scores from 1 to  $K(\{1,\ldots,K\})$ , computed across all queries.

#### 5. Results

#### <span id="page-5-0"></span>5.1. Zero-shot text-to-video retrieval

We evaluate the performance of ContextIQ for text-tovideo retrieval using the validation datasets mentioned in Sec. 4.3. Tab. 1 shows the performance of the proposed approach on the MSR-VTT dataset [73] compared to a stateof-the-art jointly trained multimodal model (LanguageBind [77]) and a popular industry solution for video retrieval (Google's Vertex API [28]). For LanguageBind we use the huggingface implementation - LanguageBind\_Video\_FT [4]. Although ContextIQ is not jointly trained on multiple modalities, it performs slightly better than Google's Vertex. We hypothesize that LanguageBind's superior performance on MSR-VTT can be attributed to its joint training on the large-scale 10M VIDAL dataset, which includes a diverse range of videos similar to the general content found in MSR-VTT, rather than being focused on entertainmentspecific content. Additionally, LanguageBind incorporates modalities such as depth, which are absent in ContextIQ.

As shown in Tab. 2, for the Condensed Movies dataset [9], we compare the results of our proposed technique with *TwelveLabs*, utilizing their *Marengo* API [63], a jointly trained multimodal model for text-to-video retrieval. For all the approaches listed in Tab. 2, we first retrieve videos for each of the text queries generated by ChatGPT (ref. Sec. 4.3) and then ask three manual validators to validate the top 5 results using the validation tool built using *Streamlit* [5] as shown in Fig. 3. We use a voting-based system among the annotators to compile the results of our validation. We can see that ContextIQ performs better than LanguageBind on all the metrics, while it performs comparable to Twelve-Labs. These findings further show that ContextIQ, a mix-

<span id="page-6-4"></span><span id="page-6-1"></span>

| Model             | Jointly<br>trained | Modalities                         | P@5  | P@10 | P@15 | P@20 | P@25 | P@30 | P@35 | P@40 | P@45 | P@50 |
|-------------------|--------------------|------------------------------------|------|------|------|------|------|------|------|------|------|------|
| CLIP (Large) [46] | Х                  | V, L                               | 100  | 100  | 99.2 | 98.8 | 98.5 | 98.8 | 96.8 | 95   | 93.3 | 90.3 |
| LanguageBind [77] | ✓                  | $L$ , $V$ , $A$ , $\delta$ , $I_r$ | 100  | 98.8 | 98.3 | 98.1 | 98.5 | 98.8 | 97.1 | 95.9 | 94.7 | 92   |
| One-Peace [68]    | ✓                  | L, V, A                            | 92.5 | 91.3 | 92.5 | 94.4 | 93.5 | 93.3 | 93.2 | 92.5 | 90.8 | 90.3 |
| ContextIQ (Ours)  | Х                  | L, V, A                            | 100  | 100  | 100  | 99.4 | 99   | 98.8 | 98.2 | 98.1 | 97.2 | 97.3 |

Table 3. Performance comparison on the curated dataset (Val-1) for ContextIQ, LanguageBind [77], One-Peace [68] and CLIP-Large [54]. V: vision, L: language, A: audio,  $\delta$ : depth,  $I_r$ : infrared.

ture of expert-based models can perform comparable to or even better than the jointly trained multimodal models.

Tab. 3 shows the performance of the proposed approach as compared to different approaches on our curated dataset (*Val-1* as described in Sec. 4.3). We use the large variant of CLIP [54] for our comparison. Note that One-Peace [68] and LanguageBind [77] are multimodal models that are trained jointly, where embeddings from all modalities reside in the same space. ContextIQ performs significantly better than the baselines, especially when we check the precision at higher k values. It is also important to highlight that CLIP, which is just a vision-language model, performs comparably to the multimodal approaches LanguageBind and One-Peace, highlighting that for most of the queries, only visual understanding results in good retrieval results.

# <span id="page-6-0"></span>5.2. Ablation: Impact of additional experts and modalities

We saw in the previous paragraph that only a vision-language model achieves comparable results on the task of video retrieval for advertising on our advertising-focused curated dataset (*Val-1*). In this section, we perform an ablation study to see the efficacy of different modalities when combined with the vision-text model encoder.

Since we want to simulate the effect of using our system in a large database of multimedia content, we utilize an internal dataset (*Val-2*) comprising over 2,000 long-form videos (movies, TV, and OTT contents) processed through our ContextIQ system (ref. Sec. 6) to generate over 100,000 scenes (videos), averaging 30 seconds in duration. We curate retrieval query sets for each additional modality, focusing on queries that highlight their individual strengths and are relevant to ad targeting. Details about the dataset are included in our github repository.

We compute the audio, video, caption and metadata embeddings for the entire dataset and store them separately. We then compare the performance of vision-only (video) embeddings with vision combined with different modalities as shown in Tab. 4. Since we do not have labeled ground truth for this internal dataset as well, we employ a similar manual validation technique which was used for validating Condensed Movies (ref. Tab. 2). For each query, we search and retrieve the top 30 videos from the dataset. Vision-only results are based on similarity scores between the text query and vision embeddings, while vision + additional modality

<span id="page-6-2"></span>

| Query set                      | Modalities | P@5  | P@10 | P@15 | P@20 | P@25 | P@30 |
|--------------------------------|------------|------|------|------|------|------|------|
| Metadata set                   | V          | 85.7 | 84.3 | 80.5 | 79.5 | 77.6 | 76.2 |
| $(\Delta_{\rm avg}=4.08)$      | V + M      | 87.9 | 86.4 | 85.0 | 83.6 | 83.0 | 82.4 |
| Caption set                    | V          | 84.2 | 79.5 | 75.4 | 76.6 | 75.4 | 74.6 |
| $(\Delta_{\rm avg}=5.42)$      | V + L      | 84.2 | 82.1 | 83.5 | 83.4 | 82.5 | 82.5 |
| Audio set                      | V          | 85.7 | 82.9 | 79.0 | 83.6 | 83.4 | 84.8 |
| $(\Delta_{\text{avg}} = 5.67)$ | V + A      | 88.6 | 87.1 | 87.6 | 89.3 | 90.3 | 90.5 |

<span id="page-6-3"></span>Table 4. Performance gain on Adding Modalities to Vision-Only System. V: vision, L: language (captions), A: audio, M: metadata

| Modality     | Overl | ision M | Modality |      |      |      |
|--------------|-------|---------|----------|------|------|------|
|              | @5    | @10     | @15      | @20  | @25  | @30  |
| Metadata (M) | 2.96  | 5.93    | 6.17     | 6.48 | 7.70 | 7.78 |
| Caption (L)  | 1.11  | 0.56    | 0.74     | 1.39 | 2.22 | 2.22 |
| Audio (A)    | 0.00  | 0.00    | 0.00     | 0.00 | 0.00 | 0.95 |

Table 5. Overlap percentage in Top-K results for vision only and vision + additional modality.

results combine scores from both modalities using the aggregation module (ref. Sec. 3.2). The top 30 videos from both methods are then annotated for correctness by 3 annotators.

Tab. 4 shows that adding an additional modality consistently improves precision across all K values compared to using a vision-only model. The precision gap between vision and vision+modality widens as K increases.  $\Delta_{avg}$  represents the average difference in precision between vision and vision+additional modality:

$$\Delta_{\text{avg}} = \frac{1}{|\mathcal{K}|} \sum_{K \in \mathcal{K}} |P_{V,K} - P_{V+X,K}| \tag{7}$$

Where, K is the set of K values  $\{5, 10, \ldots, 30\}$  and  $P_{V,K}$  is the precision for vision-only variant at K, and  $P_{V+X,K}$  is the precision for vision+additional modality.

We observe that the average precision delta is highest for audio, followed by caption and then metadata. This is due to the fact that the vision modality doesn't capture raw audio or captions (transcripts) but is better at representing metadata elements like objects, actions, places, etc.

Tab. 5 further shows that without our aggregation module, the scenes retrieved by each modality independently for the same query differ significantly from those retrieved by the vision-only variant, resulting in very low overlap. This confirms that each modality captures distinct aspects of the video. The overlap with vision follows the order: audio < caption < metadata, which aligns with expectations. These findings show that utilizing the complementarity of various modalities improves both performance and coverage.

<span id="page-7-2"></span><span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

Figure 4. End-to-End ContextIQ video retrieval system for contextual advertising (ref. Sec. 6)

## <span id="page-7-0"></span>6. Contextual Advertising

Contextual Advertising targets ads based on the content a user is viewing, improving the experience, boosting engagement, and increasing conversion rates, all without using personal information, making it privacy-friendly. Fig. 4 shows how ContextIQ is integrated into the Connected TV advertising ecosystem that enables advertisers to perform contextual advertising.

**Processing Long-Form content.** Long-form content, such as movies and shows, is processed by ContextIQ by breaking them into shorter videos using scene detection. We use PySceneDetect [16] for scene detection with default parameters. Each scene is subsequently processed through ContextIQ's multimodal embedding generation module (ref. Sec. 3.1) to generate the reference multimodal scene embeddings.

Integration into Ad Serving system. Depending on the brand campaign and the advertisements to be served, an advertiser defines a set of relevant text queries. For example, a pet food brand might have queries such as dogs, cats, pet food, etc. Using these brand-specific text queries and the multimodal embeddings, ContextIQ's multimodal search (ref. Sec. 3.2) identifies scenes where creatives can be contextually served. Additionally, these scenes can be passed through brand safety filters to ensure that brands don't get associated with sensitive/profane scenes. The selected scenes are stored in the scene-to-context lookup DB. When a viewer is watching the TV, the ad gateway looks up the scene to context lookup DB to find out the relevant context for showing advertisements; the ad gateway serves the brand's ad as shown in Fig. 4. ContextIQ can easily be extended to retrieve relevant scenes for showing an ad based on image, video or even audios. For example, an advertiser might directly use their brand advertisement as a query for video retrieval and get the relevant search results (more details in the Supplementary material).

**Brand Safety.** Ensuring brand safety is crucial in video retrieval systems for contextual advertising. Advertisers are increasingly vigilant about the environments in which their brands appear, as association with inappropriate content, such as offensive language, hate speech, negative emotions, adult material, or references to crime and terrorism, can cause significant reputational damage and erode consumer

trust. To address these concerns, our ContextIQ system integrates a safety mechanism comprising two key filters: (i) **Emotion Recognition** filter, which evaluates the emotional quality of content, ensuring alignment with the brand's messaging. (ii) **Hate Speech and Profanity Detection** filter, which blocks content containing hate speech, explicit language, or other inappropriate communication, preventing ads from appearing alongside harmful content. Since we already extract emotional and profanity information during metadata extraction (ref. Sec. 3.1.4 & 4.2), we use the same information during brand safety filtering with no additional compute.

Modularity. ContextIQ, with its diverse set of expert models, offers flexibility by allowing the use of a specific subset of models tailored to particular use cases. For example, the textual modality can be used for real-time content ingestion to serve ads during live news segments, filtering out violent content that many brands prefer to avoid. Additionally, each expert model can be fine-tuned to meet specific brand requirements. For instance, place and object detection models can be fine-tuned accordingly to support a casino brand looking to detect both a casino location and a roulette wheel within the content.

#### 7. Conclusion

This paper introduces ContextIQ, an end-to-end video retrieval system designed for contextual advertising. By leveraging multimodal experts across video, audio, captions, and metadata, ContextIQ effectively aggregates these diverse modalities to create semantically rich video representations. We demonstrate strong performance on multiple video retrieval benchmarks, achieving results better or comparable to jointly trained multimodal models without requiring extensive multimodal datasets and computational resources. Our ablation study shows the advantage of incorporating multiple modalities over a vision-only baseline. We further examine how ContextIQ extends beyond the conventional video retrieval task by integrating seamlessly into the ad ecosystem, processing streamed long-form content, offering modularity for efficient real-time ad serving, and implementing brand safety filters to ensure ads are placed within contextually appropriate and safe content.

# References

- <span id="page-8-19"></span>[1] GitHub - dimitrismistriotis/alt-profanity-check: A fast, robust library to check for offensive language in strings, dropdown replacement of "profanity-check". — github.com. [https : / / github . com / dimitrismistriotis /](https://github.com/dimitrismistriotis/alt-profanity-check) [alt-profanity-check](https://github.com/dimitrismistriotis/alt-profanity-check). [Accessed 08-09-2024]. [5](#page-4-3)
- <span id="page-8-15"></span>[2] GitHub - LAION-AI/CLAP: Contrastive Language-Audio Pretraining — github.com. [https://github.com/](https://github.com/LAION-AI/CLAP) [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP). [Accessed 08-09-2024]. [5](#page-4-3)
- <span id="page-8-20"></span>[3] GitHub - surge-ai/profanity: The world's largest profanity list. — github.com. [https://github.com/surge](https://github.com/surge-ai/profanity)[ai/profanity](https://github.com/surge-ai/profanity). [Accessed 08-09-2024]. [5](#page-4-3)
- <span id="page-8-25"></span>[4] LanguageBind/LanguageBind Video FT · Hugging Face — huggingface.co. [https : / / huggingface . co /](https://huggingface.co/LanguageBind/LanguageBind_Video_FT) [LanguageBind/LanguageBind\\_Video\\_FT](https://huggingface.co/LanguageBind/LanguageBind_Video_FT). [Accessed 08-09-2024]. [6](#page-5-4)
- <span id="page-8-24"></span>[5] Streamlit • A faster way to build and share data apps streamlit.io. <https://streamlit.io/>. [Accessed 08- 09-2024]. [6](#page-5-4)
- <span id="page-8-22"></span>[6] tae898/emoberta-large · Hugging Face — huggingface.co. [https://huggingface.co/tae898/emoberta](https://huggingface.co/tae898/emoberta-large)[large](https://huggingface.co/tae898/emoberta-large). [Accessed 08-09-2024]. [5](#page-4-3)
- <span id="page-8-21"></span>[7] AI@Meta. Llama 3 model card. 2024. [5](#page-4-3)
- <span id="page-8-14"></span>[8] Jason Ansel et al. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In *29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24)*. ACM, Apr. 2024. [5](#page-4-3)
- <span id="page-8-6"></span>[9] Max Bain, Arsha Nagrani, Andrew Brown, and Andrew Zisserman. Condensed movies: Story based retrieval with contextual embeddings, 2020. [2,](#page-1-0) [6](#page-5-4)
- <span id="page-8-16"></span>[10] Max Bain, Arsha Nagrani, Gul Varol, and Andrew Zisser- ¨ man. A clip-hitchhiker's guide to long video retrieval, 2022. [5](#page-4-3)
- <span id="page-8-2"></span>[11] Dmitry Baranchuk, Artem Babenko, and Yury Malkov. Revisiting the inverted indices for billion-scale approximate nearest neighbors, 2018. [1](#page-0-0)
- <span id="page-8-8"></span>[12] Samuel Suraj Bushi and Osmar Zaiane. Apnea: Intelligent ad-bidding using sentiment analysis. WI '19, page 76–83, New York, NY, USA, 2019. Association for Computing Machinery. [2](#page-1-0)
- <span id="page-8-23"></span>[13] Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N. Chang, Sungbok Lee, and Shrikanth S. Narayanan. Iemocap: interactive emotional dyadic motion capture database. *Language Resources and Evaluation*, 42(4):335–359, Nov. 2008. [5,](#page-4-3) [13](#page-12-0)
- <span id="page-8-17"></span>[14] Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about kinetics-600. *arXiv preprint arXiv:1808.01340*, 2018. Submitted on 3 Aug 2018. [5,](#page-4-3) [12](#page-11-0)
- <span id="page-8-18"></span>[15] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. *arXiv preprint arXiv:1907.06987*, 2019. Submitted on 15 Jul 2019 (v1), last revised 17 Oct 2022 (this version, v2). [5,](#page-4-3) [12](#page-11-0)

- <span id="page-8-26"></span>[16] Brandon Castellano. Home - PySceneDetect — scenedetect.com. <https://www.scenedetect.com/>. [Accessed 08-09-2024]. [8](#page-7-2)
- <span id="page-8-5"></span>[17] Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, Jinhui Tang, and Jing Liu. Valor: Vision-audiolanguage omni-perception pretraining model and dataset, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-4"></span>[18] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. VAST: A vision-audio-subtitle-text omni-modality foundation model and dataset. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-12"></span>[19] Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, and Dong Shen. Improving video-text retrieval by multi-stream corpus alignment and dual softmax loss, 2021. [3](#page-2-4)
- <span id="page-8-0"></span>[20] Adrian Dabrowski, Georg Merzdovnik, Johanna Ullrich, Gerald Sendera, and Edgar Weippl. Measuring cookies and web privacy in a post-gdpr world. In David Choffnes and Marinho Barcellos, editors, *Passive and Active Measurement*, pages 258–270, Cham, 2019. Springer International Publishing. [1](#page-0-0)
- <span id="page-8-3"></span>[21] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazare, Maria ´ Lomeli, Lucas Hosseini, and Herve J ´ egou. The faiss library, ´ 2024. [1](#page-0-0)
- <span id="page-8-28"></span>[22] Mai ElSherief, Caleb Ziems, David Muchlinski, Vaishnavi Anupindi, Jordyn Seybolt, Munmun De Choudhury, and Diyi Yang. Latent hatred: A benchmark for understanding implicit hate speech. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 345–363, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. [14](#page-13-0)
- <span id="page-8-11"></span>[23] Han Fang, Pengfei Xiong, Luhui Xu, and Yu Chen. Clip2video: Mastering video-text retrieval via image clip, 2021. [2](#page-1-0)
- <span id="page-8-9"></span>[24] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition, 2019. [2](#page-1-0)
- <span id="page-8-13"></span>[25] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. Multi-modal transformer for video retrieval, 2020. [3](#page-2-4)
- <span id="page-8-10"></span>[26] Yuying Ge, Yixiao Ge, Xihui Liu, Alex Jinpeng Wang, Jianping Wu, Ying Shan, Xiaohu Qie, and Ping Luo. Miles: Visual bert pre-training with injected language semantics for video-text retrieval, 2022. [2](#page-1-0)
- <span id="page-8-1"></span>[27] Google AdSense. Google adsense. [https://adsense.](https://adsense.google.com/start/) [google.com/start/](https://adsense.google.com/start/), 2024. Accessed: 2024-08-27. [1](#page-0-0)
- <span id="page-8-7"></span>[28] Google Cloud. Vertex ai: Multimodal embeddings api. [https : / / cloud . google . com / vertex](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api)  [ai/generative- ai/docs/model- reference/](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api) [multimodal- embeddings- api](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api), 2024. Accessed: 2024-08-27. [2,](#page-1-0) [6](#page-5-4)
- <span id="page-8-27"></span>[29] Keyan Guo, Alexander Hu, Jaden Mu, Ziheng Shi, Ziming Zhao, Nishant Vishwamitra, and Hongxin Hu. An investigation of large language models for real-world hate speech detection, 2024. [14](#page-13-0)

- <span id="page-9-11"></span>[30] Xudong Guo, Xun Guo, and Yan Lu. Ssan: Separable selfattention network for video representation learning, 2021. [2](#page-1-0)
- <span id="page-9-6"></span>[31] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable ´ vision learners, 2021. [1](#page-0-0)
- <span id="page-9-17"></span>[32] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 770–778, 2015. [5](#page-4-3)
- <span id="page-9-19"></span>[33] Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. spaCy: Industrial-strength Natural Language Processing in Python. 2020. [5](#page-4-3)
- <span id="page-9-0"></span>[34] E. Haglund and J. Bj ¨ orklund. Ai-driven contextual advertis- ¨ ing: Toward relevant messaging without personal data. *Journal of Current Issues & Research in Advertising*, 45(3):301– 319, 2024. [1](#page-0-0)
- <span id="page-9-18"></span>[35] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset. *arXiv preprint arXiv:1705.06950*, 2017. Submitted on 19 May 2017. [5,](#page-4-3) [12](#page-11-0)
- <span id="page-9-20"></span>[36] Jiyun Kim, Byounghan Lee, and Kyung-Ah Sohn. Why is it hate speech? masked rationale prediction for explainable hate speech detection. In Nicoletta Calzolari, Chu-Ren Huang, Hansaem Kim, James Pustejovsky, Leo Wanner, Key-Sun Choi, Pum-Mo Ryu, Hsin-Hsi Chen, Lucia Donatelli, Heng Ji, Sadao Kurohashi, Patrizia Paggio, Nianwen Xue, Seokhwan Kim, Younggyun Hahm, Zhong He, Tony Kyungil Lee, Enrico Santus, Francis Bond, and Seung-Hoon Na, editors, *Proceedings of the 29th International Conference on Computational Linguistics*, pages 6644–6655, Gyeongju, Republic of Korea, Oct. 2022. International Committee on Computational Linguistics. [5](#page-4-3)
- <span id="page-9-22"></span>[37] Taewoon Kim and Piek Vossen. Emoberta: Speaker-aware emotion recognition in conversation with roberta, 2021. [5,](#page-4-3) [13](#page-12-0)
- <span id="page-9-12"></span>[38] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling, 2021. [2](#page-1-0)
- <span id="page-9-25"></span>[39] Nicolas Lemieux and Rita Noumeir. A hierarchical learning approach for human action recognition. *Sensors*, 20(17), 2020. [12](#page-11-0)
- <span id="page-9-16"></span>[40] Dongxu Li, Junnan Li, Hung Le, Guangsen Wang, Silvio Savarese, and Steven C.H. Hoi. LAVIS: A one-stop library for language-vision intelligence. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)*, pages 31–41, Toronto, Canada, July 2023. Association for Computational Linguistics. [5](#page-4-3)
- <span id="page-9-1"></span>[41] Huiran Li and Yanwu Yang. Keyword targeting optimization in sponsored search advertising: Combining selection and matching. *Electronic Commerce Research and Applications*, 56:101209, 2022. [1](#page-0-0)
- <span id="page-9-14"></span>[42] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In *Pro-*

- *ceedings of the 40th International Conference on Machine Learning*, ICML'23. JMLR.org, 2023. [3,](#page-2-4) [4,](#page-3-4) [5,](#page-4-3) [13](#page-12-0)
- <span id="page-9-7"></span>[43] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multimodal video understanding benchmark, 2024. [1](#page-0-0)
- <span id="page-9-10"></span>[44] Jingang Liu, Chunhe Xia, Xiaojian Li, Haihua Yan, and Tengteng Liu. A bert-based ensemble model for chinese news topic prediction. BDE '20, page 18–23, New York, NY, USA, 2020. Association for Computing Machinery. [2](#page-1-0)
- <span id="page-9-2"></span>[45] Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. Use what you have: Video retrieval using representations from collaborative experts, 2020. [1,](#page-0-0) [3](#page-2-4)
- <span id="page-9-3"></span>[46] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval, 2021. [1,](#page-0-0) [2,](#page-1-0) [5,](#page-4-3) [7](#page-6-4)
- <span id="page-9-13"></span>[47] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. X-clip: End-to-end multi-grained contrastive learning for video-text retrieval, 2022. [2](#page-1-0)
- <span id="page-9-5"></span>[48] Yu. A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs, 2018. [1](#page-0-0)
- <span id="page-9-21"></span>[49] Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, and Animesh Mukherjee. Hatexplain: A benchmark dataset for explainable hate speech detection. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pages 14867–14875, 2021. [5](#page-4-3)
- <span id="page-9-8"></span>[50] Antoine Miech, Ivan Laptev, and Josef Sivic. Learning a text-video embedding from incomplete and heterogeneous data, 2020. [1,](#page-0-0) [3](#page-2-4)
- <span id="page-9-4"></span>[51] Niluthpol Chowdhury Mithun, Juncheng Li, Florian Metze, and Amit K. Roy-Chowdhury. Learning joint embedding with multimodal cues for cross-modal video-text retrieval. In *Proceedings of the 2018 ACM on International Conference on Multimedia Retrieval*, ICMR '18, page 19–27, New York, NY, USA, 2018. Association for Computing Machinery. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-24"></span>[52] OpenAI. Chatgpt (gpt-4). [https://chat.openai.](https://chat.openai.com) [com](https://chat.openai.com), 2024. Accessed: 2024-09-08. [6](#page-5-4)
- <span id="page-9-23"></span>[53] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. MELD: A multimodal multi-party dataset for emotion recognition in conversations. In Anna Korhonen, David Traum, and Llu´ıs Marquez, editors, ` *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 527– 536, Florence, Italy, July 2019. Association for Computational Linguistics. [5,](#page-4-3) [13](#page-12-0)
- <span id="page-9-9"></span>[54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. *CoRR*, abs/2103.00020, 2021. [2,](#page-1-0) [4,](#page-3-4) [7](#page-6-4)
- <span id="page-9-15"></span>[55] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 11 2019. [5](#page-4-3)

- <span id="page-10-19"></span>[56] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, October 2019. [5](#page-4-3)
- <span id="page-10-14"></span>[57] Nina Shvetsova, Brian Chen, Andrew Rouditchenko, Samuel Thomas, Brian Kingsbury, Rogerio Feris, David Harwath, James Glass, and Hilde Kuehne. Everything at once – multimodal fusion transformer for video retrieval, 2022. [2](#page-1-0)
- <span id="page-10-17"></span>[58] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. Mpnet: Masked and permuted pre-training for language understanding. *arXiv preprint arXiv:2004.09297*, 2020. [5](#page-4-3)
- <span id="page-10-1"></span>[59] Joanna Strycharz, Edith Smit, Natali Helberger, and Guda van Noort. No to cookies: Empowering impact of technical and legal knowledge on rejecting tracking cookies. *Computers in Human Behavior*, 120:106750, 2021. [1](#page-0-0)
- <span id="page-10-6"></span>[60] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality, 2024. [1](#page-0-0)
- <span id="page-10-12"></span>[61] Jing Yang Taylor Jing Wen, Ching-Hua Chuan and Wanhsiu Sunny Tsai. Predicting advertising persuasiveness: A decision tree method for understanding emotional (in)congruence of ad placement on youtube. *Journal of Current Issues & Research in Advertising*, 43(2):200–218, 2022. [2](#page-1-0)
- <span id="page-10-3"></span>[62] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training, 2022. [1](#page-0-0)
- <span id="page-10-11"></span>[63] Twelve Labs. Introducing marengo 2.6. [https://www.](https://www.twelvelabs.io/blog/introducing-marengo-2-6) [twelvelabs.io/blog/introducing- marengo-](https://www.twelvelabs.io/blog/introducing-marengo-2-6)[2-6](https://www.twelvelabs.io/blog/introducing-marengo-2-6), 2024. Accessed: 2024-08-27. [2,](#page-1-0) [6](#page-5-4)
- <span id="page-10-18"></span>[64] Ultralytics. YOLOv5: A state-of-the-art real-time object detection system. <https://docs.ultralytics.com>, 2021. [5](#page-4-3)
- <span id="page-10-13"></span>[65] Nikhita Vedula, Wei Sun, Hyunhwan Lee, Harsh Gupta, Mitsunori Ogihara, Joseph Johnson, Gang Ren, and Srinivasan Parthasarathy. Multimodal content analysis for effective advertisements on youtube, 2017. [2](#page-1-0)
- <span id="page-10-4"></span>[66] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking, 2023. [1,](#page-0-0) [12](#page-11-0)
- <span id="page-10-21"></span>[67] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. *arXiv preprint arXiv:2303.16727*, 2023. Submitted on 29 Mar 2023 (v1), last revised 18 Apr 2023 (this version, v2). [5](#page-4-3)
- <span id="page-10-9"></span>[68] Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. One-peace: Exploring one general representation model toward unlimited modalities, 2023. [2,](#page-1-0) [7](#page-6-4)
- <span id="page-10-7"></span>[69] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Chenting Wang, Guo Chen, Baoqi Pei, Ziang Yan, Rongkun Zheng, Jilan Xu, Zun Wang, Yansong Shi, Tianxiang Jiang, Songze Li, Hongjie Zhang, Yifei Huang, Yu Qiao, Yali Wang, and Limin Wang. Internvideo2: Scaling foundation models for multimodal video understanding, 2024. [1,](#page-0-0) [2](#page-1-0)

- <span id="page-10-22"></span>[70] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS '22, Red Hook, NY, USA, 2024. Curran Associates Inc. [14](#page-13-0)
- <span id="page-10-16"></span>[71] Yusong Wu\*, Ke Chen\*, Tianyu Zhang\*, Yuchen Hui\*, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In *IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP*, 2023. [2,](#page-1-0) [4,](#page-3-4) [5,](#page-4-3) [14](#page-13-0)
- <span id="page-10-5"></span>[72] Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, Guohai Xu, Ji Zhang, Songfang Huang, Fei Huang, and Jingren Zhou. mplug-2: A modularized multi-modal foundation model across text, image and video, 2023. [1](#page-0-0)
- <span id="page-10-8"></span>[73] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 5288–5296, 2016. [2,](#page-1-0) [3,](#page-2-4) [6](#page-5-4)
- <span id="page-10-0"></span>[74] Kaifu Zhang and Zsolt Katona. Contextual advertising. *Marketing Science*, 31(6):980–994, 2012. [1](#page-0-0)
- <span id="page-10-15"></span>[75] Shuai Zhao, Linchao Zhu, Xiaohan Wang, and Yi Yang. Centerclip: Token clustering for efficient text-video retrieval. In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*, SIGIR '22. ACM, July 2022. [2](#page-1-0)
- <span id="page-10-20"></span>[76] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2017. [5](#page-4-3)
- <span id="page-10-10"></span>[77] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, and Li Yuan. Languagebind: Extending video-language pretraining to nmodality by language-based semantic alignment, 2024. [2,](#page-1-0) [6,](#page-5-4) [7](#page-6-4)
- <span id="page-10-2"></span>[78] Cunjuan Zhu, Qi Jia, Wei Chen, Yanming Guo, and Yu Liu. Deep learning for video-text retrieval: a review, 2023. [1](#page-0-0)

# <span id="page-11-0"></span>A. Alternative Modality Queries to ContextIQ

The flexibility of our system allows us to effortlessly perform queries across different modalities, including video, audio, and image, ensuring any-to-any search capabilities.

Image Query: The process begins by encoding the input query image using the vision encoder of the vision-text model fθ<sup>1</sup> , resulting in an image embedding. This embedding is then compared against the vision embeddings of all available content {F<sup>v</sup> i : i = 1, 2, ..., N} using cosine similarity. The system retrieves and ranks content based on these similarity scores.

Video Query: For video queries, the system first extracts frame-level embeddings from the sampled frames of the query video using the vision encoder of the vision-text model fθ<sup>1</sup> . These frame-level embeddings are then aggregated using the previously defined aggregation function A<sup>v</sup> to generate a single video embedding. This video embedding is compared directly with the vision embedding database {F<sup>v</sup> i : i = 1, 2, ..., N} using cosine similarity. The content is then ranked according to similarity, with the most relevant videos appearing at the top of the results.

Audio Query: The process for audio queries begins by segmenting the query audio into a fixed number of segments. Each segment is encoded using the audio encoder of the audio-text model gθ<sup>2</sup> . These segment-level encodings are then aggregated using the previously defined aggregation function A<sup>a</sup> to form a single audio embedding. This aggregated embedding is compared against the audio embeddings database {F<sup>a</sup> i : i = 1, 2, ..., N} using cosine similarity. The results are then ranked based on these similarity scores.

# B. Video Action Recognition

## B.1. Simplifying Kinetics 710 classes

Reducing Kinetics 710 [\[14,](#page-8-17) [15,](#page-8-18) [35\]](#page-9-18) classes to minimize inter-class confusion can be done by either discarding irrelevant classes or combining similar ones. A hierarchical approach to combining Kinetics classes was explored in [\[39\]](#page-9-25) using a clustering method. However, this approach only provides examples rather than hierarchical clustering for the entire Kinetics dataset. In our ContextIQ system, we reduced the number of classes by collecting various signals and manually determining which classes to discard or combine. As a result, the number of classes was reduced from 710 to 185. The result is captured in this[sheet](https://github.com/AnokiAI/ContextIQ-Paper/blob/master/supplementary/video_action_recognition/simply_kinetics710_to_185_classes.xlsx) (as referred in the following paragraphs) present in our GitHub repository [https://github.com/AnokiAI/ContextIQ-Paper/.](https://github.com/AnokiAI/ContextIQ-Paper) The signals used were:

1. Relevance to contextual advertiser: Some classes, like "stretching arm" or "shuffling feet" may be too mundane, while others, like "playing oboe" or "clam digging," are too niche for a broad audience targeting. Using GPT-4, we identified and marked about 50% of classes as irrelevant for audience targeting (highlighted in red in the attached sheet). Examples of discarded classes include "Playing oboe" (niche instrument with limited audience), "Pole vault" (niche sport), and "Stretching leg" (too general for segmentation).

#### GPT4 Prompt:

*I have a list of 710 actions. Create a downloadable sheet with three columns, 710 actions, Discard, Reason. The discard should be Yes, but only if it seems less useful for detecting an action. If Discard is yes, also mention the reason (1-2 lines). I want to discard about 50% of the actions to keep the most useful half. An action is less useful if it does not seem helpful for creating audience segments for ad targeting. E.g, pinching action does not seem useful to target. Do not make any action class. Use the 710 as it is*. *[[Paste the list of 710 actions]]*

2. Groupings and correlated classes: Many classes in the Kinetics 710 set have high overlap, and their occurrence is highly correlated. For example, there are three separate classes for playing guitar, strumming guitar, and tapping guitar, which the model finds difficult to differentiate, leading to higher inter-class confusion. To address this, we used two approaches: one extends the existing Kinetics 400 [\[35\]](#page-9-18) groupings to 710 classes, and the other examines the top correlated classes during prediction.

K400 Groupings: The K400 set [\[35\]](#page-9-18) provided groupings of the 400 classes into 37 groups. However, these groupings were not extended to the additional 300 classes in the K710 set. For these additional 300 classes, we inferred their groupings by finding their text similarity with the 37 groups using the text encoder h<sup>θ</sup><sup>3</sup> used in the main paper and tagging the class to the most similar group. These classes are marked with a double asterisk in the K400 grouping column in the sheet.

Top-3 correlated classes: The VideoMAE2 model [\[66\]](#page-10-4) generates logits for 710 classes during inference. We build a co-occurrence matrix by counting every pair of classes in the Top-10 logit scores, then compute the correlation matrix. This process is applied to both the Kinetics validation set (50,000 videos) and our internal movie/TV clip set (Sec. 5.2 in the main paper). For instance, classes like dunking, dribbling, shooting, and playing basketball are highly correlated, allowing us to merge them into a single class, such as playing basketball.

- <span id="page-12-0"></span>3. Accuracy on the K710 validation set: Some classes perform poorly on the simpler Kinetics validation set (likely the same data distribution they were trained on), making them less likely to perform well on our movie clip dataset. We calculate the Top-1 and Top-3 accuracy for each class on the Kinetics set and highlight those in the bottom 25th percentile in the sheet. For instance, photobombing has a 38.8% Top-1 and 51% Top-3 accuracy, making it a candidate for discarding to reduce false positives and inter-class confusion.
- 4. Class occurrence ranking: Kinetics includes many classes that rarely occur in the wild, such as wood burning (art), stacking die, and wrestling alligator. In our large, diverse internal dataset, we found that 90% of the Top-3 search results come from only 218 classes 5. In the attached sheet, we list the occurrence count rank of each class (from 1 to 710) and highlight those beyond the top 218.

<span id="page-12-1"></span>![](_page_12_Figure_2.jpeg)

Figure 5. Cummulative Percentage Plot of Top-3 predicted actions on the internal set

Using the above four signals, the 710 classes were manually screened and reduced to **185 classes**:

- 182 classes were discarded.
- 418 classes were combined into 89 classes (see Tab. 6 for some of the obtained combined classes).
- 96 classes were retained.

Combining classes involves a trade-off between losing specificity and improving average precision @ K and prediction confidence. For instance, while predicting a broader class such as "drinking alcohol" (refer to row 2 of Table 6) can yield higher precision, it sacrifices the ability to differentiate between specific types like wine and beer.

### C. Emotion Recognition

**Text-based Emotion Recognition.** We use a pre-trained Emoberta-Large [37] model, which is trained on the MELD

<span id="page-12-2"></span>Table 6. Few examples of obtained combined classes

| Combined class     | Actions belonging to the class       |
|--------------------|--------------------------------------|
| playing cards      | playing poker, shuffling cards, card |
|                    | stacking, card throwing, dealing     |
|                    | cards, playing blackjack             |
| drinking alcohol   | uncorking champagne, bartending,     |
|                    | drinking beer, drinking shots, tast- |
|                    | ing beer, playing beer pong, pour-   |
|                    | ing beer, tasting wine, pouring      |
|                    | wine, opening bottle (not wine),     |
|                    | opening wine bottle                  |
| riding animal      | riding camel, riding elephant, rid-  |
|                    | ing mule, riding or walking with     |
|                    | horse                                |
| playing board game | playing monopoly, playing check-     |
|                    | ers, playing dominoes, playing       |
|                    | mahjong, playing scrabble            |
| cleaning floor     | cleaning floor, mopping floor,       |
|                    | sweeping floor, brushing floor,      |
|                    | vacuuming floor, sanding floor       |

[53] and IEMOCAP [13] datasets, for text-based emotion recognition as it is a speaker-aware model and shows better performance empirically on movie scene subtitles.

Leveraging Visual and Audio Cues for Emotion Recognition. The text-based models work only when there is enough text for the model to make a prediction. Moreover, it is difficult to find subtitles for some content, but still, we need to predict emotions in them for better retrieval and brand safe filtering. Since we already use the visiontext model and audio-text models for different parts of the ContextIQ system, we use these models to get some extra signals for predicting emotion. For example, we tagged the emotion joy with text queries like people smiling and people dancing, and assigned the emotion joy to all videos retrieved through the video (vision) modality using these queries. Concretely, we associate textual concepts that can be linked to different emotions and then find the scenes that have high video embedding similarity with the emotional text concept. Assume  $Q_t = \{t : e\}$  to be the text concept dictionary which contains strings t associated with different emotions e. Then, for a particular video scene x, we say that it is associated to an emotion e if,

$$f_{\theta_1}(x) \cdot f_{\theta_1}^T(t) > \tau_e \tag{8}$$

where  $f_{\theta_1}$  and  $f_{\theta_1}^T$  are the video and text encoders, respectively, of the vision-text model [42], and  $\tau_e$  is a predefined threshold for the concept-emotion pair t:e.

Empirical results show that textual emotion concepts work well only for *joy* emotion. For other emotions, either it is difficult to find emotional text concepts which are rel-

Table 7. Classification metrics for LLM, BERT and Ensemble Model

<span id="page-13-2"></span><span id="page-13-0"></span>

|           |                                   | Explicit Hate vs Normal Speech |                      |                       |          |      | Implicit Hate vs Normal Speech |                      |  |  |  |
|-----------|-----------------------------------|--------------------------------|----------------------|-----------------------|----------|------|--------------------------------|----------------------|--|--|--|
| Matria    | Metric LLM BERT Ensemble Ensemble | LLM                            | BERT                 | Ensemble              | Ensemble |      |                                |                      |  |  |  |
| Meure     | LLIVI                             | DEKI                           | $(OR, \theta = 0.7)$ | $(AND, \theta = 0.2)$ | LLIVI    | DEKI | $(OR, \theta = 0.7)$           | $(OR, \theta = 0.2)$ |  |  |  |
| Accuracy  | 83.9                              | 77.7                           | 81.5                 | 85.3                  | 75.3     | 63.4 | 73                             | 73.2                 |  |  |  |
| Precision | 78.9                              | 75.9                           | 74.5                 | 82                    | 75.2     | 66.8 | 70.7                           | 76.9                 |  |  |  |
| Recall    | 92.3                              | 81.1                           | 95.1                 | 90.2                  | 74.9     | 52.4 | 78.1                           | 66                   |  |  |  |
| F1 Score  | 85.1                              | 78.4                           | 83.5                 | 85.9                  | 75.1     | 58.8 | 74.2                           | 71                   |  |  |  |

Table 8. Differential Analysis for different prompting strategies

<span id="page-13-1"></span>

|                           | Explicit Hate vs Normal Speech |      |      |      |      | Implicit Hate vs Normal Speech |      |      |      |      |  |
|---------------------------|--------------------------------|------|------|------|------|--------------------------------|------|------|------|------|--|
| Reasoning                 | Yes                            | No   | Yes  | Yes  | Yes  | Yes                            | No   | Yes  | Yes  | Yes  |  |
| Definition of Hate Speech | Yes                            | Yes  | Yes  | Yes  | No   | Yes                            | Yes  | Yes  | Yes  | No   |  |
| Number of Examples        | 3                              | 3    | 1    | 0    | 3    | 3                              | 3    | 1    | 0    | 3    |  |
| Recall                    | 94.6                           | 97.2 | 95.2 | 93.5 | 94.9 | 76.5                           | 85.6 | 74.8 | 77.8 | 75.5 |  |
| Precision                 | 73.9                           | 65.8 | 72.3 | 70.2 | 71.0 | 70.3                           | 62.9 | 67.3 | 66.3 | 67.4 |  |
| Accuracy                  | 80.8                           | 73.4 | 79.4 | 77.1 | 78.7 | 71.9                           | 67.6 | 69.2 | 69.3 | 69.5 |  |
| F1 Score                  | 83.0                           | 78.4 | 82.2 | 80.2 | 81.2 | 73.3                           | 72.5 | 70.8 | 71.6 | 71.2 |  |

evant to that emotion, or the text concept associated to the emotion is not well represented by the vision-text model.

Similar to visual concepts, we associate audio concepts to different emotions given by  $Q_a=a:e$ , which contains audio files a and corresponding emotion e associated with that audio file. Then for a particular video scene x, we say that it is associated to an emotion e if,

$$g_{\theta_2}(x_a) \cdot g_{\theta_2}(a) > \tau_e \tag{9}$$

where  $g_{\theta_2}$  is the audio encoder of CLAP [71],  $x_a$  is the audio for the given video and  $\tau_e$  is a predefined threshold for the concept-emotion pair a:e. Note that we do not use the text encoder of CLAP because text-audio matching did not result into as good results as audio-audio matching. We have only linked audio emotion concepts to sad emotion because the rest of the emotions do not show good results empirically.

## **D. Hate Speech Detection**

**Aggregation Strategy**: To combine predictions from the BERT model, the scores for the Hate Speech and Offensive classes are summed. This aggregated score is then compared against a threshold of  $\theta=0.7$ . The final prediction is obtained by applying a logical OR operation between the thresholded BERT prediction and the predictions from the LLM to boost recall.

**Prompting Strategies**: We implement various prompting techniques to enhance the predictive performance of the LLM [29].

 Few-Shot Learning: A few examples are provided to the model to establish task context, improving its ability to accurately identify hate speech. Specifically we use three examples for the same.

- 2. **Definition of Hate Speech**: A precise definition of hate speech is included in the prompt to ensure consistent detection aligned with the dataset annotations. We use the following definition of hate speech: *Language that disparages a person or group on the basis of protected characteristics like race, gender, and cultural identity.*
- 3. **Structured JSON Output**: The model is instructed to return its response in JSON format, enabling easy parsing and seamless integration with the contextIQ system.
- 4. Chain of Thought Reasoning: The model is prompted to generate intermediate reasoning steps before determining whether content qualifies as hate speech, enhancing prediction accuracy. [70]

Various analyses were performed to evaluate the effectiveness of these strategies by using a combination of them for detection. Table 8 presents the results of these analyses. The results demonstrate that incorporating all the prompting strategies enhances detection performance, leading to improvements in accuracy, precision, and F1 score.

Validation Data and Results: We conducted validation using two datasets: an internal dataset and the implicit-hate dataset [22]. For implicit-hate, we sampled 250 examples each of Explicit Hate Speech, Implicit Hate Speech, and Normal Speech to ensure a balanced evaluation across different types of speech. In contrast, the internal dataset consisted of 11,645 examples, which, after applying a profanity

filter, was reduced to 10,645. Given the unbalanced distribution of hate speech versus normal speech on internal dataset, calculating recall was challenging. As a result, we only focused on the positive predictions generated by each model.

On the internal dataset, the BERT model identified 397 out of 10,645 examples (3.7%) as positive, while the LLM predicted 509 examples (4.8%) as positive. To assess these predictions, we randomly sampled 40 examples from each set of positive predictions, which were reviewed by two independent curators, given the subjective nature of the task. While precision varied significantly between curators owing to the subjective nature of the task, the LLM consistently outperformed the BERT model, with an average delta of 7.5%.

For the implicit-hate dataset, we evaluated various prompt templates and temperature values to enhance the performance of the LLM. A temperature value of 0.6, combined with the prompt template described [here,](https://github.com/AnokiAI/ContextIQ-Paper/blob/master/supplementary/hatespeech_detection/default_prompt_template.yaml) yielded the optimal results. Table [7](#page-13-2) presents the results for the best parameter combinations for both the LLM-based and BERT models, along with the outcomes for the ensemble models. The ensemble model outperformed the individual models, offering the flexibility to fine-tune precision and recall according to specific requirements. Additionally, the table also provides results for the ensemble model using both AND and OR operations across two different threshold values. The selection of these parameters can be guided by the desired balance between precision and recall in different scenarios.