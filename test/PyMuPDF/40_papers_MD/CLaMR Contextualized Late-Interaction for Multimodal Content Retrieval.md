## **CL A MR: Contextualized Late-Interaction for** **Multimodal Content Retrieval**

**David Wan** **Han Wang** **Elias Stengel-Eskin** **Jaemin Cho** **Mohit Bansal**
UNC Chapel Hill
```
      {davidwan, hwang, esteng, jmincho, mbansal}@cs.unc.edu

```

**Abstract**


Online video web content is richly multimodal: a single video blends vision,
speech, ambient audio, and on-screen text. Conventional retrieval systems typically
treat these modalities as independent retrieval sources, which can lead to noisy and
subpar retrieval. In this work, we explore multimodal video content retrieval, where
relevance can be scored from one particular modality or jointly across multiple
modalities simultaneously. Consequently, an effective retriever must dynamically
determine which modality (or set of modalities) best address a given query. We
introduce CL A MR, a multimodal, late-interaction retriever that jointly indexes four
modalities: video frames, transcribed speech, on-screen text, and other metadata.
CL A MR jointly encodes all modalities within a unified multimodal backbone for
improved contextualization and is trained to enhance dynamic modality selection
via two key innovations. First, to overcome the lack of training data for multimodal
retrieval, we introduce M ULTI VENT 2.0++, a large-scale synthetic training dataset
built on M ULTI VENT 2.0 (a dataset of event-centric videos in various languages
paired with English queries) with modality-targeted queries to teach modality
selection. Next, we propose a modality-aware contrastive loss that jointly trains
according to a standard contrastive objective alongside an objective for learning
correct modality usage. On the test sets of M ULTI VENT 2.0++ and MSRVTT,
we observe that conventional aggregation strategies, such as averaging similarities
for baseline retrievers, degrade performance by introducing noise from irrelevant
modalities. In contrast, CL A MR consistently outperforms existing retrievers: on
M ULTI VENT 2.0++, CL A MR improves nDCG@10 by 25.6 points over the bestperforming single-modality retriever and by 35.4 points over the best-performing
multi-modality retriever. We illustrate the downstream utility of CL A MR with
experiments on long-video QA, where we use CL A MR to retrieve relevant frames
and obtain an improvement of 3.50% over LanguageBind on Video-MME and
1.42% over dense frame sampling on LongVideoBench. [1]


**1** **Introduction**


Online platforms host a massive stream of video content that is natively _multimodal_, intertwining
visual scenes, spoken dialogue, ambient sound, on - screen text, and free-form descriptions [ 30 ].
Modern search engines and retrieval-augmented generation (RAG) systems therefore need to decide,
for every user query, _which_ of these heterogeneous sources actually contains useful data and _how_ to
exploit it [ 6 ]. However, effectively searching over and leveraging this rich multimodal content requires
combining signals from diverse sources in ways that prior work has not fully addressed. Existing
approaches often focus on a single modality (e.g., video), or convert content to text via captioning
or OCR [ 23, 32 ], which risks missing key information encoded in the original modality [ 6, 12 ].
Furthermore, current multimodal search engines that do treat different modalities as separate retrieval


1
Code and data are available in `[https://github.com/meetdavidwan/clamr](https://github.com/meetdavidwan/clamr)` .


|(a) Joint Encoding|Col2|Col3|
|---|---|---|
|**(a) Joint Encoding**|**c**|**c**|
|**(a) Joint Encoding**|||
|Joint<br>Enc|||
|Joint<br>Enc|||



Figure 1: Illustration of multimodal video content retrieval task with standard retrieval and CL A MR
with a query that is derived from the audio of the multimodal video content. Conventional retrieval
systems (top) encode each modality independently and then aggregate (e.g. mean, max, router) their
modality-specific similarity scores – a process that is easily contaminated by noise from irrelevant
modalities. By contrast, CL A MR (bottom) jointly encodes all modalities (a) and, via a late-interaction
mechanism (b), computes fine-grained, query-token-level similarities that dynamically focus on the
most relevant modality (audio and video) for the query.


sources often rely on simple heuristics for merging scores, such as maximum or reciprocal - rank
fusion (RRF) [ 8 ], as illustrated in Figure 1. These methods implicitly assume that multiple modalities
will agree on relevance, but risk drowning out valuable evidence from one modality with noise from
another, or allowing conflicting or misleading information from less relevant modalities to degrade
retrieval accuracy. In fact, as we show in Table 1, different combination methods, such as averaging
across the modalities, often lead to worse performance than using the best single modality, primarily
due to limited interaction and understanding between the modalities.


To close this gap, we introduce CL A MR (Sec. 3), a contextualized, late-interaction retriever that
jointly encodes video frames, speech transcripts, on-screen text, and other text metadata. Originally
studied in the text document retrieval domain, late-interaction (LI) models first independently encode queries and retrieval targets, then compute lightweight but fine-grained token-level similarity,
facilitating precise relevance judgments [ 19, 31 ]. This is in contrast to standard bi-encoder retrievers
that only compute cosine similarity between a pooled query and retrieval targets embeddings (Fig. 1
top). While promising, late interaction has primarily been studied in text-based contexts, with its
application in multimodal retrieval being largely restricted to single modalities like images [ 12 ] or
video frames [ 28 ]. Meanwhile, applying late interaction to retrieving multimodal video content has
remained unexplored. Inspired by recent advances in vision-language models that capture crossmodal inputs jointly [ 4, 5, 33 ], we propose to address this gap by using a single vision-language
backbone to encourage better contextualization of the modalities. As shown in Fig. 1 bottom, by
encoding _all sources together_ rather than in isolation, CL A MR learns directly from contrastive
signals which modality from the contextualized input to trust for each query, eliminating the need for
fragile combination techniques or routers [ 43 ] that require extra computation. To effectively teach
CL A MR to both retrieve the correct multimodal video content and to focus on the correct modality,
we propose a modality-aware contrastive loss for training CL A MR (Sec. 3.3). Our loss explicitly
encourages CL A MR to assign the highest similarity score to modalities containing query-relevant
information, thereby teaching CL A MR which modalities to focus on for a given query. For example,
in Fig. 2, we might generate a query derived from speech, and thus the model should learn to match
evidence encoded in the audio signal (as opposed to other modalities) to that query.


Finally, to further effectively train a late-interaction multimodal retriever that can dynamically select
between multiple modalities, we introduce synthetic training data, M ULTI VENT 2.0++ (Sec. 4),
building upon a large-scale video benchmark for event-centric video retrieval (M ULTI VENT 2.0

[ 20 ]). While M ULTI VENT 2.0 provides a massive set of multimodal data, it lacks sufficient modality

2


specific queries for training multimodal retrievers. M ULTI VENT 2.0++ addresses the lack of
large-scale training data by synthesizing queries specifically targeting different modalities for training,
and generates 371k modality-specific queries for unannotated videos from M ULTI VENT 2.0.


On the multimodal retrieval benchmark M ULTI VENT 2.0++ and popular text-video retrieval benchmark MSR-VTT [ 42 ]), CL A MR substantially outperforms all unimodal and multimodal baselines
across all retrieval metrics. For example, CL A MR surpasses strong unimodal and multimodal
retriever baselines by 25 _._ 7% nDCG@10 on M ULTI VENT 2.0++. Our ablation studies highlight
the critical roles of contextualization, modality-aware contrastive training, and the adaptability of
CL A MR when handling varying subsets of modalities. We demonstrate the downstream benefits of
CL A MR’s improved retrieval ability on long-video question answering (QA), where, given a query
about a long (up to _∼_ 60 minute) video, we use CL A MR to retrieve relevant segments. Given a
fixed frame budget, CL A MR provides improvements over LanguageBind on both VideoMME [ 13 ]
and LongVideoBench [ 40 ], two standard long-video QA benchmarks. These gains are driven by
CL A MR’s ability to retrieve more relevant segments of the video.


**2** **Related Work**


**Multimodal Retrievers.** Multimodal retrievers aim to align and retrieve information across different
modalities such as text, image, audio, and video. A key development is large-scale vision-language
pretraining with contrastive learning to align representations across modalities, as exemplified by
dual-encoder models like CLIP [ 25 ] and ALIGN [ 16 ]. These models learn joint embedding spaces for
images and text, inspiring extensions to additional modalities. For instance, ImageBind [ 14 ] extends
contrastive alignment beyond vision-text to audio and other input types, while LanguageBind [ 45 ]
uses language as a pivot to bind video and diverse modalities in a unified space. Recent retrievers
also incorporate structured signals such as OCR-extracted text [ 44 ], speech transcripts (ASR), and
video frame features [ 28 ] to handle complex content. However, dynamically selecting the most
relevant modality for each query remains challenging – most systems fuse modalities in a fixed way
or treat them independently, which can be suboptimal when only a subset of modalities is pertinent.
Emerging benchmarks like M ULTI VENT [ 20 ] emphasize this challenge by providing queries that
require retrieval via whatever modality contains the answer, underscoring the need for retrievers that
can adaptively focus on the right modality at query time. Our work addresses this gap by training a
single retriever to dynamically identify and focus on the most relevant modality per query, leveraging
modality-targeted supervision and a unified cross-modal backbone.


**Late Interaction.** Unlike standard dual encoder retrievers that match queries and documents via
fast but coarse-grained similarity in a shared embedding space [ 18, 29 ], or cross-encoders that
compute full query-document interactions at high computational cost [37], late-interaction methods
offer a middle-ground by enabling fine-grained token-level matching while retaining much of the
efficiency of dual encoders. ColBERT [ 19 ] introduces this multi-vector retrieval paradigm for text,
and ColBERTv2 [ 31 ] further improves its effectiveness and indexing efficiency. Originally developed
for monolingual text, late-interaction has since been extended to new languages and modalities.
JaColBERTv2.5 [ 7 ] explored multilingual late interactions retrievers. Similar techniques have been
adapted for vision context: ColPali [ 12 ] applies a ColBERT-style model to document images for
integrating text and image cues. These approaches allow token-level comparisons across modalities,
e.g., matching a query word to a specific image region or video segment, which is not possible
with single-vector representations. Notably, video retrieval methods like CLIP4Clip [ 22 ] leverage
pretrained CLIP features but still rely on pooled global embeddings or simple frame averaging,
whereas late-interaction models preserve multiple embeddings per item for detailed matching. Our
approach, CL A MR, differs by introducing modality-wise late interaction that computes token-level
scores separately across modalities and trains the model to select the most relevant one dynamically.
This design enables CLAMR to operate without routing heuristics or fusion rules, offering both
retrieval accuracy and interpretability in diverse multimodal settings.


**3** **CL** **A** **MR**


We propose CL A MR ( **C** ontextualized **La** te-interaction for **M** ultimodal content **R** etrieval), a novel
contextualized late-interaction multimodal retrieval framework capable of attending to different views
of multimodal web video content (e.g., frames, speech, text metadata). Unlike previous multimodal


3


Q [1]


Q [2]


Q [3]


Q [4]








|D1 D2 D3 D4|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|D<br>D<br>D<br>D<br>Q1,D1<br>Q1,D2<br>Q1,D3<br>Q1,D4<br>Q2,D1<br>Q2,D2<br>Q2,D3<br>Q2,D4<br>Q3,D1<br>Q3,D2<br>Q3,D3<br>Q3,D4<br>Q4,D1<br>Q4,D2<br>Q4,D3<br>Q4,D4<br>_Modality-Wise Late-Interaction_<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>S(Qi,Di)<br>_max_|D<br>D<br>D<br>D<br>Q1,D1<br>Q1,D2<br>Q1,D3<br>Q1,D4<br>Q2,D1<br>Q2,D2<br>Q2,D3<br>Q2,D4<br>Q3,D1<br>Q3,D2<br>Q3,D3<br>Q3,D4<br>Q4,D1<br>Q4,D2<br>Q4,D3<br>Q4,D4<br>_Modality-Wise Late-Interaction_<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>S(Qi,Di)<br>_max_|D<br>D<br>D<br>D<br>Q1,D1<br>Q1,D2<br>Q1,D3<br>Q1,D4<br>Q2,D1<br>Q2,D2<br>Q2,D3<br>Q2,D4<br>Q3,D1<br>Q3,D2<br>Q3,D3<br>Q3,D4<br>Q4,D1<br>Q4,D2<br>Q4,D3<br>Q4,D4<br>_Modality-Wise Late-Interaction_<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>S(Qi,Di)<br>_max_|D<br>D<br>D<br>D<br>Q1,D1<br>Q1,D2<br>Q1,D3<br>Q1,D4<br>Q2,D1<br>Q2,D2<br>Q2,D3<br>Q2,D4<br>Q3,D1<br>Q3,D2<br>Q3,D3<br>Q3,D4<br>Q4,D1<br>Q4,D2<br>Q4,D3<br>Q4,D4<br>_Modality-Wise Late-Interaction_<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>S(Qi,Di)<br>_max_|D<br>D<br>D<br>D<br>Q1,D1<br>Q1,D2<br>Q1,D3<br>Q1,D4<br>Q2,D1<br>Q2,D2<br>Q2,D3<br>Q2,D4<br>Q3,D1<br>Q3,D2<br>Q3,D3<br>Q3,D4<br>Q4,D1<br>Q4,D2<br>Q4,D3<br>Q4,D4<br>_Modality-Wise Late-Interaction_<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>S(Qi,Di)<br>_max_|
|D<br>D<br>D<br>D<br>Q1,D1<br>Q1,D2<br>Q1,D3<br>Q1,D4<br>Q2,D1<br>Q2,D2<br>Q2,D3<br>Q2,D4<br>Q3,D1<br>Q3,D2<br>Q3,D3<br>Q3,D4<br>Q4,D1<br>Q4,D2<br>Q4,D3<br>Q4,D4<br>_Modality-Wise Late-Interaction_<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>S(Qi,Di)<br>_max_|Q1,D1|Q1,D2|Q1,D3|Q1,D4|
|D<br>D<br>D<br>D<br>Q1,D1<br>Q1,D2<br>Q1,D3<br>Q1,D4<br>Q2,D1<br>Q2,D2<br>Q2,D3<br>Q2,D4<br>Q3,D1<br>Q3,D2<br>Q3,D3<br>Q3,D4<br>Q4,D1<br>Q4,D2<br>Q4,D3<br>Q4,D4<br>_Modality-Wise Late-Interaction_<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>S(Qi,Di)<br>_max_|Q2,D1|Q2,D2|Q2,D3|Q2,D4|
|D<br>D<br>D<br>D<br>Q1,D1<br>Q1,D2<br>Q1,D3<br>Q1,D4<br>Q2,D1<br>Q2,D2<br>Q2,D3<br>Q2,D4<br>Q3,D1<br>Q3,D2<br>Q3,D3<br>Q3,D4<br>Q4,D1<br>Q4,D2<br>Q4,D3<br>Q4,D4<br>_Modality-Wise Late-Interaction_<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>S(Qi,Di)<br>_max_|Q3,D1|Q3,D2|Q3,D3|Q3,D4|
|D<br>D<br>D<br>D<br>Q1,D1<br>Q1,D2<br>Q1,D3<br>Q1,D4<br>Q2,D1<br>Q2,D2<br>Q2,D3<br>Q2,D4<br>Q3,D1<br>Q3,D2<br>Q3,D3<br>Q3,D4<br>Q4,D1<br>Q4,D2<br>Q4,D3<br>Q4,D4<br>_Modality-Wise Late-Interaction_<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>∑ ( Sim(     ,      ), Sim(     ,      ),Sim(     ,      ))=0.6<br>S(Qi,Di)<br>_max_|Q4,D1|Q4,D2|Q4,D3|Q4,D4|









Figure 2: CL A MR with modality-wise late interaction for multimodal contrastive learning. A text
query and multimodal video document consisting of video’s visual, audio, OCR, and metadata
signals are encoded by the model. Then, token-level late interaction yields a similarity score for each
modality; the highest of these scores becomes the query-document similarity. Similarities for the
positive pair and in-batch negatives are fed to a contrastive loss.


retrieval methods that separately encode each modality, CL A MR focuses on contextualization by
encoding all modalities together and employs late-interaction to enable fine-grained retrieval. Below,
we explain task setup, CL A MR architecture, similarity computation, and training objective, in detail.


**3.1** **Task Setup**


Given a query _q_, the retriever must identify the most relevant document _d_ . Each document _d_ =
_{v, a, o, m, . . . }_ may contain _multiple_ modalities, such as video _v_, audio _a_, on-screen text _o_, textual
metadata _m_, etc. An example of such multimodal video content is depicted in the bottom part of
Fig. 2. The core retrieval challenge is to locate the relevant document, as the evidence establishing its
relevance might be found within a single modality or distributed across several.


**3.2** **Contextualized Multimodal Encoder**


To capture fine-grained visual cues, we primarily employ vision-language model (VLM). This VLM
is essential for leveraging detailed token- and patch-level interactions because it jointly encodes
all considered modalities. As illustrated in the bottom right of Fig. 2, all input modalities are first
concatenated into a single sequence – with visual inputs preceding textual inputs, based on the
model’s training regime. The VLM then processes this combined sequence to generate contextualized
hidden states for all tokens and patches. Finally, these hidden states are passed through a projection
layer to produce the final representation for each token. See Sec. 5 for more details.


**Omni-Models.** Given that modalities such as ASR are converted into text for VLMs to process,
we also explore integrating CL A MR with omni-models capable of processing additional input types
directly. Unlike VLMs, which require an initial conversion of ASR output to text, omni models such
as Qwen-Omni [ 41 ] can directly process raw audio. The setup generally follows that of using VLM,
with the exception of using pure audio instead of ASR.


**3.3** **Contextualized Late-Interaction.**


All hidden states are projected into a shared embedding space R _[D]_, where _D_ is the projection
dimension. A query yields **E** _q_ _∈_ R _[N]_ _[q]_ _[×][D]_, where _N_ _q_ is the length of the query tokens. Each document
provides one embedding matrix per modality **E** _d,m_ _∈_ R _[N]_ _[d,m]_ _[×][D]_ for _m ∈M_ . Late interaction
(LI) [ 19, 31, 12 ] compares _token-level embeddings_ instead of _pooled embeddings_ : for each query
token, its maximum cosine similarity to any document token is computed, and these maximum
similarities are then summed over all query tokens. In our task, we stack all modality embeddings


4


|Multimodal Video Content<br>Video Title: ASR: OCR: Metadata:<br>The Science of Soil Health: In our series on the science Origins of root-mediated pH Just when you thought soil<br>Nature's Way … of soil health… char soil… microbes …|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|||**Prompt a LLM to generate one**<br>**modality**‑**specific query for each modality**|||


**Query(description):**

_Soil microbes providing_
_carbon dioxide to plants_



**Query (base):**

_Soil health mineral_


_extraction video_





**Query (speech):**

_CO2 role in soil mineral_


_weathering_





**Query (OCR):**

_PH changes in soil_

_biogeochemistry_



Figure 3: Illustration of deriving modality-specific queries from multimodal video content. An LLM
uses the title, ASR, OCR, and metadata separately, and generates queries that can be answered using
_primarily_ by the designated modality.


and score them using standard LI. In this setup, each query token is matched with the most similar
document token from _any modality_ :



LI context ( _q, d_ ) =



_N_ _q_
� max _jN_ =1 _d_ � **E** [(] _q_ _[i]_ [)] _[,]_ [ [] **[E]** _[d,]_ [1] [;] _[ . . .]_ [ ;] **[ E]** _d,|M|_ []] [(] _[j]_ [)] [�] _,_ (1)

_i_ =1



where _N_ _d_ is the total number of document tokens from all modalities concatenated.


**3.4** **Training Objective: Multimodal Contrastive Learning.**


Our goal is to train the model to not only retrieve the correct document but also dynamically select the
optimal modalities. Let _{_ ( _q_ _k_ _, d_ _k_ ) _}_ _[b]_ _k_ =1 [be a batch, with one query per document. We use the standard]
InfoNCE loss [ 36 ] to train the model to retrieve the correct document from a batch that includes other
negative documents. This is achieved by bringing the representation of the correct (positive) querydocument pair closer in the embedding space while pushing representations of incorrect (negative)
pairs further apart. Illustrated in top right portion of Figure 2, the loss is formularized as follows:



_b_ _,_ (2)
~~�~~ _j_ =1 [exp] ~~�~~ _s_ _k,j_ _/τ_ ~~�~~



_L_ InfoNCE = _−_ [1]

_b_



_b_
�



exp� _s_ _k,k_ _/τ_ �

� log _b_

_k_ =1 ~~�~~ _j_ =1 [exp] ~~�~~ _s_ _k,j_



where _τ_ is a learnable temperature, and _s_ _i,j_ is the similarity score between query _q_ _i_ and document _d_ _j_ .


**Modality-Wise Late-Interaction.** Note that while the contextualized late-interaction can be directly
adapted as the similarity score, we observe that the model struggles to learn to use the modalities
effectively, as it must simultaneously learn to differentiate both between different examples and
between different modalities of the same example. Thus, we explore another more factorized
formulation _during training._ Here, we separately compute the late-interaction similarity score for
each modality and then select the maximum score. Since the modality-specific queries in our synthetic
training data, M ULTI VENT 2.0++, are designed to target a single modality, this modality-wise
approach during training guides the model to attend to one modality at a time, thereby enabling it to
focus on differentiating between distinct examples rather than simultaneously resolving modality and
example relevance. The similarity is defined as:



LI mw ( _q, d_ ) = max
_m∈M_



_N_ _q_ _N_ _d,m_
� max _j_ =1 � **E** [(] _q_ _[i]_ [)] _[,]_ **[ E]** [(] _d,m_ _[j]_ [)] � _._ (3)

_i_ =1



As illustrated in the top left portion of Fig. 2, after computing per-modality late-interaction scores
between an audio query and the different modalities of the multimodal video content, the similarity
score from the audio modality is the highest; this highest score is then used as the final similarity for
that query-document pair. As illustrated in the top right part of Fig. 2, after obtaining the similarity
score for each query-document pair in the batch (using LI mw ), these scores form a square similarity
matrix. In this matrix, the diagonal elements correspond to the positive (correct) query-document
pairings, while off-diagonal elements in each row represent negative pairings for that query.


5


**4** **M** **ULTI** **VENT 2.0++: Augmenting Training Data for Multimodal Retrieval**


To train a retriever to actively decide which modality to focus on, the training set must include
queries that are unambiguously grounded in a single modality. M ULTI VENT 2.0, however, was
not designed with this goal in mind: most of its 101K videos lack any queries, and the obvious
fallback—using the video title as a query—yields short, generic prompts that neither single out a
modality nor, in many cases, even appear in English. Among the 10K videos that _are_ annotated, only
1,504 queries are provided, a number too small to adequately train retrievers for fine-grained modality
selection. To address this limitation, we introduce M ULTI VENT 2.0++ augmenting training queries
for M ULTI VENT 2.0 on the unannotated videos.


**-**
**Synthetic Expansion of Modality** **Specific Queries.** Building on the design of the original
annotations—where each annotated video includes a ‘base’ query plus one specific query each for
audio, OCR, and metadata—we automatically extend this schema to 91k unannotated videos. For
each unannotated video, we first collect its modality sources: ASR transcripts, frame-level OCR text,
and video metadata (comprising title and human-written description). Subsequently, for each modality
source, we construct an in-context prompt consisting of ten human-written, modality-specific querycontent pairs randomly sampled from our annotated corpus. A large language model (LLM) is then
prompted with these examples to generate a base query (loosely derived from the video title) and
one new modality-specific query for each of these sources. The LLM is instructed to phrase these
generated queries such that a correct answer can be retrieved _primarily_ from the respective target
modality. Fig. 3 shows this generation pipeline. Our approach allows the LLM to generate queries
whose answers may occasionally be present in more than one modality—for instance, the term `pH`
`change` might appear in both OCR and ASR—thus encouraging the retriever to weigh corroborating
evidence rather than enforcing an artificially strict one-to-one query-modality mapping.


**LLM Choice for Synthetic Data Generation.** Because many videos contain non-English text, the
generator must both translate and condense content. We therefore use _Gemma_ _-_ _3_ _-_ _27b_ _-_ _it_ [ 34 ], whose
strong multilingual abilities make it well-suited to producing fluent, idiomatically-correct English
queries from diverse source languages. Furthermore, this model has demonstrated strong performance
in various NLP tasks, making it an appropriate choice for generating high-quality queries.


**Dataset Split.** Our training set consists of all synthetically generated queries and their associated
document, totaling 371,644 query-document pairs. From this set, we allocate 367,644 pairs for
training and 4,000 pairs for our validation set. For testing, we utilize the public benchmark split of
M ULTI VENT 2.0, which comprises 1,504 queries with available human judgments, as its private
benchmark split does not provide these. Importantly, the videos corresponding to these 1,504
M ULTI VENT 2.0 test queries were not used in the generation process of our synthetic data generation.


**5** **Experimental Setup**


**CL** **A** **MR Implementation Details.** We use `Qwen-VL-2.5-3B` [2] [ 1 ] as the backbone for CL A MR
with VLM, which offers strong multimodal accuracy at a modest size. For the Omni-model variant,
we experimented with `Qwen-Omni-3B` [3] [ 41 ], which utilizes Whisper [ 27 ] as its underlying audio
encoder. We append a 128 -dimensional linear projection layer, following ColPali [ 12 ]. We train
separate versions of CL A MR on M ULTI VENT 2.0++ and MSRVTT for 1 and 5 epochs, respectively.
Training is performed using a batch size of 16, distributed across 8 A100 80GB GPUs.To reduce
memory usage, we employ 4-bit quantization with QLoRA [ 9 ], setting the LoRA rank _r_ = 128 and
_α_ = 128 . Our implementation is built on the _transformers_ library [ 39 ]. Unless noted otherwise,
we keep default hyper-parameters, train with the 8-bit Adam optimizer, and set the learning rate to
1 _×_ 10 _[−]_ [5] for all experiments. Training on M ULTI VENT 2.0++ required approximately 10 hours,
while training on MSRVTT took about 4 hours. More details can be found in Appendix B.


**Baselines.** As single-modality baselines, we use multilingual CLIP (mCLIP) [4] from Reimers and
Gurevych [29] by processing only their corresponding modality (video, audio, OCR, or metadata). For
multi-modality baselines, we use several strong encoders: ImageBind [ 14 ], and LanguageBind [ 45 ].
For ImageBind and LanguageBind, we average the similarity scores obtained from all available


2 `[https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)`
3 `[https://huggingface.co/Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)`
4 `[https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)`


6


Table 1: Retrieval results on M ULTI VENT 2.0++ and MSRVTT. * indicates statistical significance
(p _<_ 0.05) compared to other baseline methods with a paired bootstrap test [10].


M ULTI VENT 2.0++ MSR-VTT
Method Modality R@1 R@5 R@10 nDCG@10 R@1 R@5 R@10 nDCG@10


_Single-Modality_


ICDAR + mCLIP OCR 2.9 10.4 14.7 8.1   -   -   -   Whisper + mCLIP Audio 4.5 19.7 24.5 13.9 5.2 8.7 10.8 7.7
Description + mCLIP Metadata 7.5 24.9 29.5 18.1   -   -   -   Video + mCLIP Vision 10.1 35.9 45.7 26.8 27.1 50.6 61.6 42.7
Imagebind Vision 15.4 43.0 52.1 32.8 28.9 52.8 63.3 44.9
LanguageBind Vision 14.2 39.5 47.9 30.2 40.2 64.3 74.8 56.5


_Multi-Modality_


mCLIP (avg.) All 7.9 31.9 39.7 23.0 19.5 38.3 47.0 32.2
mCLIP (router) All 7.0 29.0 34.8 20.5   -   -   -   ImageBind (avg.) All 3.9 10.6 14.0 8.5 20.4 35.7 43.0 30.9
ImageBind (router) All 8.9 22.2 27.3 17.7   -   -   -   LanguageBind (avg.) All 6.8 19.8 23.7 15.1 23.0 38.3 45.2 33.2
LanguageBind (router) All 9.8 27.3 33.2 21.0   -   -   -   Qwen VL 2.5 pooled All 21.6 74.8 81.6 52.2 36.2 62.9 73.9 53.8


_Ours_


CL A MR (Omni) All 25.5 81.1 85.2 55.7 45.5 69.8 **81.0** 62.1
CL A MR (VLM) All **26.7*** **85.1*** **88.0*** **58.5*** **46.1*** **71.3*** 79.8* **62.4***


modalities, a method we found to yield the best performance with these models. We also include
results using a router as an aggregation method. For this approach, we utilize GPT 4.1 to predict
the most relevant modality given the query and then use the similarity score from that predicted
modality as the final similarity score. Results using different modality combination techniques for
these baselines are presented in the Appendix. Finally, as an additional strong baseline, we fine-tune
the Qwen-VL 2.5 backbone (the same used for CL A MR) with a standard contrastive loss. This
involves using the embedding of the last token as the pooled representation for a sequence, a common
practice in VLM fine-tuning Bao et al. [2], Ouali et al. [24], Jiang et al. [17].


**Datasets.** Our primary evaluation dataset is M ULTI VENT 2.0++, where we train on our synthetically generated data and evaluate on the original public evaluation from M ULTI VENT 2.0. This
testing setup consists of 1,504 query-document pairs. We also include MSR-VTT [ 42 ], a standard
text-video retrieval benchmark used in several prior works [ 45, 3, 4 ]. Following prior work [ 22, 4 ],
we split the 10K examples of MSRVTT into 9K and 1K, for training and evaluation, respectively.


**Metrics.** Following standard practice in retrieval evaluation [ 21, 35 ], we evaluate the models performance using standard retrieval metrics: **Recall@k and nDCG@10** [ 15 ]. Recall@k measures whether
a relevant item appears in the top- _k_ retrieved results, while normalized Discounted Cumulative Gain
(nDCG) accounts for both the relevance and rank of retrieved items, assigning higher scores when
highly relevant items appear early in the ranked list and penalizing relevant items that appear lower.
We use the top-10 cutoff (nDCG@10) to balance sensitivity and efficiency in ranking evaluation.


**6** **Results**


**6.1** **Retrieval Results**


The results, presented in Tab. 1, demonstrate that CL A MR (VLM) consistently outperforms both
single-modality and multimodal baselines across all standard evaluation metrics. A key observation
is the challenge faced by conventional multimodal baselines when attempting to fuse information
from various modalities. For instance, models like mCLIP, ImageBind, and LanguageBind often
exhibit diminished performance compared to their vision-only version when using an average merging
strategy (avg.) for all modalities. On MSR-VTT, LanguageBind (Vision; the best performing singemodality baseline model) achieves an R@1 of 40 _._ 2%, while its multimodal average (LanguageBind
avg.) scores only 23 _._ 0% . This trend is also evident on M ULTI VENT 2.0, where ImageBind (Vision)
reaches 15 _._ 4% R@1, substantially higher than the 3 _._ 9% from ImageBind (avg.). This suggests that
naive fusion methods are susceptible to noise or suboptimal integration of complementary information
from diverse modalities, thereby hindering overall retrieval accuracy. Interestingly, employing a


7


Table 2: Ablation study on M ULTI VENT 2.0. B-C: impact of architectural and objective choices.
D-J: CL A MR trained and tested on a single modality. I-L: same models tested with all modalities.


Method Inference modality R@1 R@5 R@10 nDCG@10


(A) CL A MR All **26.66** **85.11** **88.03** **58.47**


_Architecture and training objective design_


(B) CL A MR without contextualization All 18.95 64.30 68.02 44.53
(C) CL A MR with LI context (instead of LI mw ) All 23.93 80.92 86.04 56.26


_Single-modality w. single-modality inference_


(D) CL A MR Vision Vision 16.22 57.58 65.49 40.71
(F) CL A MR Audio Audio 18.15 64.56 68.48 43.93
(G) CL A MR OCR OCR 19.68 62.10 67.95 43.19
(H) CL A MR Metadata Metadata 20.01 68.22 72.94 47.09


_Single-modality w. all-modality inference_


(I) CL A MR Vision All 23.93 76.06 82.78 53.62
(J) CL A MR Audio All 23.27 81.18 85.77 55.85
(K) CL A MR OCR All 24.40 82.38 86.37 56.97
(L) CL A MR Metadata All 22.27 80.92 85.84 55.60


router strategy for these multimodal baselines on M ULTI VENT 2.0++ shows a notable improvement
over the average merging strategy, though still falling short of vision-only performance in some cases.
For example, LanguageBind (router) shows a marked improvement with an R@1 of 9 _._ 8% compared
to LanguageBind (avg.) at 6 _._ 8%, but remains lower than LanguageBind (Vision) at 14 _._ 2% . This
indicates that while routing can be more effective than simple averaging for multimodal fusion, it
does not consistently outperform the strongest single-modality inputs for these baselines.


In stark contrast, CL A MR demonstrates superior performance by effectively leveraging multimodal
information. The VLM variant achieves the highest scores across all reported metrics on both
datasets except for R@10 on MSR-VTT where the Omni-model variant outperforms the VLM variant.
On MSR-VTT, CL A MR achieves an R@1 of 46 _._ 1%, surpassing the strongest single-modality
baseline (LanguageBind Vision) by 5 _._ 9% . The performance gains are even more pronounced on the
M ULTI VENT 2.0 dataset, where the queries target different modalities. Here, CL A MR achieves
an R@1 of 26 _._ 7%, which is 11 _._ 3% higher than the best performing single-modality baseline. These
results underscore the efficacy of our proposed approach in robustly integrating multimodal signals
for enhanced retrieval. The VLM demonstrates superior overall performance compared to the
Omni-model, particularly on M ULTI VENT 2.0++. We hypothesize this advantage stems from the
Omni-model’s architecture: accommodating speech tokens reduces its capacity for handling extended
sequence lengths, and in turn restricts batch sizes, impairing the effectiveness of contrastive learning.
Consequently, we focus primarily on the VLM for our subsequent results.


**6.2** **Ablation Studies**


To understand the contributions of different components of our proposed CL A MR architecture and
training strategy, we report ablation studies on the M ULTI VENT 2.0 dataset in Table 2.


**Impact of Contextualization.** First, we investigate the impact of contextualization, where we
jointly encode all the modalities in a single pass to the model. By removing the contextualization
mechanism from our full model (B), where we encode each modality separately and then concatenate
all the representations back together, we observed a substantial decrease in performance across all
metrics. Specifically, R@10 by 20 _._ 01% and nDCG@10 by 13 _._ 94%, highlighting contextualization’s
critical role in effectively fusing information from multiple modalities for improved retrieval.


**Impact of Late-interaction.** Next, we compare our proposed training objective with the contextualized late-interaction ( LI context ) (C). While the LI context model performs competently, our full model
(A) achieves superior results with an improvement of 1.99 in R@10 and 2.21 in nDCG@10 compared
to model (C). This suggests that our training objective facilitates a more effective learning process for
the model, enabling better integration and utilization of multimodal signals.


**Comparing Joint and Unimodal Training.** We compare the performance of models trained with
only a single modality to those trained with multiple. When restricted to their respective single


8


Table 3: Modality Accuracy on modality-specific setting.


Video ASR OCR Metadata Avg.


Router 22.4 30.9 20.0 56.9 30.9

mCLIP - max 0.0 54.3 69.1 39.2 39.5

CL A MR **58.2** **80.0** **84.3** **86.0** **76.4**


modalities during inference, these models performed considerably worse than the full multimodal
model. For instance, in row (D) CL A MR vision achieves a nDCG@10 of 40.71. Among these,
the metadata modality proves to be the most informative single source, while video is the least
informative. Interestingly, when these models are allowed to access all modalities during inference,
their performance significantly improved. For example, CL A MR vision (I) with all modalities (i.e.
not restricted to video at test-time), has its nDCG@10 from 40.71 to 53.62. This demonstrates the
model’s capability to leverage contextual information from auxiliary modalities at inference time,
even if not explicitly trained on all of them simultaneously for the primary task. This finding further
highlights the importance of rich contextual information for retrieval.


Despite these improvements, the performance of single-modality trained models still lags behind
our full CL A MR (A), which was trained with all modalities. This is true even with inference across
all modalities (I-L). For example, the best performing model in this category, CL A MR OCR with
all-modality inference (K), achieves an R@1 of 24.40, which is 2.26 points lower than the full model’s
R@1 of 26.66. This indicates that while leveraging all modalities at inference is beneficial, training
the model with comprehensive multimodal information leads to the most robust and effective retrieval
system. The most significant performance decrease occurs when training exclusively on video,
highlighting the crucial role of other modalities in multimodal video content retrieval. Training solely
on visual information evidently leads the model to under-utilize these other important modalities.


**6.3** **Query-Specific Analysis**


To better understand whether CL A MR correctly identifies and retrieves from the intended modality,
we conduct a fine-grained evaluation under modality-specific settings. This section describes how we
construct and validate modality-targeted queries, and how we use them to evaluate retrieval accuracy.


**Filtering Human-Written Queries.** We begin with a small pool of human-annotated queries from
M ULTI VENT 2.0 and apply an LLM-based filtering pipeline to verify their modality specificity. For
each query, we prompt the model to judge whether the answer is uniquely grounded in the annotated
target modality or also available in other modalities. A query passes this filter only if it is judged
answerable solely from the intended modality. For example, to assess video-grounded queries, we use
Qwen2.5-VL-72B-Inst to caption the visual content and check whether other textual modalities (ASR,
OCR, metadata) could also provide the answer. This filtering process yields a small but verified set of
modality-pure queries, which we use for preliminary analysis.


**Generating Synthetic Modality-Specific Queries.** To scale this analysis, we generate new queries
using an LLM prompted with four modality-specific documents (video, ASR, OCR, and metadata)
and instructed to produce a query answerable only by one target modality. We then reapply our
filtering step to verify that no other modality could answer the generated query. The surviving
examples are passed to human annotators for final verification. This expanded dataset allows us to
compute modality-specific retrieval accuracy at a larger scale.


**Results and Accuracy Breakdown.** We use this filtered dataset to evaluate whether a retriever
correctly attends to the intended modality when answering modality-specific queries. In Table 3,
we report modality-wise accuracy for CL A MR and a strong routing baseline. The router selects a
modality per query based on similarity to query type embeddings and executes retrieval only within
that modality, and for mCLIP we use the modality that scores the highest similarity.


CL A MR dramatically outperforms the router and mCLIP baseline across all modalities, achieving an
average accuracy of 76.4% versus 30.9%. Notably, it achieves particularly high accuracy for OCR
(84.3%) and ASR (80.0%), confirming that it learns to focus on the correct modality without explicit


9


Table 4: Long video QA results on Video-MME and LongVideoBench with different frame retrievers.


Video-MME
Frame Retriever Modality # Frames LongVideoBench
w/o subs w/ subs


No Sample   - 768 55.67 53.10 62.30


Uniform Sample   - 100 52.30 53.90 57.80
LanguageBind Vision 100   - 53.60 57.30
LanguageBind Vision + Audio 100 56.38 54.40 57.80
CL A MR Vision 100  - 55.60 59.40

CL A MR Vision + Audio 100 **57.09** **55.90** **61.30**


routing. In contrast, the router fails to adapt to the content of the query and performs poorly on
modalities like video and OCR and mCLIP fails to make use of the video modality. These results
validate that our training objective and architecture enable effective query-specific modality selection,
without the need for fragile routing heuristics.


**6.4** **Long Video QA**


**Setup.** To evaluate the effectiveness of CL A MR in a downstream scenario, we test on Long Video
Question Answering (QA) tasks using two benchmarks: the long-video subset (30 - 60 minutes in
length) of Video-MME [ 13 ] and the (900, 3600s] duration group from the dev set of LongVideoBench

[ 40 ]. Specifically, we set up a retrieval-augmented generation (RAG) pipeline: given a long video,
the retriever first selects key frames relevant to the question, which are subsequently provided as
input to a VLM (Qwen2.5-VL-7B-Inst) answerer. We compare CL A MR against several baselines:
uniform sampling, and retrieval-based methods using LanguageBind (vision only and vision+speech
modalities). LanguageBind was chosen as it is generally the second-best method in Tab. 1 on the
averaged multimodal setting when taking both datasets into account. We also include a no-sampling
baseline, where we provide the whole video as input. For this baseline, we follow the official
Qwen2.5-VL setting [ 1 ], which samples videos at 2 FPS and caps input at 768 frames per video, with
the total number of video tokens not exceeding 24,576. For Video-MME, which optionally includes
subtitle input, we evaluate both with and without subtitles. For LongVideoBench, since some queries
are grounded in subtitle content, subtitles are always provided. Performance is evaluated in terms of
QA accuracy (with and without subtitles for Video-MME).


**Results.** As shown in Tab. 4, CL A MR consistently outperforms all baseline retrievers across both
datasets. On LongVideoBench, CL A MR achieves 57.09% accuracy, surpassing uniform sampling by
4.79%. On Video-MME, CL A MR outperforms LanguageBind by 1.50% without subtitles and 3.50%
with subtitles. Overall, multimodal retrieval methods outperform single-modality ones, confirming
that leveraging multiple sources (e.g., vision and audio) helps retrieve more relevant content. For
example, even without subtitles, LanguageBind with both vision and speech inputs outperforms its
vision-only variant. CL A MR outperforms the no-sampling baseline, which feeds all 768 frames to
the vision-language model. This result indicates that full-frame inputs often include irrelevant or
distracting content, which can degrade answer accuracy [ 38 ]. By contrast, CL A MR selects a compact,
query-relevant subset of frames, promoting more focused reasoning and better QA performance.


**7** **Conclusion**


We presented CL A MR, a novel contextualized late-interaction retriever for multimodal content
retrieval that jointly encodes video frames, speech transcripts, on-screen text, and metadata within
a unified vision-language backbone. To enable the model to dynamically select the most relevant
modality for each query, we introduced M ULTI VENT 2.0++, a large-scale synthetic dataset of
modality-targeted queries built upon M ULTI VENT 2.0, and a modality-aware contrastive training
objective that explicitly guides the model to focus on the correct modality. Extensive experiments
on both M ULTI VENT 2.0++ and MSR-VTT demonstrate that CL A MR substantially outperforms
strong single-modality and multi-modality baselines. Finally, we showed that CL A MR’s improved
retrieval translates to downstream benefits in long-video QA, where retrieval of a more focused,
relevant frame set yields higher answer accuracy than uniform sampling or naive fusion strategies.


10


**Acknowledgments**


This work was supported by ARO Award W911NF2110220, ONR Grant N00014-23-1-2356, NSFCAREER Award 1846185, a Google PhD Fellowship, a Bloomberg Data Science PhD Fellowship,
the Microsoft Accelerate Foundation Models Research (AFMR) grant program, DARPA ECOLE
Program No. HR00112390060, and NSF-AI Engage Institute DRL-2112635. The views contained in
this article are those of the authors and not of the funding agency.


**References**


[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang
Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen
Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report,
2025. URL `[https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)` .


[2] Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti
Aggarwal, Subhojit Som, Songhao Piao, and Furu Wei. Vlmo: Unified visionlanguage pre-training with mixture-of-modality-experts. In S. Koyejo, S. Mohamed,
A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, _Advances in Neural Infor-_
_mation Processing Systems_, volume 35, pages 32897–32912. Curran Associates, Inc.,
2022. URL `[https://proceedings.neurips.cc/paper_files/paper/2022/file/](https://proceedings.neurips.cc/paper_files/paper/2022/file/d46662aa53e78a62afd980a29e0c37ed-Paper-Conference.pdf)`
`[d46662aa53e78a62afd980a29e0c37ed-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/d46662aa53e78a62afd980a29e0c37ed-Paper-Conference.pdf)` .


[3] Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, Jinhui Tang, and Jing
Liu. Valor: Vision-audio-language omni-perception pretraining model and dataset, 2023.


[4] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing
Liu. VAST: A vision-audio-subtitle-text omni-modality foundation model and dataset. In
_Thirty-seventh Conference on Neural Information Processing Systems_, 2023. URL `[https:](https://openreview.net/forum?id=scYa9DYUAy)`
`[//openreview.net/forum?id=scYa9DYUAy](https://openreview.net/forum?id=scYa9DYUAy)` .


[5] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz,
Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan
Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish V
Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol
Ayan, Carlos Riquelme Ruiz, Andreas Peter Steiner, Anelia Angelova, Xiaohua Zhai, Neil
Houlsby, and Radu Soricut. PaLI: A jointly-scaled multilingual language-image model. In
_The Eleventh International Conference on Learning Representations_, 2023. URL `[https:](https://openreview.net/forum?id=mWVoBz4W0u)`
`[//openreview.net/forum?id=mWVoBz4W0u](https://openreview.net/forum?id=mWVoBz4W0u)` .


[6] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multimodal retrieval is what you need for multi-page multi-document understanding, 2024.


[7] Benjamin Clavié. Jacolbertv2.5: Optimising multi-vector retrievers to create state-of-the-art
japanese retrievers with constrained resources, 2024. URL `[https://arxiv.org/abs/2407.](https://arxiv.org/abs/2407.20750)`
`[20750](https://arxiv.org/abs/2407.20750)` .


[8] Gordon V. Cormack, Charles L A Clarke, and Stefan Buettcher. Reciprocal rank fusion
outperforms condorcet and individual rank learning methods. In _Proceedings of the 32nd_
_International ACM SIGIR Conference on Research and Development in Information Retrieval_,
SIGIR ’09, page 758–759, New York, NY, USA, 2009. Association for Computing Machinery.
ISBN 9781605584836. doi: 10.1145/1571941.1572114. URL `[https://doi.org/10.1145/](https://doi.org/10.1145/1571941.1572114)`
`[1571941.1572114](https://doi.org/10.1145/1571941.1572114)` .


[9] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient
finetuning of quantized llms. _arXiv preprint arXiv:2305.14314_, 2023.


[10] Bradley Efron and Robert J. Tibshirani. _An Introduction to the Bootstrap_ . Number 57 in
Monographs on Statistics and Applied Probability. Chapman & Hall/CRC, Boca Raton, Florida,
USA, 1993.


11


[11] David Etter, Cameron Carpenter, and Nolan King. A hybrid model for multilingual ocr. In
_Document Analysis and Recognition - ICDAR 2023: 17th International Conference, San José,_
_CA, USA, August 21–26, 2023, Proceedings, Part I_, page 467–483, Berlin, Heidelberg, 2023.
Springer-Verlag. ISBN 978-3-031-41675-0. doi: 10.1007/978-3-031-41676-7_27. URL
`[https://doi.org/10.1007/978-3-031-41676-7_27](https://doi.org/10.1007/978-3-031-41676-7_27)` .


[12] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, CELINE HUDELOT,
and Pierre Colombo. Colpali: Efficient document retrieval with vision language models. In
_The Thirteenth International Conference on Learning Representations_, 2025. URL `[https:](https://openreview.net/forum?id=ogjBpZ8uSi)`
`[//openreview.net/forum?id=ogjBpZ8uSi](https://openreview.net/forum?id=ogjBpZ8uSi)` .


[13] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang,
Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui
Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. Video-mme:
The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis, 2024.
URL `[https://arxiv.org/abs/2405.21075](https://arxiv.org/abs/2405.21075)` .


[14] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala,
Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In _CVPR_,
2023.


[15] Kalervo Järvelin and Jaana Kekäläinen. Cumulated gain-based evaluation of ir techniques. _ACM_
_Trans. Inf. Syst._, 20(4):422–446, October 2002. ISSN 1046-8188. doi: 10.1145/582415.582418.
URL `[https://doi.org/10.1145/582415.582418](https://doi.org/10.1145/582415.582418)` .


[16] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan
Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning
with noisy text supervision. In Marina Meila and Tong Zhang, editors, _Proceedings of the_
_38th International Conference on Machine Learning_, volume 139 of _Proceedings of Machine_
_Learning Research_, pages 4904–4916. PMLR, 18–24 Jul 2021. URL `[https://proceedings.](https://proceedings.mlr.press/v139/jia21b.html)`
`[mlr.press/v139/jia21b.html](https://proceedings.mlr.press/v139/jia21b.html)` .


[17] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec:
Training vision-language models for massive multimodal embedding tasks, 2025. URL `[https:](https://arxiv.org/abs/2410.05160)`
`[//arxiv.org/abs/2410.05160](https://arxiv.org/abs/2410.05160)` .


[18] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, _Proceedings of the 2020_
_Conference on Empirical Methods in Natural Language Processing (EMNLP)_, pages 6769–
6781, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/
2020.emnlp-main.550. URL `[https://aclanthology.org/2020.emnlp-main.550/](https://aclanthology.org/2020.emnlp-main.550/)` .


[19] Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In _Proceedings of the 43rd International ACM SIGIR_
_Conference on Research and Development in Information Retrieval_, SIGIR ’20, page 39–48,
New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450380164. doi:
10.1145/3397271.3401075. URL `[https://doi.org/10.1145/3397271.3401075](https://doi.org/10.1145/3397271.3401075)` .


[20] Reno Kriz, Kate Sanders, David Etter, Kenton Murray, Cameron Carpenter, Kelly Van Ochten,
Hannah Recknor, Jimena Guallar-Blasco, Alexander Martin, Ronald Colaianni, Nolan King,
Eugene Yang, and Benjamin Van Durme. Multivent 2.0: A massive multilingual benchmark for
event-centric video retrieval, 2025. URL `[https://arxiv.org/abs/2410.11619](https://arxiv.org/abs/2410.11619)` .


[21] Tie-Yan Liu. Learning to rank for information retrieval. _Found. Trends Inf. Retr._, 3(3):225–331,
March 2009. ISSN 1554-0669. URL `[https://doi.org/10.1561/1500000016](https://doi.org/10.1561/1500000016)` .


[22] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip:
An empirical study of clip for end to end video clip retrieval, 2021. URL `[https://arxiv.](https://arxiv.org/abs/2104.08860)`
`[org/abs/2104.08860](https://arxiv.org/abs/2104.08860)` .


[23] Jamshed Memon, Maira Sami, Rizwan Ahmed Khan, and Mueen Uddin. Handwritten optical
character recognition (ocr): A comprehensive systematic literature review (slr). _IEEE Access_, 8:
142642–142668, 2020. doi: 10.1109/ACCESS.2020.3012542.


12


[24] Yassine Ouali, Adrian Bulat, Alexandros Xenos, Anestis Zaganidis, Ioannis Maniadis Metaxas,
Brais Martinez, and Georgios Tzimiropoulos. Vladva: Discriminative fine-tuning of lvlms,
2025. URL `[https://arxiv.org/abs/2412.04378](https://arxiv.org/abs/2412.04378)` .


[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya
Sutskever. Learning transferable visual models from natural language supervision. In Marina
Meila and Tong Zhang, editors, _Proceedings of the 38th International Conference on Machine_
_Learning_, volume 139 of _Proceedings of Machine Learning Research_, pages 8748–8763. PMLR,
18–24 Jul 2021. URL `[https://proceedings.mlr.press/v139/radford21a.html](https://proceedings.mlr.press/v139/radford21a.html)` .


[26] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya
Sutskever. Robust speech recognition via large-scale weak supervision, 2022. URL `[https:](https://arxiv.org/abs/2212.04356)`
`[//arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)` .


[27] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya
Sutskever. Robust speech recognition via large-scale weak supervision, 2022. URL `[https:](https://arxiv.org/abs/2212.04356)`
`[//arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)` .


[28] Arun Reddy, Alexander Martin, Eugene Yang, Andrew Yates, Kate Sanders, Kenton Murray,
Reno Kriz, Celso M. de Melo, Benjamin Van Durme, and Rama Chellappa. Video-colbert:
Contextualized late interaction for text-to-video retrieval, 2025. URL `[https://arxiv.org/](https://arxiv.org/abs/2503.19009)`
`[abs/2503.19009](https://arxiv.org/abs/2503.19009)` .


[29] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language_
_Processing_ . Association for Computational Linguistics, 11 2019. URL `[http://arxiv.org/](http://arxiv.org/abs/1908.10084)`
`[abs/1908.10084](http://arxiv.org/abs/1908.10084)` .


[30] Saron Samuel, Dan DeGenaro, Jimena Guallar-Blasco, Kate Sanders, Oluwaseun Eisape, Arun
Reddy, Alexander Martin, Andrew Yates, Eugene Yang, Cameron Carpenter, David Etter, Efsun
Kayi, Matthew Wiesner, Kenton Murray, and Reno Kriz. Mmmorrf: Multimodal multilingual
modularized reciprocal rank fusion, 2025. URL `[https://arxiv.org/abs/2503.20698](https://arxiv.org/abs/2503.20698)` .


[31] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia.
ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In Marine Carpuat,
Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, _Proceedings of the 2022_
_Conference of the North American Chapter of the Association for Computational Linguistics:_
_Human Language Technologies_, pages 3715–3734, Seattle, United States, July 2022. Association
for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.272. URL `[https://](https://aclanthology.org/2022.naacl-main.272/)`
`[aclanthology.org/2022.naacl-main.272/](https://aclanthology.org/2022.naacl-main.272/)` .


[32] R. Smith. An overview of the tesseract ocr engine. In _Ninth International Conference on_
_Document Analysis and Recognition (ICDAR 2007)_, volume 2, pages 629–633, 2007. doi:
10.1109/ICDAR.2007.4376991.


[33] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming
Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are
in-context learners. In _Proceedings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition (CVPR)_, pages 14398–14409, June 2024.


[34] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej,
Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard,
Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec,
Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy,
Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng, Noveen Sachdeva,
Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan
Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh
Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming
Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady,
Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander Kolesnikov, Alexei Bendebury,
Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das, Ankur Bapna,


13


Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal
Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A.
Choquette-Choo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle,
Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin
Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick Liu, Gagik
Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Pluci´nska, Harman Singh,
Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan
Nardini, Jean Pouget-Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi
Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu,
Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho,
Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min
Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan,
Noveen Sachdeva, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil
Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti,
Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan
Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth,
Siim Põder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu,
Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad
Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein
Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat
Black, Nabila Babar, Jessica Lo, Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas
Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle
Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel,
Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet,
Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud,
Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and Léonard
Hussenot. Gemma 3 technical report, 2025. URL `[https://arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786)` .


[35] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych.
BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In
_Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks_
_Track (Round 2)_, 2021. URL `[https://openreview.net/forum?id=wCu6T5xFjeJ](https://openreview.net/forum?id=wCu6T5xFjeJ)` .


[36] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive
predictive coding, 2019. URL `[https://arxiv.org/abs/1807.03748](https://arxiv.org/abs/1807.03748)` .


[37] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm:
Deep self-attention distillation for task-agnostic compression of pre-trained transformers.
In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, _Advances in_
_Neural Information Processing Systems_, volume 33, pages 5776–5788. Curran Associates,
Inc., 2020. URL `[https://proceedings.neurips.cc/paper_files/paper/2020/file/](https://proceedings.neurips.cc/paper_files/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)`
`[3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)` .


[38] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius,
and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on
long videos. _arXiv preprint arXiv:2405.19209_, 2024.


[39] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony
Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer,
Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain
Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Huggingface’s transformers:
State-of-the-art natural language processing, 2020. URL `[https://arxiv.org/abs/1910.](https://arxiv.org/abs/1910.03771)`
`[03771](https://arxiv.org/abs/1910.03771)` .


[40] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark
for long-context interleaved video-language understanding. In A. Globerson, L. Mackey,
D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, _Advances in Neural_
_Information Processing Systems_, volume 37, pages 28828–28857. Curran Associates, Inc.,
2024. URL `[https://proceedings.neurips.cc/paper_files/paper/2024/file/](https://proceedings.neurips.cc/paper_files/paper/2024/file/329ad516cf7a6ac306f29882e9c77558-Paper-Datasets_and_Benchmarks_Track.pdf)`
```
  329ad516cf7a6ac306f29882e9c77558-Paper-Datasets_and_Benchmarks_Track.
```

`[pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/329ad516cf7a6ac306f29882e9c77558-Paper-Datasets_and_Benchmarks_Track.pdf)` .


14


[41] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang,
Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-omni
technical report, 2025. URL `[https://arxiv.org/abs/2503.20215](https://arxiv.org/abs/2503.20215)` .


[42] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for
bridging video and language. In _2016 IEEE Conference on Computer Vision and Pattern_
_Recognition (CVPR)_, 2016.


[43] Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, and Sung Ju Hwang. Universalrag: Retrieval-augmented generation over multiple corpora with diverse modalities and
granularities, 2025. URL `[https://arxiv.org/abs/2504.20734](https://arxiv.org/abs/2504.20734)` .


[44] Jinxu Zhang, Yongqi Yu, and Yu Zhang. CREAM: Coarse-to-fine retrieval and multimodal efficient tuning for document VQA. In _ACM Multimedia 2024_, 2024. URL `[https:](https://openreview.net/forum?id=uxxdE9HFGI)`
`[//openreview.net/forum?id=uxxdE9HFGI](https://openreview.net/forum?id=uxxdE9HFGI)` .


[45] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang,
Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, and Li Yuan. Languagebind:
Extending video-language pretraining to n-modality by language-based semantic alignment,
2024. URL `[https://arxiv.org/abs/2310.01852](https://arxiv.org/abs/2310.01852)` .


**A** **Additional Experiments**


**A.1** **Exploration of Different Contrastive Loss Formulations**


We investigated two alternative formulations of the contrastive objective, each designed to progressively enforce the contribution of the single, most relevant modality signal.


**InfoNCE with Correct** **-** **Modality Positives.** To encourage the model to focus on the correct
modality, we keep the same denominator but replace each positive with the score computed _only_ on
the correct modality _m_ _[∗]_ _k_ [. The contrastive objective thus helps to also put the distance between the]
query and the document embedding that uses the correct modality closer:


_̸_


_̸_



exp ~~�~~ _s_ _mk,k_ ~~_[∗]_~~ _k_ _[/τ]_ ~~�~~ + ~~[�]~~ _[b]_ _j_ =1 _,_ _,j_ = _̸_ _k_ [exp] ~~�~~ _s_ _k,j_ _/τ_ ~~�~~ _[.]_ (4)


_̸_



_L_ ModPos = _−_ [1]

_b_

_̸_


_̸_



_k_ � =1 _b_ log exp ~~�~~ _s_ _m_ ~~_[∗]_~~ _k_ _[/τ]_ ~~�~~ exp+ ~~[�]~~ � _s_ _[b][m]_ _k_ =1 _,_ _k_ _[∗]_ _k_ _[/τ]_ = _̸_ � _k_


_̸_



_b_
�

_̸_


_̸_



_̸_


**InfoNCE with Modality Negatives.** To comprehensively encourage the model to _distinguish_
modalities, we treat (i) other documents, (ii) other modalities of the _same_ document, and (iii) every
modality of other documents as negatives. The loss becomes


_̸_



_̸_


~~�~~ _bj_ =1 ~~�~~ _̸_



_̸_


_,_

_._ (5)
_m∈M_ [exp] ~~�~~ _s_ _[m]_ _k,j_ _[/τ]_ ~~�~~ + ~~[�]~~ _[b]_ _j_ =1 _,j_ = _̸_ _k_ [exp] ~~�~~ _s_ _k,j_ _/τ_ ~~�~~



_̸_


_L_ ModNeg = _−_ [1]

_b_

_̸_



_̸_


� _b_ log _b_ exp� _s_ _[m]_ _k,_ _k_ _[∗]_ _k_ _[/τ]_ �

_k_ =1 ~~�~~ _j_ =1 ~~�~~ _m∈M_ [exp] ~~�~~ _s_ _[m]_ _k,j_ _[/τ]_ ~~�~~ + ~~[�]~~ _̸_



_̸_


_b_
�

_̸_



_̸_


_̸_


Together, the two objectives progressively strengthen the model’s ability to attend to the correct
modality.


**Results.** As detailed in Table 5, applying these additional constraints to the contrastive loss did
not improve retrieval performance compared to our main CL A MR (row a). In fact, increasing
the constraints led to a decrease in performance. CL A MR trained with correct-modality positives
( _L_ ModPos, row d) resulted in R@10 of 86.6 and nDCG@10 of 56.8. This is a decrease of 1.4 points
in R@10 and 1.7 points in nDCG@10 compared to the baseline CL A MR (row a, R@10: 88.0,
nDCG@10: 58.5). Employing the more stringent modality negatives ( _L_ ModNeg, row e) further reduced
performance, with R@10 dropping to 84.7 and nDCG@10 to 54.8. This represents a decrease of 3.3
points in R@10 and 3.7 points in nDCG@10 relative to the baseline (row a). These findings suggest
that the underlying assumption that a query is solely relevant to one specific modality might be overly
restrictive. The retriever appears to benefit from leveraging contextual signals from all available input
modalities rather than being forced to focus exclusively on a single “correct” one.


15


Table 5: Retrieval results on M ULTI VENT 2.0++.


Method R@1 R@5 R@10 nDCG@10


(a) CL A MR w. Qwen-2.5-VL **26.7** **85.1** **88.0** **58.5**
(b) Qwen-VL-2.5 + pooled representation 21.6 74.8 81.6 52.2
(c) CL A MR w. Qwen-Omni 25.5 81.1 85.2 55.7
(d) CL A MR w. _L_ ModPos 25.0 82.4 86.6 56.8
(e) CL A MR w. _L_ ModNeg 22.3 79.8 84.7 54.8


Table 6: Per-modality results of CL A MR on M ULTI VENT 2.0++.


R@10 nDCG@10


Base ASR OCR Metadata All Base ASR OCR Metadata All


71.4 98.1 86.4 97.8 88.0 47.4 64.2 62.6 63.3 58.5


**A.2** **Per-modality Performance Analysis**


To further understand CL A MR’s behavior, we analyze its performance on subsets of the evaluation
data, segmented by the primary modality targeted by the query (e.g., ASR, OCR, metadata, or ‘Base’
for general queries). The R@10 and nDCG@10 scores for these segments are presented in Table 6
for the CL A MR with Qwen-2.5-VL.


As shown in Tab. 6, CL A MR achieves very high R@10 scores for queries specifically targeting ASR
(98.1) and metadata (97.8), and strong performance for OCR-related queries (86.4). This indicates
that when a query has a clear signal in one of these textual modalities, the model is highly effective
at retrieving relevant documents. Queries categorized as ‘Base’—which may rely more on holistic
video understanding or a combination of visual information and less distinct textual cues—exhibit a
comparatively lower R@10 of 71.4. A similar trend is observable for nDCG@10, where ASR, OCR,
and metadata-targeted queries perform well, while ‘Base’ queries score lower.


**B** **Additional Experimental Setup Details**


**B.1** **CL** **A** **MR Implementation Details.**


We set the maximum query length to 64 tokens. Because queries are usually far shorter than
the associated documents, we follow prior work [ 19, 12 ] and mitigate this length asymmetry by
appending placeholder tokens to each query. Specifically, we add five extra tokens to help with
re-weighting the original query terms. For video input, we resize frames to 224 _×_ 224 pixels and
use the default processor to perform any extra transformations. The maximum token length for
other textual modalities (ASR, OCR, and metadata) is set to 256. For M ULTI VENT 2.0++, we
adopt the same modality configuration as M ULTI VENT 2.0, relying on the pre-extracted features
released by its authors. Concretely, each video contributes (i) up to ten key frames detected with
PY S CENE D ETECT [5], (ii) ASR transcripts generated by Whisper [ 26 ], (iii) OCR using Etter et al. [11],
and (iv) textual metadata descriptions supplied with the dataset. For MSRVTT, we only extract ASR
using Whisper V3.


**Omni-Model.** Processing audio consumes a significant number of tokens, which complicates
training procedures requiring large batch sizes. Therefore, we limited audio input to a maximum of
30 seconds, corresponding to 750 tokens. For visual input, we uniformly sampled 10 frames. The
maximum token length for OCR and metadata was set to 256. This configuration resulted in an
average input length of approximately 2048 tokens per sample, enabling an effective batch size of 16
on four A100 80GB GPUs. We use the same settings for the other hyper-parameters as CL A MR with
Qwen-VL-2.5.


5 `[https://www.scenedetect.com/](https://www.scenedetect.com/)`


16


**Prompt Type** **Prompt**


**Filtering** `You are a helpful retriever.` `Given a query and a`
```
          document, you need to determine if the document is
          relevant to the query. You only need to answer with
          ’yes’ or ’no’.
          Query: {query}
          Document: {doc}

          Answer:

```

**Generating** `Given four documents, generate a short query (less`
```
          than 10 words) that is only related to the document
          {target_id}. The other three documents should not be
          related to the query.
          Document 1: {doc_video}
          Document 2: {doc_speech}
          Document 3: {doc_ocr}
          Document 4: {doc_description}
          Query:

```

Figure 4: Prompts used for filtering relevant modality and generating synthetic modality-specific
queries.


System Prompt: `You are an assistant that creates search queries`
```
            that would help users find videos. Create a
            concise and specific query. Do not output any extra
            information.

```

User Message: `## Examples`

```
            {ICL examples}

            ## Your Task

            {Video data for this query type}
            **Query:**

```

Video Examples: `**Video Title:** {Title}`
```
            **Query:** {Query}

```

ASR Examples: `**Video Speech:** {Speech}`
```
            **Query:** {Query}

```

OCR Examples: `**Video OCR:** {OCR}`
```
            **Query:** {Query}

```

Description Examples: `**Video Description:** {Description}`
```
            **Query:** {Query}

```

Figure 5: Prompt structure for synthetic query generation for M ULTI VENT 2.0++. The prompt
begins with a system instruction, followed by a user message that incorporates in-context learning
(ICL) examples and video data corresponding to one of the four specified modality types (Video Title,
ASR, OCR, or Description).


**C** **Prompts**


The prompts employed for generating synthetic training data for M ULTI VENT 2.0++ are detailed in
Figure 5. We also provide the prompt used for the router in Figure 6.


17


Prompt `You are an expert query classifier.` `Given a user`
```
            query, determine which modality is most relevant for
            answering it. The possible modalities are: video,
            speech, ocr, description. Respond with only the
            predicted modality name.
            Here are some examples:
            {ICL Examples}
            Now, classify the following query:
            Query: {Query}
            Modality:

```

Figure 6: Prompt for router with GPT-4.1.


**C.1** **Safeguards for M** **ULTI** **VENT 2.0++**


The videos utilized are from the M ULTI VENT 2.0 dataset. We rely on the safeguarding measures
implemented by the original authors for this content and do not redistribute the videos. For our
synthetically generated queries, which were created using Gemma-3, our safeguarding strategy
included: (1) Prompt Engineering: Prompts were designed to elicit factual, descriptive, and taskrelevant queries suitable for video retrieval, thereby avoiding the generation of inappropriate outputs.
(2) Limited Scope: The queries are specific to an academic video retrieval task, a characteristic that
inherently curtails their potential for broader misuse.


**D** **Limitations and Broader Impact Statement**


This research introduces CL A MR, a multimodal retrieval model designed to dynamically leverage
multiple content modalities (video frames, audio transcripts, OCR text, and metadata) to improve
retrieval accuracy significantly. Given the broad applicability of such multimodal retrieval technologies, it has the potential for both positive and negative applications. In our work, we have taken
in the design of the prompts to mitigate risk; however, like other retrieval methods, it could be
applied in negative ways. In summary, we do not believe that our method has more potential for
misuse or negative impact than any other retrieval method, and that its improvements offer subtantial
opportunities for positive use.


Our study addresses multimodal video retrieval, training the retriever with a contrastive objective
that benefits from large batch sizes. GPU-memory limits confined us to a batch size of 16, and, in
the Omni model, required shortening the context window for non-text modalities. We expect that
techniques such as quantization, memory-efficient optimizers, and improved long-context handling
will soon enable both larger models and substantially larger batches. Likewise, ongoing advances in
late-interaction architectures and retrieval-system engineering should further boost accuracy while
reducing latency.


**E** **Licences**


M ULTI VENT is released under the Apache 2.0 license. PyTorch is under BSD-Style license and
`transformers` is under Apache license.


18


