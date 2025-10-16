# Unified Static and Dynamic Network: Efficient Temporal Filtering for Video Grounding

Jingjing Hu<sup>®</sup>, Dan Guo<sup>®</sup>, *Senior Member, IEEE*, Kun Li<sup>®</sup>, Zhan Si<sup>®</sup>, Xun Yang<sup>®</sup>, Xiaojun Chang<sup>®</sup>, *Senior Member, IEEE* and Meng Wang<sup>®</sup>, *Fellow, IEEE* 

Abstract—Inspired by the activity-silent and persistent activity mechanisms in human visual perception biology, we design a Unified Static and Dynamic Network (UniSDNet), to learn the semantic association between the video and text/audio queries in a cross-modal environment for efficient video grounding. For static modeling, we devise a novel residual structure (ResMLP) to boost the global comprehensive interaction between the video segments and queries, achieving more effective semantic enhancement/supplement. For dynamic modeling, we effectively exploit three characteristics of the persistent activity mechanism in our network design for a better video context comprehension. Specifically, we construct a diffusely connected video clip graph on the basis of 2D sparse temporal masking to reflect the "short-term effect" relationship. We innovatively consider the temporal distance and relevance as the joint "auxiliary evidence clues" and design a multi-kernel Temporal Gaussian Filter to expand the context clue into high-dimensional space, simulating the "complex visual perception", and then conduct element level filtering convolution operations on neighbour clip nodes in message passing stage for finally generating and ranking the candidate proposals. Our UniSDNet is applicable to both Natural Language Video Grounding (NLVG) and Spoken Language Video Grounding (SLVG) tasks. Our UniSDNet achieves SOTA performance on three widely used datasets for NLVG, as well as three datasets for SLVG, e.g., reporting new records at 38.88% R@1, IoU@0.7 on ActivityNet Captions and 40.26% R@1, IoU@0.5 on TACoS. To facilitate this field, we collect two new datasets (Charades-STA Speech and TACoS Speech) for SLVG task. Meanwhile, the inference speed of our UniSDNet is 1.56× faster than the strong multi-query benchmark. Code is available at: https://github.com/xian-sh/UniSDNet.

Index Terms—Natural Language Video Grounding, Spoken Language Video Grounding, Video Moment Retrieval, Video Understanding, Vision and Language

# <span id="page-0-1"></span>1 Introduction

Temporal Video Grounding (TVG), also called language-queried moment retrieval (MR), as a fundamental and challenging task in video understanding, has gained importance with the surge of online videos, attracting significant attention from both academia and industry in recent years. Generally, the TVG task refers to a natural language sentence as a query, with the goal of locating the accurate video segment that semantically corresponds to the query [3], [4],

- J. Hu, D. Guo, K. Li, and M. Wang are with Key Laboratory of Knowledge Engineering with Big Data (HFUT), Ministry of Education, School of Computer Science and Information Engineering (School of Artificial Intelligence), Hefei University of Technology (HFUT), and Intelligent Interconnected Systems Laboratory of Anhui Province (HFUT), Hefei, 230601, China (e-mail: xianhij623@gmail.com; guodan@hfut.edu.cn; kunli.hfut@gmail.com; eric.mengwang@gmail.com).
- Z. Si is with the Department of Chemistry and Centre for Atomic Engineering of Advanced Materials, Anhui University, Hefei, Anhui 230601, P.R. China (e-mail: naa0528@163.com).
- X. Yang and X. Chang are with the Department of Electronic Engineering and Information Science, School of Information Science and Technology, University of Science and Technology of China, Hefei 230026, China (email: xyang21@ustc.edu.cn; xjchang@ustc.edu.cn).
- D. Guo and M. Wang are also with the Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, Hefei, 230026, China.
- Corresponding authors: D. Guo, X. Yang, X. Chang, M. Wang.

This work was supported in part by the National Natural Science Foundation of China (62272144, 72188101, 62020106007, and U20A20183), and the Major Project of Anhui Province (202203a05020011, 2408085J040, 202423k09020001). Fundamental Research Funds for the Central Universities (JZ2024HGTG0309, JZ2024AHST0337 and JZ2023YQTD0072).

![](_page_0_Figure_14.jpeg)

<span id="page-0-0"></span>Fig. 1. A schematic illustration of the biology behind how people understand the events of a video during solving video grounding tasks. Firstly, according to the theory of GNW (*Global Neuronal Workspace*) [1], the brain engages in static multimodal information association to achieve semantic complements between multimodalities. Then the focus will be brought to the dynamic perception of the video content along the timeline, and during which three characteristics will be expressed: 1) Short-term Effect: the most recent perceptions have a high impact on the present; 2) Relevance Clues: semantically scenes will provide clues to help understand the current scene; 3) Perception Complexity: visual perception is high-dimensional and non-linear [2].

and the task is named *Natural Language Video Grounding* (*NLVG*). With the development of Automatic Speech Recognition (ASR) and Text-to-speech (TTS), speech is becoming an essential medium for Human-Computer Interaction (HCI). *Spoken Language Video Grounding* (*SLVG*) [5] has also gained a lot of attention. We find that whether using text or speech as a query, the key to solving TVG lies in video understanding and cross-modal interaction. Our work is

![](_page_1_Figure_2.jpeg)

<span id="page-1-0"></span>Fig. 2. An illustrating example for the video grounding task (query: text or audio). This video is described by four queries (events), all of which have separate semantic contexts and temporal dependencies. Other queries can provide a global context (antecedents and consequences) for the current query (e.g., query Q4). Besides, historical similar scenarios (such as in the blue dashed box) help to discover relevant event clues (time and semantic clues) for understanding the current scenario (blue solid box).

devoted to multimodal semantics-driven video understanding, namely, how to aggregate multimodal information for better video understanding?

In this work, we revisit solving TVG tasks through the lens of human visual perception biology [1], [2], as illustrated in Fig. 1. We observe that humans quickly comprehend queried events in a video, a process linked to the Global Neuronal Workspace (GNW) theory and dynamic visual perception theory in the brain's prefrontal cortex (PFC) [1], [2]. These theories describe the interplay between activity-silent and persistent activity mechanisms in the PFC [2]. The GNW theory suggests that when the brain processes multi-source data, it creates shallow correlations, allowing for semantic complementation between multimodal information. This step might not need overly complex deep networks for multimodal interactions between video and language. After that, the brain might pay attention on correlating as much useful information as possible. It will then focus on the video content and conduct dynamic visual perception that is transmitted along the Timeline Main Clue and exhibits three characteristics: 1) Short-term Effect: nearby perceptions strongly affect current perceptions; 2) Auxiliary Evidence (Relevance) Cues: semantically relevant scenes in the video provide auxiliary time and semantic cues; 3) Perception Complexity: the perception process is time-series associative and complex, demonstrating highdimensional nonlinearity [2].

Inspired by the above biological theories, we view the process of video grounding as the two-stage cross-modal semantic aggregation, beginning with the global feature interactions of video and language in text or audio modality, followed by a deeper video semantic purification based on the dynamic visual perception of the video, and thus design a unified static and dynamic framework for both NLVG and SLVG tsaks. For the static stage, static multimodal information will be comprehensively handled based on the language and video features and semantic connections between them are learned. For the dynamic stage, we further consider the aforementioned three characteristics of visual perception transmission, and integrate the key ideas of them into our model design. Specifically, as the example shown in Fig. 2, we first comprehensively communicate multiple queries and video clips to obtain contextual information for

the current query (e.g., Q4) and associate different queries to understand video scenes (e.g., query Q2 supplements query Q4 with more contextual information, in terms of semantics). This video-query understanding process is deemed as a static global interaction. Then we design a visual perception network to imitate dynamic context information transmission in the video with a dynamic filter generation network. We build a sparely connected relationship (blue arrow in Fig. 2) between video clips to reflect "Short-term Effect" (e.g., the video frames in the two dashed boxes closest to the solid blue box have the greatest impact on the current solid box frame, in terms of temporal direction and action continuity), and collect "Evidence (Relevance) Clues" (e.g., the orange and green video clips in the dashed boxes contain the cause and course of the whole video event, providing the time and semantic clues for current query sub-event) from these neighbor clips (blue dashed box in Fig. 2) by conducting a high-dimensional temporal Gaussian filtering convolution (in Section 3.3, imitating visual Perception Complexity).

Technically, existing methods primarily focus on solving a certain methodological aspect of Temporal Video Grounding tasks, such as learning self-modality language and video representation [5], [6], multimodal fusion [7], [8], cross-modal interaction [9], [10], candidate generation of proposals [11], [12], proposal-based cross-modal matching [13], [14], target moment boundary regression [15], [16], etc. Most current methods prefer to unilaterally consider the static feature interactions by employing the attention computation [5], [7], [15], [17]–[21] or graph convolution [9], [10], [16], [22] and relation computation [6], [11]–[14], [23]–[27] to associate the query and related video clips, rather than comprehensively expressing both static and dynamic visual perception simultaneously. Our work actually proposes a new paradigm for a two-stage unified staticdynamic semantic complementary new architecture.

In this paper, we propose a novel **Unified Static and Dynamic Networks (UniSDNet)** for both NLVG and SLVG. The overview of UniSDNet is shown in Fig. 3. Specifically, **for static modeling**, we propose a Static Semantic Supplement Network (S<sup>3</sup>Net), which contains a purely multilayer perceptron within the residual structure (ResMLP) and serves as a static multimodal feature aggregator to capture the association between queries and associate queries with

video clips. Unlike the traditional transformer attention [\[28\]](#page-16-9) network, this is a non-attention architecture that constitutes an efficient feedforward and facilitates data training for easy optimization of model performance-complexity tradeoffs (in Section [3.2\)](#page-3-0). **For the dynamic modeling**, we design a Dynamic Temporal Filtering Network (DTFNet) based on a Gaussian filtering GCN architecture to capture more useful contextual information in the video sequence (in Section [3.3\)](#page-4-0). We firstly construct a diffusely connected video clip graph to reflect the "*short-term effect*" relationship between video clip nodes. Then we redesign the aggregation of messages from neighboring nodes of the graph network by innovatively introducing the joint clue of the relative temporal distance r between the nodes and the relevance weight of the node a for measuring *relevance between nodes*. We employ the multi-kernel Temporal Gaussian Filter to extend the joint clue to high-dimensional space, and by performing highdimensional Gaussian filtering convolution operations on neighbor nodes, we imitate *visual perception complexity* and model fine-grained context correlations of video clips.

Notably, our proposed UniSDNet method shows encouraging performance and high inference efficiency in both NLVG and SLVG tasks, as shown in Section [5.4.](#page-11-0) Particularly, our model achieves higher efficiency (as shown in Fig. [7](#page-12-0) and Table [8\)](#page-12-1). For example, our proposed UniSDNet-M achieves 10. 31% performance gain on the R@1, IoU@0.5 metric while being 1.56× faster than multi-query training SOTA methods PTRM [\[14\]](#page-16-1) and MMN [\[25\]](#page-16-10), and, notably, the static and dynamic modules of UniSDNet-M are parameterized only by 0.53M and 0.68M (Table [3\)](#page-8-0), respectively.

Our main contributions are summarized as follows:

- We make a new attempt in solving video grounding tasks from the perspective of visual perception biology and propose a Unified Static and Dynamic Networks (UniSDNet), where the static module is a fully interactive ResMLP network that provides a global crossmodal environment for multiple queries and the video, and a Dynamic Temporal Filter Network (DTFNet) learns the fine context of the video with query attached.
- In dynamic network DTFNet, we innovatively integrate dynamic visual perception transmission biology mechanisms into the node message aggregation process of the graph network, including a newly proposed joint clue of relative temporal distance r and the node relevance weight a, and a multi-kernel Temporal Gaussian Filtering approach.
- In order to facilitate the research about the spoken language video grounding, we collect the new Charades-STA Speech and TACoS Speech datasets with diverse speakers.
- We conduct experiments on three public datasets for NLVG and one public dataset and two new datasets for SLVG, and verify the effectiveness of the proposed method. The SOTA performance on NLVG and SLVG tasks demonstrates the generalization of our model.

# **2 RELATED WORKS**

Temporal Video Grounding (TVG) includes Natural Language Video Grounding (NLVG) and Spoken Language Video Grounding (SLVG). NLVG uses text to locate video moments, while SLVG relies on spoken language. NLVG is widely studied due to advancements in natural language processing, with most existing works focus on it [\[9\]](#page-15-8)–[\[14\]](#page-16-1), [\[16\]](#page-16-3), [\[20\]](#page-16-11)–[\[22\]](#page-16-6), [\[24\]](#page-16-12)–[\[27\]](#page-16-8), [\[29\]](#page-16-13), [\[30\]](#page-16-14). SLVG, on the other hand, has gained attention recently due to its flexible speech-based querying. However, NLVG methods cannot be directly applied to SLVG without performance loss [\[5\]](#page-15-4), [\[31\]](#page-16-15), and few works address SLVG, leaving room for improvement. In this work, we consider both NLVG and SLVG tasks.

# **2.1 Natural Language Video Grounding (NLVG)**

Generally, existing popular methods for solving *NLVG* can be categorized into two main approaches: proposal-free [\[5\]](#page-15-4)– [\[8\]](#page-15-7), [\[15\]](#page-16-2), [\[17\]](#page-16-4)–[\[19\]](#page-16-16), [\[23\]](#page-16-7), [\[32\]](#page-16-17) and proposal-based [\[9\]](#page-15-8)–[\[14\]](#page-16-1), [\[16\]](#page-16-3), [\[20\]](#page-16-11)–[\[22\]](#page-16-6), [\[24\]](#page-16-12)–[\[27\]](#page-16-8), [\[29\]](#page-16-13), [\[30\]](#page-16-14), [\[33\]](#page-16-18) methods, with detailed comparative methods listed in Section [5.2.](#page-8-1) *1) Proposalfree* methods directly regress the target temporal span based on multimodal features. These proposal-free methods are mainly often divided into two main categories: Attentionbased models [\[7\]](#page-15-6), [\[17\]](#page-16-4)–[\[19\]](#page-16-16), [\[34\]](#page-16-19) and Transformer-based models [\[35\]](#page-16-20)–[\[39\]](#page-16-21). *2) Proposal-based methods* use a two-stage strategy of "generate and rank". First, they generate video moment proposals, and then rank them to obtain the best match. Herein, 2D-TAN [\[11\]](#page-15-10) is the first solution depositing possible candidate proposals via a 2D temporal map for temporal grounding and MMN [\[25\]](#page-16-10) further optimizes it for NLVG by introducing metric learning to align language and video modalities. Because of the elegance of 2D-TAN, we incorporate the concept of 2D temporal map modeling into our model, buffering the possible candidate clues. Our approach is a proposal-based architecture method. Otherwise, some proposal-based methods also focus on using Attention-based [\[8\]](#page-15-7), [\[13\]](#page-16-0), [\[23\]](#page-16-7), [\[24\]](#page-16-12), [\[29\]](#page-16-13), [\[30\]](#page-16-14) and Transformer-based [\[6\]](#page-15-5), [\[20\]](#page-16-11) architectures to address textvideo interaction and modal semantic extraction in NLVG tasks. Additionally, some approaches utilize Graph-based architectures [\[9\]](#page-15-8), [\[10\]](#page-15-9), [\[16\]](#page-16-3), [\[22\]](#page-16-6) for modeling static interactions between video clips. Although existing NLVG methods have made significant strides in video grounding, but they rely on single, static architectures [\[6\]](#page-15-5), [\[8\]](#page-15-7)–[\[10\]](#page-15-9), [\[13\]](#page-16-0), [\[16\]](#page-16-3), [\[20\]](#page-16-11), [\[22\]](#page-16-6)–[\[24\]](#page-16-12), [\[29\]](#page-16-13), [\[30\]](#page-16-14), limiting their ability to capture dynamic interactions as the video progresses.

No matter what, regardless of proposal-free or proposalbased manner, previous methods primarily emphasize feature learning with cross-modal attention [\[7\]](#page-15-6), [\[12\]](#page-15-11), [\[15\]](#page-16-2)–[\[20\]](#page-16-11), multi-level feature fusion [\[14\]](#page-16-1), [\[23\]](#page-16-7), relational computation [\[11\]](#page-15-10), [\[13\]](#page-16-0), [\[24\]](#page-16-12)–[\[26\]](#page-16-22), *etc*.; all the works are conducted *in a relatively static global perceptual mechanism mode*. Additionally, more and more methods are dedicated to capturing *the dynamics of the video*. On one side, temporal feature modeling are studied, such as using RNN to learn the temporal video relationship [\[34\]](#page-16-19), [\[40\]](#page-16-23) and conditional video feature manipulation [\[27\]](#page-16-8). On the other side, graph methods are explored for relational learning. For instance, CSMGAN [\[9\]](#page-15-8) integrates RNN for video temporal capture followed by fullconnected graph for cross-modal interaction. RaNet [\[22\]](#page-16-6) and CRaNet [\[10\]](#page-15-9) initially utilize the GC-NeXt [\[41\]](#page-16-24) to aggregate the temporal and semantic context of the video, and then a specially designed semantic graph network is used for crossmodal relational modeling. The current graph models [\[9\]](#page-15-8),

[10], [16], [22] with Graph Attention Networks (GAN) and Graph Convolutional Networks (GCN) are prominent for modeling static interactions between video clips. They overemphasize the correlation between video clip nodes but ignore the intrinsic high-dimensional time-series nature of the video In this work, we examine both static feature interactions and dynamic video representation in a unified video grounding framework, considering them in light of the motivations behind human visual perception. In the effort to achieve this, we design a lightweight ResMLP network for static semantic complements and exploit the relational learning in a video clip graph. Especially, we fresh sparse masking strategy in a 2D temporal map to build a diffusive connected video clip graph with dynamic Temporal Gaussian filtering for video grounding. Extensive experiments in Section 5 prove that this artifice is available for both NLVG and SLVG tasks. Such an integrated approach also offers broader applicability across both NLVG and SLVG tasks.

# 2.2 Spoken Language Video Grounding (SLVG)

To the best of our knowledge, the only available SLVG works at present are VGCL [5] and SIL [31], both of them have been assessed using the ActivityNet Speech dataset that has collected in VGCL's work. The VGCL proposes a proposal-free method that utilizes CPC [42] as the audio decoder and transformer encoder as the video encoder to guide audio decoding with the curriculum learning. The SIL proposes the acoustic-semantic pre-training to improve spoken language understanding and the acousticvisual contrastive learning to maximize acoustic-visual mutual information. VGCL firstly explore whether the virgin speech rather than text language can highlight relevant moments in unconstrained videos and propose the SLVG task. Compared to NLVG, the challenge of SLVG lies in the discretization of speech semantics and the audio-video interaction. The new task demonstrates that text annotations are not necessary to pilot the machine to understand video. Recently, with the development of audio pre-training, a breakthrough has been made in the discretization feature representation of speech [43]–[45].In this work, we focus on the audio-video interaction challenge of SLVG through the proposed UniSDNet. More importantly, to facilitate the research of SLVG, we collect two new audio description datasets named Charades-STA Speech and TACoS Speech that originate from the NLVG datasets of Charades-STA [3] and TACoS [46]. For more details, please refer to Section 4.

# 3 Proposed Framework

# 3.1 Task Definition & Framework Overview

The goal of the NLVG (natural language video grounding) and SLVG (spoken language video grounding) is to predict the temporal boundary  $(t^s,t^e)$  of the specific moment in the video in response to a given query in text or audio modality. Denote the input video as  $\mathcal{V}=\{v_i\}_{i=1}^T\in\mathbb{R}^{T\times d^v}$ , where  $d^v$  and T are the feature dimension and total number of video clips, respectively. Each video has an annotation set of  $\{\mathcal{Q},\mathcal{M}\}$ , in which  $\mathcal{Q}$  is a M-query set in the text or audio modality and  $\mathcal{M}$  represents the corresponding video moments of the

queried events, denoted as  $\mathcal{Q}=\{q_i\}_{i=1}^M\in\mathbb{R}^{M\times d^q}$ , and  $\mathcal{M}=\{(t_i^s,t_i^e)\}_{i=1}^M$ , where  $(t_i^s,t_i^e)$  represents the starting and ending timestamps of the m-th query,  $d^q$  is the dimension of query feature, and M is the query number.

In this paper, we present a unified framework, named Unified Static and Dynamic Network (UniSDNet), for both NLVG and SLVG tasks, focusing on video content understanding in the multimodal environment. Fig. 3 illustrates the overview of our proposed architecture. Our UniSDNet comprises the *Static Semantic Supplement Network* (S<sup>3</sup>Net) and Dynamic Temporal Filtering Network (DTFNet). It adopts a two-stage information aggregation strategy, beginning with a global interaction mode to perceive all multimodal information, followed by a graph filter to purify key visual information. Finally, we extract enhanced semantic features of the video clip for high-quality 2D video moment proposals generation. In the following subsections, we introduce the core modules, S<sup>3</sup>Net (Section 3.2), DTFNet (Section 3.3), and 2D proposal generation (Section 3.4) of our proposed unified framework.

# <span id="page-3-0"></span>3.2 Static Semantic Supplement Network

The static network S<sup>3</sup>Net is inspired by the concept of the global neuronal workspace (GNW) [1] in the human brain, which aggregates the multimodal information in the first stage of visual event recognition. In terms of the functionality of the static network for video understanding, it provides more video descriptions information and significantly fills the gap between vision-language modalities, aiding in understanding video content.

Technically, the S<sup>3</sup>Net can be seen as a fully interactive and associative process involving static queries and video features. From the aforementioned perspective, we have designed the static semantic supplement network S<sup>3</sup>Net (as shown in Fig. 3) by integrating the MLP into the residual structure (ResMLP). The incorporation of a multilayer perceptron within ResMLP enables the fulfillment of static feature's linear interaction requirement for achieving multimodal information aggregation. This setup constitutes an efficient feedforward network that facilitates data training and allows for easy optimization of model performance/complexity trade-offs. Additionally, employing a linear layer offers the advantage of having long-range filters at each layer [47].

Before feature interaction, we utilize pre-trained models (C3D [48], GloVe [49], Data2vec [50], etc.) to extract the original video and query features, which are then linearly converted into a unified feature space. This yields video and query features  $F_{\mathcal{V}} \in \mathbb{R}^{T \times d}$  and  $F_{\mathcal{Q}} \in \mathbb{R}^{M \times d}$ , respectively, with  $F_{\mathcal{V}\mathcal{Q}} = [F_{\mathcal{V}}||F_{\mathcal{Q}}] \in \mathbb{R}^{(T+M) \times d}$ . Inspired by the existing multi-modal Transformers work [51]–[53], we independently add position embeddings for video and queries, to distinguish modality-specific information. More ablation studies on adding position embeddings are discussed in Appendix B.2. Specifically, we incorporate the position embedding [28]  $P_{\mathcal{V}} \in \mathbb{R}^{T \times d}$  for video feature and  $P_{\mathcal{Q}} \in \mathbb{R}^{M \times d}$  for query feature, and concatenate them into  $P_{\mathcal{V}\mathcal{Q}} = [P_{\mathcal{V}}||P_{\mathcal{Q}}] \in \mathbb{R}^{(T+M) \times d}$ . We use MLPBlock, which is a combination of a LayerNorm layer, a Linear layer, a

![](_page_4_Figure_2.jpeg)

<span id="page-4-1"></span>Fig. 3. The architecture of the Unified Static and Dynamic Network (UniSDNet). It mainly consists of static and dynamic networks: Static Semantic Supplement Network (S³Net) and Dynamic Temporal Filtering Network (DTFNet). S³Net concatenates video clips and multiple queries into a sequence and encodes them through a lightweight single-stream ResMLP network. DTFNet is a 2-layer graph network with a dynamic Gaussian filtering convolution mechanism, which is designed to control message passing between nodes by considering temporal distance and semantic relevance as the Gaussian filtering clues when updating node features. The role of 2D temporal map is to retain possible candidate proposals and represent them by aggregating the features of each proposal moment. Finally, we perform semantic matching between the queries and proposals and rank the best ones as the predictions.

ReLU activation layer, and a Linear layer, to obtain the static interactive video clip features  $\hat{F}_{\mathcal{V}}$  and query features  $\hat{F}_{\mathcal{Q}}$ :

<span id="page-4-2"></span>
$$\tilde{F}_{\mathcal{V}\mathcal{Q}} = F_{\mathcal{V}\mathcal{Q}} + \text{LayerNorm}(F_{\mathcal{V}\mathcal{Q}}) + P_{\mathcal{V}\mathcal{Q}}, 
\hat{F}_{\mathcal{V}\mathcal{Q}} = \text{LayerNorm}(\tilde{F}_{\mathcal{V}\mathcal{Q}} + \text{MLPBlock}(\tilde{F}_{\mathcal{V}\mathcal{Q}})), 
\hat{F}_{\mathcal{V}} = \hat{F}_{\mathcal{V}\mathcal{Q}}[1:T;:] \in \mathbb{R}^{T \times d}, 
\hat{F}_{\mathcal{Q}} = \hat{F}_{\mathcal{V}\mathcal{Q}}[T+1:T+M;:] \in \mathbb{R}^{M \times d}.$$
(1)

Note that UniSDNet can accommodate any number of queries as inputs during training. Proving a single query input is the traditional training mode for NLVG and SLVG tasks. When multiple queries are fed as inputs, there are interactions among the queries, within the video (across multiple video clips), and between the queries and video. This approach enables the learning of self-modal and cross-modal semantic associations between video and queries without semantic constraints, allowing the model to leverage the complementary effects among multiple queries related to the same video content. The semantics, either in a single query or multiple queries, can offer more comprehensive semantic supplementation for a effective and efficient understanding of the entire video content.

## <span id="page-4-0"></span>3.3 Dynamic Temporal Filtering Network

The second stage (DTFNet) of UniSDNet dynamically filters out important video content, inspired by the dynamic visual perception mechanism observed in human activity [2], as introduced in Section 1. We imitate the three characteristics of this visual perception mechanism by learning a video graph network. We restate the key points of these three characteristics here: 1) Short-term Effect: nearby perceptions strongly affect current perceptions; 2) Auxiliary Evidence (Relevance) Cues: semantically relevant scenes in the video provide auxiliary time and semantic cues; 3) Perception Complexity: the perception process is time-series associative and complex, demonstrating high-dimensional nonlinearity [2]. These characteristics play a crucial role in assisting individuals in locating queried events within the video, which have been explained in Fig. 1 and Fig. 2. Graph neural networks have shown efficacy in facilitating intricate information transmission between nodes [54]. To emulate the human visual perception process, we introduce a new message passing approach between video clip nodes and propose a Dynamic Temporal Filtering Graph Network (DTFNet as depicted in Figs. 3 and 4).

To imitate the Short-term Effect, we construct a diffusive connected graph based on the 2D temporal video clip map (please see "Graph Construction" below). For discovering Auxiliary Evidence Cues, we integrate the message passed from each node's neighbors by measuring the relative temporal distance and the semantic relevance in the graph (as explained in the filter clue introduced in "How to construct  $\mathcal{F}_{filter}$ ?" below). Finally, we employ a multi-kernel Gaussian filter-generator to expand the auxiliary evidence clues to a high-dimensional space, simulating the complex visual perception capabilities of humans (explained in the filter function in "How to construct  $\mathcal{F}_{filter}$ ?" below).

# 3.3.1 Graph Construction

Let us denote a video graph  $\mathcal{G} = (\mathcal{G}_{\mathcal{V}}, \mathcal{G}_{\mathcal{E}})$  to represent the relation in the video V. In the graph G, node  $v_i$  is the i-th video clip and edge  $e_{ij}{\sim}(v_i,v_j)\in\mathcal{G}_{\mathcal{E}}$  represents whether  $v_i$  is  $v_i$ 's connective neighbor. We obtain  $\hat{F}_{\mathcal{V}}$  from the S<sup>3</sup>Net (in Eq. 1) and take it as the initialization of clip nodes in the graph, namely the initial node embedding of the graph is set to  $\mathcal{G}_{\mathcal{V}}^{(0)} = \hat{F}_{\mathcal{V}} \in \mathbb{R}^{T \times d}$ . For the graph edge set  $\mathcal{G}_{\mathcal{E}}$ , we utilize a diffusive connecting strategy [11] based on the temporal distance of two nodes, to determine the edge status  $e_{ij}$ . The temporal distance between node  $v_j$  and node  $v_i$  is defined as  $r_{ij} = ||j-i||$ , setting the hyperparameter k, for the current node  $v_i$ , we define the **short distance** as  $0 \le r_{ij} < k$  and the **long distance** as  $r_{ij} \geq k$ . Based on these two distances, there are two types of edge connections: (1) Dense connectivity for nodes with a short distance: when  $0 \le r_{ij} < k$ , we densely connect two nodes, i.e.,  $\mathcal{G}_{\mathcal{E}_{short}} = \{e_{ij} \mid 0 \leq r_{ij} < k\}$ . (2) Sparse connectivity for nodes with a long distance: when  $r_{ij} \geq k$ , we connect them at exponentially spaced intervals, *i.e.*, the following conditions should be met when  $e_{ij}$  exists:

<span id="page-4-3"></span>
$$\mathcal{G}_{\mathcal{E}_{long}} = \{e_{ij}\}, \quad s.t. \begin{cases} i \mod 2^{n+1} = 0 \\ r_{ij} \mod (2^n k) = 0 \\ 2^n k \le r_{ij} < 2^{n+1} k \end{cases}$$
 (2)

where  $n=(0,1,\cdots,\lceil log_2\frac{T}{k}\rceil-1).$   $\lceil \cdot \rceil$  is the ceil function. we obtain a sparsely connected edge set  $\mathcal{G}_{\mathcal{E}}=\mathcal{G}_{\mathcal{E}_{short}}\cup \mathcal{G}_{\mathcal{E}_{long}}.$  Please note that we model forward along the timeline, resulting in that the edge set  $\mathcal{G}_{\mathcal{E}}$  is reflected as a upper triangular adjacency matrix. For more explanation and discussion on the edge construction, please see Appendix B.1.

# 3.3.2 Temporal Filtering Graph Learning

We build L-layer graph filtering convolutions in our implementation. During training, the node embedding  $\mathcal{G}_{\mathcal{V}}^{(l)} = \{v_i^{(l)}\}_{i=1}^T$  is optimized at each graph layer,  $1 \leq l \leq L$ . In this part, we introduce a Gaussian Radial **Filter-Generator**  $\mathcal{F}_{filter}$  shown in Fig. 4 to imitate the dynamic flashback process of video for visual perception. There are two core technical difficulties to be resolved below.

How to construct  $\mathcal{F}_{filter}$ ? Since visual perception is transmitted along the timeline, we consider the relative time interval between nodes as the primary clue. Additionally, similar scenes work appropriately on the comprehension of current scene, so we take into account the semantic relevance between graph nodes as auxiliary clue. Specifically, we compute the two clues of the relative temporal distance  $r_{ij}$  of node  $v_j$  and node  $v_i$  ( $r_{ij} = ||j - i||$ ) and the relevance weight  $a_{ij}$  of this two-node pair measured by the  $cos(\cdot)$  similarity function. We combine them as the joint clue  $d_{ij} = (1 - a_{ij}) \cdot r_{ij}$ . To mimic the dynamic nature, continuity (high dimensionality), and non-linearity (complexity) of visual perception transmission, we use the filter-generating network to dynamically generate highdimensional filter operators that control message passing between nodes, rather than directly applying the simple discrete scalar  $d_{ij}$  to compute message aggregation weights, which is insufficient to express these properties. The filtergenerator (as illustrated in Fig. 4) is given in the form of  $\mathcal{F}_{filter}(d_{ij}): \mathbb{R} \to \mathbb{R}^h$ . Gaussian function has already been exploited in deep neural networks, such as Gaussian kernel grouping [55], learnable Gaussian fucntion [21], Gaussian radial basis function [56] that have been proven to be effective in simulating high-dimensional nonlinear information in various scenes. Inspired by these works, we adopt multikernel Gaussian radial basis to extend the influence of the clue  $d_{ij}$  into high-dimensional space, thereby reflecting the continuous complexity of the perception process. Specifically, we design a temporal Gaussian basis function to build the  $\mathcal{F}_{filter}$  and expand the joint clue  $d_{ij}$  to a high dimension vector  $f_{ij} \in \mathbb{R}^h$  in message passing process. We express the form of a single kernel temporal Gaussian as  $\phi(d_{ij},z)=\exp(-\gamma(d_{ij}-z)^2)$ , where  $\gamma$  is a Gaussian coefficient that reflects the amplitude of Gaussian kernel function and controls the gradient descent speed of the function value, z is a bias we added to avoid a plateau at the beginning of training due to the highly correlated Gaussian filters. Furthermore, we expand it to multiple-kernel Gaussian function  $\Phi(d_{ij}, Z) = \exp(-\gamma (d_{ij} - z_k)^2), k \in [1, h]$  to fully represent the complex nonlinear of video perception. Based on the single kernel term, we construct h kernel functions, more studies on the settings of  $(\gamma, z, h)$  are in the Section 5.5.3. The way we generate the filter  $f_{ij}$  of node  $v_i$  to node  $v_i$  through the multi-kernel Gaussian filer is:

<span id="page-5-3"></span>
$$f_{ij} = \mathcal{F}_{filter}(d_{ij}) = (\phi_1(d_{ij}), \phi_2(d_{ij}), \cdots, \phi_h(d_{ij})).$$
 (3)

![](_page_5_Picture_7.jpeg)

Fig. 4. The process of (a) node message aggregation in the Dynamic Temporal Filtering graph and (b) dynamic filter-generator Filter, which is built based on the joint clue of relevance weight  $a_{ij}$  and relative temporal distance  $r_{ij}$  between two nodes. This joint clue is expanded into high dimensions representation through a multi-kernel Gaussian radial basis function.

<span id="page-5-1"></span>How to update the nodes in graph  $\mathcal{G}_{\mathcal{V}}$ ? In the stage of message passing on l-th layer, we update each node representation by aggregating its neighbor node message to obtain  $\mathcal{G}_{\mathcal{V}}^{(l)}$ . For node  $v_i$ , its neighbor set is  $\{v_j \mid v_j \in \mathcal{N}(v_i)\}$  corresponding to the adjacency map  $\mathcal{G}_{\mathcal{E}}$ . With the multi-kernel Gaussian filter  $f_{ij}$ , the update of node feature  $v_i$  on l-th graph layer is described as:

$$v_i^{(l)} = \sigma \left( \sum_{j \in \mathcal{N}(v_i)} \text{FFN}_1(f_{ij}) \odot \text{FFN}_0(v_j^{(l-1)}) \right), \quad (4)$$

where  $\odot$  represents element-wise multiplication and  $\sigma$  is a ReLU activation function. So far, a video graph with spatiotemporal context correlation of video clips is learned.

# <span id="page-5-0"></span>3.4 2D Proposal Generation

**Proposal Generation.** After obtaining the updated video clip features from the above DTFNet module, we implement the moment sampling [11] on the features to generate a 2D temporary proposal map  $M^{\text{2D}} \in \mathbb{R}^{T \times T \times d}$  that indicates all candidate moments (2D Proposal Generation in Fig. 3). The element  $m_{ij}$  in the map  $M^{\text{2D}}$  indicates the candidate proposal  $[v_i, \cdots, v_j]$ . For each moment  $m_{ij}$ , we consider all the clips in the moment interval and the boundary feature is further added to the moment representation (Eq. 5). Afterwards, a stack of 2D convolutions is used to encode the moment feature. For the detailed ablation studies about the moment sampling strategy, please refer to Section 5.5.4.

<span id="page-5-2"></span>
$$m_{ij} = \text{MaxPool}(v_i^L, v_{i+1}^L, \cdots, v_j^L) + v_i^L + v_j^L \in \mathbb{R}^d,$$

$$M^{\text{2D}} = CNN(m_{ij}) \in \mathbb{R}^{T \cdot T \cdot d}.$$
(5)

**Modality Alignment Measurement.** We calculate the relevance of each {query, moment proposal} pair according to the semantic similarity, generating new 2D moment score maps for the M-queries. Specifically, a  $1 \times 1$  convolution and an FFN are respectively used to project the moment feature  $M^{2D}$  and the query feature  $\hat{F}_{\mathcal{Q}}$  into the same dimensional vectors  $S^{\mathcal{M}} \in \mathbb{R}^{T \times T \times d}$  and  $S^{\mathcal{Q}} \in \mathbb{R}^{M \times d}$ . Following MMN [25], we use cosine similarity to measure the semantic similarity between queries and moment proposals, it is

defined as  $\tilde{S} = \operatorname{CoSine}(S^{\mathcal{M}}, S^{\mathcal{Q}})$ . Thereby, M similarity score maps for input M queries are computed:

<span id="page-6-1"></span>
$$S^{\mathcal{M}} = \operatorname{Norm}(\operatorname{Conv2d}_{1\times 1}(M^{2D})) \in \mathbb{R}^{T \cdot T \cdot d},$$

$$S^{\mathcal{Q}} = \operatorname{Norm}(\operatorname{FNN}(\hat{F}_{\mathcal{Q}})) \in \mathbb{R}^{M \cdot d},$$

$$\tilde{S} = \operatorname{CoSine}(S^{\mathcal{M}}, S^{\mathcal{Q}}) = \{\tilde{s}^{1}, \tilde{s}^{2}, \dots, \tilde{s}^{M}\} \in \mathbb{R}^{(T \times T) \cdot M}.$$

$$(6)$$

where for each query  $q_i$ , the proposal corresponding to the maximum value in  $\tilde{s}^i$  is selected as the best match for the given query  $q_i$ . There are some other semantic similarity functions for measuring modal alignment. Please refer to Appendix B.3 for relevant ablation study.

# 3.5 Training and Inference

Our UniSDNet is proposal-based, thereby we optimize the score map  $\tilde{S}$  with IoU regression loss and contrastive learning loss. Following 2D-TAN [11], we compute the groundtruth IoU Map  $\text{IoU}^{\text{GT}} = \{IoU^i\}_{i=1}^{M} \in \mathbb{R}^{(T \times T) \cdot M}$  corresponding to queries. That is, we compute the value of intersection over union between each candidate moment and the target moment  $(t_{gt}^s, t_{gt}^e)$ , and scale this value to (0,1), with total N moment scores. The IoU prediction loss is

$$\mathcal{L}_{iou} = \frac{1}{N} \sum_{i=1}^{N} (iou_i \cdot \log(y_i) + (1 - iou_i) \cdot \log(1 - y_i)), \quad (7)$$

where  $iou_i$  is the groundtruth from  $IoU^{GT}$ , and  $y_i$  is the predicted IoU value from  $\tilde{S}$  in Eq. 6.

Besides, we adopt contrastive learning [25] as an auxiliary constraint, to fully utilize the positive and negative samples between queries and moments to provide more supervised signals. The noise contrastive estimation [42] is used to estimate two conditional distributions p(q|m) and p(m|q). The former represents the probability that a query q matches the video moment m when giving m, and the latter represents the probability that a video moment m matches the query q when giving q.

$$\mathcal{L}_{contra} = -\left(\sum_{q \in Q^B} \log p(m_q|q) + \sum_{m \in M^B} \log p(q_m|m)\right), \quad (8)$$

where  $Q^B$  and  $M^B$  are the sets of queries and moments in a training batch.  $m_q \in \{m_q^+, m_q^-\}$ ,  $m_q^+$  is the moment matched to query q (solo positive sample) and  $m_q^-$  denotes the moment unmatched to q in the training batch (multiple negative samples). The definition of  $q_m \in \{q_m^+, q_m^-\}$  for moment m is similar to that of  $m_q \in \{m_q^+, m_q^-\}$ . The objective of contrastive learning is to guide the representation learning of video and queries and effectively capture mutual matching information between modalities. As a result, the total loss is  $\mathcal{L} = \mathcal{L}_{iou} + \mathcal{L}_{contra}$ . Non-Maximum Suppression (NMS) threshold is 0.5 during inference.

# <span id="page-6-0"></span>4 DATASETS

To validate the effectiveness of our proposed unified static and dynamic framework for both NLVG and SLVG tasks, we conduct experiments on the popular video grounding benchmarks. There are three classic benchmarks for NLVG task, *i.e.*, *ActivityNet Captions* [57], *Charades-STA* [3], and *TACoS* [46] datasets. For SLVG task, only the *ActivityNet* 

Speech dataset [5] is publicly available, an extension of ActivityNet Captions dataset used for NLVG task. To accelerate SLVG development, we collect **two new Speech datasets**: Charades-STA Speech and TACoS Speech based on the original Charades-STA and TACoS datasets.

# 4.1 Existing Datasets for NLVG Task and SLVG Task

The dataset benchmarks used for the NLVG task consist of the untrimmed video and its annotations (text sentence descriptions and video moment pairs). (1) ActivityNet Captions [57] dataset includes 19,209 videos sourced from YouTube's open domain collection, initially proposed by [57] for dense video captioning task and later utilized for video grounding task. The dataset is divided according to the partitioning scheme in [7], [11]; it comprises 37,417, 17,505, and 17,031 sentence-moment pairs for training, validation, and testing, respectively. (2) Charades-STA [3] dataset consists of 9,848 relatively short indoor videos from Charades dataset [59] originally designed for action recognition and localization. It is extended by [3] to include language descriptions for the NLVG task, including 12,408 and 3,720 sentence-moment pairs for training and testing, respectively. (3) TACoS [46] dataset focuses on 127 activities within a kitchen, constructed based on the MPII-Compositive dataset [60]. Following the split outlined in [11], the dataset includes 10,146, 4,589, and 4,083 sentencemoment pairs for training, validation and testing, respectively. Compared to ActivityNet Captions and Charades-STA, the TACoS features longer and more annotated queries for each video, with an average of 286.59s and 130.53 per video in the training set.

Currently, there is only one dataset, *ActivityNet Speech* proposed by Xia *et al.* [5], publicly available for the SLVG task. The dataset is collected based on the ActivityNet Captions dataset [57], consisting of 37,417, 17,505, and 17,031 audio-moment pairs for training, validation, and testing (as the same split as in [57]), where audio is obtained by 58 volunteers (28 male and 30 female) reading the text fluently in a clean surrounding environment.

#### 4.2 New Collected Datasets for SLVG Task

In this work, we collected two new datasets to facilitate SLVG research. Unlike the *ActivityNet Speech* [5] with manual text-to-speech reading, we use *machine simulation* to synthesize audio subtitle datasets and release two **new** *Charades-STA Speech* and *TACoS Speech* datasets<sup>1</sup>. The considerations for adopting the machine simulation are:

- High-quality synthesised voice. Thanks to advancements in text-to-speech (TTS) technology [58], [61], TTS is capable of closely simulating the human voice, effectively capturing and expressing intricate voice characteristics, including speaking style and tone, and generating a high-quality synthesised voice.
- Diverse readers. We randomly select a "reader" from the CMU ARCTIC database<sup>2</sup> to "read" text sentences
- <span id="page-6-2"></span>1. Charades-STA Speech dataset is available at https://zenodo.org/records/8019213 and TACoS Speech dataset is available at https://zenodo.org/records/8022063
- <span id="page-6-3"></span>CMU ARCTIC database is available at http://www.festvox.org/ cmu\_arctic/

TABLE 1

Data statistics of three widely used datasets for NLVG task, ActivityNet Captions, Charades-STA and TACoS datasets.

<span id="page-7-4"></span>

| Datasets                  | Domain  | #      | ¥ Videos |       | #      | Sentence | s      | Av      | erage Ler | ıgth   | Average | Queries | per Video |
|---------------------------|---------|--------|----------|-------|--------|----------|--------|---------|-----------|--------|---------|---------|-----------|
| Datasets                  | Domain  | Train  | Val      | Test  | Train  | Val      | Test   | Video   | Words     | Moment | Train   | Val     | Test      |
| ActivityNet Captions [57] | Open    | 10,009 | 4,917    | 4,885 | 37,421 | 17,505   | 17,031 | 117.60s | 14.41     | 37.14s | 3.74    | 3.56    | 3.49      |
| Charades-STA [3]          | Indoors | 5,336  | -        | 1,334 | 12,408 | -        | 3,720  | 30.60s  | 7.22      | 8.09s  | 2.33    | -       | 2.79      |
| TACoS [46]                | Cooking | 75     | 27       | 25    | 9,790  | 4,436    | 4001   | 286.59s | 9.42      | 27.88s | 130.53  | 164.30  | 160.04    |

#### TABLE 2

<span id="page-7-1"></span>Data statistics of datasets for SLVG task. Charades-STA Speech\* and TACoS Speech\* are new datasets collected by us, using machine simulation [58] from CMU ARCTIC database, offering more diverse pronunciations than AcitivityNet Speech.

| Datasets               | Domain  | #      | Videos |       |        | # Audios |        | Av      | erage Ler | ngth   | Audio Source  |
|------------------------|---------|--------|--------|-------|--------|----------|--------|---------|-----------|--------|---------------|
| Datasets               | Domain  | Train  | Val    | Test  | Train  | Val      | Test   | Video   | Audio     | Moment | Audio Source  |
| ActivityNet Speech [5] | Open    | 10,009 | 4,917  | 4,885 | 37,421 | 17,505   | 17,031 | 117.60s | 6.22s     | 37.14s | 58 Volunteers |
| Charades-STA Speech*   | Indoors | 5,336  | -      | 1,334 | 12,408 | -        | 3,720  | 30.60s  | 2.33s     | 8.09s  | 3,869 Readers |
| TACoS Speech*          | Cooking | 75     | 27     | 25    | 9790   | 4436     | 4001   | 286.59s | 2.89s     | 27.88s | 126 Readers   |

![](_page_7_Figure_7.jpeg)

<span id="page-7-3"></span>Fig. 5. Statistics on the query number size of each video in training set for NLVG&SLVG datasets (1k=1,000). The datasets can be divided into three categories: large query size (TACoS & TACoS Speech, most sizes are 110), middle query size (ActivityNet Captions & ActivityNet Speech, most sizes are 3), and small query size (Charades-STA & Charades-STA Speech, most sizes are 1, and the query description is often ambiguous and semantically insufficient as the video is too short with mostly 30s duration for manually annotating events).

in Charades-STA and TACoS datasets. The database contains 7,931 vocal embeddings with different English pronunciation characteristics.

 Cost savings and high-quality annotation. With the strong ability of TTS technology to prevent errors like word mispronunciations, incoherent sentence delivery, and audio-text mismatches caused by manual annotation, the necessity for manual text reading, recording, and file annotation processes is mitigated. Machine emulation reduces the cost of manual annotation and avoids manual reading errors.

Based on the above considerations, we adopt the TTS technology "microsoft/speecht5\_tts" to collect the audio description of the text query with a random virtual "reader" in the CMU ARCTIC database to guarantee the diversity of voice, style, and tone. Compared to the *ActivityNet Speech* dataset, the *Charades-STA Speech* and *TACoS Speech* datasets we collected have more diverse pronunciations. The average of each speech recording is 2.33 seconds and 2.89 seconds in the *Charades-STA Speech* and *TACoS Speech* datasets, respectively. It is important to note that the partitioning of both the *Charades-STA Speech* and *TACoS Speech* datasets is consistent with their source datasets *Charades-STA* [3] and *TACoS* [46].

![](_page_7_Figure_13.jpeg)

<span id="page-7-2"></span>Fig. 6. Statistics on the query length (counted by word number) in training set for NLVG&SLVG datasets (1k=1000). The query length of the ActivityNet Captions dataset are generally long (mostly 14 words and mostly 6s per query), having more detailed descriptions compared to the other two datasets.

We have summarized the statistics of these two new datasets for SLVG task in Table 2.

# <span id="page-7-5"></span>4.3 Datasets Analysis

First of all, please note that the SLVG datasets are derived from the NLVG datasets, sharing the same video and query sentence. The main difference between them is the modality of query used: SLVG use audio-moment pairs, while NLVG use text-moment pairs. The datasets exhibit distinct characteristics in the following aspects: (1) Video *Duration*. The average video duration is counted in Table 2 with the datasets ActivityNet, Charades-STA, and TACoS of 117.60s, 30.60s, and 286.59s, respectively. The minimum video duration in the Charades-STA implies a stricter judgment of event boundaries than the other two datasets. (2) Query Length (counted by word number in a text sentence or audio duration). Generally, the longer the audio duration, the more words in the text annotation, and the richer the information provided by the query to describe the video. Notably, the ActivityNet Speech dataset has longer queries (mostly 14 words and mostly 6s per query as shown in Fig. 6), providing more detailed descriptions. (3) Query *Number*. Fig. 5 shows the distributions of the video's query numbers, the datasets can be divided into three categories: large (TACoS), medium (ActivityNet Captions), and small (Charades-STA). Particularly, the Charades-STA is minimal with at most 1 query per video, suggesting a potential limitation in description detail provided for the video.

<span id="page-7-0"></span><sup>3.</sup> Source code of Microsoft TTS technology is available at https://huggingface.co/microsoft/speecht5\_tts

TABLE 3

<span id="page-8-0"></span>The hyperparameter settings of UniSDNet framework for different NLVG&SLVG datasets with the specific pre-extracted video features. It is worth noting that the number of parameters in the static (S<sup>3</sup>Net) and dynamic (DTFNet) modules of UniSDNet is extremely small on all datasets.

| Datasets                   | #Clips | Static S <sup>3</sup> Net | Dynan   | nic DTFNet  | 2D      | Proposal Gen | eration     |                        | #Parar     | neters                  |
|----------------------------|--------|---------------------------|---------|-------------|---------|--------------|-------------|------------------------|------------|-------------------------|
|                            |        | Hidden size               | #Layers | Hidden size | #Layers | Kernel size  | Hidden size | S <sup>3</sup> Net 3.2 | DTFNet 3.3 | Proposal Generation 3.4 |
| ActivityNet Captions (C3D) | 64     | 1024                      | 2       | 256         | 4       | 9            | 512         | 0.53M                  | 0.68M      | 76.79M                  |
| Charades-STA (VGG)         | 16     | 1024                      | 2       | 512         | 3       | 5            | 512         | 1.05M                  | 2.68M      | 20.19M                  |
| Charades-STA (C3D)         | 16     | 1024                      | 2       | 512         | 3       | 5            | 512         | 1.05M                  | 2.68M      | 20.19M                  |
| Charades-STA (I3D)         | 64     | 1024                      | 2       | 256         | 2       | 17           | 512         | 0.53M                  | 0.68M      | 113.91M                 |
| TACoS (C3D)                | 128    | 1024                      | 2       | 256         | 3       | 5            | 512         | 0.53M                  | 0.68M      | 16.65M                  |

#### <span id="page-8-2"></span>5 EXPERIMENTS

# <span id="page-8-3"></span>5.1 Experimental setup

**Evaluation Metrics.** Following the convention in the video grounding and video moment retrieval tasks [3], [7], [19], we compute the "R@h, IoU@u" and "mIoU" for performance evaluation of both NLVG and SLVG tasks. The metric "R@h, IoU@u" denotes the percentage of samples that have at least one correct answer in the top-h choices, where the criterion for correctness is that the moment IoU between the predicted result and the groundtruth is greater than a threshold u. Mathematically, "R@h, IoU@u" is defined as:

$$R@h, IoU@u = \frac{1}{N_q} \sum_{i=1}^{N_q} r(h, u, q_i), \tag{9}$$

where  $N_q$  denotes the number of queries in the test set and  $q_i$  represents the i-th query. In the top h predicted moments of query  $q_i$ , if the moment IoU between prediction and groundtruth is larger than u,  $r(h,u,q_i)$  equals 1; otherwise,  $r(h,u,q_i)$ =0. Specifically, we set  $h \in \{1,5\}$  and  $u \in \{0.3,0.5,0.7\}$ . Also, we use mIoU, the average IoU between the prediction and groundtruth across the test set, as an indicator to compare overall performance:

$$mIoU = \frac{1}{N_q} \sum_{i=1}^{N_q} IoU_i, \qquad (10)$$

where  $N_q$  is the total number of queries, and  $IoU_i$  is the IoU value of the predicted moment for the i-th query.

Hyperparameter Settings. Table 3 shows hyperparameter settings of UniSDNet. For data preparation, we evenly sample 64 and 128 video clips for ActivityNet Captions dataset with C3D features, and 16, 16, and 64 video clips for the Charades-STA dataset with VGG, C3D, and I3D features, respectively. In the static module, we conduct two ResMLP blocks (N=2), and feature hidden size is set to 1024. In the dynamic module, DTFNet has two graph layers, Based on the average clips of target moments in training set, hyperparameter k in Eq. 2 – dividing value between short and long distances in video graph - is set to 16. More discussion and ablation studies of k are in Appendix B.1. We empirically set hyperparameter  $\gamma$  to 10.0, Gaussian kernels number h to 50, and generate h biases with equal steps from 0 with step 0.1. For dynamic filter  $\mathcal{F}_{filter}$ , settings of convolution layers, kernel size, and hidden size for 2D proposal generation are listed in Table 3. Parameters size of S<sup>3</sup>Net (Section 3.2), DTFNet (Section 3.3) and proposal generation (Section 3.4) are also provided in Table 3.

**Implementation Details.** For a fair comparison, we utilize the same video features provided by 2D-TAN [11],

which includes 500-dim C3D feature [48] on ActivityNet Captions, 4096-dim VGG feature [64] on Charades-STA, and 500-dim C3D feature on TACoS from [9]. Besides, there are currently other popular C3D feature and I3D feature [65] available on Charades-STA, so we also use the 4096-dim C3D feature from [18] and 1024-dim I3D feature provided by [19]. Following previous work [25], we use the GloVe [49] and BERT [62] to extract textual feature. For the audio feature, we use the HuggingFace [66] implementation of Data2vec [50] with pre-trained model "facebook/data2vecaudio-base-960h" for SLVG. Specifically, we set the audio sampling rate to 16,000 Hz, and use the python audio standard library "librosa" to read the original audio and input it into the Data2vec model to obtain the audio sequence embedding. Additionally, we use LayerNorm and AvgPool operations to aggregate the entire audio representation. The feature dimensions of both text and audio are 768.

Training and Inference Settings. In this work, we delve into both single-query and multi-query training. For the M-query annotations  $\mathcal{Q}=\{q_i\}_{i=1}^M$  associated with video  $\mathcal{V}$ , we specify the number of queries fed into model training at a time to be m. When m=1, this corresponds to single query training, designated as UniSDNet-S. Conversely, for multi-query training, where m>1, specifically when m=M, all queries relating to video  $\mathcal{V}$  are simultaneously fed into the model, referred to as UniSDNet-M. It is important to underscore that during the inference phase, regardless of UniSDNet-S or UniSDNet-M, the evaluation process is a fair single query input that determines the prediction of a uniquely corresponding moment, consistent with the conventional settings of the NLVG & SLVG tasks [3], [7], [19].

We use the AdamW [67] to optimize the proposed model. For ActivityNet Captions and TACoS datasets, the learning rate and batch size are set to  $8\times 10^{-4}$  and 12, respectively. For Charades-STA dataset, we set the learning rate and batch size to  $1\times 10^{-4}$ , and 48, respectively. We train the model (whether UniSDNet-S or UniSDNet-M) with the upper-limit of 15 epochs on ActivityNet Captions and Charades-STA datasets and 200 epochs on TACoS. All experiments are conducted with a GeForce RTX 2080Ti GPU.

#### <span id="page-8-1"></span>5.2 Comparison with state-of-the-arts for NLVG Task

We compare our UniSDNet with the state-of-the-art methods for *NLVG* and divide them into two groups. **1) Proposal-free methods**: VSLNet [17], LGI [19], DRN [18], CP-Net [7], VSLNet-L [15], BPNet [23], VGCL [5], METML [6], MA3SRN [8]. **2) Proposal-based methods**: 2D-TAN [11], CSMGAN [9], MS-2D-TAN [12], MSAT [20], RaNet [22], I<sup>2</sup>N [24], FVMR [13], SCDM [29], MMN [25], MGPN [30],

TABLE 4

<span id="page-9-0"></span>Comparison with the state-of-the-arts on the *ActivityNet Captions* and *TACoS* datasets for *NLVG* task. ‡ denotes multi-query training mode, others are single-query training mode. UniSDNet-S is single-query training result, and UniSDNet-M is multi-query training result. We evaluate our model with two different text feature: GloVe [49] and BERT [62].

|                |                       |           |       |            |       |                 | Activi  | yNet Ca | ptions         |          |       |       |       |          | TACoS        |       |          |              |
|----------------|-----------------------|-----------|-------|------------|-------|-----------------|---------|---------|----------------|----------|-------|-------|-------|----------|--------------|-------|----------|--------------|
|                | Methods               | Venue     | Text  | Video      | 0.3   | R@1, IoU<br>0.5 | 0.7     | 0.3     | @5, IoU<br>0.5 | @<br>0.7 | mIoU  | 0.3   | 0.5   | @<br>0.7 | 0.3          | 0.5   | @<br>0.7 | mIoU         |
|                | VSLNet [17]           | ACL'20    | GloVe | C3D        | 63.16 | 43.22           | 26.16   | -       | -              | -        | 43.19 | 29.61 | 24.27 | 20.03    | -            | -     | -        | 24.11        |
| proposal-free  | LGI [19]              | CVPR'20   | -     | C3D        | 58.52 | 41.51           | 23.07   | -       | -              | -        | 41.13 | -     | -     | -        | -            | -     | -        | -            |
| 4              | CPNet [7]             | AAAI'21   | GloVe | C3D        | -     | 40.56           | 21.63 - | -       | -              | -        | 40.65 | 42.61 | 28.29 | -        | -            | -     | -        | 28.69        |
| Sa             | VSLNet-L [15]         | TPAMI'21  | GloVe | C3D        | -     | 43.86           | 27.51   | -       | -              | -        | 44.06 | 47.11 | 36.34 | 26.42    | -            | -     | -        | 36.61        |
| ğ              | VGCL [5]              | ACM MM'22 | GloVe | C3D        | 60.57 | 42.96           | 25.68   | -       | -              | -        | 43.34 | -     | -     | -        | -            | -     | -        | -            |
| <u> </u>       | METML [6]             | EACL'23   | BERT  | I3D        | 60.61 | 43.74           | 27.04   | -       | -              | -        | 44.05 | -     | -     | -        | -            | -     | -        | -            |
| _              | MA3SRN [8]            | TMM'23    | GloVe | C3D+Object | -     | 51.97           | 31.39   | -       | 84.05          | 68.11    | -     | 47.88 | 37.65 | -        | 66.02        | 54.27 | -        | -            |
|                | 2D-TAN [11]           | AAAI'20   | GloVe | C3D        | 59.45 | 44.51           | 26.54   | 85.53   | 77.13          | 61.96    | -     | 37.29 | 25.32 | -        | 57.81        | 45.04 | -        | -            |
|                | CSMGAN [9]            | ACM MM'20 | GloVe | C3D        | 68.52 | 49.11           | 29.15   | 87.68   | 77.43          | 59.63    | -     | 33.90 | 27.09 | -        | 53.98        | 41.22 | -        | -            |
|                | MS-2D-TAN [12]        | TPAMI '21 | GloVe | C3D        | 61.04 | 46.16           | 29.21   | 87.30   | 78.80          | 60.85    | -     | 45.61 | 35.77 | 23.44    | 69.11        | 57.31 | 36.09    | -            |
|                | MSAT [20]             | CVPR'21   | -     | C3D        | -     | 48.02           | 31.78   | -       | 78.02          | 63.18    | -     | 48.79 | 37.57 | -        | 67.63        | 57.91 | -        | -            |
|                | RaNet [22]            | EMNLP'21  | GloVe | C3D        | -     | 45.59           | 28.67   | -       | 75.93          | 62.97    | -     | 43.34 | 33.54 | -        | 67.33        | 55.09 | -        | -            |
| _              | I <sup>2</sup> N [24] | TIP'21    | GloVe | C3D        | -     | -               | -       | -       | -              | -        | -     | 31.47 | 29.25 | -        | 52.65        | 46.08 | -        | -            |
| e e            | FVMR [13]             | ICCV'21   | GloVe | C3D        | 60.63 | 45.00           | 26.85   | 86.11   | 77.42          | 61.04    | -     | 41.48 | 29.12 | -        | 64.53        | 50.00 | -        | -            |
| )as            | SCDM [29]             | TPAMI'22  | GloVe | C3D        | 55.25 | 36.90           | 20.28   | 78.79   | 66.84          | 42.92    | -     | 27.64 | 23.27 | -        | 40.06        | 33.49 | -        | -            |
| 듵              | MGPN [30]             | SIGIR'22  | GloVe | C3D        | -     | 47.92           | 30.47   | -       | 78.15          | 63.56    | -     | 48.81 | 36.74 | -        | 71.46        | 59.24 | -        | -            |
| os.            | SPL [16]              | ACM MM'22 | GloVe | C3D        | -     | 52.89           | 32.04   | -       | 82.65          | 67.21    | -     | 42.73 | 32.58 | -        | 64.30        | 50.17 | -        | -            |
| proposal-based | DCLN [26]             | ICMR'22   | GloVe | C3D        | 65.58 | 44.41           | 24.80   | 84.65   | 74.04          | 56.67    | -     | 44.96 | 28.72 | -        | 66.13        | 51.91 | -        | -            |
| ם              | CRaNet [10]           | TCSVT'23  | GloVe | C3D        | -     | 47.27           | 30.34   | -       | 78.84          | 63.51    | -     | 47.86 | 37.02 | -        | 70.78        | 58.39 | -        | -            |
|                | PLN [27]              | ACM MM'23 | GloVe | C3D        | 59.65 | 45.66           | 29.28   | 85.66   | 76.65          | 63.06    | 44.12 | 43.89 | 31.12 | -        | 65.11        | 52.89 | -        | 29.70        |
|                | $M^2DCapsN$ [33]      | TNNLS'23  | GloVe | C3D        | 61.53 | 47.03           | 29.99   | -       | 76.64          | 62.83    | -     | 46.41 | 32.58 | -        | 66.32        | 52.91 | -        | -            |
|                | MMN‡ [25]             | AAAI'22   | BERT  | C3D        | -     | 48.59           | 29.26   | -       | 79.50          | 64.76    | -     | 39.24 | 26.17 | -        | 62.03        | 47.39 | -        | -            |
|                | PTRM‡ [14]            | AAAI'23   | BERT  | C3D        | 66.41 | 50.44           | 31.18   | -       | -              | -        | 47.68 | -     | -     | -        | -            | -     | -        | -            |
|                | DFM‡ [63]             | ACM MM'23 | BERT  | C3D        | -     | 45.92           | 32.18   | -       | -              | -        | -     | 40.04 | 28.57 | 14.77    | -            | -     | -        | 27.35        |
|                | UniSDNet-S (Our       |           | GloVe | C3D        | 68.59 | 52.73           | 31.08   | 89.57   | 84.19          | 72.52    | 50.13 | 51.44 | 36.37 | 23.47    | 76.56        | 61.06 | 36.22    | 35.83        |
|                | UniSDNet-S (Our       | -,        | BERT  | C3D        | 68.66 | 52.35           | 32.25   | 89.74   | 83.35          | 70.61    | 50.22 | 53.46 | 36.24 | 23.48    | 76.96        | 63.06 | 36.34    | 36.47        |
|                | UniSDNet-M (Ou        |           | GloVe | C3D        | 74.07 | 57.67           | 35.64   | 90.49   | 84.46          | 72.47    | 53.68 | 53.59 | 38.34 | 23.79    | 79.01        | 64.83 | 36.89    | <u>37.54</u> |
|                | UniSDNet-M (Ou        | rs)       | BERT  | C3D        | 75.85 | 60.75           | 38.88   | 91.17   | 85.34          | 74.01    | 55.47 | 55.56 | 40.26 | 24.12    | <u>77.08</u> | 64.01 | 37.02    | 38.88        |

TABLE 5

<span id="page-9-1"></span>Comparison with the state-of-the-arts on the *Charades-STA* dataset for *NLVG* task. ‡ denotes multi-query training mode. Both MMN and our method originate from the exploitation of 2D temporal map. The single-query (UniSDNet-S) and multi-query (UniSDNet-M) training results on this dataset are closest compared to the other two NLVG datasets, due to its distribution of the query number concentrated in 1 (as shown in Fig. 5).

|                   |                            |             | Video       | Feature      | : VGG       |       |             | Vide         | o Feature   | e: C3D       |       |             | Vide         | o Featur    | e: I3D       |       |
|-------------------|----------------------------|-------------|-------------|--------------|-------------|-------|-------------|--------------|-------------|--------------|-------|-------------|--------------|-------------|--------------|-------|
|                   | Methods                    | R@1,<br>0.5 | IoU@<br>0.7 | R@5,<br>0.5  | IoU@<br>0.7 | mIoU  | R@1,<br>0.5 | IoU@<br>0.7  | R@5,<br>0.5 | IoU@<br>0.7  | mIoU  | R@1,<br>0.5 | IoU@<br>0.7  | R@5,<br>0.5 | IoU@<br>0.7  | mIoU  |
| proposal-<br>free | DRN [18]                   | -           | -           | -            | -           | -     | 45.40       | 26.40        | 88.01       | 55.38        | -     | 53.09       | 31.75        | 89.06       | 60.05        | -     |
| pos               | LGI [19]                   | -           | -           | -            | -           | -     | -           | -            | -           | -            | -     | 59.46       | 35.48        | -           | -            | 51.38 |
| ў.<br>Щ           | BPNet [23]                 | -           | -           | -            | -           | -     | 38.25       | 20.51        | -           | -            | 38.03 | 50.75       | 31.64        | -           | -            | 46.34 |
| īd                | CPNet [7]                  | -           | -           | -            | -           | -     | 40.32       | 22.47        | -           | -            | 37.36 | 60.27       | 38.74        | -           | -            | 52.00 |
|                   | 2D-TAN [11]                | 42.80       | 23.25       | 80.54        | 54.14       | -     | -           | -            | -           | -            | -     | -           | -            | -           | -            | -     |
|                   | MS-2D-TAN [12]             | 45.65       | 27.20       | 86.72        | 56.42       | -     | 41.10       | 23.25        | 81.53       | 48.55        | -     | 60.08       | 37.39        | 89.06       | 59.17        | -     |
|                   | FVMR [13]                  | -           | -           | _            | -           | _     | 38.16       | 18.22        | 82.18       | 44.96        | -     | 55.01       | 33.74        | 89.17       | 57.24        | -     |
| ਚ                 | $I^2N$ [24]                | -           | -           | _            | -           | -     | -           | -            | -           | -            | -     | 56.61       | 34.14        | 81.48       | 55.19        | -     |
| Se                | CPL [21]                   | -           | -           | _            | -           | -     | -           | -            | -           | -            | -     | 49.05       | 22.61        | 84.71       | 52.37        | -     |
| proposal-based    | PLN [27]                   | 45.43       | 26.26       | 86.32        | 57.02       | 41.28 | -           | -            | -           | -            | -     | 56.02       | 35.16        | 87.63       | 62.34        | 49.09 |
| ig j              | PTRM‡ [14]                 | 47.77       | 28.01       | -            | -           | 42.77 | -           | -            | -           | -            | -     | -           | -            | -           | -            | -     |
| õ.                | CRaNet [10]                | 47.12       | 27.39       | 83.51        | 58.33       | -     | -           | -            | -           | -            | -     | 60.94       | 41.32        | 89.97       | 65.19        | -     |
| ro                | M <sup>2</sup> DCapsN [33] | 43.17       | 25.13       | 79.35        | 55.86       | -     | 40.81       | 23.98        | 77.93       | 53.52        | -     | 55.03       | 31.61        | 84.33       | 63.71        | -     |
| ď                 | MMN‡ [25]                  | 47.31       | 27.28       | 83.74        | 58.41       | _     | -           | -            | -           | -            | -     | -           | -            | -           | -            | -     |
| I                 | UniSDNet-S (Ours)          | 47.34       | 27.45       | 84.68        | 58.41       | 43.32 | 48.71       | <u>27.31</u> | 82.77       | <u>57.58</u> | 43.16 | 59.41       | 38.58        | 89.52       | <u>70.65</u> | 52.07 |
|                   | UniSDNet-M (Ours)          | 48.41       | 28.33       | <u>84.76</u> | 59.46       | 44.41 | 49.57       | 28.39        | 84.70       | 58.49        | 44.29 | 61.02       | <u>39.70</u> | 89.97       | 73.20        | 52.69 |

SPL [16], DCLN [26], CPL [21], PTRM [14], CRaNet [10], PLN [27],  $M^2DCapsN$  [33], DFM [63]. The best and second-best results are marked in **bold** and <u>underlined</u> in experimental tables. The detailed test results of  $R@1, IoU@\{0.3, 0.5, 0.7\}$  on three NLVG datasets are reported in Table 4 and Table 5. Since most works do not report R@1, IoU@0.1 performance, we have removed it from the table. Notably, our method performs well on all metrics on the three NLVG datasets. For more prediction distributions of our model and other existing methods on NLVG task, see Appendix A.

# 5.2.1 Results on the ActivityNet Captions dataset

The ActivityNet Captions is the largest open domain dataset for NLVG. As shown in Table 4, our UniSDNet-S has achieved satisfactory performance to current SOTA methods, but at a very low cost of 0.53M for static modules and 0.68M for dynamic modules (Table 3). If the -M (multiquery) mode is utilized in training, there will be a significant increase in performance (UniSDNet-M achieves the best performance with scores of 38.88 and 55.47 in terms of R@1, IoU@0.7, and mIoU, respectively), note that regardless of UniSDNet-S and -M, they are tested in the same fair way, *i.e.*, single-query reasoning at a time. And a lot of work has also released M-query training modes such as MMN [25], PTRM [14] and DFM [68], but their performances are significantly worse than these of our UniSDNet-M due to our efficient modelling of multimodal information. Since we adopt a proposal-based backend to favor modal alignment between the video moment and the query, we prefer to compare our method with recently proposed

TABLE 6

<span id="page-10-0"></span>Comparison with state-of-the-art methods on three datasets for *SLVG* task, in which *Charades-STA Speech\** and *TACoS Speech\** are our new collected datasets, described in Section 4. ‡ denotes multi-query training mode, † denotes our reproduced results using the released code.

| Dataset                  | Method                  | Audio Feature   | Video Feature |       | R@1, IoU@    |              |       | R@5, IoU@ |       | mIoU         |
|--------------------------|-------------------------|-----------------|---------------|-------|--------------|--------------|-------|-----------|-------|--------------|
| Dataset                  |                         |                 | video reature | 0.3   | 0.5          | 0.7          | 0.3   | 0.5       | 0.7   |              |
|                          | VGCL [5]                | CPC [42]        |               | 49.80 | 30.05        | 16.63        | -     | -         | -     | 35.36        |
|                          | ISL [31]                | Mel Spectrogram |               | 49.46 | 30.26        | 15.22        | 82.28 | 63.73     | 35.48 | 34.52        |
| ActivityNet Speech [5]   | VSLNet [17]             | Mel Spectrogram | C3D           | 46.75 | 29.08        | 16.24        | -     | -         | -     | 34.01        |
| Activity (vet Speech [8] | VSLNet <sup>†</sup>     | Data2vec [44]   | CSD           | 51.02 | 30.38        | 17.45        | -     | -         | -     | 37.04        |
|                          | MMN‡ [12]†              | Data2vec        |               | 51.98 | 35.69        | 20.77        | 85.46 | 75.29     | 56.87 | 37.81        |
|                          | UniSDNet-S              | Data2vec        |               | 64.83 | <u>47.82</u> | <u>27.49</u> | 90.69 | 84.16     | 72.12 | <u>47.31</u> |
|                          | UniSDNet-M              | Data2vec        |               | 72.27 | 56.29        | 33.29        | 90.41 | 84.28     | 72.42 | 52.22        |
|                          | VSLNet <sup>†</sup>     |                 |               | 53.06 | 32.43        | 17.69        | -     | -         | -     | 37.22        |
|                          | $MMN^{\ddagger\dagger}$ | Data2vec        | I3D           | 53.23 | 35.53        | 20.09        | 83.77 | 72.76     | 55.88 | 38.24        |
|                          | UniSDNet-S              |                 |               | 64.16 | 49.28        | 27.94        | 90.05 | 83.38     | 67.09 | 47.47        |
|                          | UniSDNet-M              |                 |               | 69.83 | 54.93        | 33.20        | 90.38 | 84.21     | 71.76 | 51.19        |
|                          | VSLNet <sup>†</sup>     |                 |               | 50.27 | 38.76        | 23.25        | -     | -         | -     | 35.78        |
|                          | $MMN^{\ddagger\dagger}$ | Data2vec        | VGG           | 56.16 | 42.74        | 24.14        | 91.25 | 80.96     | 55.97 | 39.15        |
|                          | UniSDNet-S              | Datazvec        | VGG           | 59.19 | 45.08        | 25.91        | 92.02 | 82.47     | 57.34 | 41.26        |
|                          | UniSDNet-M              |                 |               | 60.73 | 46.37        | 26.72        | 92.66 | 82.31     | 57.66 | 42.28        |
|                          | VSLNet <sup>†</sup>     |                 |               | 52.42 | 40.70        | 22.36        | -     | -         | -     | 36.91        |
| Charades-STA Speech* 4   | $MMN^{\dagger\dagger}$  | Data2vec        | C3D           | 52.28 | 39.44        | 21.80        | 85.24 | 74.16     | 48.23 | 36.09        |
| Churuues-3171 Speech 4   | UniSDNet-S              |                 |               | 56.37 | 41.85        | 24.06        | 86.61 | 76.24     | 52.39 | 39.21        |
|                          | UniSDNet-M              |                 |               | 58.20 | 43.66        | 25.05        | 92.23 | 82.15     | 55.86 | 40.56        |
|                          | VSLNet <sup>†</sup>     |                 |               | 65.46 | 47.55        | 28.98        | -     | -         | -     | 45.40        |
|                          | $MMN^{\dagger\dagger}$  | Data2vec        | I3D           | 64.27 | 51.75        | 31.26        | 93.46 | 85.90     | 62.69 | 45.84        |
|                          | UniSDNet-S              | Datazvec        | 13D           | 67.37 | 53.63        | 33.87        | 94.54 | 87.45     | 67.77 | 48.13        |
|                          | UniSDNet-M              |                 |               | 67.45 | 53.82        | 34.49        | 94.81 | 87.90     | 69.30 | 48.27        |
|                          | VSLNet <sup>†</sup>     |                 |               | 29.39 | 20.59        | 10.92        | -     | -         | -     | 21.10        |
|                          | $MMN^{\ddagger\dagger}$ | Data2vec        | VGG           | 30.12 | 20.07        | 11.62        | 56.24 | 40.64     | 22.17 | 21.21        |
|                          | UniSDNet-S              |                 |               | 38.94 | 23.07        | 11.02        | 68.13 | 50.31     | 24.97 | 27.59        |
|                          | UniSDNet-M              |                 |               | 40.29 | 26.34        | 12.85        | 67.36 | 51.41     | 26.24 | 28.40        |
|                          | VSLNet <sup>†</sup>     |                 |               | 38.14 | 27.87        | 16.35        | -     | -         | -     | 27.28        |
| TACoS Speech* 4          | $MMN^{\dagger\dagger}$  | D. ( - 2        | Cap           | 31.72 | 23.82        | 12.55        | 59.16 | 45.36     | 22.89 | 22.58        |
| 1AC03 3peech 4           | UniSDNet-S              | Data2vec        | C3D           | 47.04 | 31.77        | 17.42        | 73.78 | 60.88     | 32.69 | 33.25        |
|                          | UniSDNet-M              |                 |               | 51.66 | 37.77        | 20.44        | 76.38 | 63.48     | 33.64 | 36.86        |
|                          | VSLNet <sup>†</sup>     |                 |               | 30.54 | 18.87        | 10.67        | -     | -         | -     | 19.88        |
|                          | $MMN^{\dagger\dagger}$  | Data2vec        | I3D           | 29.39 | 20.37        | 10.82        | 54.46 | 42.41     | 21.14 | 20.86        |
|                          | UniSDNet-S              |                 |               | 40.11 | 25.19        | 11.37        | 67.58 | 50.36     | 24.62 | 27.93        |
|                          | UniSDNet-M              |                 |               | 41.74 | 26.34        | 12.25        | 69.26 | 51.26     | 24.94 | 29.27        |

proposal-based methods, especially MMN [25], PTRM [14], etc. And our research on recent NLVG work has found that proposal-based methods predominate, as shown in Table 4. Compared to other proposal-based methods, our UniSDNet-M performs the best and has substantial improvements in all metrics due to the unique static and dynamic modes.

# 5.2.2 Results on the TACoS dataset

TACoS (Cooking dataset) has the longest video length (approx. 5 min) and the highest number of events (>100)per video (more details in Table 1). As shown in Table 4, the proposed UniSDNet-S (BERT) performs well with R@1, IoU@0.3 being 53.46, and UniSDNet-M achieves the best results across all metrics (e.g., 38.88 on mIoU), indicating that our model is better able to construct multiquery multimodal environmental semantics for video understanding. For proposal-based method MSAT [20] with good performance of 37.57 on R@1, IoU@0.5. It focuses only on static feature interactions with a transformer encoder. In contrast, our UniSDNet-M uses the lightweight MLP- and dynamic GCN-based network to construct deeper cross-modal associations, and performs better than MSAT, achieving improvements of 6.77 and 2.69 in R@1, IoU@0.3and R@1, IoU@0.5 metrics, respectively.

## 5.2.3 Results on the Charades-STA dataset

For the Charades-STA dataset, we report the fair comparison results of our method under VGG, C3D, and I3D features in

Table 5. Notably, the different characteristics of Charades-STA compared to the other two NLVG datasets are analysed in Fig. 5, Fig. 6 and Section 4.3, including smallest query number size, shortest query length and shortest video duration with an average of 30.60s, so that more subtle human movements need to be identified, resulting in that the models are sensitive to different visual features. Despite under this limitation, for the VGG and C3D visual features, our method achieves the best performance on the stringent metric R@1, e.g., 28.33 and 28.39 R@1, IoU@0.7on VGG and C3D feature, respectively. For the I3D video features, our UniSDNet-M achieves an outstanding record in R@1, IoU@0.5 and R@5, IoU@0.7, that are 61.02 and 73.20, demonstrating the robustness and generalization of our model. Moreover, we specifically make a fair comparison of ours with MMN [25] based on the same 2D temporal proposal map. Compared with MMN, our UniSDNet has improvements of 1.05  $\uparrow$  in R@1, IoU@0.7 with VGG feature.

# 5.3 Comparison with state-of-the-arts for SLVG Task

We compare our UniSDNet with the state-of-the-art methods for SLVG, including VGCL [5], SIL [31], VSLNet [17] and MMN [25] methods, where VGCL and SIL both have been assessed on the *ActivityNet Speech* dataset. In order to make fair comparison and add richer results, we reconstruct VSLNet [17] and MMN [25] models for the SLVG task,

where VSLNet is a classic proposal-free method, and MMN is a classic proposal-based method.

In addition, existing NLVG methods evaluated different video features in the experiments, including VGG, C3D, and I3D fatures. To validate our UniSDNet on the SLVG task dataset effect, we evaluate it using all existing available video features. Since there is no existing work reporting VGG video feature results on the ActivityNet Captions and ActivityNet Speech datasets, we followed them and do not report this result. And in Table [6,](#page-10-0) except for the video features presented in the implementation details, the other video features on different datasets are taken from the MS-2D-TAN [\[12\]](#page-15-11). The detailed test results on ActivityNet Speech dataset and our newly collected two datasets Charades-STA Speech and TACoS Speech are listed in Table [6.](#page-10-0) Our method perform the best stably under different features. See Appendix A for visualizations of the results.

# *5.3.1 Results on the ActivityNet Speech dataset*

The results on the ActivityNet Speech dataset are delineated in Table [6,](#page-10-0) where we evaluate a broader array of audio features, including Contrastive Predictive Coding (CPC) [\[42\]](#page-16-25), Mel Spectrogram, and Data2vec [\[44\]](#page-16-44) audio features. This analysis aims to elucidate the variations in performance attributable to different pre-extracted audio features. It is observed that our UniSDNet-S and UniSDNet-M achieves state-of-the-art performances across all evaluated metrics (*e.g*., 33.29 on R@1, IoU@0.7). Compared to VGCL [\[5\]](#page-15-4) and ISL [\[31\]](#page-16-15), our UniSDNet-M exhibits a remarkable enhancement, improving by approximately 20 points in mIoU. This significant gain underscores the superior efficacy of our integrated static and dynamic framework in addressing the SLVG task. The reconstructed VSLNet method, which utilizes Data2vec audio features, demonstrates an improvement of approximately 1 point in R@1, IoU@0.7 compared to the VSLNet method that uses audio Mel Spectrogram as input. When we account for the differences in input audio features and utilize the common Data2vec audio features, our UniSDNet-M outperforms VSLNet and MMN with scores of 15.84 and 12.52 on R@1IoU@0.7, respectively. This highlights the effectiveness of our method in associating cross-modal information between audio and video.

# *5.3.2 Results on Two New Speech datasets*

To advance research in SLVG, we conduct experiments on newly collected datasets, *Charades-STA Speech* and *TACos Speech*, as detailed in Section [4](#page-6-0) (Table [2\)](#page-7-1) and depicted in Table [6.](#page-10-0) Our UniSDNet-M achieves SOTA performance across all evaluated SLVG datasets, (*e.g*., R@1, IoU@0.7 of 34.49 and 20.44 on the Charades-STA Speech and TACoS Speech, respectively). This underscores its exceptional versatility across a variety of dataset environments. When compared to VSLNet, our UniSDNet-M exhibits superior performance, enhancing the mIoU by margins of 2.87 and 9.58 on the Charades-STA Speech and TACoS Speech datasets, respectively. Furthermore, in a direct comparison with the baseline model MMN, our UniSDNet-M demonstrates significantly better performance, with mIoU improvements of 3.13 and 14.28 on the Charades-STA Speech and TACoS Speech datasets, respectively. These improvements further highlight the efficacy of our static and dynamic framework

in bridging cross-modal information between audio and video, showcasing not only its accuracy but also its ability to effectively associate diverse modalities.

# <span id="page-11-0"></span>**5.4 Model Efficiency**

To better distinguish our model from other proposal-based models, we conduct an efficiency comparison on the ActivityNet Captions dataset in both single-query and multiquery training modes. The results are presented in Table [8.](#page-12-1) Additionally, the specific parameters of the various modules within our UniSDNet are elaborated in Fig. [7](#page-12-0) and Table [3.](#page-8-0) From the analysis in Table [8,](#page-12-1) it is evident that our UniSDNet offers moderate parameters and exhibits the fastest inference speed 0.009 s/query, regardless of the training mode (single-query or multi-query). It is worth noting that our UniSDNet-M has only half the number of model parameters compared to the proposal-based multi-query MMN and PTRM models. Nonetheless, our UniSDNet-M achieves a remarkable 35.71% improvement in efficiency over MMN. Compared to the PTRM approach that employs multiquery training, our UniSDNet exhibits notable accuracy enhancement, with an increase of 10.31% in R@1, IoU@0.5. Meanwhile, under single-query training, UniSDNet-S also has 9.02% of performance gain on R@1, IoU@0.5 while being 4.67× faster than single-query training SOTA method.

## **5.5 Ablation Studies**

In this section, we conduct in-depth ablation to analyze each component and specified parameter of UniSDNet. The experiments are conducted in multi-query training mode.

## *5.5.1 Ablation Study on Static and Dynamic Modules*

We remove the static (Section [3.2\)](#page-3-0) and dynamic modules (Section [3.3\)](#page-4-0) separately to investigate their contribution to cross-modal associativity modeling in our model. The results of NLVG and SLVG are reported in Table [7.](#page-12-3) In NLVG, the single static module outperforms the baseline (without static and dynamic modules) with improvements of 4.91 and 7.48 in R@1, IoU@0.7 and mIoU, respectively. In addition, the single dynamic module exhibits improvements of 7.41 and 8.32 than the baseline on R@1, IoU@0.7 and mIoU, which demonstrates its effectiveness of dynamic temporal modeling in the video. When combining the static and dynamic modules, all the performance metrics are further improved, such as setting new SOTA records 38.88 in R@1, IoU@0.7 and 55.47 in mIoU for NLVG. In SLVG, we can observe similar conclusions. These results demonstrate that both static and dynamic modules indeed have a mutual promoting effect on improving accuracy.

#### *5.5.2 Ablation Study on Static Network Variants*

In the static network, transformer architecture [\[28\]](#page-16-9) or the recent S4 architecture [\[69\]](#page-17-7) can also be used as long-range filter. We have tested the effect of Transformer or S4 as a static network as shown in Table [9.](#page-12-4) From the results, in terms of performance and efficiency, Transformer is close to our method, but our results are better. We speculate that the reason is that our network also includes the second stage of graph filtering. The static network uses a lightweight

<span id="page-12-3"></span>TABLE 7
Ablation studies of the static (Section 3.2) and dynamic (Section 3.3) modules on the ActivityNet Captions and ActivityNet Speech datasets.

| Task | Static | Dynamic | R@1, IoU@0.3   | R@1, IoU@0.5   | R@1, IoU@0.7   | R@5, IoU@0.3   | R@5, IoU@0.5   | R@5, IoU@0.7   | mIoU           |
|------|--------|---------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
|      | X      | X       | 61.22<br>72.32 | 44.46<br>55.18 | 26.76<br>31.67 | 87.19<br>90.99 | 78.63<br>84.65 | 63.60<br>71.46 | 43.98<br>51.46 |
| NLVG | X      | ĵ.      | 72.74          | 55.99          | 34.17          | 90.51          | 84.65<br>83.95 | 71.42          | 52.30          |
|      | ✓      | ✓       | 75.85          | 60.75          | 38.88          | 91.16          | 85.34          | 74.01          | 55.47          |
|      | X      | X       | 53.63<br>64.83 | 35.91<br>47.82 | 20.51<br>27.49 | 84.71<br>90.19 | 74.21<br>84.16 | 55.95<br>72.12 | 38.23<br>47.31 |
| SLVG | ×      | ĵ.      | 63.77          | 49.68          | <u>29.32</u>   | 89.84          | 83.33          | 70.30          | 47.55          |
|      | ✓      | ✓       | 72.27          | 56.29          | 33.29          | 90.41          | 84.28          | 72.42          | 52.22          |

<span id="page-12-1"></span>TABLE 8
Model efficiency comparison on the *ActivityNet Captions* dataset. "Infer. Speed" denotes the average inference time per query.

| Query  | Method            | Model Size    | Infer. Speed (s/query) | R@1, IoU@0.5 |
|--------|-------------------|---------------|------------------------|--------------|
| Single | 2D-TAN [11]       | 21.62M        | 0.061                  | 44.51        |
|        | MS-2D-TAN [12]    | 479.46M       | 0.141                  | 46.16        |
|        | MSAT [20]         | 37.19M        | 0.042                  | 48.02        |
|        | MGPN [30]         | 5.12M         | 0.115                  | 47.92        |
|        | UniSDNet-S (Ours) | 76.52M        | <b>0.009</b>           | <b>52.35</b> |
| Multi  | MMN [25]          | 152.22M       | 0.014                  | 48.59        |
|        | PTRM [14]         | 152.25M       | 0.038                  | 50.44        |
|        | UniSDNet-M (Ours) | <b>76.52M</b> | <b>0.009</b>           | <b>60.75</b> |

![](_page_12_Figure_6.jpeg)

<span id="page-12-0"></span>Fig. 7. Model Size vs. R@1, IoU@0.5 Accuracy Comparison of 2D Proposal-based Methods. Our UniSDNet-S has 9.02% of performance gain on the R@1, IoU@0.5 metric while being 4.67× faster than single-query training SOTA method. Also, UniSDNet-M significantly outperforms other recent 2D proposal-based NLVG methods [11], [12], [14], [20], [25], [30] on ActivityNet Captions dataset. UniSDNet-M achieves 10.31% of performance gain on the R@1, IoU@0.5 metric while being  $1.56 \times$  faster than multi-query training SOTA methods. The diameter of the circle indicates the model size (M).

and stable network (more detailed configuration in Table 3), which is more conducive to model training. Using Transformer as a static network increases the weight and instability factors [47] of the network.

# <span id="page-12-2"></span>5.5.3 Dynamic Network Variants and Hyperparameters

**Different Graph Networks.** Our dynamic network implementation is based on the graph structure. We compare it with the currently popular graph structures, GCN [70] and GAT [54], and test other variants of our graph filter, namely **D** and **MLP**. Additionally, our proposed temporal filtering graph contains more parametric details, which are analyzed

<span id="page-12-4"></span>TABLE 9
Different static networks Comparison on ActivityNet Captions dataset.

| Method                 | Infer. Speed | R     | @1, IoU | @     | mIoU  |
|------------------------|--------------|-------|---------|-------|-------|
| Method                 | (s/query)    | 0.3   | 0.5     | 0.7   | mioc  |
| Transformer [28]       | 0.024        | 75.17 | 59.98   | 38.38 | 54.97 |
| S4 [69]                | 0.030        | 70.41 | 55.11   | 34.93 | 51.40 |
| Our S <sup>3</sup> Net | 0.009        | 75.85 | 60.75   | 38.88 | 55.47 |

![](_page_12_Figure_13.jpeg)

<span id="page-12-5"></span>Fig. 8. R@1, IoU@0.5 results of different message passing strategies in our Graph on *ActivityNet Captions* and *ActivityNet Speech* datasets.

in Section 5.5.3. Specifically, the message aggregation definitions of these graphs are listed below:

• GCN: 
$$v_i^{(l+1)} = \sigma\left(\sum_{j \in (v_i)} \frac{1}{\sqrt{c_i c_j}} \cdot v_j^{(l)}\right)$$
;

• GAT: 
$$v_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(v_i)} a_{ij}^{(l)} \cdot v_j^{(l)}\right)$$
;

• **D:** 
$$v_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(v_i)} \frac{1}{d_{ij}^{(l)} + 1} \cdot v_j^{(l)}\right);$$

• MLP: 
$$v_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(v_i)} \text{MLP}(d_{ij}^{(l)}) \odot v_j^{(l)}\right)$$
;

• Our DTFNet: 
$$v_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(v_i)} \Phi(d_{ij}^{(l)}) \odot v_j^{(l)}\right)$$
,

where in these definitions,  $\sigma$  is the activation function,  $c_i$  is the degree of node  $v_i$  for GCN, and  $a_{ij}^{(l)} \in \mathbb{R}$  is the attention weight for GAT. The variant  $d_{ij}^{(l)} = (1-a_{ij}^{(l)}) \cdot ||j-i|| \in \mathbb{R}$  has been defined in Section 3.3, which denotes the joint clue of temporal distance and relevance between two nodes. In particular,  $\mathrm{MLP}(d_{ij}^{(l)}) \in \mathbb{R}^h$  and  $\Phi(d_{ij}^{(l)}) \in \mathbb{R}^h$  are two different ways of expanding the  $d_{ij}^{(l)}$  dimension. Observing Fig. 8, w/o Graph denotes the Dynamic

Observing Fig. 8, w/o Graph denotes the Dynamic Network is removed from the whole framework, and its performance is the worst. The vanilla GCN tracts all the neighbor nodes equally with a convolution operation to aggregate neighbor information. GAT is a weighted attention aggregation method [54]. Our method outperforms

<span id="page-13-1"></span>TABLE 10 Different Gaussian kernel number h and step z on the ActivityNet Captions dataset.

| #V am ala | Ctom | R     | @1, IoU | @     | R     | .@5, IoU | @     | mIoU  |
|-----------|------|-------|---------|-------|-------|----------|-------|-------|
| #Kernels  | Step | 0.3   | 0.5     | 0.7   | 0.3   | 0.5      | 0.7   | mioc  |
| 25        | 0.1  | 75.12 | 60.20   | 38.02 | 91.20 | 85.82    | 74.68 | 54.91 |
| 50        | 0.1  | 75.62 | 60.75   | 38.88 | 90.94 | 85.34    | 74.01 | 55.47 |
| 100       | 0.1  | 74.88 | 59.54   | 38.53 | 91.29 | 85.96    | 74.75 | 54.99 |
| 200       | 0.1  | 74.28 | 59.60   | 38.62 | 91.33 | 85.91    | 75.05 | 54.96 |
| 25        | 0.2  | 75.11 | 60.01   | 38.13 | 91.25 | 85.48    | 74.31 | 54.99 |
| 50        | 0.2  | 75.12 | 60.31   | 38.66 | 90.95 | 85.11    | 73.86 | 55.25 |
| 100       | 0.2  | 75.30 | 59.73   | 38.47 | 91.65 | 86.07    | 75.16 | 55.13 |
| 200       | 0.2  | 74.69 | 59.99   | 39.03 | 91.30 | 85.63    | 74.86 | 55.18 |

<span id="page-13-2"></span>TABLE 11 Different Gaussian coefficient  $\gamma$  on the ActivityNet Captions dataset.

| Gaussian Coefficient | R     | @1, IoU | @     | R     | 1@5, IoU | @     | mIoU   |
|----------------------|-------|---------|-------|-------|----------|-------|--------|
| Gaussian Coemicient  | 0.3   | 0.5     | 0.7   | 0.3   | 0.5      | 0.7   | IIIIOC |
| 5.0                  | 75.76 | 60.80   | 39.23 | 91.14 | 85.43    | 74.33 | 55.51  |
| 10.0                 | 75.85 | 60.75   | 38.88 | 91.16 | 85.34    | 74.01 | 55.47  |
| 25.0                 | 75.87 | 60.77   | 39.30 | 91.16 | 85.23    | 74.06 | 55.52  |
| 50.0                 | 75.84 | 60.98   | 38.83 | 91.04 | 85.27    | 73.98 | 55.51  |
| 75.0                 | 75.74 | 60.57   | 38.63 | 90.98 | 85.26    | 73.86 | 55.29  |
| average              | 75.81 | 60.77   | 38.97 | 91.10 | 85.31    | 74.05 | 55.46  |
| standard deviation   | 0.06  | 0.15    | 0.28  | 0.08  | 0.08     | 0.17  | 0.10   |

**GCN** and **GAT** by 2.61 and 1.97 on R@1, IoU@0.5 for NLVG, and by 3.00 and 2.33 on R@1, IoU@0.5 for SLVG, respectively. For D and MLP, we discuss the Gaussian filter setup in our method. In the setting of D, we directly use the message aggregation wight  $f_{ij}^{(l)}=1/(d_{ij}^{(l)}+1)$  to replace  $f_{ij}^{(l)} = \mathcal{F}_{filter}(d_{ij}^{(l)})$  in Eq. 3, which indicates that we still consider the same joint clue of temporal distance and relevance between two nodes  $d_{ij}^{(l)}$  but remove the Gaussian filtering calculation from our method. This replacement results in a decrease of 1.23 and 1.26 on R@1, IoU@0.5 for NLVG and SLVG, respectively. **MLP** uses the operation  $MLP(d_{ij}^{(l)})$  to replace the Gaussian basis function  $\phi(d_{ij}^{(l)})$  in Eq. 3. In this way, we realize the convolution kernel rather than Gaussian kernel in the dynamic filter. Compared to Ours, MLP has a decreased performance of 1.59 and 1.34 on R@1, IoU@0.5for NLVG and SLVG, respectively. Overall, our proposed dynamic filtering network offers irreplaceable benefits.

Hyperparameters in the Dynamic Temporal Filter  $\mathcal{F}_{filter}$ . In this work, we employ the multi-kernel Gaussian  $\Phi(x) =$  $\exp(-\gamma(x-z_k)^2), k \in [1,h]$  (Section 3.3), and there are three variables  $(z_k, h, \gamma)$ : different bias  $z_k$  for total h Gaussian kernels and a Gaussian coefficient  $\gamma$ . To meet the constraint of nonlinear correlated Gaussian kernels, we randomly set biases at equal intervals (e.g., 0.1 or 0.2) starting from 0.0, sweep the value of from 25 to 200 and set the global range of  $z_k$  values to [0, 5] in our experiments, as shown in Table 10. And we can find that the best setting is h = 50, we speculate that our method achieves the best results when number of Gaussian kernels h is close to the number of graph nodes. Gaussian coefficient  $\gamma$  reflects the amplitude of the Gaussian kernel function that controls the gradient descent speed of the function value. It can be found that from Table 11, when  $\gamma = 25.0$ , our model achieves the best performance with mIoU at 55.52. We also list the average and standard deviation of the five experimental results and select  $\gamma = 10.0$  as the empirical setting as its results are

![](_page_13_Figure_8.jpeg)

<span id="page-13-4"></span>Fig. 9. Different feature sampling strategies for 2D proposal generation. (a) Only the content feature. (b) The content and boundary features are fused by addition operation. (c) The content and boundary features are fused with concatenation operation.

![](_page_13_Figure_10.jpeg)

<span id="page-13-3"></span>Fig. 10. The results across different graph layer on the AcvitivtyNet Captions dataset for NLVG. From top to bottom, the metrics are R@1, IoU@0.3, IoU@0.5, and IoU@0.7, respectively.

closest to the average. To summarize, in our experiments, the final settings of variables  $(h, \gamma)$  are set to 50 and 10.0, and  $z_k$  is set at an equal interval of 0.1.

**Dynamic Graph Layer.** We investigate the influence of the graph layer of our dynamic module. As shown in Fig. 10, we observe that our model achieves the best result (e.g., R@1, IoU@0.7 is 38.88) when the total number of graph layer is set to 2. It is speculated that on the basis of informative context modelling by the static module, two-layers dynamic graph module is enough for relational learning of the video. Additionally, graph convolutional networks generally experience over-smoothing problem as the number of layers increases, leading to a performance decline [71]. Our model exhibits good stability on the  $1\sim$ 6-th graph layers.

# <span id="page-13-0"></span>5.5.4 Ablation Study on Proposals Generation

To analyze the sensitivity of the feature sampling strategy for 2D proposals generation, we evaluate the effects of moment content and boundary features. As shown in Fig. 9, we conduct experiments with different proposal generation strategies: (a) only the content feature; (b) the addition of content and two boundary features; (c) the concatenation of content and boundary features. Here, the content feature refers to  $\mathrm{Gen}(v_i^L, v_{i+1}^L, \cdots, v_j^L)$  with Gen being 1D Conv or MaxPool [11], where  $v_i^L$  and  $v_j^L$  are the start i-th and ending j-th video clip features, respectively. The experimental results for both NLVG and SLVG tasks are summarized in Table 12. For NLVG, the MaxPool strategy outperforms convolution, e.g., 38.88 vs. 38.20 in terms of R@1, IoU@0.7

<span id="page-14-0"></span>TABLE 12
Comparison of different proposal generation strategies on the ActivityNet Captions and ActivityNet Speech datasets.

| Task | Generation | Features          | Fusion        | R     | R@1, IoU | @     | R     | @5, IoU | @     | mIoU  |
|------|------------|-------------------|---------------|-------|----------|-------|-------|---------|-------|-------|
| lask | Generation | reatures          | rusion        | 0.3   | 0.5      | 0.7   | 0.3   | 0.5     | 0.7   | IIIOC |
|      | Conv       | Content           | -             | 75.30 | 60.27    | 38.20 | 90.86 | 85.16   | 73.17 | 55.13 |
|      | Conv       | Content, Boundary | Addition      | 75.85 | 60.70    | 38.75 | 90.85 | 85.05   | 73.25 | 55.41 |
| NLVG | Conv       | Content, Boundary | Concatenation | 74.76 | 60.30    | 38.80 | 90.70 | 84.96   | 73.00 | 55.15 |
| NLVG | MaxPool    | Content           | -             | 75.62 | 60.40    | 38.99 | 90.94 | 85.22   | 73.97 | 55.39 |
|      | MaxPool    | Content, Boundary | Addition      | 75.85 | 60.75    | 38.88 | 91.16 | 85.34   | 74.01 | 55.47 |
|      | MaxPool    | Content, Boundary | Concatenation | 75.13 | 59.96    | 38.25 | 91.26 | 85.59   | 73.91 | 54.98 |
|      | Conv       | Content           | -             | 71.02 | 55.24    | 32.88 | 90.38 | 84.25   | 71.38 | 51.66 |
|      | Conv       | Content, Boundary | Addition      | 72.27 | 56.29    | 33.29 | 90.41 | 84.28   | 72.42 | 52.22 |
| SLVG | Conv       | Content, Boundary | Concatenation | 71.45 | 55.79    | 33.20 | 90.55 | 84.16   | 71.48 | 51.76 |
| SLVG | MaxPool    | Content           | -             | 71.26 | 55.25    | 33.74 | 90.49 | 84.29   | 72.46 | 51.80 |
|      | MaxPool    | Content, Boundary | Addition      | 72.60 | 56.64    | 32.61 | 90.82 | 84.89   | 72.48 | 52.04 |
|      | MaxPool    | Content, Boundary | Concatenation | 69.85 | 53.96    | 32.05 | 90.36 | 84.12   | 72.24 | 50.68 |

![](_page_14_Figure_4.jpeg)

![](_page_14_Figure_5.jpeg)

- <span id="page-14-1"></span>(a) NLVG (on ActivityNet Captions)
- (b) SLVG (on ActivityNet Speech)

Fig. 11. Results of different training query number m of a video for NLVG and SLVG. Because the query number distribution of ActivityNet Captions is concentrated in the [3,8] (Fig. 5), we test m=1,3,5,8. When m=1, the training mode is single-query, and when m>1, the training mode is multi-query. "Full" represents all query inputs for a video simultaneously during training. A more detailed comparison of the single-query and multi-query mode methods is given in Table 8.

when the model using content feature. Additionally, addition performs better than concatenation, *e.g.*, 55.47 *vs.* 54.98 when the model uses the content and boundary features. SLVG shows similar results. Therefore, we use content and boundary features to generate proposals through MaxPool and Conv for both NLVG and SLVG.

#### 5.5.5 Ablation Study on Training Mode

In this work, we adopt two training mode: single-query and multi-query training, as described in experimental setup part (Section 5.1, Training and Inference Mode). The number of queries is an important variable, in order to explore the effect of our UniSDNet in single-query and multi-query modes, we conduct experiments with different number of queries on the ActivityNet Captions and ActivityNet Speech datasets. The results are shown in Fig. 11. It can be observed that for single-query training, our model is comparable with state-of-art MSAT [20] and VSLNet [17], achieving scores of 52,35 and 47.82 on R@1, IoU@0.7 in the NLVG and SLVG tasks, respectively. As the query number upper limit increases, the performance of our model significantly improves, which demonstrates the effectiveness of our model in utilizing multimodal information.

#### 5.6 Extended Evaluation on QVHighlights Dataset

We also validate our model on the most recently publicized NLVG dataset QVHighlights [38] for multi-tasks:

<span id="page-14-2"></span>TABLE 13
Performance comparison on QVHighlights *test* split. \*: introduce audio modality.

|                  |       |       | MR    |       |       | I     | HD      |
|------------------|-------|-------|-------|-------|-------|-------|---------|
| Method           | R     | :1    |       | mAP   |       | >= Ve | ry Good |
|                  | @0.5  | @0.7  | @0.5  | @0.75 | Avg.  | mAP   | HIT@1   |
| BeautyThumb [72] | -     | -     | -     | -     | -     | 14.36 | 20.88   |
| DVSE [73]        | -     | -     | -     | -     | -     | 18.75 | 21.79   |
| MCN [4]          | 11.41 | 2.72  | 24.94 | 8.22  | 10.67 | -     | -       |
| CAL [35]         | 25.49 | 11.54 | 23.40 | 7.65  | 9.89  | -     | -       |
| XML+ [36]        | 46.69 | 33.46 | 47.89 | 34.67 | 34.90 | 35.38 | 55.06   |
| CLIP [37]        | 16.88 | 5.19  | 18.11 | 7.00  | 7.67  | 31.30 | 61.04   |
| Moment-DETR [38] | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 | 35.69 | 55.60   |
| UMT* [39]        | 56.23 | 41.18 | 53.83 | 37.01 | 36.12 | 38.18 | 59.99   |
| MH-DETR [74]     | 60.05 | 42.48 | 60.75 | 38.13 | 38.38 | 38.22 | 60.51   |
| QD-DETR [75]     | 62.40 | 44.98 | 62.52 | 39.88 | 39.86 | 38.94 | 62.40   |
| UniVTG [76]      | 58.86 | 40.86 | 57.60 | 35.59 | 35.47 | 38.20 | 60.96   |
| UniSDNet (Ours)  | 63.49 | 46.63 | 62.86 | 42.51 | 41.33 | 39.80 | 64.66   |

both moment retrieval (MR, also called temporal video grounding) and highlight detection (HD) tasks. Following the practice [38], [39], the commonly used metric for moment retrieval is Recall@K, IoU=[0.5, 0.7], and mean average precision (mAP). HIT@1 is also used to evaluate the highlight detection by computing the hit ratio of the highest-scored clip. The other settings such as pre-extracted Slowfast video and CLIP text features, the number of transformer decoder layers and loss weights are the same with Moment-DETR [38]. The comparison with exiting works are listed in Table 13. From the results, our model achieves superior performance to state-of-art models, achieving R@1, IoU@0.7 of 63.03 for MR, and HIT@1 of 62.56 for HD, demonstrating its strong universality for both tasks.

# <span id="page-14-3"></span>5.7 Qualitative Results

We provide the qualitative results of our UniSDNet on the ActivityNet Captions dataset with a video named "v\_q81H-V1\_gGo" for NLVG, as shown in Fig. 12. MMN [25] exhibits significant semantic bias, making it impossible to distinguish between Q2 and Q3. Our **Only Static** accurately predicts the moments, which is thanks to the effective static learning of the semantic association between queries and video moments. Our **Only Dynamic** performs well in the three queries too, thanks to the fine dynamic learning of the video sequence context. The results of the full model **Ours** for all queries are the closest to **Groundtruth (GT)**. It shows that the full model can integrate the advantageous aspects

![](_page_15_Figure_2.jpeg)

<span id="page-15-12"></span>Fig. 12. Qualitative examples of our UniSDNet. The right figures display the groundtruth IoU maps and the predicted score maps by our UniSDNet.

of static (differentiating different query semantics and supplementing video semantics) and dynamic (differentiating and associating the related contexts in the video) modules to achieve more accurate target moment prediction. The quantitative results confirm the effectiveness of our unified static and static methods in solving both NLVG and SLVG tasks. More examples are unfolded in Appendix C.

#### 6 FUTURE DIRECTIONS

As a fundamental cross-modal task, TVG research remains focusing on effectively integrating multimodal data for accurate temporal localization. Language-queried video grounding dominates current research due to advanced language models. In the future, several promising directions can advance TVG: First, expanding to more flexible query modes - incorporating audio, images, and video clips can enhance the model's ability to handle diverse inputs and improve generalization. Second, addressing fine-grained video grounding is essential for real-world applications, requiring detailed temporal-spatial interactions and complex scene dynamics capture, by developing larger fine-grained datasets and more sophisticated models. Third, long-form video understanding, remains challenging, as current methods are typically designed for short videos struggle with extended duration content. Additionally, leveraging advances in large vision-language models (VLMs) like GPT-4V can better align visual and textual features, and explore more complementary modality information. Finally, improving model efficiency in computation and memory is crucial for scaling TVG systems to larger datasets and more complex scenarios.

# 7 CONCLUSION

In this paper, we propose a novel Unified Static and Dynamic Network (UniSDNet) for efficient video grounding. We can adopt either single-query or multi-query mode and achieve model performance/complexity trade-offs; it benefits from both "static" and "dynamic" associations between queries and video semantics in a cross-modal environment. We adopt a ResMLP architecture that comprehensively considers mutual semantic supplement through video-queries interaction (static mode). Afterwards, we utilize a dynamic

Temporal Gaussian filter convolution to model nonlinear high-dimensional visual semantic perception (dynamic mode). The static and dynamic manners complement each other, ensuring effective 2D temporary proposal generation. We also contribute two new Charades-STA Speech and TACoS Speech datasets for SLVG task. UniSDNet is evaluated on both NLVG and SLVG. For both of them we achieve new state-of-the-art results. We believe that our work is a new attempt and inspire similar video tasks in the design of neural networks guided by visual perception biology.

#### REFERENCES

- <span id="page-15-2"></span>[1] G. Deco, D. Vidaurre, and M. L. Kringelbach, "Revisiting the global workspace orchestrating the hierarchical organization of the human brain," *Nature human behaviour*, vol. 5, no. 4, pp. 497–511, 2021
- <span id="page-15-3"></span>[2] J. Barbosa, H. Stein, R. L. Martinez, A. Galan-Gadea, S. Li, J. Dalmau, K. C. Adam, J. Valls-Solé, C. Constantinidis, and A. Compte, "Interplay between persistent activity and activity-silent dynamics in the prefrontal cortex underlies serial biases in working memory," *Nature neuroscience*, vol. 23, no. 8, pp. 1016–1024, 2020.
- <span id="page-15-0"></span>[3] J. Gao, C. Sun, Z. Yang, and R. Nevatia, "TALL: temporal activity localization via language query," in *ICCV*, 2017, pp. 5277–5285.
   [4] L. Anne Hendricks, O. Wang, E. Shechtman, J. Sivic, T. Darrell, and
- <span id="page-15-1"></span>[4] L. Anne Hendricks, O. Wang, E. Shechtman, J. Sivic, T. Darrell, and B. Russell, "Localizing moments in video with natural language," in *ICCV*, 2017, pp. 5803–5812.
- <span id="page-15-4"></span>[5] Y. Xia, Z. Zhao, S. Ye, Y. Zhao, H. Li, and Y. Ren, "Video-guided curriculum learning for spoken video grounding," in ACM Multimedia, 2022, pp. 5191–5200.
- <span id="page-15-5"></span>[6] C. Rodriguez, E. Marrese-Taylor, B. Fernando, H. Takamura, and Q. Wu, "Memory-efficient temporal moment localization in long videos," in EACL, 2023, pp. 1901–1916.
- <span id="page-15-6"></span>[7] K. Li, D. Guo, and M. Wang, "Proposal-free video grounding with contextual pyramid network," in AAAI, vol. 35, no. 3, 2021, pp. 1902–1910.
- <span id="page-15-7"></span>[8] D. Liu, X. Fang, W. Hu, and P. Zhou, "Exploring optical-flow-guided motion and detection-based appearance for temporal sentence grounding," *IEEE Trans. Multim.*, vol. 25, pp. 8539–8553, 2023
- <span id="page-15-8"></span>[9] D. Liu, X. Qu, X.-Y. Liu, J. Dong, P. Zhou, and Z. Xu, "Jointly crossand self-modal graph attention network for query-based moment localization," in ACM Multimedia, 2020, pp. 4070–4078.
- <span id="page-15-9"></span>[10] X. Sun, J. Gao, Y. Zhu, X. Wang, and X. Zhou, "Video moment retrieval via comprehensive relation-aware network," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 33, no. 9, pp. 5281–5295, 2023.
- <span id="page-15-10"></span>[11] S. Zhang, H. Peng, J. Fu, and J. Luo, "Learning 2d temporal adjacent networks for moment localization with natural language," in AAAI, 2020, pp. 12870–12877.
- <span id="page-15-11"></span>[12] S. Zhang, H. Peng, J. Fu, Y. Lu, and J. Luo, "Multi-scale 2d temporal adjacency networks for moment localization with natural language," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 12, pp. 9073–9087, 2021.

- <span id="page-16-0"></span>[13] J. Gao and C. Xu, "Fast video moment retrieval," in *ICCV*, 2021, pp. 1523–1532.
- <span id="page-16-1"></span>[14] M. Zheng, S. Li, Q. Chen, Y. Peng, and Y. Liu, "Phrase-level temporal relationship mining for temporal sentence localization," in *AAAI*, 2023, pp. 3669–3677.
- <span id="page-16-2"></span>[15] H. Zhang, A. Sun, W. Jing, L. Zhen, J. T. Zhou, and R. S. M. Goh, "Natural language video localization: A revisit in span-based question answering framework," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 8, pp. 4252–4266, 2021.
- <span id="page-16-3"></span>[16] D. Liu and W. Hu, "Skimming, locating, then perusing: A humanlike framework for natural language video localization," in *ACM Multimedia*, 2022, pp. 4536–4545.
- <span id="page-16-4"></span>[17] H. Zhang, A. Sun, W. Jing, and J. T. Zhou, "Span-based localizing network for natural language video localization," in *ACL*, 2020, pp. 6543–6554.
- <span id="page-16-43"></span>[18] R. Zeng, H. Xu, W. Huang, P. Chen, M. Tan, and C. Gan, "Dense regression network for video grounding," in *ICCV*, 2020, pp. 10 287–10 296.
- <span id="page-16-16"></span>[19] J. Mun, M. Cho, and B. Han, "Local-global video-text interactions for temporal grounding," in *ICCV*, 2020, pp. 10 810–10 819.
- <span id="page-16-11"></span>[20] M. Zhang, Y. Yang, X. Chen, Y. Ji, X. Xu, J. Li, and H. T. Shen, "Multi-stage aggregated transformer network for temporal language localization in videos," in *ICCV*, 2021, pp. 12 669–12 678.
- <span id="page-16-5"></span>[21] M. Zheng, Y. Huang, Q. Chen, Y. Peng, and Y. Liu, "Weakly supervised temporal sentence grounding with gaussian-based contrastive proposal learning," in *ICCV*, 2022, pp. 15 555–15 564.
- <span id="page-16-6"></span>[22] J. Gao, X. Sun, M. Xu, X. Zhou, and B. Ghanem, "Relation-aware video reading comprehension for temporal language grounding," in *EMNLP*, 2021, pp. 3978–3988.
- <span id="page-16-7"></span>[23] S. Xiao, L. Chen, S. Zhang, W. Ji, J. Shao, L. Ye, and J. Xiao, "Boundary proposal network for two-stage natural language video localization," in *AAAI*, vol. 35, no. 4, 2021, pp. 2986–2994.
- <span id="page-16-12"></span>[24] K. Ning, L. Xie, J. Liu, F. Wu, and Q. Tian, "Interaction-integrated network for natural language moment localization," *IEEE Trans. Image Process.*, vol. 30, pp. 2538–2548, 2021.
- <span id="page-16-10"></span>[25] Z. Wang, L. Wang, T. Wu, T. Li, and G. Wu, "Negative sample matters: A renaissance of metric learning for temporal grounding," in *AAAI*, 2022, pp. 2613–2623.
- <span id="page-16-22"></span>[26] B. Zhang, B. Jiang, C. Yang, and L. Pang, "Dual-channel localization networks for moment retrieval with natural language," in *ICMR*, 2022, pp. 351–359.
- <span id="page-16-8"></span>[27] Q. Zheng, J. Dong, X. Qu, X. Yang, Y. Wang, P. Zhou, B. Liu, and X. Wang, "Progressive localization networks for languagebased moment localization," *ACM Trans. Multim. Comput. Commun. Appl.*, vol. 19, no. 2, pp. 1–21, 2023.
- <span id="page-16-9"></span>[28] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *NeurIPS*, vol. 30, 2017.
- <span id="page-16-13"></span>[29] Y. Yuan, L. Ma, J. Wang, W. Liu, and W. Zhu, "Semantic conditioned dynamic modulation for temporal sentence grounding in videos," *IEEE Trans. Pattern Anal. Mach. Intel.*, vol. 44, no. 5, pp. 2725–2741, 2022.
- <span id="page-16-14"></span>[30] X. Sun, X. Wang, J. Gao, Q. Liu, and X. Zhou, "You need to read again: Multi-granularity perception network for moment retrieval in videos," in *SIGIR*, 2022, pp. 1022–1032.
- <span id="page-16-15"></span>[31] Y. Wang, W. Lin, S. Zhang, T. Jin, L. Li, X. Cheng, and Z. Zhao, "Weakly-supervised spoken video grounding via semantic interaction learning," in *ACL*, 2023, pp. 10 914–10 932.
- <span id="page-16-17"></span>[32] K. Li, D. Guo, and M. Wang, "Vigt: proposal-free video grounding with a learnable token in the transformer," *Science China Information Sciences*, vol. 66, no. 10, p. 202102, 2023.
- <span id="page-16-18"></span>[33] N. Liu, X. Sun, H. Yu, F. Yao, G. Xu, and K. Fu, "M2dcapsn: Multimodal, multichannel, and dual-step capsule network for natural language moment localization," *IEEE Trans. Neural Networks Learn. Syst.*, 2023.
- <span id="page-16-19"></span>[34] Y. Yuan, T. Mei, and W. Zhu, "To find where you talk: Temporal sentence localization in video with attention based location regression," in *AAAI*, 2019, pp. 9159–9166.
- <span id="page-16-20"></span>[35] V. Escorcia, M. Soldan, J. Sivic, B. Ghanem, and B. C. Russell, "Temporal localization of moments in video collections with natural language," *CoRR*, vol. abs/1907.12763, 2019.
- <span id="page-16-46"></span>[36] J. Lei, L. Yu, T. L. Berg, and M. Bansal, "TVR: A large-scale dataset for video-subtitle moment retrieval," in *ECCV*, 2020, pp. 447–463.
- <span id="page-16-47"></span>[37] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark *et al.*, "Learning transferable visual models from natural language supervision," in *ICML*, 2021, pp. 8748–8763.

- <span id="page-16-45"></span>[38] J. Lei, T. L. Berg, and M. Bansal, "Detecting moments and highlights in videos via natural language queries," in *NeurIPS*, vol. 34, 2021, pp. 11 846–11 858.
- <span id="page-16-21"></span>[39] Y. Liu, S. Li, Y. Wu, C.-W. Chen, Y. Shan, and X. Qie, "Umt: Unified multi-modal transformers for joint video moment retrieval and highlight detection," in *ICCV*, 2022, pp. 3042–3051.
- <span id="page-16-23"></span>[40] C. Rodriguez, E. Marrese-Taylor, F. S. Saleh, H. Li, and S. Gould, "Proposal-free temporal moment localization of a naturallanguage query in video using guided attention," in *ICCV*, 2020, pp. 2464–2473.
- <span id="page-16-24"></span>[41] M. Xu, C. Zhao, D. S. Rojas, A. Thabet, and B. Ghanem, "G-tad: Sub-graph localization for temporal action detection," in *ICCV*, 2020, pp. 10 156–10 165.
- <span id="page-16-25"></span>[42] A. van den Oord, Y. Li, and O. Vinyals, "Representation learning with contrastive predictive coding," *CoRR*, vol. abs/1807.03748, 2018.
- <span id="page-16-26"></span>[43] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in *NeurIPS*, vol. 33, 2020, pp. 12 449–12 460.
- <span id="page-16-44"></span>[44] A. Baevski, W. Hsu, Q. Xu, A. Babu, J. Gu, and M. Auli, "data2vec: A general framework for self-supervised learning in speech, vision and language," in *ICML*, vol. 162, 2022, pp. 1298–1312.
- <span id="page-16-27"></span>[45] C. Wang, Y. Wu, Y. Qian, K. Kumatani, S. Liu, F. Wei, M. Zeng, and X. Huang, "Unispeech: Unified speech representation learning with labeled and unlabeled data," in *ICML*, 2021, pp. 10 937– 10 947.
- <span id="page-16-28"></span>[46] M. Regneri, M. Rohrbach, D. Wetzel, S. Thater, B. Schiele, and M. Pinkal, "Grounding action descriptions in videos," *Trans. Assoc. Comput. Linguistics*, vol. 1, pp. 25–36, 2013.
- <span id="page-16-29"></span>[47] H. Touvron, P. Bojanowski, M. Caron, M. Cord, A. El-Nouby, E. Grave, G. Izacard, A. Joulin, G. Synnaeve, J. Verbeek, and H. Jégou, "Resmlp: Feedforward networks for image classification with data-efficient training," *IEEE Trans. Pattern Anal. Mach. Intel.*, vol. 45, no. 4, pp. 5314–5321, 2023.
- <span id="page-16-30"></span>[48] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3d convolutional networks," in *ICCV*, 2015, pp. 4489–4497.
- <span id="page-16-31"></span>[49] J. Pennington, R. Socher, and C. D. Manning, "Glove: Global vectors for word representation," in *EMNLP*, 2014, pp. 1532–1543.
- <span id="page-16-32"></span>[50] A. Baevski, W.-N. Hsu, Q. Xu, A. Babu, J. Gu, and M. Auli, "Data2vec: A general framework for self-supervised learning in speech, vision and language," in *ICML*, 2022, pp. 1298–1312.
- <span id="page-16-33"></span>[51] P. Wang, S. Wang, J. Lin, S. Bai, X. Zhou, J. Zhou, X. Wang, and C. Zhou, "One-peace: Exploring one general representation model toward unlimited modalities," *arXiv preprint arXiv:2305.11172*, 2023.
- [52] W. Kim, B. Son, and I. Kim, "Vilt: Vision-and-language transformer without convolution or region supervision," in *ICML*. PMLR, 2021, pp. 5583–5594.
- <span id="page-16-34"></span>[53] J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu, "Coca: Contrastive captioners are image-text foundation models," *arXiv preprint arXiv:2205.01917*, 2022.
- <span id="page-16-35"></span>[54] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention networks," *CoRR*, vol. abs/1710.10903, 2017.
- <span id="page-16-36"></span>[55] F. Long, T. Yao, Z. Qiu, X. Tian, J. Luo, and T. Mei, "Gaussian temporal awareness networks for action localization," in *ICCV*, 2019, pp. 344–353.
- <span id="page-16-37"></span>[56] K. Schütt, P. Kindermans, H. E. S. Felix, S. Chmiela, A. Tkatchenko, and K. Müller, "Schnet: A continuous-filter convolutional neural network for modeling quantum interactions," in *NeurIPS*, 2017, pp. 991–1001.
- <span id="page-16-38"></span>[57] R. Krishna, K. Hata, F. Ren, L. Fei-Fei, and J. Carlos Niebles, "Dense-captioning events in videos," in *ICCV*, 2017, pp. 706–715.
- <span id="page-16-41"></span>[58] J. Ao, R. Wang, L. Zhou, C. Wang, S. Ren, Y. Wu, S. Liu, T. Ko, Q. Li, Y. Zhang *et al.*, "Speecht5: Unified-modal encoder-decoder pre-training for spoken language processing," in *ACL*, 2022, pp. 5723–5738.
- <span id="page-16-39"></span>[59] G. A. Sigurdsson, G. Varol, X. Wang, A. Farhadi, I. Laptev, and A. Gupta, "Hollywood in homes: Crowdsourcing data collection for activity understanding," in *ECCV*, 2016, pp. 510–526.
- <span id="page-16-40"></span>[60] M. Rohrbach, M. Regneri, M. Andriluka, S. Amin, M. Pinkal, and B. Schiele, "Script data for attribute-based recognition of composite activities," in *ECCV*, 2012, pp. 144–157.
- <span id="page-16-42"></span>[61] V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu, A. Elkahky, Z. Ni, A. Vyas, M. Fazel-Zarandi, A. Baevski, Y. Adi,

- X. Zhang, W. Hsu, A. Conneau, and M. Auli, "Scaling speech technology to 1, 000+ languages," *CoRR*, vol. abs/2305.13516, 2023.
- <span id="page-17-2"></span>[62] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "Distilbert, a distilled version of BERT: smaller, faster, cheaper and lighter," *CoRR*, vol. abs/1910.01108, 2019.
- <span id="page-17-5"></span>[63] X. Wang, Z. Wu, H. Chen, X. Lan, and W. Zhu, "Mixup-augmented temporally debiased video grounding with content-location disentanglement," in *ACM Multimedia*, 2023, pp. 4450–4459.
- <span id="page-17-0"></span>[64] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," vol. abs/1409.1556, 2014.
- <span id="page-17-1"></span>[65] J. Carreira and A. Zisserman, "Quo vadis, action recognition? a new model and the kinetics dataset," in *ICCV*, 2017, pp. 6299– 6308.
- <span id="page-17-3"></span>[66] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, and J. Brew, "Huggingface's transformers: State-of-the-art natural language processing," *CoRR*, vol. abs/1910.03771, 2019.
- <span id="page-17-4"></span>[67] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," *CoRR*, vol. abs/1711.05101, 2017.
- <span id="page-17-6"></span>[68] W. Jing, A. Sun, H. Zhang, and X. Li, "MS-DETR: natural language video localization with sampling moment-moment interaction," in *ACL*, 2023, pp. 1387–1400.
- <span id="page-17-7"></span>[69] A. Gu, K. Goel, and C. Ré, "Efficiently modeling long sequences with structured state spaces," *CoRR*, vol. abs/2111.00396, 2021.
- <span id="page-17-8"></span>[70] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," vol. abs/1609.02907, 2016.
- <span id="page-17-9"></span>[71] Q. Li, Z. Han, and X.-M. Wu, "Deeper insights into graph convolutional networks for semi-supervised learning," in *AAAI*, 2018, pp. 3538–3545.
- <span id="page-17-10"></span>[72] Y. Song, M. Redi, J. Vallmitjana, and A. Jaimes, "To click or not to click: Automatic selection of beautiful thumbnails from videos," in *CIKM*, 2016, pp. 659–668.
- <span id="page-17-11"></span>[73] W. Liu, T. Mei, Y. Zhang, C. Che, and J. Luo, "Multi-task deep visual-semantic embedding for video thumbnail selection," in *ICCV*, 2015, pp. 3707–3715.
- <span id="page-17-12"></span>[74] Y. Xu, Y. Sun, Y. Li, Y. Shi, X. Zhu, and S. Du, "MH-DETR: video moment and highlight detection with cross-modal transformer," *CoRR*, vol. abs/2305.00355, 2023.
- <span id="page-17-13"></span>[75] W. Moon, S. Hyun, S. Park, D. Park, and J.-P. Heo, "Querydependent video representation for moment retrieval and highlight detection," in *ICCV*, 2023, pp. 23 023–23 033.
- <span id="page-17-14"></span>[76] K. Q. Lin, P. Zhang, J. Chen, S. Pramanick, D. Gao, A. J. Wang, R. Yan, and M. Z. Shou, "Univtg: Towards unified video-language temporal grounding," in *ICCV*, 2023, pp. 2794–2804.

![](_page_17_Picture_18.jpeg)

**Kun Li** is currently pursuing the Ph.D. degree in the School of Computer Science and Information Engineering, Hefei University of Technology, China. His research interests include multimedia content analysis, computer vision, and video understanding. He regularly serves as a PC Member for top-tier conferences in multimedia and artificial intelligence, like ACM Multimedia, IJCAI, AAAI, CVPR, ICCV, and ECCV.

![](_page_17_Picture_20.jpeg)

**Zhan Si** received the B.E. degree from Shenyang University of Chemical Technology, China, in 2022. He is currently pursuing the master's degree in the School of Chemistry and Chemical Engineering, Anhui University, China. His research interests include AI for science and computational chemistry.

![](_page_17_Picture_22.jpeg)

**Xun Yang** is currently a Professor with the Department of Electronic Engineering and Information Science, University of Science and Technology of China (USTC). His research interests include information retrieval, cross-media analysis and reasoning, and computer vision. He served as the Area Chair for the ACM Multimedia 2022. He also serves as the Associate Editor for the IEEE TRANSACTIONS ON BIG DATA journal.

![](_page_17_Picture_24.jpeg)

**Jingjing Hu** received the B.E. degree in the Internet of Things from Northeastern University, China, in 2022. She is currently pursuing the master's degree in the School of Computer Science and Information Engineering, Hefei University of Technology, China. Her research interests include multimedia content analysis, computer vision. She currently serves as a reviewer of ACM Multimedia conference.

![](_page_17_Picture_26.jpeg)

**Xiaojun Chang** (Senior Member, IEEE) is currently a Professor at the School of Information Science and Technology (USTC). He has spent most of his time working on exploring multiple signals (visual, acoustic, and textual) for automatic content analysis in unconstrained or surveillance videos. He has achieved top performances in various international competitions, such as TRECVID MED, TRECVID SIN, and TRECVID AVS.

![](_page_17_Picture_28.jpeg)

**Dan Guo** (Senior Member, IEEE) is currently a Professor with the School of Computer Science and Information Engineering, Hefei University of Technology, China. Her research interests include computer vision, machine learning, and intelligent multimedia content analysis. She serves as a PC Member and for top-tier conferences and prestigious journals in multimedia and artificial intelligence, like ACM Multimedia, IJCAI, AAAI, CVPR and ECCV. She also serves as a SPC Member for IJCAI 2021.

![](_page_17_Picture_30.jpeg)

**Meng Wang** (Fellow, IEEE) is currently a Professor with the Hefei University of Technology, China. His current research interests include multimedia content analysis, computer vision, and pattern recognition. He was a recipient of the ACM SIGMM Rising Star Award 2014. He is an Associate Editor of the IEEE TRANS-ACTIONS ON KNOWLEDGE AND DATA EN-GINEERING, the IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECH-NOLOGY, and the IEEE TRANSACTIONS ON

NEURAL NETWORKS AND LEARNING SYSTEMS.

# **APPENDIX A**

# OVERALL PREDICTION ANALYSIS FOR BOTH NLVG AND SLVG TASKS

Fig. 13 shows the temporal distribution of target moments on the ActivtyNet Captions, Charades-STA, and TACoS datasets for NLVG task, the distribution of target moments varies among the these datasets, and our method has good predictive performance than MMN [25] on all these datasets, indicating that the model has good robustness. Fig. 14 shows the temporal distribution of target moments on the ActivtyNet Speech, Charades-STA Speech, and TACoS Speech datasets for SLVG task, it is clear to see that when audio is used as the query, the MMN approach is clearly missing some important moment regions in the predicted response on the ActivityNet Speech and TACoS Speech datasets, whereas our approach responds more comprehensively to the moment peak regions shown by groundtruth. It is worth noting that the distribution of video grounding results using text and audio as queries differ for both MMN and our UniSDNet-M. It is reasonable to assume that the predictions using text are closest to groundtruth, text-based TVG outperforms audio-based TVG due to its superior representation capability from pre-training technique.

![](_page_18_Figure_5.jpeg)

<span id="page-18-0"></span>Fig. 13. The distribution of predicted moments by our UniSDNet-M and MMN [25] on ActivtyNet Captions, Charades-STA, and TACoS datasets for NLVG task. While MMN's predictions are more centrally biased towards regions of high density, our model fits the true distribution of the target moments to a greater extent.

![](_page_19_Figure_2.jpeg)

<span id="page-19-0"></span>Fig. 14. The distribution of predicted moments by our UniSDNet-M and MMN [25] on ActivtyNet Speech, Charades-STA Speech, and TACoS Speech datasets for SLVG task. While the prediction of MMN clearly ignores some important regions, *e.g.*, on the ActivtyNet Speech dataset, very few results correspond to the left centre and right centre of the distributions for [start index = 0, end index = 15] and [start index = 40, end index = 60], our model also fits the true distribution of the target moments to a greater extent when using the spoken language as the query.

# APPENDIX B

# ADDITIONAL EXPERIMENTAL RESULTS

In this section, we conduct a series of ablation studies to evaluate the hyperparameter k in graph construction, as well as various model settings including the method of adding positional encodings in the feature encoding stage and the semantic matching function in the model's decoding stage.

#### B.1 Ablation Study on Hyperparameters k in Graph Construction

The hyperparameter k in Eq. 2 of the main paper, the dividing value between short and long distances in the video graph, is a empirical parameter, which is tuned on validation set with the final model are tested on test set. Table 14 shows the ablation experiments on hyperparameter k, and Fig. 15 visualizes the video graph connectivity matrix with different k value. From the experimental results, the optimal value of k on all three datasets is 16. Either too small or too large a k value can impair performance, a small k value overly focuses on short-distance information, neglecting long-distance

dependencies in videos, while a large k value adds more redundant edges, increasing the difficulty for the model to recognize video events.

![](_page_20_Figure_3.jpeg)

<span id="page-20-1"></span>Fig. 15. The adjacency matrices for different k values (4, 8, 16, and 32), with the number of video clips T fixed at 64 (row i corresponds to node  $v_i$ , column i corresponds to node  $v_j$ ). Each sub-figure is a binary value  $\{0,1\}$  that shows valid connections between start and end indexes in the video graph.

<span id="page-20-0"></span>TABLE 14

Ablation study on hyperparameter k for NLVG task.  $L_V$  denotes the average duration of videos in a dataset. T is the number of sampled clips, which is consistent with the settings in [9], [11], [19] for experiment fairness. Here, we use currently popular video features (ActivityNet Captions, C3D) [11], (Charades-STA, I3D) [19], (TACoS, C3D) [9] respectively.

| Detect      | T (-)    | /TI | ,  |         | R       | @1      |         |         | R       | @5      |         |       |
|-------------|----------|-----|----|---------|---------|---------|---------|---------|---------|---------|---------|-------|
| Dataset     | $L_V(s)$ | T   | k  | IoU@0.1 | IoU@0.3 | IoU@0.5 | IoU@0.7 | IoU@0.1 | IoU@0.3 | IoU@0.5 | IoU@0.7 | mIoU  |
|             |          |     | 4  | 89.79   | 73.59   | 57.92   | 35.50   | 96.26   | 90.59   | 84.53   | 73.20   | 53.45 |
| ActivityNet |          |     | 8  | 90.12   | 74.97   | 60.03   | 37.20   | 96.26   | 90.75   | 84.66   | 73.21   | 54.53 |
| Captions    | 117.60   | 64  | 16 | 90.28   | 75.85   | 60.75   | 38.88   | 96.32   | 91.17   | 85.34   | 74.01   | 55.47 |
| •           |          |     | 32 | 90.04   | 74.87   | 60.05   | 36.92   | 96.09   | 90.72   | 84.45   | 72.77   | 54.48 |
|             |          |     | 64 | 89.99   | 74.79   | 59.67   | 37.17   | 96.08   | 90.42   | 84.48   | 72.17   | 54.44 |
|             |          |     | 4  | 78.04   | 70.94   | 58.15   | 37.42   | 98.06   | 95.91   | 89.60   | 70.99   | 50.99 |
| Charades-   |          |     | 8  | 78.44   | 71.29   | 58.44   | 38.98   | 97.82   | 95.81   | 89.68   | 72.72   | 51.60 |
| STA         | 30.60    | 64  | 16 | 79.44   | 72.18   | 61.02   | 39.70   | 97.55   | 95.35   | 89.97   | 73.20   | 52.69 |
|             |          |     | 32 | 78.17   | 70.83   | 58.23   | 39.33   | 97.72   | 95.97   | 90.30   | 73.23   | 51.67 |
|             |          |     | 64 | 78.68   | 71.53   | 58.76   | 38.33   | 98.06   | 96.42   | 89.73   | 71.96   | 51.66 |
|             |          |     | 4  | 68.03   | 53.34   | 37.69   | 21.94   | 88.63   | 76.63   | 63.03   | 35.67   | 37.17 |
|             |          |     | 8  | 69.78   | 53.19   | 38.64   | 22.14   | 87.83   | 75.43   | 63.16   | 35.14   | 37.84 |
| TACoS       | 286.59   | 128 | 16 | 70.78   | 55.56   | 40.26   | 24.12   | 89.85   | 77.08   | 64.01   | 37.02   | 38.88 |
|             |          |     | 32 | 66.73   | 53.06   | 37.72   | 21.87   | 88.75   | 75.88   | 63.31   | 35.34   | 36.90 |
|             |          |     | 64 | 68.93   | 51.59   | 34.82   | 18.22   | 88.58   | 76.66   | 63.01   | 32.94   | 35.50 |

# B.2 Ablation Study on Adding Position Embeddings for Video and Query

The function of position embedding (PE) is to help the model understand the relative position and order of different elements in the sequence, and thus better capture the semantic information in the sequence. We add sine position embedding [28] to the input video clip and query sequence, in order to enhance the temporal relationships between video sequences and the logical relationships between queries. Considering that most existing multimodal Transformers add independent PEs to different modalities, in order to distinguish modality-specific information, and arcitecturally, the static module with ResMLP structure of our model is similar to Transformer in processing multi-modal sequences in parallel [47]. We simply follow existing work, adding independent PEs for video and queries, as shown in Fig. 16. Specifically, we denote the PE for each video clip  $v_i$  or query  $q_i$  as:

$$PE(o_i) = \begin{cases} sin(i/10000^{j/d}), & \text{if } j \text{ is even} \\ cos(i/10000^{j/d}), & \text{if } j \text{ is odd} \end{cases}$$
(11)

where  $PE(o_i) \in \mathbb{R}^{1 \times d}$ ,  $o_i$  denotes  $v_i$  or  $q_i$ , and j varies from 1 to d dimension. We set up two different ways to add PE: adding Independent PE and adding Joint PE. These two ways of adding PE correspond to "w/. Independent PE" and "w/. Joint PE" in Table 15, respectively. The results demonstrate that the inclusion of PEs significantly improves the model's performance. On the ActivityNet Captions dataset, the R@1, IoU@0.7 score improved from 30.16% to 36.96%. Similarly, on the TACoS dataset, the R@1, IoU@0.5 score increased from 30.94% to 36.84%. The setting of "w/. Independent PE" gives better results than that of "w/. Joint PE", which demonstrates the superiority of adding independent PE.

![](_page_21_Figure_2.jpeg)

<span id="page-21-0"></span>Fig. 16. Illustration of two different ways to add position embedding (PE) for different modality features (query and video).

#### TABLE 15

<span id="page-21-1"></span>Ablation Study on adding position embedding for different modality features. The setting of "w/o. PE" refers to the model without any position embedding; the setting of "w/. Joint PE" refers to the model with the joint position encoding added to all modalities; the setting of "w/. Independent PE" refers to the model with independent positional embedding for different modalities (queries and video clips).

| Dataset                 | Model Setting      | R@1     |         |         |         |         |         |         |         |       |
|-------------------------|--------------------|---------|---------|---------|---------|---------|---------|---------|---------|-------|
|                         |                    | IoU@0.1 | IoU@0.3 | IoU@0.5 | IoU@0.7 | IoU@0.1 | IoU@0.3 | IoU@0.5 | IoU@0.7 | mIoU  |
| ActivityNet<br>Captions | w/o. PE            | 88.42   | 70.11   | 54.12   | 30.16   | 95.93   | 89.89   | 81.43   | 67.82   | 49.91 |
|                         | w/. Joint PE       | 89.88   | 74.18   | 58.29   | 36.96   | 96.14   | 90.12   | 83.67   | 72.08   | 53.96 |
|                         | w/. Independent PE | 90.28   | 75.85   | 60.75   | 38.88   | 96.32   | 91.17   | 85.34   | 74.01   | 55.47 |
| Charades-<br>STA        | w/o. PE            | 71.99   | 63.60   | 50.56   | 31.18   | 97.61   | 94.68   | 86.53   | 64.78   | 45.09 |
|                         | w/. Joint PE       | 77.96   | 70.81   | 57.80   | 36.53   | 98.09   | 96.08   | 90.13   | 71.18   | 50.81 |
|                         | w/. Independent PE | 79.44   | 72.18   | 61.02   | 39.70   | 97.55   | 95.35   | 89.97   | 73.20   | 52.69 |
| TACoS                   | w/o. PE            | 62.48   | 45.21   | 30.94   | 16.97   | 88.30   | 75.01   | 58.69   | 31.22   | 31.99 |
|                         | w/. Joint PE       | 65.96   | 50.81   | 36.84   | 19.70   | 90.85   | 79.18   | 65.28   | 35.02   | 35.59 |
|                         | w/. Independent PE | 70.78   | 55.56   | 40.26   | 24.12   | 89.85   | 77.08   | 64.01   | 37.02   | 38.88 |

# B.3 Ablation Study on Modality Alignment Measurement Method

In this part, we investigate different cross-modal semantic similarity matching methods. The ablation study in Table 16 compares cosine similarity used in "Section 3.4 2D Proposal Generation", Eq. 6 of the main paper, with other similarity measures, including (1) Mean Hadamard product:  $\operatorname{Hada}_{\operatorname{Mean}}(S^{\mathcal{M}}, S^{\mathcal{Q}}) = \frac{1}{d} \sum_{i=1}^{d} (S_{i}^{\mathcal{M}} \odot S_{i}^{\mathcal{Q}})$ . (2) Euclidean distance measures the straight-line distance between two vectors and is defined as:  $\operatorname{E-Dis}(S^{\mathcal{M}}, S^{\mathcal{Q}}) = \sqrt{\sum_{i=1}^{d} (S_{i}^{\mathcal{M}} - S_{i}^{\mathcal{Q}})^{2}}$ . (3) Manhattan distance, also known as L1 distance, is calculated as:  $\operatorname{M-Dis}(S^{\mathcal{M}}, S^{\mathcal{Q}}) = \sum_{i=1}^{d} |S_{i}^{\mathcal{M}} - S_{i}^{\mathcal{Q}}|$ . From Table 16, it is evident that cosine similarity performs best, and the Hadamard product provides competitive results. Based on these findings, we confirm that cosine similarity is an effective measure for our semantic matching module. Nevertheless, the alternative similarity measures provide valuable insights and potential areas for further exploration.

<span id="page-21-2"></span>TABLE 16 Ablation Study on different similarity measure functions. We report the experimental results of similarity measures: "cosine" CoSine( $\cdot, \cdot$ ), "Mean Hadamard product" Hada<sub>Mean</sub>( $\cdot, \cdot$ ), "Euclidean distance" E-Dis( $\cdot, \cdot$ ), and "Manhattan distance" M-Dis( $\cdot, \cdot$ ).

| Dataset                 | Semantic Matching                   | R@1     |         |         |         | R@5     |         |         |         |       |
|-------------------------|-------------------------------------|---------|---------|---------|---------|---------|---------|---------|---------|-------|
|                         | Measure Method                      | IoU@0.1 | IoU@0.3 | IoU@0.5 | IoU@0.7 | IoU@0.1 | IoU@0.3 | IoU@0.5 | IoU@0.7 | mIoU  |
|                         | $CoSine(\cdot, \cdot)$              | 90.28   | 75.85   | 60.75   | 38.88   | 96.32   | 91.17   | 85.34   | 74.01   | 55.47 |
| ActivityNet<br>Captions | $Hada_{\mathrm{Mean}}(\cdot,\cdot)$ | 89.08   | 74.23   | 58.89   | 36.85   | 95.94   | 90.59   | 84.73   | 73.16   | 54.04 |
|                         | $\text{E-Dis}(\cdot,\cdot)$         | 89.37   | 74.68   | 58.40   | 35.83   | 95.96   | 90.32   | 84.16   | 71.26   | 53.57 |
|                         | $\text{M-Dis}(\cdot,\cdot)$         | 89.06   | 73.21   | 56.68   | 34.01   | 96.29   | 90.66   | 84.45   | 72.93   | 52.69 |
|                         | $CoSine(\cdot, \cdot)$              | 79.44   | 72.18   | 61.02   | 39.70   | 97.55   | 95.35   | 89.97   | 73.20   | 52.69 |
| Charades-<br>STA        | $Hada_{\mathrm{Mean}}(\cdot,\cdot)$ | 78.68   | 71.53   | 58.76   | 38.33   | 98.06   | 96.42   | 89.73   | 71.96   | 51.66 |
|                         | $E	ext{-Dis}(\cdot,\cdot)$          | 78.01   | 70.59   | 57.28   | 37.37   | 98.31   | 96.29   | 90.27   | 71.64   | 50.82 |
|                         | $\text{M-Dis}(\cdot,\cdot)$         | 77.72   | 69.95   | 57.47   | 37.26   | 97.77   | 95.43   | 89.25   | 70.97   | 50.48 |
|                         | $CoSine(\cdot, \cdot)$              | 70.78   | 55.56   | 40.26   | 24.12   | 89.85   | 77.08   | 64.01   | 37.02   | 38.88 |
| TACoS                   | $Hada_{\mathrm{Mean}}(\cdot,\cdot)$ | 69.51   | 54.19   | 38.59   | 23.03   | 89.65   | 78.78   | 64.48   | 35.87   | 37.60 |
|                         | $E	ext{-Dis}(\cdot,\cdot)$          | 68.03   | 53.34   | 37.69   | 22.94   | 89.63   | 77.63   | 64.03   | 35.67   | 37.17 |
|                         | $M	ext{-Dis}(\cdot,\cdot)$          | 66.58   | 53.11   | 37.44   | 22.87   | 88.78   | 75.43   | 62.76   | 35.09   | 36.90 |

# APPENDIX C More Visualization of Prediction Results

In order to clearly demonstrate the specific role of our proposed unified static and dynamic networks in cross-modal video grounding, we provide more challenging visualization cases in this section as a supplement to Sec. 5.7.

# **C.1 Visualization on ActivityNet Captions for NLVG**

**Video Sample with Complex Scene Transitions.** The ActivityNet Captions dataset contains a large amount of open-world videos with more shot transitions. We choose typical samples of this type for visualisation and analysis. As shown in Fig [17](#page-23-0) (a), there are multiple scene transitions in video sample "ID: v\_rKtktLDSOpA" from the ActivityNet Captions dataset and different events have serious intersection in the temporal sequence of video. For example, there is an intersection between the end of the moment corresponding to Q1 and the beginning of the moment corresponding to Q2 and another big intersection exists between the moments corresponding to Q2 and Q3. From Fig. [17,](#page-23-0) **MMN** [\[25\]](#page-16-10) makes a serious prediction for Q1, locating the moment corresponding to Q2. Meanwhile, when predicting Q3, **MMN** omits the temporal region intersected with Q2 but correct temporal region also belonged to the moment of Q3 for the final prediction. Compared to **MMN**, our **Only Static** and **Only Dynamic** predict more accurate moments for each query, and they can accurately comprehend the intersection of Q2 and Q3. **Only Static** performs better at identifying transitions, while **Only Dynamic** performs better at recognizing overlapping events. Our **Full Model** performs best in these challenging scenarios because it combines the advantages of **Only Static** and **Only Dynamic**.

**Video Sample with Similar Scenes.** For the NLVG task that employs textual queries, it is also challenging to use the semantic guidance of the text to distinguish some video clips that are similar in the front and back frames (without transitions). As shown in Fig. [17](#page-23-0) (b), the frames in video sample "ID: v\_UajYunTsr70" from the ActivityNet Captions dataset also have high similarity, you can find it to locate the corresponding moment corresponding to Q1: "*A cat is sitting on top of a white sheet.*" **MMN** is basically unable to distinguish the video content for the three different queries. It almost predicts the entire video for each query. Even through our **Only Static** performs poorly in this situation too, our **Only Dynamic** performs much better than MMN. Finally, our **Full model** locates the most accurate target moment. This is thanks to our model that combines the advantages of static and dynamic modules, especially for that the latter learns a tighter contextual correlation of video in this case.

# **C.2 Visualization on ActivityNet Speech for SLVG.**

We also provide quantitative results of our UniSDNet on SLVG to demonstrate the effectiveness of our model in the video grounding task based on spoken language.

**Video Sample with Noisy Background.** When using audio as a query, we prefer to analyze how well the model understands the interaction between audio and video by performing visualizations of video cases that contain more background noise. We instantiate the video sample "ID: v\_FsS8cQbfKTQ" from the ActivityNet Speech dataset in Fig. [18](#page-24-0) (a) using audio queries under noisy background interference. We can see that **MMN** predicts the video clips corresponding to Q2 and Q3 with significant deviations, and the predicted moments totally do not intersect with **GT** at all. This video is a challenging case. Compared to **MMN**, **Only Static** and **Only Dynamic** coverage the queried moment but have somewhat boundary shifts, exhibits a strong advantage, as it correctly predicts the relative positions of all events has a large intersection ratio with **GT** video clips. Compared to **MMN**, our **Full model** exhibits the best prediction results for all queries, as it correctly predicts the queried moment and has a large intersection ratio with **GT** video clips. From the 2D map in the figure, it can be seen that our model still performs well in video grounding task based on audio queries, fully demonstrating the generalization of our model.

**The Videos with Continuous and Varied Actions.** Similarly to the NLVG task, we analyse the video case without transitions but with continuous action changes for the SLVG task to quantify the model's ability of identifying event boundaries. Taking video sample "ID: v\_UJwWjTvDEpQ" from the ActivityNet Speech dataset in Fig. [18](#page-24-0) (b) as an example, the video shows a scene with a clean background, but in which a boy's actions are continuously changing. In this case, for different event divisions, it is necessary to finely distinguish the contentual semantics of the boy's actions and the differences between them. **MMN** fails to recognize such densly varied actions and incorrectly assigns the entire video as the answer (*e.g*., Q1 and Q2). **Our Static** predicts the approximate location of each event. **Our Dynamic** exhibits excellent performance in distinguishing the semantics of continuous actions, it not only correctly distinguishes the semantic centers of three events, but also more accurately predicts the boundaries of each event, compared to **MMN** and **Our Static**. Inspiring, **Full Model** achieves the most accurate prediction of the location and semantic boundaries of events, this is thanks to the combination of static and dynamic modes, which deepens the understanding of video context and enables the model to distinguish different action semantics.

# **C.3 More Visualization of Plethoric Multi-query Cases**

**Visualization Examples on the TACoS Dataset.** Taking the video sample "ID: s27-d50" in Fig. [19](#page-25-0) (a) as an example, we provide the grounding results of our model and MMN. Note that the total duration of the video is 82.11 s, which includes 119 query descriptions. Limited by page size and layout, we select and show 6 very challenging queries here. The video depicts a person cooking in a kitchen. **MMN** experiences a significant prediction error in the moment corresponding to the query Q88. On the contrary, our **Full model** accurately determines the relative positions of the video segments corresponding to all queries. The qualitative results highlight the effectiveness of learning semantic associations between multi-queries (*i.e*., multi-queries contextualization) for cross-modal video grounding.

**Visualization Examples on the Charades-STA Dataset.** The video sample "ID: U5T4M" in Fig. [19](#page-25-0) (b) has a duration of 19.58 s, which describes the indoor activities of a person, and contains 7 queries. Our **Full model** infers the localization

![](_page_23_Figure_2.jpeg)

<span id="page-23-0"></span>Fig. 17. Qualitative examples on ActivityNet Captions for NLVG. (a) The video contains complex scene transitions and overlap. (b) The video scenes that are difficult to distinguish. **MMN** makes significant errors in predicting the location range of the queried events, *i.e.*, Q1 and Q3 in cases (a) and (b), respectively. Our **Only Static** has an advantage in predicting transitions (Q1 in case (a)), our **Only Dynamic** performs better in predicting overlapping. It is difficult to distinguish scenarios (Q2 and Q3 in both cases (a) and (b)). Our **Full Model** performs best in both challenging scenarios, as it combines the advantages of static (query semantic differentiation) and dynamic (video sequence context association) modules.

results of all queries corresponding to the video at once. In all queries, Q1 and Q2 are similar descriptions of an event, respectively. The same situation also includes queries of Q4 and Q5, Q6 and Q7. Our **Full model** accurately predicts the boundaries of each query, and effectively distinguishing the semantics among similar but with slightly different events.

![](_page_24_Figure_2.jpeg)

<span id="page-24-0"></span>Fig. 18. Qualitative examples on ActivityNet Speech for SLVG. (a) The scenes that contains a noisy background. (b) The Videos with Continuous and Varied Actions. **MMN** makes significant errors in predicting the location (Q2 and Q3 in case (a)) and location coverage areas of events (Q1 and Q2 in case (b)). These two cases are challenging. Encouragingly, our **Full Model** achieves the best performance in these video grounding cases based on audio queries, which confirms the effectiveness and generalization of our unified static and dynamic methods in this task.

![](_page_25_Figure_2.jpeg)

<span id="page-25-0"></span>Fig. 19. Quantitative examples of plethoric multi-query cases. **(a)** Examples on the TACoS dataset for NLVG. **(b)** Examples on the Charades-STA dataste for NLVG. **MMN** has a significant semantic bias when predicting Q7 in case (a), and Q4, Q5, Q7 in case (b), there is also a large positional deviation in predicting Q88 in case (a), and Q1, Q3 in case (b). Our **Full Model** correctly predicts the location of all the queried events, and the predicted moment interval is closest to that of **GT**, this is thanks to model capacity of mutual learning of video and multiple queries and effectively capturing the video context associated with multiple queries.