# **Known-Item Video Search via Query-to-Modality Mapping**

Kong-Wah Wan Institute for Infocomm Research 1 Fusionopolis Way Singapore 138632 kongwah@i2r.a-star.edu.sg

Yan-Tao Zheng Institute for Infocomm Research 1 Fusionopolis Way Singapore 138632 yzheng@i2r.a-star.edu.sg

Lekha Chaisorn Institute for Infocomm Research 1 Fusionopolis Way Singapore 138632 clekha@i2r.a-star.edu.sg

# **ABSTRACT**

We introduce a novel query-to-modality mapping approach to the TRECVid 2010 known-Item video search (KIS) task. To search for a specific target video, a KIS query is verbose with many multi-modal attributes. Issuing all search terms to a retrieval engine will confuse the search criteria in different modalities and result in "topic drift". We propose decomposing a KIS query into a set of short uni-modal subqueries and issue them to the search index of the corresponding modality features, such as text-based metadata, visualbased high-level features. To do so, we introduce novel syntactic query features and cast the query-to-modality mapping as a classification problem. Retrieval results on the TRECVid 2010 KIS dataset shows that our approach outperforms existing methods by a significant margin.

## **Categories and Subject Descriptors**

H.3.3 [**Information Search and Retrieval**]: Query formulation

## **General Terms**

Algorithms, Languages, Performance

## **Keywords**

Multimedia Search, Query Segmentation

# **1. INTRODUCTION**

The known-item video search (KIS) models the scenario in which the searcher has seen and known of a video before, but does not know where to look in the video corpus [1]. He begins his search session by issuing a lengthy text query describing the target video (Figure 1). The search engine returns a list of videos, where only one is the correct answer.

The average KIS query is verbose and comprises of many multi-modal search attributes. A running example is TRECV

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. To copy otherwise, to republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee.

*MM'11,* November 28–December 1, 2011, Scottsdale, Arizona, USA. Copyright 2011 ACM 978-1-4503-0616-4/11/11 ...\$10.00.

![](_page_0_Picture_18.jpeg)

**Figure 1: The Video Known-Item-Search scenario**

KIS query 24: *"Find the video with a man wearing a red jacket, glasses and a white baseball cap, powerwashing his fence"*. Issuing all the search terms into a retrieval engine will confuse it with many search modalities and result in "topic drift". To address this problem, we propose decomposing a KIS query into short uni-modal subqueries, each embodying a different *attribute type* of the target object, and issued to a retrieval engine that is indexed on features of the corresponding modality type. Using our example, the query is segmented into a text-attribute set: {*"powerwashing fence"*}, and a visual-attribute-set: {*"red jacket", "glassess", "white baseball cap"*}. The former is issued to a metadata search engine, and the latter to a High-Level-Feature-(HLF) based system. The two result lists can then be fused.

We assume there are grammatical and syntactic patterns governing the description of queried objects and their attributes, e.g. *"man in white shirt", "girl with brown hair"*, and use REGEX surface patterns [15] and syntactic features to capture these lexico-syntactic rules. On a text-attributed segment, we learn a classifier to identify the key concepts for text search. Query terms are represented by standard statistical and part-of-speech features, and dependency parsing. For visual-attributed segments, we apply visual detectors for the detected key concepts with query-to-detector mapping.

We validate our proposed method on the TRECVid 2010 dataset [1] comprising 8000 Internet Archive videos totaling 300 hours. Results show that our method outperforms standard baseline methods by a significant margin.

<sup>∗</sup> Area chair: Qi Tian

### 2. RELATED WORK

Known-item search (or known-item finding) has been a popular information seeking task in text retrieval. In this task, the user attempts to locate a web page, an email, or a report that he has known of [11]. Recently, the TRECVid forum extends this concept to formulate the known item search task for video. Because of the multi-modality nature of video, queries in video KIS are much more complex because they contain search terms on both the topical semantics and audio/visual attributes of video content.

There are recent attempts to apply existing video retrieval techniques on video KIS. Fusion of meta-data text and visual features are explored by Snoek et. al [14]. Ngo et. al applied query-to-concept mapping to leverage a set of 130 high level concepts for visual search [10]. In contrast, we focus on handling complex and verbose KIS queries and mapping query terms to their search modalities.

Analyzing complex queries has been a popular research topic in information retrieval. Risvik  $et.\ al$  first described web query term selection using mutual information and segment frequency [12]. Bendersky and Croft developed a probabilistic model to identify query keywords using features from TF, IDF, etc [2]. Bergsman and Wang applied supervised learning on similar features plus dependency features to segment a query into compound concepts [3]. In this paper, while we have used similar features and learning approach, we introduce additional novel syntactic dependency features to improve keyword extraction, and undertake the additional task of locating the A/V search terms.

#### 3. QUERY-TO-MODALITY MAPPING

Without loss of generality, we constrain our discussion to two modality types: text (video metadata) and visual HLF.

#### 3.1 Ouery Terms for Visual Modality

We introduce ways to identify the visual attributes (VA) of KIS queries as query terms of visual modality. First, there is a noticeable consistency in KIS queries: VA phrases are usually anchored by active verbs (VBG) such as "wearing a...", "standing by...", "sitting on..." or prepositions (PP) like "with", "in". Therefore, we first learn a set of surface patterns from an annotated data set of training queries manually marked with visual attributes.

Next, because surface patterns cannot handle long-range dependencies, we augment the REGEX rules with POS tags, and impose the rule that VA segments must not contain proper-noun (NNP) phrases. Then we analyze the term dependency relations from a dependency parser [9] to capture conjunctional constructs (Figure 2). For example, in "a man wearing a red jacket, glasses, a white baseball cap", "wearing" is connected with "red jacket", "glassess" and "baseball cap", via the dobj relation. Semantically, it becomes: "a man wearing a red jacket, wearing glassess, wearing a white baseball cap", making "glasses" and "a white baseball cap" part of the VA. In this expanded form, our surface patterns can properly identify all three "wearing" VA of the man.

#### 3.2 Ouerv Terms for Text Modality

The semantic content of a user's information need is usually expressed by the key terms in the query. In the following, we describe how the key terms can be selected. We also assume they can be captured by a ranking function.

![](_page_1_Figure_11.jpeg)

Figure 2: Typed dependency tree of KIS query 24

### 3.2.1 Selecting Key Query Terms

Noun Phrase (NP): The key concepts in a query are often embodied in the NP [2]. We extract NP using a parser [9]. If there are proper nouns (NNP), only they are selected as a compound keyword enclosed in quotes. Hence, given: members of the Coast Guard sing the slow hit song, the two resulting keywords are: {"Coast Guard"}, {slow hit song}

**Dependency-Relations:** Keywords are also extracted from the dependency parse (DP) tree. This can uncover related terms over multi-word and POS boundaries. For example, given: tavern with round tables and dancing candlesticks, the DP captures two keywords: {tavern tables}, {tavern candlesticks} connected by the prep\_with relation.

### 3.2.2 Ranking Key Query Terms

We adopt a supervised learning approach and make use of the following term weighting features to derive the final key query term rank:

tf(K): frequency of term K in the corpus.

idf(K): inverse of the document frequency (df) of the query term in the corpus:  $idf(K) = \log \frac{N}{df(K)}$ 

wig(K): weighted information gain of the query term K, based on the relative normalized entropy of the concept in corpus C and in the collection D of the top 50 retrieved documents when K is issued as query:

$$wig(K) = \frac{\frac{1}{50} \sum_{d \in D} p(K|d) - \log p(K|C)}{-\log p(K|C)}$$
(1)

 $ng\_wiki(K)$ : Given K, we form all valid segmentations S comprising n-gram (n=1..4) sub-segment s, and score S by an exponential sum of the frequency count(s) of s in the English Wikipedia 2010-10-11 dump:

$$ng\_wiki(K) = \max_{S} \left\{ \sum_{s \in S} |s|^{|s|} \cdot count(s) \right\}$$
 (2)

pos(K): the histogram of the POS tags of the three adjacent terms for each word of the key query term.

 $dp\_in(K)$ : total number of dependency links of K in the dependency tree. We expect that key terms are semantically related to many other terms, and characterized by their hub-like structures in the dependency tree.

### 3.3 Building HLF Detectors from Key Visual Query Terms

Similar to learning key concepts from textual attributes, key visual concepts in VA can be selected from noun phrases, e.g. *shirt, chair, sky, beach*, for which visual detectors can be built to augment retrieval. For this, we can turn to existing detectors [7]. However, there are problems with this

![](_page_2_Figure_0.jpeg)

Figure 3: Top: Sample attribute images for training. Botton: The top five attribute predictons in images. Correct predictions are shown in bold.

approach. First, for many visual keywords such as *hair*, santa-cap, there is no detector available. Second, concepts in a VA segment are mostly local visual attributes of objects, and differ from the global descriptors in existing detectors.

For the purpose of showing the utility of visual-attributed analysis in this paper, we follow recent trends in describing objects with attributes [8], to learn new detectors for the key visual concepts. We choose 20 key visual concepts with the smallest semantic gap, estimated in terms of their interand intra-class visual variation [5]: sky, beach, brick wall, santa-cap, country-side, shirt, chair, courtyard, pants, suit computer, screen, apartment, sofa, door, hair, coat, sunglass, beard, microphone. For a keyword that is outside this list, we simply discard its visual detection.

For each keyword, we downloaded and manually prepared Google images for training. As negative examples, we randomly sampled from other keyword classes. All attribute images are resized to 128x128, and a broad variety of visual features is used: LAB-color, texton and SIFT visual words. Each feature type is respectively quantized into 64, 128, 300 K-means centroids, forming a 492-d bag-of-word image vector. A linear SVM is used as the classifier. To detect the presence of an attribute in an image, the attribute classifier is scanned over multiple locations and scales. The final output is a 20-d histogram score vector of keyword attributes. Figure 3 shows some examples of training attribute images, and the attribute predictions in images.

### 3.4 Fusion

We use linear fusion to combine retrieval results from all N attributes: Denote  $f_k(x)$  to be the  $k^{th}$ -modality retrieval score of input video x. The final retrieval score is:  $R(x) = \sum_{k=1}^{N} \alpha_k f_k(x)$ , where  $\alpha_k$  is the weight of the  $k^{th}$ -modality retrieval method and  $\sum_k \alpha_k = 1$ . The weights for  $\alpha$  is determined by a grid-search maximizing the  $\mathbf{AP}$  of R on the labeled training set, where each  $\alpha_k$  is constrained to  $\{0,0.05,0.1,\ldots,1.0\}$ .

# 4. EVALUATION

# 4.1 Query-to-Modality Mapping Results

We evaluate the proposed methods using the TRECVid 2010 dataset. This comprises of 8000 Internet Archive videos totaling 300 hours. Each video has a metadata text descrip-

Table 1: Query-to-Modality Mapping results

| Visual query term classification  |          |           |          |  |  |  |  |
|-----------------------------------|----------|-----------|----------|--|--|--|--|
|                                   | Recall   | Precision | Accuracy |  |  |  |  |
| CRF++ [13]                        | 85.6     | 81.2      | 81.8     |  |  |  |  |
| Our method                        | 85.7     | 81.7      | 82.5     |  |  |  |  |
| Textual query term classification |          |           |          |  |  |  |  |
|                                   | Proposed | idf       | wig      |  |  |  |  |
| Accuracy                          | 72.3     | 52.4      | 65.7     |  |  |  |  |
| Mean Reciprocal Rank              | 81.4     | 71.2      | 77.3     |  |  |  |  |

tion averaging only 50 words. To facilitate the evaluation of intermediate results, we also develop two ground truth datasets: **D1**: annotation of query terms and its corresponding modality, and **D2**: optimal subqueries for text modality.

Query Terms for Visual Modality: The query dataset is splitted into a train-set (query 1 to 200), and the others used as test-set. To measure classification performance, we use standard precision, recall, and accuracy of correct assignment. We compare our method with an CRF++ stopphrase removal method [13]. Comparable results are shown in Table 1, and indicate that a simple method based on surface patterns and syntactic features can produce competitive results with a state-of-art sequential tagger.

Query Terms for Text Modality: For the supervised learning tool, we use the TiMBL Memory-Based Learning software [4], We use 10-fold cross validation, and the average results over all ten runs are reported. Results are evaluated using Accuracy and  $Mean\ Reciprocal\ Rank$ . To see the advantage of our supervised learning approach, we also compare against a simple non-supervised weighting method based on two features: the inverse document frequency (idf) and the weighted information gain (wig) (see Section 3.2.2). Table 1 tabulates the results. They show that the supervised learning approach produces the best classifier.

#### 4.2 Known-Item Video Search Results

In this sub-section and the next, we evaluate the performance of known-item search using the proposed method. The evaluation criteria for KIS is the mean inverted rank (MIR), at which the known item is found.

Retrieval on Text Modality: We first utilize the original query that undergoes standard query pre-processing such as stopword removal. We then issue the subqueries obtained from the proposed method, the human-annotated subqueries in **D1**, and the optimal subqueries in **D2** to the text-metadata-based retrieval indexes. For the subqueries generated from the proposed method, we adjust its length to make it include top 1, 2, 3 and 4 ranked key query terms.

Table 2 illustrates the accuracies of all the runs. We make three observations from the results. First, issuing using the original queries (with many modalities) yield the lowest retrieval, than issuing with sub-queries (with segmented modalities). This confirms the effectiveness of our proposed method. Second, retrieval using the proposed subqueries is only slightly lower than that from using human-annotated subqueries. This shows that our proposed method can effectively identify subqueries that is comparable to human annotation. Third, retrieval empirically peaks at a subquery of length two and deteriorates when more query terms are used. This is likely because most video metadata is very short and contain only one or two key concepts. For the

Table 2: Retrieval performance of text (video metadata) modality. Mean Inverted Rank @ 100 from the proposed method, human-annotation D1, and the optimal subqueries D2.

| ſ | <u> </u> | O / 1   | <u> </u>  |       | D     | 1     |       |
|---|----------|---------|-----------|-------|-------|-------|-------|
| ١ | Original | Optimal | numan-    |       | Prop  | osed  |       |
|   |          | queries | annotated |       |       |       |       |
|   | queries  | D2      | D1        | top-1 | top-2 | top-3 | top-4 |
| ĺ | 27.7     | 38.5    | 36.7      | 33.8  | 35.7  | 35.4  | 35.2  |

rest of the paper, all reported text-attributed retrieval results are based on the top-2 ranked key terms.

### 4.3 Using More Modalities

In this section, we examine the retrieval benefit of using more attribute types. Here are the additional attribute types and their brief descriptions:

Auditory-based ASR modality: There are grammatical constructs common to spoken attributes, e.g. "saying that...", that can detected using the surface pattern approach in Section 3.1. Spoken attributes are appropriately issued to a retrieval engine indexed on the spoken speech transcripts. As an Automatic Speech Recognition (ASR) engine, we use the HTK version 3.4 release from Cambridge University Engineering Department.

Visual-based OCR modality: These are characterized by the query snippets "with the logo...", "the text...appears on the screen", and are suitable for a retrieval engine indexed on the detected Optical Character Recognition (OCR) text in the video frames. We use a gray scale OCR engine, with multi-frame integration and a character segmenter using region-growing. The recognizer is a neural network trained using backpropagation.

Visual-based HLF concept detectors: Finally, we also use the 130-HLF-detectors from [7]. As discussed in Section 3.3, it is also important to note the difference between these global detectors and our VA detectors in Section 3.3. For query-to-HLF mapping, we match a query against an expanded gloss description with related Wikipedia text.

Table 3 shows the results of the various combination of modalities using linear fusion. It is clear that when more search modalities are utilized, we obtain better retrieval. The modality that contributes most to retrieval is the ASR modality. The modality with the least contribution is the 130-HLF-detectors. This concurs with the findings of other participants of the KIS tasks [10, 6, 14]. We also list the relevant results of these participants for comparison. The video-metadata-based retrieval results from [10, 14] are similar to the results that we obtain using the original query without any concept detection (See Table 2).

### 5. CONCLUSIONS

We have presented a query-to-modality mapping approach to the TRECVid 2010 known-Item video search (KIS) task. Our method deviates from the conventional query-to-HLF mapping by generalizing the query mapping to the various modality types in a multimedia domain, and formulating the mapping as a classification task based on surface pattern and syntactic features. This is in recognition that a sub-query of a particular modality type is better issued to a search engine that is indexed on features of the corresponding modality type. This query modality alignment can also

Table 3: Mean inverted rank (MIR) of different runs. Linear fusion is used to combine all modality. See text for the explanation of the acronyms.

| MIR@1 | MIR@10                       | MIR@100                                                   |
|-------|------------------------------|-----------------------------------------------------------|
| 21.3  | 24.4                         | 27.7                                                      |
| 27.3  | 33.4                         | 35.7                                                      |
| 28.0  | 34.5                         | 37.3                                                      |
| 32.5  | 37.5                         | 39.2                                                      |
| 33.4  | 39.2                         | 41.0                                                      |
| 33.4  | 39.8                         | 41.3                                                      |
| _     | -                            | 29.4                                                      |
| -     | _                            | 25.8                                                      |
| _     | _                            | 25.0                                                      |
|       | 27.3<br>28.0<br>32.5<br>33.4 | 21.3 24.4   27.3 33.4   28.0 34.5   32.5 37.5   33.4 39.2 |

be seen as a generalization of the query segmentation approach traditionally used by the text community for query analysis. We empirically show on the TRECVid 2010 KIS dataset that our approach can greatly enhance retrieval over existing methods.

Several issues are worthy of further investigation. The current retrieval is based on a linear fusion of search results by indexes of different modalities. The inter-relation among search attributes of different modalities can be further analyzed for a more intelligent fusion.

#### 6. REFERENCES

- [1] TRECVID 2010 Known-Item Search. http://www-nlpir.nist.gov/projects/tv2010/tv2010.html#kis.
- [2] M. Bendersky and W. B. Croft. Discovering key concepts in verbose queries. In *Proc of SIGIR*, pages 491–498, 2008.
- [3] S. Bergsma and Q. I. Wang. Learning noun phrase query segmentation. In *Proc of EMNLP-CoNLL*, pages 819–826, 2007.
- [4] W. Daelemans, J. Zavrel, K. van der Sloot, and A. van den Bosch. Timbl: Tilburg memory based learner version 6.3, reference guide. ILK Technical Report ILK-1001, 2010.
- [5] J. Feng, Y. Zheng, and S. Yan. Towards a universal detector by mining concepts with small semantic gaps. In *Proc of ACM Multimedia*, pages 1707–1710, 2010.
- [6] X. Guo, Y. Chen, W. Liu, Y. Mao, H. Zhang, K. Zhou, L. Wang, Y. Hua, Z. Zhao, Y. Zhao, and A. Cai. Bupt-mcprl at treevid 2010. In Proc of TRECVID Workshop, 2010.
- [7] Y.-G. Jiang, J. Yang, C.-W. Ngo, and A. G. Hauptmann. Representations of keypoint-based semantic concept detection: A comprehensive study. *IEEE Transactions on Multimedia*, 12:42–53, 2010.
- [8] C. Lampert, H. Nickisch, and S. Harmeling. Learning to detect unseen object classes by between-class attribute transfer. Proc CVPR, 2009.
- [9] M. Marneffe, B. MacCartney, and C. Manning. Generating typed dependency parses from phrase structure parses. In *Proc* of *LREC*. 2006.
- [10] C.-W. Ngo, S.-A. Zhu, H.-K. Tan, W.-L. Zhao, and X.-Y. Wei. Vireo at treevid 2010: Semantic indexing, known-item search, and content-based copy detection. In *Proc of TRECVID Workshop*, Nov 2010.
- [11] P. Ogilvie and J. Callan. Combining document representations for known-item search. In *Proc of SIGIR*, SIGIR '03, pages 143–150, 2003.
- [12] K. Risvik, T. Mikolajewski, and P. Boros. Query segmentation for web search. In Proc of WWW, 2003.
- [13] H. Samuel and W. Bruce. Evaluating verbose query processing techniques. In *Proc of SIGIR*, pages 291–298, 2010.
- [14] C. Snoek, K. Sande, O. Rooij, B. Huurnink, E. Gavves, D. odijk, M. Rijke, T. Gevers, M. Worring, D. Koelma, and A. Smeulders. The mediamill treevid 2010 semantic video search engine. In *Proc of TRECVID Workshop*, 2010.
- [15] M. Soubbotin and S. Soubbotin. Use of patterns for detection of likely answer strings: A systematic approach. In *Proc of TREC*, 2020.