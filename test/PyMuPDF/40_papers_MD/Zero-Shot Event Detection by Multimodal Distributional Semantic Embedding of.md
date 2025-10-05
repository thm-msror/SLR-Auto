## **Zero-Shot Event Detection by Multimodal Distributional Semantic Embedding of** **Videos**

**Mohamed Elhoseiny** _[§‡]_ **, Jingen Liu** _[‡]_ **, Hui Cheng** _[‡]_ **, Harpreet Sawhney** _[‡]_ **, Ahmed Elgammal** _[§]_

m.elhoseiny@cs.rutgers.edu, _{_ jingen.liu,hui.cheng,harpreet.sawhney _}_ @sri.com, elgammal@cs.rutgers.edu
_§_ Rutgers University, Computer Science Department _‡_ SRI International, Vision and Learning Group



**Abstract**


We propose a new zero-shot Event Detection method by
Multi-modal Distributional Semantic embedding of videos.
Our model embeds object and action concepts as well as other
available modalities from videos into a distributional semantic space. To our knowledge, this is the first Zero-Shot event
detection model that is built on top of distributional semantics and extends it in the following directions: (a) semantic
embedding of multimodal information in videos (with focus
on the visual modalities), (b) automatically determining relevance of concepts/attributes to a free text query, which could
be useful for other applications, and (c) retrieving videos by
free text event query (e.g., ”changing a vehicle tire”) based on
their content. We embed videos into a distributional semantic space and then measure the similarity between videos and
the event query in a free text form. We validated our method
on the large TRECVID MED (Multimedia Event Detection)
challenge. Using only the event title as a query, our method
outperformed the state-of-the-art that uses big descriptions
from 12.6% to 13.5% with MAP metric and 0.73 to 0.83 with
ROC-AUC metric. It is also an order of magnitude faster.


**Introduction**

Every minute, hundreds of hours of video are uploaded to
video archival site such as YouTube (Google2014 ). Developing methods to automatically understand the events captured in this large volume of videos is necessary and meanwhile challenging. One of the important tasks in this direction is event detection in videos. The main objective of
this task is to determine the relevance of a video to an event
based on the video content (e.g., feeding an animal, birthday
party; see Fig. 1). The cues of an event in a video could include visual objects, scene, actions, detected speech (by Automated Speech Recognition(ASR)), detected text (by Optical Character Recognition (OCR)), and audio concepts (e.g.
music and water concepts).
Search and retrieval of videos for arbitrary events using
only free-style text and unseen text in particular has been a
dream in computational video and multi-media understanding. This is referred as “zero-shot event detection”, because
there is no positive exemplar videos to train a detector. Due
to the proliferation of videos, especially consumer-generated


Copyright c _⃝_ 2016, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.



(a) Grooming an Animal


(1) “brushing dog”, weight = 0.67


(2) “combing dog”, weight = 0.66


(3) “clipping nails”, weight = 0.52



(b) Birthday Party


(1) “cutting cake”, weight = 0.72


(2) “blowing candles”, weight = 0.65


(3) “opening presents”, weight = 0.59



Figure 1: Top relevant Concepts from a pre-defined multimedia concept repository and their automatically-assigned
weights as a part of our Event Detection method


videos (e.g., YouTube), zero-shot search and retrieval of
videos has become an increasingly important problem.
Several research works have been proposed to facilitate
performing the zero-shot learning task by establishing an intermediate semantic layer between events or generally categories (i.e., concepts or attributes) and the low-level representation of a multimedia content from the visual perspective. (Lampert, Nickisch, and Harmeling 2009) and (Farhadi
et al. 2009) were the two first to use attribute learning representation for the zero-shot setting for object recognition
in still images. Attributes were similarly adopted for recognizing human actions (Liu, Kuipers, and Savarese 2011); attributes are generalized and denoted by concepts in this context. Later, (Liu et al. 2013) proposed Concept Based Event
Retrieval (CBER) for videos InTheWild. Even though these
methods facilitate zero-shot event detection, they only capture the visual modality and more importantly they assume
that the relevant concepts for a query event are manually defined. This manual definition of concepts, also known as
semantic query editing, is a tedious task and may be biased
due to the limitation of human knowledge. Instead, we aim
at automatically generating relevant concepts by leveraging
information from distributional semantics.

Recently, several systems were proposed for zero-shot
event detection methods (Wu et al. 2014; Jiang et al. 2014b;
2014a; Chen et al. 2014; Habibian, Mensink, and Snoek


2014). These approaches rely on the whole text description
of an event where relevant concepts are specified; see example event descriptions used in these approaches in the Supplementary Materials (SM) [1] ( explicitly define the event explication, scene, objects, activities, and audio). In practice,
however, we think that typical use of event queries under
this setting should be similar to text-search, which is based
on few words instead that we model their connection to the

multimodal content in videos.
The main question addressed in this paper is how to use
an event text query (i.e. just the event title like “birthday
party” or “feeding an animal”) to retrieve a ranked list of
videos based on their content. In contrast to (Lampert, Nickisch, and Harmeling 2009; Liu et al. 2013), we do not manually assign relevant concepts for a given event query. Instead, we leverage information from a distributional semantic space (Mikolov et al. 2013b) trained on a large text corpus to embed event queries and videos to the same space,
where similarity between both could be directly estimated.
Furthermore, we only assume that query comes in the form
of an “unstructured” few-keyword query (in contrast to (Wu
et al. 2014; Jiang et al. 2014b; 2014a)). We abbreviate
our method as EDiSE (Event-detection by Distributional Semantic Embedding of videos).
**Contributions.** The contributions of this paper can be
listed as follows: (1) Studying how to use few-keyword
unstructured-text query to detect/retrieve videos based on
their multimedia content, which is novel in this setting. We
show how relevant concepts to that event query could be automatically retrieved through a distributional semantic space
and got assigned a weight associated with the relevance;
see Fig. 1 “Birthday” and “Grooming an Animal” example
events. (2) To the best of our knowledge, our work is the
first attempt to model the connection between few keywords
and multimodal information in videos by distributional semantics . We study and propose different similarity metrics
in the distributional semantic space to enable event retrieval
based on (a) concepts, (b) ASR, and (c) OCR in videos. Our
unified framework is capable of embedding all of them into
the same space; see Fig. 2. (3) Our method is also very fast,
which makes it applicable to both large number of videos
and concepts ( _i.e_ . 26.67 times faster than the state of the
art (Jiang et al. 2014a)).


**Related Work**

Attribute methods for zero-shot learning are based on manually specifying the attributes for each category (e.g., (Lampert, Nickisch, and Harmeling 2009; Parikh and Grauman 2011)). Other methods focused on attribute discovery (Rohrbach, Stark, and Schiele 2011; Rohrbach,
Ebert, and Schiele 2013) and then apply the same mechanism. Recently, several methods were proposed to perform zero shot recognition by representing unstructured text
in document terms ( _e.g_ . (Elhoseiny, Saleh, and Elgammal 2013; Mensink, Gavves, and Snoek 2014)) One drawback of the TFIDF (Salton and Buckley 1988) in (Elho

1
Supplementary Materials (SM) could be found here https://

sites.google.com/site/mhelhoseiny/EDiSE_supp.zip



seiny, Saleh, and Elgammal 2013) and hardly matching tag
terms in (Mensink, Gavves, and Snoek 2014; Rohrbach et al.
2010) is that they do not capture semantically related terms
that our model can relate in noisy videos instead of still images. Also, WordNet (Miller 1995), adopted in (Rohrbach
et al. 2010), does not connect objects with actions (e.g., person blowing candle), making it hard to apply in our setting
and heavily depending on predefined information like WordNet.

There has been a recent interest especially in the computational linguistics’ community in word-vector representation
( e.g., (Bengio et al. 2006)), which captures word semantics based on context. While word-vector representation is
not new, recent algorithms (e.g. (Mikolov et al. 2013b;
2013a)) enabled learning these vectors from billions of
words, which makes them much more semantically accurate. As a result, these models got recently adopted
in several tasks including translation (Mikolov, Le, and
Sutskever 2013) and web search (Shen et al. 2014). Several computer vision researchers explored using these wordvector representation to perform Zero-Shot learning in the
object recognition (e.g. (Frome et al. 2013; Socher et al.
2013; Norouzi et al. 2014)). They embed the object class
name into the word-vector semantic space learnt by models like (Mikolov et al. 2013b). It is worth mentioning
that these zero-shot learning approaches (Frome et al. 2013;
Socher et al. 2013) and also the aforementioned work (Elhoseiny, Saleh, and Elgammal 2013) assume that during training, there is a set of training classes and test classes. Hence,
they learn a transformation to correlate the information between both domains (textual and visual). In contrast, zeroshot setting of event retrieval rely mainly on the event information without seeing any training events, as assumed in
recent zero-shot event retrieval methods (e.g., (Dalton, Allan, and Mirajkar 2013; Jiang et al. 2014b; Wu et al. 2014;
Liu et al. 2013)). Hence, there does not exist seen events to
learn such transformation from. Differently, we also model
multimodal connection from free text query to video information.

In the context of videos, (Wu et al. 2014) proposed a
method for zero-shot event detection by using the salient
words in the whole structured event description, where relevant concept are already defined in the event structured text
description; also see Eq. 1 in (Wu et al. 2014). Similarly, (Dalton, Allan, and Mirajkar 2013) adopted a MarkovRandom-Field language model proposed by (Metzler and
Croft 2005). One drawback of this model is that it performs
an intensive processing for each new concept. This is since
it determines the relevance of the concept to a query event
by creating a text document to represent each concept. This
document is created by web-querying the concept name and
some of its keywords and merging the top retrieved pages.
In contrast, our model does not require this step to determine
relevance of an event to a query. Once the language model
is trained, any concept can be instantly added and captured
in our multimodal semantic embedding of videos.
In contrast to both (Wu et al. 2014) and (Dalton, Allan,
and Mirajkar 2013), we focus on retrieving videos only with
the event title (i.e., few-words query) and without semantic


editing. The key difference is in modeling and embedding
concepts to allow zero-shot event retrieval. In (Wu et al.
2014) and (Dalton, Allan, and Mirajkar 2013), the semantic space is a vector whose dimensionality is the number of
the concepts. Our idea is to embed concepts, video information, and the event query into a distributional semantic space
whose dimensionality is independent of the number of concepts. This property, together with the semantic properties
captured by distributional semantics, feature our approach
with two advantages (a) scalability to any concept size. Having new concepts does not affect the representation dimensionality (i.e., in all our experiments concepts, videos, event
queries are embedded to _M_ dimensional space; _M_ is few
hundreds in our experiments). (b) facilitating automatic determination of relevant concepts given an unstructured short
event query: For example, being able to automatically determine that “blowing a candle” concept is a relevant concept to
“birthday party” event. (Wu et al. 2014) and (Dalton, Allan,
and Mirajkar 2013) used the complete text description of an
event for retrieval that explicitly specifies relevant concepts.
There is a class of models that improve zero-shot Event
Detection performance by reranking. Jiang et al. proposed
multimodal pseudo relevance feedback (Jiang et al. 2014b)
and self-paced reranking (Jiang et al. 2014a) algorithms.
The main assumption behind these models is that all unlabeled test examples are available and the top few examples
by a given initial ranking have high top K precision (K _∼_
10). This means that reranking algorithms can not update
confidence of a video for an event without knowing the confidences of the remaining videos to perform reranking. In
contrast, our goal is different which is to directly model the
probability of a few-keyword event-query given an arbitrary
video. Hence, our work does not require an initial ranking
and can compute the conditional probability of a video without any information about other videos. Our method is also
26.67 times faster, as detailed in our experiments.


**Method**

**Problem Definition**


Given an arbitrary event query _e_ and a video _v_ (e.g. just
”birthday party”), our objective is to model _p_ ( _e|v_ ). We start
by defining the representation of event query _e_, the concept
set **c**, the video _v_ in our setting.
**Event-Query representation** _e_ **:** We use the unstructured
event title to represent an event query for concept based retrieval. Our framework also allows additional terms specifically for ASR or OCR based retrieval. While we show retrieval on different modalities, concept based retrieval is our
main focus in this work. The few-keyword event query for
concept based retrieval is denoted by _e_ _c_, while query keywords for OCR and ASR are denoted by _e_ _o_ and _e_ _a_, respectively. Hence, under our setting _e_ = _{e_ _c_ _, e_ _o_ _, e_ _a_ _}_ .
**Concept Set c:** We denote the whole concept set in our
setting as **c**, which include visual concepts **c** _v_ and audio concepts **c** _d_, i.e., **c** = _{_ **c** _v_ _,_ **c** _d_ _}_ . The visual concepts include
object, scene and action concepts. The audio concepts include acoustic related concepts like water sound. We performed an experiment on a set of audio concepts trained



on MFCC audio features (Davis and Mermelstein 1980;
Logan and others 2000). However, we found their performance _≈_ 1% MAP, and hence we excluded them from our final experiments. Accordingly, our final performance mainly
relies on the visual concepts for concept based retrieval; i.e.,
**c** _d_ = _∅_ . We denote each member _c_ _i_ _∈_ **c** as the definition
of the _i_ _[th]_ concept in **c** . _c_ _i_ is defined by the _i_ _[th]_ concept’s
name and optionally some related keywords; see examples
in SM. Hence, **c** = _{c_ 1 _, · · ·, c_ _N_ _}_ is the the set of concept
definitions, where _N_ is the number concepts.
**Video Representation:** For our zero-shot purpose, a
video _v_ is defined by three pieces of information, which are
video OCR denoted by _v_ _o_, video ASR denoted by _v_ _a_, and
video concept representation denoted by _v_ _c_ . _v_ _o_ and _v_ _a_ are
the detected text in OCR and ASR, respectively. We used
(Myers et al. 2005) to extract _v_ _o_ and (van Hout et al. 2013)
to extract _v_ _a_ . In this paper, we mainly focus on the visual
video content, which is the most challenging. The video
concept based representation _v_ _c_ is defined as


_v_ _c_ = [ _p_ ( _c_ 1 _|v_ ) _, p_ ( _c_ 2 _|v_ ) _, · · ·, p_ ( _c_ _N_ _|v_ )] (1)


where _p_ ( _c_ _i_ _|v_ ) is a conditional probability of concept _c_ _i_ given
video _v_, detailed later. We denote _p_ ( _c_ _i_ _|v_ ) by _v_ _c_ _[i]_ [.]
In zero-shot event detection setting, we aim at recognizing events in videos without training examples based on its
multimedia content including still-image concepts like objects and scenes, action concepts, OCR, and ASR [2] . Given
a video _v_ = _{v_ _c_ _, v_ _o_ _, v_ _a_ _}_, our goal is to compute _p_ ( _e|v_ ) by
embedding both the event query _e_ and information of video
_v_ of different modalities ( _v_ _c_, _v_ _o_, and _v_ _o_ ) into a distributional
semantic space, where relevance of _v_ to _e_ could be directly
computed; see Fig. 2. Specifically, our approach is to model
_p_ ( _e|v_ ) as a function _F_ of _θ_ ( _e_ ), _ψ_ ( _v_ _c_ ), _θ_ ( _v_ _o_ ), and _θ_ ( _v_ _a_ ),
which are the distributional semantic embedding of _e_, _v_ _c_,
_v_ _o_, and _v_ _a_, respectively


_p_ ( _e|v_ ) _∝F_ � _θ_ ( _e_ ) _, ψ_ ( _v_ _c_ ) _, θ_ ( _v_ _o_ ) _, θ_ ( _v_ _a_ )� (2)


We remove the stop words from _e_, _v_ _o_, _v_ _a_ before applying the
embedding _θ_ ( _·_ ). The rest of this section is organized as follows. First, we present the distributional semantic manifold
and the embedding function _θ_ ( _·_ ) which is applied to _e_, _v_ _a_,
_v_ _o_, and the concept definitions **c** in our framework. Then,
we show how to determine automatically relevant concepts
to an event title query and assign a relevance weight to them,
as illustrated in Fig. 1. We present this concept relevance
weighting in a separate section since it might be generally
useful for other applications. Finally, we present the details
of _p_ ( _e|v_ ) where we derive _v_ _c_ embedding ( _i.e_ . _ψ_ ( _v_ _c_ )), which
is based on the proposed concept relevance weighting.


**Distributional Semantic Model &** _θ_ ( _·_ ) **Embedding**
We start by the distributional semantic model by (Mikolov
et al. 2013b; 2013a) to train our semantic manifold. We denote the trained semantic manifold by _M_ _s_, and the vectorization function that maps a word to _M_ _s_ space as _vec_ ( _·_ ). We
denote the dimensionality of the real vector returned from


2 Note that OCR and ASR are not concepts. They are rather
detected text in video frames and speech


Figure 2: EDiSE Approach


_vec_ ( _·_ ) by _M_ . These models learn a vector for each word
_w_ _n_, such that _p_ ( _w_ _n_ _|_ ( _w_ _i−L_ _, w_ _i−L_ +1 _, · · ·, w_ _i_ + _L−_ 1 _, w_ _i_ + _L_ ) is
maximized over the training corpus; 2 _×L_ is the context window size. Hence similarity between _vec_ ( _w_ _i_ ) and _vec_ ( _w_ _j_ ) is
high if they co-occurred a lot in context of size 2 _× L_ in the
training text-corpus (i.e., semantically similar words share
similar context). Based on the trained _M_ _s_ space, we define how to embed the event query _e_, and **c** . Each of _e_ _c_, _e_ _a_,
and _e_ _o_ is set of one or more words. Each of these words
can be directly embedded into _M_ _s_ manifold by _vec_ ( _·_ ) function. Accordingly, we represent these sets of word vectors
for each of _e_ _c_, _e_ _a_, and _e_ _o_ as _θ_ ( _e_ _c_ ), _θ_ ( _e_ _a_ ), and _θ_ ( _e_ _o_ ). We
denote _{θ_ ( _e_ _c_ ) _, θ_ ( _e_ _a_ ) _, θ_ ( _e_ _o_ ) _}_ by _θ_ ( _e_ ). Regarding embedding
of **c**, each concept _c_ _[∗]_ _∈_ **c** is defined by its name and optionally some related keywords. Hence, the corresponding word
vectors are then used to define _θ_ ( _c_ _[∗]_ ) in _M_ _s_ space.


**Relevance of Concepts to Event Query**


Let us define a similarity function between _θ_ ( _c_ _[∗]_ ) and _θ_ ( _e_ _c_ )
as _s_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _[∗]_ )). We propose two functions to measure
the similarity between _θ_ ( _e_ _c_ ) and _θ_ ( _c_ _[∗]_ ). The first one is inspired by an example by (Mikolov et al. 2013b) to show the
quality of their language model, where they indicated that
_vec_ (“king”) + _vec_ (“woman”) _−_ _vec_ (“man”) is closest to
_vec_ (“queen”). Accordingly, we define a version of _s_ ( _X, Y_ ),
where the sets _X_ and _Y_ are firstly pooled by the sum operation; we denote the sum pooling operation on a set by an
overline. For instance, _X_ = [�] _i_ _[x]_ _[i]_ [ and] _[ Y]_ [ =][ �] _i_ _[y]_ _[j]_ [, where]

_x_ _i_ and _y_ _j_ are the word vectors of the _i_ _[th]_ element in _X_ and
the _j_ _[th]_ element in _Y_, respectively. Then, cosine similarity
between _X_ and _Y_ is computed. We denote this version as
_s_ _p_ ( _·, ·_ ); see Eq. 3. Fig. 3 shows how _s_ _p_ ( _·, ·_ ) could be used
to retrieve the top 20 concepts relevant to _θ_ (“Grooming An
Animal”) in _M_ _s_ space. The figure also shows embedding
of the query and the relevant concept sets in 3D PCA visualization. _θ_ ( _e_ _c_ =“Grooming An Animal”) and each of _θ_ ( _c_ _i_ )
for the most relevant 20 concepts are represented by their



Figure 3: PCA visualization in 3D of the ”Grooming an
Animal” event (in green) and its most 20 relevant concepts
in _M_ _s_ space using _s_ _p_ ( _·, ·_ ). The exact _s_ _p_ ( _θ_ (“Grooming An
Animal”) _, θ_ ( _c_ _i_ )) is shown between parenthesis


corresponding pooled vectors ( _θ_ ( _e_ _c_ ) and _θ_ ( _c_ _i_ )) _∀i_ ), normalized to unit length under L2 norm. Another idea is to define _s_ ( _X, Y_ ) as a similarity function between the _X_ and _Y_
sets. For robustness (Torki and Elgammal 2010), we used
percentile-based Hausdorff point set metric, where similarity between each pair of points is computed by the cosine
similarity. We denote this version by _s_ _t_ ( _X, Y_ ); see Eq. 3.
We used _l_ = 50% (i.e., median).



_i_ _[x]_ _[i]_ [)] [T] [(][�]

_i_ _[x]_ _[i]_ _[∥∥]_ ~~[�]~~



( [�]
_s_ _p_ ( _X, Y_ ) =



_j_ _[y]_ _[j]_ [)]



_∥_ ~~[�]~~



~~T~~

_j_ _[y]_ _[j]_ [)] _X_ _Y_ (3)

_j_ _[y]_ _[j]_ _[∥]_ [=] _∥X∥∥Y ∥_



_l_ % _x_ [T] _i_ _[y]_ _[j]_ _l_ % _x_ [T] _i_ _[y]_ _[j]_
_s_ _t_ ( _X, Y_ ) = _min{min_ _j_ _m_ _i_ _[ax]_ _∥x_ _i_ _∥∥∥y_ _j_ _∥_ _[,]_ _min_ _i_ _m_ _j_ _[ax]_ _∥x_ _i_ _∥∥y_ _j_ _∥_ _[}]_


**Event Detection** _p_ ( _e|v_ )
In practice, we decomposed _p_ ( _e|v_ ) into _p_ ( _e_ _c_ _|v_ ), _p_ ( _e_ _o_ _|v_ ),
_p_ ( _e_ _a_ _|v_ ), which makes the problem reduces to deriving
_p_ ( _e_ _c_ _|v_ ) (concept based retrieval), _p_ ( _e_ _o_ _|v_ ) (OCR based retrieval), and _p_ ( _e_ _a_ _|v_ ) (ASR based retrieval) under _M_ _s_ . We
start by _p_ ( _e_ _c_ _|v_ ) then we will how later in this section how
_p_ ( _e_ _o_ _|v_ ), and _p_ ( _e_ _a_ _|v_ ) could be estimated.
**Estimating** _p_ ( _e_ _c_ _|v_ ) **:** In our work, concepts are linguistic
meanings that have corresponding detection functions given
the video _v_ . From Fig. 3, _M_ _s_ space could be viewed as a
space of meanings captured by a training text-corpus, where
only sparse points in that space has a corresponding visual
detection functions given _v_, which are the concepts **c** (e.g.,
“blowing a candle”). For zero shot event detection, we aim
at exploiting these sparse points by the information captured
by _s_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _[i]_ _∈_ **c** )) in _M_ _s_ space. We derive _p_ ( _e_ _c_ _|v_ ) from
probabilistic perspective starting from marginalizing _p_ ( _e_ _c_ _|v_ )
over the concept set **c**



_p_ ( _e_ _c_ _|v_ ) _∝_ �



� _p_ ( _e_ _c_ _|c_ _i_ ) _p_ ( _c_ _i_ _|v_ ) _∝_ �


_c_ _i_ _c_ _i_



_s_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _i_ )) _v_ _c_ _[i]_
(4)

_c_ _i_



where _p_ ( _e|c_ _i_ ) _∀i_ are assumed to be proportional to
_s_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _i_ )) in our framework. From semantic embedding perspective, each video _v_ is embedded into _M_ _s_ by the


set _ψ_ ( _v_ _c_ ) = _{θ_ _v_ ( _c_ _i_ ) = _v_ _c_ _[i]_ _[θ]_ [(] _[c]_ _[i]_ [)] _[,][ ∀][c]_ _[i]_ _[∈]_ **[c]** _[}]_ [, where] _[ v]_ _c_ _[i]_ _[θ]_ [(] _[c]_ _[i]_ [)][ is]
a set of the same points in _θ_ ( _c_ _i_ ) scaled with _v_ _c_ _[i]_ [;] _[ ψ]_ [(] _[v]_ _[c]_ [)][ could]
be then directly compared with _θ_ ( _e_ _c_ ); see Eq. 5



_p_ ( _e_ _c_ _|v_ ) _∝_ �



_s_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _i_ )) _v_ _c_ _[i]_

_c_ _i_



_c_ _i_ (5)

_∝_ _s_ _[′]_ ( _θ_ ( _e_ _c_ ) _, ψ_ ( _v_ _c_ ) = _{θ_ _v_ ( _c_ _i_ ) _, ∀c_ _i_ _∈_ **c** _}_ )



where _s_ _[′]_ ( _θ_ ( _e_ _c_ ) _, ψ_ ( _p_ ( **c** _|v_ )) = [�] _i_ _[s]_ [(] _[θ]_ [(] _[e]_ _[c]_ [)] _[, θ]_ _[v]_ [(] _[c]_ _[i]_ [))][ and] _[ s]_ [(] _[·][,][ ·]_ [)]

could be replaced by _s_ _p_ ( _·, ·_ ), _s_ _t_ ( _·, ·_ ), or any other measure in
_M_ _s_ space. An interesting observation is that when _s_ _p_ ( _·, ·_ )



_T_
is chosen, _p_ ( _e_ _c_ _|v_ ) _∝_ _[θ]_ [(] _[e]_ _[c]_ [)]



_i_



_∥_ [(] _θe_ _[e]_ _[c]_ _c_ [)] _∥_ � � _i_



**Scene Concepts** _p_ ( _s_ _i_ _|f_ ) _, s_ _i_ _∈_ **c** _s_ **:** We represented scene
concepts ( _p_ ( _s_ _i_ _|f_ )) as bag of word representation on static
features (i.e., SIFT (Lowe 2004) and HOG (Dalal and Triggs
2005)) with 10000 codebooks. We used TRECVID 500 SIN
concepts concepts, including scene categories like “city”
and “hall” way; these concepts are provided by provided by
TRECVID2011 SIN track.
**Action Concepts** _p_ ( _a_ _i_ _|u_ ) _, a_ _i_ _∈_ **c** _a_ **:** We use both manually
annotated (i.e. strongly supervised) and automatically annotated (i.e. weekly supervised) concepts; detailed in SM. We
have _∼_ 500 action concepts; please refer to (Liu et al. 2013)
for the action concept detection method that we adopt.


**Video level concept probabilities** _p_ ( **c** _|v_ ) We represent
probabilities of the **c** _v_ set given a video _v_ by a pooling
operation over the the chunks or the frames of the videos
similar to (Liu et al. 2013). In our experiments, we evaluated both max and average pooling. Specifically, _p_ ( _o_ _i_ _|v_ ) =
_ρ_ ( _{p_ ( _o_ _i_ _|f_ _k_ ) _, f_ _k_ _∈_ _v}_ ), _p_ ( _s_ _l_ _|v_ ) = _ρ_ ( _{p_ ( _s_ _i_ _|f_ _k_ ) _, f_ _k_ _∈_ _v}_ ),
_p_ ( _a_ _k_ _|v_ ) = _ρ_ ( _{p_ ( _a_ _i_ _|u_ _k_ ) _, u_ _k_ _∈_ _v}_, where _p_ ( _o_ _i_ _|v_ ) and _p_ ( _s_ _l_ _|v_ )
are the video level probabilities of for the _i_ _[th]_ object and the
_l_ _[th]_ scene concepts respectively, pooled over frames _f_ _k_ _∈_ _v_ .
_{f_ _k_ _∈_ _v}_ are selected every M frames in _v_ (M= 250).
_p_ ( _a_ _k_ _|v_ ) is the video level probability of the _k_ _[th]_ action concept, pooled over a set of video chunks _{u_ _k_ _∈_ _v}_ . The chunk
size is set to the mean chunk length of all concept training
chunks. Finally, _ρ_ is the pooling function. We denote average and max pooling as _ρ_ _a_ ( _·_ ) and _ρ_ _m_ ( _·_ ), respectively.


**EDiSE Computational Performance Benefits**

Here we discuss the computational complexity of concept
based EDISE, and ASR/OCR based EDiSE. The fusion part
is negligible since it is constant time.


**Concept based EDiSE**
The computational complexity for computing _p_ ( _e_ _c_ _|v_ ) is
mainly linear in the number of videos, denoted by _|V |_ . We
here detail why computational complexity of _p_ ( _e_ _c_ _|v_ ) is almost constant and hence video retrieval is almost _O_ ( _|V |_ ).
From Eq. 5, _p_ ( _e_ _c_ _|v_ ) has a computational complexity of
_O_ ( _N ·_ _Q_ ) for on e video, where _Q_ is the computational complexity of computing _s_ ( _·, ·_ ) and _N_ is the number of concepts.
We detail next the computational complexity of _s_ _p_ ( _·, ·_ ) and
_s_ _t_ ( _·, ·_ ) for the whole set of videos _|V |_ .


**Complexity of** _p_ ( _e_ _c_ _|v_ ) **for** _s_ _p_ ( _·, ·_ ) Let’s assume that
there _θ_ ( _e_ _c_ ) set has _|e_ _c_ _|_ terms and _θ_ ( _c_ _i_ ) has _|c_ _i_ _|_ terms.
Then, the computational complexity of _s_ _p_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _i_ )) is


Figure 4: Concept probabilities from videos ( _p_ ( **c** _|v_ ))



_θ_ ( _c_ _i_ )
_∥θc_ _i_ _∥_ _[v]_ _c_ _[i]_ � which is a direct



similarity between _θ_ ( _e_ _c_ ) representing the query and the em
bedding of _ψ_ ( _v_ _c_ ) as [�] _i_ _∥θθc_ ( _c_ _ii_ ) _∥_ _[v]_ _c_ _[i]_ [; see proof in Appendix A.]

_s_ _p_ ( _·, ·_ ) performs consistently better than _s_ _t_ ( _·, ·_ ) in our experiments. In practice, we only include _θ_ _v_ ( _c_ _i_ ) in _ψ_ ( _v_ _c_ ) such
that _c_ _i_ is among the top R concepts with highest _p_ ( _e_ _c_ _|c_ _i_ ).
This is assuming that the remaining concepts are assigned
_p_ ( _e_ _c_ _|c_ _i_ ) = 0 which makes those items vanish; we used R=5.
Hence, only a few concept detectors needs to be computed
for on _v_ which is a computational advantage.
**Estimating** _p_ ( _e_ _o_ _|v_ ) and _p_ ( _e_ _a_ _|v_ ) **:** Both _v_ _o_ and _v_ _a_ can be
directly embedded into _M_ _s_ since they are sets of words.
Hence, we can model _p_ ( _e_ _o_ _|v_ ) and _p_ ( _e_ _a_ _|v_ ) as follows

_p_ ( _e_ _o_ _|v_ ) _∝_ _s_ _d_ ( _θ_ ( _e_ _o_ ) _, θ_ ( _v_ _o_ )) _, p_ ( _e_ _a_ _|v_ ) _∝_ _s_ _d_ ( _θ_ ( _e_ _a_ ) _, θ_ ( _v_ _a_ )) (6)

where _s_ _d_ ( _X, Y_ ) = [�] _ij_ _[x]_ _i_ _[T]_ _[y]_ _[j]_ [.] We found this similarity

function more appropriate for ASR/OCR text since they normally contains more text compared to concept definition.
We also exploited an interesting property in _M_ _s_ that nearest words to an arbitrary point can be retrieved. Hence, we
automatically augment _e_ _a_ and _e_ _o_ with the nearest words to
the event title in _M_ _s_ using cosine similarity before retrieval.
We found this trick effective in practice since it automatically retrieve relevant words that might appear in _v_ _o_ or _v_ _a_ .
**Fusion:** We fuse _p_ ( _e_ _c_ _|v_ ), _p_ ( _e_ _o_ _|v_ ), and _p_ ( _e_ _a_ _|v_ ) by
weighted geometric mean with focus on visual concepts, i.e.
_w_ +1 ~~[�]~~
_p_ ( _e|v_ ) = _p_ ( _e_ _c_ _|v_ ) _[w]_ ~~[�]~~ _p_ ( _e_ _o_ _|v_ ) _p_ ( _e_ _a_ _|v_ )); _w_ = 6 . _p_ ( _e_ _c_ _|v_ ),



where _s_ _d_ ( _X, Y_ ) = [�]



_p_ ( _e_ _c_ _|v_ ) _[w]_ ~~[�]~~



_w_
_p_ ( _e|v_ ) = _p_ ( _e_ _c_ _|v_ ) _[w]_ ~~[�]~~ _p_ ( _e_ _o_ _|v_ ) _p_ ( _e_ _a_ _|v_ )); _w_ = 6 . _p_ ( _e_ _c_ _|v_ ),

_p_ ( _e_ _c_ _|v_ ), and _p_ ( _e_ _c_ _|v_ ) involves the similarity between _θ_ ( _e_ )
and each of _ψ_ ( _v_ _c_ ), _θ_ ( _v_ _o_ ), and _θ_ ( _v_ _a_ ), leading to Eq. 2 view.



**Visual Concept Detection functions (** _p_ ( **c** _|v_ ) **)**
We leverage the information from three types of visual concepts in **c** _v_ : object concepts **c** _o_, action concepts **c** _a_, and
scene concepts **c** _s_ . Hence, **c** = **c** _v_ = _{_ **c** _o_ _∪_ **c** _a_ _∪_ **c** _s_ _}_ ; the
list of concepts are attached in SM. We define object and
scene concept probabilities per video frame, and action concepts per video chunks. The rest of this section summarizes
the concept detection for objects and scenes per frame _f_, and
action concepts per video chunk _u_ . Then, we show how they
can be reduced to video level probabilities. Fig. 4 shows example high confidence concepts in a “Birthday Party” video.
**Object Concepts** _p_ ( _o_ _i_ _|f_ ) _, o_ _i_ _∈_ **c** _o_ **:** We involve 1000 Overfeat (Sermanet et al. 2014) object concept detectors which
maps to 1000-ImageNet categories. We also adopt the concept detectors of face, car and person from a publicity available detector (i.e., (Felzenszwalb, McAllester, and Ramanan 2008))


_O_ ( _M_ ( _|e_ _c_ _|_ + _|c_ _i_ _|_ ). _|c_ _i_ _|_ and and _|e_ _c_ _|_ are usually few terms
in our case ( _<_ 10). Hence the computational complexity of
_s_ _p_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _i_ )) is _O_ ( _M_ ), where _M_ is the dimensionality of
the word vectors. In our experiments _M_ = 300. Given the
complexity of _s_ _p_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _i_ )), the computational complexity of _p_ ( _e_ _c_ _|v_ ) will be _O_ ( _N · M_ ), where _N_ is the number of
concepts. Hence, the computational complexity for computing _p_ ( _e_ _c_ _|v_ ) for _|V |_ videos is _O_ ( _|V | · N · M_ ). However, for a
given event, only few concepts are relevant, which are computed based on _s_ _p_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _i_ )) and only few concepts 5 in
our case are sufficient for event zero shot retrieval, retrieved
by Nearest Neighbor search of _c_ _i_ _∈_ **c** that is close the _e_ _c_ .
Hence the computational complexity reduced to _O_ ( _|V |·M_ ),
_M_ = 300 for the GoogleNews word2vec model that we
used. Hence, the complexity for _|V |_ videos is basically linear _O_ ( _|V |_ ), given _M_ is a constant and _M << |V |_ .


**Complexity of** _p_ ( _e_ _c_ _|v_ ) **for** _s_ _t_ ( _·, ·_ ) The previous argument
applies here in all elements except the complexity of the similarity function _s_ _t_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _i_ )), which is _O_ ( _M_ ( _|e_ _c_ _| · |c_ _i_ _|_ ).
Assuming that _|e_ _c_ _| · |c_ _i_ _|_ is bounded by a constant, then the
complexity of _|V |_ videos is also _O_ ( _|V | · M_ ), but with a bigger constant compared to _s_ _p_ ( _·, ·_ ) (linear in _|V |_ for constant
_M << |V |_ ).


**ASR/OCR based EDiSE**

The computational complexity of _s_ _d_ ( _θ_ ( _e_ _o_ ) _, θ_ ( _v_ _o_ )) and
_s_ _d_ ( _θ_ ( _e_ _a_ ) _, θ_ ( _v_ _a_ )) are _O_ ( _|e_ _o_ _|·|v_ _o_ _|·_ _M_ ) and _O_ ( _|e_ _a_ _|·|v_ _a_ _|·_ _M_ ),
respectively. There is no concepts for ASR/OCR based retrieval. Hence, the computational complexity of _p_ ( _e_ _o_ _, v_ ) and
_p_ ( _e_ _a_ _|v_ ) are _O_ ( _|V |·|e_ _o_ _|·|v_ _o_ _|·_ _M_ ) and _O_ ( _|V |·|e_ _a_ _|·|v_ _a_ _|·_ _M_ ),
respectively. Since _|e_ _o_ _| ≪|V |_, _|v_ _o_ _| ≪|V |_, _|e_ _a_ _| ≪|V |_,
_|v_ _a_ _| ≪|V |_, and _M ≪|V |_, the dominating factor in the
complexity for both _p_ ( _e_ _o_ _, v_ ) and _p_ ( _e_ _a_ _|v_ ) will be _|V |_ .


**Experiments**

We evaluated our method on the large TRECVID MED
(Felzenszwalb, McAllester, and Ramanan 2013). We show
the MAP (Mean Average Precision) and ROC AUC performance of the designated MEDTest set (Felzenszwalb,
McAllester, and Ramanan 2013), containing more than
25,000 videos. Unless otherwise mentioned, our results are
on TRECVID MED2013. There are two distributional semantic models in our experiments, trained on Wikipedia and
GoogleNews using (Mikolov et al. 2013b). The Wikipedia
model got trained on 1 billion words resulting in a vocabulary of size of _≈_ 120,000 words and word vectors of 250
dimensions. The GoogleNews model got trained on 100 billion words resulting in a vocabulary of size 3 million words



and word vectors of 300 dimensions. The objective of having two models is to compare how well our EDiSE method
performs depending on the size of the training corpus, used
to train the language model. In the rest of this section, we
present Concepts, OCR, ASR, and fusion results.


**Concept based Retrieval**
All the results in this section were generated by automatically retrieved concepts using only the event title. We start
by comparing different settings of our method against (Dalton, Allan, and Mirajkar 2013). We used the language model
in (Dalton, Allan, and Mirajkar 2013) for concept based
retrieval to rank the concepts. This indicates that _p_ ( _e|c_ _i_ )
in Eq. 4 is computed by the language model in (Metzler
and Croft 2005) as adopted in (Dalton, Allan, and Mirajkar 2013), that we compare with under exactly the same
setting. For our model, we evaluated the two pooling operations _ρ_ _m_ ( _·_ ) and _ρ_ _p_ ( _·_ ) and also the two different similarity
measures on _M_ _s_ space _s_ _p_ ( _·, ·_ ) and _s_ _t_ ( _·, ·_ ). Furthermore, we
evaluated the methods on both Wikipedia and GNews language models. In order to have conclusive experiments on
these eight settings of our model compared to (Dalton, Allan, and Mirajkar 2013), we performed all of them on the
four different sets of concepts (i.e. each has the same concept detectors; completely consistent comparison); see Table 1. Details about these concept sets are attached in SM.
There are a number of observations. (1) using GNews
(the bigger text corpus) language model is consistently better than using the Wikipedia language model. This indicates
when the word embedding model is trained with a bigger
text corpus, it captures more semantics and hence more accurate in our setting. (2) max pooling _ρ_ _m_ ( _·_ ) behaves consistently better than average pooling _ρ_ _a_ ( _·_ ). (3) _s_ _p_ ( _·, ·_ ) similarity measure is consistently better than _s_ _t_ ( _·, ·_ ), which we
see very interesting since this indicates that our hypothesis
of using the vector operations on _M_ _s_ manifold better represent _p_ ( _e|c_ _i_ ). Hence, we recommend finally to use the model
trained on the larger corpus, _ρ_ _m_ ( _·_ ) for concept pooling, and
use _s_ _p_ ( _·, ·_ ) to measure the performance on _M_ _s_ manifold. (4)
our model’s final setting is consistently better than (Dalton,
Allan, and Mirajkar 2013). The final MED13 ROC AUC
performance is 0.834. MAP for MED13 Events 31 to 40
(E31:40) is 5.97%. Detailed figures are attached in SM.
Our next experiment shows the final MAP performance
using the recommended setting for our framework on the
whole set of concepts, detailed earlier and in SM. Table 2
shows our final performance compared with (Dalton, Allan,
and Mirajkar 2013) on the same concept detectors. It is not
hard to see that our method performs more than double the



Table 1: MED2013 MAP performance on four concept sets (event title query)


|TRECVID MED 2013|Ours-Gnews|Col3|Ours-Wiki|Col5|(Dalton etal, 2013)|
|---|---|---|---|---|---|
|TRECVID MED 2013|_ρm_(_·_)|_ρa_(_·_)|_ρm_(_·_)|_ρa_(_·_)|_ρa_(_·_)|
|TRECVID MED 2013|_sp_(_·, ·_)<br>_st_(_·, ·_)|_sp_(_·, ·_)<br>_st_(_·, ·_)|_sp_(_·, ·_)<br>_st_(_·, ·_)|_sp_(_·, ·_)<br>_st_(_·, ·_)|_sp_(_·, ·_)<br>_st_(_·, ·_)|
|**Concepts G1 (152 concepts)**<br>**Concepts G2 (101 concepts)**<br>**Concepts G3 (60 concepts)**<br>**Concepts G4 (56 concepts)**|**4.29**<br>3.94%<br>**1.74**<br>1.20<br>**1.72**<br>1.33%<br>**1.22**<br>0.95|2.39%<br>2.38%<br>1.56%<br>1.20%<br>1.28%<br>1.16%<br>0.84%<br>0.69%|3.14%<br>2.13%<br>1.09%<br>0.96%<br>1.21%<br>0.88%<br>0.87%<br>0.76%|1.85%<br>1.70%<br>0.66%<br>0.60%<br>0.88%<br>0.74%<br>0.67%<br>0.56%|2.57%<br>1.17%<br>1.54%<br>0.83%|


Table 2: MED2013 full concept set MAP Performance
(auto-weighted versus manually-weighted concepts)

|Ours (auto)|Col2|Dalton etal,13(auto)|Col4|Dalton etal,13(manual)|Col6|Overfeat|Col8|
|---|---|---|---|---|---|---|---|
|**8.36%**|**8.36%**|3.40%|3.40%|7.4%|7.4%|2.43%|2.43%|
|SUN|Object Rank|Object Rank|Classeme|_CD_~~_DT_~~|_WSCD−SIF T_<br>_Y ouT ube_|_WSCD−SIF T_<br>_Y ouT ube_||
|0.48%|0.77%|0.77%|0.84%|2.28%|3.48%|3.48%|3.48%|



MAP performance of (Dalton, Allan, and Mirajkar 2013)
under the same concept set. Even when manual semantic
editing is applied to (Dalton, Allan, and Mirajkar 2013), our
performance is still better without semantic editing. We also
show the performance on the same events of different concepts (i.e. SUN (Patterson and Hays 2012), Object Rank (Li
et al. 2010), Classeme (Torresani, Szummer, and Fitzgibbon 2010)), and the best performing concepts in (Wu et al.
2014) (i.e., _CD_ _[DT]_, _WSC_ _Y ouT ube_ _[D][−][SIF T]_ [). These numbers are as]
reported in (Wu et al. 2014). The results indicate the value of
our concepts and approach compared to (Wu et al. 2014) and
their concepts. We also report our performance using Overfeat concepts only to retrieve videos for the same events.
This shows the value of involving action and scene concepts
compared to only still image concepts like Overfeat for zeroshot event detection. We highlight that the results in (Wu et
al. 2014) uses the whole event description which explicitly
includes names of relevant concepts.


**ASR and OCR based Retrieval**


First, we compared our OCR and ASR retrieval trained on
both Wikipedia and GoogleNews language model. Table 3



shows that the GoogleNews model MED13 MAP is better
than the Wikipedia Model MAP in both ASR and OCR,
which is consistent with our concept retrieval results. Fig. 5
shows the GoogleNews MED13 AP per event for both OCR
and ASR. We further show our AP performance on MED14
events 31 to 40 in Fig. 5.
In order to show the value our semantic modeling, we
computed the performance of string matching method as a
baseline, which basically increment the score for every exact match in the the detected text to words in the query.
While, both our model and the matching model use the
same query words and ASR/OCR detection, semantic properties captured by _M_ _s_ boosts the performance compared to
string matching; see table 3. This is since semantically relevant terms to the query have a high cosine similarity in _M_ _s_
(i.e., high _vec_ ( _w_ _i_ ) [T] _vec_ ( _w_ _j_ ) if _w_ _i_ is semantically related to


Table 3: ASR & OCR Retrieval MAP on _M_ _s_ using GNews,
Wikipedia, and using word matching

|Col1|GNews MED2013|Wiki MED2013|matching MED2013|
|---|---|---|---|
|OCR|**4.81%**|3.85%|1.8%|
|ASR|**4.23%**|1.50%|3.77%|



Table 4: ASR & OCR MAP performance using GNews corpus compared to (Wu et al. 2014)(prefix E indicates Event)


|Col1|MED13|MED13 (Wu et al. 2014)|MED14(E31:40)|
|---|---|---|---|
|OCR|**4.81%**|4.30%|2.5%|
||MED13|MED13 (Wu et al. 2014)|MED14 (E31:40)|
|ASR|**4.23%**|3.66%|5.97%|



(a) MED2013 (b) MED2014 (E31:40)


Figure 5: ASR & OCR AP Performance (Google News)


Figure 6: ASR & OCR AUCs on MED2013: Ours (GoogleNews) vs keyword Matching (the same query)


_w_ _j_ ). On the other hand, hard matching basically assumes
that _vec_ ( _w_ _i_ ) [T] _vec_ ( _w_ _j_ ) = 1 if _w_ _i_ = _w_ _j_, 0 otherwise. We
also computed the ROC AUC metric for our method and the
hard matching method on ASR and OCR; see Fig. 6. For
ASR, average AUC is 0.623 for ours and 0.567 for Matching (9.9% gain). For OCR, average AUC is 0.621 for ours
and 0.53 for Matching (17.1% gain). We report our GNews
model results compared with (Wu et al. 2014) to indicate
that, we achieve state-of-the-art MED13 MAP performance
or even better for ASR/OCR; see table 4. The table also
shows our ASR&OCR MED14 (E31:40) MAP.


**Fusion Experiments and Related Systems**
In table 5, we start by presenting a summary of our earlier
ASR/OCR results on MED13 Test. Comparing OCR and
ASR performances to Concepts performance, it is not hard
to see that OCR/ASR have much lower average AUC zeroshot performance compared to concepts which are visual in
our work. This indicates that OCR/ASR produces much
higher false negatives compared to visual concepts. When
we fused our all OCR and ASR confidences, we achieved
10.7% MAP performance, however, the average AUC performance is as low as 0.67. We achieved lower MAP for
our concepts 8.36% MAP but the average AUC performance
is as high as 0.834. This indicates that measuring retrieval
performance on MAP performance only is not informative,
so one approach might achieve a high MAP but lower average AUC and vice versa. We further achieved the best
performance of our system by fusing all Concepts, OCR,
and ASR to achieve 13.1% MAP and 0.830 average AUC.
We found our system achieves better than the state of the
art system (Wu et al. 2014) 4.0% gain in MAP, but significantly in average AUC; see 13.6% gain to (Wu et al. 2014)
in table 5.
We also discuss CPRF (Yang and Hanjalic 2010), MMPRF (Jiang et al. 2014b), and SPaR (Jiang et al. 2014a)
reranking systems in contrast to our system that does not
involve reranking. The initial retrieval performance is 3.9%
MAP without reranking. Interestingly, we achieved a performance of 13.1% MAP also without reranking. The reranking methods assumes high top 5-10 precision of the initial
ranking and that all test videos are available. Without any of
these assumptions, our system without reranking performs
6.7%, 3.0%, and 0.2% better than CPRF (Yang and Hanjalic
2010), MMPRF (Jiang et al. 2014b), and SPaR (Jiang et al.
2014a) re-ranking systems; see table 5. Unfortunately, ROC
AUC performances are not available for these method to
compare with. Regarding efficiency, given _v_ _c_ representation
of videos, our concept retrieval experiment on our whole
concept set it takes _≈_ 270 seconds on a 16 cores Intel Xeon
processor (64GB RAM) to the retrieval task on 20 events altogether. This is more than the time that SPaR (Jiang et al.
2014a) takes to rerank one event on an Intel Xeon processor(16GB RAM); see (Jiang et al. 2014a). Since, we detect
the MED13 events in _≈_ 270 given _v_ _c_ representation of videos
and as reported in (Jiang et al. 2014a), their average detection time per event for MED13 is _≈_ 5 minutes assuming feature representation of videos (i.e., 360 seconds per event =
7200 seconds per 20 events). This indicates that our system



Table 5: Fusion Experiments and Comparison to State of the
Art Systems

|Method|MAP|AUC|
|---|---|---|
|Our Concept retrieval (event title query)<br>Concept retrieval (Dalton etal, 2013) (event title query)<br>Concept retrieval (Dalton etal, 2013) (manual concepts)|8.36%<br>3.4 %<br>7.4%|0.834<br>-<br>-|
|Our ASR GNews<br>Our OCR GNews<br>Our ASR Matching<br>Our OCR Matching|4.81%<br>4.23%<br>2.77%<br>1.8%|0.623<br>0.621<br>0.567<br>0.536|
|Our ASR and OCR all fused|10.6|0.670|
|**Our Full (Concepts+ASR+OCR) (No reranking)**<br>**Our Full + SPaR reranking (Jiang et al. 2014a)**|**13.1%**<br>**13.5%**|**0.830**<br>**0.790**|
|Full system (Wu et al. 2014)|12.6|0.730|
|**Reranking Systems**|**Reranking Systems**|**Reranking Systems**|
|Without Reranking (Jiang et al. 2014b)<br>CPRF (Yang and Hanjalic 2010)<br>Full Reranking system MMPRF(Jiang et al. 2014b)<br>Full Reranking system SPaR(Jiang et al. 2014a)|3.9%<br>6.4%<br>10.1%<br>12.9%|-<br>-<br>-<br>-|



is 26.67X faster than (Jiang et al. 2014a) in MED13 detection. Finally, when we applied SPaR on our output as an
initial ranking, we found that it improves MAP (from 13.1%
to 13.5%) but hurts ROC AUC (from 0.83 to 0.79). This
indicates that reranking has a limited/harmful effect on the
performance of our method. We think is since our method
already achieve a high performance without re-ranking; see
SM for details about the features in this experiment.


**Conclusion**


We proposed a method for zero shot event detection by
distributional semantic embedding of video modalities and
with only event title query. By fusing all modalities, our
method outperformed the state of the art on the challenging
TRECVID MED benchmark. Based on this notion, we also
showed how to automatically determine relevance of concepts to an event based on the distributional semantic space.


**Acknowledgements.** This work has been supported by the
Intelligence Advanced Research Projects Activity (IARPA)
via Department of Interior National Business Center contract number D11-PC20066. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation
thereon. The views and conclusions contained herein are
those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/NBC, or the U.S.
Government. This work is also partially funded by NSFUSA award # 1409683.


**References**


Bengio, Y.; Schwenk, H.; Sen´ecal, J.-S.; Morin, F.; and Gauvain,
J.-L. 2006. Neural probabilistic language models. In _Innovations_
_in Machine Learning_ . 137–186.

Chen, J.; Cui, Y.; Ye, G.; Liu, D.; and Chang, S.-F. 2014. Eventdriven semantic concept discovery by exploiting weakly tagged internet images. In _ICMR_ .

Dalal, N., and Triggs, B. 2005. Histograms of oriented gradients
for human detection. In _CVPR_ .


Dalton, J.; Allan, J.; and Mirajkar, P. 2013. Zero-shot video retrieval using content and concepts. In _CIKM_ .

Davis, S., and Mermelstein, P. 1980. Comparison of parametric
representations for monosyllabic word recognition in continuously
spoken sentences. _Acoustics, Speech and Signal Processing, IEEE_
_Transactions on_ 28(4):357–366.

Elhoseiny, M.; Saleh, B.; and Elgammal, A. 2013. Write a classifier: Zero-shot learning using purely textual descriptions. In _ICCV_ .

Farhadi, A.; Endres, I.; Hoiem, D.; and Forsyth, D. 2009. Describing objects by their attributes. In _CVPR_ .

Felzenszwalb, P.; McAllester, D.; and Ramanan, D. 2008. A discriminatively trained, multiscale, deformable part model. In _CVPR_ .

Felzenszwalb, P.; McAllester, D.; and Ramanan, D. 2013. Trecvid
2013 an overview of the goals, tasks, data, evaluation mechanisms
and metrics. In _TRECVID 2013_, 1–8. NIST.

Frome, A.; Corrado, G. S.; Shlens, J.; Bengio, S.; Dean, J.;
Mikolov, T.; et al. 2013. Devise: A deep visual-semantic embedding model. In _NIPS_ .

Google2014. Youtube, howpublished = ”https:
//www.youtube.com/yt/press/statistics.html”,
year = 2014, note = ”[online; 11/06/2014]”.

Habibian, A.; Mensink, T.; and Snoek, C. G. 2014. Composite
concept discovery for zero-shot video event detection. In _ICMR_ .

Jiang, L.; Meng, D.; Mitamura, T.; and Hauptmann, A. G. 2014a.
Easy samples first: Self-paced reranking for zero-example multimedia search. In _ACM Multimedia_ .

Jiang, L.; Mitamura, T.; Yu, S.-I.; and Hauptmann, A. G. 2014b.
Zero-example event search using multimodal pseudo relevance
feedback. In _ICMR_ .

Lampert, C. H.; Nickisch, H.; and Harmeling, S. 2009. Learning
to detect unseen object classes by between-class attribute transfer.
In _CVPR_ .

Li, L.-J.; Su, H.; Fei-Fei, L.; and Xing, E. P. 2010. Object bank: A
high-level image representation for scene classification & semantic
feature sparsification. In _NIPS_ .

Liu, J.; Yu, Q.; Javed, O.; Ali, S.; Tamrakar, A.; Divakaran, A.;
Cheng, H.; and Sawhney, H. 2013. Video event recognition using
concept attributes. In _WACV_ .

Liu, J.; Kuipers, B.; and Savarese, S. 2011. Recognizing human
actions by attributes. In _CVPR_ .

Logan, B., et al. 2000. Mel frequency cepstral coefficients for
music modeling. In _ISMIR_ .

Lowe, D. G. 2004. Distinctive image features from scale-invariant
keypoints. _IJCV_ .

Mensink, T.; Gavves, E.; and Snoek, C. G. 2014. Costa: Cooccurrence statistics for zero-shot classification. In _CVPR_ .

Metzler, D., and Croft, W. B. 2005. A markov random field model
for term dependencies. In _ACM SIGIR_ .

Mikolov, T.; Chen, K.; Corrado, G.; and Dean, J. 2013a. Efficient
estimation of word representations in vector space. _ICLR_ .

Mikolov, T.; Sutskever, I.; Chen, K.; Corrado, G. S.; and Dean, J.
2013b. Distributed representations of words and phrases and their
compositionality. In _NIPS_ .

Mikolov, T.; Le, Q. V.; and Sutskever, I. 2013. Exploiting similarities among languages for machine translation. _arXiv preprint_
_arXiv:1309.4168_ .

Miller, G. A. 1995. Wordnet: a lexical database for english. _Com-_
_munications of the ACM_ 38(11):39–41.



Myers, G. K.; Bolles, R. C.; Luong, Q.-T.; Herson, J. A.; and Aradhye, H. B. 2005. Rectification and recognition of text in 3-d scenes.
_IJDAR_ 7.

Norouzi, M.; Mikolov, T.; Bengio, S.; Singer, Y.; Shlens, J.; Frome,
A.; Corrado, G. S.; and Dean, J. 2014. Zero-shot learning by
convex combination of semantic embeddings. In _ICLR_ .

Parikh, D., and Grauman, K. 2011. Interactively building a discriminative vocabulary of nameable attributes. In _CVPR_ .

Patterson, G., and Hays, J. 2012. Sun attribute database: Discovering, annotating, and recognizing scene attributes. In _CVPR_ .

Rohrbach, M.; Stark, M.; Szarvas, G.; Gurevych, I.; and Schiele,
B. 2010. What helps where–and why? semantic relatedness for
knowledge transfer. In _CVPR_ .

Rohrbach, M.; Ebert, S.; and Schiele, B. 2013. Transfer learning
in a transductive setting. In _NIPS_ .

Rohrbach, M.; Stark, M.; and Schiele, B. 2011. Evaluating knowledge transfer and zero-shot learning in a large-scale setting. In
_CVPR_ .

Salton, G., and Buckley, C. 1988. Term-weighting approaches in
automatic text retrieval. _Information processing & management_ .

Sermanet, P.; Eigen, D.; Zhang, X.; Mathieu, M.; Fergus, R.; and
LeCun, Y. 2014. Overfeat: Integrated recognition, localization and
detection using convolutional networks. In _ICLR_ .

Shen, Y.; He, X.; Gao, J.; Deng, L.; and Mesnil, G. 2014. A convolutional latent semantic model for web search. Technical report,
Technical Report MSR-TR-2014-55, Microsoft Research.

Socher, R.; Ganjoo, M.; Manning, C. D.; and Ng, A. 2013. Zeroshot learning through cross-modal transfer. In _NIPS_ .

Torki, M., and Elgammal, A. 2010. Putting local features on a
manifold. In _CVPR_ .

Torresani, L.; Szummer, M.; and Fitzgibbon, A. 2010. Efficient
object category recognition using classemes. In _ECCV_ .

van Hout, J.; Akbacak, M.; Castan, D.; Yeh, E.; and Sanchez,
M. 2013. Extracting spoken and acoustic concepts for multimedia event detection. In _ICASSP_ .

Wu, S.; Bondugula, S.; Luisier, F.; Zhuang, X.; and Natarajan,
P. 2014. Zero-shot event detection using multi-modal fusion of
weakly supervised concepts. In _CVPR_ .

Yang, L., and Hanjalic, A. 2010. Supervised reranking for web
image search. In _ACM Multimedia_ .


**Appendix A: Proof** _p_ ( _e_ _c_ _|v_ ) **for** _s_ ( _·, ·_ ) = _s_ _p_ ( _·, ·_ )
We start by Eq. 5 while replacing _s_ ( _·, ·_ ) as _s_ _p_ ( _·, ·_ ).


_p_ ( _e_ _c_ _|v_ ) _∝_ � _s_ _p_ ( _θ_ ( _e_ _c_ ) _, θ_ ( _c_ _i_ )) _p_ ( _c_ _i_ _|v_ )


_i_



� �

_i_



(7)
_θ_ ( _c_ _i_ )

_c_
_∥θc_ _i_ _∥_ _[v]_ _[i]_ �



_∝_
�


_i_



_T_ ~~_T_~~
_θ_ ( _e_ _c_ ) _θ_ ( _c_ _i_ )
_∥θe_ _c_ _∥∥θc_ _i_ _∥_ _[v]_ _c_ _[i]_ _[∝]_ _[θ]_ _∥_ [(] _θe_ _[e]_ _[c]_ _c_ [)] _∥_



which is the dot product between _[θ]_ _∥_ [(] _θe_ _[e]_ _[c]_ _c_ [)] _∥_ _T_ [representing the]



event embedding, and [�] _i_ _∥θθc_ ( _c_ _ii_ _∥_ ) _[v]_ _c_ _[i]_ [representing the video]

embedding, which is a function of _ψ_ ( _v_ _c_ _[i]_ [) =] _[ {][θ]_ _[v]_ [(] _[c]_ _[i]_ [) =]
_θ_ ( _c_ _i_ ) _v_ _c_ _[i]_ _[}]_ [. This equation clarifies our notion of distributional]
semantic embedding of videos and relating it to event title



event embedding, and [�] _i_



_i_


