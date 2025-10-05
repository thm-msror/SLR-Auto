## **Type-to-Track: Retrieve Any Object** **via Prompt-based Tracking**

**Pha Nguyen** [1] **, Kha Gia Quach** [2] **, Kris Kitani** [3] **, Khoa Luu** [1]

1 CVIU Lab, University of Arkansas 2 pdActive Inc. 3 Robotics Institute, Carnegie Mellon University
1 `{panguyen, khoaluu}@uark.edu` 2 `kquach@ieee.org` 3 `kkitani@cs.cmu.edu`

```
             uark-cviu.github.io/Type-to-Track

```

(a) Appearance on a fine-grained scale (b) and on a class-agnostic scale.


Figure 1: An example of the responsive _Type-to-Track_ . The user provides a video sequence and a prompting
request. During tracking, the system is able to discriminate appearance attributes to track the target subjects
accordingly and iteratively responds to the user’s tracking request. Each box color represents a unique identity.


**Abstract**


One of the recent trends in vision problems is to use natural language captions
to describe the objects of interest. This approach can overcome some limitations
of traditional methods that rely on bounding boxes or category annotations. This
paper introduces a novel paradigm for Multiple Object Tracking called _Type-to-_
_Track_, which allows users to track objects in videos by typing natural language
descriptions. We present a new dataset for that Grounded Multiple Object Tracking
task, called _GroOT_, that contains videos with various types of objects and their
corresponding textual captions describing their appearance and action in detail.
Additionally, we introduce two new evaluation protocols and formulate evaluation
metrics specifically for this task. We develop a new efficient method that models
a transformer-based eMbed-ENcoDE-extRact framework ( _MENDER_ ) using the
third-order tensor decomposition. The experiments in five scenarios show that our
_MENDER_ approach outperforms another two-stage design in terms of accuracy
and efficiency, up to 14.7% accuracy and 4 _×_ speed faster.


**1** **Introduction**


Tracking the movement of objects in videos is a challenging task that has received significant attention
in recent years. Various methods have been proposed to tackle this problem, including deep learning
techniques. However, despite these advances, there is still room for improvement in intuitiveness
and responsiveness. One potential way to improve object tracking in videos is to incorporate user
input into the tracking process. Traditional Visual Object Tracking (VOT) methods typically require


37th Conference on Neural Information Processing Systems (NeurIPS 2023).


Table 1: Comparison of current datasets. # denotes the number of the corresponding item. **Bold** numbers are the
best number in each sub-block, while **highlighted** numbers are the best across all sub-blocks.

|Datasets|Task NLP #Videos #Frames #Tracks #AnnBoxes #Words #Settings|
|---|---|
|**OTB100** [8]<br>**VOT-2017** [9]<br>**GOT-10k** [10]<br>**TrackingNet** [11]|SOT<br>✗<br>100<br>59K<br>100<br>59K<br>-<br>-<br>SOT<br>✗<br>60<br>21K<br>60<br>21K<br>-<br>-<br>SOT<br>✗<br>10K<br>1.5M<br>10K<br>1.5M<br>-<br>-<br>SOT<br>✗<br>**30K**<br>**14.43M**<br>**30K**<br>**14.43M**<br>-<br>-|
|**MOT17** [12]<br>**TAO** [13]<br>**MOT20** [14]<br>**BDD100K** [15]|**MOT**<br>✗<br>14<br>11.2K<br>1.3K<br>0.3M<br>-<br>-<br>**MOT**<br>✗<br>1.5K<br>**2.2M**<br>8.1K<br>0.17M<br>-<br>-<br>**MOT**<br>✗<br>8<br>13.41K<br>3.83K<br>2.1M<br>-<br>-<br>**MOT**<br>✗<br>**2K**<br>318K<br>**130.6K**<br>**3.3M**<br>-<br>-|
|**LaSOT** [6]<br>**TNL2K** [7]<br>**Ref-DAVIS** [16]<br>**Refer-YTVOS** [17]|SOT<br>✓<br>1.4K<br>**3.52M**<br>1.4K<br>**3.52M**<br>9.8K<br>1<br>SOT<br>✓<br>2K<br>1.24M<br>2K<br>1.24M<br>10.8K<br>1<br>VOS<br>✓<br>150<br>94K<br>400+<br>-<br>10.3K<br>**2**<br>VOS<br>✓<br>**4K**<br>1.24M<br>**7.4K**<br>131K<br>**158K**<br>**2**|
|**Ref-KITTI** [18]<br>**GroOT (Ours)**|**MOT**<br>✓<br>18<br>6.65K<br>-<br>-<br>3.7K<br>1<br>**MOT**<br>✓<br>**1,515**<br>**2.25M**<br>**13.3K**<br>**2.57M**<br>**256K**<br>**5**|



users to manually select objects in the video by points [ 1 ], bounding boxes [ 2, 3 ], or trained object
detectors [ 4, 5 ]. Thus, in this paper, we introduce a new paradigm, called _Type-to-Track_, to this task
that combines responsive typing input to guide the tracking of objects in videos. It allows for more
intuitive and conversational tracking, as users can simply type in the name or description of the object
they wish to track, as illustrated in Fig. 1. Our intuitive and user-friendly _Type-to-Track_ approach has
numerous potential applications, such as surveillance and object retrieval in videos.


We present a new Grounded Multiple Object Tracking dataset named _GroOT_ that is more advanced
than existing tracking datasets [ 6, 7 ]. _GroOT_ contains videos with various types of multiple objects
and detailed textual descriptions. It is 2 _×_ larger and more diverse than any existing datasets, and it
can construct many different evaluation settings. In addition to three easy-to-construct experimental
settings, we propose two new settings for prompt-based visual tracking. It brings the total number
of settings to five, which will be presented in Section 5. These new experimental settings challenge
existing designs and highlight the potential for further advancements in our proposed research topic.


In summary, this work addresses the use of natural language to guide and assist the Multiple Object
Tracking (MOT) tasks with the following contributions. First, a novel paradigm named _Type-to-Track_
is proposed, which involves responsive and conversational typing to track any objects in videos.
Second, a new _GroOT_ dataset is introduced. It contains videos with various types of objects and their
corresponding textual descriptions of 256K words describing definition, appearance, and action. Next,
two new evaluation protocols that are tracking by _retrieval prompts_ and _caption prompts_, and three
class-agnostic tracking metrics are formulated for this problem. Finally, a new transformer-based
eMbed-ENcoDE-extRact framework ( _MENDER_ ) is introduced with third-order tensor decomposition
as the first efficient approach for this task. Our contributions in this paper include a novel paradigm, a
rich semantic dataset, an efficient methodology, and challenging benchmarking protocols with new
evaluation metrics. These contributions will be advantageous for the field of Grounded MOT by
providing a valuable foundation for the development of future algorithms.


**2** **Related Work**


**2.1** **Visual Object Tracking Datasets and Benchmarks**


**Datasets.** To develop and train VOT models for the computer vision task of tracking objects in videos,
various datasets have been created and widely used. Some of the most popular datasets for VOT
are OTB [ 19, 8 ], VOT [ 9 ], GOT [ 10 ], MOT challenges [ 12, 14 ] and BDD100K [ 15 ]. Visual object
tracking has two sub-tasks: _Single Object Tracking_ (SOT) and _Multiple Object Tracking_ (MOT).
Table 1 shows that there is a wide variety of object tracking datasets in both types available, each
with its own strengths and weaknesses. Existing datasets with NLP [ 6, 7 ] only support the SOT task,
while our _GroOT_ dataset supports MOT with approximately 2 _×_ larger in description size.


**Benchmarks.** Current benchmarks for tracking can be broadly classified into two main categories:
_Tracking by Bounding Box_ and _Tracking by Natural Language_, depending on the type of initialization.


2


Table 2: Comparison of key features of tracking
methods. **Cls-agn** is for class-agnostic, while **Feat**
is for the approach of feature fusion and **Stages**
indicates the number of stages in the model design
incorporating NLP into the tracking task. **NLP**
indicates how text is utilized for the tracker: _assist_
(w/ box) or can _initialize_ (w/o box).



Table 3: Statistics of _GroOT_ ’s settings.





|Datasets|#Videos #Frames #Tracks #AnnBoxes #Words Parts|
|---|---|
|**MOT17**_∗∗_<br>Train<br>Test<br>**Total**|7<br>5,316<br>546_∗_<br>112,297_∗_<br>3,792<br>(1)<br>7<br>5,919<br>785_∗_<br>188,076_∗_<br>5,757<br>(2)<br>14<br>11,235<br>1,331_∗_<br>300,373_∗_<br>9,549|
|**TAO**_∗∗_<br>Train<br>Val<br>Test<br>**Total**|500<br>764,526<br>2,645<br>54,639<br>19,222<br>(3)<br>993<br>1,460,666<br>5,485<br>113,112<br>39,149<br>(4)<br>914<br>2,221,846<br>7,972<br>164,650<br>-<br>2,407<br>4,447,038<br>16,089<br>332,401<br>58,371|
|**MOT20**_∗∗_<br>Train<br>Test<br>**Total**|4<br>8,931<br>2,332_∗_<br>1,336,920_∗_<br>-<br>(5)<br>4<br>4,479<br>1,501_∗_<br>765,465_∗_<br>-<br>(6)<br>8<br>13,410<br>3,833_∗_<br>2,102,385_∗_<br>-|
|**GroOT**_∗∗_<br>**nm**<br>**syn**<br>**def**<br>**cap**<br>**retr**|1,515<br>2,249,837<br>13,294<br>2,570,509<br>21,424<br>_all_<br>1,515<br>2,249,837<br>13,294<br>2,570,509<br>53,540<br>_all_<br>1,515<br>2,249,837<br>13,294<br>2,570,509<br>99,218<br>_all_<br>1,507<br>2,236,427<br>9,461<br>468,124<br>67,920<br>_w/o MOT20_<br>993<br>1,460,666<br>1,952<br>-<br>13,935<br>_uses (4)_|


**MENDER** **MOT** **init** ✓ **attn** **single** _∗_ [Statistics from the official site, including objects other than human.](https://motchallenge.net/)

_∗∗_ [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/)

|Approach|Task NLP Cls-agn Feat Stages|
|---|---|
|GTI [27]<br>SOT<br>assist<br>✗<br>concat<br>**single**<br>TransVLT [28]<br>SOT<br>assist<br>✗<br>**attn**<br>**single**|GTI [27]<br>SOT<br>assist<br>✗<br>concat<br>**single**<br>TransVLT [28]<br>SOT<br>assist<br>✗<br>**attn**<br>**single**|
|TrackFormer [4]|**MOT**<br>–<br>✗<br>–<br>–|
|MDETR+TFm<br>TransRMOT [18]<br>**MENDER**|**MOT**<br>**init**<br>✓<br>**attn**<br>two<br>**MOT**<br>**init**<br>✓<br>**attn**<br>two<br>**MOT**<br>**init**<br>✓<br>**attn**<br>**single**|



Previous benchmarks [ 20, 19, 8, 9, 21, 22, 22, 23 ] were limited to test videos before the emergence
of deep trackers. The first publicly available benchmarks for visual tracking were OTB-2013 [ 19 ]
and OTB-2015 [ 8 ], consisting of 50 and 100 video sequences, respectively. GOT-10k [ 10 ] is a
benchmark featuring 10K videos classified into 563 classes and 87 motions. TrackingNet [ 11 ], a
subset of the object detection benchmark YT-BB [ 24 ], includes 31K sequences. Furthermore, there
are long-term tracking benchmarks such as OxUvA [ 25 ] and LaSOT [ 6 ]. OxUvA spans 14 hours
of video in 337 videos, comprising 366 object tracks. On the other hand, LaSOT [ 6 ] is a languageassisted dataset consisting of 1.4K sequences with 9.8K words in their captions. In addition to
these benchmarks, TNL2K [ 7 ] includes 2K video sequences for natural language-based tracking and
focuses on expressing the attributes. LaSOT [ 6 ] and TNL2K [ 7 ] support one benchmarking setting
with their provided prompts, while our _GroOT_ dataset supports five settings. Ref-KITTI [ 18 ] is built
upon the KITTI [ 26 ] dataset and contains only two categories, including `car` and `pedestrian`, while
our _GroOT_ dataset focuses on category-agnostic tracking, and outnumbers the frames and settings.


A similar task with a different nomenclature to the Grounded MOT task is Referring Video Object
Segmentation (Ref-VOS) [ 16, 17 ], which primarily measures the overlapping area between the ground
truth and prediction for a single foreground object in each caption, with less emphasis on densely
tracking multiple objects over time. In contrast, our proposed _Type-to-Track_ paradigm is distinct
in its focus on _responsively_ and _conversationally_ typing to track any objects in videos, requiring
maintaining the temporal motions of multiple objects of interest.


**2.2** **Grounded Object Tracking**


**Grounded Vision-Language Models** accurately map language concepts onto visual observations
by understanding both vision content and natural language. For instance, visual grounding [ 29 ]
seeks to identify the location of nouns or short phrases (such as a black hat or a blue bird) within an
image. Grounded captioning [ 30, 31, 32 ] can generate text descriptions and align predicted words
with object regions in an image. Visual dialog [ 33 ] enables meaningful dialogues with humans about
visual content using natural, conversational language. Some visual dialog systems may incorporate
referring expression recognition [34] to resolve expressions in questions or answers.


**Grounded Single Object Tracking** is limited to tracking a single object with box-initialized and
language-assisted methods. The GTI [ 27 ] framework decomposes the tracking by language task into
three sub-tasks: Grounding, Tracking, and Integration, and generates tubelet predictions frame-byframe. AdaSwitcher [ 7 ] module identifies tracking failure and switches to visual grounding for better
tracking. [ 35 ] introduce a unified system using attention memory and cross-attention modules with
learnable semantic prototypes. Another transformer-based approach [ 28 ] is presented including a
cross-modal fusion module, task-specific heads, and a proxy token-guided fusion module.


**2.3** **Discussion**


Most existing datasets and benchmarks for object tracking are limited in their coverage and diversity
of language and visual concepts. Additionally, the prompts in the existing Grounded SOT benchmarks
do not contain variations in covering many objects in a single prompt, which limits the application of
existing trackers in practical scenarios. To address this, we present a new dataset and benchmarking


3








(a) Our MOT17 [12] subset sample with captions in both action and appearance types.


(b) Our TAO [13] subset samples with captions. **Best viewed in color and zoom in.**


Figure 2: Example sequences and annotations in our dataset.



(a) Our MOT17 [12] subset.


(b) Our TAO [13] subset.


Figure 3: Some words in our language
description.



metrics to support the emerging trend of the Grounded MOT, where the goal is to align language
descriptions with fine-grained regions or objects in videos.


As shown in Table 2, most of the recent methods for the Grounded SOT task are not class-agnostic,
meaning they require prior knowledge of the object. GTI [ 27 ] and TransVLT [ 28 ] need to input
the initial bounding box, while TrackFormer [ 4 ] need the pre-defined category. The operation used
in [ 27 ] to fuse visual and textual features is _concatenation_ which can only support prompts describing
a single object. A Grounded MOT can be constructed by integrating a grounded object detector, i.e.
MDETR [ 36 ], and an object tracker, i.e. TrackFormer [ 4 ]. However, this approach is low-efficient
because the visual features have to be extracted multiple times. In contrast, our proposed MOT
approach _MENDER_ formulates third-order _attention_ to adaptively focus on many targets, and it is an
efficient _single-stage_ and _class-agnostic_ framework. The scope of _class-agnostic_ in our approach is
constructing a large vocabulary of concepts via a visual-textual corpus, following [37, 38, 39].


**3** **Dataset Overview**


**3.1** **Data Collection and Annotation**


Existing object tracking datasets are typically designed for specific types of video scenes [ 40, 41, 42,
43, 44, 2 ]. To cover a diverse range of scenes, _GroOT_ was created using official videos and bounding
box annotations from the MOT17 [ 12 ], TAO [ 13 ] and MOT20 [ 14 ]. The MOT17 dataset comprises
14 sequences with diverse environmental conditions such as crowded scenes, varying viewpoints,
and camera motion. The TAO dataset is composed of videos from seven different datasets, such
as the ArgoVerse [ 45 ] and BDD [ 15 ] datasets containing outdoor driving scenes, while LaSOT [ 6 ]
and YFCC100M [ 46 ] datasets include in-the-wild internet videos. Additionally, the AVA [ 47 ],
Charades [ 48 ], and HACS [ 49 ] datasets include videos depicting human-human and human-object
interactions. By combining these datasets, _GroOT_ covers multiple types of scenes and encompasses a
wide range of 833 objects. This diversity allows for a wide range of object classes with captions to be
included, making it an invaluable resource for training and evaluating visual grounding algorithms.


We release our textual description annotations in COCO format [ 50 ]. Specifically, a new key
`‘captions’` which is a list of strings is attached to each `‘annotations’` item in the official
annotation. In the MOT17 subset, we attempt to maintain two types of caption for well-visible
objects: one describes the _appearance_ and the other describes the _action_ . For example, the caption
for a well-visible person might be `[‘a man wearing a gray shirt’, ‘person walking on`
`the street’]` as shown in Fig. 2a. However, 10% of tracklets only have one caption type, and
3% do not have any captions due to their low visibility. The physical characteristics of a person or
their personal accessories, such as their clothing, bag color, and hair color are considered to be part
of their appearance. Therefore, the appearance captions include verbs `‘carrying’` or `‘holding’`
to describe personal accessories. In the TAO subset, objects other than humans have one caption


4


describing appearance, for instance, `[‘a red and black scooter’]` . Objects that are human
have the same two types of captions as the MOT17 subset. An example is shown in Fig. 2b. These
captions are consistently annotated throughout the tracklets. Fig. 3 is the word-cloud visualization of
our annotations.


**3.2** _**Type-to-Track**_ **Benchmarking Protocols**


Let **V** be a video sample lasts _t_ frames, where **V** = **I** _t_ _| t < |_ **V** _|_ and **I** _t_ be the image sample at a
� �
particular time step _t_ . We define a request prompt **P** that describes the objects of interest, and **T** _t_ is
the set of tracklets of interest up to time step _t_ . The _Type-to-Track_ paradigm requires a tracker network
_T_ ( **I** _t_ _,_ **T** _t−_ 1 _,_ **P** ) that efficiently take into account **I** _t_, **T** _t−_ 1, and **P** to produce **T** _t_ = _T_ ( **I** _t_ _,_ **T** _t−_ 1 _,_ **P** ) .
To advance the task of multiple object retrieval, another benchmarking set is created in addition to
the _GroOT_ dataset. While training and testing sets follow a _One-to-One_ scenario, where each caption
describes a single tracklet, the new retrieval set contains prompts that follow a _One-to-Many_ scenario,
where a short prompt describes multiple objects. This scenario highlights the need for diverse
methods to improve the task of multiple object retrieval. The retrieval set is provided with a subset of
tracklets in the TAO validation set and three custom _retrieval prompts_ that change throughout the


tracking process in a video _{_ **P** _t_ 1 =0 _,_ **P** _t_ 2 _,_ **P** _t_ 3 _}_, as depicted in Fig. 1(a). The _retrieval prompts_ are
generated through a semi-automatic process that involves: (i) selecting the most commonly occurring
category in the video, and (ii) cascadingly filtering to the object that appears for the longest duration.
In contrast, the _caption prompts_ are created by joining tracklet captions in the scene and keeping
it consistent throughout the tracking period. We name these two evaluation scenarios as _tracklet_
_captions_ **cap** and _object retrieval_ **retr** . With three more easy-to-construct scenarios, five scenarios
in total will be studied for the experiments in Section 5. Table 3 presents the statistics of the five
settings, and the data portions are highlighted in the corresponding colors.


**3.3** **Class-agnostic Evaluation Metrics**


As indicated in [ 51 ], long-tailed classification is a very challenging task in imbalanced and large-scale
datasets such as TAO. This is because it is difficult to distinguish between similar fine-grained classes,
such as bus and van, due to the class hierarchy. Additionally, it is even more challenging to treat every
class independently. The traditional method of evaluating tracking performance leads to inadequate
benchmarking and undesired tracking results. In our _Type-to-Track_ paradigm, the main task is not to
classify objects to their correct categories but to retrieve and track the object of interest. Therefore,
to alleviate the negative effect, we reformulate the original per-category metrics of MOTA [ 52 ],
IDF1 [53], HOTA [54] into class-agnostic metrics:



_t_ [GT] _[t]_



_t_ [(][GT] _[CLS]_ [1] [)] _[t]_
(1)



1
MOTA =
_|CLS_ _[n]_ _|_



_CLS_ _[n]_
�


_cls_



1 _−_ �
�



_t_ [(][FN] _[t]_ [ +][ FP] _[t]_ [ +][ IDS] _[t]_ [)]
~~�~~ _t_ [GT] _[t]_



_t_ [(][FN] _[t]_ [ +][ FP] _[t]_ [ +][ IDS] _[t]_ [)] _[CLS]_ [1]
~~�~~ _t_ [(][GT] _[CLS]_ [1] [)] _[t]_



�



�
_cls_ [, CA-MOTA][ = 1] _[−]_



1
IDF1 =
_|CLS_ _[n]_ _|_



_CLS_ _[n]_
�


_cls_



� 2 _×_ IDTP2 + _×_ IDFP IDTP + IDFN



(2 _×_ IDTP) _CLS_ 1
CA-IDF1 =

� _cls_ [,] (2 _×_ IDTP + IDFP + IDFN) _CLS_ 1


(2)



1
HOTA =
_|CLS_ _[n]_ _|_



_CLS_ _[n]_
�


_cls_



� ~~_√_~~



DetA _·_ AssA
�



CA-HOTA = ~~�~~
_cls_ [,]



(DetA _CLS_ 1 ) _·_ (AssA _CLS_ 1 ) (3)



where _CLS_ _[n]_ is the category set, size _n_ is reduced to 1 by combining all elements: _CLS_ _[n]_ _→_ _CLS_ [1] .


**4** **Methodology**


**4.1** **Problem Formulation**


Given the image **I** _t_ and the request prompt **P** describing the objects of interest, which can adaptively
change between { **P** _t_ 1, **P** _t_ 2, **P** _t_ 3 } in the **retr** setting, and _K_ is the prompt’s length _|_ **P** _|_ = _K_,
let _enc_ ( _·_ ) and _emb_ ( _·_ ) be the visual encoder and the word embedding model to extract features
of image tokens and prompt tokens, respectively. The resulting outputs, _enc_ ( **I** _t_ ) _∈_ R _[M]_ _[×][D]_ and


5


_emb_ ( **P** ) _∈_ R _[K][×][D]_, where _D_ is the length of feature dimensions. A list of region-prompt associations
**C** _t_, which contains objects’ bounding boxes and their confident scores, can be produced by Eqn. (4) :



**C** _t_ = _dec_ _enc_ ( **I** _t_ ) _×_ [¯] _emb_ ( **P** ) [⊺] _, enc_ ( **I** _t_ ) = **c** _i_ = ( _c_ _x_ _, c_ _y_ _, c_ _w_ _, c_ _h_ _, c_ _conf_ ) _i_ _| i < M_
_γ_ � � � �



(4)
_t_



where ( _×_ [¯] ) is an operation representing the region-prompt correlation, that will be elaborated in the
next section, _dec_ _γ_ [(] _[·][,][ ·]_ [)] [ is an object decoder taking the similarity and the image features to decode to]

object locations, thresholded by a scoring parameter _γ_ (i.e. _c_ _conf_ _≥_ _γ_ ). For simplicity, the cardinality
of the set of objects _|_ **C** _t_ _|_ = _M_, implying each image token produces one region-text correlation.


We define **T** _t_ = **tr** _j_ = ( _tr_ _x_ _, tr_ _y_ _, tr_ _w_ _, tr_ _h_ _, tr_ _conf_ _, tr_ _id_ ) _j_ _| j < N_
� � _t_ [produced by the tracker] _[ T]_ [,]

where _N_ = _|_ **T** _t_ _|_ is the cardinality of current tracklets. _i_, _j_, _k_, and _t_ are consistently denoted as
indexers for objects, tracklets, prompt tokens, and time steps for the rest of the paper.


**Remark 1** _**Third-order Tensor Modeling.**_ _Since the Type-to-Track paradigm requires three input com-_
_ponents_ **I** _t_ _,_ **T** _t−_ 1 _, and_ **P** _, an_ _**auto-regressive single-stage end-to-end framework**_ _can be formulated_
_via third-order tensor modeling._


To achieve this objective, a combination of initialization, object decoding, visual encoding, feature
extraction, word embedding, and aggregation can be formulated as in Eqn. (5):



**T** _t_ =



_initialize_ ( **C** _t_ ) _t_ = 0


_dec_ **1** _D×D×D_ _×_ 1 _enc_ ( **I** _t_ ) _×_ 2 _ext_ ( **T** _t−_ 1 ) _×_ 3 _emb_ ( **P** ) _, enc_ ( **I** _t_ ) _∀t >_

� _γ_ � �



_dec_

_γ_



(5)
**1** _D×D×D_ _×_ 1 _enc_ ( **I** _t_ ) _×_ 2 _ext_ ( **T** _t−_ 1 ) _×_ 3 _emb_ ( **P** ) _, enc_ ( **I** _t_ ) _∀t >_ 0
� �



where _ext_ ( _·_ ) denotes the visual feature extractor of the set of tracklets, _ext_ ( **T** _t−_ 1 ) _∈_ R _[N]_ _[×][D]_,
**1** _D×D×D_ is an all-ones tensor has size _D × D × D_, ( _×_ _n_ ) is the _n_ -mode product of the thirdorder tensor [ 55 ] to aggregate many types of token [1], and _initialize_ ( _·_ ) is the function to ascendingly
assign unique identities to tracklets for the first time those tracklets appear.


Let _T ∈_ R _[M]_ _[×][N]_ _[×][K]_ be the resulting tensor _T_ = **1** _D×D×D_ _×_ 1 _enc_ ( **I** _t_ ) _×_ 2 _ext_ ( **T** _t−_ 1 ) _×_ 3 _emb_ ( **P** ) .
The objective function can be expressed as the log softmax of the positive region-tracklet-prompt
triplet over all possible triplets, defined in Eqn. (6):



� [�]



_θ_ _enc,ext,emb_ _[∗]_ [= arg] _θ_ _enc,ext,emb_ max



log� _K_ _N_ exp( _M_ _T_ _ijk_ )

� ~~�~~ _l_ ~~�~~ _n_ ~~�~~ _m_ [exp(] _[T]_ _[lnm]_ [)]



(6)



where _θ_ denotes the network’s parameters, the combination of the _i_ _[th]_ image token, the _j_ _[th]_ tracklet,
and the _k_ _[th]_ prompt token is the correlated triplet.


In the next subsection, we elaborate our model design for the tracking function _T_ ( **I** _t_ _,_ **T** _t−_ 1 _,_ **P** ),
named _MENDER_, as defined in Eqn. (5), and loss functions for the problem objective in Eqn. (6).


**4.2** **MENDER for Multiple Object Tracking by Prompts**


The correlation in Eqn. (5) has the cubic time and space complexity _O_ ( _n_ [3] ), which can be intractable
as the input length grows and hinder the model scalability.


**Remark 2** _**Correlation Simplification.**_ _Since both_ _enc_ ( _·_ ) _and_ _ext_ ( _·_ ) _are visual encoders, the_
_region-prompt correlation can be equivalent to the tracklet-prompt correlation. Therefore, the_
_region-tracklet-prompt correlation tensor T can be simplified to lower the computation footprint._


To design that goal, the extractor and encoder share network weights for computational efficiency:



_ext_ ( **T** _t−_ 1 ) _j_ = _ext_ _{_ **tr** _j_ _}_ _t−_ 1 = _enc_ ( **I** _t−_ 1 ) _i_ : **c** _i_ _�→_ **tr** _j_, therefore ( _T_ : _j_ : ) _t−_ 1 = ( _T_ _i_ :: ) _t_ : **c** _i_ _�→_ **tr** _j_ 2
� � � � � �



(7)
where _T_ : _j_ : and _T_ _i_ :: are lateral and horizontal slices. In layman’s terms, the **region-prompt** correlation
at the time step _t −_ 1 is equivalent to the **tracklet-prompt** correlation at the time step _t_, as visualized
in Fig. 4(a). Therefore, one practically needs to model the **region-tracklet** and **tracklet-prompt**


1
implemented by a single Python code with Numpy: `np.einsum(‘ai, bj, ck -> abc’, P, I, T)` .
2 If **P** changes, the equivalence still holds true, see Appendix for the full algorithm.


6


(a) Because the tracklet set **T** _t−_ 1 pools visual features
of the image **I** _t−_ 1, the **region-prompt** is equivalent with
**tracklet-prompt** (only need to filter unassigned objects).



(b) The structure of our proposed _MENDER_ . It employs a visual backbone to
extract visual features and a word embedding to extract textual features. We
model the **tracklet-prompt** correlation _ext_ ( **T** _t−_ 1 ) _×_ [¯] _emb_ ( **P** ) [⊺] instead of
the **region-prompt** to avoid unnecessary computation caused by `no-object`
tokens [56]. **Best viewed in color and zoom in.**



Figure 4: The _auto-regressive_ manner takes advantage of the equivalent components. Simplifying the correlation
in (a) turns the solution to _MENDER_ in (b), and reduces complexity to _O_ ( _n_ [2] ) where _n_ denotes the size of tokens.


correlations which reduces time and space complexity from _O_ ( _n_ [3] ) to _O_ ( _n_ [2] ), significantly lowering
computation footprint. We alternatively rewrite the decoding step in Eqn. (5) as follows:



_enc_ ( **I** _t_ ) _×_ [¯] _ext_ ( **T** _t−_ 1 ) [⊺] [�] _×_ _ext_ ( **T** _t−_ 1 ) _×_ [¯] _emb_ ( **P** ) [⊺] [�] _, enc_ ( **I** _t_ )
�
��



**T** _t_ = _dec_
_γ_



�



_∀t >_ 0 (8)



**Correlation Representations.** In our approach, the correlation operation ( _×_ [¯] ) is modelled by the
_multi-head cross-attention_ mechanism [ 57 ], as depicted in Fig. 4(b). The attention matrix can be
computed as:



(9)
�



_σ_ ( **X** ) _×_ [¯] _σ_ ( **Y** ) = _A_ **X** _|_ **Y** = softmax



⊺
_σ_ ( **X** ) _× W_ _Q_ **[X]** � _×_ � _σ_ ( **Y** ) _× W_ _K_ **[Y]** �
� [�] ~~_√_~~ _D_



~~_√_~~



_D_



where **X** and **Y** tokens are one of these types: region, tracklet, prompt. _σ_ ( _·_ ) is one of the operations
_enc_ ( _·_ ), _emb_ ( _·_ ), _ext_ ( _·_ ) as the corresponding operation to **X** or **Y** . Superscript _W_ _Q_, _W_ _K_, and _W_ _V_ are
the projection matrices corresponding to **X** or **Y** as in the attention mechanism.


Then, the attention weight from the image **I** _t_ to the prompt **P** are computed by the matrix multiplication for _A_ **I** _|_ **T** and _A_ **T** _|_ **P** to aggregate the information from two matrices as in Eqn. (8) . The result is
the matrix _A_ **I** _|_ **T** _×_ **T** _|_ **P** = _A_ **I** _|_ **T** _× A_ **T** _|_ **P** that shows the correlation between each input or output. Then,
the resulting attention matrix _A_ **I** _|_ **T** _×_ **T** _|_ **P** is used to produce the object representations at time _t_ :


**Z** _t_ = _A_ **I** _|_ **T** _×_ **T** _|_ **P** _×_ � _emb_ ( **P** ) _× W_ _V_ **[P]** � + _A_ **I** _|_ **T** _×_ � _ext_ ( **T** _t−_ 1 ) _× W_ _V_ **[T]** � (10)


**Object Decoder** _dec_ ( _·_ ) utilizes context-aware features **Z** _t_ that are capable of preserving identity
information while adapting to changes in position. The tracklet set **T** _t_ is defined in the _auto-regressive_
manner to adjust to the movements of the object being tracked as in Eqn. (8) . For decoding the final
output at any frame, the decoder transforms the object representation by a 3-layer FFN to predict
bounding boxes and confidence scores for frame _t_ :



**T** _t_ = **tr** _j_ = ( _tr_ _x_ _, tr_ _y_ _, tr_ _w_ _, tr_ _h_ _, tr_ _conf_ ) _j_
� � _t_



_tr_ _conf_ _≥γ_
= FFN **Z** _t_ + _enc_ ( **I** _t_ ) (11)
� �



where the identification information of tracklets, represented by _tr_ _id_, is not determined directly by
the FFN model. Instead, the _tr_ _id_ value is set when the tracklet is first initialized and maintained till
its end, similar to _tracking-by-attention_ approaches [4, 58, 59, 60].


7


**4.3** **Training Losses**


To achieve the training objective function as in Eqn. (6), we formulate the objective function into two
loss functions _L_ **I** _|_ **T** and _L_ **T** _|_ **P** for correlation training and one loss _L_ _GIoU_ for decoder training:


_L_ = _γ_ **T** _|_ **P** _L_ **T** _|_ **P** + _γ_ **I** _|_ **T** _L_ **I** _|_ **T** + _γ_ _GIoU_ _L_ _GIoU_ (12)


where _γ_ **T** _|_ **P**, _γ_ **I** _|_ **T**, and _γ_ _GIoU_ are corresponding coefficients, which are set to 0 _._ 3 by default.


**Alignment Loss** _L_ **T** _|_ **P** is a contrastive loss, which is used to assure the alignment of the ground-truth
object feature and caption pairs ( **T** _,_ **P** ) which can be obtained in our dataset. There are two alignment
losses used, one for all objects normalized by the number of positive prompt tokens and the other for
all prompt tokens normalized by the number of positive objects. The total loss can be expressed as:


_L_ **T** _|_ **P** =



exp � _ext_ ( **T** ) [⊺] _j_ _[×][ emb]_ [(] **[P]** [)] _[k]_ �


_K_

�

[⊺]



_K_
�



_l_ exp ~~�~~ _ext_ ( **T** ) [⊺] _j_ _[×][ emb]_ [(] **[P]** [)] _[l]_ ~~�~~



exp _emb_ ( **P** ) [⊺] _k_ _[×][ ext]_ [(] **[T]** [)] _[j]_
� �


_N_

�

[⊺]



_N_
�



exp _emb_ ( **P** ) [⊺] _k_ _[×][ ext]_ [(] **[T]** [)] _[l]_
_l_ ~~�~~ ~~�~~



1
_−_ _|_ **P** [+] _|_



_|_ **P** [+] _|_
� log


_k_



_|_ **P** [+] _|_
�



_|_ **T** [+] _|_
� log

_j_



�



1
_−_ _|_ **T** [+] _|_



�



(13)


where **P** [+] and **I** [+] are the sets of positive prompts and image tokens corresponding to the selected
_enc_ ( **I** ) _i_ and _emb_ ( **P** ) _k_, respectively.


**Objectness Losses.** To model the track’s temporal changes, our network learns from training samples
that capture both appearance and motion generated by two adjacent frames:



exp � _ext_ ( **T** ) [⊺] _j_ _[×][ enc]_ [(] **[I]** [)] _[i]_ �
� ~~�~~ _Nl_ [exp] ~~�~~ _ext_ ( **T** ) [⊺] _j_ _[×][ enc]_ [(] **[I]** [)] _[l]_ ~~�~~



�



_L_ **I** _|_ **T** = _−_



_N_
� log

_j_



_N_
�



, and _L_ _GIoU_ =



_N_
� _ℓ_ _GIoU_ ( **tr** _j_ _,_ **obj** _i_ ) (14)

_j_



_L_ **I** _|_ **T** is the log-softmax loss to guide the tokens’ alignment as similar to Eqn. (13) . In the _L_ _GIoU_ loss,
**obj** _i_ is the ground truth object corresponding to **tr** _j_ . The optimal assignment between **tr** _j_ or **obj** _i_
to the ground truth object is computed efficiently by the Hungarian algorithm, following DETR [ 56 ].
_ℓ_ _GIoU_ is the Generalized IoU loss [61].


**5** **Experimental Results**


**5.1** **Implementation Details**


**Experimental Scenarios.** We create three types of prompt: _category name_ **nm**, _category synonyms_


**syn**, _category definition_ **def** . One _tracklet captions_ **cap** scenario is constructed by our detailed


annotations and one more _objects retrieval_ **retr** scenario is given in our custom request prompts as
described in Subsec. 3.2. The dataset contains 833 classes, each has a name and a corresponding set of
synonyms that are different names for the same category, such as `[man`, `woman`, `human`, `pedestrian`,
`boy`, `girl`, `child]` for `person` . Additionally, each category is described by a _category definition_
sentence. This definition makes the model deal with the variations in the text prompts. We join the
names, synonyms, definitions, or captions and filter duplicates to construct the prompt. Trained models
use as the same type as testing. We annotated the raw tracking data of the best-performant tracker
(i.e., BoT-SORT [ 62 ] at 80.5% MOTA and 80.2% IDF1) at the time we constructed experiments and
used it as the sub-optimal ground truth of MOT17 and MOT20 (parts _(2, 4)_ in Table 3). That is also
the raw data we used to evaluate all our ablation studies.


**Datasets and Metrics.** RefCOCO+ [ 63 ] and Flickr30k [ 64 ] serve as pre-trained datasets for acquiring
a vocabulary of visual-textual concepts [ 37 ]. The _ext_ ( _·_ ) operation is not involved in this training
step. After obtaining a pre-trained model from RefCOCO+ and Flickr30k, we train and evaluate
our model for the proposed _Type-to-Track_ task on all five scenarios on our _GroOT_ dataset and the
first-three scenarios for MOT20 [ 14 ]. The tracking performance is reported in class-agnostic metrics
CA-MOTA, CA-IDF1, and CA-HOTA as in Subsec. 3.3 and mAP50 as defined in [13]..


**Tokens Production.** _emb_ ( _·_ ) utilizes RoBERTa [ 65 ] to convert the text input into a sequence of
numerical tokens. The tokens are fed into the RoBERTa-base model for text encoding using a


8


Table 4: Ablation studies. **sim** indicates whether
the correlation is the _simplified_ Eqn. (8) or the
Eqn. (5) . See 5.1 for the abbreviations. The two
first settings get only one word for the request
prompt, therefore, tensor _T_ is an unsqueezed
matrix, resulting in no difference in **nm** ( ✗ ) vs



Table 5: Comparisons to the two-stage baseline design. In
each dataset, the from-top-to-bottom scenarios are **syn**,

|Col1|GroOT - MOT17 Subset|
|---|---|
|MDETR + TFm<br>**MENDER**|62.60<br>64.70<br>519<br>1382<br>0.793<br>2.2<br>**65.10**<br>**71.10**<br>**554**<br>**1348**<br>**0.874**<br>**10.3**|
|MDETR + TFm<br>**MENDER**|62.60<br>64.70<br>519<br>1382<br>0.793<br>2.2<br>**67.30**<br>**72.40**<br>**568**<br>**1322**<br>**0.877**<br>**10.3**|
|MDETR + TFm<br>**MENDER**|44.80<br>45.20<br>193<br>1945<br>0.619<br>2.1<br>**59.50**<br>**54.80**<br>**201**<br>**1734**<br>**0.688**<br>**7.8**|


|Col1|GroOT - TAO Subset|
|---|---|
|MDETR + TFm<br>**MENDER**|21.30<br>33.20<br>2945<br>5834<br>0.184<br>3.1<br>**25.70**<br>**36.10**<br>**3212**<br>**5048**<br>**0.198**<br>**11.2**|
|MDETR + TFm<br>**MENDER**|14.60<br>21.40<br>1944<br>6493<br>0.137<br>3.1<br>**16.80**<br>**27.70**<br>**2547**<br>**6118**<br>**0.158**<br>**10.5**|
|MDETR + TFm<br>**MENDER**|15.30<br>23.60<br>2132<br>6354<br>0.156<br>3.0<br>**20.70**<br>**32.00**<br>**3103**<br>**5192**<br>**0.182**<br>**8.7**|
|MDETR + TFm<br>**MENDER**|25.70<br>26.40<br>513<br>3993<br>0.387<br>3.1<br>**32.90**<br>**39.30**<br>**645**<br>**3194**<br>**0.430**<br>**11.5**|



**GroOT**         - MOT20 Subset


|Col1|GroOT - MOT17 Subset|
|---|---|
|**nm**<br>✗/✓|67.00<br>71.20<br>544<br>1352<br>0.876<br>10.3|
|**syn**<br>✗/✓|65.10<br>71.10<br>554<br>1348<br>0.874<br>10.3|
|**def**<br>✗<br>✓|67.00<br>72.10<br>556<br>1343<br>0.876<br>5.8<br>**67.30**<br>**72.40**<br>**568**<br>**1322**<br>**0.877**<br>**10.3**|
|**cap**<br>✗<br>✓|58.20<br>53.20<br>**289**<br>1751<br>0.674<br>3.4<br>**59.50**<br>**54.80**<br>201<br>**1734**<br>**0.688**<br>**7.8**|


|Col1|GroOT - TAO Subset|
|---|---|
|**nm**<br>✓|27.30<br>37.20<br>3523<br>4284<br>0.212<br>11.2|
|**syn**<br>✓|25.70<br>36.10<br>3212<br>5048<br>0.198<br>11.2|
|**def**<br>✗<br>✓|15.20<br>27.30<br>2452<br>6253<br>0.154<br>6.2<br>**16.80**<br>**27.70**<br>**2547**<br>**6118**<br>**0.158**<br>**10.5**|
|**cap**<br>✗<br>✓|20.30<br>31.80<br>2943<br>5242<br>0.188<br>4.3<br>**20.70**<br>**32.00**<br>**3103**<br>**5192**<br>**0.184**<br>**8.7**|
|**retr**<br>✗<br>✓|32.40<br>38.40<br>630<br>3238<br>0.423<br>7.6<br>**32.90**<br>**39.30**<br>**645**<br>**3194**<br>**0.430**<br>**11.5**|


|Col1|GroOT - MOT20 Subset|
|---|---|
|**nm**<br>✗/✓|72.40<br>67.50<br>823<br>2498<br>0.826<br>7.6|
|**syn**<br>✗/✓|70.90<br>65.30<br>809<br>2509<br>0.823<br>7.6|
|**def**<br>✗<br>✓|72.90<br>67.70<br>**823**<br>2489<br>0.826<br>4.3<br>**72.10**<br>**67.10**<br>812<br>**2503**<br>**0.825**<br>**7.6**|



12-layer transformer network with 768 hidden units and 12 self-attention heads per layer. _enc_ ( _·_ )
is implemented using a ResNet-101 [ 66 ] as the backbone to extract visual features from the input
image. The output of the ResNet is processed by a Deformable DETR encoder [ 67 ] to generate visual
tokens. For each dimension, we use sine and cosine functions with different frequencies as positional
encodings, similar to [68]. A feature resizer combining a list of (Linear _,_ LayerNorm _,_ Dropout) is
used to map to size _D_ = 512 for all token producers.


**5.2** **Ablation Study**


**Comparisons in Different Scenarios.** Table 4 shows comparisons in the performance of different
prompt inputs. For MOT17 and MOT20, the _category name_ is `‘person’`, while _category definition_ is
`‘a human being’` . Since the prompt by _category definition_ is short, it does not differ much from the

**nm** setting. However, the **syn** setting shuffles between some words, resulting in a slight decrease

in CA-MOTA and CA-IDF1. The **cap** setting results in prompts that contain more diverse and
complex vocabulary, and more context-specific information. It is more difficult for the model to
accurately localize the objects and identify their identity within the image, as it needs to take into
account a wider range of linguistic cues, resulting in a decrease in performance compared to **def**
(59.5% CA-MOTA and 54.8% CA-IDF1 vs 67.3% CA-MOTA and 72.4% CA-IDF1 on MOT17).


For TAO, the **def** setting has a significant number of variations and many tenuous connections in
the scene context, for example, `‘an aircraft that has a fixed wing and is powered by`
`propellers or jets’` for the `airplane` category. Therefore, it results in a decrease in performance (16.8% CA-MOTA and 27.7% CA-IDF1) compared to **cap** (20.7% CA-MOTA and 32.0%

CA-IDF1), because the **cap** setting is more specific on the object level than category level. The best

performant setting is **nm** (27.3% CA-MOTA and 37.2% CA-IDF1), where names are combined.


**Simplied Attention Representations.** Table 4 also presents the effectiveness of different attention
representations of the full tensor _T_ (denoted by ✗ ) and the simplified correlation (denoted by ✓ ). The
performance is reported with frame per second (FPS), which is self-measured on one GPU NVIDIA
RTX 3060 12GB. Overall, the performance of simplified correlation is witnessed with a superior
speed of up to 2 _×_ (7.8 FPS vs 3.4 FPS of **cap** on MOT17 and 11.5 FPS vs 7.6 FPS of **retr** on
TAO), resulting in and a slight increase in accuracy due to attention stability, and precision gain.


9


Table 6: Comparisons to the state-of-the-art approaches on the _category name_ **nm** setting.

|Approach|Cls-agn|CA-IDF1 CA-MOTA CA-HOTA MT ML AssA DetA LocA IDs|
|---|---|---|
|ByteTrack [69]<br>TrackFormer [4]<br>QuasiDense [70]<br>CenterTrack [71]<br>TraDeS [72]<br>CTracker [73]|✗<br>✗<br>✗<br>✗<br>✗<br>✗|77.3<br>80.3<br>63.1<br>957<br>516<br>52.7<br>55.6<br>81.8<br>3,378<br>68.0<br>74.1<br>57.3<br>1,113<br>246<br>54.1<br>60.9<br>82.8<br>2,829<br>66.3<br>68.7<br>53.9<br>957<br>516<br>52.7<br>55.6<br>81.8<br>3,378<br>64.7<br>67.8<br>52.2<br>816<br>579<br>51.0<br>53.8<br>81.5<br>3,039<br>63.9<br>69.1<br>52.7<br>858<br>507<br>50.8<br>55.2<br>81.8<br>3,555<br>57.4<br>66.6<br>49.0<br>759<br>570<br>45.2<br>53.6<br>81.3<br>5,529|
|**MENDER**|✓|67.1<br>65.0<br>53.9<br>678<br>648<br>54.4<br>53.6<br>83.4<br>3,266|



**5.3** **Comparisons with A Baseline Design**


Due to the new proposed topic, no current work has the same scope or directly solves our problem.
Therefore, we compare our proposed _MENDER_ against a two-stage baseline tracker in Table 5. We
use current SOTA methods to develop this approach, i.e., MDETR [ 36 ] for the grounded detector,
while TrackFormer [ 4 ] for the object tracker. It is worth noting that our _MENDER_ relies on direct
regression to locate and track the object of interest, without the need for an explicit grounded object
detection stage. Table 5 shows our proposed _MENDER_ outperforms the baseline on both CA-MOTA
and CA-IDF1 metrics in all four settings _category synonyms_, _category definition_, _tracklet captions_
and _object retrieval_ (25.7% vs. 21.3%, 16.8% vs. 14.6%, 20.7% vs. 15.3% and 32.9% vs. 25.7%
CA-MOTA on TAO), while can maintain up to 4 _×_ run-time speed (10.3 FPS vs 2.2 FPS). The results
indicate that training a single-stage network enhances efficiency and reduces errors by avoiding
separate feature extractions for both detection and tracking steps.


**5.4** **Comparisons with State-of-the-Art Approaches**


The _category name_ **nm** setting is also the official MOT benchmark. Table 6 is the comparison on
the _category name_ setting on the official leaderboard of MOT17, comparing our proposed _MENDER_
with other state-of-the-art approaches, including ByteTrack [ 69 ] and TrackFormer [ 4 ]. Note that
our proposed _MENDER_ is one of the first attempts at the Grounded MOT task, not to achieve the
top rankings on the general MOT leaderboard. In contrast, other SOTA approaches benefit from the
efficient single-category design in their separate object detectors, while our single-stage design is
agnostic to the category and for flexible textual input. Compared to TrackFormer [ 4 ], our proposed
_MENDER_ only demonstrates a marginal decrease in identity assignment (67.1% vs 68.0% CA-IDF1
and 53.9% vs 57.3% CA-HOTA). The decrease in the MOTA detection metric stems from our
detector’s design, which is a detector integrating prompts as a flexible input.


**6** **Conclusion**


We have presented a novel problem of _Type-to-Track_, which aims to track objects using natural
language descriptions instead of bounding boxes or categories, and a large-scale dataset to advance
this task. Our proposed _MENDER_ model reduces the computational complexity of third-order
correlations by designing an efficient attention method that scales quadratically w.r.t the input sizes.
Our experiments on three datasets and five scenarios demonstrate that our model achieves state-ofthe-art accuracy and speed for class-agnostic tracking.


**Limitations.** While our proposed metrics effectively evaluate the proposed _Type-to-Track_ problem,
they may not be ideal for measuring precision-recall characteristics in retrieval tasks. Additionally, the
lack of the question-answering task in data and problem formulation may limit the algorithm to not
being able to provide language feedback such as clarification or alternative suggestions. Additional
benchmarks incorporating question-answering are excellent research avenues for future work. While
the performance of our proposed _MENDER_ may not be optimal for well-defined categories, it paves
the way for exploring new avenues in open vocabulary and open-world scenarios [74].


**Broader Impacts.** The _Type-to-Track_ problem and the proposed _MENDER_ model have the potential
to impact various fields, such as surveillance and robotics, where recognizing object interactions is a
crucial task. By reformulating the problem with text support, the proposed methodology can improve
the intuitiveness and responsiveness of tracking, making it more practical for video input support in
large-language models [ 75 ] and real-world applications similar to ChatGPT. However, it could bring
potential negative impacts related to human trafficking by providing a video retrieval system via text.


10


**Acknowledgment.** This work is partly supported by NSF Data Science, Data Analytics that are Robust and
Trusted (DART), and Google Initiated Research Grant. We also thank Utsav Prabhu and Chi-Nhan Duong
for their invaluable discussions and suggestions and acknowledge the Arkansas High-Performance Computing
Center for providing GPUs.


**References**


[1] Pha Nguyen, Thanh-Dat Truong, Miaoqing Huang, Yi Liang, Ngan Le, and Khoa Luu. Self-supervised
domain adaptation in crowd counting. In _2022 IEEE International Conference on Image Processing (ICIP)_,
pages 2786–2790. IEEE, 2022. 2, 24


[2] Kha Gia Quach, Pha Nguyen, Huu Le, Thanh-Dat Truong, Chi Nhan Duong, Minh-Triet Tran, and Khoa
Luu. Dyglip: A dynamic graph model with link prediction for accurate multi-camera multiple object
tracking. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages
13784–13793, 2021. 2, 4


[3] Kha Gia Quach, Huu Le, Pha Nguyen, Chi Nhan Duong, Tien Dai Bui, and Khoa Luu. Depth perspectiveaware multiple object tracking. _arXiv preprint arXiv:2207.04551_, 2022. 2


[4] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. Trackformer: Multiobject tracking with transformers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pages 8844–8854, 2022. 2, 3, 4, 7, 10, 23


[5] Pha Nguyen, Kha Gia Quach, John Gauch, Samee U Khan, Bhiksha Raj, and Khoa Luu. Utopia:
Unconstrained tracking objects without preliminary examination via cross-domain adaptation. _arXiv_
_preprint arXiv:2306.09613_, 2023. 2


[6] Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, and
Haibin Ling. Lasot: A high-quality benchmark for large-scale single object tracking. In _Proceedings of the_
_IEEE Conference on Computer Vision and Pattern Recognition_, pages 5374–5383, 2019. 2, 3, 4


[7] Xiao Wang, Xiujun Shu, Zhipeng Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, and Feng Wu. Towards more flexible and accurate object tracking with natural language: Algorithms and benchmark. In
_Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pages
13763–13773, June 2021. 2, 3, 19


[8] Yi Wu, Jongwoo Lim, and Ming-Hsuan Yang. Object tracking benchmark. _IEEE Transactions on Pattern_
_Analysis and Machine Intelligence_, 37(9):1834–1848, 2015. 2, 3


[9] Matej Kristan, Jiri Matas, Aleš Leonardis, Tomas Vojir, Roman Pflugfelder, Gustavo Fernandez, Georg
Nebehay, Fatih Porikli, and Luka Cehovin. A novel performance evaluation methodology for single-target [ˇ]
trackers. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 38(11):2137–2155, Nov 2016.
2, 3


[10] Lianghua Huang, Xin Zhao, and Kaiqi Huang. Got-10k: A large high-diversity benchmark for generic
object tracking in the wild. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 2019. 2, 3


[11] Matthias Muller, Adel Bibi, Silvio Giancola, Salman Alsubaihi, and Bernard Ghanem. Trackingnet:
A large-scale dataset and benchmark for object tracking in the wild. In _Proceedings of the European_
_Conference on Computer Vision (ECCV)_, pages 300–317, 2018. 2, 3


[12] A. Milan, L. Leal-Taixé, I. Reid, S. Roth, and K. Schindler. MOT16: A benchmark for multi-object
tracking. _arXiv:1603.00831 [cs]_, March 2016. arXiv: 1603.00831. 2, 4, 16


[13] Achal Dave, Tarasha Khurana, Pavel Tokmakov, Cordelia Schmid, and Deva Ramanan. Tao: A large-scale
benchmark for tracking any object. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow,_
_UK, August 23–28, 2020, Proceedings, Part V 16_, pages 436–454. Springer, 2020. 2, 4, 8, 16


[14] Patrick Dendorfer, Hamid Rezatofighi, Anton Milan, Javen Shi, Daniel Cremers, Ian Reid, Stefan Roth,
Konrad Schindler, and Laura Leal-Taixé. Mot20: A benchmark for multi object tracking in crowded scenes.
_arXiv preprint arXiv:2003.09003_, 2020. 2, 4, 8


[15] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and
Trevor Darrell. BDD100K: A diverse driving dataset for heterogeneous multitask learning. In _IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition (CVPR)_, June 2020. 2, 4


[16] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video object segmentation with language referring
expressions. In _Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia,_
_December 2–6, 2018, Revised Selected Papers, Part IV 14_, pages 123–141. Springer, 2019. 2, 3


[17] Seonguk Seo, Joon-Young Lee, and Bohyung Han. Urvos: Unified referring video object segmentation
network with a large-scale benchmark. In _Computer Vision–ECCV 2020: 16th European Conference,_
_Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16_, pages 208–223. Springer, 2020. 2, 3


11


[18] Dongming Wu, Wencheng Han, Tiancai Wang, Xingping Dong, Xiangyu Zhang, and Jianbing Shen.
Referring multi-object tracking. In _Proceedings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pages 14633–14642, 2023. 2, 3


[19] Yi Wu, Jongwoo Lim, and Ming Hsuan Yang. Object tracking benchmark. _IEEE Transactions on Pattern_
_Analysis & Machine Intelligence_, 37(9):1834, 2015. 2, 3


[20] Pengpeng Liang, Erik Blasch, and Haibin Ling. Encoding color information for visual tracking: Algorithms
and benchmark. _IEEE Transactions on Image Processing_, 24(12):5630–5644, 2015. 3


[21] A Li, M Lin, Y Wu, MH Yang, and S Yan. NUS-PRO: A New Visual Tracking Challenge. _IEEE_
_Transactions on Pattern Analysis and Machine Intelligence_, 38(2):335–349, 2016. 3


[22] Siyi Li and Dit-Yan Yeung. Visual object tracking for unmanned aerial vehicles: A benchmark and new
motion models. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 31, 2017. 3


[23] Hamed Kiani Galoogahi, Ashton Fagg, Chen Huang, Deva Ramanan, and Simon Lucey. Need for speed:
A benchmark for higher frame rate object tracking. In _Proceedings of the IEEE International Conference_
_on Computer Vision_, pages 1125–1134, 2017. 3


[24] Esteban Real, Jonathon Shlens, Stefano Mazzocchi, Xin Pan, and Vincent Vanhoucke. Youtubeboundingboxes: A large high-precision human-annotated data set for object detection in video. In
_proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, pages 5296–5305, 2017.
3


[25] Jack Valmadre, Luca Bertinetto, Joao F Henriques, Ran Tao, Andrea Vedaldi, Arnold WM Smeulders,
Philip HS Torr, and Efstratios Gavves. Long-term tracking in the wild: A benchmark. In _Proceedings of_
_the European Conference on Computer Vision (ECCV)_, pages 670–685, 2018. 3


[26] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti
dataset. _The International Journal of Robotics Research_, 32(11):1231–1237, 2013. 3


[27] Zhengyuan Yang, Tushar Kumar, Tianlang Chen, Jingsong Su, and Jiebo Luo. Grounding-trackingintegration. _IEEE Transactions on Circuits and Systems for Video Technology_, 31(9):3433–3443, 2020. 3,
4


[28] Haojie Zhao, Xiao Wang, Dong Wang, Huchuan Lu, and Xiang Ruan. Transformer vision-language
tracking via proxy token guided cross-modal fusion. _Pattern Recognition Letters_, 2023. 3, 4


[29] Fenglin Liu, Xian Wu, Shen Ge, Xuancheng Ren, Wei Fan, Xu Sun, and Yuexian Zou. Dimbert: learning
vision-language grounded representations with disentangled multimodal-attention. _ACM Transactions on_
_Knowledge Discovery from Data (TKDD)_, 16(1):1–19, 2021. 3


[30] Wenqiao Zhang, Haochen Shi, Siliang Tang, Jun Xiao, Qiang Yu, and Yueting Zhuang. Consensus graph
representation learning for better grounded image captioning. In _Proceedings of the AAAI Conference on_
_Artificial Intelligence_, volume 35, pages 3394–3402, 2021. 3


[31] Wenhui Jiang, Minwei Zhu, Yuming Fang, Guangming Shi, Xiaowei Zhao, and Yang Liu. Visual cluster
grounding for image captioning. _IEEE Transactions on Image Processing_, 31:3920–3934, 2022. 3


[32] Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang.
Grit: A generative region-to-text transformer for object understanding. _arXiv preprint arXiv:2212.00280_,
2022. 3


[33] Haonan Yu, Haichao Zhang, and Wei Xu. Interactive grounded language acquisition and generalization in
a 2d world. In _International Conference on Learning Representations_, 2018. 3


[34] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects
in photographs of natural scenes. In _Proceedings of the 2014 conference on empirical methods in natural_
_language processing (EMNLP)_, pages 787–798, 2014. 3


[35] Yihao Li, Jun Yu, Zhongpeng Cai, and Yuwen Pan. Cross-modal target retrieval for tracking by natural
language. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_,
pages 4931–4940, 2022. 3


[36] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion.
Mdetr-modulated detection for end-to-end multi-modal understanding. In _Proceedings of the IEEE/CVF_
_International Conference on Computer Vision_, pages 1780–1790, 2021. 4, 10, 23


[37] Alireza Zareian, Kevin Dela Rosa, Derek Hao Hu, and Shih-Fu Chang. Open-vocabulary object detection
using captions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_,
pages 14393–14402, 2021. 4, 8


[38] Muhammad Maaz, Hanoona Bangalath Rasheed, Salman Hameed Khan, Fahad Shahbaz Khan,
Rao Muhammad Anwer, and Ming-Hsuan Yang. Multi-modal transformers excel at class-agnostic object
detection. _arXiv_, 2021. 4


12


[39] Tanmay Gupta, Amita Kamath, Aniruddha Kembhavi, and Derek Hoiem. Towards general purpose vision
systems: An end-to-end task-agnostic vision-language architecture. In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_, pages 16399–16409, 2022. 4


[40] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc
Van Gool. The 2017 DAVIS Challenge on Video Object Segmentation. _arXi_, 2017. 4


[41] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In _ICCV_, 2019. 4


[42] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang.
YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark. _arXiv_, 2018. 4


[43] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille,
Philip HS Torr, and Song Bai. Occluded video instance segmentation: A benchmark. _International Journal_
_of Computer Vision_, 130(8):2022–2039, 2022. 4


[44] Namdar Homayounfar, Justin Liang, Wei-Chiu Ma, and Raquel Urtasun. Videoclick: Video object
segmentation with a single click. _arXiv preprint arXiv:2101.06545_, 2021. 4


[45] Ming-Fang Chang, John Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett,
De Wang, Peter Carr, Simon Lucey, Deva Ramanan, et al. Argoverse: 3d tracking and forecasting with
rich maps. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages
8748–8757, 2019. 4


[46] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian
Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. _Communications of the ACM_,
59(2):64–73, 2016. 4


[47] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra
Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, et al. Ava: A video dataset of
spatio-temporally localized atomic visual actions. In _Proceedings of the IEEE conference on computer_
_vision and pattern recognition_, pages 6047–6056, 2018. 4


[48] Gunnar A Sigurdsson, Gül Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood
in homes: Crowdsourcing data collection for activity understanding. In _Computer Vision–ECCV 2016:_
_14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14_,
pages 510–526. Springer, 2016. 4


[49] Hang Zhao, Antonio Torralba, Lorenzo Torresani, and Zhicheng Yan. Hacs: Human action clips and
segments dataset for recognition and temporal localization. In _Proceedings of the IEEE/CVF International_
_Conference on Computer Vision_, pages 8668–8678, 2019. 4


[50] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár,
and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014:_
_13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_, pages
740–755. Springer, 2014. 4


[51] Siyuan Li, Martin Danelljan, Henghui Ding, Thomas E Huang, and Fisher Yu. Tracking every thing in the
wild. In _Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022,_
_Proceedings, Part XXII_, pages 498–515. Springer, 2022. 5, 24


[52] Keni Bernardin and Rainer Stiefelhagen. Evaluating multiple object tracking performance: the clear mot
metrics. _EURASIP Journal on Image and Video Processing_, 2008:1–10, 2008. 5


[53] Ergys Ristani, Francesco Solera, Roger Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures
and a data set for multi-target, multi-camera tracking. In _European conference on computer vision_, pages
17–35. Springer, 2016. 5


[54] Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixé, and
Bastian Leibe. Hota: A higher order metric for evaluating multi-object tracking. _International journal of_
_computer vision_, 129(2):548–578, 2021. 5


[55] Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. _SIAM review_, 51(3):455–500,
2009. 6


[56] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey
Zagoruyko. End-to-end object detection with transformers. In _Computer Vision–ECCV 2020: 16th_
_European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16_, pages 213–229. Springer,
2020. 7, 8


[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_,
30, 2017. 7


[58] Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. Motr: End-to-end
multiple-object tracking with transformer. In _European Conference on Computer Vision (ECCV)_, 2022. 7,
23


13


[59] Pha Nguyen, Kha Gia Quach, Chi Nhan Duong, Son Lam Phung, Ngan Le, and Khoa Luu. Multicamera multi-object tracking on the move via single-stage global association approach. _arXiv preprint_
_arXiv:2211.09663_, 2022. 7


[60] Pha Nguyen, Kha Gia Quach, Chi Nhan Duong, Ngan Le, Xuan-Bac Nguyen, and Khoa Luu. Multi-camera
multiple 3d object tracking on the move for autonomous vehicles. In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_, pages 2569–2578, 2022. 7


[61] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese.
Generalized intersection over union. In _The IEEE Conference on Computer Vision and Pattern Recognition_
_(CVPR)_, June 2019. 8


[62] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Bot-sort: Robust associations multi-pedestrian tracking.
_arXiv preprint arXiv:2206.14651_, 2022. 8


[63] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in
referring expressions. In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The_
_Netherlands, October 11-14, 2016, Proceedings, Part II 14_, pages 69–85. Springer, 2016. 8


[64] Bryan A. Plummer, Liwei Wang, Christopher M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and
Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-tosentence models. _IJCV_, 123(1):74–93, 2017. 8


[65] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv_
_preprint arXiv:1907.11692_, 2019. 8, 23


[66] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.
In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 770–778, 2016.
9


[67] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable {detr}: Deformable
transformers for end-to-end object detection. In _International Conference on Learning Representations_,
2021. 9, 23


[68] Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, and Huaxia Xia.
End-to-end video instance segmentation with transformers. In _Proceedings of the IEEE/CVF conference_
_on computer vision and pattern recognition_, pages 8741–8750, 2021. 9


[69] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and
Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In _Proceedings of_
_the European Conference on Computer Vision (ECCV)_, 2022. 10


[70] Jiangmiao Pang, Linlu Qiu, Xia Li, Haofeng Chen, Qi Li, Trevor Darrell, and Fisher Yu. Quasi-dense
similarity learning for multiple object tracking. In _Proceedings of the IEEE/CVF conference on computer_
_vision and pattern recognition_, pages 164–173, 2021. 10


[71] Xingyi Zhou, Vladlen Koltun, and Philipp Krähenbühl. Tracking objects as points. In _European Conference_
_on Computer Vision_, pages 474–490. Springer, 2020. 10


[72] Jialian Wu, Jiale Cao, Liangchen Song, Yu Wang, Ming Yang, and Junsong Yuan. Track to detect and
segment: An online multi-object tracker. In _Proceedings of the IEEE/CVF conference on computer vision_
_and pattern recognition_, pages 12352–12361, 2021. 10


[73] Jinlong Peng, Changan Wang, Fangbin Wan, Yang Wu, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li,
Feiyue Huang, and Yanwei Fu. Chained-tracker: Chaining paired attentive regression results for end-to-end
joint multiple-object detection and tracking. In _Computer Vision–ECCV 2020: 16th European Conference,_
_Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16_, pages 145–161. Springer, 2020. 10


[74] Siyuan Li, Tobias Fischer, Lei Ke, Henghui Ding, Martin Danelljan, and Fisher Yu. Ovtrack: Openvocabulary multiple object tracking. In _Proceedings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pages 5567–5577, 2023. 10


[75] OpenAI. Gpt-4 technical report. _arXiv_, 2023. 10


[76] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Tubedetr: Spatio-temporal
video grounding with transformers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pages 16442–16453, 2022. 21


[77] Jingkuan Song, Ruimin Lang, Xiaosu Zhu, Xing Xu, Lianli Gao, and Heng Tao Shen. 3d self-attention for
unsupervised video quantization. In _Proceedings of the 43rd International ACM SIGIR Conference on_
_Research and Development in Information Retrieval_, pages 1061–1070, 2020. 21


[78] Haoyu Lan, Alzheimer Disease Neuroimaging Initiative, Arthur W Toga, and Farshid Sepehrband. Threedimensional self-attention conditional gan with spectral normalization for multimodal neuroimaging
synthesis. _Magnetic resonance in medicine_, 86(3):1718–1733, 2021. 21


14


[79] Yang Liu, Idil Esen Zulfikar, Jonathon Luiten, Achal Dave, Deva Ramanan, Bastian Leibe, Aljoša Ošep,
and Laura Leal-Taixé. Opening up open world tracking. In _Proceedings of the IEEE/CVF Conference on_
_Computer Vision and Pattern Recognition_, pages 19045–19055, 2022. 24


15


## **Appendix**

**7** **Dataset Taxonomy**


**nm** **syn** **def** **cap** **retr**



_category_


_name_



_category_

_synonyms_



_category_
_definition_



_tracklet_

_appearance_



_tracklet_


_action_



_objects_
_retrieval_



Figure 5: Full list of the types of caption to construct the request prompt for the corresponding settings.





**Prompt types:** _appearance_ _action_ _appearance_ _object:_ _human: action_

_appearance_ _and appearance_


Figure 6: Types of prompt for the construction of our settings for each dataset.


The reason for creating two new evaluation scenarios **cap** and **retr** is that they are more specific on the object
level than on the category level. This is because defining objects by category synonyms and category name and
definition is not sufficient to accurately describe them, and leads to ambiguous results. By focusing on the object
level, the benchmarking sets can provide more accurate and meaningful evaluations of multiple object retrieval
methods.


We include a comprehensive taxonomy of prompt types used to construct our settings. However, the **retr**
setting on the MOT17 could not be constructed because test annotations for this dataset are not available. To
construct this setting, bounding boxes will be filtered to the corresponding _retrieval prompt_ when it changes.


Section 8 describes how to construct this _retrieval prompt_ . The MOT20 dataset requires extensive annotations
and has a larger number of low-visible people due to the crowd view. Therefore, its annotations are not ready to
be released at the moment.


**8** **Annotation Process**


Instead of collecting new videos, we add annotations to the widely used MOT17 [12] and TAO [13] evaluation
sets. These sets contains diverse and relatively long videos with fast-moving objects, camera motion, various
object sizes, frequent object occlusions, scale changes, motion blur, and similar objects. Another advantage
is that there are typically multiple objects that are present throughout the full sequence, which is desirable for
long-term tracking scenarios.


We entrust 10 professional annotators to annotate all frames. All annotations are manually verified.


16


Then, we post-process the annotations to construct the _retrieval prompts_ . Retrieval prompts are short phrases
or sentences used to retrieve relevant information from the video. The process of generating these prompts
involves two main steps:


1. Select the most commonly occurring category in the video. This is done to ensure that the generated
prompts are relevant to the content of the video and that they capture the main objects or scenes in the
video. For example, if the video is about a soccer game, the most commonly occurring category might
be `‘soccer players’` or `‘soccer ball’` .

2. Filter the category selected in the first step to the object that appears for the longest duration. This
is likely done to ensure that the generated prompts are specific and focused on a particular object or
scene in the video. For example, if the most commonly occurring category in a soccer game video is
`‘soccer players’`, the longest appearing player is selected as the focus of the retrieval prompt.


**9** **Data Format**


**9.1** `categories`


1 `categories` `[{`
2 `‘frequency ’` `: str` `,`
3 `‘id ’` `: int` `,`
4 `‘synset ’` `: str` `,`
5 `‘image_count ’` `: int` `,`
6 `‘instance_count ’` `: int` `,`


10 `}]`


The categories field of the annotation structure stores a mapping of category id to the category name, synonyms,
and definitions. The categories field is structured as an array of dictionaries. Each dictionary in the array
represents a single category.


The keys and values of the dictionary are:


       - `‘frequency’` : A string value that indicates the frequency of the category in the dataset.

       - `‘id’` : An integer value that represents the unique ID assigned to the category.

       - `‘synset’` : A string value that contains a unique identifier for the category.

       - `‘image_count’` : An integer value that indicates the number of images in the dataset that belong to
the category.

       - `‘instance_count’` : An integer value that indicates the number of instances of the category that
appear in the dataset.

       - `‘name’` : A string value that represents the name of the category.

       - `‘synonyms’` : An array of string values that contains synonyms of the category name.

       - `‘def’` : A string value that provides a definition of the category.


**9.2** `annotations`


1 `annotations` `[{`
2 `‘id ’` `: int` `,`
3 `‘image_id ’` `: int` `,`
4 `‘category_id ’` `: int` `,`
5 `‘scale_category ’` `: str` `,`
6 `‘track_id ’` `: int` `,`
7 `‘video_id ’` `: int` `,`
8 `‘segmentation ’` `: [` `polygon` `]` `,`
9 `‘area ’` `: float` `,`
10 `‘bbox ’` `: [` `x` `, y` `, width` `, height` `]` `,`
11 `‘iscrowd ’` `: 0 or 1` `,`


12 `‘captions’:` `[str]`


13 `}]`


17


An object instance annotation is a record that describes a single instance of an object in an image or video. It is
structured as a dictionary that contains a series of key-value pairs, where each key corresponds to a specific field
in the annotation. The fields included in the annotation are:


       - `‘id’` : An integer value that represents the unique ID assigned to the annotation.


       - `‘image_id’` : An integer value that represents the ID of the image that the object instance is part of.


       - `‘category_id’` : An integer value that represents the ID of the category to which the object instance
belongs.


       - `‘scale_category’` : A string value that represents the scale of the object instance with respect to the
category.


       - `‘track_id’` : An integer value that represents the ID of the track to which the object instance belongs.


       - `‘video_id’` : An integer value that represents the ID of the video that the object instance is part of.


       - `‘segmentation’` : An array of polygon coordinates that represent the segmentation mask of the object
instance.


       - `‘area’` : A float value that represents the area of the object instance.


       - `‘bbox’` : An array of four values that represent the bounding box coordinates of the object instance.


       - `‘iscrowd’` : A binary value (0 or 1) that indicates whether the object instance is a single object or a
group of objects.


       - `‘captions’` : An array of string values that contains annotated textual descriptions of the object
instance. The first caption is implicitly annotated as appearance, while the next one is action.


**9.3** `images`


1 `images` `[{`
2 `‘id ’` `: int` `,`
3 `‘frame_index ’` `: int` `,`
4 `‘video_id ’` `: int` `,`
5 `‘file_name ’` `: str` `,`
6 `‘width ’` `: int` `,`
7 `‘height ’` `: int` `,`
8 `‘video ’` `: str` `,`


9 `‘prompt’:` `str`


10 `}]`


The `images` annotations are used to construct request prompts by using the image index at a particular timestamp.
To do this, we use the ‘images" field in the annotation structure, which contains information about the images in
the dataset.


Each image in the dataset is represented as a dictionary object with the following fields:


       - `‘id’` : an integer ID for the image


       - `‘frame_index’` : an integer value representing the frame index or time stamp index of the image


       - `‘video_id’` : an integer ID for the video the image belongs to


       - `‘file_name’` : a string value representing the name of the image file


       - `‘width’` : an integer value representing the width of the image in pixels


       - `‘height’` : an integer value representing the height of the image in pixels


       - `‘video’` : a string value representing the name of the video the image belongs to


       - `‘prompt’` : a string value representing the request prompt for the video at a particular time stamp
which is indexed by `‘frame_index’` .


The `‘prompt’` field is the key field used to construct the request prompt, and it is generated based on the
information in the annotations for the objects in the image. By using the annotations to generate the prompt, it
becomes possible to retrieve specific data about the objects in the image, such as their category, location, and
size.


18


Figure 7: Samples in TNL2K [ 7 ] dataset. The annotations are not meaningful and not discriminative. This
dataset also overlooks many moving objects that are present in the video but are not annotated.


Figure 8: Samples in our _GroOT_ dataset cover almost all moving objects with discriminative captions and a
variety of object types. Labels are shown in the following format: `track_id:np.random.choice(captions)` .


**10** **Examples**


**10.1** **Data Samples**


In Fig. 7, we present some samples from the TNL2K [ 7 ] dataset. This dataset only contains SOT annotations,
which are less meaningful than our dataset. For example, the annotations for some objects in the images, such as
`‘the batman’`, `‘the first person on the left side’`, and `‘the animal riding a motor’`, can be
confusing for both viewers and algorithms. In some cases, the same caption describes two different objects. For
instance, in a video game scene, two opponents are annotated with the same caption `‘the target person`
`the player against with’` . Additionally, this dataset overlooks some large moving objects present in the
video. Therefore, while the TNL2K dataset provides some useful data, it also has significant limitations in terms
of the clarity, discrimination, and consistency of the annotations, and the scope of the annotated objects.


19


On the other hand, Fig. 8 shows some samples from our _GroOT_ dataset, which covers almost all moving objects
in the video and provides distinct captions. The dataset includes a variety of object types and provides accurate
and comprehensive annotations such as `‘white tissues on a table’`, `‘a bottle on the table’`, etc.
This allows for more effective training and evaluation of Grounded MOT algorithms.


**10.2** **Annotations**


Table 7: Examples of annotations in the _GroOT_ dataset.

|Col1|MOT17|
|---|---|
|`‘name’`|`‘person’`|
|`‘synonyms’`|`[‘baby’, ‘child’, ‘boy’, ‘girl’, ‘man’, ‘woman’, ‘perdestrian’, ‘human’]`|
|`‘definition’`|`‘a human being’`|
|`‘captions’`|`[‘man walking on sidewalk’, ‘man wearing a orange shirt’]`|


|Col1|TAO|
|---|---|
|`‘name’`|`‘backpack’`|
|`‘synonyms’`|`[‘backpack’, ‘knapsack’, ‘packsack’, ‘rucksack’, ‘haversack’]`|
|`‘definition’`|`‘a bag carried by a strap on your back or shoulder’`|
|`‘captions’`|`[‘a black colored bag’, ‘the bag is yellow in color’]`|



Table 7 provides examples of annotations in the _GroOT_ dataset. For instance, the MOT17 subset has annotations for the object class _‘person’_ with synonyms including `‘baby’`, `‘child’`, `‘boy’`, `‘girl’`, `‘man’`,
`‘woman’`, `‘pedestrian’` and `‘human’` . The definition for this class is `‘a human being’` and example captions could include `‘man walking on sidewalk’` or `‘man wearing an orange shirt’` . On the other
hand, the TAO subset has annotations for the object class `‘backpack’`, with synonyms such as `‘knapsack’`,
`‘packsack’`, `‘rucksack’`, and `‘haversack’` . The definition for this class is `‘a bag carried by a strap`
`on your back or shoulder’` and example captions could include `‘a black colored bag’` or `‘the bag`
`is yellow in color’` .


**10.3** **Run-time Prompts**


Table 8 presents examples of how the annotations described earlier can be used to construct request prompts
during runtime. In MOT17 and MOT20 subsets, the only category is `‘person’` with randomly selected synonyms `‘man’` and `‘woman’` and the definition `‘a human being’` . The captions for the MOT17 subset include
`‘a man in a suit’`, `‘man wearing an orange shirt’` and `‘a woman in a black shirt and pink`
`skirt’`, while the captions for the MOT20 subset are not annotated.


For TAO subset, the categories in the first example on a driving scene include `‘bus’`, `‘bicycle’` and `‘person’`
with the synonyms being `‘autobus’`, `‘bicycle’` and `‘pedestrian’`, respectively. The definitions for these
categories are `‘a vehicle carrying many passengers; used for public transport’`, `‘a motor`
`vehicle with two wheels and a strong frame’` and `‘a human being’`, respectively. The captions
include `‘a black van’`, `‘silver framed bicycle’`, and `‘person wearing black pants’`, while the
retrieval is `‘people crossing the street’` .


Example 2 shows another example of how annotations can be used to construct request prompts. The
categories in this example include `‘man’`, `‘cup’`, `‘chair’`, `‘sandwich’` and `‘eyeglass’` with the synonyms
being `‘person’`, `‘cup’`, `‘chair’`, `‘sandwich’` and `‘spectacles’`, respectively. The definitions for
these categories are `‘a human being’`, `‘a small open container usually used for drinking;`
`usually has a handle’`, `‘a seat for one person, with a support for the back’`, `‘two (or`
`more) slices of bread with a filling between them’` and `‘optical instrument consisting`
`of a frame that holds a pair of lenses for correcting defective vision’`, respectively.
The joint captions include `‘a man wearing a gray shirt’`, `‘a white cup on the table’`, `‘wooden`
`chair in white room’`, `‘the sandwich is triangle’` and `‘an eyeglass on the table’`, while the
retrieval prompt is `‘a man sitting on a chair eating a sandwich with a cup and an eyeglass`
`in front of him’` .


20


Table 8: Examples of constructing request prompts in the proposed evaluation settings.

|Col1|MOT17|MOT20|
|---|---|---|
|**nm**|`‘person’`|`‘person’`|
|**syn**|`[‘man’, ‘woman’]`|`[‘man’, ‘woman’]`|
|**def**|`[‘a human being’]`|`[‘a human being’]`|
|**cap**|`[‘a man in a suit’, ‘man wearing an orange shirt’,`<br>`‘a woman in a black shirt and pink skirt’]`|N/A|



**TAO**


_Example 1_

|nm|[‘bus’, ‘bicycle’, ‘person’]|
|---|---|
|**syn**|`[‘autobus’, ‘bicycle’, ‘perdestrian’]`|
|**def**|`[‘a vehicle carrying many passengers; used for public transport’,`<br>`‘a motor vehicle with two wheels and a strong frame’,`<br>`‘a human being’]`|
|**cap**|`[‘a black van’, ‘silver framed bicycle’, ‘person wearing black pants’]`|
|**retr**|`‘people crossing the street’`|



_Example 2_

|nm|[‘man’, ‘cup’, ‘chair’, ‘sandwich’, ‘eyeglass’]|
|---|---|
|**syn**|`[‘person’, ‘cup’, ‘chair’, ‘sandwich’, ‘spectacles’]`|
|**def**|`[‘a human being’,`<br>`‘a small open container usually used for drinking; usually has a handle’,`<br>`‘a seat for one person, with a support for the back’,`<br>`‘two (or more) slices of bread with a filling between them’,`<br>`‘optical instrument consisting of a frame that holds a pair`<br>`of lenses for correcting defective vision’]`|
|**cap**|`[‘a man wearing a gray shirt’,`<br>`‘a white cup on the table’,`<br>`‘wooden chair in white room’,`<br>`‘the sandwich is triangle’,`<br>`‘an eyeglasses on the table’]`|
|**retr**|`‘a man sitting on a chair eating a sandwich`<br>`with a cup and an eyeglass in front of him’`|



**11** **Methodolody**


**11.1** **3D Transformers**


**Third-order Tensor Modeling.** Our design of third-order tensor to handle three input components **I** _t_, **T** _t−_ 1,
and **P** influences the design of a novel 3D Transformer. Current temporal visual-text modeling [ 76, 77, 78 ] uses
two dimensions and computes interactions between video and text features, which are then spanned over the
temporal domain. However, our approach is different because it handles three components individually, which
allows for more flexibility and a more nuanced understanding of the data. By modeling as the _n_ -mode product
of the third-order tensor to aggregate many types of tokens, we have presented a general methodology that can
be scaled to multi-modality. The use of the 3D Transformer model, which allows for interactions between these
features over time, can improve the performance of multi-modal models by enabling them to consider a wider
range of input features and their temporal dependencies. Therefore, our design of third-order tensor modeling
has the potential for further research in multi-modality applications.


**11.2** **Symmetric Alignment Loss**


Both the _Alignment Loss_ _L_ **T** _|_ **P** and the _Objectness Loss_ _L_ **I** _|_ **T** are log-softmax loss functions because they
both aim to maximize the similarity of the alignments. The _Alignment Loss_ has two terms, one for all objects
normalized by the number of positive prompt tokens and the other for all prompt tokens normalized by the


21


number of positive objects. In this way, the loss is symmetric and penalizes equally both types of misalignments,
especially for _different modalities_ .


On the other hand, the _Objectness Loss_ only computes from one side and is not necessarily symmetric because
there is a _single modality_ in this case. It only needs to focus on the quality of the object alignment to the image
and does not need to take into account the quality of the image alignment to the object. Consider two objects _A_
and _B_ are equivalent. If we want to maximize the similarity between object _A_ and the correct alignment, we can
achieve this by computing the loss on _A_ with _B_ or _B_ with _A_ . The similarity between object _A_ and object _B_ is
maximized in both cases.


**12** **Additional Details**


**12.1** **Implementation Details**


**Algorithm 1** The inference pipeline of _MENDER_


**Input:** Video **V**, set of tracklets **T** _←_ ∅, set of prompts _{_ **P** _t_ 1 =0 _,_ **P** _t_ 2 _,_ **P** _t_ 3 _}_, _γ_ = 0 _._ 7,
_γ_ _reassign_ = 0 _._ 75, _t_ _tlr_ = 30
1: **for** _t ∈{_ 0 _, · · ·, |_ **V** _| −_ 1 _}_ **do**
2: **if** _t ∈{t_ 1 _, t_ 2 _, t_ 3 _}_ **then**
3: Select **P** _←_ **P** _t_
4: **end if**
5: Draw **I** _t_ _∈_ **V**
6: **if T** = ∅ **then**

7: **if** _t_ = 0 **then**
8: **T** _inactive_ _←_ ∅
9: **else**
10: _% This case happens when_ **P** _changed to a completely new prompt without covering_
_any old tracklets, returning an empty_ **T** _at a timestamp_ _t ≥_ 0 _in line 23. Then the_
_reinitialization is performed as in line 13 to line 14._
11: Pass

12: **end if**
13: **C** _←_ _dec_ _enc_ ( **I** _t_ ) _×_ [¯] _emb_ ( **P** ) [⊺] _, enc_ ( **I** _t_ )
_γ_ � �

14: **T** _←_ _initialize_ ( **C** _t_ ) _% Obtaining tracklet tr_ _id_ _’s_
15: **else**
16: **T** _prev_ _←_ **T** + **T** _inactive_
17: _% If_ **P** _does not change or it covers a subset of the previous objects, our MENDER forward_
_has the ability to attend to the correct targets._



_enc_ ( **I** _t_ ) _×_ [¯] _ext_ ( **T** _prev_ ) [⊺] [�] _×_ _ext_ ( **T** _prev_ ) _×_ [¯] _emb_ ( **P** ) [⊺] [�] _, enc_ ( **I** _t_ )
�
��



18: **T** _←_ _dec_

_γ_



�



19: _% Obtaining tracklet tr_ _id_ _’s_
20: `matched_pairs` _,_ `unmatched_lists` _←_ _cascade_ _ _matching_ ( **T** _,_ **T** _prev_ _, γ_ _reassign_ )
21: `m_new` _,_ `m_old` _←_ `matched_pairs`
22: `unm_new` _,_ `unm_old` _←_ `unmatched_lists`

23: **T** _←_ _update_ ( **T** [ `m_new` ] _,_ **T** _prev_ [ `m_old` ]) + _initialize_ **T** [ `unm_new` ]
� �

24: **T** _inactive_ _←_ _remove_ _ _deprecation_ ( **T** _inactive_ _, t_ _tlr_ ) + **T** _prev_ [ `unm_old` ]
25: **end if**

26: **end for**


**Pseudo-Algorithm.** Alg. 1 is the pseudo-code for our _MENDER_ algorithmic design, a Grounded Multiple
Object Tracker that performs online multiple object tracking via text initialization. The pseudocode provides a
high-level overview of the steps involved in our _MENDER_ method.


**Prompt Change without Losing Track.** If **P** changes to a new prompt between _{_ **P** _t_ 1 _,_ **P** _t_ 2 _,_ **P** _t_ 3 _}_ that still
covers a subset of the objects from the previous prompt, then the **region-prompt** correlation is still partially
equivalent to the **tracklet-prompt** correlation. In this case, our _MENDER_ can still attend to the correct targets
even with the new prompt, because it is trained to maximize the correct pairs which are influenced by the
_Alignment Loss_ and _Objectness Loss_ .


22


Table 9: Traditional metrics struggle to evaluate tracking performance in the presence of uneven datasets and
misclassified categories, leading to biased and extremely poor results.


**P** **sim.** MOTA IDF1 CA-MOTA CA-IDF1 MT IDs mAP FPS


**GroOT**             - MOT17 Subset


**nm** ✗/✓ 67.00 71.20 67.00 71.20 544 1352 0.876 10.3


**syn** ✗/✓ 65.10 71.10 65.10 71.10 554 1348 0.874 10.3


**GroOT**             - TAO Subset


**nm** ✓ 3.10 -53.20 27.30 37.20 3523 4284 0.212 11.2


**syn** ✓ 3.00 -57.10 25.70 36.10 3212 5048 0.198 11.2


**GroOT**             - MOT20 Subset


**nm** ✗/✓ 72.40 67.50 72.40 67.50 823 2498 0.826 7.6


**syn** ✗/✓ 70.90 65.30 70.90 65.30 809 2509 0.823 7.6


However, if the prompt **P** changes completely and no longer covers any of the objects from the previous prompt,
then our _MENDER_ needs to reinitialize the process by recomputing the **region-prompt** . This means that the
algorithm needs to start over with the new **region-prompt** correlation and determine which objects to attend to,
as in line 13 to line 14.


**Tracklets Management.** Our approach involves the _tracking-by-attention_ paradigm [ 4, 58 ] that enables us to
re-identify tracklets for a short period, without requiring any specific re-identification training. This can be
achieved by decoding tracklet features for a maximum number of _t_ _tlr_ tolerant frames. During this tolerance,
these tracklets are considered inactive, but they can still contribute to output trajectories when their re-assignment
score exceeds _γ_ _reassign_ .


**Training Process.** We follow the same training setting as [ 67 ] with a batch size of 4, 40 epochs, and different
learning rates for the word embedding model, and the rest of the network, specifically, the learning rates are
0.00005, and 0.0001, respectively. We configure different max numbers for each type of token: 250 for text
queries, 500 for image queries, and 500 for tracklet queries. The training takes about 4 days for MOT17 and 7
days for MOT20 and TAO on 4 GPUs NVIDIA A100.


**Text Tokenizer.** _MENDER_ employs RoBERTa Tokenizer [ 65 ] to convert textual input into a sequence of text
tokens. This is done by dividing the text into a sequence of subword units using a pre-existing vocabulary. Each
subword is then mapped to a unique numerical token ID using a lookup table. The tokenizer adds special tokens

`[CLS]` and `[SEP]` to the beginning and end of the sequence, respectively. To encode the prompt for **def** and

**cap** settings, the `[CLS]` token is used to represent each sentence in the prompt list, as in Table 7 and Table 8.

For **nm** and **syn**, we join the words by `‘.` `’` and use the word features, following [36].


23


**12.2** **Negative Effects of the Long-tail Challenge on Tracking**


The imbalance in the TAO’s distribution has negative effects on the performance of tracking algorithms and the
evaluation of tracking metrics. Here are the negative effects of the long-tail problem on large-scale tracking
datasets:


**Inaccurate Classification.** Large-scale tracking datasets like TAO contain numerous rare and semantically
similar categories [ 79 ]. The classification performance for these categories is inaccurate due to the challenges
of imbalanced datasets and distinguishing fine-grained classes [ 51, 1 ]. The inaccurate classification results
in suboptimal tracking, where objects may be misclassified. This hinders the accurate evaluation of tracking
algorithms, as classification is a prerequisite for conducting association and evaluating tracking performance.


**Suboptimal Tracking.** Current MOT methods and metrics typically associate objects with the same class
predictions. In the case of large-scale datasets with inaccurate classification, this association strategy leads to
suboptimal tracking. Even if the tracker localizes and tracks the object perfectly, it still receives a low score if
the class prediction is wrong. As a result, the performance of trackers in tracking rare or semantically similar
classes becomes negligible, and the evaluation is dominated by the performance of dominant classes.


**Inadequate Benchmarking.** The prevalent strategies in MOT evaluation group tracking results based on class
labels and evaluate each class separately. However, in large-scale datasets with inaccurate classification, this
approach leads to inadequate benchmarking. Trackers that perform well in terms of localization and association
but have inaccurate class predictions may receive low scores, even though their tracking results are valuable. For
example, the trajectories of wrongly classified or unknown objects can still be useful for tasks such as collision
avoidance in autonomous vehicles [51].


Table 9 presents our findings which indicate that the performance of the Grounded MOT system is very poor on
the traditional benchmarking metrics (0.17% to 0.45% MOTA and -45.60% to -62.10% IDF1 on TAO). The
benchmarking metrics for this task should be designed to differentiate between the two tasks of classification
and tracking. By separating these tasks, the CA-MOTA and CA-IDF1 can help to provide a more accurate
assessment of tracking performance.


**13** **Qualitative Results**


Figure 9: Qualitative results using detailed prompts. Each box color represents a unique tracklet identity. (a)
**Green arrows** indicate true positive tracklets, while **red arrows** indicate false negative tracklets. (b) **Green**
**lines** indicate correct attended caption of each tracklet, while the **red line** indicate the incorrect attended caption.


Fig. 9 shows two qualitative results in the Grounded Multiple Object Tracking problem with detailed request
prompts. Fig. 9(a) is the **def** setting and Fig. 9(b) is the **cap** setting. See the supplementary video for more
qualitative results.


**Failed Cases.** Fig. 9 also shows some failed cases of our _MENDER_ . Fig. 9(a) indicates IDSwitch error by the
**red arrows** . We also map the result tracklets to their attended caption. Fig. 9(b) shows the incorrect attended
caption, which is highlighted by the **red line** .


24


