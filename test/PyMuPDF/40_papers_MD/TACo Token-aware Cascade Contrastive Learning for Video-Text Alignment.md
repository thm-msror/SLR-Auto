## **TACo: Token-aware Cascade Contrastive Learning for Video-Text Alignment**



Jianwei Yang
Microsoft Research


jianwyan@microsoft.com



Yonatan Bisk

Carnegie Mellon University


ybisk@cs.cmu.edu



Jianfeng Gao
Microsoft Research


jfgao@microsoft.com


𝐾 [′] hard negatives



**Abstract**


_Contrastive learning has been widely used to train_
_transformer-based vision-language models for video-text_
_alignment and multi-modal representation learning. This_
_paper presents a new algorithm called_ _**T**_ _oken-_ _**A**_ _ware_
_**C**_ _ascade c_ _**o**_ _ntrastive learning (_ _**TACo**_ _) that improves con-_
_trastive learning using two novel techniques. The first is the_
_token-aware contrastive loss which is computed by taking_
_into account the syntactic classes of words. This is mo-_
_tivated by the observation that for a video-text pair, the_
_content words in the text, such as nouns and verbs, are_
_more likely to be aligned with the visual contents in the_
_video than the function words. Second, a cascade sampling_
_method is applied to generate a small set of hard nega-_
_tive examples for efficient loss estimation for multi-modal_
_fusion layers._ _To validate the effectiveness of_ _**TACo**_ _, in_
_our experiments we finetune pretrained models for a set of_
_downstream tasks including text-video retrieval (YouCook2,_
_MSR-VTT and ActivityNet), video action step localization_
_(CrossTask), video action segmentation (COIN). The results_
_show that our models attain consistent improvements across_
_different experimental settings over previous methods, set-_
_ting new state-of-the-art on three public text-video retrieval_
_benchmarks of YouCook2, MSR-VTT and ActivityNet._


**1. Introduction**


Aligning or grounding language to videos is a challenging topic in the context of vision-language (VL) research
as it requires the model to understand contents, dynamics,
and causality presented in videos [3]. Inspired by the success of BERT [10] in natural language processing, there is a
growing interest in applying transformer-based multi-modal
models for video-text alignment and representation learning [41, 40, 60, 33, 14, 28]. These models are typically
pretrained on large amounts of noisy video-text pairs using
contrastive learning [35, 34], and then applied in a zeroshot manner or finetuned for various downstream tasks,
such as text-video retrieval [52], video action step localization [61], video action segmentation [43], video question



Figure 1: The proposed token-aware cascade contrastive
learning pipeline. We compute three contrastive losses: 1)
sentence-level loss _L_ 1 over all negative examples; 2) tokenlevel loss _L_ 2 on content words (noun, verb) over all negative examples; 3) sentence-level loss _L_ 3 over hard negative
examples sampled based on _L_ 1 and _L_ 2 online.


answering [44, 27] and video captioning [58].

In this paper, we present a new variant of contrastive learning, **T** oken- **A** ware **C** ascade c **o** ntrastive learning ( **TACo** ) to improve the video-text alignment for both
large-scale pretraining and downstream specific tasks. As
the name indicates, **TACo** makes two modifications to the
conventional contrastive learning used in video-language
domain. The first is the token-aware contrastive loss which
is computed by taking into account the syntactic classes of
words. This is motivated by the observation that, given
a video and its corresponding text, content words, such
as nouns and verbs, are more likely than function words
to be aligned with (or grounded to) visual contents in the
video. Conventional contrastive learning typically compute
the loss after aggregating over all the words in the text and
frames in the video (loss _L_ 1 or _L_ 3 in Fig. 1). In contrast,
the token-aware contrastive loss is computed using only a
subset of words whose syntactic classes belong to a predefined set ( _e.g._, nouns and verbs), which forces the grounding of individual words to the video (loss _L_ 2). For example,
we pay particular attention to the words “add”, “tomatos”,
“pan” and “stir” in Fig. 1.







…


𝐾−1 negatives



















Text-video pair


The second technique we introduce is a cascade sampling method to find a small set of hard negative examples for training the multi-modal fusion layers. Consider
a batch of _K_ video-text pairs. For each of the video-text
pairs, the ideal case is that we use the remaining _K −_ 1
negative videos or texts to compute the contrastive loss after multi-modal fusion. However, the cost of computing the
contrastive loss quickly becomes prohibitive when it is coupled with multi-modal fusion layers, considering its high
complexity _O_ ( _K_ [2] _× L_ [2] ) where _L_ is total number of visual
and textual tokens. A conventional way to address this is
using random sampling to select a small subset of negative
pairs. In this paper, instead of random sampling, we propose a cascade sampling method as shown in the top-right
of Fig. 1 to efficiently select a small set of hard negative
examples _on the fly_ during training. It leverages the videotext alignment scores computed in _L_ 1 and _L_ 2 before multimodal fusion layers, and helps to learn the multi-modal fusion layers more effectively without any extra overhead.
We perform a comprehensive empirical study to validate the effectiveness of **TACo** in both pretraining and
dataset-specific scenarios. We apply **TACo** and different
variants of contrastive losses to train or pretrain and finetune on various downstream tasks including text-video retrieval (YouCook2, MSR-VTT and ActivityNet) [58, 52,
12], video action step localization (CrossTask) [61] and action segmentation (COIN) [43]. Our results show that **TACo**
improves the text-video retrieval performance over current
state-of-the-art across three benchmarks. Furthermore, the
learned multi-modal representation and video representation can be effectively transferred to CrossTask and COIN,
and achieve better or comparable performance to current
state-of-the-art methods.


**2. Related work**


**Video-language pretraining** . Realistic application scenarios around videos have prompted emergence of various video-language tasks, such as text-video retrieval [30,
55, 53], video question answering [21, 27], video captioning [54, 59], _etc_ . Inspired by the success of BERT for largescale pretraining in language domain [10], transformers
have been employed in the video-language domain [41, 60,
33, 28] as well as image-language domain [42, 32, 57, 29].
Combined with large scale datasets, _e.g._ Howto100M [35]
this approach has proven to be effective on various downstream tasks. Depending on the tasks of interest, some approaches train a multi-modal transformer using a combination of multiple losses including video-text alignment [41,
60, 33, 28], masked token (words/frames/objects) prediction [41, 60, 33], and frame order prediction [28], _etc_ .
Some other approaches exploited various contrastive learning techniques to directly optimize the feature space without multi-modal fusion [35, 34, 31, 14]. In most of previ


ous works, these two approaches were explored separately.
Very recently, an updated version of [33] used two independent alignment losses before and after multi-modal fusion in
a single framework. In this paper, however, these two losses
cooperate closely with each other during training in that the
earlier stage helps to discover the hard negatives while the
multi-modal layers with more capacity help to tackle those
hard samples particularly.


**Video-text alignment** . Aligning videos to text requires the
model to understand motion and temporal coherence. Some
works have relied on attention mechanisms to extract key
information from videos [45, 55], while others preserve visual information by composing pairwise joint representation using 3D tensors [53] or use multi-level video encoders
to separately encode the spatial and temporal cues [11].
These models usually rely on a rank or margin loss to learn
the correct alignment for video-text pairs. Another line
of work learns fine-grained or hierarchical alignment between videos and texts [56, 49, 6]. In [49], the authors
proposed a fine-grained alignment by extracting the nouns
and verbs from action phrase in a sentence and projecting
them into a shared space with videos. Alternatively, the authors in [6] extract a hierarchical semantic graph and apply
graph reasoning to achieve the alignment at different levels. Similar ideas have been also proposed in the imagetext alignment by decomposing the images and texts into
sub-tokens [26, 50]. Thus far, it has not been studied how
these task-specific architectures can be integrated into largescale pretraining. In this paper, we are the first to propose
a simple yet effective token-aware contrastive loss for finegrained alignment for pretraining and downstream tasks.


**Negative sampling** . Key to efficient contrastive training
is a good source of negative examples. Most of current approaches use random sampling strategies for training videotext alignment [60, 33]. However, in the domain of imagetext retrieval, a few works tried hard negative sampling to
choose the hardest negatives for training. In [2, 13], the
authors computed the alignment scores for all image-text
pairs in a mini-batch and use the hardest negative sample
to compute the marginal loss. However, this strategy can
only be applied without multi-modal fusion. In those models which have multi-modal fusion layers for better representations [32, 8], the authors instead compute the matching score offline and then use it to sample hard negatives
for finetuning image-text retrieval model, which however is
difficult for large-scale pretraining due to the high computational cost. In this paper, our cascade hard negative mining is particularly designed to address these issues as we
efficiently select the hard negative samples online before
multi-modal fusion and send them to the fusion layers for
computing the loss. As we will show in our experiments,
this technique can be seamlessly applied to both large-scale
pretraining and downstream tasks.



2


**3. Method**


**3.1. Framework**


As depicted in Fig. 1, our model has three components:

**Video encoding module** _f_ _θ_ _v_ . It is implemented by a stack
of self-attention layers parameterized by _θ_ _v_ . Here, we assume the input video features have been already extracted
using some pre-trained models such as 2D CNN ( _e.g._,
ResNet [18]) or 3D CNN ( _e.g._, I3D [4], S3D [51]). Given
the input video embeddings, video encoder starts with a linear layer to project them to the same dimension _d_ as following self-attention layers. We denote the output of our
video encoder for a video clip by a sequence of _m_ features,
_**x**_ = _{x_ [1] _, ..., x_ _[m]_ _} ∈_ R _[m][×][d]_ . The number of features _m_ de- _̸_
pends on the choice of sampling frame rate and the video
feature extractor, which we will discuss in Sec. 4.

**Language encoding module** _f_ _θ_ _t_ . We use pretrained tokenizer [48] and BERT [10] to tokenize the input texts and
extract textual features, respectively. Given a raw sentence,
we append a “[CLS]” and “[SEP]” to the beginning and
end, respectively. At the top, we can obtain a sequence
of _n_ textual features _**y**_ = _{y_ [1] _, ..., y_ _[n]_ _} ∈R_ _[n][×][d]_ . We ensure the output feature dimension of video encoder to be
identical to that of language encoder. During training, we
update the parameters _θ_ _t_ in our language encoder to adapt
to the texts in specific domain, _e.g.,_ cooking instructions in
YouCook2 [58].

**Multi-modal fusion module** _f_ _θ_ _m_ . It also consists of selfattention layers with learnable parameters _θ_ _m_ . It takes video
features _**x**_ _∈_ R _[m][×][d]_ and text features _**y**_ _∈_ R _[n][×][d]_ from two
separate modalities as inputs and output the ( _m_ + _n_ ) features _**z**_ = _{z_ 1 _, ..., z_ ( _m_ + _n_ ) _} ∈R_ [(] _[m]_ [+] _[n]_ [)] _[×][d]_ . To help it to distinguish the video and language tokens, we use a token type
embedding layer to learn two embeddings and add them to
the visual and textual tokens, separately. Similar to original
Transformer [47], we include a positional embedding layer
to encode the absolute token positions in the input sequence.

The above three components comprise our video-text
alignment model which is then trained with the proposed
token-aware cascade contrastive loss. We start with a brief

review of conventional contrastive learning and then introduce the proposed technique.


**3.2. Contrastive learning: a revisit**


Given a set of _N_ video-text pairs _{_ ( _v_ _i_ _, t_ _i_ ) _}_ _[N]_ _i_ =1 [, our goal]
is to learn an optimal scoring function _s_ such that paired
video and text ( _v_ _i_ _, t_ _i_ ) have higher scores than all the other
unmatched pairs ( _v_ _j_ _, t_ _k_ ) _, j ̸_ = _k_ . From the probabilistic
perspective, aligning _v_ _i_ to _t_ _i_ is equivalent to maximizing
the conditional probability _p_ ( _v_ _i_ _|t_ _i_ ) while minimizing the
probability for all negative pairs _p_ ( _v_ _j_ _|t_ _i_ ) _, j ̸_ = _i_ . Accord


ing to [15, 37], _p_ ( _v_ _j_ _|t_ _i_ ) can be approximated by:


exp _[s]_ [(] _[v]_ _[j]_ _[,t]_ _[i]_ [)]
_p_ ( _v_ _j_ _|t_ _i_ ) _∼_ ~~�~~ _Nk_ =1 [exp] _[s]_ [(] _[v]_ _[k]_ _[,t]_ _[i]_ [)] (1)


where _s_ ( _v, t_ ) is the alignment score between _v_ and _t_ ; the
denominator is a sum over all possible videos, which is a
partition function for normalization. Adding cross-entropy
loss on _p_ ( _v_ _j_ _|t_ _i_ ), we can then derive the NCE loss [15]:


_̸_



_L_ _nce_ =


_∼_


_̸_



_N_
� _−_ log _p_ ( _v_ _i_ _|t_ _i_ )


_i_ =1


_̸_



_N_


_−_

� log


_i_ =1 _̸_



exp _[s]_ [(] _[v]_ _[i]_ _[,t]_ _[i]_ [)]
� exp _[s]_ [(] _[v]_ _[i]_ _[,t]_ _[i]_ [)] + ~~[�]~~ _k_ = _̸_ _i_



exp _[s]_ [(] _[v]_ _[i]_ _[,t]_ _[i]_ [)] + ~~[�]~~ _̸_



_k_ = _̸_ _i_ [exp] _[s]_ [(] _[v]_ _[k]_ _[,t]_ _[i]_ [)]



(2)

_̸_ �



_̸_


The denominator in Eq. 2 requires a sum over all videos
in a dataset, which is intractable in practice. Therefore, we
usually compute the NCE loss on a mini-batch of _K_ ( _K ≪_
_N_ ) video-text pairs sampled from the whole dataset. Ideally, we want to learn the parameters _θ_ = _{θ_ _v_ _, θ_ _t_ _, θ_ _m_ _}_
of the model to minimize the above NCE loss, such that
∆= _s_ ( _v_ _i_ _, t_ _i_ ) _−_ _s_ ( _v_ _j_ _, t_ _i_ ) is maximized over all tuples
( _t_ _i_ _, v_ _i_ _, v_ _j_ ) _, j ̸_ = _i_ . A number of previous works used the
above formula for contrastive learning [34, 60]. Meanwhile, there are some variants of computing contrastive loss
in video-langauge representation learning. For example,

[28, 14] omits the denominator and incorporate a margin
s.t. _s_ ( _v_ _i_ _, t_ _i_ ) _> s_ ( _v_ _j_ _, t_ _i_ ) + _δ, ∀j ̸_ = _i_ in a mini-batch. [33]
optimizes binary cross-entropy (BCE) by assigning ( _v_ _i_ _, t_ _i_ )
a positive label (1) and other pairs a negative label (0).


**3.3.** **TACo** : our approach


The way of using contrastive learning in previous works
has two issues. First, the loss is computed at sentencelevel by taking ‘[CLS]’ token [14] or the maximum over
all tokens [34] in a sentence. Clearly, the content words
( _e.g._, nouns, verbs) are more likely to align with the visual
contents or concepts in the videos compared with function
words ( _e.g._, stop words). Second, the high computational
cost in multi-modal fusion layers hinder the usage of large
batch of negative samples, which however is essential to
contrastive learning [34, 17, 7]. Motivated by these two issues, we introduce **TACo**, a simple yet effective method to
improve the contrastive learning. We elaborate below how
these contrastive losses are computed.
Given the _K_ video-text pairs _{_ ( _v_ _i_ _, t_ _i_ ) _}_ _[K]_ _i_ =1 [in a mini-]
batch, we first use our video encoder _f_ _θ_ _v_ and language encoder _f_ _θ_ _t_ to obtain a batch of video features
_X_ = _{x_ 1 _, ..., x_ _K_ _} ∈R_ _[K][×][m][×][d]_ and text features _Y_ =
_{y_ 1 _, ..., y_ _K_ _} ∈R_ _[K][×][n][×][d]_, respectively. Then, we average
all tokens of a video clip _v_ _i_ to get ¯ _x_ _i_ _∈R_ [1] _[×][d]_, and take the
first ‘[CLS]’ token for each text _t_ _i_ to get ¯ _y_ _i_ _∈R_ [1] _[×][d]_ . Based



_̸_


3


on ¯ _x_ and ¯ _y_, we compute the sentence-level contrastive loss:


_̸_


_̸_


_̸_



this as the summary of two modalities and then compute the
contrastive loss:


_̸_


_̸_


_̸_



_̸_ _L_ 3 = _−_


_̸_


_̸_



exp _[x]_ [¯] _[i]_ _[·][y]_ [¯] _[i]_ _[/τ]_ [1]
� exp _[x]_ [¯] _[i]_ _[·][y]_ [¯] _[i]_ _[/τ]_ [1] + ~~[�]~~ _j_ = _̸_ _i_


_̸_


_̸_



_j_ = _̸_ _i_ [exp] _[x]_ [¯] _[j]_ _[·][y]_ [¯] _[i]_ _[/τ]_ [1]


_̸_


_̸_



_L_ 1 = _−_


_̸_


_̸_


_̸_



_K_
� log


_i_ =1 _̸_


_̸_


_̸_



exp _[x]_ [¯] _[i]_ _[·][y]_ [¯] _[i]_ _[/τ]_ [1] + ~~[�]~~ _̸_


_̸_


_̸_



_̸_ 


_̸_


_̸_



_̸_  exp _[w][·][z]_ _i,i_ _[cls]_

 exp _[w][·][z]_ _i,i_ _[cls]_ + ~~[�]~~ _j_ = _̸_ _i_


_̸_



_̸_

exp _[w][·][z]_ _i,i_ _[cls]_ + ~~[�]~~ _̸_


_̸_



_̸_


_j,i_
_j_ = _̸_ _i_ [exp] _[w][·][z]_ _[cls]_


_̸_



_̸_ �


_̸_


_̸_



(3)


_̸_


_̸_


_̸_



_K_

_̸_ � log


_i_ =1 _̸_


_̸_



_̸_  (5)

_̸_ 


_̸_



_̸_


where _τ_ 1 is a scalar temperature parameter. In Eq. 3, the _̸_
computation is simply a number of dot-products between
video and text features. Giving such efficiency, we can use
all the _K −_ 1 negative samples in a mini-batch to compute
the loss. Through this, we optimize _θ_ _v_ and _θ_ _t_ so as to project
the video and text samples into an aligned feature space.
The ‘[CLS]’ token and average of video tokens in Eq. 3
overlooks the differences across tokens and frames, and thus
may not provide the pressure to push individual tokens ( _e.g._,
nouns and verbs) to ground on the specific video contents.
To encourage correct alignment, in addition to the sentencelevel loss, we introduce a token-level contrastive loss:


_̸_



_̸_


_̸_


� log

_p∈P_ _i_ _̸_



_̸_


_̸_





_̸_



_̸_


_̸_


exp _[s]_ [(] _[x]_ _[i]_ _[,y]_ _i_ _[p]_ [)] _[/τ]_ [2]


exp _[s]_ [(] _[x]_ _[i]_ _[,y]_ _i_ ~~_[p]_~~ [)] _[/τ]_ [2] + ~~[�]~~ exp

 _j_ = _̸_ _i_



_̸_


_̸_


~~[�]~~ exp _[s]_ [(] _[x]_ _[j]_ _[,y]_ _i_ ~~_[p]_~~ [)] _[/τ]_ [2]

_j_ = _̸_ _i_



_̸_


_̸_


_L_ 2 = _−_


_̸_



_̸_


_̸_


_K_
�


_i_ =1

_̸_



_̸_


_̸_


exp _[s]_ [(] _[x]_ _[i]_ _[,y]_ _i_ ~~_[p]_~~ [)] _[/τ]_ [2] + ~~[�]~~

_̸_



_̸_


_̸_






_̸_ 



_̸_


_̸_


_̸_

(4)
where _τ_ 2 is another scalar temperature parameter; _P_ _i_ is the
indices of tokens of interest in _i_ -th text, and _y_ _i_ _[p]_ [is the] _[ p]_ [-th]
token embedding in _i_ -th text. _s_ ( _·_ ) measures the similarity
between video features and specific token embedding _y_ _i_ _[p]_ [. It]
first computes the dot-product between _y_ _i_ _[p]_ _[∈R]_ [1] _[×][d]_ [ and all]
_m_ video tokens _x ∈R_ _[m][×][d]_, and then take the maximum
over _m_ scores to get the final alignment score. Through
Eq. 4, the model uses individual tokens as anchors to align
with video, which is complementary to the sentence-level
loss in Eq. 3. Similar to Eq. 3, we can compute this tokenlevel contrastive loss efficiently, and thus use all the _K −_ 1
negative samples. As a whole, these two losses are used to
optimize _θ_ _v_ and _θ_ _t_ in a token-aware manner.
**Token of interest** . In Eq. 4, we need to decide which tokens
should be included in _P_ _i_ . In this paper, we heuristically
select _nouns_ and _verbs_ as the targets considering they are
more “concrete” in the videos. In practice, _nouns_ or _verbs_
usually have different discriminativeness even if they are all
the same type. For example, “man” is a _noun_ but is less informative than “gymnast”. To reflect this, we further assign
different words with different weights by computing their
inverse document frequency (idf) [22]. A higher idf means
it is more unique across the corpus, and hence will weigh
more when computing the token-level contrastive loss. Another practical issue for computing the loss is that the tokens
are usually sub-words due to the BERT tokenizer. Hence,
for all tokens that belongs to the same word, we will assign
the same weights accordingly.
After computing the token-aware contrastive loss, we
feed the features from separate modalities to multi-modal
fusion layers to enable more interactions between them two.
Similar to previous work [60], we take the feature corresponding to the “[CLS]” in the ( _m_ + _n_ ) outputs. We regard



_̸_


_̸_


where _z_ _j,i_ _[cls]_ [is the multi-modal fusion output for “[CLS]” to-]
ken taking _x_ _j_ and _y_ _i_ as inputs; _w ∈R_ [1] _[×][d]_ is the parameter in a linear layer [1] . Based on Eq. 5, we optimize all parameters in our model _θ_ = _{θ_ _v_ _, θ_ _t_ _, θ_ _m_ _}_ in collaboration
with Eq. 3 and Eq. 4.
In Eq. 5, a practical challenge is that we can hardly use
all ( _K −_ 1) negative samples in the mini-batch, due to the
high computational and memory cost in the multi-modal fusion. The _O_ ( _d_ ( _m_ + _n_ ) [2] ) complexity of self-attention layer
makes it intractable to pass all _K × K_ pairs into the multimodal layers. Previous work solved this by performing random sampling to cut the number of negative samples to _K_ _[′]_ .
However, randomly choosing negative samples may result

_̸_ in sub-optimal learning since the pairs are scarce. We there
fore introduce a cascade sampling strategy to find hard negatives instead of random ones.

**Cascade hard negative sampling** . To reduce the computational cost in Eq. 5, we choose among all possible videotext pairs a small subset which are most difficult. However,
computing the alignment scores for all pairs using Eq. 5 and
then select the hard negatives is a “chicken-and-egg” problem. Instead, we propose to use the similarities between all
video-text pairs computed in Eq. 3 and Eq. 4 as the guidance. Specifically, for each text-video pair ( _v_ _j_ _, t_ _i_ ), we take
their global similarity ¯ _x_ _j_ _·_ ¯ _y_ _i_ computed in Eq. 3 and tokenlevel similarity by aggregating [�] _p∈P_ _i_ _[s]_ [(] _[x]_ _[j]_ _[, y]_ _i_ _[p]_ [)][ for all to-]
kens of interest in _t_ _i_ . Then we sum the two similarities as
the alignment score for the given pair. For each text, we
choose the top _K_ _[′]_ aligned negative videos and vice versa.
The resulting 2 _K ×_ ( _K_ _[′]_ + 1) pairs are then fed into the
multi-modal fusion layers. Through this strategy, we can effectively select the difficult negative samples on the fly at no
extra cost. Since the multi-modal fusion layers has more capacity (parameters) to distinguish these hard negatives from
positive ones, our sampling strategy naturally prompts the
cooperation between the three contrastive losses.
Finally, we present a comprehensive comparison to differentiate our model with previous works with respect to the
used contrastive learning method in Table 1.


**3.4. Objective**


The training objective in our method is finding optimal
_θ_ = _{θ_ _v_ _, θ_ _t_ _, θ_ _m_ _}_ by minimizing the combination of the
above three contrastive losses:



_̸_


_̸_


_̸_


arg min
_θ_ _v_ _,θ_ _t_ _,θ_ _m_



_̸_


_̸_


_̸_


_N_
� ( _L_ 1 + _λ_ _t_ _L_ 2 + _L_ 3 ) (6)


_i_ =1



_̸_


_̸_


_̸_


1 for clarity, we omit the bias term in the formula



_̸_


_̸_


_̸_


4


Method Token-aware Early stage Later stage Cascade Loss


VideoBert [41]     BCE
CBT [40]     NCE
TJVE [35]     Margin
MIL-NCE [34]     NCE
ActBert [60]     BCE
UniVL [33]     NCE
MMT [14]     Margin
**TACo** (Ours)     NCE


Table 1: A comparison of video-language pretraining methods regarding contrastive learning strategies. “Early stage”
and “Later stage” mean computing the loss before and after
multi-modal fusion, respectively. “Cascade” means using
cascade hard negative sampling.


where _λ_ _t_ is the weight of token-level loss (0.5 by default).
During inference, we make the prediction by summing the
alignment scores from all the three scoring functions.


**4. Experimental setup**


**4.1. Datasets**


In our experiments, we train and evaluate our model on
the following established benchmarks:

- **YouCook2** [58] consists of 2k videos about routine cooking activities of 89 recipes. Each video contains multiple
video clips annotated with text descriptions by human annotators. Following [35, 34], we train our models on the training split, and report the text-video retrieval performance on
around 3.5k validation clips.

- **MSR-VTT** [52] contains 10k video clips associated with
200k sentences. There are two validation splits used in previous work. In [31, 14], the training set has 9k clip-text pairs
with the remaining 1k pairs for evaluation, which we denote
by _split1_ . In [53, 35, 34], 1k clip-text pairs are sampled from
the 3k pairs in test set for evaluation, while the original 7k
pairs are used for training. We denote this by _split2_ . We
report text-video retrieval results using both splits.

- **ActivityNet** [25]. It consists of 20K YouTube videos, each
of which is associated with multiple human-annotated captions. Following [56, 14], we concatenate all the captions
for a video into a paragraph and evaluate the paragraphvideo retrieval on the “val1” split.

- **Howto100M** [35]. We compare with previous work under
the pretraining protocol on Howto100M [35, 34, 60, 33]. It
was collected from YouTube and contains over 1.2M nar
rated videos associated with automatically generated transcripts. Each video contains over 100 clips on average.

To further verify the transferrability or our learned multimodal representation from Howto100M, we also evaluate the action step localization and action segmentation on
CrossTask [61] and COIN [43], respectively.



**4.2. Settings**


Previous work use a variety of different video and language representations which we find significantly affect the
final performance. We summarize different choices below:

- **Video representations** . For 2D CNN, Resnet-152 [18]
is used to extract feature map and then globally pooled to
2048-d [35, 33]. For 3D features, commonly used models are I3D [5], R(2+1)D [46] and S3D [51]. In [60],
the authors further extract objects from the video clips.
In [31, 14], the authors use collaborative experts to extract
features from audio, scene, OCR, face, speech, etc.

- **Language representations** . There are primarily four variants: 1) GoogleNews pretrained word2vec (w2v) [36] used
in [31, 35, 34]; 2) LSTM or Bidirectional LSTM [19];
3) pretrained BERT [10] used in [41, 60, 33, 14] and 4)
OpenAI-GPT [38] used in [31].
In this paper, we use a pretrained BERT-base model for
language representation as in [60, 33]. For video features,
following [35, 34, 33], we extract 2D CNN features using
Resnet-152 (R-152) pretrained on ImageNet [9]. For 3D
CNN features, we use I3D (with Resnext-101 backbone)
pretrained on Kinetics-400 [23] and S3D [51] pretrained on
Howto100M [34]. The off-the-shelf pretrained weights are
provided by [16] and [34]. For simplicity, we denote them
by I3D-X101 and S3D-HM in the following.
Another discrepancy among different methods is the
number of self-attention layers used in the model. In [60],
the authors use 12 multi-modal self-attention layers while
6 video encoder layers and 2 multi-modal fusion layers are
used in [33]. Differently, 4 multi-modal self-attention layers
are used in [14]. In this paper, for all our ablation studies
below, we use 1 and 2 self-attention layers for our video
encoder and multi-modal fusion, respectively. To compare
with previous work on specific dataset, we use 2 video
encoding layers. While pretraining the model with largescale dataset Howto100M [35], we increase to 4 video encoding layers for comparable model capacity to previous
works [60, 33, 14]. Note that this largest model is still
smaller than or on par with the aforementioned methods.


**4.3. Implementation details**


For YouCook2 and MSR-VTT, the maximum number of
video and text tokens are set to 48 and 30, respectively. For
paragraph-video retrieval on ActivityNet, we set them both
to 256. The 2D R-152 feature is extracted for one frame

per second, and then globally pooled to 2048-d. For 3D
CNN features, we follow [35] to sample video frames at 24
fps and extract an I3D-X101 feature every 16 frames. This
results in 1.5 2048-d feature per second. For Eq. 3 and 4,
we set the temperatures _τ_ 1 and _τ_ 2 both equal to 1.
**Training on separate datasets** . In this setting, we train
models from scratch using the training set provided in
YouCook2, MSR-VTT and ActivityNet separately. We train



5


YouCook2 MSR-VTT ( _split1_ )


Video Representation R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_ R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_


R-152, Baseline 4.1 13.2 19.4 81.0 16.4 42.6 55.8 8.0
R-152, Ours **4.6 14.1 20.4 71.0** **18.9 46.2 58.8** **7.0**


I3D-X101, Baseline 2.1 8.1 12.7 125.0 14.7 40.83 53.2 9.0
I3D-X101, Ours **2.6 8.9 13.2 115.0 20.6 44.0 56.9** **7.0**


R-152+I3D-X101, Baseline 4.2 13.5 20.0 75.0 16.6 45.4 58.5 7.0
R-152+I3D-X101, Ours **4.7 14.3 21.9 68.0** **23.1 50.5 64.0** **5.0**


S3D-HM, Baseline 13.8 37.2 51.1 10.0 18.7 47.2 62.2 6.0
S3D-HM, Ours **16.1 40.3 52.2** **9.0** **23.9 51.4 65.0** **5.0**


R-152+S3D-HM, Baseline 13.3 35.8 48.9 11.0 21.4 48.1 61.5 6.0
R-152+S3D-HM, Ours **15.8 39.8 52.4 10.0** **24.5 52.8 65.5** **5.0**


Table 2: Text-video retrieval performance on YouCook2
and MSR-VTT with different feature types. S3D pretrained
on HowTo100M outperforms others with large margin.


the model for 30k iterations with batch size 128. For each

training sample, we use our cascade sampling strategy to
sample 8 hard negatives. We use Adam [24] as the optimizer with initial learning rate 1 _e_ _[−]_ [4] . A linear learning rate
decay is applied after 5k warm-up iterations. The weight
decay is set to 1 _e_ _[−]_ [5] .
**Pretraining and finetuning** . We pretrain our model on
Howto100M [35]. Since the original annotated video clips
in Howto100M are usually short with a few seconds, we
merge the adjacent clips so that the resulted text has at least
10 words. We use Adam [24] as the optimizer with initial learning rate 1 _e_ _[−]_ [4] . We train the model for 500k iterations with batch size 64, and also sample 8 hard negatives
for each sample using our cascade sampling strategy. After
pretraining, we finetune the pretrained models on different
datasets using the same setting as above except for a lower
initial learning rate 2 _e_ _[−]_ [5] and less finetuning iterations 20k.
**Evaluation metrics** . For text-video retrieval, we use Recalls at different points (Recall@n or Rn, with n as a specific number) and Median Rank (MR) as the metrics following previous works [60, 33]. In all tables, we use _↑_ or _↓_ to
indicate higher or lower is better, respectively.


**5. Results**


We first evaluate text-video retrieval performance and
then study whether the learned representations can be transferred to other tasks on CrossTask and COIN.


**5.1. Text-video retrieval**


**5.1.1** **Comparing with baselines**


We first show the comparisons with baselines to inspect the
effects of different components in our model.
**Video representations** . We train our model with different video representations as described above and compare it
with the baseline model which has identical architecture but

merely trained with _L_ 3 as depicted in Eq. 5. The baseline



YouCook2 MSR-VTT ( _split1_ )


Losses Cascade R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_ R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_


_L_ 1 n/a 14.1 35.7 48.8 11.0 22.9 49.7 61.7 6.0
_L_ 3 n/a 13.3 35.8 48.9 11.0 21.4 48.1 61.5 6.0
_L_ 1 + _L_ 3  13.9 37.4 50.7 10.0 22.5 50.8 64.1 5.0
_L_ 1 + _L_ 3  15.0 38.7 51.3 10.0 23.7 51.3 63.9 5.0
_L_ 1 + _L_ 2 + _L_ 3  **15.8 39.8 52.4 10.0** **24.5 52.8 65.5** 5.0


Table 3: Text-video retrieval performance with different
technique ensembles. It shows that using our proposed two
techniques produce best results. All experiments use R152+S3D-HM video features.


YouCook2 MSR-VTT ( _split1_ )


Token of Interest R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_ R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_


None 15.0 38.7 51.3 10.0 23.7 51.3 63.9 5.0

det+adp 14.7 38.5 51.2 10.0 23.3 51.0 63.5 5.0
noun 15.4 39.3 51.8 10.0 24.0 51.8 65.1 5.0

verb 15.3 39.0 51.4 10.0 23.9 52.1 64.8 5.0

noun+verb **15.8 39.8 52.4** 10.0 **24.5 52.8 65.5** 5.0


Table 4: Text-video retrieval performance with different tokens of interest for computing token-level contrastive loss.
“det” means determiner; “adp” means adposition. We use
the same video features as in Table 3.


contrastive learning method has been adopted in a number
of previous works [60, 33]. This comparison can verify the
effectiveness of our proposed contrastive learning method
considering two models have exactly the same number of
parameters. In Table 2, we can see our proposed method
outperforms baseline across all feature types introduced in
Sec. 4.2 on both YouCook2 and MSR-VTT. Note that our

model uses exactly the same number of parameters to the
baseline model. These consistent improvements demonstrate the effectiveness and generalization ability of our proposed method. As mentioned above, we also observe the
text-video retrieval performance significantly depends on
the feature types. We can find 3D features (I3D-X101 and
S3D-HM) in general outperform 2D feature (R-152), which
is expected since 2D feature does not capture the motions in
the videos. Among all three feature types, S3D-HM outperforms the other two with large margin, which demonstrates
the potential to learn good video representation by pretraining on large-scale noisy dataset (Howto100M [35]). Because Howto100M mainly contains instructional videos, it
is more close to YouCook2 than MSR-VTT, and hence we
see more gain on YouCook2. These comparisons indicate
video representations matter much to the final performance.
**Component Analysis** . In our method, we combine _L_ 1, _L_ 2,
and _L_ 3 during training and inference. Here, we study how
they perform separately and contribute to the final performance. In Table 2, we use R-152+S3D-HM as the video
feature and report the results with different loss combina


6


YouCook2
Model Lang. Video

R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_


Random – – 0.0 0.2 0.3 1675

TVJE [35] w2v R-152+I3D-X101 4.2 13.7 21.5 65
UniVL(v1) [33] BERT R-152+I3D-X101 3.4 10.8 17.8 76
**TACo** (Ours) BERT R-152+I3D-X101 **4.9** **14.7** **21.7** **63**


UniVL(v3) [33] BERT S3D-HM 7.7 23.9 34.7 21
**TACo** (Ours) BERT S3D-HM **16.6 40.3** **53.1** **9.0**


Table 5: Comparing text-video retrieval on YouCook2.


MSR-VTT
Model Lang. Video

R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_


Random – – 0.1 0.5 1.0 500.0

JSFusion [53] BiLSTM R-152 10.2 31.2 43.2 13.0
JPoSE [49] w2v TSN+Flow 14.3 38.1 53.0 9.0
TVJE [35] w2v R-152+I-101 12.1 35.0 48.0 12.0
UniVL(v1) _[∗]_ [33] BERT R-152+I-101 14.6 39.0 52.6 10.0
**TACo** (Ours) BERT R-152+I-101 **19.2 44.7** **57.2** **7.0**


CE [31] GPT Collaborative Experts 20.9 48.8 62.4 6.0
MMT [14] BERT Collaborative Experts 24.6 54.0 67.1 **4.0**
**TACo** (Ours) BERT R-152+S3D-HM **26.7 54.5** **68.2** **4.0**


Table 6: Comparing text-video retrieval on MSR-VTT. The
upper block and bottom block use _split2_ and _split1_, respectively. We report them separately for fair comparison.


ActivityNet
Model Lang. Video

R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_


Random - - 0.02 0.1 1.02 2458

DenseCap [25] LSTM C3D 14.0 32.0 65.0 34
FSE [56] GRU C3D+TSN-Inception 18.2 44.8 89.1 7.0
CE [31] GPT Collaborative Experts 18.2 47.7 91.4 6.0
MMT [14] BERT Collaborative Experts 22.7 54.2 93.2 5.0
**TACo** (Ours) BERT R-152+S3D-HM **25.8 56.3** **93.8** **4.0**


Table 7: Comparing text-video retrieval on ActivityNet.


tions. As we can see, solely using _L_ 1 (row 1) or _L_ 2 (row
2) for contrastive learning results in sub-optimal video-text
alignment. Simply combining them together (row 3) improves the performance on two datasets. This implies that
different levels of contrastive learning can be complementary to each other, which supports our earlier hypothesis that
these two losses are synergistic with each other for a better
video-text alignment. When incorporating the hard negative
mining via our cascade sampling strategy (row 4), it further improves the performance. Finally, we can see adding
token-level contrastive loss _L_ 3 can further improve the performance across all settings (row 5).
**Tokens of Interest** . We further study the effect of different
tokens of interest on the model performance. By default,
our model uses the noun and verb as the tokens of inter
est to compute the token-level contrast loss. Here, we vary
them to other types such as adposition (adp) and determiner
(det) for investigation. In Table 4, we replace “noun+verb”



Figure 2: Zero-shot performance on YouCook2 and MSRVTT for different settings. score-1-5 correspond to the five
settings in Table 3 from top to bottom.


with “det+adp”, “noun” and “verb” and report the numbers
on two text-video retrieval datasets. As we can see, using
“det+adp” as the target tokens is worse than the baseline
without any token-level contrastive loss. “noun” and “verb”
can both improve the performance while “noun” is slightly
better than “verb”. Finally, combining noun and verb together achieves the best performance. These results align
with our intuition to use nouns and verbs as the target token
for fine-grained alignment between texts and videos considering they are usually grounded to video contents.


**5.1.2** **Comparing with state-of-the-art**


We compare with previous works under three protocols: 1)
training and evaluating on separate datasets; 2) pretraining
on Howto100M and evaluating zero-shot performance and
3) finetuning pretrained model on separate datasets.
**Results on separate datasets** . We separately show the
comparisons on YouCook2, MSR-VTT and ActivityNet in
Table 5, 6 and 7. For a fair comparison with previous works,
we use the same or similar features as listed in the tables.

As we can see, our method outperforms all previous work
across all datasets. These results validates its effectiveness

to learn video-text alignment. Note that previous works either use a variety of loss functions [33, 28] or a collection of
multiple features [31, 14]. In contrast, we achieve the best
performance using a _simpler_ contrastive learning pipeline
with smaller model size. This supports our earlier claim
on the efficiency. Comparing the numbers in Table 2, Table 5 and Table 6, we can find our model achieves better performance with the same video features when using deeper
video encoder (2 layers _v.s._ 1 layer).



7


Model Video YouCook2 MSR-VTT ActivityNet

R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_ R1 _↑_ R5 _↑_ R10 _↑_ MR _↓_ R1 _↑_ R5 _↑_ R50 _↑_ MR _↓_


TJVE [35] R-152+I-101 6.1 17.3 24.8 46.0 7.5 21.2 29.6 38.0 – – – –
ActBERT [60] O-101+ R(2+1)D 9.6 26.7 38.0 19.0 8.6 23.4 33.1 36.0 – – – –
MIL-NCE [34] S3D-HM 15.1 38.0 51.2 10.0 **9.9** 24.0 32.4 29.5 – – – –
**TACo** (Ours) S3D-HM **19.9 43.2 55.7** **8.0** 9.8 **25.0 33.4** **29.0** – – – –


TJVE [35] R-152+I3D-X101 8.2 24.5 35.3 24.0 14.9 40.2 52.8 9.0 – – – –
UniVL(v3) [33] S3D-HM 28.9 57.6 70.0 **4.0** 21.2 49.6 63.1 6.0 – – – –
**TACo** (Ours) S3D-HM **29.6 59.7 72.7** **4.0** **24.8 52.1 64.0** **5.0** 28.3 56.8 92.6 4.0


MMT [14] Collaborative Experts – – – – 26.6 57.1 69.6 **4.0** 28.7 **61.4 94.5** 3.3
**TACo** (Ours) R-152+S3D-HM 27.3 56.5 68.8 **4.0** **28.4 57.8 71.2** **4.0** **30.4** 61.2 93.4 **3.0**


Table 8: A complete comparison of **TACo** under zero-shot and finetuning evaluation
protocols. Note that the zero-shot and upper part of finetuned performance for MSRVTT is on _split2_, while the bottom is on _split1_ for fair comparison.



Method CrossTask COIN


Alayrac _et al._ [1] 13.3 –
Zhukov _et al._ [61] 22.4 –
Supervised [61] 31.6 –
NN-Viterbi [39] – 21.2
CBT [40] – 53.9
TVJE [35] 33.6 –
MIL-NCE [34] 40.5 61.0
ActBert [60] 41.4 57.0
UniVL(v3) [33] 42.0 **70.0**


**TACo** (Ours) **42.5** 68.4


Table 9: Action step localization
on CrossTask (avg. recall) and action segmentation on COIN (acc.).



**Zero-shot and finetuned performance** . In Table 8, we
show the comparisons across different models pretrained on
Howto100M. In the upper part of the table, we compare the
zero-shot performance on YouCook2 and MSR-VTT. We do
not evaluate on ActivityNet since it has different number of
input video tokens compared with the pretrained model and
thus is not directly compatible to the pretrained model. As
we can see, **TACo** outperforms previous works significantly
on YouCook2 and slightly on MSR-VTT. Since YouCook2
has closer domain gap to Howto100M than MSR-VTT, the
improvement brought by large-scale pretraining is more significant. However, on MSR-VTT, our model still outperforms MIL-NCE [34] which uses the same video features.
In Fig. 2, we show the zero-shot performance on YouCook2
and MSR-VTT when pretraining our models with different contrastive losses as listed in Table 3. Accordingly, it
shows our proposed contrastive losses gradually improve
the performance, and combining all techniques achieves the
best performance. Based on the pretrained model, we further finetune it on specific datasets. In our experiments, we
use two feature S3D-HM and R-152+S3D-HM, to compare
with the methods with the same/similar settings. As we can
see, our model using S3D-HM outperforms UniVL [33]
using the same feature but more video encoder layers (6).
Different from zero-shot results, we observe more improvement on MSR-VTT than YouCook2 after finetuning. This
implies that finetuning on specific datasets can compensate
the domain gap to the pretraining datasets. To compare with
the methods using features extracted from collaborative experts [14], we enrich our video representation by adding
2D R-152 feature, which achieves better performance on
MSR-VTT, and better Recall@1 and Median Rank on ActivityNet. Note that this combination hurts the performance
on YouCook2, and we witnessed a similar trend for models
without pretraining in Table 2. Finally, comparing with the
results without pretraining in Table 5, 6 and 7, we clearly
find large-scale pretraining and finetuning brings substantial improvements consistently.



**5.2. Other video-related tasks**


Following [35, 60, 33], we evaluate action step localization performance on CrossTask dataset [61]. It covers 18
tasks and each video contains multiple video segments annotated with action steps and natural language descriptions.
Similar to [35, 60, 33], we use our model to compute the
similarity between each frame and the action step descriptions, which results in a score matrix. Using the official
algorithm provided by [61], we can find the optimal framewise order of action steps for a video. By comparing it
with the ground-truth annotations, we compute the recall for
each task and then do the average. According to the results
in Table 9, our model achieves the best performance compared with previous works. This indicates that our model
can learn good video-language representations.
We further evaluate our pretrained model on action segmentation task on COIN dataset, following [34, 60]. Unlike
the above task, action segmentation does not rely on texts,
and thus can be used to evaluate the learned video representation. As shown in Table 9, our method significantly
outperforms MIL-NCE and ActBert, and achieves comparable performance to UniVL. This indicates that our model
is also a good video representation learner.


**6. Conclusion**


In this paper, we introduced **TACo**, a simple yet effective contrastive learning method for learning video-text
alignment. It is aimed at addressing two existing issues
in current contrastive learning pipelines: missing finegrained alignment and inefficient sampling for multi-modal
fusion. Without introducing any extra parameters, our
method achieved promising results on three text-video retrieval benchmarks under various evaluation protocols. We
further demonstrated the learned representations can be effectively transferred to other tasks such as action step localization and segmentation. Based on all these encouraging
results, we believe **TACo** is a good alternative to conventional contrastive learning pipeline.



8


**References**


[1] Jean-Baptiste Alayrac, Piotr Bojanowski, Nishant Agrawal,
Josef Sivic, Ivan Laptev, and Simon Lacoste-Julien. Unsupervised learning from narrated instruction videos. In _Pro-_
_ceedings of the IEEE Conference on Computer Vision and_
_Pattern Recognition_, pages 4575–4583, 2016. 8

[2] Srikar Appalaraju and Vineet Chaoji. Image similarity
using deep cnn and curriculum learning. _arXiv preprint_
_arXiv:1709.08761_, 2017. 2

[3] Yonatan Bisk, Ari Holtzman, Jesse Thomason, Jacob Andreas, Yoshua Bengio, Joyce Chai, Mirella Lapata, Angeliki Lazaridou, Jonathan May, Aleksandr Nisnevich, Nicolas
Pinto, and Joseph Turian. Experience grounds language. In
_Proceedings of the 2020 Conference on Empirical Methods_
_in Natural Language Processing (EMNLP)_, 2020. 1

[4] Joao Carreira and Andrew Zisserman. Quo Vadis, Action
Recognition? A New Model and the Kinetics Dataset. In
_IEEE Conference on Computer Vision and Pattern Recogni-_
_tion (CVPR)_, 05 2017. 3

[5] Joao Carreira and Andrew Zisserman. Quo vadis, action
recognition? a new model and the kinetics dataset. In _pro-_
_ceedings of the IEEE Conference on Computer Vision and_
_Pattern Recognition_, pages 6299–6308, 2017. 5

[6] Shizhe Chen, Yida Zhao, Qin Jin, and Qi Wu. Fine-grained
video-text retrieval with hierarchical graph reasoning. In
_Proceedings of the IEEE/CVF Conference on Computer Vi-_
_sion and Pattern Recognition_, pages 10638–10647, 2020. 2

[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning
of visual representations. _arXiv preprint arXiv:2002.05709_,
2020. 3

[8] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy,
Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter:
Universal image-text representation learning. In _European_
_Conference on Computer Vision_, pages 104–120. Springer,
2020. 2

[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In _2009 IEEE conference on computer vision and_
_pattern recognition_, pages 248–255. Ieee, 2009. 5

[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In _Proceedings of the_
_2019 Conference of the North American Chapter of the As-_
_sociation for Computational Linguistics: Human Language_
_Technologies, Volume 1 (Long and Short Papers)_, 10 2019.
1, 2, 3, 5

[11] Jianfeng Dong, Xirong Li, Chaoxi Xu, Shouling Ji, Yuan
He, Gang Yang, and Xun Wang. Dual encoding for zeroexample video retrieval. In _Proceedings of the IEEE Con-_
_ference on Computer Vision and Pattern Recognition_, pages
9346–9355, 2019. 2

[12] Bernard Ghanem Fabian Caba Heilbron, Victor Escorcia and
Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In _Proceedings_
_of the IEEE Conference on Computer Vision and Pattern_
_Recognition_, pages 961–970, 2015. 2




[13] Fartash Faghri, David J Fleet, Jamie Ryan Kiros, and Sanja
Fidler. Vse++: Improving visual-semantic embeddings with
hard negatives. _arXiv preprint arXiv:1707.05612_, 2017. 2

[14] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia

Schmid. Multi-modal transformer for video retrieval. In _Eu-_

_ropean Conference on Computer Vision (ECCV)_, volume 5.
Springer, 2020. 1, 2, 3, 5, 7, 8, 12

[15] Michael Gutmann and Aapo Hyv¨arinen. Noise-contrastive
estimation: A new estimation principle for unnormalized
statistical models. In _Proceedings of the Thirteenth Inter-_
_national Conference on Artificial Intelligence and Statistics_,
pages 297–304, 2010. 3

[16] Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. Can
spatiotemporal 3d cnns retrace the history of 2d cnns and
imagenet? In _Proceedings of the IEEE Conference on Com-_
_puter Vision and Pattern Recognition (CVPR)_, pages 6546–
6555, 2018. 5

[17] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross
Girshick. Momentum contrast for unsupervised visual representation learning. In _Proceedings of the IEEE/CVF Con-_
_ference on Computer Vision and Pattern Recognition_, pages
9729–9738, 2020. 3

[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. 12 2015. 3, 5

[19] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term
memory. _Neural computation_, 9(8):1735–1780, 1997. 5

[20] Matthew Honnibal and Ines Montani. spaCy 2: Natural language understanding with Bloom embeddings, convolutional
neural networks and incremental parsing. To appear, 2017.
11

[21] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and
Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in
visual question answering. In _Proceedings of the IEEE Con-_
_ference on Computer Vision and Pattern Recognition_, pages
2758–2766, 2017. 2

[22] Karen Sp¨arck Jones. A statistical interpretation of term
specificity and its application in retrieval. _Journal of Doc-_
_umentation_, 28:11–21, 1972. 4

[23] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang,
Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola,
Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman,
and Andrew Zisserman. The Kinetics Human Action Video

Dataset. _arXiv:1705.06950_, 05 2017. 5

[24] Diederik P Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. _arXiv preprint arXiv:1412.6980_,
2014. 6

[25] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and
Juan Carlos Niebles. Dense-captioning events in videos. In
_Proceedings of the IEEE international conference on com-_
_puter vision_, pages 706–715, 2017. 5, 7, 11

[26] Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, and Xiaodong He. Stacked cross attention for image-text matching.
In _Proceedings of the European Conference on Computer Vi-_
_sion (ECCV)_, pages 201–216, 2018. 2

[27] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg.
Tvqa: Localized, compositional video question answering.
_arXiv preprint arXiv:1809.01696_, 2018. 1, 2



9


[28] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu,
and Jingjing Liu. Hero: Hierarchical encoder for video+
language omni-representation pre-training. _arXiv preprint_
_arXiv:2005.00200_, 2020. 1, 2, 3, 7

[29] Xiujun Li, Xi Yin, Chunyuan Li, Xiaowei Hu, Pengchuan
Zhang, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong,
Furu Wei, Yejin Choi, and Jianfeng Gao. Oscar: Objectsemantics aligned pre-training for vision-language tasks.
_arXiv preprint arXiv:2004.06165_, 2020. 2

[30] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
_European conference on computer vision_, pages 740–755.
Springer, 2014. 2

[31] Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew
Zisserman. Use what you have: Video retrieval using
representations from collaborative experts. _arXiv preprint_
_arXiv:1907.13487_, 2019. 2, 5, 7

[32] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks. In _11 pages, 5 fig-_
_ures_, 08 2019. 2

[33] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan
Duan, Tianrui Li, Xilin Chen, and Ming Zhou. Univilm:
A unified video and language pre-training model for multimodal understanding and generation. _arXiv:2002.06353_,
2020. 1, 2, 3, 5, 6, 7, 8, 12

[34] A. Miech, J. B. Alayrac, L. Smaira, I. Laptev, J. Sivic, and
A. Zisserman. End-to-end learning of visual representations
from uncurated instructional videos. In _2020 IEEE/CVF_

_Conference on Computer Vision and Pattern Recognition_
_(CVPR)_, pages 9876–9886, June 2020. 1, 2, 3, 5, 8

[35] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac,
Makarand Tapaswi, Ivan Laptev, and Josef Sivic.
HowTo100M: Learning a Text-Video Embedding by
Watching Hundred Million Narrated Video Clips. In _ICCV_,
06 2019. 1, 2, 5, 6, 7, 8

[36] Tomas Mikolov, Kai Chen, Greg S. Corrado, and Jeffrey
Dean. Efficient estimation of word representations in vector
space. In _International Conference on Learning Representa-_
_tions_, 2013. 5

[37] Andriy Mnih and Yee Whye Teh. A fast and simple algorithm for training neural probabilistic language models.
_arXiv preprint arXiv:1206.6426_, 2012. 3

[38] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya
Sutskever. Improving language understanding by generative
pre-training, 2018. 5

[39] Alexander Richard, Hilde Kuehne, Ahsan Iqbal, and Juergen Gall. Neuralnetwork-viterbi: A framework for weakly
supervised video learning. In _Proceedings of the IEEE Con-_
_ference on Computer Vision and Pattern Recognition_, pages
7386–7395, 2018. 8

[40] Chen Sun, Fabien Baradel, Kevin Murphy, and Cordelia
Schmid. Learning video representations using contrastive
bidirectional transformer. _arXiv preprint arXiv:1906.05743_,
2019. 1, 5, 8




[41] C. Sun, A. Myers, C. Vondrick, K. Murphy, and C. Schmid.
Videobert: A joint model for video and language representation learning. In _2019 IEEE/CVF International Conference_
_on Computer Vision (ICCV)_, pages 7463–7472, 2019. 1, 2, 5

[42] Hao Tan and Mohit Bansal. LXMERT: Learning CrossModality Encoder Representations from Transformers. In
_EMNLP_, 08 2019. 2

[43] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng,
Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin:
A large-scale dataset for comprehensive instructional video
analysis. In _Proceedings of the IEEE Conference on Com-_
_puter Vision and Pattern Recognition_, pages 1207–1216,
2019. 1, 2, 5

[44] Makarand Tapaswi, Yukun Zhu, Rainer Stiefelhagen,
Antonio Torralba, Raquel Urtasun, and Sanja Fidler.
Movieqa: Understanding stories in movies through questionanswering. In _Proceedings of the IEEE conference on_
_computer vision and pattern recognition_, pages 4631–4640,
2016. 1

[45] Atousa Torabi, Niket Tandon, and Leonid Sigal. Learning
language-visual embedding for movie understanding with
natural-language. _arXiv preprint arXiv:1609.08124_, 2016.
2

[46] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann
LeCun, and Manohar Paluri. A closer look at spatiotemporal
convolutions for action recognition. In _Proceedings of the_
_IEEE conference on Computer Vision and Pattern Recogni-_
_tion_, pages 6450–6459, 2018. 5

[47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia
Polosukhin. Attention is all you need. 06 2017. 3

[48] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim
Rault, R´emi Louf, Morgan Funtowicz, Joe Davison, Sam
Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien
Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama
Drame, Quentin Lhoest, and Alexander M. Rush. Huggingface’s transformers: State-of-the-art natural language processing. _ArXiv_, abs/1910.03771, 2019. 3

[49] Michael Wray, Diane Larlus, Gabriela Csurka, and Dima
Damen. Fine-grained action retrieval through multiple partsof-speech embeddings. In _Proceedings of the IEEE Inter-_
_national Conference on Computer Vision_, pages 450–459,
2019. 2, 7

[50] Hao Wu, Jiayuan Mao, Yufeng Zhang, Yuning Jiang, Lei
Li, Weiwei Sun, and Wei-Ying Ma. Unified visual-semantic
embeddings: Bridging vision and language with structured
meaning representations. In _Proceedings of the IEEE Con-_
_ference on Computer Vision and Pattern Recognition_, pages
6609–6618, 2019. 2

[51] Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and
Kevin Murphy. Rethinking spatiotemporal feature learning:
Speed-accuracy trade-offs in video classification. In _Pro-_
_ceedings of the European Conference on Computer Vision_
_(ECCV)_, pages 305–321, 2018. 3, 5

[52] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large
video description dataset for bridging video and language.



10


In _2016 IEEE Conference on Computer Vision and Pattern_
_Recognition (CVPR)_, pages 5288–5296. IEEE, 2016. 1, 2, 5,
11

[53] Youngjae Yu, Jongseok Kim, and Gunhee Kim. A joint sequence fusion model for video question answering and retrieval. In _Proceedings of the European Conference on Com-_
_puter Vision (ECCV)_, pages 471–487, 2018. 2, 5, 7

[54] Youngjae Yu, Hyungjin Ko, Jongwook Choi, and Gunhee
Kim. Video captioning and retrieval models with semantic
attention. _arXiv preprint arXiv:1610.02947_, 6(7), 2016. 2

[55] Youngjae Yu, Hyungjin Ko, Jongwook Choi, and Gunhee
Kim. End-to-end concept word detection for video captioning, retrieval, and question answering. In _Proceedings of the_
_IEEE Conference on Computer Vision and Pattern Recogni-_
_tion_, pages 3165–3173, 2017. 2

[56] Bowen Zhang, Hexiang Hu, and Fei Sha. Cross-modal and
hierarchical modeling of video and text. In _Proceedings_
_of the European Conference on Computer Vision (ECCV)_,
pages 374–390, 2018. 2, 5, 7

[57] Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J. Corso, and Jianfeng Gao. Unified vision-language pretraining for image captioning and vqa. In _Thirty-Fourth AAAI_
_Conference on Artificial Intelligence_, 2019. 2

[58] Luowei Zhou, Chenliang Xu, and Jason J Corso. Towards
automatic learning of procedures from web instructional
videos. In _AAAI 2018_, 2017. 1, 2, 3, 5, 11

[59] Luowei Zhou, Yingbo Zhou, Jason J Corso, Richard Socher,
and Caiming Xiong. End-to-end dense video captioning with
masked transformer. In _Proceedings of the IEEE Conference_
_on Computer Vision and Pattern Recognition_, pages 8739–
8748, 2018. 2

[60] Linchao Zhu and Yi Yang. Actbert: Learning global-local
video-text representations. In _CVPR_, 2020. 1, 2, 3, 4, 5, 6,
8, 12

[61] Dimitri Zhukov, Jean-Baptiste Alayrac, Ramazan Gokberk
Cinbis, David Fouhey, Ivan Laptev, and Josef Sivic. Crosstask weakly supervised learning from instructional videos.
In _Proceedings of the IEEE Conference on Computer Vision_
_and Pattern Recognition_, pages 3537–3545, 2019. 1, 2, 5, 8


**A. Tokens of interest**


Dataset Noun Verb All


YouCook2 [58] 378 168 2,144
MSR-VTT [52] 4,415 1,463 15,740
ActivityNet [25] 2,602 1,021 9,059

Table 10: Token statistics for each dataset.


We extract tokens of interest (T.O.I) using the pos-tagger
provided by Spacy [20]. In Table 10, we show the statistics
of tokens for three datasets. For each token that is tagged
at _VERB_ or _NOUN_, we compute the inverse document frequency (idf) by:


_|D|_
_idf_ ( _token_ ) = log (7)
1 + _|{d ∈_ _D_ : _token ∈_ _d}|_



where _D_ is the full set of corpus, which are the captions
in the training set for a dataset; the denominator counts the
number of captions which contain a specific token. Based
on Eq. 7, we can compute the idf for each token of interest. The smaller the idf, the more frequent it appears in the
corpus. We do not compute the tf term since usually a token only appears once in a single sentence. The full list of
tokens and corresponding idfs can be found in Fig. 4. For
a given sentence, we first assign the computed idfs to its
nouns and verbs and then normalize the idfs, which are then
used to weigh the token-level contrastive losses.


**B. Contribution of three contrastive losses**


Loss R@1 R@5 R@10 MR


Early stage only 14.1 35.7 48.8 11.0
Later stage only 13.3 35.8 48.9 11.0


Early stage 15.3 39.3 51.9 **10.0**
Token-level 15.0 39.5 51.4 11.0

Later stage 14.3 38.4 50.6 11.0
Fused **15.8** **39.8** **52.4** **10.0**

Table 11: Text-video retrieval performance using separate
alignment scores on YouCook2.


In this part, we investigate the contributions of three contrastive losses used in our model. After we train the video
text alignment model using all three losses, we report the
performance using separate alignment scores in Table 11.
For reference, the top two rows are the performance for using early stage only and later stage only contrastive learning
to train the model. The bottom four rows are the separate
performance at different stages for our model. As we can
see, combining three contrastive losses during training can
boost the performance for both early and later stage (row
3 _v.s._ row 1, row 5 _v.s._ row 2). This indicates that the
three losses are synergistic to each other for a better videotext alignment. On the other hand, the early stage alignment
achieves better performance than other two (token-level and
later stage), while the fused score is the best. We suspect
that this is because early stage alignment is trained with all
text-video pairs at sentence-level. In contrast, token-level
contrast focuses on single tokens and the multi-modal fusion layers merely see a small part of hard text-video pairs.


**C. Effect of cascade sampling**

The proposed cascade sampling helps the later stage contrastive learning to focus on hard negative samples. As
shown in our main submission, adding cascade sampling
will improve the performance. We suspect this is because
cascade sampling helps learn a better later stage alignment.
To verify this, we compare the later stage alignment across
three different settings: 1) merely applying later stage contrastive loss; 2) combine early state and later stage con


11


trastive losses and 3) using cascade sampling for later stage
contrastive loss. We report the results on YouCook2 in Table 12. Here, note that we only use the later stage alignment
scores for evaluating the performance. As we can see, combining early stage and later stage together slightly improves
the performance. This is probably because early stage contrastive loss helps to learn a better video and language encoder, from which the multi-modal module takes better representations for cross-modal fusion. After applying the cascade sampling for the later stage contrastive loss, the performance is further improved. Since our cascade sampling
strategy can send more difficult samples to the later stage,
the cross-modal fusion layers can learn more discriminative representations for video-text alignment. These results validate that the hard negative mining through cascade
sampling indeed helps to improve the later-stage text-video
alignment, and hence the final performance.


Setting R@1 R@5 R@10 MR


Later stage only 13.3 35.7 48.8 11.0
Early stage + Later stage 13.6 35.9 49.1 11.0
Cascade sampling **14.5** **38.3** **50.7** 11.0

Table 12: Text-video retrieval performance on YouCook2
only using later stage alignment score for different settings.


**D. Effect of video encoder layers**


In our main paper, we noticed the number of video encoder layers affects the final performance. To have a more
comprehensive study, we use R-152 and S3D-HM as the 2D
and 3D features and train the video-text alignment model on
YouCook2 with different video encoder layers. As shown
in Table 13, using more video encoder layers can significantly boost the text-video retrieval performance. Particularly, when no video encoder layers are used, the model can
hardy capture the long-range temporal dynamics, and thus
performs poorly. Once we add one video encoder layer, the
performance improves significantly. With the increase of
encoder layers, the performance is further improved, which
is reasonable since more video encoder layers can encode
more complicated video contents and dynamics.


#video #params. FLOPs YouCook2
enc. layers (M) (G) R@1 R@5 R@10 MR


0 126.5 3.86 14.0 35.7 49.5 11.0

1 133.6 4.11 15.8 39.8 52.4 10.0

2 140.7 4.45 15.9 **40.5** 53.8 **9.0**

4 154.9 5.14 **16.4 40.5** **54.3** **9.0**

Table 13: Text-video retrieval performance on YouCook2
with different video encoder layers using R-152+S3D-HM.



**E. Comparing model size and FLOPs**

Finally, we attempt to compare the model sizes and computational costs for different methods. Unfortunately, all
previous methods did not report FLOPs and only MMT [14]
discussed #params. However, the results in Table 13 imply
that bigger model can usually achieve better performance.
Therefore, it is necessary to have a comparison of model
size and computational cost between our model and those
from other methods. For other methods which do not report
the numbers, we estimate them based on the descriptions in
the original paper. Table 14 summarizes the comparisons
and also reports the #params and FLOPs (all underlined
numbers are estimated based on the descriptions in original papers). As shown, our largest model has comparable
size and FLOPs to others.


mm #params. FLOPs
method text video
(M) (G)
self cross


ActBert [60] 12 12 0 24 369.1 13.80
MMT [14] 12 4 0 0 133.3 4.63
UniVL [33] 12 6 2 0 169.0 5.82
Our largest 12 4 2 0 154.9 5.14

Table 14: Comparison of model size and FLOPs. “mm”
means multi-modal fusion, and “self” means self-attention
layers while “cross” means cross-modal attention.


**F. Visualizations**

We visualize the text-video retrieval results by varying
the weights for the token-level alignment scores during testing. In Fig. 3, we show two text-video retrieval examples
on YouCook (top) and MSR-VTT (bottom). From top to
bottom, the five rows in each block correspond to the top
five retrieved results from the whole test set. As we can
see, when we gradually increase the weight for the tokenlevel alignment score, there are more related videos appearing in the top five candidates. For YouCook2, when we set
the weight equal to 0.0, the third and fifth video are not
well-aligned with the query since they are both not about
“tomato”. When we increase the weight to 0.1, we can observe the the fourth video moves to the third place. After we
increase the weight to 0.5, we can see all top-5 videos are
about cutting tomato. Similarly, for MSR-VTT, we can see
the last three videos are not about “two people talking on a
table”. When we increase the weight to 0.1, the fifth video is
replaced with a more matched video. Keeping increase the
weight to 0.5, we can obtain the top 5 videos all about “two
people talking with each other on a table”. These visualizations demonstrate the efficacy of our proposed token-level
contrastive learning.



12


**Query** : cut the tomato and put it inside a bowl



cut a tomato into quarters remove the seeds chop finely and add to the bowl


chop a tomato into thin slices


chop some red onions red pepper and green pepper into square pieces


chop up the tomatoes



cut a tomato into quarters remove the seeds chop finely and add to the bowl cut a tomato into quarters remove the seeds chop finely and add to the bowl



chop a tomato into thin slices


chop up the tomatoes


chop some red onions red pepper and green pepper into square pieces



chop a tomato into thin slices


chop up the tomatoes


slice tomatoes into thin slices


cut tomatoes and place them in a bowl



cut the pepperoni in half cut the pepperoni in half



Weight=0.0 Weight=0.1 Weight=0.5


**Query** : two people are talking with each other on a table



a man and a woman trying some sake


there is a woman is talking with two guys


leonardo dicaprio is portrayed as two different characters in this film


a girl in a studio singing


a cartoon man plays a card game with his friends



a man and a woman trying some sake


there is a woman is talking with two guys


leonardo dicaprio is portrayed as two different characters in this film


a girl in a studio singing


two men talking about investors on a show



there is a woman is talking with two guys


a man and another man speak to each other in a room


a man and a woman trying some sake


two men talking about investors on a show


a man and woman arguing about fake arms used in a performance



Weight=0.0 Weight=0.1 Weight=0.5


Figure 3: Text-video retrieval results given a query on YouCook2 (top) and MSR-VTT (bottom). In each block, we show top
5 ranked videos from top to bottom. From left to right, we gradually increase the token-level alignment weight from 0.0 to 0.1
and then 0.5 (default in our main experiments). The change of the top 5 results demonstrate the benefit of token-level contrast
when performing text-video retrieval. Below each video (depicted by three side-by-side frames), we show the associated
descriptions provided in the original dataset. Better viewed by enlarging the figure.


13


Figure 4: Token inverse document frequency (IDF) for noun and verb in YouCook2 and MSR-VTT. For clarity, we evenly
sample the tokens and show their IDFs. From left to right, the noun/verb becomes more and more frequent gradually.


14


