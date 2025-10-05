## **End-to-End Audio Visual Scene-Aware Dialog using** **Multimodal Attention-Based Video Features**

**Chiori Hori** _[†]_ **, Huda Alamri** _[∗†]_ **, Jue Wang** _[†]_ **, Gordon Wichern** _[†]_ **,**
**Takaaki Hori** _[†]_ **, Anoop Cherian** _[†]_ **, Tim K. Marks** _[†]_ **,**
**Vincent Cartillier** _[∗]_ **, Raphael Gontijo Lopes** _[∗]_ **, Abhishek Das** _[∗]_ **,**
**Irfan Essa** _[∗]_ **, Dhruv Batra** _[∗]_ **Devi Parikh** _[∗]_ **,**


_†_ Mitsubishi Electric Research Laboratories (MERL), Cambridge, MA, USA
_∗_ School of Interactive Computing, Georgia Tech


**Abstract**


Dialog systems need to understand dynamic visual scenes in order to have conversations with users about the objects and events around them. Scene-aware dialog
systems for real-world applications could be developed by integrating state-ofthe-art technologies from multiple research areas, including: end-to-end dialog
technologies, which generate system responses using models trained from dialog
data; visual question answering (VQA) technologies, which answer questions about
images using learned image features; and video description technologies, in which
descriptions/captions are generated from videos using multimodal information. We
introduce a new dataset of dialogs about videos of human behaviors. Each dialog
is a typed conversation that consists of a sequence of 10 question-and-answer
(QA) pairs between two Amazon Mechanical Turk (AMT) workers. In total, we
collected dialogs on _∼_ 9 _,_ 000 videos. Using this new dataset, we trained an end-toend conversation model that generates responses in a dialog about a video. Our
experiments demonstrate that using multimodal features that were developed for
multimodal attention-based video description enhances the quality of generated
dialog about dynamic scenes (videos). Our dataset, model code and pretrained
models will be publicly available for a new Video Scene-Aware Dialog challenge.


**1** **Introduction**


Spoken dialog technologies have been applied in real-world human-machine interfaces including
smart phone digital assistants, car navigation systems, voice-controlled smart speakers, and humanfacing robots [ 1, 2, 3 ]. Generally, a dialog system consists of a pipeline of data processing modules,
including automatic speech recognition, spoken language understanding, dialog management, sentence generation, and speech synthesis. However, all of these modules require significant hand
engineering and domain knowledge for training. Recently, end-to-end dialog systems have been
gathering attention, and they obviate this need for expensive hand engineering to some extent. In endto-end approaches, dialog models are trained using only paired input and output sentences, without
relying on pre-designed data processing modules or intermediate internal data representations such as
concept tags and slot-value pairs. End-to-end systems can be trained to directly map from a user’s
utterance to a system response sentence and/or action. This significantly reduces the data preparation
and system development cost. Several types of sequence-to-sequence models have been applied to
end-to-end dialog systems, and it has been shown that they can be trained in a completely data-driven
manner. End-to-end approaches have also been shown to better handle flexible conversations between
the user and the system by training the model on large conversational datasets [4, 5].


Preprint. Work in progress.


In these applications, however, all conversation is triggered by user speech input, and the contents of
system responses are limited by the training data (a set of dialogs). Current dialog systems cannot
understand dynamic scenes using multimodal sensor-based input


such as vision and non-speech audio, so machines using such dialog systems cannot have a conversation about what’s going on in their surroundings. To develop machines that can carry on a conversation
about objects and events taking place around the machines or the users, dynamic scene-aware dialog
technology is essential.


To interact with humans about visual information, systems need to understand both visual scenes
and natural language inputs. One naive approach could be a pipeline system in which the output of a
visual description system is used as an input to a dialog system. In this cascaded approach, semantic
frames such as "who" is doing "what" and "where" must be extracted from the video description
results. The prediction of frame type and the value of the frame must be trained using annotated data.
In contrast, the recent revolution of neural network models allows us to combine different modules
into a single end-to-end differentiable network. We can simultaneously input video features and user
utterances into an encoder-decoder-based system whose outputs are natural-language responses.


Using this end-to-end framework, _visual question answering_ (VQA) has been intensively researched
in the field of computer vision [ 6, 7, 8 ]. The goal of VQA is to generate answers to questions about
an imaged scene, using the information present in a single static image. As a further step towards
conversational visual AI, the new task of _visual dialog_ was introduced [ 9 ], in which an AI agent holds
a meaningful dialog with humans about an image using natural, conversational language [ 10 ]. While
VQA and visual dialog take significant steps towards human-machine interaction, they only consider
a single static image. To capture the semantics of dynamic scenes, recent research has focused on
_video description_ (natural-language descriptions of videos). The state of the art in video description
uses a multimodal attention mechanism that selectively attends to different input modalities (feature
types), such spatiotemporal motion features and audio features, in addition to temporal attention [ 11 ].


In this paper, we propose a new research target, a dialog system that can discuss dynamic scenes
with humans, which lies at the intersection of multiple avenues of research in natural language
processing, computer vision, and audio processing. To advance this goal, we introduce a new model
that incorporates technologies for multimodal attention-based video description into an end-to-end
dialog system. We also introduce a new dataset of human dialogues about videos. We are making our
dataset, code, and model publicly available for a new Video Scene-Aware Dialog Challenge.


**2** **Audio Visual Scene-Aware Dialog Dataset**


We collected text-based conversations data about short videos for Audio Visual Scene-Aware Dialog
(AVSD) as described in [ 12 ] using from an existing video description dataset, Charades [ 13 ], for
Dialog System Technology Challenge the 7th edition (DSTC7) [1] . Charades is an untrimmed and
multi-action dataset, containing 11,848 videos split into 7985 for training, 1863 for validation, and
2,000 for testing. It has 157 action categories, with several fine-grained actions. Further, this dataset
also provides 27,847 textual descriptions for the videos, each video is associated with 1–3 sentences.
As these textual descriptions are only available in the training and validation set, we report evaluation
results on the validation set.


The data collection paradigm for dialogs was similar to the one described in [ 9 ], in which for each
image, two different Mechanical Turk workers interacted via a text interface to yield a dialog. In [ 9 ],
each dialog consisted of a sequence of questions and answers about an image. In the video sceneaware dialog case, two Amazon Mechanical Turk (AMT) workers had a discussion about events in a
video. One of the workers played the role of an answerer who had already watched the video. The
answerer answered questions asked by another AMT worker – the questioner. The questioner was not
allowed to watch the whole video but only the first, middle and last frames of the video which were
single static images. After having a conversation to to ask about the events that happened between the
frames through 10 rounds of QA, the questioner summarized the events in the video as a description.


In total, we collected dialogs for 7043 videos from the Charades training set and all of the validation
set (1863 videos). Since we did not have scripts for the test set, we split the validation set into 732


1 http://workshop.colips.org/dstc7/call.html


2


Output word sequence


**Answer:**


**“He is watching TV.”** A n

















y i-1


y i


y i+1





































Figure 1: Our multimodal-attention based video scene-aware dialog system


and 733 videos and used them as our validation and test sets respectively. See Table 1 for statistics.
The average numbers of words per question and answer are 8 and 10, respectively.


Table 1: Video Scene-aware Dialog Dataset on Charades

|Col1|training validation test|
|---|---|
|#dialogs<br>#turns<br>#words|6,172<br>732<br>733<br>123,480<br>14,680<br>14,660<br>1,163,969<br>138,314<br>138,790|



**3** **Video Scene-aware Dialog System**


We built an end-to-end dialog system that can generate answers in response to user questions about
events in a video sequence. Our architecture is similar to the Hierarchical Recurrent Encoder in
Das _et al._ [ 9 ]. The question, visual features, and the dialog history are fed into corresponding
LSTM-based encoders to build up a context embedding, and then the outputs of the encoders are fed
into a LSTM-based decoder to generate an answer. The history consists of encodings of QA pairs.
We feed multimodal attention-based video features into the LSTM encoder instead of single static
image features. Figure 1 shows the architecture of our video scene-aware dialog system.


**3.1** **End-to-end Conversation Modeling**


This section explains the neural conversation model of [ 4 ], which is designed as a sequence-tosequence mapping process using recurrent neural networks (RNNs). Let _X_ and _Y_ be input and
output sequences, respectively. The model is used to compute posterior probability distribution
_P_ ( _Y |X_ ) . For conversation modeling, _X_ corresponds to the sequence of previous sentences in a
conversation, and _Y_ is the system response sentence we want to generate. In our model, both _X_ and
_Y_ are sequences of words. _X_ contains all of the previous turns of the conversation, concatenated in
sequence, separated by markers that indicate to the model not only that a new turn has started, but
which speaker said that sentence. The most likely hypothesis of _Y_ is obtained as


_Y_ ˆ = arg max (1)
_Y ∈V_ _[∗]_ _[P]_ [(] _[Y][ |][X]_ [)]



= arg max
_Y ∈V_ _[∗]_



_|Y |_
� _P_ ( _y_ _m_ _|y_ 1 _, . . ., y_ _m−_ 1 _, X_ ) _,_ (2)


_m_ =1



where _V_ _[∗]_ denotes a set of sequences of zero or more words in system vocabulary _V_ .


3


Let _X_ be word sequence _x_ 1 _, . . ., x_ _T_ and _Y_ be word sequence _y_ 1 _, . . ., y_ _M_ . The encoder network is
used to obtain hidden states _h_ _t_ for _t_ = 1 _, . . ., T_ as:


_h_ _t_ = LSTM ( _x_ _t_ _, h_ _t−_ 1 ; _θ_ _enc_ ) _,_ (3)


where _h_ 0 is initialized with a zero vector. LSTM( _·_ ) is a LSTM function with parameter set _θ_ _enc_ .


The decoder network is used to compute probabilities _P_ ( _y_ _m_ _|y_ 1 _, . . ., y_ _m−_ 1 _, X_ ) for _m_ = 1 _, . . ., M_

as:


_s_ 0 = _h_ _T_ (4)
_s_ _m_ = LSTM ( _y_ _m−_ 1 _, s_ _m−_ 1 ; _θ_ _dec_ ) (5)
_P_ ( **y** _|y_ 1 _, . . ., y_ _m−_ 1 _, X_ ) = softmax( _W_ _o_ _s_ _m_ + _b_ _o_ ) _,_ (6)


where _y_ 0 is set to `<eos>`, a special symbol representing the end of sequence. _s_ _m_ is the _m_ -th decoder
state. _θ_ _dec_ is a set of decoder parameters, and _W_ _o_ and _b_ _o_ are a matrix and a vector. In this model,
the initial decoder state _s_ 0 is given by the final encoder state _h_ _T_ as in Eq. (4), and the probability is
estimated from each state _s_ _m_ . To efficiently find _Y_ [ˆ] in Eq. (1), we use a beam search technique since
it is computationally intractable to consider all possible _Y_ .


In the scene-aware-dialog scenario, a scene context vector including audio and visual features is also
fed to the decoder. We modify the LSTM in Eqs. (4)–(6) as


_s_ _n,_ 0 = **0** [¯] (7)

_s_ _n,m_ = LSTM �[ _y_ _n,m_ [⊺] _−_ 1 _[, g]_ _n_ [⊺] []] [⊺] _[, s]_ _[n,m][−]_ [1] [;] _[ θ]_ _[dec]_ � _,_ (8)

_P_ ( **y** **n** _|y_ _n,_ 1 _, . . ., y_ _n,m−_ 1 _, X_ ) = softmax( _W_ _o_ _s_ _n,m_ + _b_ _o_ ) _,_ (9)


where _g_ _n_ is the concatenation of question encoding _g_ _n_ [(] _[q]_ [)] [, audio-visual encoding] _[ g]_ _n_ [(] _[av]_ [)] and history
encoding _g_ _n_ [(] _[h]_ [)] for generating the _n_ -th answer _A_ _n_ = _y_ _n,_ 1 _, . . ., y_ _n,|Y_ _n_ _|_ . Note that unlike Eq. (4), we
feed all contextual information to the LSTM at every prediction step. This architecture is more
flexible since the dimensions of encoder and decoder states can be different.

_g_ _n_ [(] _[q]_ [)] is encoded by another LSTM for the _n_ -th question, and _g_ _n_ [(] _[h]_ [)] is encoded with hierarchical LSTMs,
where one LSTM encodes each question-answer pair and then the other LSTM summarizes the
question-answer encodings into _g_ _n_ [(] _[h]_ [)] [. The audio-visual encoding is obtained by multi-modal attention]
described in the next section.


**3.2** **Multimodal-attention based Video Features**


To predict a word sequence in video description, prior work [ 14 ] extracted content vectors from
image features of VGG-16 and spatiotemporal motion features of C3D, and combined them into one
vector in the fusion layer as:



�



_g_ _n_ [(] _[av]_ [)] = tanh



_K_
� _d_ _k,n_
� _k_ =1



_,_ (10)



where
_d_ _k,n_ = _W_ _ck_ [(] _[λ]_ _[D]_ [)] _c_ _k,n_ + _b_ [(] _ck_ _[λ]_ _[D]_ [)] _,_ (11)
and _c_ _k,n_ is a context vector obtained using the _k_ -th input modality.


We call this approach Naïve Fusion, in which multimodal feature vectors are combined using
projection matrices _W_ _ck_ for _K_ different modalities (input sequences _x_ _k_ 1 _, . . ., x_ _kL_ for _k_ = 1 _, . . ., K_ ).


To fuse multimodal information, prior work [ 11 ] proposed method extends the attention mechanism.
We call this fusion approach _multimodal attention_ . The approach can pay attention to specific
modalities of input based on the current state of the decoder to predict the word sequence in video
description. The number of modalities indicating the number of sequences of input feature vectors is
denoted by _K_ .


The following equation shows an approach to perform the attention-based feature fusion:



�



_g_ _n_ [(] _[av]_ [)] = tanh



_K_
� _β_ _k,n_ _d_ _k,n_
� _k_ =1


4



_._ (12)


The similar mechanism for temporal attention is applied to obtain the multimodal attention weights
_β_ _k,n_ :

_β_ _k,n_ = ~~�~~ _Kκ_ ex =1 p( [exp(] _v_ _k,n_ _[v]_ ) _[κ,n]_ [)] _,_ (13)


where
_v_ _k,n_ = _w_ _B_ [⊺] [tanh(] _[W]_ _[B]_ _[g]_ _n_ [(] _[q]_ [)] + _V_ _Bk_ _c_ _k,n_ + _b_ _Bk_ ) _._ (14)

Here the multimodal attention weights are determined by question encoding _g_ _n_ [(] _[q]_ [)] and the context
vector of each modality _c_ _k,n_ as well as temporal attention weights in each modality. _W_ _B_ and _V_ _Bk_ are
matrices, _w_ _B_ and _b_ _Bk_ are vectors, and _v_ _k,n_ is a scalar. The multimodal attention weights can change
according to the question encoding and the feature vectors (shown in Figure 1).


This enables the decoder network to attend to a different set of features and/or modalities when
predicting each subsequent word in the description. Naïve fusion can be considered a special case of
Attentional fusion, in which all modality attention weights, _β_ _k,n_, are constantly 1.


**4** **Experiments for Multimodal attention-based Video Features**


To select best video features for the video scene-aware dialog system, we firstly evaluate the performance of video description using multimodal attention-based video features in this paper.


**4.1** **Datasets**


We evaluated our proposed feature fusion using the MSVD (YouTube2Text) [ 15 ], MSR-VTT [ 16 ],
and Charades [13] video data sets.


_•_ MSVD (YouTube2Text) covers a wide range of topics including sports, animals, and music.
We applied the same condition defined by [ 15 ]: a training set of 1,200 video clips, a
validation set of 100 clips, and a test set of the remaining 670 clips.


_•_ MSR-VTT is split into training, validation, and testing sets of 6,513, 497, and 2,990 clips
respectively. However, approvimatebly 12 of the MSR-VTT videos on YouTube have
been removed. We used the available data consists of 5,763, 419, and 2,616 clips for train,
validation, and test respectively defined by [11].


_•_ Charades [ 13 ] is split into 7985 clips for training and 1863 clips for validation. provides
27,847 textual descriptions for the videos, As these textual descriptions are only available in
the training and validation set, we report the evaluation results on the validation set.


Details of textual descriptions are summarized in Table 2.


Table 2: Sizes of textual descriptions in MSVD (YouTube2Text), MSR-VTT and Charades

|Dataset|#Clips|#Description|#Descriptions<br>per clip|#Word|Vocabulary<br>size|
|---|---|---|---|---|---|
|MSVD<br>MSR-VTT<br>Charades|1,970<br>10,000<br>9,848|80,839<br>200,000<br>16,140|41.00<br>20.00<br>1.64|8.00<br>9.28<br>13.04|13,010<br>29,322<br>2,582|



**4.2** **Video Processing**


We used a sequence of 4096-dimensional feature vectors of the output from the fully-connected fc7
layer of a VGG-16 network pretrained on the ImageNet dataset for the image features.


The pretrained C3D [ 17 ] model is used to generate features for model motion and short-term
spatiotemporal activity. The C3D network reads sequential frames in the video and outputs a
fixed-length feature vector every 16 frames. 4096-dimensional features of activation vectors from
fully-connected fc6-1 layer was applied to spatiotemporal features.


5


In addition to the VGG-16 and C3D features, we also adopted the state-of-the-art I3D features [ 18 ],
spatiotemporal features that were developed for action recognition. The I3D model inflates the 2D
filters and pooling kernels in the Inception V3 network along their temporal dimension, building 3D
spatiotemporal ones. We used the output from the "Mixed_5c" layer of the I3D network to be used as
video features in our framework. As a pre-processing step, we normalized all the video features to
have zero mean and unit norm; the mean was computed over all the sequences in the training set for
the respective feature.


In the experiments in this paper, we treated I3D-rgb (I3D features computed on a stack of 16 video
frame images) and I3D-flow (I3D features computed on a stack of 16 frames of optical flow fields) as
two separate modalities that are input to our multimodal attention model. To emphasize this, we refer
to I3D in the results tables as I3D (rgb-flow).


**4.3** **Audio Processing**


While the original MSVD (YouTube2Text) dataset does not contain audio features, we were able
to collect audio data for 1,649 video clips (84% of the dataset) from the video URLs. In our
previous work on multimodal attention for video description, we used two different types of audio
features: concatenated mel-frequency cepstral coefficient (MFCC) features [ 19 ], and SoundNet [ 20 ]
features [ 21 ]. In this paper, we also evaluate features extracted using a new state-of-the-art model,
Audio Set VGGish [22].


Inspired by the VGG image classification architecture (Configuration A without the last group of
convolutional/pooling layers), the Audio Set VGGish model operates on 0.96 s log Mel spectrogram
patches extracted from 16 kHz audio, and outputs a 128-dimensional embedding vector. The model
was trained to predict an ontology of labels from only the audio tracks of millions of YouTube videos.
In this work, we overlap frames of input to the VGGish network by 50%, meaning an Audio Set
VGGish feature vector is output every 0.48 s. For SoundNet [ 20 ], in which a fully convolutional
architecture was trained to predict scenes and objects using a pretrained image model as a teacher,
we take as input to the audio encoder the output of the second-to-last convolutional layer, which
gives a 1024-dimensional feature vector every 0.67 s, and has a receptive field of approximately 4.16
s. For raw MFCC features, sequences of 13-dimensional MFCC features are extracted from 50 ms
windows, every 25 ms, and then 20 consecutive frames are concatenated into a 260-dimensional
vector and normalized to zero mean/unit variance (computed over the training set) and used as input
to the BLSTM audio encoder.


**4.4** **Experimental Setup**


The caption generation model, i.e., the decoder network, is trained to minimize the cross entropy
criterion using the training set. Image features and deep audio features (SoundNet and VGGish) are
fed to the decoder network through one projection layer of 512 units, while MFCC audio features
are fed to a BLSTM encoder (one projection layer of 512 units and bidirectional LSTM layers of
512 cells) followed by the decoder network. The decoder network has one LSTM layer with 512
cells. Each word is embedded to a 256-dimensional vector when it is fed to the LSTM layer. In this
video description task, we used L2 regularization for all experimental conditions and used RMSprop
optimization.


**4.5** **Evaluation**


The quality of the automatically generated sentences will be evaluated with objective measures to
measure the similarity between the generated sentences and ground truth sentences. We will use the
evaluation code for MS COCO caption generation [2] for objective evaluation of system outputs, which
is a publicly available tool supporting various automated metrics for natural language generation such
as BLEU, METEOR, ROUGE_L, and CIDEr.


2 `[https://github.com/tylin/coco-caption](https://github.com/tylin/coco-caption)`


6


Table 3: Video description evaluation results on the MSVD (YouTube2Text) test set.


**MSVD (YouTube2Text) Full Dataset**

|Modalities (feature types)|Col2|Col3|Evaluation metric|Col5|Col6|
|---|---|---|---|---|---|
|Image|Spatiotemporal|Audio|BLEU4|METEOR|CIDEr|
|VGG-16<br>VGG-16|C3D<br>C3D|MFCC|0.524<br>0.539|0.320<br>0.322|0.688<br>0.674|
||I3D (rgb-ﬂow)||0.525|0.330|0.742|
||I3D (rgb-ﬂow)|MFCC<br>SoundNet<br>VGGish|0.527<br>0.529<br>**0.554**|0.325<br>0.319<br>**0.332**|0.702<br>0.719<br>**0.743**|



Table 4: Video description evaluation results on MSR-VTT Subset. Approximately 12% of the
MSR-VTT videos have been removed from YouTube, so we train and test on the remaining Subset of
MSR-VTT videos that we were able to download. The normalization for the visual features was not
applied to MSR-VTT in this experiments.


**MSR-VTT Subset**

|Modalities (feature types)|Col2|Col3|Evaluation metric|Col5|Col6|
|---|---|---|---|---|---|
|Image|Spatiotemporal|Audio|BLEU4|METEOR|CIDEr|
|VGG-16|C3D|MFCC|**0.397**|0.255|0.400|
||I3D (rgb-ﬂow)||0.347|0.241|0.349|
||I3D (rgb-ﬂow)|MFCC<br>SoundNet<br>VGGish|0.364<br>0.366<br>0.390|0.253<br>0.246<br>**0.263**|0.393<br>0.387<br>**0.417**|



**4.6** **Results and Discussion**


Tables 3, 4, and 5 show the evaluation results on the MSVD (YouTube2Text), MSR-VTT Subset, and
Charades datasets. The I3D spatiotemporal features outperformed the combination of VGG-16 image
features and C3D spatiotemporal features. We also tried a combination of VGG-16 image features
plus I3D spatiotemporal features, but we do not report those results because they did not improve
performance over I3D features alone. We believe this is because I3D features already include enough
image information for the video description task. In comparison to C3D, which uses the VGG-16 base
architecture and was trained on the Sports-1M dataset [ 23 ], I3D uses a more powerful Inception-V3
network architecture and was trained on the larger (and cleaner) Kinectics [ 24 ] dataset. As a result,
I3D has demonstrated state-of-the-art performance for the task of human action recognition in video
sequences [ 18 ]. Further, the Inception-V3 architecture has significantly fewer network parameters
than the VGG-16 network, making it more efficient.


In terms of audio features, the Audio Set VGGish model provided the best performance. While we
expected the deep features (SoundNet and VGGish) to provide improved performance compared to
MFCC, there are several possibilities as to why VGGish performed better than SoundNet. First, the
VGGish model was trained on more data, and had audio specific labels, whereas SoundNet used
pre-trained image classification networks to provide labels for training the audio network. Second,
the large Audio Set ontology used to train VGGish likely provides the ability to learn features more
relevant to text descriptions than the broad scene/object labels used by SoundNet.


Table 5: Video description evaluation results on Charades.


**Charades Dataset**

|Modalities (feature types)|Col2|Col3|Evaluation metric|Col5|Col6|
|---|---|---|---|---|---|
|Image|Spatiotemporal|Audio<br>|BLEU4|METEOR|CIDEr|
||I3D (rgb-ﬂow)||0.094|0.149|0.236|
||I3D (rgb-ﬂow)|MFCC<br>SoundNet<br>VGGish|0.098<br>-<br>**0.100**|0.156<br>-<br>**0.157**|0.268<br>-<br>**0.270**|



Since it is intractable to enumerate all possible word sequences in vocabulary _V_, we usually limit
them to the _n_ -best hypotheses generated by the system. Although in theory the distribution _P_ ( _Y_ _[′]_ _|X_ )
should be the true distribution, we instead estimate it using the encoder-decoder model.


7


Table 6: System response generation evaluation results with objective measures.

|Input features|Attentional<br>fusion|BLEU1 BLEU2 BLEU3 BLEU4 METEOR ROUGE_L CIDEr|
|---|---|---|
|QA<br>QA + Captions<br>QA + VGG16<br>QA + I3D<br>QA + I3D<br>QA + I3D + VGGish<br>QA + I3D + VGGish|-<br>-<br>-<br>no<br>yes<br>no<br>yes|0.236<br>0.142<br>0.094<br>0.065<br>0.101<br>0.257<br>0.595<br>0.245<br>0.152<br>0.103<br>0.073<br>0.109<br>0.271<br>0.705<br>0.231<br>0.141<br>0.095<br>0.067<br>0.102<br>0.259<br>0.618<br>0.246<br>0.153<br>0.104<br>0.073<br>0.109<br>0.269<br>0.680<br>0.250<br>0.157<br>0.108<br>0.077<br>0.110<br>0.274<br>0.724<br>0.249<br>0.155<br>0.106<br>0.075<br>0.110<br>0.275<br>0.701<br>**0.256**<br>**0.161**<br>**0.109**<br>**0.078**<br>**0.113**<br>**0.277**<br>**0.727**|



**5** **Experiments for Video-scene-aware Dialog**


In this paper, we extended an end-to-end dialog system to scene-aware dialog with multimodal fusion.
As shown in Fig. 1, we embed the video and audio features selected in Section 2.


**5.1** **Conditions**


We evaluated our proposed system with the dialog data for Charades we collected. Table 1 shows
the size of each data set. We compared the performance between models trained from various
combinations of the QA text, visual and audio features. In addition, we tested an efficacy of
multimodal-attention mechanism for dialg response generation. We employed an ADAM optimizer

[ 25 ] with the cross-entropy criterion and iterated the training process up to 20 epochs. For each of
the encoder-decoder model types, we selected the model with the lowest perplexity on the expanded
development set.


We used the parameters of the LSTMs with #layer=2 and #cells=128 for encoding history and
question sentences. Video features were projected to 256 dimensional feature space before modality
fusion. The decoder LSTM had a structure of #layer=2 and #cells=128 as well.


**5.2** **Evaluation Results**


Table 6 shows the response sentence generation performance of our models, training and decoding
methods using objective measures, BLEU1-4, METEOR, ROUGE_L, and CIDEr, which were
computed with the evaluation code for MS COCO caption generation as done for video description.
We investigated different input features including question-answering dialog history plus last question
(QA), human-annotated captions (Captions), video features of VGG16 or I3D rgb and flow features
(I3D), and audio features (VGGish).


First we evaluated response generation quality with only QA features as a baseline without any
video scene features. Then, we added the caption features to QA, and the performance improved
significantly. This is because each caption provided the scene information in natural language and
helped the system answer the question correctly. However, such human annotations are not available
for real systems.


Next we added VGG16 features to QA, but they did not increase the evaluation scores from those of
QA-only features. This result indicates that QA+VGG16 is not enough to let the system generate
better responses than those of QA+Captions. After that, we replaced VGG16 with I3D, and obtained
a certain improvement from the QA-only case. As in the video description, it has been shown
that the I3D features are also useful for scene-aware dialog. Furthermore, we applied the multimodal attention mechanism (attentional fusion) for I3D rgb and flow features, and obtained further
improvement in all the metrics.


Finally, we examined the efficacy of audio features. The table shows that VGGish obviously
contributed to increasing the response quality especially when using the attentional fusion. The
following example of system response was obtained with or without VGGish features, which worked
better for the questions regarding audios:


8


```
Question: was there audio ?
Ground truth: there is audio, i can hear music and background noise .
I3D: no, there is no sound in the video .
I3D+VGGish: yes there is sound in the video .

```

**6** **Conclusion**


In this paper, we propose a new research target, a dialog system that can discuss dynamic scenes
with humans, which lies at the intersection of multiple avenues of research in natural language
processing, computer vision, and audio processing. To advance this goal, we introduce a new model
that incorporates technologies for multimodal attention-based video description into an end-to-end
dialog system. We also introduce a new dataset of human dialogues about videos. Using this
new dataset, we trained an end-to-end conversation model that generates system responses in a
dialog about an input video. Our experiments demonstrate that using multimodal features that were
developed for multimodal attention-based video description enhances the quality of generated dialog
about dynamic scenes. We are making our data set and model publicly available for a new Video
Scene-Aware Dialog challenge.


**References**


[1] Michael F McTear, “Spoken dialogue technology: enabling the conversational user interface,”
_ACM Computing Surveys (CSUR)_, vol. 34, no. 1, pp. 90–169, 2002.


[2] Steve J Young, “Probabilistic methods in spoken–dialogue systems,” _Philosophical Transactions_
_of the Royal Society of London A: Mathematical, Physical and Engineering Sciences_, vol. 358,
no. 1769, pp. 1389–1402, 2000.


[3] Victor Zue, Stephanie Seneff, James R Glass, Joseph Polifroni, Christine Pao, Timothy J
Hazen, and Lee Hetherington, “Juplter: a telephone-based conversational interface for weather
information,” _IEEE Transactions on speech and audio processing_, vol. 8, no. 1, pp. 85–96,
2000.


[4] Oriol Vinyals and Quoc Le, “A neural conversational model,” _arXiv preprint arXiv:1506.05869_,
2015.


[5] Ryan Lowe, Nissan Pow, Iulian Serban, and Joelle Pineau, “The ubuntu dialogue corpus:
A large dataset for research in unstructured multi-turn dialogue systems,” _arXiv preprint_
_arXiv:1506.08909_, 2015.


[6] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence
Zitnick, and Devi Parikh, “VQA: Visual Question Answering,” in _International Conference on_
_Computer Vision (ICCV)_, 2015.


[7] Peng Zhang, Yash Goyal, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh, “Yin and
Yang: Balancing and answering binary visual questions,” in _Conference on Computer Vision_
_and Pattern Recognition (CVPR)_, 2016.


[8] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh, “Making the V
in VQA matter: Elevating the role of image understanding in Visual Question Answering,” in
_Conference on Computer Vision and Pattern Recognition (CVPR)_, 2017.


[9] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José M. F. Moura, Devi
Parikh, and Dhruv Batra, “Visual dialog,” _CoRR_, vol. abs/1611.08669, 2016.


[10] Abhishek Das, Satwik Kottur, José M.F. Moura, Stefan Lee, and Dhruv Batra, “Learning
cooperative visual dialog agents with deep reinforcement learning,” in _International Conference_
_on Computer Vision (ICCV)_, 2017.


[11] Chiori Hori, Takaaki Hori, Teng-Yok Lee, Ziming Zhang, Bret Harsham, John R. Hershey,
Tim K. Marks, and Kazuhiko Sumi, “Attention-based multimodal fusion for video description,”
in _The IEEE International Conference on Computer Vision (ICCV)_, Oct 2017.


[12] Huda Alamri, Vincent Cartillier, Raphael Gontijo Lopes, Abhishek Das, Jue Wang, Irfan Essa,
Dhruv Batra, Devi Parikh, Anoop Cherian, Tim K Marks, and Chiori Hori, “Audio visual
scene-aware dialog (avsd) challenge at dstc7,” _arXiv preprint arXiv:1806.00525_, 2018.


9


[13] Gunnar A. Sigurdsson, Gül Varol, Xiaolong Wang, Ivan Laptev, Ali Farhadi, and Abhinav
Gupta, “Hollywood in homes: Crowdsourcing data collection for activity understanding,” _ArXiv_,
2016.

[14] Haonan Yu, Jiang Wang, Zhiheng Huang, Yi Yang, and Wei Xu, “Video paragraph captioning
using hierarchical recurrent neural networks,” _CoRR_, vol. abs/1510.07712, 2015.

[15] Sergio Guadarrama, Niveda Krishnamoorthy, Girish Malkarnenkar, Subhashini Venugopalan,
Raymond Mooney, Trevor Darrell, and Kate Saenko, “Youtube2text: Recognizing and describing arbitrary activities using semantic hierarchies and zero-shot recognition,” in _Proceedings of_
_the IEEE International Conference on Computer Vision_, 2013, pp. 2712–2719.

[16] Jun Xu, Tao Mei, Ting Yao, and Yong Rui, “Msr-vtt: A large video description dataset for
bridging video and language,” in _Proceedings of the IEEE Conference on Computer Vision and_
_Pattern Recognition (CVPR)_, 2016.

[17] Du Tran, Lubomir D. Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri, “Learning
spatiotemporal features with 3d convolutional networks,” in _2015 IEEE International Con-_
_ference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015_, 2015, pp.
4489–4497.

[18] Joao Carreira and Andrew Zisserman, “Quo vadis, action recognition? a new model and the
kinetics dataset,” in _CVPR_, 2017.

[19] Chiori Hori, Takaaki Hori, Teng-Yok Lee, Ziming Zhang, Bret Harsham, John R Hershey,
Tim K Marks, and Kazuhiko Sumi, “Attention-based multimodal fusion for video description,”
in _ICCV_, 2017.

[20] Yusuf Aytar, Carl Vondrick, and Antonio Torralba, “Soundnet: Learning sound representations
from unlabeled video,” in _NIPS_, 2016.

[21] Chiori Hori, Takaaki Hori, Tim K Marks, and John R Hershey, “Early and late integration of
audio features for automatic video description,” in _ASRU_, 2017.

[22] S. Hershey, S. Chaudhuri, D. P. W. Ellis, J. F. Gemmeke, A. Jansen, R. C. Moore, M. Plakal,
D. Platt, R. A. Saurous, B. Seybold, M. Slaney, R. J. Weiss, and K. Wilson, “CNN architectures
for large-scale audio classification,” in _ICASSP_, 2017.

[23] Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, and
Li Fei-Fei, “Large-scale video classification with convolutional neural networks,” in _Proceedings_
_of the IEEE conference on Computer Vision and Pattern Recognition_, 2014, pp. 1725–1732.

[24] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al., “The kinetics human
action video dataset,” _arXiv_, 2017.

[25] Diederik Kingma and Jimmy Ba, “Adam: A method for stochastic optimization,” _arXiv preprint_
_arXiv:1412.6980_, 2014.


10


