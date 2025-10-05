## **A Simple Transformer-Based Model for Ego4D** **Natural Language Queries Challenge**

Sicheng Mo, Fangzhou Mu, Yin Li


University of Wisconsin-Madison


**Abstract.** This report describes Badgers@UW-Madison, our submission to the Ego4D Natural Language Queries (NLQ) Challenge. Our solution inherits the point-based event representation from our prior work
on temporal action localization [12], and develops a Transformer-based
model for video grounding. Further, our solution integrates several strong
video features including SlowFast [2], Omnivore [3] and EgoVLP [6].
Without bells and whistles, our submission based on a single model
achieves 12 _._ 64% Mean R@1 and is ranked 2 _nd_ on the public leaderboard.
Meanwhile, our method garners 28.45% (18.03%) R@5 at tIoU=0.3 (0.5),
surpassing the top-ranked solution by up to 5.5 absolute percentage
points.


**1** **Introduction**


Given an untrimmed video and a text query, the Ego4D Natural Language
Queries (NLQ) task seeks to localize the temporal window within the video
where the answer to the query is evident [4]. NLQ provides a first step towards
searching our egocentric visual experience using natural language, and thus opens
up the opportunity for a new generation of AI assistants. Similar to text grounding in videos (a.k.a. video grounding [5]), NLQ requires the understanding and
reasoning of the text query and video content, yet under the interference of
ego-motion within egocentric video, thereby posing additional challenges.
Prior solutions to NLQ adopted sophisticated model design for video grounding [6,7]. Our solution instead explores a minimalist design based on Transformer [11]. Specifically, we re-purpose ActionFormer, our prior work on temporal action localization [12] for video grounding. The resulting model considers
every moment in the video as an event candidate, measures their relevance to
the text query, and regresses the event boundaries from foreground moments.
We show that our simple model works surprisingly well on the Ego4D NLQ
task. Further, we experiment with different video features including SlowFast [2],
Omnivore [3] and EgoVLP [6], and integrate them into our model to further
boost the performance. Our final submission based on a single model achieves
12 _._ 64% Mean R@1 and is ranked 2 _nd_ on the public leaderboard. Meanwhile, our
solution presents the highest R@5 results on the leaderboard and surpasses the
top-ranked solution by up to 5.5 absolute percentage points.


2 Mo et al.












|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||Fe|ature Matchi|ng|||
||||||||













**Transformer Encoder**


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|Pro|jection Using|Convolution|s|








|Col1|Col2|Col3|
|---|---|---|
|ojection Usin|g Convolutio|ns|
||||


|Col1|Col2|Col3|Col4|
|---|---|---|---|
||Downs<br>(opti|LP<br>Norm<br>ample<br>onal)<br>||
|||||
||Layer|Norm|Norm|
||**+**|**+**|**+**|
||M|M|M|
||Layer|Norm|Norm|
||Multi-<br>Atte<br>**+**|Multi-<br>Atte<br>**+**|Multi-<br>Atte<br>**+**|
||Input<br>Embeddings|Input<br>Embeddings|Input<br>Embeddings|



**Fig. 1. Overview of our model.** Our model consists of a video branch and a text
branch, both based on transformers. A similarity score is computed between every
video and text embeddings. Those with high similarity are fused and undergo further
processing for event boundary regression.


**2** **Approach**


Our method represents a video as a sequence of feature vectors, where each vector
is derived from a short clip using pre-trained video backbones. Our Transformerbased model further processes this sequence of video features, computes their
similarity to an embedding of the text query, and decodes temporal event segments from moments with high similarity scores.
Figure 1 illustrates the design of our model. Our model borrows its video
encoder from Actionformer [12] and runs an additional Transformer-based text
encoder to obtain an embedding of the text query. The video and text embeddings are subsequently fused and further examined by shared classification and
regression heads as in ActionFormer, yielding event predictions.


**Input Video/Text Representations.** We use the official release of Slowfast [2]
and Omnivore [3] features as our video representation. We additionally include
video features from EgoVLP [6], a video-language pre-training method tailored
to egocentric videos. We fuse the features via simple concatenation and feed
them into the video branch of our grounding model. For text queries, we extract
token-wise embeddings using CLIP [10] and feed them into the text branch.


**Video Encoder.** Our video encoder adopts the same design as our prior work [12].
It consists of two embedding convolutions followed by seven Transformer blocks
with local self-attention. The last five blocks additionally perform 2x downsampling with depth-wise strided convolutions, resulting in a feature pyramid of
six levels. More details can be found in [12].


**Text Encoder.** Our text encoder is also a Transformer network, consisting of
a linear feature projection layer followed by multiple Transformer layers. The
output text embeddings share the same length as the input.


Simple Transformer for Ego4D NLQ 3


**Table 1. Results on the Ego4D NLQ dataset.** All results on the test set are
evaluated on the submission server. Results from concurrent works are colored.

|Model Split Video Feature|R@1|R@5|
|---|---|---|
|Model<br>Split Video Feature|IoU = 0.3 IoU = 0.5 mean|IoU = 0.3 IoU = 0.5|
|EgoVLP<br>Val<br>EgoVLP<br>ReLER@ZJU-Alibaba Val<br>SlowFast<br>ReLER@ZJU-Alibaba Val<br>Omnivore<br>ReLER@ZJU-Alibaba Val<br>Ensemble<br>Ours<br>Val<br>SlowFast<br>Ours<br>Val<br>Omnivore<br>Ours<br>Val<br>EgoVLP<br>Ours<br>Val<br>Fused|10.84<br>6.81<br>8.83<br>10.79<br>6.74<br>8.77<br>10.74<br>6.87<br>8.81<br>11.33<br>7.05<br>9.19<br>9.96<br>6.63<br>8.30<br>11.46<br>7.28<br>9.37<br>12.03<br>7.15<br>9.59<br>15.72<br>10.12<br>12.92|18.84<br>13.45<br>13.19<br>8.85<br>13.47<br>8.72<br>14.77<br>8.98<br>25.86<br>16.80<br>28.42<br>17.71<br>28.34<br>17.71<br>34.64<br>23.64|
|EgoVLP<br>Test<br>EgoVLP<br>ReLER@ZJU-Alibaba Test<br>Ensemble<br>CONE (3rd)<br>Test<br>Red Panda (1st)<br>Test<br>Ours (2nd)<br>Test<br>Fused|10.46<br>6.24<br>8.35<br>12.89<br>8.14<br>10.52<br>15.26<br>9.24<br>12.25<br>**16.46**<br>**10.06**<br>**13.26**<br>15.71<br>9.57<br>12.64|16.76<br>11.29<br>15.41<br>9.94<br>26.42<br>16.51<br>22.95<br>16.11<br>**28.45**<br>**18.03**|



**Feature Fusion.** We inject textual information into video embeddings at the
top of the encoders using Adaptive Attention Normalization (AdaAttN) [8]. Intuitively, AdaAttN aligns the distribution of video / text features weighted by their
attention scores. Consequently, the modulated video features are made aware of
the text query, hence can inform event classification and offset regression.

**Classification and Regression Heads.** The heads adopt the same convolutional design and are shared across pyramid levels, similar to those in [12]. The
classification head outputs a binary score for each point on the pyramid, whereas
the regression head predicts the distances from a foreground point to the event’s
onset and offset.

**Loss Function.** Our loss function is a summation of three terms, including the
same classification and regression losses from ActionFormer [12] and an additional, novel NCE loss inspired by the winning entry of 1 _st_ Ego4D NLQ Challenge. To evaluate the NCE loss, we label a moment as positive if it lies inside
a ground-truth event and negative otherwise. We pair the moments with the
text embedding to form positive and negative training pairs. We apply the NCE
loss on the video embeddings _immediately after feature fusion_ to encourage early
separation of positive and negative moments.

**Implementation Details.** Our model maintains an embedding dimension of
512 throughout the network. The Transformer layers employ 16 heads for selfattention and AdaAttN. The model is trained using the AdamW optimizer [9] for
9 epochs. We use a mini-batch size of 16 and a learning rate of 1e-3 with linear
warm-up and consine decay. At inference time, the model outputs at most 2,000
event predictions for each video. The initial predictions are further combined
and refined with SoftNMS [1], yielding final predictions subject to evaluation.


**3** **Experiments and Results**


We now describe our experiments on the Ego4D NLQ dataset. We first present
an ablation study of our method using different video features, then compare
our results to other methods on the leaderboard.


4 Mo et al.















**Fig. 2. Failure cases of our model.** ‘GT’ are input text queries and ground-truth
events displayed as key frames. ‘P#’ are the top-5 predictions with text descriptions
derived from video frames within the prediction window.













**Dataset and Evaluation Metrics.** The Ego4D NLQ dataset contains 227
hours of videos with 5.9K clips and 22.5K pairs of events and text queries. The
video clips last either 8 or 20 minutes. Text queries are generated from 13 predefined templates. We follow the official train/val/test splits in our experiments.
We train our model on the train split when reporting results on the val split, and
train on the combined train/val splits when reporting results on the test split.
We report Recall@1 (R@1) and Recall@5 (R@5) at tIoU thresholds 0.3 and 0.5.


**Results.** Table 1 summarizes our results. On the val split, our method is on par
with the champion from last challenge (ReLER@ZJU-Alibaba [7]) in R@1 scores
but is better in R@5. Among the three types of features (SlowFast, Omnivore
and EgoVLP), EgoVLP alone yields the best results thanks to the in-domain
pre-training. An ensemble of all three features further boosts all metrics by a
significant amount. On the test split, our model ultimately achieves 12 _._ 64% Mean
R@1, 2.12% absolute percentage points higher than the last champion [7]. Our
Mean R@1 score lags behind the top-ranked solution by a tiny gap of 0.62%.
Meanwhile, our R@5 scores beat all entries on the leaderboard, and in particular,
surpasses the top-ranked solution by up to 5.5 absolute percentage points.


**Limitations and Discussion.** Some failure cases of our model are shown in

Figure 2. A common failure mode is that our model sticks to the moments where
the object of interest is present, yet failed to localize the moment from which the
answer to the question can be deduced. We conjecture that explicit reasoning
about the text queries might be necessary to avoid those errors. Another interesting observation is that the R@5 scores of our model are considerably higher
than other methods, suggesting that learning stronger matching / classification
heads is a promising future direction.


**4** **Conclusion**


In this report, we presented our solution to the Ego4D NLQ task. Our solution
combines a variant of a latest temporal action localization backbone with strong
video features, resulting in impressive empirical results on the public learderboard while maintaining a minimalist model design. We hope our model design
and results can shed light on video grounding and egocentric vision.


Simple Transformer for Ego4D NLQ 5


**References**


1. Bodla, N., Singh, B., Chellappa, R., Davis, L.S.: Soft-nms – improving object
detection with one line of code. In: International Conference on Computer Vision
(ICCV) (2017)
2. Feichtenhofer, C., Fan, H., Malik, J., He, K.: Slowfast networks for video recognition. In: International Conference on Computer Vision (ICCV) (2019)
3. Girdhar, R., Singh, M., Ravi, N., van der Maaten, L., Joulin, A., Misra, I.: Omnivore: A single model for many visual modalities. In: Conference on Computer
Vision and Pattern Recognition (CVPR) (2022)
4. Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., Hamburger, J., Jiang, H., Liu, M., Liu, X., Martin, M., Nagarajan, T., Radosavovic,
I., Ramakrishnan, S.K., Ryan, F., Sharma, J., Wray, M., Xu, M., Xu, E.Z., Zhao,
C., Bansal, S., Batra, D., Cartillier, V., Crane, S., Do, T., Doulaty, M., Erapalli,
A., Feichtenhofer, C., Fragomeni, A., Fu, Q., Gebreselasie, A., Gonz´alez, C., Hillis,
J., Huang, X., Huang, Y., Jia, W., Khoo, W., Kol´aˇr, J., Kottur, S., Kumar, A.,
Landini, F., Li, C., Li, Y., Li, Z., Mangalam, K., Modhugu, R., Munro, J., Murrell,
T., Nishiyasu, T., Price, W., Ruiz, P., Ramazanova, M., Sari, L., Somasundaram,
K., Southerland, A., Sugano, Y., Tao, R., Vo, M., Wang, Y., Wu, X., Yagi, T.,
Zhao, Z., Zhu, Y., Arbel´aez, P., Crandall, D., Damen, D., Farinella, G.M., Fuegen, C., Ghanem, B., Ithapu, V.K., Jawahar, C.V., Joo, H., Kitani, K., Li, H.,
Newcombe, R., Oliva, A., Park, H.S., Rehg, J.M., Sato, Y., Shi, J., Shou, M.Z.,
Torralba, A., Torresani, L., Yan, M., Malik, J.: Ego4d: Around the world in 3,000
hours of egocentric video. In: Computer Vision and Pattern Recognition (CVPR)
(2022)
5. Lan, X., Yuan, Y., Wang, X., Wang, Z., Zhu, W.: A survey on temporal sentence
grounding in videos. Transactions on Multimedia Computing, Communications,
and Applications (TOMM) (2021)
6. Lin, K.Q., Wang, A.J., Soldan, M., Wray, M., Yan, R., Xu, E.Z., Gao, D., Tu, R.,
Zhao, W., Kong, W., Cai, C., Wang, H., Damen, D., Ghanem, B., Liu, W., Shou,
M.Z.: Egocentric video-language pretraining. In: Neural Information Processing
Systems (NeurIPS) (2022)
7. Liu, N., Wang, X., Li, X., Yang, Y., Zhuang, Y.: Reler@ zju-alibaba submission to
the ego4d natural language queries challenge 2022. arXiv preprint arXiv:2207.00383
(2022)
8. Liu, S., Lin, T., He, D., Li, F., Wang, M., Li, X., Sun, Z., Li, Q., Ding, E.: Adaattn:
Revisit attention mechanism in arbitrary neural style transfer. In: International
Conference on Computer Vision (ICCV) (2021)
9. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International
Conference on Learning Representations (ICLR) (2019)
10. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from
natural language supervision. In: International Conference on Machine Learning
(ICML) (2021)
11. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser,
�L., Polosukhin, I.: Attention is all you need. Neural Information Processing Systems
(NeurIPS) (2017)
12. Zhang, C., Wu, J., Li, Y.: Actionformer: Localizing moments of actions with transformers. In: European Conference on Computer Vision (ECCV) (2022)


