# HumanOmni: A Large Vision-Speech Language Model for Human-Centric Video Understanding

## Abstract


In human-centric scenes, the ability to simultaneously understand visual and auditory information is crucial. While recent omni models can process multiple modalities, they generally lack effectiveness in human-centric scenes due to the absence of large-scale, specialized datasets and non-targeted architectures. In this work, we developed HumanOmni, the industry's first human-centric Omnimultimodal large language model. We constructed a dataset containing over 2.4 million human-centric video clips with detailed captions and more than 14 million instructions, facilitating the understanding of diverse human-centric scenes. Hu-manOmni includes three specialized branches for understanding different types of scenes. It adaptively fuses features from these branches based on user instructions, significantly enhancing visual understanding in scenes centered around individuals. Moreover, HumanOmni integrates audio features to ensure a comprehensive understanding of environments and individuals. Our experiments validate HumanOmni's advanced capabilities in handling human-centric scenes across a variety of tasks, including emotion recognition, facial expression description, and action understanding. Our model will be open-sourced to facilitate further development and collaboration within both academia and industry.



## Main Text
In human-centric scenes, the ability to simultaneously understand visual and auditory information is crucial. While recent omni models can process multiple modalities, they generally lack effectiveness in human-centric scenes due to the absence of large-scale, specialized datasets and non-targeted architectures. In this work, we developed HumanOmni, the industry's first human-centric Omnimultimodal large language model. We constructed a dataset containing over 2.4 million human-centric video clips with detailed captions and more than 14 million instructions, facilitating the understanding of diverse human-centric scenes. Hu-manOmni includes three specialized branches for understanding different types of scenes. It adaptively fuses features from these branches based on user instructions, significantly enhancing visual understanding in scenes centered around individuals. Moreover, HumanOmni integrates audio features to ensure a comprehensive understanding of environments and individuals. Our experiments validate HumanOmni's advanced capabilities in handling human-centric scenes across a variety of tasks, including emotion recognition, facial expression description, and action understanding. Our model will be open-sourced to facilitate further development and collaboration within both academia and industry.

In the era of rapid digital and intelligent development, understanding human-centric scenes has become increasingly critical. These scenes extend beyond video chat 
[26]
 to encompass education, healthcare, social interactions, and entertainment. In these human-centric scenes, vision and speech are typically present simultaneously. For certain tasks, both visual and auditory information provide significant benefits, such as in emotion recognition 
[9,
11,
32,
66]
 and speaker-specific speech recognition. Speaker-specific speech recognition builds upon automatic speech recognition by incorporating additional description about the speaker. We are currently defining this task and collecting such a dataset, with plans to release it in the next version of our work.

Current methods predominantly focus on Vision-Language models 
[1,
16,
24,
25,
30,
36,
37,
43,
[55]
[56]
[57]
67]
, which effectively handle visual and textual information but generally lack the capability to process audio inputs. This limitation results in an incomplete understanding of scenes. In recent years, some omni models 
[17,
18,
28,
51,
54]
 have been proposed to address multiple modalities, including visual, auditory, and textual data. However, these models often emphasize generic scenes and lack targeted training for human-centric scenes. Additionally, they do not incorporate specialized model designs, leading to weaker performance in understanding such scenarios.

In Fig. 
1
, we illustrate the HumanOmni pipeline, which is capable of processing a multimodal input encompassing textual, auditory and visual data.

For visual component, facial expressions, body movements, individual attributes, and interactions with the environment are crucial elements for understanding human-centric video content. Different types of features are critical for different tasks; for instance, emotion analysis heavily relies on facial expressions, action recognition focuses more on body movements, and social interaction analysis depends on interactions between individuals and their environment or objects. To address these diverse requirements, we have designed three specialized branches: the Face-related Branch, Bodyrelated Branch, and Interaction-related Branch. These branches capture distinct features to enhance the model's performance across various human-related tasks. Leveraging advanced visual encoders SigLIP 
[58]
 and large language models Qwen2.5 
[45]
, which exhibit strong feature extraction and representation capabilities, our branch architectures remain flexible and do not require task-specific modifications. To guide each branch to focus on specific tasks, we train them using different video clips and instructions, ensuring that each branch specializes in extracting different types of features, as detailed in the training section.

In particular, while the three branches share a generic architecture, they differ in their visual projector components. The face-related branch employs a detail-sensitive projector MLP2xGeLU 
[23]
 to better capture subtle facial changes. In contrast, the body-related branch and interaction-related branch utilize a spatial-temporal projector STC 
[12]
, handling continuous actions and interaction scenes. Importantly, despite using two different types of projectors, the features derived from both approaches remain spatially and temporally aligned, ensuring consistency and effectiveness in feature fusion.

The features from these three branches are complementary to some extent. However, directly concatenating them would lead to an excessive number of visual tokens, imposing additional computational and analytical burdens on the LLM. While simply summing the features is one approach, we have devised a more sophisticated method for feature fusion. Inspired by LLAVA-Octopus 
[65]
, we use the rich information contained in the user instructions to dynamically adjust the weighting of features Figure 
1
: Pipeline of HumanOmni. HumanOmni is a vision-speech language model that focus on human-centric scenes. For the visual component, we pre-trained three distinct branches using separate data. The features from these branches are fused based on user instructions. HumanOmni also supports audio input, enhancing its ability to fully understand complex human-centric scenes.

from each branch. For example, when the instruction pertains to emotion recognition, the model places greater emphasis on features from the face-related branch; for interaction scenes, it prioritizes the interaction-related branch.

Specifically, to process user instructions, we employ BERT 
[15]
 for encoding the commands. We focus on the [CLS] token produced by BERT, which encapsulates the semantic essence of the instruction. We chose BERT as our text encoder due to its robust pre-training, enabling it to capture deep semantic information from text. BERT utilizes a bidirectional transformer architecture to encode input text, with the [CLS] token effectively summarizing the semantics of the entire sentence. This provides a strong foundation for subsequent weight generation processes.

Next, we introduce two MLPs for generating feature weights. The first MLP receives the [CLS] token as input and, through multiple layers of neural network processing, produces intermediate feature representations that capture high-level semantic details from the instructions. The second MLP then takes these intermediate representations as input and further refines them to generate final weight values, each corresponding to one of the visual projectors. These generated weights are used to dynamically adjust and combine the visual features extracted by the three projectors, selecting the most suitable features for the task at hand. Suppose the three projectors extract features denoted as F 1 , F 2 and F 3 , and the generated weights are w 1 , w 2 and w 3 , respectively. Then, the final visual representation F is given by:

This instruction-driven feature fusion approach enhances the model's flexibility and adaptability while ensuring efficient resource utilization. It allows the model to automatically adjust its focus on different types of features based on task requirements.

For the auditory component, we follow 
[54]
 utilizing the audio preprocessor and encoder from Whisper-large-v3 
[40]
 to process audio data. Specifically, the audio input first undergoes preliminary processing through the audio preprocessor, generating a format suitable for encoding. Subsequently, the preprocessed audio data is encoded using Whisper's encoder, extracting robust audio features.

To ensure that audio features can be effectively integrated with visual and textual features in the same domain, we employ MLP2xGeLU as the projector. This projector maps the audio features into the text domain.

For the text, we directly used the corresponding text encoder module from the LLM to encode the text. Consequently, the audio tokens, along with visual and text tokens, are concatenated within a unified representation space using specific tokens to distinguish between features from different modalities, and then fed into the LLM decoder for further processing. We employ scene detection and segmentation to divide the video into clips to prevent unnatural temporal changes caused by instantaneous scene transitions. Then, the clips with relatively low resolution are removed, and the key frames detection algorithms are applied, which helps to quantify the temporal changes in clips. To further improve learning efficiency, we generate brief captions based on advanced multimodal model, and eliminate the clips similar in contexts. Finally, in addition to being automatically annotated with human and face bounding boxes, the remaining video clips will be processed by several state-of-the-art multimodal models to generate detailed captions. Subsequently, a large language model will be used to synthesize the common content across these captions, while filtering out unique content that may result from model hallucinations.

Although there are currently many multimodal annotated datasets, including OCR and visual navigation, there is a lack of a large-scale human-centric dataset with fine-grained annotations, limiting the development of human-centric video understanding. Based on the existing large-scale video datasets, we have carefully designed a data processing workflow and present the largest human-centric dataset for comprehensive human-centric video understanding.

We form our dataset based on the existing web-scale dataset: Panda-70M 
[10]
 which covers various scenes and contents. Despite initial processing and caption generation, much useful information, particularly related to human subjects, remains underutilized. Existing research has demonstrated that data quality is crucial for model performance; simply increasing the quantity of low-quality data does not lead to significant performance improvements. Therefore, we have further optimized our dataset to create a human-centric collection with the following characteristics: high video resolution (above 480P), rich temporal dynamics, inclusion of face and body detection bounding boxes, and captions verified through a dual-check process. Our data processing includes the following step and an illustration is in Fig. 
2
.

-Stage 1: Temporal processing. Temporal dynamics is a key feature distinguishing video data from static images. Videos lacking temporal changes offer limited learning value, while those with excessive and unrealistic temporal variations can confuse models. To extract naturally smooth temporal changes in the video, we employed scene recognition algorithms to identify multiple scenes within videos and segment them into clips, as shown in the center of Fig. 
2
. Scene detection and segmentation avoid drastic changes from multiple different perspectives in the video clip. Besides, it ensures that clips with intense temporal changes are broken down into shorter segments, which can then be filtered out.

Furthermore, we extracted keyframes for each clip to find the temporal variations. The clips with minimal temporal changes would have very few keyframes identified, allowing us to screen out such Caption: In the video, the character exhibits a sense of happiness and relaxation. His facial expressions and body language convey satisfaction with the exercise, indicating a positive attitude towards physical fitness and personal challenge. His face appears relaxed, with a smooth, uncreased forehead. His eyes are bright and focused, and a subtle smile on his lips. The character is dressed in workout gear that includes a black tank top, black shorts, and vibrant red sneakers. He hangs from the pull-up bar with fully extended arms, setting himself ready for the exercise. He skillfully performs hanging leg raises by bringing his knees toward his chest. Regrading interactions, he begins by approaching the pull-up bar, gripping it with both hands to ensure a secure hold. Once ready, he lifts himself into position, hanging with fully extended arms. Throughout the exercise, his movements are fluid and controlled, indicating a practiced routine.

{"Face-related": [ { "emotion": { "question": "What emotion is the character in the video expressing?", "answer": " Happiness and relaxation." } }, { "expression": { "question": "How does the character's expression convey his emotional state?", "answer": "His facial expressions show satisfaction with the exercise, indicating a positive attitude. "} }, { "appearance": { "question": "Describe the appearance of the character's face.", "answer": "The character's face is relaxed, with bright, focused eyes and a subtle smile." } } ],

"Body-related": [ { "appearance": { "question": "Describe the character's outfit.", "answer": "The character is dressed in workout gear that includes a black tank top, black shorts, and vibrant red sneakers." } }, { "pose": { "question": " Describe the character's pose.", "answer": "He hangs from the pull-up bar with fully extended arms, setting himself up." } }, { "action": { "question": "Describe the character's action.", "answer": "He skillfully performs hanging leg raises by bringing his knees toward his chest}],

"Interaction-related": [ { " Interaction ": { "question": "Describe the interactions between the character and the environment.", "answer": "The character interacts with the pull-up bar by approaching it and gripping it securely with both hands. He then lifts himself into a hanging position with fully extended arms. While hanging, he performs leg raises, bringing his knees to his chest, using the bar for leverage. His movements are fluid and controlled, indicating familiarity with the equipment. Throughout the exercise, his relaxed expression and subtle smile suggest a comfortable and positive engagement with t he environment." } } ] } Figure 
3
: Instruction Data Generation Process for face-related, body-related, and interaction-related branches. We generate structured instruction data by leveraging Qwen2.5 with specifically designed prompts to process the detailed captions we have previously obtained. videos based on the number of detected keyframes. We also removed clips with resolutions below 480P to enhance the overall quality of the data.

-Stage 2: Reducing Redundancy. To improve learning efficiency, we eliminated redundant video segments with similar contexts or meanings, reducing data redundancy as shown in the bottom of Fig. 
2
. Specifically, we used an advanced multimodal model, QWen2-VL-72B 
[44]
, to generate brief descriptions for each clip, focusing on the general contextual information. By calculating the semantic similarity between these brief descriptions using language model, we were able to filter out clips with high semantic similarity and repeated patterns.

-Stage 3: Fine-grained Annotations. For the remaining data, we generated detailed descriptions using the advanced multimodal model (QWen2-VL-72B) to maximize the utility of the data as shown on the right side of Fig. 
2
. Given the well-known issue of hallucination in large models, we implemented a dual verification method to eliminate hallucinations from detailed captions. Specifically, we used an additional multimodal large model 
[49]
 to generate multiple detailed descriptions for each clip. Given that the content of hallucinations is not related to the actual content of in the video, we hypothesize that hallucinations generated by different models are distinct. Hence, we heuristically employed a large language model (QWen2.5-72B 
[45]
) to summarize the common points across the different detailed descriptions, ensuring the accuracy of the final descriptions and

Happy minary Annotations: This video showcases the emotio anges of a young man with fair skin, short light brown ell-defined facial features, and a black top. e beginning of the video, his eyes are slightly squinted, relaxed eyelids, expressing a sense of ease and pleasure. e video progresses, the corners of his eyes gradually lift, is eyes widen, radiating a feeling of joy and excitement. yebrows also rise, particularly the right one, which lifts r, and the wrinkles between his brows deepen slightly, i ting a sense of surprise and happiness. outh transitions from a smile to a hearty laugh, with his lip lifting to reveal white teeth, and his jaw relaxing an ing downward. The corners of his mouth pull up notice and his entire facial muscles work together to convey th ful emotion, appearing very natural and relaxed. middle of the video, his head tilts slightly to the right a n returns to an upright position, seemingly expressing a of relaxation and ease. e video continues, his expression takes on a hint of play ss, with his upper lip lifting slightly, his lower lip gently inward, and his mouth closing slightly, forming a smil he can hardly suppress. The fine lines at the corners of es become more pronounced, further enhancing the joyf osphere. ly, his head tilts forward slightly, his gaze remains bright ull of joy, and his overall expression shifts from relaxed py, ultimately immersing him in a broad, joyful smile.

This vi deo shows an elderly man with gray hair that is short and neatly kept, wh ite eyebrows, and wrinkles on his fa ce and forehead. The man looks at th e camera with a smile, his mouth cor ners upturned, teeth showing, and ey ebrows raised.

Speaker Through the aforementioned steps, we collected 2.4M human-centric video clips with captions. We then utilize Qwen2.5-72B to generate structured data from detail captions for the pre-training of different branches, as illustrated in the Fig. 
3
. We systematically constructed instruction pairs for the face branch, body branch, and interaction branch by utilizing the structured data.

For the face-related branch, we filtered out videos that did not include descriptions of faces, emotions, or expressions. This resulted in a dataset of 1.78M videos. We then created detailed instruction pairs for facial features, emotions, and expressions, totaling 4.12M instruction pairs. These pairs were used to train the face-related branch.

For the body-related branch, we applied a similar method. We filtered out videos that lacked information on human appearance attributes, actions, or poses, leaving us with 2.21M videos. We then created specific instruction pairs for human appearance attributes, actions, and poses, resulting in a total of 5.75M instruction pairs. These pairs were used to train the body-related branch.

Finally, for the interaction-related module, we created specific instruction pairs for interactions with the external environment. Additionally, to ensure that our caption information was fully utilized, we incorporated detailed captions as instruction data in this module. This process resulted in a total of 4.8M instruction pairs, including 2.4M interaction instruction pairs and 2.4M detailed caption instruction pairs. These pairs were used to train the interaction-related module.

All of these instructions were used during the pre-training phase. The instructions come from captions that we double-checked, ensuring higher accuracy. By reasonably segmenting and filtering video clips, we also improved video quality, which helps the model better understand human-centric scenes.

Additionally, we manually annotated a subset of our human-centric video data by randomly sampling 50K video clips containing both vision and speech. These clips were annotated for emotion recognition, detailed facial expression descriptions, and speaker-specific speech annotations. Due to varying annotation difficulties, we completed emotion annotations for all 50K videos, detailed facial expression descriptions for 5K videos, and speaker-specific speech annotations for 20K videos, resulting in a total of 75K instruction pairs. The process is illustrated in Fig. 
4
. These annotated data were used in the fine-tuning of the visual component, further enhancing its feature extraction capabilities. Because these data include both visual and speech components and are manually annotated for high quality, they were also utilized in Cross-Modal Interaction Integration.

To build a multimodal large model capable of accurately understanding human-centric video information and possessing cross-modal interaction capabilities, our training strategy is divided into three stages. Initially, we focus on pretraining and fine-tuning the model's visual abilities using a substantial amount of human-centric video data, enabling the model to learn rich spatio-temporal feature representations and patterns of human behavior, thereby achieving a deep understanding of human-related details in video content. Next, we conduct standalone audio capability training with audio data, allowing the model to recognize and interpret speech. Finally, we perform cross-modal interaction training by integrating auditory and visual data, enhancing the model's ability to process and associate information across different modalities, ensuring it can provide accurate understanding and responses in complex multimedia environments.

Our model's visual component includes three specialized branches: face-related branch, body-related branch, and interaction-related branch. For each branch, we generated specific instruction data as described in the above section. This instruction data was used to pre-train each branch, during which only the projector parameters were updated. The aim was to keep all other parameters identical across the three branches to facilitate integration during fine-tuning.

During fine-tuning, we used manually annotated data consisting of 50K emotion recognition instructions and 5K facial expression description instructions, along with general Oryx 
[33]
 fine-tuning data. We integrated the three branches using an instruction-driven fusion module. In this process, we froze the parameters of the visual encoder and BERT, while training the parameters of the three projectors, the large language model, and two MLPs that generate the fusion weights.

During this phase, even though some of the videos contain both auditory and visual information, we only utilized the visual part.

In this stage, we aim to align the modalities between text and audio, enhancing the model's ability to understand and respond to audio in various contexts. We exclusively sample audio data from tasks such as automatic audio captioning, automatic speech recognition, and sound event classification, resulting in a total of approximately 18,000 hours of data used to train the audio projector.

Specifically, we utilize the WavCaps 
[35]
 dataset, which provides around 7,500 hours of annotated audio, offering detailed captions that describe the audio events. This dataset plays a crucial role in helping the model understand and generate descriptive audio analyses. We also select multiple comprehensive ASR datasets including WenetSpeech 
[59]
, GigaSpeech 
[6]
, CommonVoice15 
[4]
,

and LibriSpeech 
[39]
. These datasets cover extensive and diverse speech data, which are important in training models for speech recognition tasks. For SEC, the VGGSound 
[7]
 dataset is chosen due to its extensive collection of audio events. For the different tasks, we designed multiple question templates to prompt the model in generating captions, performing speech recognition, and classifying sounds, which in turn enables the model to thoroughly understand and process human-related audio information.

Task Type Datasets Duration (hours) AAC WavCaps 
[35]
 ~7.5k ASR WenetSpeech 
[59]
, GigaSpeech 
[6]
, CommonVoice15 
[4]
, LibriSpeech 
[39]
 ~10k SEC VGGSound 
[7]
 ~0.5k Table 
1
: Details of audio datasets for training audio projectors.

We use the encoder and the audio preprocessor from the Whisper-large-v3 
[40]
 as the audio encoder and processor. Specifically, We resample each audio to a frequency of 16kHz and convert the waveform into 128-channel mel-spectrogram using a window size of 25ms and a hop size of 10ms. To reduce the token length of the audio, we introduce an average pooling layer with a stride of 3, resulting in each audio frame from the audio encoder corresponding to a 60ms segment of the original audio. We use two linear layers to connect this to the LLM decoder. Additionally, we wrap each audio embedding with a pair of special tokens to indicate the start and end positions of the audio embedding.  

To enhance our model's video-audio interaction capabilities, we synthesized a series of visualauditory cross-modal interaction data. For audio data, we collected a diverse dataset covering various audio tasks, including samples from the audio pre-training phase, emotion recognition datasets, and audio question-answering datasets, totaling 7,000 hours of audio. For video data, we used all the aforementioned manually annotated 20K speaker-specific speech recognition data, as well as the instruction data used for visual fine-tuning. Additionally, we incorporate multi-modal emotion recognition datasets, converting classification labels with GPT-4o into a question-answer format, which includes DFEW 
[19]
, MAFW 
[32]
, CAER 
[21]
, and FERV39k 
[48]
.

To better distinguish features from different modalities, we encapsulate the embeddings of audio and visual data using distinct special tokens. We initialize the visual projectors and LLM decoder with parameters obtained from the Visual Capability Construction phase, while the audio projector is initialized with parameters from the Auditory Capability Development phase. During this training phase, we jointly fine-tune the LLM decoder, all projectors and two multi-layer perceptrons (MLPs) that generate the fusion weights to optimize their performance in handling multi-modal inputs.

To ensure that our HumanOmni can understand both scenes that include visual and auditory information and those with only visual input, for each video that contains audio, we also generate a version without audio for training. The model determines which modality to use based on special tokens in the instructions. Additionally, if either the auditory or visual part is missing, we fill in with default tokens to ensure consistent and complete inputs. This design allows the model to maintain stable performance across different modality combinations.

We evaluated HumanOmni's ability to understand audio-visual inputs on several human-related tasks, such as emotion recognition, facial expression description, and action understanding. We also tested HumanOmni's performance on speech recognition using only audio inputs. Finally, we explored how different modalities affect model performance across these human-centric tasks.

Both DFEW and MAFW are video-clip-based datasets designed for Dynamic Facial Emotion Recognition task, with DFEW providing a 7-dimensional expression distribution vector and MAFW providing an 11-dimensional expression distribution vector for each video clip.  As shown in Tab. 2, while VLM methods possess broader capabilities, they still exhibit a performance gap compared to specialized methods in dynamic emotion recognition tasks. In this task, both video and audio information play crucial roles, which is where the HumanOmni model excels. Experimental results demonstrate that HumanOmni significantly outperforms existing video-language multimodal models, audio-language multimodal large models, recently proposed omni model and specialized methods in this field. Moreover, it also shows a clear advantage over recently proposed Omni models for emotion recognition.

Facial expressions refer to external features displayed through facial muscle movements, such as smiling or frowning, while emotions denote internal emotional states, such as happiness or sadness.

Although facial expressions are one way to convey emotions, not all expressions directly correspond to specific emotional states, and the same expression can represent different emotions in various contexts. In this evaluation, we utilized the recently proposed DFEC dataset for facial expression description and adopted the evaluation methods recommended by DFEC.

In Tab. 3, our experimental results show that the HumanOmni model with combined video and audio input not only outperforms other open-source models but also surpasses the FaceTrack-MM 
[64]
 method proposed in DFEC, achieving superior performance in facial expression description tasks.

MVBench is a comprehensive video understanding benchmark covering 20 tasks organized in the form of multiple-choice questions. From this extensive suite of challenges, we select a specialized benchmark focusing on human-related subtasks, demonstrating in Tab.  (FA). This refined selection aims to provide a focused evaluation framework for the nuanced aspects of human activity recognition within video content.

The experimental results show that on the MVBench dataset, HumanOmni significantly outperforms nearly all mainstream methods with the same parameter size, with the exception of a few methods that utilized the full MVBench dataset.

Speech recognition capability is a crucial component of human-computer interaction. To demonstrate the advantages of our approach within the domain of speech recognition, Tab. 5 presents results from four widely recognized benchmarks in this field: LibriSpeech 
[39]
, WenetSpeech 
[59]
, and Fleurs 
[14]
. These benchmarks are specifically chosen for their distinct characteristics and contributions to evaluating speech recognition systems across different languages and contexts. LibriSpeech focuses on English speech recognition, while Fleurs is dedicated to evaluating cross-lingual speech representations. From the table, it can be seen that our method is leading among the current Omni models. However, compared to proprietary speech recognition approaches, current audio-visual methods can still be improved. 

Here we explored modalities efects on human-centric task performance. In Table 
6
, we evaluated HumanOmni's performance on emotion recognition, facial expression description, and action understanding under different input modalities. As expected, in the emotion recognition task, single-modal configurations (using only video or only audio) performed notably lower compared to the multi-modal configuration that used both video and audio inputs. For facial expression description, even when using only video input, the HumanOmni model maintained excellent performance, only slightly lower than with combined inputs. This is because facial expression recognition primarily relies on visual information, with limited added value from audio data. In the action understanding task, where actions are mainly represented by visual content, the contribution of audio was even more limited, as confirmed by our experimental results. These results demonstrate HumanOmni's robust performance across different input modalities. Additionally, they show that for all tasks, the combined visual-auditory input consistently achieved the best results, underscoring the necessity of joint audio and video inputs in human-centric scenes. 

In this work, we developed HumanOmni, the first human-centric multi-modal large language model. We constructed a dataset containing over 2.4 million human-centric video clips annotated with more than 14 million detailed captions and instructions to facilitate the understanding of diverse humancentered scenes. HumanOmni features a specialized architecture with three branches: a face-related branch, a body-related branch, and an interaction-related branch. Each branch addresses specific categories of human-centric scenes. By using user instructions to guide the adaptive fusion of features from these branches, HumanOmni significantly enhances its robustness across various scenarios. Additionally, HumanOmni supports joint audio and video input, enabling a more comprehensive understanding of scenes. We evaluated HumanOmni's performance through extensive experiments on multiple human-centric tasks, demonstrating its effectiveness in understanding complex humancentered interactions. To promote community-driven development and further research, we will open-source our code and model.