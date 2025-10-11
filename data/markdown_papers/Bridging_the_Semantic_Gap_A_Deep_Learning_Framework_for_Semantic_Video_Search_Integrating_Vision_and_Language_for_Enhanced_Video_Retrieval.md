# **Bridging the Semantic Gap: A Deep Learning Framework for Semantic Video Search Integrating Vision and Language for Enhanced Video Retrieval**

J ARJUN ANNAMALAI, Department of Computing Technologies, School of Computing, College of Engineering and Technology, SRM institute of Science and Technology, Kattankulathur, Chennai, India, 603203

S SHARON SNEHA, Department of Computing Technologies, School of Computing, College of Engineering and Technology, SRM institute of Science and Technology, Kattankulathur, Chennai, India, 603203

Dr. C. ASHOKKUMAR Department of Computing Technologies, School of Computing, College of Engineering and Technology, SRM institute of Science and Technology, Kattankulathur, Chennai, India, 603203

### **ABSTRACT:**

**Traditional video search engines often rely on tags or manual annotations for content retrieval, limiting the accuracy and efficiency of search results. Moreover, keyword-centric searches may not adeptly capture the nuanced and intricate queries users pose when seeking specific video content. The envisioned video search system integrates machine learning and natural language processing components to enable efficient and effective video retrieval based on user queries. The pipeline includes video processing employing Vision Transformer with GPT-2 (ViT-GPT2) architecture to analyze intricate details within video frames, Speech-to-Text (STT) models for transcribing spoken content, and the combination of transcriptions with video captions to form textual descriptions. These descriptions are then embedded using BERT, a transformer-based model, to grasp contextual relationships. Semantic matching is achieved through deep learning models, and a ranking mechanism based on similarity scores facilitates efficient retrieval. Leveraging datasets such as the MSR-VTT dataset, spanning visual recognition domains, contributes to the evaluation and training of the system's components. The system's efficacy is evaluated using metrics such as precision, recall, F1-score, and ranking metrics, with datasets spanning visual recognition, speech transcription, and semantic similarity domains. The comprehensive evaluation approach ensures the robustness and relevance of the video search system, aligning with user expectations and real-world use cases.** 

*Index Terms - Video search engine, Machine learning, Natural language processing, Vision Transformer with Generative Pre-trained. Transformer 2 (ViT-GPT2), Frame feature extraction, Speech-to-Text (STT) models, BERT (transformer-based model), Semantic matching* 

### **I. INTRODUCTION:**

In the era of digital content proliferation, the sheer volume of videos available online presents both an opportunity and a challenge for users seeking relevant and engaging content. Traditional video search engines often rely on metadata, tags, or manual annotations for content retrieval, limiting the accuracy and efficiency of search results. Additionally, keyword-based searches may not capture the nuanced and complex queries users have when searching for specific video content. This project embarks on the development of an Advanced Multi-Modal Video Retrieval System, propelled by state-of-the-art deep learning methodologies with a focus on semantic matching. Our goal is to bridge the semantic gap between user queries and video content, enabling a more intuitive and context-aware search experience.

The foundational layer of our system involves the extraction of rich visual features using Vision Transformer with GPT-2 (ViT-GPT2) architecture to analyze intricate details within video frames. Simultaneously, Speech-to-Text (STT) models transcribe spoken content, providing a textual representation of the auditory component. By fusing these modalities, our system generates comprehensive textual descriptions that encapsulate both visual and auditory elements.

To empower semantic understanding, we leverage BERT (Bidirectional Encoder Representations from Transformers), a transformer-based model renowned for its contextual language understanding capabilities [15]. BERT transforms the textual descriptions into embeddings that capture nuanced relationships and context, forming a powerful foundation for subsequent semantic matching. Further, BERT embeddings facilitate a fine-grained understanding of contextual semantics, enhancing the precision and relevance of the semantic matching process.

### **II. LITERATURE REVIEW:**

Recent advancements in multimedia research have showcased innovative methodologies across various tasks. For instance, the Spatiotemporal Vision Transformer (STVT) addresses the need for efficient video summarization by integrating inter-frame and intra-frame attention mechanisms [1]. Similarly, the Interpretable Spatial-Temporal Video Transformer (ISTVT) focuses on robust Deepfake detection through decomposed spatial-temporal self-attention and interpretability [2]. Meanwhile, a transformer-based approach for sentiment recognition in online social networks enhances sentiment learning performance by fusing features from different types of images [3]. Moreover, a multi-scale convolution and vision transformer method for deepfake detection demonstrates superior performance across datasets of varying quality [4]. Furthermore, a system employing colearning methods for multimodal fusion overcomes performance limitations and data imbalances among modalities [5]. These studies collectively underline the importance of advanced techniques such as transformers, attention mechanisms, and fusion strategies in addressing challenges across multimedia applications.

The comparative analysis and application of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) across diverse domains. In Digital Holography (DH), the determination of object distance for auto-focusing has been addressed using deep learning architectures, with ViT showing promise by drastically reducing the distance compared to conventional methods [6]. Adversarial robustness between ViT- and CNN-based models has been explored through novel metrics, revealing ViT-based models to exhibit higher robustness [7]. In remote sensing applications like wetland mapping, WetMapFormer combines CNNs and ViTs effectively, achieving precise mapping while mitigating computational costs [8]. Similarly, ViTs have demonstrated superior performance in classifying X-ray images for automated clinical diagnosis, outperforming CNN-based models [9]. Additionally, efforts have been made to bridge the gap between CNNs and ViTs, with DMFormer proposing a Dynamic Multi-level Attention mechanism to enhance the efficiency and performance of ViT-based architectures [10]. These studies collectively underscore the versatility and efficacy of ViTs rather than CNNs.

Recent research has explored diverse aspects of video processing, retrieval, and semantic understanding using advanced neural network architectures. For instance, the Semantic Grouping Network (SGN) for video captioning introduces a method that effectively groups video frames based on partially decoded captions, achieving state-of-theart performances in caption generation [11]. On the topic of video retrieval, a shift towards semantic similarity-based methods is proposed to enhance the relevance of retrieved videos/captions to queries, with implications for large-scale retrieval datasets [12]. Additionally, advancements in video moment retrieval are highlighted by a cross-modal neural architecture search approach, which automates the design of architectures for complex cross-modal matching tasks, leading to significant performance improvements [13]. Furthermore, Semantics-Aware Spatial-Temporal Binaries (S2 Bin) for cross-modal video retrieval offers a novel framework that considers both spatial-temporal context and semantic relationships, demonstrating superior performance compared to existing methods [14]. In the realm of natural language processing, SGPT and Sentence-BERT models are introduced for semantic search tasks, showcasing substantial improvements in sentence embeddings and information retrieval over traditional keyword-based searches [15]. These studies collectively highlight the growing importance of semantic understanding and advanced neural network architectures in enhancing various aspects of video processing and information retrieval tasks.

# **III. DATASET:**

# **Microsoft Research Video-to-Text (MSR-VTT) dataset:**

The MSR-VTT dataset [16] stands as a cornerstone in the advancement of video understanding models, providing a rich and diverse collection of over 10,000 video clips sourced from a variety of media forms, including movies, TV shows, and user-generated content. This dataset is accompanied by an extensive corpus of human-annotated captions, averaging 20 per video clip and totalling over 200,000 captions. These captions offer detailed semantic descriptions of the visual content, providing invaluable context for training and evaluating video understanding models. With its vast and meticulously curated collection, the MSR-VTT dataset serves as an essential benchmark for assessing the performance of models in video understanding tasks.

### **IV. PROPOSED METHODOLOGY:**

Our Multi-Modal Video Retrieval System incorporates deep learning methodologies, particularly leveraging BERT for semantic matching, to redefine the landscape of video search. The methodology unfolds in several key steps, ensuring a comprehensive and effective approach to multi-modal content understanding.

![](_page_2_Figure_2.jpeg)

*Figure 1: Architecture Diagram* 

### **A. Video Pre-processing:**

# • Frame Extraction:

we utilize OpenCV, a computer vision library in Python, to sequentially read frames from each video. By defining a frame rate parameter, we extract frames at regular intervals, enabling temporal analysis of the visual content. These extracted frames serve as the foundational input for subsequent visual feature extraction. Each frame encapsulates unique visual information, and collectively they provide a comprehensive representation of the video content.

# • Vision Transformer (ViT) Encoder:

The Vision Transformer (ViT) is a deep learning model primarily designed for computer vision tasks, such as image classification, object detection, and image captioning [14]. As an encoder, ViT processes the individual frames extracted from the video. ViT divides each frame into fixedsize non-overlapping patches and flattens them into sequences. These patches serve as the input tokens for the transformer architecture. The patches are then passed through multiple transformer layers, which consist of self-attention mechanisms and feed-forward neural networks [14]. The selfattention mechanism allows ViT to capture relationships between different patches, enabling it to understand spatial dependencies and contextual information within the frames. As the encoder, ViT transforms the visual input into a sequence of encoded representations, capturing high-level features and semantic information from the video frames.

# • Generative Pre-trained Transformer 2 (GPT-2) Decoder:

The encoded visual features from the Vision Transformer (ViT) are then concatenated with a special token denoting the beginning of the sequence, forming the input to the GPT-2 decoder [11]. Operating in an autoregressive manner, GPT-2 generates text sequentially by predicting the next token in the sequence based on previously generated tokens and the encoded visual features [11]. Each transformer layer within the GPT-2 decoder applies a self-attention mechanism to capture dependencies between different tokens, allowing the model to attend to relevant parts of the input sequence and learn contextual relationships between words. Following the self-attention mechanism, the output is passed through a feed-forward neural network (FFNN) to capture complex patterns and relationships in the data. The decoder then generates probabilities for each token in the vocabulary, representing the likelihood of that token being the next word in the sequence [11]. During training, the model selects the token with the highest probability as the next word, while during inference, it can sample from the probability distribution to introduce diversity in the generated captions.

![](_page_2_Figure_12.jpeg)

*Figure 2: ViT Encoder and GPT-2 Decoder* 

# • Caption Generation:

The frame-level captions are then temporally aligned, ensuring a seamless transition between consecutive frames and maintaining continuity in the video caption. Through contextual analysis and fusion techniques, the individual frame captions are aggregated into a cohesive narrative, capturing the essence of the entire video content comprehensively. Language modelling and refinement further enhance the coherence and readability of the aggregated caption, providing a concise and informative textual summary of the entire video content.

### **B. Audio Pre-processing:**

- Audio Transcription: We transcribed the audio tracks associated with each video into textual representations using the Google Speech-to-Text API. This allowed us to convert spoken words into text, which could then be processed alongside the visual data.
- Text Cleaning: Transcribed Text is cleaned and preprocessed by removing punctuation, special characters, and irrelevant information to ensure consistency and accuracy.

![](_page_3_Picture_3.jpeg)

*Figure 3: Video STT Transcription*

## **C. Multi-Modal Feature Extraction:**

Multi-modal feature extraction is the process of extracting features from multiple modalities, such as images and audio and combining these features to create a comprehensive representation of the input data. Concatenation can be a simple and effective technique for multi-modal feature extraction.

Table 1: Video Captions generated based on the MSR-VTT dataset

| Sample Video Frames | Captions                          |
|---------------------|-----------------------------------|
|                     | visual model: dogs are playing    |
|                     | audio model: barking noise        |
|                     | V+A(concatenating): dogs are      |
|                     | playing and barking in the garden |
|                     | visual model: a man               |
|                     | audio model: a man is talking     |
|                     | V+A(concatenating): a man is      |
|                     | talking in his phone              |

- Feature Vectors for Each Modality: The features extracted from each modality are represented as vectors. Let's denote these as V (visual) and A (auditory).
- Concatenation of Feature Vectors: Concatenation involves combining the feature vectors from different modalities into a single vector. Mathematically, the concatenated feature vector C is obtained as: C = [V, A]
- Unified Multi-Modal Representation: The concatenated feature vector C serves as a unified representation that contains information from all modalities. [V1, V2, ..., V\_n, A1, A2, ..., A\_m]

### **D. BERT-Based Textual Embeddings:**

![](_page_3_Picture_13.jpeg)

*Figure 4: BERT Embeddings*

The first step in BERT-based textual embedding generation is tokenization, where the user input query text is broken down into smaller units called tokens. As the tokenized input progresses through the layers of the BERT model, each token's contextual embeddings are generated by considering its surrounding tokens in a bidirectional manner. By leveraging self-attention mechanisms within the transformer architecture, BERT assigns different weights to each token based on its relevance to the entire input sequence, effectively capturing the semantic relationships between words and phrases.

After processing through the transformer layers, the contextual embeddings of all tokens are pooled together to obtain a fixed-dimensional representation of the entire input text. This pooling operation condenses the contextual information captured by BERT into a single vector, effectively summarizing the semantic content of the input query. The resulting embeddings encapsulate the nuanced semantic relationships within the textual content, providing a comprehensive representation of the input query's meaning. This step captures the nuanced semantic relationships within the textual content, forming a robust foundation for subsequent semantic matching.

### **E. Semantic Matching Model:**

![](_page_3_Picture_18.jpeg)

*Figure 5: Cosine Similarity* 

The user query and semantic index representations are passed through the BERT model to generate contextual embeddings for each token in the input text. After obtaining the embeddings, a pooling layer may be applied to aggregate the token-level embeddings into a fixed-size representation for the entire query or index.

With the pooled representations of the user query and semantic index obtained, cosine similarity is calculated between these representations. Cosine similarity is a metric used to measure the similarity between two vectors by computing the cosine of the angle between them. In the context of information retrieval, cosine similarity is often used to compare the similarity of document vectors or embedding representations. A higher cosine similarity score indicates greater similarity between the user query and the items in the semantic index.

### F. Fine-Tuning with Domain-Specific Data:

Fine-tuning with domain-specific data is an integral aspect of enhancing the adaptability and efficacy of the semantic matching models within the video retrieval system. This process involves retraining the models using additional labelled data that is specific to the domain of interest, thereby allowing the system to better understand the intricacies and nuances inherent in the target video content. By incorporating domain-specific annotations, user interactions, or curated datasets relevant to the domain, the models can learn to discern subtle patterns and relationships that are characteristic of the domain. This fine-tuning process serves to optimize the models' performance and relevance in retrieving videos based on user queries within the specified domain.

#### V. EXPERIMENTAL RESULTS AND ANALYSIS

We have achieved high accuracy in understanding and processing natural language queries and are capable of with context-aware handling complex queries comprehension. We successfully extracted meaningful visual features from videos using deep learning models. Efficiently handled video segmentation and scene detection tasks. Established a robust semantic mapping between textual descriptions and visual features. Achieved effective alignment in a unified embedding space, facilitating crossmodal retrieval. Integrated NLP, video analysis, and semantic mapping components seamlessly.

$$Precision = \frac{Number\ of\ relevant\ videos\ retrieved}{Total\ number\ of\ videos\ retrieved} = 0.6$$

So, the precision value is 60%. This means that out of all the videos retrieved by the system, 60% of them are relevant to the user's query.

$$Recall = \frac{Number of relevant videos retrieved}{Total number of relevant videos} = 0.7$$

The recall value is 70%. This indicates that the system successfully retrieved 70% of the relevant videos from the dataset.

$$F1 - Score = \frac{2 \times Precision \times Recall}{Precision + Recall} = 0.65$$

The F1-score is 0.65. This harmonic mean of precision and recall provides a balanced measure of the system's performance in retrieving relevant videos while minimizing false positives and false negatives.

The cosine similarity metric is used to measure the similarity between the user query and semantic index for video retrieval. The result of cosine similarity ranges from -1 to 1. A value closer to 1 indicates a higher similarity between the feature vectors representing two videos, suggesting that they are more alike in content.

![](_page_4_Figure_13.jpeg)

Figure 6: Cosine Similarity between User Query and Semantic index

![](_page_4_Figure_15.jpeg)

Figure 7: Semantic Search Results based on Cosine Similarity

High performance in processing natural language queries and retrieving relevant videos. Provided an intuitive and user-friendly interface for interacting with the system. Implemented features such as autocomplete suggestions and visual representations of search results. Developed an efficient indexing system and search engine for video

retrieval. Achieved fast and accurate retrieval of relevant videos based on user queries. It demonstrates significant advancements in video search technology, particularly in integrating natural language understanding with deep learning. The system achieves precise and efficient retrieval of videos based on user-generated queries, offering a seamless and intuitive search experience.

### **VI. Limitations:**

Lack of User Feedback Mechanisms: The system may lack robust mechanisms for capturing user feedback and preferences, limiting its ability to adapt and improve over time based on user interactions and feedback.

Domain-Specific Limitations: The system's effectiveness may vary across different content domains, as it may not be equally adept at handling diverse types of video content or user queries. Certain domains or niche content types may pose unique challenges that the system may struggle to address effectively.

Complex User Queries: The system may struggle to accurately interpret and respond to more intricate or ambiguous user queries. Complex queries with nuanced semantics, multiple intents, or vague descriptors may pose challenges in semantic understanding and representation.

Video Content Nuances: The system may encounter difficulties in capturing subtle nuances, context-dependent meanings, or cultural references embedded within the videos. These nuances, which contribute to the richness and complexity of video content, pose challenges in semantic understanding and representation, potentially leading to mismatches or misinterpretations in retrieval results.

### **VII. CONCLUSION AND FUTURE WORK**

In conclusion, the development of the advanced multi-modal video retrieval system represents a significant breakthrough in the field of multimedia information retrieval. By leveraging cutting-edge deep learning methodologies, such as Vision Transformer with GPT-2 architecture and BERT (semantic matching model), the system has demonstrated remarkable efficacy in bridging the semantic gap between user queries and video content [11, 15]. Through rigorous evaluation, the system has exhibited high precision, recall and F1-score, affirming its capability to accurately retrieve relevant videos based on user input.

The comprehensive analysis of experimental results underscores the system's robustness and effectiveness across diverse query types and content domains [15]. These findings not only validate the system's ability to provide users with precise and contextually relevant search results but also highlight its potential to revolutionize video retrieval technology. With its intuitive interface and seamless integration of advanced deep learning models, the system offers users a more efficient, accurate, and personalized means of accessing digital video content.

Looking ahead, there are several promising avenues for future research and development to further enhance the capabilities of the video retrieval system. One potential direction is the integration of additional modalities, such as text, tags, and metadata, to provide a more comprehensive understanding of video content and improve retrieval accuracy. Moreover, exploring advanced semantic matching techniques, including graph-based models and deep reinforcement learning, could further enhance the system's ability to understand complex user queries and video content nuances.

## **REFERENCES**

- [1] Walsh, Hannah S and Sequoia R. Andrade, "Semanc Search With Sentence-BERT for Design Informaon Retrieval.," in *Internaonal Design Engineering Technical Conferences and Computers and Informaon in Engineering Conference. Vol. 86212. American Society of Mechanical*, 2022.
- [2] Hsu, Tzu-Chun, Yi-Sheng Liao and Chun-Rong Huang, "Video summarizaon with spaotemporal vision transformer.," in *IEEE Transacons on Image Processing*, 2023.
- [3] Zhao, Cairong, Chuan Wang, Guosheng Hu, Haonan Chen, Chun Liu and Jinhui Tang, "ISTVT: interpretable spaal-temporal video transformer for deepfake detecon.," in *IEEE Transacons on Informaon Forensics and Security 18*, 2023.
- [4] Alzamzami, Famah and Abdulmotaleb El Saddik, "Transformer-based feature fusion approach for mulmodal visual senment recognion using tweets in the wild.," in *IEEE Access*, 2023.
- [5] Lin, Hao, Wenmin Huang, Weiqi Luo and Wei Lu, "DeepFake detecon with mul-scale convoluon and vision transformer.," in *Digital Signal Processing 134*, 2023.
- [6] Yoon, JunHo, GyuHo Choi and Chang Choi, "Mulmedia analysis of robustly opmized mulmodal transformer based on vision and language co-learning.," in *Informaon Fusion 100*, 2023.
- [7] Cuenat, Stéphane and Raphaël Couturier, "Convoluonal neural network (cnn) vs vision transformer (vit) for digital holography.," in *2nd Internaonal Conference on Computer, Control and Robocs (ICCCR). IEEE*, 2022.
- [8] Heo, Jaehyuk, Seungwan Seo and Pilsung Kang, "Exploring the differences in adversarial robustness between ViT-and CNN-based models using novel metrics.," in *Computer Vision and Image Understanding 235*, 2023.
- [9] Jamali, Ali, Swalpa Kumar Roy and Pedram Ghamisi, "WetMapFormer: A unified deep CNN and vision transformer for complex wetland mapping.," in *Internaonal Journal of Applied Earth Observaon and Geoinformaon 120*, 2023.
- [10] Uparkar, Om and et al., "Vision transformer outperforms deep convoluonal neural network-based model in classifying X-ray images.," in *Procedia Computer Science 218*, 2023.
- [11] Zimian Wei, Hengyue Pan and et al. , "DMFormer: Closing the gap Between CNN and Vision Transformers.," in *ICASSP 2023-2023 IEEE Internaonal Conference on Acouscs, Speech and Signal Processing (ICASSP)*, 2023.
- [12] Ryu, Hobin and et al., "Semanc grouping network for video caponing.," in *proceedings of the AAAI Conference on Arficial Intelligence.*, 2021.
- [13] Wray, Michael, Hazel Doughty and Dima Damen, "On semanc similarity in video retrieval.," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pa<ern Recognion*, 2021.
- [14] Yang, Xun and et al., "Video moment retrieval with cross-modal neural architecture search.," in *IEEE Transacons on Image Processing 31*, 2022.
- [15] Qi, Mengshi and et al., "Semancs-aware spaal-temporal binaries for cross-modal video retrieval.," in *EEE Transacons on Image Processing 30*, 2021.
- [16] Xu, Jun and et al., "MSR-VTT: A Large Video Descripon Dataset for Bridging Video and Language," in *IEEE Conference on Computer Vision and Pa<ern Recognion (CVPR)*, 2016.
- [17] Aggarwal, Akshay and et al., "Video capon based searching using end-to-end dense caponing and sentence embeddings.," in *Symmetry 12, no. 6*, 2020.
- [18] Shang, Xindi and et al., "Mulmodal video summarizaon via me-aware transformers.," in *Proceedings of the 29th ACM Internaonal Conference on Mulmedia*, 2021.
- [19] Lin, Zhijie and en al., "Weakly-supervised video moment retrieval via semanc compleon network.," in *Proceedings of the AAAI Conference on Arficial Intelligence. Vol. 34. No. 07.*, 2020.
- [20] Yousaf, Kanwal and Tabassam Nawaz, "A deep learning-based approach for inappropriate content detecon and classificaon of youtube videos.," in *IEEE Access 10*, 2022.