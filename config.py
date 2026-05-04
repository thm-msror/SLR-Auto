# config.py

# ----------------- Extraction Categories -----------------
FIXED_CATEGORIES = {
    "Video length + Processing Time (per Video)": "Maximum video duration handled and the average time taken to process each video.",
    "Video Segmentation to Clips": "Methods used to divide the video into smaller, searchable clips or segments.",
    "Frame Sampling Method": "Strategy for selecting frames from the video (e.g., fixed rate, keyframes, adaptive).",
    "Spatio-temporal Analysis (motion + frame interaction)": "Techniques for analyzing motion and temporal dependencies between frames.",
    "Only does visual analysis (plus text user query)": "Whether the system relies solely on visual features and text queries.",
    "Does Video+Audio Analysis (speech and background?)": "Whether the system integrates both visual and auditory data (speech, background noise, cues).",
    "Retrieval Level (video, clip, etc)": "The granularity of the search results (e.g., entire video, specific clip, single frame).",
    "Has Interactive QA Features (back and forth with chatbot)": "Presence of a conversational interface or interactive feedback loop.",
    "Model Generalization/Guessing Strategy": "How the model infers context or 'guesses' (e.g., VLM knowing a person is driving based on surroundings)."
}

# ----------------- Research Questions -----------------
RESEARCH_QUESTIONS = """
How effectively can an integrated pipeline, combining LLM query interpretation, multimodal tagging, and retrieval, support semantic clip search across long-form video archives?
What indexing and retrieval strategies ensure accurate, efficient, and scalable clip retrieval in large video archives?
To what extent can an LLM generate meaningful, context-aware answers from retrieved clips to enhance user understanding and usability?
To what extent do existing long-form video retrieval systems utilize the integration of four distinct modalities (video-text captioning, object detection, transcription, and audio cues) for semantic search?
"""

# ----------------- PICOC Components -----------------
# Population: Long-form video content across education, sports, security, etc.
# Intervention: object detection, action recognition, ASR, NLP, LLMs
# Comparison: Traditional metadata/timestamp-based search or keyword-only video retrieval.
# Outcome: Methods enabling semantic, natural query-based clip retrieval, scene summarization, and improved retrieval accuracy.
# Context: Applied AI research in computer vision, multimedia information retrieval, and multimodal systems

# ----------------- Search Terms -----------------
VIDEO_TERMS = [
    "long-form video retrieval", "long video content analysis", "video RAG", 
    "multimodal video indexing", "video clip retrieval", "video archive search",
    "extended video understanding", "video narrative retrieval"
]
METHOD_TERMS = [
    "audio-visual fusion", "VLM", "vision-language models", "audio cues",
    "environmental audio", "audio labeling", "sound event detection", 
    "audio-visual correspondence", "visual-audio alignment", "cross-modal grounding", 
    "multimodal video understanding", "video summarization", "synopsis generation",
    "system architecture", "end-to-end framework", "retrieval pipeline"
]
LANGUAGE_TERMS = [
    "retrieval-augmented generation", "RAG", "natural language query", 
    "semantic query", "LLM", "NLP", "multimodal tagging", "grounded retrieval"
]

# ----------------- Generated Search Queries -----------------
BOOLEAN_QUERY = '("long-form video retrieval" OR "video RAG") AND ("audio-visual fusion" OR "VLM") AND ("retrieval-augmented generation" OR "system architecture") AND ("clip retrieval" OR "multimodal indexing")'

QUERIES = [
    '"video RAG" "long-form video" "audio-visual fusion" "system"',
    '"video RAG" "multimodal video indexing" "LLM" "RAG" architecture',
    '"long-form video" "retrieval-augmented generation" "audio cues" "visual"',
    '"end-to-end framework" "long-form video retrieval" "multimodal"',
    '"video RAG" "semantic clip search" "VLM" "audio labeling"',
    '"long-form video archives" "semantic clip search" "LLM" "VLM" "multimodal"',
    '"long-form video archives" "synopsis generation" "summarization" "VLM"',
    '"long-form video archives" "environmental audio" "audio cues" "retrieval"',
    '"long-form video archives" "semantic clip search" "LLM query interpretation" "video-text captioning"',
    '"long-form video archives" "semantic clip search" "LLM query interpretation" "object detection"',
    '"long-form video archives" "semantic clip search" "LLM query interpretation" transcription',
    '"long-form video archives" "semantic clip search" "LLM query interpretation" "audio cues"',
    '"long-form video archives" "semantic clip search" "multimodal tagging" "video-text captioning"',
    '"long-form video archives" "semantic clip search" "multimodal tagging" "object detection"',
    '"long-form video archives" "semantic clip search" "multimodal tagging" transcription',
    '"long-form video archives" "semantic clip search" "multimodal tagging" "audio cues"',
    '"long-form video archives" "multimodal indexing" "LLM query interpretation" "video-text captioning"',
    '"long-form video archives" "multimodal indexing" "LLM query interpretation" "object detection"',
    '"long-form video archives" "multimodal indexing" "LLM query interpretation" transcription',
    '"long-form video archives" "multimodal indexing" "LLM query interpretation" "audio cues"',
    '"long-form video archives" "multimodal indexing" "multimodal tagging" "video-text captioning"',
    '"long-form video archives" "multimodal indexing" "multimodal tagging" "object detection"',
    '"long-form video archives" "multimodal indexing" "multimodal tagging" transcription',
    '"long-form video archives" "multimodal indexing" "multimodal tagging" "audio cues"',
    '"long-form video" "clip retrieval" "modalities integration" "video-text captioning"',
    '"long-form video" "clip retrieval" "modalities integration" "object detection"',
    '"long-form video" "clip retrieval" "modalities integration" transcription',
    '"long-form video" "clip retrieval" "modalities integration" "audio cues"',
    '"3-hour long videos" "semantic search" "multimodal retrieval" "video-text captioning"',
    '"3-hour long videos" "semantic search" "multimodal retrieval" "object detection"',
    '"3-hour long videos" "semantic search" "multimodal retrieval" transcription',
    '"3-hour long videos" "semantic search" "multimodal retrieval" "audio cues"',
]

# ----------------- Inclusion/Exclusion Criteria -----------------
CRITERIA = [
    "INCLUDE: Peer-reviewed research (journals, conferences)",
    "INCLUDE: Specifically targets long-form video archives (e.g., >1 hour, 3-hour long videos, or untrimmed archives)",
    "INCLUDE: Integrates at least 4 modalities (video-text/captioning, object detection, transcription/ASR, and audio cues) for retrieval",
    "INCLUDE: Implements or evaluates Computer Vision (detection, action recognition, scene understanding) for retrieval",
    "INCLUDE: Implements or evaluates Audio/ASR (speech recognition, audio-visual events) for retrieval",
    "INCLUDE: Implements or evaluates NLP/LLM for semantic query processing or answer generation",
    "INCLUDE: Presents an implemented system, experiment, or evaluation with real video data",
    "INCLUDE: Allows natural-language queries or retrieval by meaning (not just metadata)",
    "EXCLUDE: Studies relying solely on manual annotations, titles, or tags without video content analysis",
    "EXCLUDE: Purely theoretical papers without empirical implementation",
    "EXCLUDE: Focus only on video compression, storage, or non-retrieval analytics",
    "EXCLUDE: Focus on short-form clips only (e.g., 10-second web clips) without archival scalability",
    "EXCLUDE: Non-English publications"
]

# ----------------- Limits -----------------
MAX_QUERIES = 200
PER_QUERY_RESULTS = 50
TOP_N = 40 # Number of papers to fully read (selective)

# ----------------- Paths -----------------
RUNS_DIR = "data/runs"
