import json

def load_json(filepath):
    """Load JSON file into Python object."""
    print(f" Loading {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    
read_papers = load_json(r"data\4_read_papers\full_read.json")

gaps = ["video_segmentation", "frame_sampling_method", "input_video_length", "spatiotemporal_analysis", "visual_analysis", 
"speech_audio_analysis", "sound_audio_analysis", "qa_interaction", "retrieval_level", "inferring_method", 
"video_representation", "model_pipeline", "dataset", "comparisons", "hyperparameters", "environment", 
"repository", "authors"]
def gap_df(gaps):
    gap_answers = {}
    for gap in gaps:
        gap_answers[f"{gap}"] = ""
        for paper in read_papers:
            gap_answers[f"{gap}"] += f'In {paper.get("paper_file")}, {paper.get("extraction").get(gap).get("answer")} \n\n'
    return gap_answers

print(gap_df(gaps)["qa_interaction"])



