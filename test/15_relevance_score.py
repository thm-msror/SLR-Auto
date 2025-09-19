def relevance_checker(paper_data):
    """
    Adds a 'relevance' score to the given JSON-like dictionary based on the
    sum of values in the 'inclusion_exclusion_criteria' section.
    """
    criteria = paper_data.get("llm_screening", {}).get("inclusion_exclusion_criteria", {})
    relevance_score = sum(criteria.values())

    # Add the relevance score to the JSON
    paper_data["llm_screening"]["relevance"] = relevance_score
    return paper_data


input_data = {
    "llm_screening": {
        "review_paper": False,
        "inclusion_exclusion_criteria": {
            "Video retrieval/QA/semantic search": 5,
            "Uses CV (detection, action recog, scene understanding)": 3,
            "Uses Audio/ASR (speech, audio-visual events)": 0,
            "Uses NLP/LLM for query or answers": 4,
            "Has system/experiment with real video data": 5,
            "Supports natural-language/semantic queries": 5,
            "Multimodal fusion (vision+audio+NLP)": 3,
            "Explicit semantic or multimodal retrieval": 5,
            "NOT Metadata/keyword-only search": 5,
            "NOT purely theoretical": 5,
            "NOT focused on compression/storage/non-retrieval": 5,
            "NOT in full English text": 5,
            "NOT a pure review with no new empirical work": 5
        },
        "reason_of_relevance": "This paper focuses on video retrieval using natural language queries and has conducted experiments on real video data.",
        "task_type": "clip retrieval",
        "modalities": ["video", "text"],
        "key_technologies": ["spatial preposition grounding", "geometric feature analysis"],
        "datasets": [],
        "application": ["surveillance video search"],
        "limitations": [],
        "notes": "The paper proposes a method to ground spatial prepositions in geometric features for video retrieval via natural language queries."
    }
}

# Run the function
result = relevance_checker(input_data)

# Print result
from pprint import pprint
pprint(result)  # Output: 55
