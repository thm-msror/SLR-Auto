import re
import json

trial =  "{\n\"review_paper\": false,\n\"inclusion_exclusion_criteria\": {\n  \"Video retrieval/QA/semantic search\": 4,\n  \"Uses CV (detection, action recog, scene understanding)\": 2,\n  \"Uses Audio/ASR (speech, audio-visual events)\": 0,\n  \"Uses NLP/LLM for query or answers\": 2,\n  \"Has system/experiment with real video data\": 4,\n  \"Supports natural-language/semantic queries\": 3,\n  \"Multimodal fusion (vision+audio+NLP)\": 2,\n  \"Explicit semantic or multimodal retrieval\": 4,\n  \"NOT Metadata/keyword-only search\": 4,\n  \"NOT purely theoretical\": 4,\n  \"NOT focused on compression/storage/non-retrieval\": 4,\n  \"NOT in full English text\": 4,\n  \"NOT a pure review with no new empirical work\": 4,\n},\n\"task_type\": \"clip retrieval\",\n\"modalities\": [\"video\"],\n\"key_technologies\": [\"Query-By-Example (QBE), Video Ontology\"],\n\"datasets\": [\"TRECVID 2009\"],\n\"application\": [],\n\"limitations\": [],\n\"notes\": \"Proposes using a video ontology to improve QBE retrieval by filtering out irrelevant shots.\",\n\"reason_of_relevance\": \"Surely there according to inclusion criteria due to its focus on video retrieval and use of computer vision.\"\n}"

def strip_json_comments(json_with_comments: str) -> str:
    cleaned_lines = []
    for line in json_with_comments.splitlines():
        # Remove anything after '#' unless it's inside a string (basic check)
        quote_open = False
        clean_line = ""
        for i, char in enumerate(line):
            if char == '"' and (i == 0 or line[i - 1] != '\\'):  # Toggle on unescaped "
                quote_open = not quote_open
            if char == '#' and not quote_open:
                break  # Comment found outside string
            clean_line += char
        cleaned_lines.append(clean_line.rstrip())
    cleaned_lines = "\n".join(cleaned_lines)
    
    return re.sub(r',(\s*[}\]])', r'\1', cleaned_lines) #remove trailing commas

parsed_output = json.loads(strip_json_comments(trial))
print(parsed_output)
