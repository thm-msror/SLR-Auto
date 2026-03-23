from __future__ import annotations

from typing import List, Optional

from atlas.results.prompts import SYNTHESIZE_CATEGORY_PROMPT
from atlas.utils.gpt_client import call_gpt_chat


DEFAULT_SYNTHESIS_PROMPT = (
    "You are an expert research synthesis assistant. "
    "Given a category label and a list of items (notes, findings, abstracts), "
    "write a concise synthesis of the category. Focus on common themes, "
    "methods, signals, and gaps. Use 1 short paragraph (4-8 sentences). "
    "Do NOT include bullets, numbering, or headings. "
    "Do NOT add caveats or meta commentary. Output only the synthesis text."
)


def synthesize_category(
    items: List[str],
    category_name: Optional[str] = None,
    prompt_text: str = SYNTHESIZE_CATEGORY_PROMPT,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 500,
) -> str:
    cleaned_items = [item.strip() for item in items if item and item.strip()]
    if not cleaned_items:
        raise ValueError("At least one non-empty item is required for synthesis.")

    system = (prompt_text or "").strip()
    if not system:
        system = DEFAULT_SYNTHESIS_PROMPT

    user = _build_synthesis_user_content(cleaned_items, category_name)

    return call_gpt_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )


def _build_synthesis_user_content(items: List[str], category_name: Optional[str]) -> str:
    lines = []
    if category_name:
        lines.append(f"Category: {category_name.strip()}")
    lines.append("Items:")
    for i, item in enumerate(items, start=1):
        lines.append(f"{i}. {item}")
    return "\n".join(lines)


def testCLI() -> None:
    sample_items = [
        """
        "GroundNLQ @ Ego4D Natural Language Queries Challenge 2023"
        The paper addresses video segmentation and representation by employing a two-stage pre-training strategy for egocentric video feature extractors and a grounding model. It uses pre-trained models like EgoVLP and InternVideo to extract video features and integrates these features into a text-aware video feature pyramid using a multi-scale transformer encoder. This approach enables the model to capture temporal intervals of varying lengths, facilitating effective segmentation of long-form videos.
        Supporting quotes:
        - We employ video feature extractors that have been pre-trained on Ego4D narrations, namely, InternVideo [3] and EgoVLP [2]." (§2.1)
        - The feature pyramid output is a combined result of all 6 layers’ outputs with the inputs. This produces 7 text-aware video feature sequence levels of varying lengths for moment prediction." (§2.2)
        - To address long videos, related works attempted to adapt literature models by sparse sampling [1, 4] or window-slicing [6]." (§1)
        - --
        """,


        """
        "REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding"
        The paper introduces REVISOR, a framework designed to enhance long-form video understanding by segmenting videos into critical intervals for detailed analysis. It employs a two-stage reasoning process: initial inference to identify key video segments and reflective reasoning to refine understanding using these segments. The Dual Attribution Decoupled Reward (DADR) mechanism ensures accurate segment selection by aligning reasoning with relevant video evidence.
        Supporting quotes:
        - REVISOR comprises an MLLM that collaborates with a visual toolbox. In Stage 1, the MLLM performs an initial reasoning step, identifies video segments requiring further examination, and invokes the visual toolbox to resample key frames from these segments as supplementary inputs." (Sec. 3.1)
        - The segment S = [tstart, tend] specifies the start and end timestamps of the video interval that warrants closer examination." (Sec. 3.1)
        - The CSSR enforces causal alignment between the model’s reasoning and the selected video evidence, rewarding correctness only when the answer is derived exclusively from those segments." (Sec. 3.2)
        """,


        """
        "Toward Scalable Video Narration: A Training-Free Approach Using Multimodal Large Language Models"
        The paper introduces VideoNarrator, a training-free pipeline that segments videos into uniform temporal chunks (e.g., every 10 seconds) and generates dense captions for each segment. This segmentation approach ensures precise temporal boundaries for each caption, enabling structured video representation. The modular design of VideoNarrator allows integration of off-the-shelf multimodal large language models (MLLMs) and visual-language models (VLMs) to enhance the quality of segment-level narrations.
        Supporting quotes:
        - Videos are segmented into chunks with uniform intervals (i.e., S seconds), and the MLLM generates the caption for each segment individually." (Page 2, Figure 2)
        - This strategy naturally yields captions with accurate and readily available temporal boundaries, where each caption spans from t(i)st to t(i)end = t(i)st + S." (Page 3)    
        - The modular design offers seamless integration of off-the-shelf MLLMs and VLMs, each fulfilling specialized roles, thereby contributing to generate more reliable and relevant video narrations." (Page 4)
        """,


        """
        "Zero-Shot Long-Form Video Understanding through Screenplay"
        The paper introduces MM-Screenplayer, which organizes video content into higher-level semantic scenes rather than individual shots, addressing the limitations of previous storytelling methods that overlooked temporal relationships. This approach enables a deeper comprehension of the video’s narrative by merging shots into coherent scenes using a Scene-Level Scripts Generation module.
        Supporting quotes:
        - Unlike previous storytelling methods, we organize video content into scenes as the basic unit, rather than just visually continuous shots." (Abstract)
        - The concept of scene is fundamental in the design of screenplays, as it provides a higher-level semantic decision of the video." (Section 2.2)
        - For example, in the film Titanic (1997), the sequence leading up to the ship’s collision with the iceberg consists of numerous quickly shifting shots. If these shots are viewed in isolation rather than treated as part of a cohesive scene, the true narrative is lost." (Section 2.2)
        - --
        """
    ]

    synthesis = synthesize_category(
        items=sample_items,
        category_name="Video Segmentation and Representation",
    )

    print("\nSynthesis:\n")
    print(synthesis)


if __name__ == "__main__":
    testCLI()
