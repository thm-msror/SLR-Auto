from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.utils.gpt_client import call_gpt_chat


DEFAULT_DISCUSSION_CONCLUSION_PROMPT = (
    "You are an expert academic writing assistant. "
    "Given an introduction and results section from a systematic literature review, "
    "write exactly two short paragraphs. "
    "The first paragraph must be a discussion paragraph focused on research gaps, challenges, and future research directions. "
    "The second paragraph must be a conclusion paragraph for the SLR. "
    "Output only the two paragraphs with no headings, bullets, numbering, labels, or markdown."
)

DEFAULT_INTRODUCTION = (
    "Graph Retrieval-Augmented Generation (GraphRAG) has emerged as a promising paradigm for integrating "
    "structured knowledge graphs into generative AI systems, enabling more accurate and context-aware outputs. "
    "Despite its potential, the implementation of GraphRAG varies widely, with different approaches to clustering, "
    "subgraph identification, and community detection playing a critical role in optimizing context aggregation. "
    "These methods are particularly important for improving the scalability of GraphRAG systems when applied to "
    "large-scale knowledge graphs, where computational efficiency and data complexity pose significant challenges. "
    "Furthermore, the evaluation of GraphRAG systems relies on diverse datasets and benchmarks, yet standardization "
    "in these metrics remains limited, complicating cross-comparisons. Addressing the limitations of current "
    "GraphRAG approaches requires a deeper understanding of their algorithmic foundations, scalability techniques, "
    "and evaluation frameworks, which collectively shape their effectiveness in real-world applications."
)

DEFAULT_RESULTS = (
    "The exploration of GraphRAG implementations reveals a diverse array of frameworks, architectures, and "
    "methodologies tailored to enhance the capabilities of retrieval-augmented generation systems. These "
    "advancements include hierarchical retrieval mechanisms, community embedding strategies, and domain-specific "
    "adaptations, which collectively aim to improve the precision and relevance of information retrieval processes "
    "[1], [3]. For instance, hierarchical retrieval approaches have been shown to refine the granularity of "
    "retrieved data, while community embedding techniques enable the system to capture nuanced relationships "
    "within knowledge graphs, thereby facilitating reasoning-intensive tasks [3], [4]. These innovations "
    "underscore the versatility of GraphRAG systems in adapting to varied application contexts.\n\n"
    "Transitioning to clustering and subgraph identification, the focus shifts to methodologies that enable the "
    "efficient segmentation of knowledge graphs into meaningful structures. Techniques such as the Louvain "
    "algorithm, multi-level pruning, and hierarchical retrievers have emerged as pivotal tools for identifying "
    "communities and subgraphs within large-scale graphs [16], [47]. These methods not only enhance retrieval "
    "efficiency but also support reasoning processes by isolating relevant graph segments for targeted analysis. "
    "By leveraging hierarchical structures, GraphRAG systems can achieve improved computational performance and "
    "more accurate retrieval outcomes [16].\n\n"
    "Building on the concept of hierarchical graph structures, community detection and context aggregation further "
    "refine the contextual understanding within GraphRAG systems. Community detection algorithms, coupled with "
    "hierarchical graph representations, enable multi-hop reasoning and enhance the relevance of retrieved "
    "information [3], [15]. These approaches facilitate the aggregation of contextually rich data, which is "
    "particularly beneficial for applications requiring deep semantic understanding or complex reasoning [15]. "
    "The integration of these techniques into GraphRAG systems demonstrates their potential to bridge gaps in "
    "contextual comprehension and retrieval precision.\n\n"
    "Scalability techniques for large knowledge graphs represent another critical area of development. Methods such "
    "as dynamic pruning, hybrid indexing, incremental fine-tuning, and efficient graph traversal have been "
    "introduced to optimize GraphRAG systems for handling extensive datasets [14], [16]. These techniques address "
    "the computational challenges associated with large-scale knowledge graphs, ensuring that retrieval processes "
    "remain efficient and cost-effective. By reducing the computational overhead, these scalability strategies "
    "enable GraphRAG systems to maintain high performance even when applied to expansive and complex graph structures "
    "[14].\n\n"
    "To evaluate the efficacy of these systems, datasets and benchmarks play a pivotal role. Resources such as the "
    "RGB corpus, Wikidata5M, QALD-h, and domain-specific datasets like EmoryNLP or Cail2022 provide standardized "
    "platforms for testing GraphRAG implementations [17], [33]. Metrics such as Hits@k, F1 scores, and Rouge are "
    "commonly employed to assess retrieval accuracy, reasoning capabilities, and overall system performance [33]. "
    "These benchmarks not only validate the effectiveness of GraphRAG systems but also facilitate comparative "
    "analyses across different implementations, driving further innovation in the field.\n\n"
    "Despite these advancements, limitations persist in current GraphRAG approaches. Challenges such as high "
    "computational costs, hallucinations, semantic redundancy, fragmented retrieval, and difficulties in handling "
    "complex relationships or domain-specific data remain significant obstacles [33], [46]. These issues highlight "
    "the need for continued research and development to address the inherent complexities of knowledge graph-based "
    "retrieval systems. Efforts to mitigate these limitations are crucial for ensuring the reliability and "
    "scalability of GraphRAG systems across diverse applications [46].\n\n"
    "Finally, the domain-specific applications of GraphRAG underscore its transformative potential in specialized "
    "fields such as healthcare [39]. By leveraging the contextual richness of knowledge graphs, GraphRAG systems "
    "have been employed to enhance medical diagnostics, drug interaction analysis, and semantic data interpretation "
    "[11], [39]. These applications demonstrate the adaptability of GraphRAG systems to meet the unique demands of "
    "various domains, paving the way for innovative solutions in areas ranging from legal knowledge graph completion "
    "to intelligent tutoring systems [40], [42]. The breadth of these applications highlights the growing "
    "importance of GraphRAG in addressing complex, domain-specific challenges."
)


def generate_discussion_conclusion(
    introduction: str,
    results: str,
    prompt_text: str = DEFAULT_DISCUSSION_CONCLUSION_PROMPT,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 700,
) -> Dict[str, str]:
    cleaned_intro = (introduction or "").strip()
    cleaned_results = (results or "").strip()
    if not cleaned_intro:
        raise ValueError("introduction is required.")
    if not cleaned_results:
        raise ValueError("results is required.")

    raw = call_gpt_chat(
        messages=[
            {"role": "system", "content": (prompt_text or "").strip() or DEFAULT_DISCUSSION_CONCLUSION_PROMPT},
            {
                "role": "user",
                "content": (
                    "Introduction:\n"
                    f"{cleaned_intro}\n\n"
                    "Results:\n"
                    f"{cleaned_results}\n\n"
                    "Write exactly two short paragraphs."
                ),
            },
        ],
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    discussion, conclusion = parse_discussion_conclusion(raw)
    return {
        "discussion": discussion,
        "conclusion": conclusion,
        "raw": raw.strip(),
    }


def parse_discussion_conclusion(text: str) -> tuple[str, str]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"```[a-zA-Z]*\n?", "", cleaned)
    cleaned = cleaned.replace("```", "").strip()

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", cleaned) if part.strip()]
    if len(paragraphs) == 2:
        return paragraphs[0], paragraphs[1]
    if len(paragraphs) > 2:
        return paragraphs[0], " ".join(paragraphs[1:]).strip()
    if len(paragraphs) == 1:
        return _split_single_paragraph(paragraphs[0])
    raise ValueError("Empty discussion/conclusion response.")


def _split_single_paragraph(text: str) -> tuple[str, str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if len(sentences) < 2:
        raise ValueError("Could not split response into discussion and conclusion.")

    midpoint = max(1, len(sentences) // 2)
    discussion = " ".join(sentences[:midpoint]).strip()
    conclusion = " ".join(sentences[midpoint:]).strip()
    if not discussion or not conclusion:
        raise ValueError("Could not split response into discussion and conclusion.")
    return discussion, conclusion


def testCLI() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    output = generate_discussion_conclusion(
        introduction=DEFAULT_INTRODUCTION,
        results=DEFAULT_RESULTS,
    )
    print(output["discussion"])
    print("")
    print(output["conclusion"])


if __name__ == "__main__":
    testCLI()
