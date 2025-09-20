import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from src.utils import *

def get_relevant(paper_list, k: int) :
    cleaned_papers = []

    for item in paper_list:
        try:
            # Safely parse date
            published_str = item['paper']['published']
            published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))

            # Use 0 as default if relevance_score is missing
            relevance_score = item.get('relevance_score', 0)

            # Add a temp field for sorting
            item['__sort_date'] = published_dt
            item['__sort_score'] = relevance_score

            cleaned_papers.append(item)
        except (KeyError, TypeError, ValueError):
            # Skip items with missing/invalid data
            continue

    # Sort by relevance_score (desc), then date (desc)
    sorted_papers = sorted(
        cleaned_papers,
        key=lambda x: (-x['__sort_score'], -x['__sort_date'].timestamp())
    )

    # Clean up helper fields before returning
    for item in sorted_papers:
        item.pop('__sort_date', None)
        item.pop('__sort_score', None)

    return sorted_papers[:k]

def paper_table(papers):
    """
    Generate a Markdown table summarizing papers.
    
    Args:
        papers (list[dict]): List of paper objects (with 'paper' and 'llm_screening' keys).
    
    Returns:
        str: Markdown table as a string.
    """
    def list_to_text(str):
        return ", ".join(list(str))
    
    # Define table header
    header = (
        "| Title | Authors | Links | Citations / Refs | Notes | "
        "Relevance (Score + Reason) | Key Tech | Datasets | Application | Limitations |\n"
        "|-------|---------|-------|------------------|-------|"
        "---------------------------|----------|----------|-------------|-------------|"
    )
    
    rows = []
    for item in papers:
        paper = item.get("paper", {})
        screen = item.get("llm_screening", {})

        relevance_score = item.get("relevance_score", 0)
        reason_of_relevance = screen.get("reason_of_relevance", "")
        
        # publication year
        published = paper.get("published", "")
        year = published[:4] if published else ""
        link = paper.get("link", "")
        doi = paper.get("doi", "DOI")
        openalex_id = paper.get("openalex_id", "")
        openalex_lookup = paper.get("openalex_lookup", "")
        link_md = f"[DOI]({link}) [OpenAlex]({openalex_id})" 

        # citations & refs
        citations = paper.get("citations", "")
        ref_count = paper.get("reference_count", "")
        citeref = f"{citations} / {ref_count}"

        # extracted info
  
        key_tech = list_to_text(screen.get("key_technologies", ""))
        datasets = list_to_text(screen.get("datasets", ""))
        application = list_to_text(screen.get("application", ""))
        limitations = list_to_text(screen.get("limitations", ""))
        notes = screen.get("general_notes", "")

        rows.append(
            f"| {paper.get('title','')} "
            f"| {year}, {paper.get('authors','')[:100]} "
            f"| {link_md}"
            f"| {citeref} "
            f"| {notes} "
            f"| {relevance_score}: {reason_of_relevance} "
            f"| {key_tech} "
            f"| {datasets} "
            f"| {application} "
            f"| {limitations} |"
        )

    return header + "\n" + "\n".join(rows)


data = load_json("data/screened_articles/screened_6702_2025-09-20T06-19-06.json")

save_md(paper_table(get_relevant(data, 60)))

