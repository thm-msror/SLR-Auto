import re

def clean_paper_md(md):

    # Remove References 
    md = re.sub(r'(?i)^#\s*References\b.*?(?=^#|\Z)', '', md, flags=re.DOTALL | re.MULTILINE).strip()

    # Remove Markdown links like [something](#page-4-0)
    md = re.sub(r'\[.*?\]\(.*?page.*?\)', '', md).strip()
    
    # Remove HTML tags like <span> or </span>
    md = re.sub(r'<.*?>', '', md).strip()

    # Remove heading tag # 
    md = md.replace("#", "").strip()

    return md
