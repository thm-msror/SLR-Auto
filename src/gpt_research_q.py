from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from gpt_client import call_gpt_chat


Node = Tuple[str, object]


def term(value: str) -> Node:
    return ("TERM", value)


def not_(node: Node) -> Node:
    return ("NOT", node)


def and_(left: Node, right: Node) -> Node:
    return ("AND", left, right)


def or_(left: Node, right: Node) -> Node:
    return ("OR", left, right)


OPERATORS = {"AND", "OR", "NOT"}


def tokenize(query: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    n = len(query)
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue
        if ch == '"':
            i += 1
            start = i
            while i < n and query[i] != '"':
                i += 1
            phrase = query[start:i]
            tokens.append(f"\"{phrase}\"")
            i = i + 1 if i < n and query[i] == '"' else i
            continue
        # bare token
        start = i
        while i < n and (not query[i].isspace()) and query[i] not in "()":
            i += 1
        token = query[start:i]
        tokens.append(token)
    return tokens


def parse_boolean_query(query: str) -> Node:
    tokens = tokenize(query)
    pos = 0

    def peek() -> Optional[str]:
        return tokens[pos] if pos < len(tokens) else None

    def consume() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of query.")
        tok = tokens[pos]
        pos += 1
        return tok

    def parse_primary() -> Node:
        tok = peek()
        if tok is None:
            raise ValueError("Unexpected end of query.")
        if tok == "(":
            consume()
            node = parse_or()
            if peek() != ")":
                raise ValueError("Unmatched '(' in query.")
            consume()
            return node
        if tok.upper() in OPERATORS:
            raise ValueError(f"Unexpected operator '{tok}'.")
        consume()
        return term(tok)

    def parse_not() -> Node:
        tok = peek()
        if tok and tok.upper() == "NOT":
            consume()
            return not_(parse_not())
        return parse_primary()

    def parse_and() -> Node:
        node = parse_not()
        while True:
            tok = peek()
            if tok and tok.upper() == "AND":
                consume()
                node = and_(node, parse_not())
            else:
                break
        return node

    def parse_or() -> Node:
        node = parse_and()
        while True:
            tok = peek()
            if tok and tok.upper() == "OR":
                consume()
                node = or_(node, parse_and())
            else:
                break
        return node

    ast = parse_or()
    if pos != len(tokens):
        raise ValueError(f"Unexpected token '{tokens[pos]}'.")
    return ast


def to_nnf(node: Node) -> Node:
    kind = node[0]
    if kind == "TERM":
        return node
    if kind == "NOT":
        inner = node[1]
        if inner[0] == "TERM":
            return node
        if inner[0] == "NOT":
            return to_nnf(inner[1])
        if inner[0] == "AND":
            return or_(to_nnf(not_(inner[1])), to_nnf(not_(inner[2])))
        if inner[0] == "OR":
            return and_(to_nnf(not_(inner[1])), to_nnf(not_(inner[2])))
    if kind == "AND":
        return and_(to_nnf(node[1]), to_nnf(node[2]))
    if kind == "OR":
        return or_(to_nnf(node[1]), to_nnf(node[2]))
    raise TypeError(f"Unsupported node: {node}")


Clause = Tuple[Set[str], Set[str]]


def dnf_clauses(node: Node) -> List[Clause]:
    node = to_nnf(node)
    kind = node[0]
    if kind == "TERM":
        return [({node[1]}, set())]
    if kind == "NOT":
        inner = node[1]
        if inner[0] != "TERM":
            raise ValueError("NOT should only apply to terms after NNF.")
        return [(set(), {inner[1]})]
    if kind == "OR":
        return dnf_clauses(node[1]) + dnf_clauses(node[2])
    if kind == "AND":
        left = dnf_clauses(node[1])
        right = dnf_clauses(node[2])
        combined: List[Clause] = []
        for l_pos, l_neg in left:
            for r_pos, r_neg in right:
                pos = set(l_pos) | set(r_pos)
                neg = set(l_neg) | set(r_neg)
                if pos.intersection(neg):
                    continue
                combined.append((pos, neg))
        return combined
    raise TypeError(f"Unsupported node: {node}")


def clause_to_query(pos: Iterable[str], neg: Iterable[str]) -> str:
    parts: List[str] = []
    parts.extend(pos)
    parts.extend(f"-{t}" for t in neg)
    return " ".join(parts)


def boolean_to_queries(boolean_query: str, max_queries: int = 50) -> List[str]:
    ast = parse_boolean_query(boolean_query)
    clauses = dnf_clauses(ast)
    queries = [clause_to_query(pos, neg) for pos, neg in clauses]
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    if len(deduped) > max_queries:
        return deduped[:max_queries]
    return deduped


def extract_boolean_query(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise ValueError("Empty GPT response.")
    # Try JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("boolean_query", "query", "boolean"):
                if key in data and isinstance(data[key], str):
                    return data[key].strip()
        if isinstance(data, str):
            return data.strip()
    except json.JSONDecodeError:
        pass
    # Try fenced blocks
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            candidate = parts[1].strip()
            if candidate:
                return candidate.splitlines()[0].strip()
    # Try label
    for line in text.splitlines():
        if ":" in line:
            label, value = line.split(":", 1)
            if label.strip().lower() in {"boolean query", "boolean", "query"}:
                return value.strip()
    # Fallback: first line
    return text.splitlines()[0].strip()


def build_boolean_query_from_questions(questions_text: str) -> str:
    system = load_prompt("prompts/rq_query.txt").strip()
    if not system:
        system = (
            "You are a research assistant. Convert research questions into a single "
            "Boolean search query. Use AND/OR/NOT, group synonyms with OR, and quote "
            "phrases. Return ONLY the Boolean query without explanation."
        )
    user = f"Research questions:\n{questions_text}\n\nBoolean query:"
    raw = call_gpt_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return extract_boolean_query(raw)


def read_multiline_input(prompt: str) -> str:
    print(prompt)
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def load_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8")



def testCLI() -> None:
    questions = read_multiline_input(
        "Paste research questions end with empty line: "
        "\n(Press enter to select default question)"
    )
    if not questions:
        questions = DEFAULT_QUESTION
        print(f"...Using default question: {questions}")
    boolean_query = build_boolean_query_from_questions(questions)
    print("\nBoolean query:\n" + boolean_query)
    queries = boolean_to_queries(boolean_query, max_queries=50)
    print("\nExpanded queries:")
    for i, q in enumerate(queries):
        print(f" {i+1}.) {q}")


DEFAULT_QUESTION = (
    "How can AI systems efficiently index, retrieve, and semantically understand long-form video content at scale?",
    "What are the current techniques to retrieve a specific clip from a long video?"
)


if __name__ == "__main__":
    testCLI()
