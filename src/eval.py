"""
Evaluation harness for Naive RAG. Runs every Q&A pair in eval/qa_pairs.json
through the real pipeline (query.py) and scores it two ways:

- Answerable questions: three LLM-as-judge quality scores (faithfulness,
  context relevance, answer relevance), replicating RAGAS's core metrics
  without the framework itself. ragas 0.4.3 turned out to have a broken
  dependency chain (it hard-imports Google Vertex AI support that no longer
  exists in langchain-community, which is itself now deprecated/sunset) —
  not worth fighting a version-pin battle for a heavy dependency tree when
  these three metrics are each just "ask an LLM to rate X given Y."
- Unanswerable questions (no answer exists in the corpus): checked for
  correct refusal instead — these exercise the strict-grounding prompt's
  negative-rejection behavior, not answer quality.

Every question also gets latency and tokens/sec, regardless of answerability.

    .venv/Scripts/python.exe src/eval.py [path/to/qa_pairs.json]

# ponytail: judge scores come from the same local model that generated the
# answers (self-judging) — a known limitation, not an oversight. Treat
# scores as directional (did this change help or hurt?) rather than
# absolute. Upgrade path: score with a separate/stronger judge model if
# absolute scores ever need to be trusted on their own.
"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator.generator import generate
from src.query import query

DEFAULT_QA_PATH = "eval/qa_pairs.json"
RESULTS_DIR = "eval"

_FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated answer is faithful to the given context — every claim in the answer must be supported by the context, with no invented or outside information.

Context:
{context}

Answer:
{answer}

Rate faithfulness from 0.0 to 1.0 (1.0 = fully supported by the context, 0.0 = entirely unsupported/fabricated).
Respond with ONLY a number between 0.0 and 1.0."""

_CONTEXT_RELEVANCE_PROMPT = """You are evaluating whether retrieved context is relevant to a question.

Question: {question}

Context:
{context}

Rate context relevance from 0.0 to 1.0 (1.0 = the context directly addresses the question, 0.0 = completely unrelated).
Respond with ONLY a number between 0.0 and 1.0."""

_ANSWER_RELEVANCE_PROMPT = """You are evaluating whether an answer addresses the question asked (regardless of factual correctness).

Question: {question}

Answer: {answer}

Rate answer relevance from 0.0 to 1.0 (1.0 = directly and completely addresses the question, 0.0 = completely unrelated/off-topic).
Respond with ONLY a number between 0.0 and 1.0."""


def _judge_score(prompt: str) -> float:
    result = generate([{"role": "user", "content": prompt}])
    match = re.search(r"\d*\.?\d+", result.text)
    if not match:
        raise ValueError(f"judge didn't return a parseable score: {result.text!r}")
    return max(0.0, min(1.0, float(match.group())))


def faithfulness(context: str, answer: str) -> float:
    return _judge_score(_FAITHFULNESS_PROMPT.format(context=context, answer=answer))


def context_relevance(question: str, context: str) -> float:
    return _judge_score(_CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context))


def answer_relevance(question: str, answer: str) -> float:
    return _judge_score(_ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer))


def _percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * pct))]


def run(qa_path: str = DEFAULT_QA_PATH) -> dict:
    qa_pairs = json.loads(Path(qa_path).read_text(encoding="utf-8"))

    per_question = []
    for pair in qa_pairs:
        question = pair["question"]
        expect_answerable = pair.get("expect_answerable", True)

        start = time.perf_counter()
        result = query(question)
        latency = time.perf_counter() - start
        tokens_per_sec = result.completion_tokens / latency if latency > 0 else 0.0

        context = "\n\n".join(c.text for c in result.chunks)
        row = {
            "question": question,
            "answer": result.answer,
            "expect_answerable": expect_answerable,
            "latency_s": latency,
            "tokens_per_sec": tokens_per_sec,
        }

        if expect_answerable:
            row["faithfulness"] = faithfulness(context, result.answer)
            row["context_relevance"] = context_relevance(question, context)
            row["answer_relevance"] = answer_relevance(question, result.answer)
        else:
            row["correctly_rejected"] = "don't know" in result.answer.lower()

        per_question.append(row)
        tag = "OK" if expect_answerable else "REJECT"
        print(f"  [{tag}] {question[:65]}")

    return _summarize(per_question)


def _summarize(per_question: list[dict]) -> dict:
    answerable = [r for r in per_question if r["expect_answerable"]]
    unanswerable = [r for r in per_question if not r["expect_answerable"]]
    latencies = [r["latency_s"] for r in per_question]

    summary = {
        "n_questions": len(per_question),
        "avg_latency_s": sum(latencies) / len(latencies),
        "p95_latency_s": _percentile(latencies, 0.95),
        "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in per_question) / len(per_question),
    }
    if answerable:
        summary["avg_faithfulness"] = sum(r["faithfulness"] for r in answerable) / len(answerable)
        summary["avg_context_relevance"] = sum(r["context_relevance"] for r in answerable) / len(answerable)
        summary["avg_answer_relevance"] = sum(r["answer_relevance"] for r in answerable) / len(answerable)
    if unanswerable:
        summary["rejection_accuracy"] = sum(r["correctly_rejected"] for r in unanswerable) / len(unanswerable)

    return {"summary": summary, "per_question": per_question}


if __name__ == "__main__":
    qa_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QA_PATH
    print(f"Running eval on {qa_path}...")
    report = run(qa_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(RESULTS_DIR) / f"results_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Summary ---")
    for k, v in report["summary"].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nSaved: {out_path}")
