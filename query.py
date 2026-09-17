"""Retrieval Interface.

Exposes an async API over the LightRAG dual-level index and demonstrates the two
modes that exercise it fully:

* ``hybrid`` -- low-level entity retrieval (local) fused with high-level
  relationship/theme retrieval (global) over the knowledge graph.
* ``mix``    -- the same graph retrieval plus dense vector retrieval over the raw
  text chunks, i.e. graph structure and semantic text search together.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

from lightrag import LightRAG, QueryParam
from lightrag.prompt import PROMPTS
from lightrag.utils import logger, setup_logger

from config import Settings
from rag_factory import rag_session

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]

DEMO_QUESTIONS: tuple[str, ...] = (
    "What is Sparse Attention Routing and on which datasets was it evaluated?",
    "Who authored the paper on Sparse Attention Routing and where was it published?",
)


@dataclass
class QueryResult:
    """Answer plus the retrieval context that produced it."""

    mode: str
    question: str
    answer: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = False

    def summary(self) -> str:
        return (
            f"[{self.mode}] ok={self.ok} entities={len(self.entities)} "
            f"relationships={len(self.relationships)} chunks={len(self.chunks)}"
        )


class QueryInterface:
    """Async query facade over a LightRAG instance."""

    def __init__(
        self,
        rag: LightRAG,
        *,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        response_type: str = "Multiple Paragraphs",
    ) -> None:
        self._rag = rag
        self._top_k = top_k
        self._chunk_top_k = chunk_top_k
        self._response_type = response_type

    def _param(self, mode: QueryMode, **overrides: Any) -> QueryParam:
        # QueryParam supports positional construction, so always pass keywords.
        params: dict[str, Any] = {
            "mode": mode,
            "enable_rerank": False,  # no rerank model configured
            "response_type": self._response_type,
        }
        if self._top_k is not None:
            params["top_k"] = self._top_k
        if self._chunk_top_k is not None:
            params["chunk_top_k"] = self._chunk_top_k
        params.update(overrides)
        return QueryParam(**params)

    async def ask(
        self, question: str, mode: QueryMode = "hybrid", **kw: Any
    ) -> QueryResult:
        """Run retrieval + generation and return answer with its context."""
        logger.info("Query (%s): %s", mode, question)
        result = await self._rag.aquery_llm(question, param=self._param(mode, **kw))
        status = result.get("status")
        data = result.get("data") or {}
        llm_response = result.get("llm_response") or {}
        answer = str(llm_response.get("content") or "").strip()
        failed = (
            not answer or answer == PROMPTS["fail_response"] or "[no-context]" in answer
        )
        ok = status == "success" and not failed
        if not ok:
            logger.warning(
                "Query (%s) did not produce a grounded answer: status=%s message=%s",
                mode,
                status,
                result.get("message"),
            )
        query_result = QueryResult(
            mode=mode,
            question=question,
            answer=answer,
            entities=list(data.get("entities") or []),
            relationships=list(data.get("relationships") or []),
            chunks=list(data.get("chunks") or []),
            ok=ok,
        )
        logger.info(query_result.summary())
        return query_result

    async def hybrid(self, question: str, **kw: Any) -> QueryResult:
        """Dual-level graph retrieval (local entities + global themes)."""
        return await self.ask(question, "hybrid", **kw)

    async def mix(self, question: str, **kw: Any) -> QueryResult:
        """Graph retrieval fused with dense vector retrieval over text chunks."""
        return await self.ask(question, "mix", **kw)

    async def retrieve(
        self, question: str, mode: QueryMode = "hybrid", **kw: Any
    ) -> dict[str, Any]:
        """Return retrieval context only (no answer generation)."""
        result = await self._rag.aquery_data(question, param=self._param(mode, **kw))
        return result.get("data") or {}

    async def compare_modes(self, question: str) -> dict[str, QueryResult]:
        """Run ``hybrid`` and ``mix`` concurrently for the same question."""
        hybrid_result, mix_result = await asyncio.gather(
            self.hybrid(question), self.mix(question)
        )
        return {"hybrid": hybrid_result, "mix": mix_result}


def format_result(result: QueryResult) -> str:
    lines = [
        f"=== mode={result.mode} ok={result.ok} ===",
        f"Q: {result.question}",
        f"A: {result.answer}",
        result.summary(),
    ]
    return "\n".join(lines)


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Query the Graph-RAG index.")
    parser.add_argument("question", nargs="*", help="Question(s); defaults to demo set")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logger("lightrag", level=args.log_level, enable_file_logging=False)
    questions = args.question or list(DEMO_QUESTIONS)
    settings = Settings()
    async with rag_session(settings) as rag:
        interface = QueryInterface(rag)
        for question in questions:
            results = await interface.compare_modes(question)
            for result in results.values():
                print(format_result(result))
                print()


if __name__ == "__main__":
    asyncio.run(main())
