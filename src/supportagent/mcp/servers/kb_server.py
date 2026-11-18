"""Knowledge-base MCP server: retrieval over policy docs (tool), raw docs (resources), answer template (prompt).

Retrieval is BM25 by default. Set SA_RETRIEVAL_MODE=hybrid (or dense | bm25+) and optionally SA_EMBEDDER to use the opt-in
hybrid retriever (ADR-0012); or pass `retrieval=` to make_server. Unavailable backends fall back to BM25 with a warning."""
from __future__ import annotations

from pathlib import Path

from ...config import POLICY_DIR
from ...rag.hybrid import get_retriever
from ..protocol import MCPServer


def make_server(tokens: dict | None = None, jwt_secret: str | None = None, retrieval: str | None = None) -> MCPServer:
    srv = MCPServer("kb", tokens=tokens, jwt_secret=jwt_secret)
    idx = get_retriever(str(POLICY_DIR), retrieval)

    @srv.tool("search_policy", "Search company policy documents; returns cited passages.",
              {"type": "object", "properties": {"query": {"type": "string", "maxLength": 300},
                                                "top_k": {"type": "integer", "minimum": 1, "maximum": 5}},
               "required": ["query"]}, scope="kb:read")
    def search_policy(args):
        return {"results": idx.search(args["query"], int(args.get("top_k", 3)))}

    for f in sorted(Path(POLICY_DIR).glob("*.md")):
        srv.resource(f"policy://{f.stem}", f.stem.replace("_", " ").title(), f"Full text of {f.name}",
                     (lambda p=f: p.read_text()))
    srv.prompt("policy_answer", "Answer a policy question using only retrieved passages",
               [{"name": "question", "required": True}],
               lambda a: f"Answer using only cited passages from search_policy. Question: {a.get('question','')}")
    return srv


def main() -> None:
    make_server().run_stdio()


if __name__ == "__main__":
    main()
