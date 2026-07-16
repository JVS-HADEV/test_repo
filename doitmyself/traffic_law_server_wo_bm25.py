"""도로교통법 상담 Agent (벡터 전용, BM25 없음)를 FastAPI 서버로 올립니다.

하이브리드 버전(`traffic_law_server.py`)과 동일한 엔드포인트 구조를 따릅니다.
핵심 로직은 `traffic_law_agent_wo_bm25.py`에 있고, 이 파일은 HTTP 연결만 담당합니다.

실행 방법
---------
    cd doitmyself
    uvicorn traffic_law_server_wo_bm25:app --reload --port 8001

- 브라우저 채팅 UI : http://127.0.0.1:8001/
- Swagger 문서     : http://127.0.0.1:8001/docs
- 헬스체크         : http://127.0.0.1:8001/health

참고: 하이브리드 서버(포트 8000)와 동시에 띄울 수 있도록 기본 포트를 8001로 둡니다.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import traffic_law_agent_wo_bm25 as agent_module

app = FastAPI(
    title="도로교통법 상담 Agent API (Vector-only)",
    description="LangGraph 벡터 RAG 기반 도로교통법 상담 챗봇 서버 (BM25 미사용)",
    version="1.0.0",
)

HTML_PATH = Path(__file__).with_name("traffic_law_chat_wo_bm25.html")


# ── 요청/응답 스키마 ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="상담 질문")
    thread_id: str | None = Field(
        default=None, description="대화 맥락을 유지할 세션 ID. 생략하면 새 세션이 발급됩니다."
    )
    show_sources: bool = Field(default=True, description="검색 근거를 응답에 포함할지 여부")


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    category: str | None = None
    issues: list[str] = []
    missing_facts: list[str] = []
    sources: list[dict] = []
    response_seconds: float
    retriever: str | None = None


class NewSessionResponse(BaseModel):
    thread_id: str


class ResetResponse(BaseModel):
    ok: bool
    thread_id: str


# ── 순수 함수: 서버 없이도 그대로 테스트/재사용 가능 ─────────────────────────
def new_session() -> dict:
    return {"thread_id": str(uuid.uuid4())}


def chat(question: str, thread_id: str | None, show_sources: bool) -> dict:
    tid = thread_id or str(uuid.uuid4())
    result = agent_module.consult_detailed(question, thread_id=tid)
    if not show_sources:
        result = {**result, "sources": []}
    return result


def reset_session(thread_id: str) -> dict:
    ok = agent_module.reset_thread(thread_id)
    return {"ok": ok, "thread_id": thread_id}


def health() -> dict:
    return {"ok": True, **agent_module.stats()}


# ── 엔드포인트: 위 함수를 그대로 연결 ────────────────────────────────────────
@app.get("/")
def home():
    """브라우저에서 채팅 UI를 보여줍니다."""
    return FileResponse(HTML_PATH)


@app.get("/health")
def api_health():
    return health()


@app.post("/session", response_model=NewSessionResponse)
def api_new_session():
    """새 대화 세션(thread_id)을 발급합니다."""
    return new_session()


@app.post("/chat", response_model=ChatResponse)
def api_chat(payload: ChatRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question은 비어 있을 수 없습니다.")
    try:
        return chat(payload.question, payload.thread_id, payload.show_sources)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - 상담 실패를 500으로 명확히 알림
        raise HTTPException(status_code=500, detail=f"상담 처리 중 오류: {exc}") from exc


@app.delete("/chat/{thread_id}", response_model=ResetResponse)
def api_reset_session(thread_id: str):
    """해당 세션의 대화 메모리를 초기화합니다."""
    return reset_session(thread_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("traffic_law_server_wo_bm25:app", host="127.0.0.1", port=8001, reload=True)
