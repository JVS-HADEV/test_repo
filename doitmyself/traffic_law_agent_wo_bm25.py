"""도로교통법 상담 Agent 핵심 로직 — 벡터 검색만 사용 (BM25 없음).

`traffic_law_wo_BM25.ipynb`의 로직을 서버에서 재사용할 수 있게 옮긴 모듈입니다.
하이브리드 버전(`traffic_law_agent.py`)과 동일하게 법령 PDF / 판례 CSV를 로드하고
LangGraph 상담 그래프를 컴파일하지만, 검색은 Chroma 임베딩 유사도만 사용합니다.

이 파일을 import하면 다음이 1회 수행됩니다.
1. 법령 PDF / 판례 CSV 로드 & 청킹
2. Chroma 벡터 스토어 준비 (이미 임베딩되어 있으면 재사용)
3. LangGraph 상담 그래프 컴파일

이후 `consult()` / `consult_detailed()`를 호출할 때마다 그래프만 실행됩니다.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("traffic_law_agent_wo_bm25")

BASE_DIR = Path(__file__).resolve().parent

# ── 환경 설정 ───────────────────────────────────────────────────────────────
load_dotenv()
load_dotenv(BASE_DIR.parent / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일에 OPENAI_API_KEY=... 를 추가한 뒤 "
        "다시 실행하세요."
    )

CHAT_MODEL = os.getenv("TRAFFIC_LAW_CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("TRAFFIC_LAW_EMBEDDING_MODEL", "text-embedding-3-small")

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# ── 데이터 경로 ─────────────────────────────────────────────────────────────
_data_candidates = [
    BASE_DIR / "common_dataset",
    Path("./common_dataset"),
    Path("./doitmyself/common_dataset"),
]
DATA_DIR = next((p.resolve() for p in _data_candidates if p.exists()), None)
if DATA_DIR is None:
    tried = ", ".join(str(p.resolve()) for p in _data_candidates)
    raise FileNotFoundError(f"common_dataset 폴더를 찾을 수 없습니다. 확인한 위치: {tried}")

LAW_PDF = DATA_DIR / "도로교통법_법률_제21246호.pdf"
CASE_CSV = DATA_DIR / "판례목록.csv"
# 노트북과 동일: 하이브리드와 같은 Chroma 디렉터리를 공유해도 무방합니다.
# (임베딩 내용은 같고, 검색 시 BM25를 쓰지 않을 뿐입니다.)
DB_DIR = DATA_DIR.parent / ".traffic_law_z_chroma"

for _path in (LAW_PDF, CASE_CSV):
    if not _path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {_path}")

# ── 문서 로딩 & 청킹 (traffic_law_wo_BM25.ipynb와 동일 로직) ─────────────────
ARTICLE_RE = re.compile(r"제\s*(\d+)\s*조(?:의\s*(\d+))?(?:\([^)]*\))?")


def article_label(match: re.Match) -> str:
    return f"제{match.group(1)}조" + (f"의{match.group(2)}" if match.group(2) else "")


def compact_text(text: str) -> str:
    """PDF 줄바꿈/공백을 검색에 적합하게 정리합니다."""
    text = text.replace("\x00", " ")
    text = re.sub(r"--\s*\d+\s*of\s*\d+\s*--", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 1100, overlap: int = 160) -> list[str]:
    """문장 경계를 우선 보존하는 간단하고 재현 가능한 청킹입니다."""
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("다. ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 2
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


def load_law_documents(pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []
    current_article = "미상"
    for page_no, page in enumerate(reader.pages, start=1):
        text = compact_text(page.extract_text() or "")
        if not text:
            continue
        matches = list(ARTICLE_RE.finditer(text))
        if matches:
            current_article = article_label(matches[0])
        for chunk_no, chunk in enumerate(split_text(text)):
            chunk_articles = [article_label(m) for m in ARTICLE_RE.finditer(chunk)]
            article = ", ".join(dict.fromkeys(chunk_articles)) or current_article
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source_type": "법령",
                    "source": pdf_path.name,
                    "page": page_no,
                    "article": article,
                    "chunk": chunk_no,
                    "effective_date": "2026-07-01",
                },
            ))
        if matches:
            current_article = article_label(matches[-1])
    return documents


def load_case_documents(csv_path: Path) -> list[Document]:
    frame = pd.read_csv(csv_path, encoding="utf-8-sig").fillna("")
    required = {"제목", "사건번호", "선고일자", "조문번호", "판시사항"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"판례 CSV 필수 열 누락: {sorted(missing)}")

    documents: list[Document] = []
    for row_no, row in frame.iterrows():
        content = (
            f"사건명: {row['제목']}\n사건번호: {row['사건번호']}\n"
            f"선고일자: {row['선고일자']}\n관련 조문: {row['조문번호']}\n"
            f"판시사항: {row['판시사항']}"
        ).strip()
        documents.append(Document(
            page_content=content,
            metadata={
                "source_type": "판례",
                "source": csv_path.name,
                "row": int(row_no) + 2,
                "case_name": str(row["제목"]),
                "case_number": str(row["사건번호"]),
                "decision_date": str(row["선고일자"]),
                "articles": str(row["조문번호"]),
            },
        ))
    return documents


def dataset_fingerprint(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode())
        digest.update(str(path.stat().st_mtime_ns).encode())
    return digest.hexdigest()[:12]


def stable_id(doc: Document) -> str:
    raw = f"{doc.metadata}|{doc.page_content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class VectorRetriever:
    """Chroma 임베딩 유사도만 사용하는 순수 벡터 Retriever입니다. (BM25 없음)"""

    def __init__(self, documents: list[Document], vector_store: Chroma):
        self.documents = documents
        self.vector_store = vector_store

    def search(self, query: str, k: int = 6) -> list[Document]:
        return self.vector_store.similarity_search(
            query,
            k=min(k, len(self.documents)),
        )


def source_label(doc: Document) -> str:
    meta = doc.metadata
    if meta["source_type"] == "법령":
        return f"법령 {meta.get('article', '조문 미상')} (PDF p.{meta['page']})"
    return f"판례 {meta.get('case_number', '사건번호 미상')} ({meta.get('decision_date', '선고일 미상')})"


def source_detail(doc: Document) -> dict:
    """API 응답에 넣기 좋은 형태로 출처를 구조화합니다."""
    meta = doc.metadata
    return {
        "type": meta["source_type"],
        "label": source_label(doc),
        "excerpt": doc.page_content[:200],
    }


def format_context(law_hits: list[Document], case_hits: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(law_hits, start=1):
        blocks.append(f"[L{i} | {source_label(doc)}]\n{doc.page_content}")
    for i, doc in enumerate(case_hits, start=1):
        blocks.append(f"[C{i} | {source_label(doc)}]\n{doc.page_content}")
    return "\n\n".join(blocks)


logger.info("도로교통법 문서 로딩 중... (%s)", DATA_DIR)
law_documents = load_law_documents(LAW_PDF)
case_documents = load_case_documents(CASE_CSV)
logger.info("법령 청크 %d개 / 판례 %d건", len(law_documents), len(case_documents))

fingerprint = dataset_fingerprint(LAW_PDF, CASE_CSV)
DB_DIR.mkdir(parents=True, exist_ok=True)

law_store = Chroma(
    collection_name=f"traffic_law_{fingerprint}",
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)
case_store = Chroma(
    collection_name=f"traffic_cases_{fingerprint}",
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)

# 같은 데이터로 서버를 재시작할 때 중복 임베딩하지 않습니다.
if law_store._collection.count() == 0:
    law_store.add_documents(law_documents, ids=[stable_id(d) for d in law_documents])
if case_store._collection.count() == 0:
    case_store.add_documents(case_documents, ids=[stable_id(d) for d in case_documents])

law_retriever = VectorRetriever(law_documents, law_store)
case_retriever = VectorRetriever(case_documents, case_store)

logger.info(
    "법령 벡터 %d개 / 판례 벡터 %d개 준비 완료 (vector-only, no BM25)",
    law_store._collection.count(),
    case_store._collection.count(),
)


# ── LangGraph 상담 워크플로 (traffic_law_wo_BM25.ipynb와 동일 로직) ──────────
class LegalIssueAnalysis(BaseModel):
    category: Literal[
        "교통사고", "음주·약물", "무면허", "속도·신호", "주정차",
        "개인형이동장치", "면허행정", "기타"
    ] = Field(description="가장 중요한 상담 유형")
    issues: list[str] = Field(description="검토할 법적 쟁점")
    search_queries: list[str] = Field(description="법령·판례 검색용 한국어 질의 1~3개")
    likely_articles: list[str] = Field(description="질문에서 추정되는 조문 번호. 모르면 빈 목록")
    missing_facts: list[str] = Field(description="판단에 중요하지만 질문에 없는 사실")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    analysis: dict
    law_hits: list[Document]
    case_hits: list[Document]
    context: str


ANALYSIS_SYSTEM = """당신은 대한민국 도로교통 사건의 쟁점을 구조화하는 분석 Agent다.
사용자의 최신 질문과 대화 맥락에서 검색에 필요한 사실만 추출한다.
검색 질의에는 구체적인 행위, 결과, 관련 조문 후보, 처벌/행정처분 쟁점을 포함한다.
없는 사실을 추정하지 말고 missing_facts에 기록한다."""

ANSWER_SYSTEM = """당신은 대한민국 도로교통법 전문 상담 AI다.
아래 원칙을 반드시 지켜라.

- 제공된 검색 근거만으로 답하고 법령 문구, 벌금액, 판례 결론을 만들지 않는다.
- 현행법 기준일은 2026-07-01(법률 제21246호)임을 명시한다.
- 먼저 '잠정 판단'을 짧게 말하고, 그 다음 '적용 근거', '추가로 필요한 사실', '권장 조치' 순서로 쓴다.
- 모든 핵심 법적 판단 문장 끝에 [L1], [C1]처럼 실제 제공된 근거 ID를 붙인다.
- 법령과 판례가 충돌하거나 판례가 과거 법령을 다루면 현행 법령을 우선하고 시점 차이를 경고한다.
- 자료에 시행령·시행규칙·교통사고처리특례법·특정범죄가중처벌법 등이 없으면 해당 법률의 확인 필요성을 분명히 밝힌다.
- 사실이 부족하면 단정하지 말고 조건부로 설명한 뒤 필요한 질문을 최대 5개 제시한다.
- 사고 직후 상황이면 인명 구조, 119·112 신고, 정차, 피해자 구호, 위험 방지, 인적사항 제공을 우선 안내한다.
- 최종 형량·과실비율·유무죄를 확정하지 않는다.
- 마지막에 '이 답변은 제공 자료에 기초한 일반 정보이며 법률 자문이 아닙니다.'라고 쓴다.
"""

structured_analyzer = llm.with_structured_output(LegalIssueAnalysis)


def latest_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("사용자 질문이 없습니다.")


def analyze_node(state: AgentState) -> dict:
    recent = state["messages"][-6:]
    analysis = structured_analyzer.invoke([
        SystemMessage(content=ANALYSIS_SYSTEM),
        *recent,
    ])
    return {"analysis": analysis.model_dump()}


def retrieve_node(state: AgentState) -> dict:
    user_question = latest_user_text(state["messages"])
    analysis = LegalIssueAnalysis.model_validate(state["analysis"])
    queries = analysis.search_queries or [user_question]
    if analysis.likely_articles:
        queries.append(" ".join(analysis.likely_articles + analysis.issues))

    def collect(retriever: VectorRetriever, limit: int) -> list[Document]:
        unique: dict[str, Document] = {}
        for query in queries[:4]:
            for doc in retriever.search(query, k=limit):
                unique.setdefault(stable_id(doc), doc)
        return list(unique.values())[:limit]

    law_hits = collect(law_retriever, 7)
    case_hits = collect(case_retriever, 5)
    return {
        "law_hits": law_hits,
        "case_hits": case_hits,
        "context": format_context(law_hits, case_hits),
    }


def answer_node(state: AgentState) -> dict:
    analysis = LegalIssueAnalysis.model_validate(state["analysis"])
    prompt = f"""[쟁점 분석]
{analysis.model_dump_json(indent=2)}

[검색 근거]
{state['context']}

[사용자 질문]
{latest_user_text(state['messages'])}

검색 근거의 ID를 정확히 인용해 한국어로 답하라."""
    response = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        *state["messages"][-6:-1],
        HumanMessage(content=prompt),
    ])
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("analyze", analyze_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("answer", answer_node)
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)

_checkpointer = MemorySaver()
agent = builder.compile(checkpointer=_checkpointer)
logger.info(
    "Agent graph ready (vector-only, chat=%s, embedding=%s)",
    CHAT_MODEL,
    EMBEDDING_MODEL,
)


# ── 외부(서버/노트북)에서 사용할 함수 ─────────────────────────────────────
def consult(question: str, thread_id: str = "default", show_sources: bool = False) -> str:
    """상담 질문을 실행합니다. 같은 thread_id는 후속 질문의 맥락을 공유합니다."""
    if not question or not question.strip():
        return "질문을 입력해 주세요."
    result = agent.invoke(
        {"messages": [HumanMessage(content=question.strip())]},
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = str(result["messages"][-1].content)
    if show_sources:
        labels = [source_label(d) for d in result.get("law_hits", []) + result.get("case_hits", [])]
        answer += "\n\n---\n검색된 자료\n" + "\n".join(f"- {label}" for label in labels)
    return answer


def consult_detailed(question: str, thread_id: str = "default") -> dict:
    """FastAPI 응답용 구조화된 결과(답변 + 출처 + 소요 시간)를 반환합니다."""
    if not question or not question.strip():
        raise ValueError("질문을 입력해 주세요.")

    started_at = time.perf_counter()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question.strip())]},
        config={"configurable": {"thread_id": thread_id}},
    )
    elapsed = round(time.perf_counter() - started_at, 3)

    law_hits = result.get("law_hits", [])
    case_hits = result.get("case_hits", [])
    analysis = result.get("analysis", {})

    return {
        "thread_id": thread_id,
        "answer": str(result["messages"][-1].content),
        "category": analysis.get("category"),
        "issues": analysis.get("issues", []),
        "missing_facts": analysis.get("missing_facts", []),
        "sources": [source_detail(d) for d in law_hits + case_hits],
        "response_seconds": elapsed,
        "retriever": "vector-only",
    }


def reset_thread(thread_id: str) -> bool:
    """해당 thread_id의 대화 메모리를 지웁니다. 존재 여부와 무관하게 True를 반환합니다."""
    _checkpointer.delete_thread(thread_id)
    return True


def stats() -> dict:
    """서버 상태/헬스체크용 인덱스 정보."""
    return {
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "retriever": "vector-only (no BM25)",
        "law_chunks": law_store._collection.count(),
        "case_chunks": case_store._collection.count(),
        "law_pdf": LAW_PDF.name,
        "case_csv": CASE_CSV.name,
    }
