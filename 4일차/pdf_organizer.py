"""
PDF 분류·정리 프로그램

1. pdf_samples 폴더의 PDF 목록 파악
2. RAG로 핵심 조각을 수집한 뒤 요약
3. 대분류(논문/보도자료/학칙 등) 폴더로 묶기
4. PDF를 해당 폴더로 이동

4.5 pdf 정리 agent.py 의 RAG 함수를 재사용합니다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parent
DOC_LIBRARY = BASE / "samples" / "pdf_samples"
MANIFEST_PATH = DOC_LIBRARY / "_catalog" / "organize_manifest.json"

load_dotenv(BASE.parent / ".env")
client = OpenAI()

# 4.5 pdf 정리 agent.py 모듈 로드
AGENT_PATH = BASE / "4.5 pdf 정리 agent.py"
_spec = importlib.util.spec_from_file_location("pdf_agent", AGENT_PATH)
pdf_agent = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(pdf_agent)

build_pdf_index = pdf_agent.build_pdf_index
search_chunks = pdf_agent.search_chunks

RAG_QUERIES = [
    "이 문서의 유형과 목적",
    "이 문서의 주제와 핵심 내용",
    "이 문서가 규정 보도자료 논문 중 무엇인지",
]

# 허용되는 대분류 폴더명
ALLOWED_CATEGORIES = ["논문", "보도자료", "학칙", "기타"]

CATEGORY_ALIASES = {
    "규정": "학칙",
    "학칙": "학칙",
    "대학원학칙": "학칙",
    "보도": "보도자료",
    "보도자료": "보도자료",
    "논문": "논문",
    "paper": "논문",
    "기타": "기타",
    "기타문서": "기타",
}


def list_pdfs() -> List[Path]:
    """pdf_samples 아래 PDF 수집 (_catalog 제외, 하위 폴더 포함)."""
    pdfs = []
    for pdf in DOC_LIBRARY.rglob("*.pdf"):
        if "_catalog" in pdf.parts:
            continue
        pdfs.append(pdf)
    return sorted(pdfs, key=lambda p: str(p).lower())


def rag_collect_context(pdf_name: str, top_k: int = 4) -> str:
    """RAG 검색으로 요약에 쓸 핵심 조각을 수집한다."""
    chunks = build_pdf_index(pdf_name)
    if not chunks:
        return ""

    seen = set()
    collected: List[str] = []
    for query in RAG_QUERIES:
        for hit in search_chunks(query, chunks, top_k=top_k):
            chunk_id = hit.get("chunk_id")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            collected.append(hit["text"])

    if not collected:
        collected = [item["text"] for item in chunks[:5]]

    return "\n\n---\n\n".join(collected)[:8000]


def summarize_pdf(pdf_name: str) -> str:
    """RAG로 수집한 조각을 바탕으로 PDF를 요약한다."""
    context = rag_collect_context(pdf_name)
    categories = ", ".join(ALLOWED_CATEGORIES)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 문서 분류를 돕는 요약 도우미다. "
                    "아래 조각만 근거로 요약하라. "
                    "반드시 아래 형식을 지켜라:\n"
                    "문서 대분류: (다음 중 하나만) %s\n"
                    "한 줄 요약: (1문장)\n"
                    "세부 주제는 적지 말고, 문서 종류(대분류) 판단에 집중하라."
                    % categories
                ),
            },
            {
                "role": "user",
                "content": "PDF 파일명: %s\n\n[검색된 조각]\n%s" % (pdf_name, context),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def normalize_category(name: str) -> str:
    """폴더명을 허용된 대분류 중 하나로 정규화한다."""
    cleaned = sanitize_folder_name(name)
    if cleaned in ALLOWED_CATEGORIES:
        return cleaned
    if cleaned in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[cleaned]
    for allowed in ALLOWED_CATEGORIES:
        if allowed in cleaned or cleaned in allowed:
            return allowed
    return "기타"


def assign_folder_names(summaries: Dict[str, str]) -> Dict[str, str]:
    """요약을 보고 대분류 폴더명을 배정한다."""
    summary_block = "\n\n".join(
        "- %s\n  요약: %s" % (pdf_name, summary)
        for pdf_name, summary in summaries.items()
    )
    categories = ", ".join(ALLOWED_CATEGORIES)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 문서 분류 전문가다. "
                    "각 PDF를 대분류 폴더 하나에만 배정하라. "
                    "세부 주제별 폴더는 만들지 말 것. "
                    "예: 모든 학술 논문은 '논문', 회사 보도자료는 '보도자료', 학칙·규정은 '학칙'. "
                    "폴더명은 반드시 다음 중 하나만 사용: %s. "
                    "반드시 JSON만 출력: {\"pdf파일명.pdf\": \"대분류\", ...}"
                    % categories
                ),
            },
            {
                "role": "user",
                "content": "다음 PDF 요약을 보고 대분류 폴더명을 정하라.\n\n%s" % summary_block,
            },
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    folder_map = json.loads(raw)

    cleaned: Dict[str, str] = {}
    for pdf_name in summaries:
        folder = folder_map.get(pdf_name, "기타")
        cleaned[pdf_name] = normalize_category(folder)
    return cleaned


def sanitize_folder_name(name: str, max_len: int = 10) -> str:
    """Windows에서 사용 가능한 폴더명으로 정리 (10글자 이내)."""
    name = re.sub(r'[<>:"/\\|?*]', "", name.strip())
    name = re.sub(r"\s+", "", name)
    if not name:
        name = "기타문서"
    return name[:max_len]


def move_pdfs(
    pdf_paths: Dict[str, Path],
    folder_map: Dict[str, str],
    dry_run: bool = False,
) -> List[dict]:
    """PDF를 지정된 대분류 폴더로 이동한다."""
    results = []

    for pdf_name, folder_name in folder_map.items():
        src = pdf_paths.get(pdf_name)
        if src is None or not src.exists():
            results.append({"pdf_name": pdf_name, "status": "missing"})
            continue

        dest_dir = DOC_LIBRARY / folder_name
        dest = dest_dir / pdf_name

        if dry_run:
            results.append(
                {
                    "pdf_name": pdf_name,
                    "folder": folder_name,
                    "from": str(src),
                    "to": str(dest),
                    "status": "dry_run",
                }
            )
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.move(str(src), str(dest))
        results.append(
            {
                "pdf_name": pdf_name,
                "folder": folder_name,
                "from": str(src),
                "to": str(dest),
                "status": "moved",
            }
        )
    return results


def save_manifest(
    summaries: Dict[str, str],
    folder_map: Dict[str, str],
    move_results: List[dict],
) -> None:
    """분류 결과를 _catalog/organize_manifest.json 에 저장한다."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "organized_at": datetime.now().isoformat(timespec="seconds"),
        "documents": [
            {
                "pdf_name": pdf_name,
                "summary": summaries[pdf_name],
                "folder": folder_map[pdf_name],
                "move": next((r for r in move_results if r["pdf_name"] == pdf_name), {}),
            }
            for pdf_name in summaries
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def organize_pdfs(dry_run: bool = False) -> None:
    print("=" * 60)
    print("PDF 분류·정리 시작")
    print("대상 폴더:", DOC_LIBRARY)
    print("=" * 60)

    pdfs = list_pdfs()
    pdf_paths = {pdf.name: pdf for pdf in pdfs}
    print("\n[1] PDF 파일 목록 (%d개)" % len(pdfs))
    for pdf in pdfs:
        print("  -", pdf.relative_to(DOC_LIBRARY))

    if not pdfs:
        print("정리할 PDF가 없습니다.")
        return

    print("\n[2] RAG 기반 요약 생성 (대분류 판단용)")
    summaries: Dict[str, str] = {}
    for pdf in pdfs:
        print("  요약 중:", pdf.name)
        summaries[pdf.name] = summarize_pdf(pdf.name)
        print("   →", summaries[pdf.name][:80], "...")

    print("\n[3] 대분류 폴더 배정 (%s)" % ", ".join(ALLOWED_CATEGORIES))
    folder_map = assign_folder_names(summaries)
    for pdf_name, folder in folder_map.items():
        print("  %s → [%s]" % (pdf_name, folder))

    print("\n[4] PDF 이동")
    move_results = move_pdfs(pdf_paths, folder_map, dry_run=dry_run)
    for item in move_results:
        print("  %s → %s (%s)" % (item["pdf_name"], item.get("folder", "-"), item["status"]))

    if not dry_run:
        save_manifest(summaries, folder_map, move_results)
        print("\n분류 결과 저장:", MANIFEST_PATH)

    print("\n완료!")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 기반 PDF 분류·정리")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 이동 없이 결과만 출력",
    )
    args = parser.parse_args()
    organize_pdfs(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
