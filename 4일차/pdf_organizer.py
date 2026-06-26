"""
PDF 분류·정리 프로그램

1. pdf_samples 폴더의 PDF 목록 파악
2. RAG로 핵심 조각을 수집한 뒤 요약
3. 비슷한 주제끼리 10글자 이내 폴더명으로 묶기
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
    "이 문서의 주제와 핵심 내용",
    "저자가 다루는 문제와 결론",
    "문서의 분야와 키워드",
    "문서의 유형과 목적",
]


def list_root_pdfs() -> List[Path]:
    """pdf_samples 루트에 있는 PDF만 수집 (_catalog 제외)."""
    return sorted(DOC_LIBRARY.glob("*.pdf"))


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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 문서 분류를 돕는 요약 도우미다. "
                    "아래 조각만 근거로 3~5문장 한국어 요약을 작성하라. "
                    "문서 유형(규정/보도자료/논문 등)과 핵심 주제를 반드시 포함하라."
                ),
            },
            {
                "role": "user",
                "content": f"PDF 파일명: {pdf_name}\n\n[검색된 조각]\n{context}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def assign_folder_names(summaries: Dict[str, str]) -> Dict[str, str]:
    """요약을 보고 비슷한 문서끼리 같은 10글자 이내 폴더명을 배정한다."""
    summary_block = "\n\n".join(
        f"- {pdf_name}\n  요약: {summary}" for pdf_name, summary in summaries.items()
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 문서 분류 전문가다. "
                    "비슷한 주제/유형의 PDF는 같은 폴더명을 사용하라. "
                    "폴더명 규칙: 한국어, 공백 없이, 10글자 이내, Windows 폴더명 가능. "
                    "반드시 JSON만 출력: {\"pdf파일명.pdf\": \"폴더명\", ...}"
                ),
            },
            {
                "role": "user",
                "content": f"다음 PDF 요약을 보고 폴더명을 정하라.\n\n{summary_block}",
            },
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    folder_map = json.loads(raw)

    cleaned: Dict[str, str] = {}
    for pdf_name in summaries:
        folder = folder_map.get(pdf_name, "기타문서")
        cleaned[pdf_name] = sanitize_folder_name(folder)
    return cleaned


def sanitize_folder_name(name: str, max_len: int = 10) -> str:
    """Windows에서 사용 가능한 폴더명으로 정리 (10글자 이내)."""
    name = re.sub(r'[<>:"/\\|?*]', "", name.strip())
    name = re.sub(r"\s+", "", name)
    if not name:
        name = "기타문서"
    return name[:max_len]


def unique_folder_path(base_name: str, existing: set) -> str:
    """같은 이름이 있으면 숫자 접미사를 붙인다."""
    candidate = base_name
    counter = 2
    while candidate in existing:
        suffix = str(counter)
        candidate = base_name[: max(1, 10 - len(suffix))] + suffix
        counter += 1
    existing.add(candidate)
    return candidate


def move_pdfs(folder_map: Dict[str, str], dry_run: bool = False) -> List[dict]:
    """PDF를 지정된 폴더로 이동한다."""
    results = []
    used_names = set()

    for pdf_name, folder_name in folder_map.items():
        src = DOC_LIBRARY / pdf_name
        if not src.exists():
            results.append({"pdf_name": pdf_name, "status": "missing"})
            continue

        folder_name = unique_folder_path(folder_name, used_names)
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

    pdfs = list_root_pdfs()
    print("\n[1] PDF 파일 목록 (%d개)" % len(pdfs))
    for pdf in pdfs:
        print("  -", pdf.name)

    if not pdfs:
        print("정리할 PDF가 없습니다.")
        return

    print("\n[2] RAG 기반 요약 생성")
    summaries: Dict[str, str] = {}
    for pdf in pdfs:
        print("  요약 중:", pdf.name)
        summaries[pdf.name] = summarize_pdf(pdf.name)
        print("   →", summaries[pdf.name][:80], "...")

    print("\n[3] 폴더명 배정 (10글자 이내, 유사 주제 묶음)")
    folder_map = assign_folder_names(summaries)
    for pdf_name, folder in folder_map.items():
        print("  %s → [%s]" % (pdf_name, folder))

    print("\n[4] PDF 이동")
    move_results = move_pdfs(folder_map, dry_run=dry_run)
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
