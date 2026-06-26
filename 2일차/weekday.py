import calendar
import re
import sys
from datetime import datetime

# 요일 헤더 (월요일부터 시작)
WEEKDAY_SHORT = ["월", "화", "수", "목", "금", "토", "일"]

# 터미널 색상 코드 (ANSI 이스케이프 시퀀스)
BLUE = "\033[34m"   # 파란색 (토요일)
RED = "\033[31m"    # 빨간색 (일요일)
RESET = "\033[0m"   # 색상 초기화

# 색상 코드를 제거할 때 사용하는 정규식
ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def enable_ansi_colors() -> None:
    """Windows 터미널에서 ANSI 색상 출력을 활성화한다."""
    if sys.platform == "win32":
        import os

        os.system("")


def visible_width(text: str) -> int:
    """색상 코드를 제외한 실제 화면에 보이는 글자 수를 반환한다."""
    return len(ANSI_RE.sub("", text))


def pad_visible(text: str, width: int) -> str:
    """색상 코드가 있어도 달력 열이 맞도록 공백을 채운다."""
    return text + " " * (width - visible_width(text))


def colorize(text: str, column: int) -> str:
    """열 위치에 따라 토요일(5)은 파란색, 일요일(6)은 빨간색으로 표시한다."""
    if column == 5:
        return f"{BLUE}{text}{RESET}"
    if column == 6:
        return f"{RED}{text}{RESET}"
    return text


def print_month_calendar(year: int, month: int) -> None:
    """입력받은 년·월의 달력을 출력한다."""
    calendar.setfirstweekday(calendar.MONDAY)  # 한국 달력처럼 월요일부터 시작
    weeks = calendar.monthcalendar(year, month)  # 주 단위 2차원 리스트 (0은 빈 칸)

    print(f"\n{'=' * 32}")
    print(f"        {year}년 {month}월 달력")
    print(f"{'=' * 32}")

    # 요일 헤더 출력 (토·일요일 색상 적용)
    header = "".join(
        pad_visible(colorize(f"{w:>4}", i), 4) for i, w in enumerate(WEEKDAY_SHORT)
    )
    print(header)
    print("-" * 32)

    # 각 주(행)별로 날짜 출력
    for week in weeks:
        row = ""
        for column, day in enumerate(week):
            if day == 0:
                cell = "    "  # 해당 월에 속하지 않는 빈 칸
            else:
                cell = colorize(f"{day:4}", column)
            row += pad_visible(cell, 4)
        print(row)


def main() -> None:
    enable_ansi_colors()

    try:
        year = int(input("년을 입력하세요: "))
        month = int(input("월을 입력하세요: "))

        # 유효한 년·월인지 확인 (잘못된 값이면 ValueError 발생)
        datetime(year, month, 1)
        print_month_calendar(year, month)
    except ValueError:
        print("올바른 날짜를 입력해주세요.")


if __name__ == "__main__":
    main()
