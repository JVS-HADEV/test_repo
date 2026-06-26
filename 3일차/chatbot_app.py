import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import pytz
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

# Python 3.9+ 호환

# .env 로드 (3일차 폴더 또는 상위 폴더)
BASE_DIR = Path(__file__).resolve().parent
for env_path in (BASE_DIR / ".env", BASE_DIR.parent / ".env"):
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

MODEL = "gpt-4o-mini"
MAX_TOOL_ROUNDS = 5
DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful assistant with tools for city time, weather, and US stock prices. "
    "Use tools when needed. Answer in Korean unless the user asks otherwise."
)

CITY_TZ = {
    "서울": "Asia/Seoul",
    "seoul": "Asia/Seoul",
    "뉴욕": "America/New_York",
    "new york": "America/New_York",
    "도쿄": "Asia/Tokyo",
    "tokyo": "Asia/Tokyo",
    "런던": "Europe/London",
    "london": "Europe/London",
}

WEATHER_DESCRIPTIONS = {
    0: "맑음",
    1: "대체로 맑음",
    2: "부분적으로 흐림",
    3: "흐림",
    45: "안개",
    48: "짙은 안개",
    51: "이슬비",
    53: "이슬비",
    55: "강한 이슬비",
    61: "약한 비",
    63: "비",
    65: "강한 비",
    71: "약한 눈",
    73: "눈",
    75: "강한 눈",
    80: "소나기",
    95: "뇌우",
}


def get_city_time_basic() -> str:
    """도시 현재 시간을 반환(기본 버전: 시간대 미반영)"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_city_time_tz(city: str) -> str:
    key = city.strip().lower()
    tz_name = CITY_TZ.get(key)
    if not tz_name:
        return json.dumps({"error": "지원하지 않는 도시: %s" % city}, ensure_ascii=False)

    now = datetime.now(pytz.timezone(tz_name)).strftime("%Y-%m-%d %H:%M:%S")
    return json.dumps(
        {"city": city, "timezone": tz_name, "current_time": now},
        ensure_ascii=False,
    )


def get_us_stock_price(ticker: str) -> str:
    symbol = ticker.strip().upper()
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="2d")
        if hist.empty:
            return json.dumps({"error": "%s 데이터가 없습니다." % symbol}, ensure_ascii=False)

        latest = hist.iloc[-1]
        close_price = float(latest["Close"]) if "Close" in latest else None
        open_price = float(latest["Open"]) if "Open" in latest else None

        return json.dumps(
            {
                "ticker": symbol,
                "open": round(open_price, 2) if open_price is not None else None,
                "close": round(close_price, 2) if close_price is not None else None,
                "currency": "USD",
                "source": "yfinance",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def get_current_weather(city: str) -> str:
    """Open-Meteo API로 도시 현재 날씨 조회 (API 키 불필요)"""
    city_query = city.strip()
    try:
        geo_response = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_query, "count": 1, "language": "ko", "format": "json"},
            timeout=10.0,
        )
        geo_response.raise_for_status()
        results = geo_response.json().get("results", [])
        if not results:
            return json.dumps({"error": "도시를 찾을 수 없습니다: %s" % city}, ensure_ascii=False)

        location = results[0]
        latitude = location["latitude"]
        longitude = location["longitude"]
        city_name = location.get("name", city_query)

        weather_response = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=10.0,
        )
        weather_response.raise_for_status()
        current = weather_response.json().get("current", {})
        weather_code = current.get("weather_code")

        return json.dumps(
            {
                "city": city_name,
                "temperature_c": current.get("temperature_2m"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "weather": WEATHER_DESCRIPTIONS.get(weather_code, "알 수 없음"),
                "weather_code": weather_code,
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "source": "open-meteo",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_city_time_basic",
            "description": "현재 시간을 반환합니다. (시간대 미반영)",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_time_tz",
            "description": "도시의 시간대를 반영해 현재 시간을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_stock_price",
            "description": "미국 주식 티커를 입력받고 최근 주가 정보를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "도시 이름을 입력받고 현재 날씨를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]

TOOL_FUNCTIONS: Dict[str, Callable[..., str]] = {
    "get_city_time_basic": get_city_time_basic,
    "get_city_time_tz": get_city_time_tz,
    "get_us_stock_price": get_us_stock_price,
    "get_current_weather": get_current_weather,
}


def rerun_app() -> None:
    """Streamlit 구버전(st.experimental_rerun) 호환."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def sidebar_button(label: str, **kwargs: Any) -> bool:
    """use_container_width 미지원 Streamlit 버전 호환."""
    try:
        return st.button(label, use_container_width=True, **kwargs)
    except TypeError:
        return st.button(label, **kwargs)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("`.env` 파일에 `OPENAI_API_KEY=sk-...` 를 설정하세요.")
        st.stop()
    return OpenAI(api_key=api_key)


def assistant_message_to_dict(message: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return data


def run_agent(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    temperature: float,
) -> Tuple[str, int]:
    total_tokens = 0

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            messages=messages,
            tools=TOOLS,
        )
        if response.usage:
            total_tokens += response.usage.total_tokens

        message = response.choices[0].message
        if not message.tool_calls:
            content = message.content or ""
            messages.append({"role": "assistant", "content": content})
            return content, total_tokens

        messages.append(assistant_message_to_dict(message))
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments or "{}")
            function = TOOL_FUNCTIONS.get(function_name)
            if function is None:
                result = json.dumps({"error": "알 수 없는 함수: %s" % function_name}, ensure_ascii=False)
            else:
                result = function(**function_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    fallback = "도구 호출이 너무 많아 답변을 중단했습니다."
    messages.append({"role": "assistant", "content": fallback})
    return fallback, total_tokens


def init_session_state() -> None:
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    if "api_messages" not in st.session_state:
        st.session_state.api_messages = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0


def main() -> None:
    st.set_page_config(page_title="Tool Chatbot", page_icon="💬", layout="centered")
    st.title("💬 OpenAI Tool 챗봇")
    st.caption("3일차 실습 · Streamlit + OpenAI API + Tool Calling")

    init_session_state()

    with st.sidebar:
        st.header("설정")
        system_message = st.text_area(
            "System 메시지",
            value=DEFAULT_SYSTEM_MESSAGE,
            height=120,
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
        )
        st.metric("이번 세션 사용 토큰", "%s" % f"{st.session_state.total_tokens:,}")
        st.caption("※ 남은 토큰이 아니라, 이 챗봇에서 쓴 토큰 합계입니다.")
        st.markdown("**사용 가능한 Tool**")
        st.markdown(
            "- `get_city_time_basic` : 현재 시간\n"
            "- `get_city_time_tz` : 도시별 시간\n"
            "- `get_us_stock_price` : 미국 주식 조회\n"
            "- `get_current_weather` : 현재 날씨"
        )
        if sidebar_button("대화 초기화"):
            st.session_state.display_messages = []
            st.session_state.api_messages = []
            st.session_state.total_tokens = 0
            rerun_app()

    for message in st.session_state.display_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("메시지를 입력하세요"):
        st.session_state.display_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        api_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_message},
            *st.session_state.api_messages,
            {"role": "user", "content": prompt},
        ]
        history_len = len(st.session_state.api_messages)

        with st.chat_message("assistant"):
            try:
                with st.spinner("답변 생성 중..."):
                    client = get_client()
                    answer, used_tokens = run_agent(client, api_messages, temperature)
            except Exception as exc:
                st.error("API 호출 실패: %s" % exc)
                st.session_state.display_messages.pop()
                st.stop()

            st.markdown(answer)
            if used_tokens:
                st.caption("이번 호출 토큰: %s" % f"{used_tokens:,}")
                st.session_state.total_tokens += used_tokens

        st.session_state.api_messages.extend(api_messages[history_len + 1 :])
        st.session_state.display_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
