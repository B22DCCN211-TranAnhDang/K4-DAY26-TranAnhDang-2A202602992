# cspell: disable
"""Minh hoạ Function Calling qua OpenRouter.

Model chỉ quyết định gọi tool nào; app Python mới là nơi thực thi tool.

Cách chạy (PowerShell):
    $env:OPENROUTER_API_KEY="..."
    python weather_function_calling.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def load_local_env() -> None:
    """Đọc file .env cùng thư mục mà không ghi đè biến môi trường đã có."""
    env_file = Path(__file__).with_name(".env")
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


load_local_env()
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý thời tiết thân thiện, trả lời bằng tiếng Việt tự nhiên. "
    "Khi người dùng hỏi về thời tiết của một hoặc nhiều thành phố, hãy gọi công cụ get_weather "
    "cho từng thành phố được hỏi. "
    "Dùng emoji phù hợp (🌧️ 🌤️ 💨 💧). "
    "Tóm tắt ngắn gọn, dễ hiểu, và đưa ra lời khuyên thực tế "
    "(ví dụ: mang ô, mặc áo mỏng, ...)."
)

# 1. App tự định nghĩa schema của tool. OpenRouter dùng schema tương thích OpenAI.
GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Lấy thời tiết hiện tại của một thành phố",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Tên thành phố",
                }
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

TOOLS: list[Any] = [GET_WEATHER_TOOL]


# 2. App tự thực thi tool (trong thực tế sẽ gọi API thời tiết thật).
def get_weather(city: str) -> str:
    """Trả về thời tiết mock của *city*. Dùng làm tool cho model."""
    mock_data = {
        "Hà Nội": {
            "nhiệt_độ": "29°C",
            "thời_tiết": "trời mưa nhẹ",
            "độ_ẩm": "82%",
            "gió": {"hướng": "Đông Nam", "tốc_độ": "12 km/h"},
        },
        "Hồ Chí Minh": {
            "nhiệt_độ": "33°C",
            "thời_tiết": "mưa rào",
            "độ_ẩm": "75%",
            "gió": {"hướng": "Tây Nam", "tốc_độ": "15 km/h"},
        },
        "Đà Nẵng": {
            "nhiệt_độ": "30°C",
            "thời_tiết": "nhiều mây",
            "độ_ẩm": "78%",
            "gió": {"hướng": "Đông", "tốc_độ": "10 km/h"},
        },
    }

    default = {"nhiệt_độ": "28°C", "thời_tiết": "không có dữ liệu chi tiết"}
    return json.dumps({"city": city, **mock_data.get(city, default)}, ensure_ascii=False)


def run(prompt: str) -> str:
    """Gửi prompt tới OpenRouter và xử lý vòng lặp function calling."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Chưa đặt biến môi trường OPENROUTER_API_KEY")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    message = response.choices[0].message

    # 3. Model yêu cầu tool; app tự chạy hàm rồi gửi kết quả lại model.
    while message.tool_calls:
        messages.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            if tool_call.type != "function":
                continue
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  [model yêu cầu] {name}({args})")

            if name != "get_weather":
                result = json.dumps({"error": f"Tool không được hỗ trợ: {name}"})
            else:
                result = get_weather(city=str(args["city"]))

            print(f"  [app thực thi]  -> {result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

    return message.content or "Model không trả về nội dung."


if __name__ == "__main__":
    question = "Thời tiết Hà Nội và Đà Nẵng hôm nay thế nào?"
    print(f"User: {question}\n")
    print("Trả lời:", run(question))
