import os
import json

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from prompts import DIRECTION_MAP, AUTO_DETECT_PROMPT

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
MODEL = os.getenv("MODEL", "qwen-plus")

app = FastAPI(title="职能沟通翻译助手")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/translate")
async def translate(request: Request):
    body = await request.json()
    content = body.get("content", "").strip()
    direction = body.get("direction", "")

    if not content:
        return {"error": "请输入要翻译的内容"}

    system_prompt = DIRECTION_MAP.get(direction)
    if not system_prompt:
        return {"error": f"不支持的翻译方向: {direction}"}

    async def event_stream():
        try:
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                stream=True,
                temperature=0.7,
                max_tokens=2000,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    data = json.dumps(
                        {"content": chunk.choices[0].delta.content},
                        ensure_ascii=False,
                    )
                    yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/detect")
async def detect_role(request: Request):
    """自动识别输入内容属于产品视角还是开发视角。"""
    body = await request.json()
    content = body.get("content", "").strip()

    if not content:
        return {"role": "unknown"}

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": AUTO_DETECT_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=10,
        )
        detected = response.choices[0].message.content.strip().lower()
        role = "product" if "product" in detected else "dev"
        return {"role": role}
    except Exception as e:
        return {"role": "unknown", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
