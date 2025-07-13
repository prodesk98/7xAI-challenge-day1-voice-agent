import os

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core import process

app = FastAPI(
    title="Voice Agent",
    description="An AI-powered voice agent for various tasks",
    version="1.0.0",
)

_path = os.path.dirname(os.path.abspath(__file__))

app.mount("/public", StaticFiles(directory="%s/public" % _path), name="static")
templates = Jinja2Templates(directory="%s/public/templates" % _path)

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(ws: WebSocket):
    await ws.accept()
    print(
        "WebSocket connection established. Ready to receive audio data."
    )
    try:
        while True:
            data = await ws.receive_bytes()
            if data == b'ping':
                print("Received ping, sending pong.")
                await ws.send_bytes(b"pong")
                continue

            print(f"Received audio data of length {len(data)} bytes.")
            await ws.send_bytes(await process(data))
    except Exception as e:
        print(f"Error during WebSocket communication: {e}")


@app.get("/")
async def root():
    return templates.TemplateResponse("index.html", {"request": {}})
