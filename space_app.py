import os

import gradio as gr
import uvicorn
from fastapi.responses import RedirectResponse

from app.config import settings
from app.db import find_nearest
from app.embedding import embed_texts
from app.main import app as api_app


def demo_check(text: str):
    text = (text or "").strip()
    if not text:
        return []
    try:
        vector = embed_texts([text])[0]
        rows = find_nearest(vector, settings.TOP_K_DEFAULT)
    except Exception as exc:
        raise gr.Error(f"Check failed: {exc}")
    return [
        [
            row[1],
            row[2],
            round(float(row[3]), 3),
            "possible duplicate" if float(row[3]) >= settings.DUPLICATE_THRESHOLD else "",
        ]
        for row in rows
    ]


with gr.Blocks(title="KSL Connect - Duplicate Detection") as demo:
    gr.Markdown(
        "## KSL Connect: duplicate detection\n"
        "Type a word or phrase (English or Swahili) to see the closest "
        "existing glosses. Nothing is saved from this page."
    )
    box = gr.Textbox(label="Gloss", placeholder="e.g. thank you / asante")
    table = gr.Dataframe(
        headers=["Existing gloss", "Language", "Similarity", "Flag"],
        interactive=False,
    )
    box.submit(demo_check, inputs=box, outputs=table)
    gr.Button("Check").click(demo_check, inputs=box, outputs=table)


@api_app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


app = gr.mount_gradio_app(api_app, demo, path="/ui")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
