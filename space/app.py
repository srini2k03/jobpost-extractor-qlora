"""Gradio demo (Hugging Face Spaces, free CPU tier).

Runs the fine-tuned GGUF Q4_K_M model with llama-cpp-python. Paste a posting and get the JSON,
plus a view that highlights where each extracted value was found in the text.

Environment:
  JOBX_GGUF_REPO   HF model repo with the GGUF files (default <HF_USERNAME>/jobx-qwen2.5-1.5b-gguf)
  JOBX_GGUF_FILE   file pattern (default *Q4_K_M.gguf)
  JOBX_BACKEND     "regex" to run without a model (rule-based fallback, clearly labelled in the UI)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# On the Space, the deploy workflow copies src/jobx next to this file; locally it is pip-installed.
if (HERE / "jobx").exists():
    sys.path.insert(0, str(HERE))

import gradio as gr  # noqa: E402

from jobx.serve.extractor import Extractor  # noqa: E402
from jobx.serve.highlight import highlighted  # noqa: E402
from jobx.serve.llm import LLM, RegexLLM  # noqa: E402

LABEL_COLORS = {
    "company": "#4c78a8",
    "role": "#f58518",
    "skill": "#54a24b",
    "location": "#e45756",
    "salary": "#b279a2",
    "experience": "#9d755d",
    "work_mode": "#72b7b2",
    "employment_type": "#eeca3b",
    "education": "#ff9da6",
}


def load_examples() -> list[str]:
    p = HERE / "examples.json"
    return [e["text"] for e in json.loads(p.read_text())] if p.exists() else []


def default_llm() -> tuple[LLM, str]:
    if os.environ.get("JOBX_BACKEND", "").lower() == "regex":
        return RegexLLM(), "Rule-based fallback (no model loaded)."
    try:
        from jobx.serve.llm import llm_from_env

        llm = llm_from_env()
        return llm, f"Model: `{llm.name}` (llama.cpp, CPU)."
    except Exception as e:  # model repo missing or llama-cpp unavailable: keep the demo usable
        return RegexLLM(), f"Model unavailable ({type(e).__name__}); showing the rule-based fallback."


def make_predict(extractor: Extractor) -> Any:
    def predict(text: str) -> tuple[dict[str, Any], list[tuple[str, str | None]], str]:
        if not text or not text.strip():
            return {}, [], "Paste a job posting first."
        r = extractor.extract(text)
        status = (
            f"{'valid' if r.valid else 'INVALID'} JSON after {r.attempts} attempt(s), "
            f"{r.latency_ms:.0f} ms, backend `{r.backend}`"
        )
        if r.errors:
            status += f". Errors: {'; '.join(r.errors)}"
        return r.data or {}, highlighted(text, r.data), status

    return predict


def build_demo(llm: LLM | None = None, banner: str | None = None) -> gr.Blocks:
    if llm is None:
        llm, banner = default_llm()
    extractor = Extractor(llm, max_retries=1)
    predict = make_predict(extractor)
    with gr.Blocks(title="Job posting to JSON") as demo:
        gr.Markdown(
            "# Job posting to JSON\nMessy Indian job postings (LPA / lakh / per-month salaries, Hinglish, "
            "WhatsApp forwards) to one strict JSON object. Fine-tuned Qwen2.5-1.5B (QLoRA), served as GGUF Q4 "
            "on CPU. [Code and evaluation](https://github.com/srini2k03/jobpost-extractor-qlora)\n\n"
            + (banner or "")
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="Job posting", lines=14, placeholder="Paste a posting...")
                btn = gr.Button("Extract", variant="primary")
                gr.Examples(examples=[[e] for e in load_examples()], inputs=[inp])
            with gr.Column():
                status = gr.Markdown()
                with gr.Tab("JSON"):
                    out_json = gr.JSON(label="Extracted fields")
                with gr.Tab("Highlighted fields"):
                    out_hl = gr.HighlightedText(
                        label="Where each value was found", color_map=LABEL_COLORS, combine_adjacent=False
                    )
        btn.click(predict, inputs=[inp], outputs=[out_json, out_hl, status])
    return demo


if __name__ == "__main__":
    build_demo().launch()
