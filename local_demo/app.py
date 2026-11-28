"""Launch the local Gradio comparison."""
import gradio as gr

from .config import Settings
from .service import Demo


def build_app(demo):
    databases = sorted(demo.schemas)
    with gr.Blocks(title="NL to MQL") as app:
        gr.Markdown("# Natural language to MongoDB")
        with gr.Row():
            database = gr.Dropdown(databases, value="allergy_1" if "allergy_1" in databases else databases[0],
                                   label="Spider database")
            retry = gr.Checkbox(value=True, label="Allow one schema-feedback retry")
        question = gr.Textbox(label="Question", placeholder="How many students are there?")
        submit = gr.Button("Generate", variant="primary")
        with gr.Row():
            base = gr.JSON(label="Zero-shot Qwen · local GGUF")
            fallback = gr.JSON(label="Fallback BM25 · validated Spider training examples")
        with gr.Row():
            r3 = gr.JSON(label="Registered BGE RAG")
            ft = gr.JSON(label="Fine-tuned Qwen + R3")
        gr.Markdown("Validation checks JSON and schema. The demo does not execute queries.")

        def generate(db, text, retry_enabled):
            try:
                return demo.answer(db, text, retry_enabled)
            except ValueError as error:
                raise gr.Error(str(error)) from error

        inputs, outputs = [database, question, retry], [base, fallback, r3, ft]
        submit.click(generate, inputs, outputs, concurrency_id="local-qwen", concurrency_limit=1)
        question.submit(generate, inputs, outputs, concurrency_id="local-qwen", concurrency_limit=1)
    return app


def main():
    settings = Settings.from_env()
    build_app(Demo(settings)).queue().launch(server_name="127.0.0.1", server_port=settings.port, share=False)


if __name__ == "__main__":
    main()
