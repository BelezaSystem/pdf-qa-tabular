import argparse
import os
from dotenv import load_dotenv
import gradio as gr

from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings

# Reuso da lógica de resposta do projeto
from main import responder, CAMINHO_DB

load_dotenv()


def build_stack(use_local_embeddings: bool, no_llm: bool):
    api_key = os.environ.get("OPENAI_API_KEY")

    # Embeddings (sempre locais por padrão para compatibilidade com DB criado)
    if use_local_embeddings or not api_key:
        embeddings = FastEmbedEmbeddings()
        print("Usando embeddings locais (FastEmbed) - dimensão 384")
    else:
        embeddings = OpenAIEmbeddings()
        print("Usando embeddings OpenAI - dimensão 1536")

    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)
    modelo = None if no_llm else ChatOpenAI(temperature=0)
    return db, modelo


def make_chat_handler(db, modelo):
    def handle(message, history, top_only, debug):
        pergunta = (message or "").strip()
        if not pergunta:
            return "Digite uma pergunta."
        try:
            resposta = responder(pergunta, db, modelo, debug=bool(debug), top_only=bool(top_only))
            return resposta
        except Exception as e:
            return f"Erro ao processar a pergunta: {e}"
    return handle


def main():
    parser = argparse.ArgumentParser(description="Chat web para QA sobre PDFs")
    parser.add_argument("--local-embeddings", action="store_true", help="Usa embeddings locais (FastEmbed)")
    parser.add_argument("--no-llm", action="store_true", help="Não utilizar LLM; respostas por recuperação/extrator")
    parser.add_argument("--port", type=int, default=7860, help="Porta do servidor (default: 7860)")
    args = parser.parse_args()

    db, modelo = build_stack(use_local_embeddings=args.local_embeddings, no_llm=args.no_llm)
    handler = make_chat_handler(db, modelo)

    demo = gr.ChatInterface(
        fn=handler,
        title="PDF QA Chat",
        description=(
            "Faça perguntas sobre os PDFs da pasta base/. "
            "Use as opções para controlar a resposta: Top-only e Debug."
        ),
        additional_inputs=[
            gr.Checkbox(label="Top-only (melhor correspondência)", value=False),
            gr.Checkbox(label="Debug", value=False),
        ],
    )

    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()