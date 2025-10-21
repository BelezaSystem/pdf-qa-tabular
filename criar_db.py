import os
import shutil
import argparse
import json
import re
import unicodedata
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
# Modo tabular
import pdfplumber
# Utilitários compartilhados do main
from main import remove_acentos, normalizar, tokens, tokens_stemmed, classificacao_norm, extrair_anexo

load_dotenv()

PASTA_BASE = "base"
DB_DIR = "db"
ROWS_INDEX = os.path.join(DB_DIR, "rows.json")

# --- Normalização --- (utilizando utilitários do main.py)


# --- Criação do DB ---

def criar_db(chunk_size: int = 800, chunk_overlap: int = 150, force: bool = False, tabular: bool = False, normalize: bool = True, local_embeddings: bool = False):
    if force and os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR, ignore_errors=True)
        print(f"Removido diretório de base existente: {DB_DIR}")

    if not os.path.isdir(PASTA_BASE):
        print(f"Pasta de base não encontrada: {PASTA_BASE}")
        return

    if db_existe():
        print(f"Base vetorial já existe em '{DB_DIR}'. Pulando criação.")
        return

    if tabular:
        documentos = carregar_documentos_tabulares(PASTA_BASE, normalize=normalize)
    else:
        documentos = carregar_documentos()

    if not documentos:
        print("Nenhum documento encontrado para indexar.")
        return

    if tabular:
        chunks = documentos  # já é 1 linha = 1 documento
        print(f"Total de linhas indexadas como documentos: {len(chunks)}")
    else:
        chunks = dividir_chunks(documentos, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            print("Falha ao dividir documentos em chunks.")
            return

    vetorizar_chunks(chunks, use_local_embeddings=local_embeddings)

    # Persistir índice auxiliar se modo tabular
    if tabular:
        os.makedirs(DB_DIR, exist_ok=True)
        rows = [doc.metadata for doc in chunks if isinstance(doc, Document)]
        with open(ROWS_INDEX, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Índice auxiliar escrito em {ROWS_INDEX} com {len(rows)} entradas.")


# --- Utilitários ---

def db_existe():
    if not os.path.isdir(DB_DIR):
        return False
    try:
        return any(os.scandir(DB_DIR))
    except Exception:
        return True


def carregar_documentos():
    try:
        # Listagem explícita dos PDFs na pasta base (para diagnóstico)
        pdfs = [p for p in os.listdir(PASTA_BASE) if p.lower().endswith(".pdf")]
        print(f"PDFs na base: {len(pdfs)} -> {pdfs}")
        carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
        documentos = carregador.load()
        # Contagem por arquivo de origem
        src_counts = {}
        for d in documentos:
            src = d.metadata.get("source") or d.metadata.get("file_path") or "(desconhecido)"
            src_counts[src] = src_counts.get(src, 0) + 1
        print(f"Documentos carregados: {len(documentos)}; por arquivo: {src_counts}")
        return documentos
    except Exception as e:
        print(f"Erro ao carregar documentos: {e}")
        return []


def dividir_chunks(documentos, chunk_size: int, chunk_overlap: int):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = separador_documentos.split_documents(documentos)
    print(f"Total de chunks: {len(chunks)} (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def vetorizar_chunks(chunks, use_local_embeddings: bool = False):
    try:
        embeddings = FastEmbedEmbeddings() if use_local_embeddings else OpenAIEmbeddings()
        Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=DB_DIR,
        )
        print("Banco de dados vetorial criado em disco.")
    except Exception as e:
        print(f"Erro ao criar o banco de dados vetorial: {e}")


# --- Modo Tabular ---

# Helpers específicos removidos: usar main.classificacao_norm e main.extrair_anexo


def _doc_from_text(texto: str, pdf_path: str, page_ix: int, t_ix: int, r_ix: int, normalize: bool) -> Document:
    if not texto:
        return None
    texto_proc = remove_acentos(texto) if normalize else texto
    # Heurística: primeira célula/descrição
    atividade = texto.split("|")[0].strip() if "|" in texto else texto.strip()
    classif = classificacao_norm(texto)
    anexo = extrair_anexo(texto)
    meta = {
        "source": os.path.relpath(pdf_path),
        "page": page_ix,
        "table_index": t_ix,
        "row_index": r_ix,
        "atividade": atividade,
        "atividade_norm": normalizar(atividade),
        "tokens": " ".join(tokens(atividade)),
                    "tokens_stemmed": " ".join(tokens_stemmed(atividade)),
                    "has_estetic": ("estetic" in tokens_stemmed(atividade)),
        "classificacao": classif.upper() if classif else None,
        "classificacao_norm": classif,
        "anexo": ("ANEXO " + anexo) if anexo else None,
        "anexo_romano": anexo,
        "preview": texto[:220],
    }
    return Document(page_content=texto_proc, metadata=meta)


def carregar_documentos_tabulares(base_dir: str, normalize: bool = True) -> List[Document]:
    docs: List[Document] = []
    pdfs = [os.path.join(base_dir, p) for p in os.listdir(base_dir) if p.lower().endswith(".pdf")]
    print(f"[TABULAR] PDFs encontrados: {len(pdfs)} -> {[os.path.basename(p) for p in pdfs]}")
    if not pdfs:
        return docs

    for pdf_path in pdfs:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"[TABULAR] Processando: {os.path.basename(pdf_path)} (paginas={len(pdf.pages)})")
                for page_ix, page in enumerate(pdf.pages, start=1):
                    # 1) Tentar extrair tabelas estruturadas
                    tables = page.extract_tables() or []
                    for t_ix, table in enumerate(tables, start=1):
                        for r_ix, row in enumerate(table, start=1):
                            cells = [c.strip() if isinstance(c, str) else "" for c in row]
                            texto = " | ".join([c for c in cells if c])
                            d = _doc_from_text(texto, pdf_path, page_ix, t_ix, r_ix, normalize)
                            if d:
                                docs.append(d)
                    # 2) Fallback: extrair texto da página e dividir em linhas
                    texto_pg = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    for r_ix, linha in enumerate(texto_pg.splitlines(), start=1):
                        linha = linha.strip()
                        if not linha:
                            continue
                        ln_norm = normalizar(linha)
                        if len(linha) < 12 and "anexo" not in ln_norm:
                            continue
                        d = _doc_from_text(linha, pdf_path, page_ix, t_ix=0, r_ix=r_ix, normalize=normalize)
                        if d:
                            docs.append(d)
        except Exception as e:
            print(f"Falha ao processar '{pdf_path}': {e}")
    print(f"[TABULAR] Total de linhas/documentos extraídos: {len(docs)}")
    return docs


def main():
    parser = argparse.ArgumentParser(description="Criar base vetorial a partir de PDFs")
    parser.add_argument("--force", action="store_true", help="Recria a base removendo o diretório existente")
    parser.add_argument("--chunk-size", type=int, default=800, help="Tamanho dos chunks (default: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Sobreposição dos chunks (default: 150)")
    parser.add_argument("--tabular", action="store_true", help="Extrai linhas de tabelas com pdfplumber e indexa cada linha como documento")
    parser.add_argument("--no-normalize", action="store_true", help="Não aplicar normalização de texto (acentos) no modo tabular")
    parser.add_argument("--local-embeddings", action="store_true", help="Usa embeddings locais (FastEmbed) em vez de OpenAI")
    args = parser.parse_args()

    criar_db(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force=args.force,
        tabular=args.tabular,
        normalize=not args.no_normalize,
        local_embeddings=args.local_embeddings,
    )


if __name__ == "__main__":
    main()