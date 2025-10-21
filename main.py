import os
import argparse
import re
import unicodedata
from dotenv import load_dotenv
import nltk
from nltk.stem import RSLPStemmer
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Inicializar stemmer português
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/rslp')
except LookupError:
    nltk.download('rslp', quiet=True)

stemmer = RSLPStemmer()

CAMINHO_DB = "db"
SCORE_THRESHOLD = 0.4
K_RESULTADOS = 8

prompt_template = (
    "Você é um assistente que responde de forma objetiva e precisa.\n"
    "Use exclusivamente as informações fornecidas na base para responder.\n"
    "Se houver trechos pertinentes na base, responda usando-os rigorosamente (cite trechos).\n"
    "Somente diga que não encontrou dados suficientes se realmente não houver nenhum trecho relevante.\n\n"
    "Pergunta:\n{pergunta}\n\n"
    "Base de conhecimento:\n{base_conhecimento}\n\n"
    "Inclua referências das fontes usadas ao final."
)

# --- Utilidades de normalização / tokenização ---
STOPWORDS_PT = {
    "de","do","da","dos","das","e","ou","para","com","em","no","na","nos","nas","por","ao","aos","às","uma","um","o","a","os","as"
}

def remove_acentos(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizar(s: str) -> str:
    return remove_acentos(s).lower()

def tokens(s: str):
    return [t for t in re.findall(r"\w+", normalizar(s)) if len(t) > 2 and t not in STOPWORDS_PT]

def tokens_stemmed(s: str):
    """Tokenização com stemming para normalizar singular/plural"""
    base_tokens = tokens(s)
    return [stemmer.stem(t) for t in base_tokens]

# --- Helpers de classificação/anexo reutilizáveis ---
def classificacao_norm(s: str):
    s = normalizar(s)
    if "alto" in s:
        return "alto"
    if "medio" in s:
        return "medio"
    if "baixo" in s:
        return "baixo"
    return None

def extrair_anexo(s: str):
    s = normalizar(s)
    m = re.search(r"anexo\s+([ivxlcdm]+)", s)
    return m.group(1).upper() if m else None

# --- Recuperação ---

def obter_docs(db, pergunta):
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": SCORE_THRESHOLD, "k": K_RESULTADOS},
    )
    docs = retriever.invoke(pergunta)
    if docs:
        return docs, "threshold"

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.3},
    )
    docs = retriever.invoke(pergunta)
    if docs:
        return docs, "mmr"

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=K_RESULTADOS)
    docs = [doc for doc, _ in resultados]
    return docs, "similarity"


def obter_top_match(db, pergunta):
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=1)
    if not resultados:
        return None, None
    doc, score = resultados[0]
    return doc, score

# --- Construção de base ---

def construir_base_conhecimento(docs):
    if not docs:
        return None, []
    textos = []
    fontes = []
    for doc in docs:
        textos.append(doc.page_content)
        fonte = doc.metadata.get("source") or doc.metadata.get("file_path") or "Fonte desconhecida"
        fontes.append(fonte)
    base = "\n\n----\n\n".join(textos)
    return base, fontes

# --- Extrator tabular simples para TABELA SEMA ---

def extrair_linhas_relevantes(pergunta: str, docs, max_results: int = 5):
    ts = tokens(pergunta)
    ts_stemmed = tokens_stemmed(pergunta)
    if not ts:
        return []
    resultados = []
    for doc in docs:
        fonte = doc.metadata.get("source") or doc.metadata.get("file_path") or "Fonte desconhecida"
        for linha in doc.page_content.splitlines():
            ln = normalizar(linha)
            if not ln.strip():
                continue
            # score por cobertura de tokens (normal + stemmed)
            cobertura_normal = sum(1 for t in ts if t in ln)
            ln_tokens_stemmed = tokens_stemmed(linha)
            cobertura_stemmed = sum(1 for t in ts_stemmed if t in ln_tokens_stemmed)
            # usar o melhor score entre normal e stemmed
            cobertura = max(cobertura_normal, cobertura_stemmed)
            score = cobertura / max(1, len(ts))
            if score >= 0.5:
                risco_map = {"alto": "ALTO", "medio": "MÉDIO", "baixo": "BAIXO"}
                cls = classificacao_norm(ln)
                risco = risco_map.get(cls) if cls else None
                anexo = extrair_anexo(ln)
                resultados.append({
                    "texto": linha.strip(),
                    "fonte": fonte,
                    "score": score,
                    "risco": risco,
                    "anexo": anexo,
                })
    # deduplicar por texto e ordenar
    vistos = set()
    unicos = []
    for r in sorted(resultados, key=lambda x: x["score"], reverse=True):
        if r["texto"] not in vistos:
            vistos.add(r["texto"])
            unicos.append(r)
        if len(unicos) >= max_results:
            break
    return unicos

# --- Helpers ---

def _format_snippet(texto: str, max_len: int = 220) -> str:
    s = " ".join(texto.split())
    return s[:max_len] + ("…" if len(s) > max_len else "")

# --- Resposta ---

def responder(pergunta, db, modelo, debug: bool = False, top_only: bool = False):
    if top_only:
        doc, score = obter_top_match(db, pergunta)
        if not doc:
            return "Não consegui encontrar informação relevante na base para responder."
        fonte = doc.metadata.get("source") or doc.metadata.get("file_path") or "Fonte desconhecida"
        percent = f"{(score or 0)*100:.1f}%"
        # tentar extrair linhas da melhor fonte
        matches = extrair_linhas_relevantes(pergunta, [doc], max_results=3)
        if matches:
            linhas_fmt = []
            for m in matches:
                detalhes = []
                if m["risco"]:
                    detalhes.append(f"Classificação: {m['risco']}")
                if m["anexo"]:
                    detalhes.append(f"Anexo: ANEXO {m['anexo']}")
                detalhes_str = f" ({'; '.join(detalhes)})" if detalhes else ""
                linhas_fmt.append(f"- {m['texto']}{detalhes_str}\n  Fonte: {m['fonte']}")
            return (
                f"Melhor correspondência (similaridade {percent}):\n" +
                "\n".join(linhas_fmt) +
                "\n\nReferências:\n- " + fonte
            )
        # fallback: mostrar preview do doc
        return (
            f"Melhor correspondência (similaridade {percent}):\n" +
            f"Fonte: {fonte}\n" +
            f"Preview: {_format_snippet(doc.page_content)}\n\n" +
            "Referências:\n- " + fonte
        )

    # modo padrão
    docs, estrategia = obter_docs(db, pergunta)
    base_conhecimento, fontes = construir_base_conhecimento(docs)

    if debug:
        print(f"[DEBUG] Estratégia: {estrategia} | docs={len(docs)} | threshold={SCORE_THRESHOLD} | k={K_RESULTADOS}")
        for i, d in enumerate(docs[:5], start=1):
            fonte = d.metadata.get("source") or d.metadata.get("file_path") or "Fonte desconhecida"
            print(f"[DEBUG] Doc#{i} fonte={fonte} preview= {_format_snippet(d.page_content)}")

    matches = extrair_linhas_relevantes(pergunta, docs, max_results=3)
    if matches:
        linhas_fmt = []
        for m in matches:
            detalhes = []
            if m["risco"]:
                detalhes.append(f"Classificação: {m['risco']}")
            if m["anexo"]:
                detalhes.append(f"Anexo: ANEXO {m['anexo']}")
            detalhes_str = f" ({'; '.join(detalhes)})" if detalhes else ""
            linhas_fmt.append(f"- {m['texto']}{detalhes_str}\n  Fonte: {m['fonte']}")
        resposta = (
            "Com base na tabela recuperada, encontrei registros diretamente relacionados:\n" +
            "\n".join(linhas_fmt)
        )
        fontes_unicas = list(dict.fromkeys([m["fonte"] for m in matches]))
        if fontes_unicas:
            resposta += "\n\nReferências:\n- " + "\n- ".join(fontes_unicas)
        return resposta

    if base_conhecimento is None:
        return "Não consegui encontrar informação relevante na base para responder."

    # Modo sem LLM: compõe resposta a partir das prévias recuperadas
    if modelo is None:
        previews = []
        for d in docs[:3]:
            fonte = d.metadata.get("source") or d.metadata.get("file_path") or "Fonte desconhecida"
            previews.append(f"- Fonte: {fonte}\n  Preview: {_format_snippet(d.page_content)}")
        resposta = "Prévia dos documentos mais relevantes:\n" + "\n".join(previews)
        fontes_unicas = list(dict.fromkeys(fontes))
        if fontes_unicas:
            resposta += "\n\nReferências:\n- " + "\n- ".join(fontes_unicas)
        return resposta

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | modelo | StrOutputParser()
    texto_resposta = chain.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})

    fontes_unicas = list(dict.fromkeys(fontes))
    if fontes_unicas:
        texto_resposta += "\n\nReferências:\n- " + "\n- ".join(fontes_unicas)
    return texto_resposta

# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Assistente de QA sobre PDFs")
    parser.add_argument("--debug", action="store_true", help="Imprime logs de recuperação e prévias dos documentos")
    parser.add_argument("--top-only", action="store_true", help="Retorna apenas a melhor correspondência com porcentagem de similaridade")
    parser.add_argument("--local-embeddings", action="store_true", help="Usa embeddings locais (FastEmbed) em vez de OpenAI")
    parser.add_argument("--no-llm", action="store_true", help="Não usar LLM; resposta só com recuperação/extrator")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")

    # Seleção de embeddings (OpenAI ou locais)
    if args.local_embeddings:
        embeddings = FastEmbedEmbeddings()
    else:
        if not api_key:
            print("OPENAI_API_KEY não encontrado; usando embeddings locais (FastEmbed).")
            embeddings = FastEmbedEmbeddings()
        else:
            embeddings = OpenAIEmbeddings()

    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)

    # Seleção de LLM (ou modo sem LLM)
    modelo = None if args.no_llm else ChatOpenAI(temperature=0)
    if modelo is None and not args.no_llm:
        print("Aviso: sem OPENAI_API_KEY, o modo LLM não está disponível. Use --no-llm para operar sem LLM.")

    print("Digite sua pergunta (ou 'sair' para encerrar).")
    while True:
        pergunta = input("> ").strip()
        if not pergunta or pergunta.lower() == "sair":
            print("Até mais!")
            break

        resposta = responder(pergunta, db, modelo, debug=args.debug, top_only=args.top_only)
        print("\nResposta da IA:\n" + resposta + "\n")


if __name__ == "__main__":
    main()
