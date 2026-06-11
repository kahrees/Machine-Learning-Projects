# HSRG PDF Summarizer — corrected script
# Bugs fixed:
#   1. doc_store/pipeline recreated on every Streamlit rerun  → @st.cache_resource
#   2. LLMGateway methods orphaned outside class              → fixed indentation
#   3. env_key stores key values instead of key names         → use string literals
#   4. Genre case mismatch ("Fiction" vs "fiction")           → .lower() at widget
#   5. Model name case mismatch ("Kimi" vs "kimi")            → .lower() at widget

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from haystack import Pipeline
from haystack.components.converters.pypdf import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from haystack_integrations.components.embedders.mistral.document_embedder import (
    MistralDocumentEmbedder,
)
from haystack_integrations.components.embedders.mistral.text_embedder import (
    MistralTextEmbedder,
)

load_dotenv()
warnings.filterwarnings("ignore")


# ── FIX 1: Cache the ingestion pipeline so it survives Streamlit reruns ───────
#
# Without @st.cache_resource every rerun creates a fresh InMemoryDocumentStore,
# so the chunks written during ingestion are immediately thrown away when the
# next widget interaction triggers a rerun. The decorator ensures the same
# pipeline + store instance is returned on every call after the first.

@st.cache_resource
def build_ingestion_pipeline():
    """
    Build and return the Haystack indexing pipeline alongside the document store.
    Called once per session; subsequent calls return the cached pair.
    """
    _doc_store = InMemoryDocumentStore()

    converter = PyPDFToDocument()
    cleaner   = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=False,
    )
    splitter  = DocumentSplitter(split_length=800, split_overlap=80)
    embedder  = MistralDocumentEmbedder(
        api_key=Secret.from_env_var("MISTRAL_EMBED_API_KEY"),
        model="mistral-embed",
    )
    writer    = DocumentWriter(document_store=_doc_store)

    _pipeline = Pipeline()
    _pipeline.add_component("converter", converter)
    _pipeline.add_component("cleaner",   cleaner)
    _pipeline.add_component("splitter",  splitter)
    _pipeline.add_component("embedder",  embedder)
    _pipeline.add_component("writer",    writer)

    _pipeline.connect("converter", "cleaner")
    _pipeline.connect("cleaner",   "splitter")
    _pipeline.connect("splitter",  "embedder")
    _pipeline.connect("embedder",  "writer")

    return _pipeline, _doc_store


# ── Upload helper ──────────────────────────────────────────────────────────────

def upload_pdfs():
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload files for summarization",
    )

    if not uploaded_files:
        return []

    temp_dir    = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
    saved_paths = []

    for upfile in uploaded_files:
        file_path = temp_dir / upfile.name
        with open(file_path, "wb") as f:
            f.write(upfile.getvalue())
        saved_paths.append(file_path)

    st.success(f"Saved {len(saved_paths)} file(s) to temporary storage")
    return saved_paths


# ── Ingestion ──────────────────────────────────────────────────────────────────

def ingestion(pipeline, sources):
    pipeline.warm_up()
    result = pipeline.run({"converter": {"sources": sources}})
    return result["writer"]["documents_written"]


# ── FIX 2: LLMGateway — all methods properly indented inside the class ─────────
#
# Previously __init__, _get_client, generate, and get_context_window were defined
# at module level (0 indentation) after a "# Generators" comment, making
# LLMGateway a class with only CONFIGS and no callable methods.
#
# FIX 3: env_key now stores the environment variable NAME (a string literal)
# rather than the resolved value.  _get_client calls os.environ.get(name),
# which requires the name, not the value it holds.

class LLMGateway:
    """
    Thin abstraction over the three LLM APIs.
    Each config entry points to an OpenAI-compatible endpoint.
    """

    CONFIGS = {
        "kimi": {
            "base_url": "https://openrouter.ai/api/v1",
            "model":    "moonshotai/kimi-k2.6:free",
            "env_key":  "KIMI_API_KEY",           # FIX 3: string name, not variable
        },
        "deepseek": {
            "base_url": "https://openrouter.ai/api/v1",
            "model":    "deepseek/deepseek-chat",
            "env_key":  "DEEPSEEK_API_KEY",        # FIX 3
        },
        "mistral": {
            "base_url": "https://api.mistral.ai/v1",
            "model":    "mistral-small-latest",
            "env_key":  "MISTRAL_API_KEY",         # FIX 3
        },
        # "qwen": {
        #     "base_url": "https://openrouter.ai/api/v1",
        #     "model":    "qwen/qwen3-235b-a22b:free",
        #     "env_key":  "QWEN_API_KEY",
        # },
    }

    def __init__(self):                            # FIX 2: 4-space indent = inside class
        self._clients: Dict[str, OpenAI] = {}

    def _get_client(self, model_name: str) -> OpenAI:   # FIX 2
        if model_name not in self._clients:
            cfg     = self.CONFIGS[model_name]
            api_key = os.environ.get(cfg["env_key"])    # FIX 3: looks up name correctly
            if not api_key:
                raise ValueError(
                    f"Missing API key: set {cfg['env_key']} in your .env file."
                )
            self._clients[model_name] = OpenAI(
                api_key=api_key,
                base_url=cfg["base_url"],
            )
        return self._clients[model_name]

    def generate(                                       # FIX 2
        self,
        prompt:      str,
        model_name:  str,
        max_tokens:  int   = 2000,
        temperature: float = 0.3,
    ) -> str:
        client   = self._get_client(model_name)
        cfg      = self.CONFIGS[model_name]
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def get_context_window(self, model_name: str) -> int:   # FIX 2
        windows = {
            "kimi":     2_000_000,
            "deepseek":    64_000,
            "mistral":    128_000,
            # "qwen":      32_000,
        }
        return windows.get(model_name, 32_000)


@st.cache_resource
def get_llm_gateway() -> LLMGateway:
    """
    Cache the gateway so OpenAI clients are created once and reused,
    rather than re-instantiated on every Streamlit rerun.
    """
    return LLMGateway()


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_comprehensive(document_store, genre="nonfiction", top_k=5):
    retriever  = InMemoryEmbeddingRetriever(document_store=document_store)
    q_embedder = MistralTextEmbedder(
        api_key=Secret.from_env_var("MISTRAL_EMBED_API_KEY"),
        model="mistral-embed",
    )

    # genre arrives already lowercased (normalised at the widget — FIX 4)
    if genre == "fiction":
        sub_queries = [
            "main characters and conflict",
            "plot progression and climax",
            "themes and setting",
            "ending and resolution",
        ]
    else:
        sub_queries = [
            "introduction and problem statement",
            "methodology and approach",
            "key results and evidence",
            "conclusions and recommendations",
        ]

    all_docs = []
    seen_ids = set()

    for sq in sub_queries:
        embedding = q_embedder.run(text=sq)["embedding"]
        result    = retriever.run(query_embedding=embedding, top_k=top_k)
        for doc in result["documents"]:
            if doc.id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc.id)

    all_docs.sort(key=lambda d: d.meta.get("page_number", 0))
    return all_docs


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_full_prompt(text, genre):
    if genre == "fiction":
        system = (
            "Provide a literary analysis alongside a comprehensive summary. "
            "Cover the protagonist's arc, major conflicts, themes, and resolution. "
            "No bullet points."
        )
    else:
        system = (
            "Provide an executive summary covering "
            "the central thesis, methodology, key findings, and conclusions. "
            "Use structured analysis and provide bullet points as needed."
        )
    return f"{system}\n\n{text}\n\nProvide a summary of all file(s)."


def _build_map_prompt(chunk, genre):
    if genre == "fiction":
        return (
            "Summarize this passage from a fictional work. Focus on plot events, "
            f"character actions, and setting:\n\n{chunk}"
        )
    return (
        "Summarize this passage from a non-fiction document. Focus on key "
        f"arguments, evidence, and conclusions:\n\n{chunk}"
    )


def _build_reduce_prompt(summaries, genre):
    combined = "\n\n---\n\n".join(summaries)
    if genre == "fiction":
        system = (
            "Synthesize these section summaries into a cohesive narrative overview. "
            "Preserve chronological order, identify the protagonist's arc, major "
            "conflicts, and thematic resolution. Write flowing prose."
        )
    else:
        system = (
            "Synthesize these section summaries into an executive summary. Identify "
            "the central thesis, key evidence, methodology highlights, and actionable "
            "conclusions. Use structured paragraphs."
        )
    return f"{system}\n\nSource summaries:\n\n{combined}\n\nProduce a section summary."


# ── Summarizer ─────────────────────────────────────────────────────────────────

def summarize(document_store, genre, model_name, gateway: LLMGateway):
    """
    Orchestrates whole-document summarisation.
    Uses Direct Mode if content fits the model's context window;
    otherwise falls back to Map-Reduce.

    genre and model_name must arrive already lowercased.
    gateway is the cached LLMGateway instance.
    """
    chunks = retrieve_comprehensive(document_store, genre=genre)

    if not chunks:
        return "No documents found. Please upload and index a PDF first."

    full_text     = "\n\n".join([d.content for d in chunks])
    rough_tokens  = len(full_text.split()) * 1.3
    context_limit = gateway.get_context_window(model_name)   # FIX 2: called on instance

    # Direct Mode (using 70 % of the context window as a safety buffer)
    if rough_tokens < (context_limit * 0.7):
        prompt = _build_full_prompt(full_text, genre)
        return gateway.generate(prompt, model_name, max_tokens=3000)  # FIX 2

    # Map-Reduce Mode
    chunk_summaries = []
    for chunk in chunks:
        mini_prompt   = _build_map_prompt(chunk.content, genre)
        mini_summary  = gateway.generate(mini_prompt, model_name, max_tokens=500)  # FIX 2
        chunk_summaries.append(mini_summary)

    reduce_prompt = _build_reduce_prompt(chunk_summaries, genre)
    return gateway.generate(reduce_prompt, model_name, max_tokens=3000)   # FIX 2


# ── Streamlit app ──────────────────────────────────────────────────────────────

# FIX 1: obtain the stable cached pipeline and document store
pipeline, doc_store = build_ingestion_pipeline()
gateway             = get_llm_gateway()

# Session state — engine_ready is the only flag needed here;
# doc_store is now managed by the cache, not session state.
if "engine_ready" not in st.session_state:
    st.session_state.engine_ready = False

st.set_page_config(page_title="HSRG PDF Summarizer", layout="wide")
st.title("HSRG PDF Summarizer")

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox(
    "Select LLM",
    options=["Kimi", "Deepseek", "Mistral"],
    index=0,
)

genre = st.sidebar.radio(
    "Content Type",
    options=["Nonfiction", "Fiction"],
    index=0,
    help="Determines chunking strategy and prompt templates",
)

# ── Upload ─────────────────────────────────────────────────────────────────────

st.header("Upload Documents")
upload = upload_pdfs()

# ── Index ──────────────────────────────────────────────────────────────────────

st.header("Index Documents")
ingest_disabled = len(upload) == 0

if st.button("Run Haystack Ingestion Pipeline", disabled=ingest_disabled):
    with st.spinner("Reading your document..."):
        docs_written = ingestion(pipeline, sources=upload)

    st.success(f"Pipeline complete: {docs_written} chunks indexed.")
    st.session_state.engine_ready = True

# ── Summarize ──────────────────────────────────────────────────────────────────

st.header("Generate Summary")

if not st.session_state.engine_ready:
    st.info("Upload and ingest PDFs to enable summarisation.")
else:
    if st.button("Summarize"):
        with st.spinner(f"Generating summary via {model_name}..."):
            summary = summarize(
                document_store=doc_store,
                genre=genre.lower(),            # FIX 4: "Fiction"   → "fiction"
                model_name=model_name.lower(),  # FIX 5: "Kimi"      → "kimi"
                gateway=gateway,                # FIX 2: pass instance
            )

        st.markdown("### Summary")
        st.markdown(summary)

        with st.expander("Retrieved Chunks (Debugging)"):
            docs = retrieve_comprehensive(
                doc_store,
                genre=genre.lower(),            # FIX 4 here too
            )
            for i, doc in enumerate(docs, 1):
                page = doc.meta.get("page_number", "N/A")
                st.markdown(f"**Chunk {i}** — Page {page}")
                st.text(doc.content[:300] + "...")
                st.divider()