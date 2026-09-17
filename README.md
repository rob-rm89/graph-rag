# Unified Graph-RAG System

A modular Python pipeline that bridges semantic text retrieval with visual knowledge mapping:

* **Ingestion** (`ingestion.py`) indexes academic text into a persistent local
  [LightRAG](https://github.com/HKUDS/LightRAG) store and forces the extraction of
  **LitGraph bibliographic metadata** (papers, authors, venues, years, cited works) as
  explicit graph nodes and edges alongside semantic entities.
* **Retrieval** (`query.py`) exposes an async interface over LightRAG's dual-level index and
  demonstrates the `hybrid` (low-level entities + high-level themes) and `mix` (graph +
  dense chunk retrieval) query modes.
* **Visual translation** (`canvas_exporter.py`) reads the `NetworkXStorage` graph, runs a
  Fruchterman-Reingold layout, removes bounding-box overlaps, and writes an
  Obsidian-compatible **JSON Canvas 1.0** file.
* **Verification** (`verify_integrity.py`) is a standalone checker for the exported canvas.

Input documents may be plain text (`.txt`, `.md`) or PDF (`.pdf`, text layer extracted with
pypdf; scanned PDFs without a text layer are skipped with a warning).

Storage backends are explicit and local: `NetworkXStorage` (graph), `NanoVectorDBStorage`
(vectors), `JsonKVStorage` and `JsonDocStatusStorage` (key/value and document status).

## Layout

| File | Responsibility |
|---|---|
| `config.py` | `Settings` (from `.env`), LitGraph entity-type registry, extraction-prompt guidance, canvas colour presets |
| `llm.py` | OpenAI-compatible LLM/embedding adapters; routes extraction to `gpt-4o-mini` and answers to `gpt-4o` via LightRAG role configs |
| `rag_factory.py` | `build_rag()` / `rag_session()`: explicit storages, `initialize_storages()` + `initialize_pipeline_status()` |
| `ingestion.py` | Profiling prompt -> `BibliographicRecord` -> `ainsert_custom_kg` -> `ainsert` |
| `query.py` | `QueryInterface.hybrid()`, `.mix()`, `.retrieve()`, `.compare_modes()` |
| `canvas_exporter.py` | `CanvasExporter`: layout, overlap removal, JSON Canvas serialisation |
| `verify_integrity.py` | Canvas integrity CLI (exit code 0/1) |
| `offline_backend.py` | Deterministic stub LLM, hashing embeddings and codepoint tokenizer for key-free runs |
| `main.py` | CLI orchestrator: `ingest`, `query`, `export`, `all` |
| `data/` | Input corpus (`.txt`, `.md`, `.pdf` via pypdf); ships with a synthetic sample paper |
| `output/` | Exported `knowledge_graph.canvas` |
| `tests/` | Unit tests plus an offline end-to-end integration test |

## Quick start

```bash
uv sync
```

Run the whole pipeline **without any API key** using the deterministic offline backend:

```bash
uv run python main.py --backend stub all
```

This ingests `data/`, answers the demo questions in `hybrid` and `mix` mode, writes
`output/knowledge_graph.canvas`, and prints the integrity report. Open the canvas by
copying it into any Obsidian vault.

To use a real model, copy `.env.example` to `.env`, set `OPENAI_API_KEY` (and optionally
`OPENAI_BASE_URL` for an OpenAI-compatible endpoint), then:

```bash
uv run python main.py ingest
uv run python main.py query "Who authored the paper and where was it published?" --mode both
uv run python main.py export --out output/knowledge_graph.canvas
uv run python verify_integrity.py output/knowledge_graph.canvas
```

Each module also runs standalone (`python ingestion.py`, `python query.py "..."`,
`python canvas_exporter.py --working-dir rag_storage`).

## LitGraph schema

| Kind | Entity types | Canvas colour |
|---|---|---|
| Bibliographic | `Paper` (6 purple), `Author` (4 green), `Venue` (5 cyan), `Year` (3 yellow), `CitedWork` (1 red) | per type |
| Semantic | `Concept`, `Method`, `Dataset`, `Metric`, `Organization` | 2 (orange) |

Relationships written by the profiling pass use the keywords `authored_by`, `published_in`,
`published_year`, `cites` and `affiliated_with`. Edges touching a bibliographic node are
green ("4"), purely semantic edges are orange ("2"); every edge has `toEnd: "arrow"` and
points away from the paper.

Extraction is steered twice ("dual pass"):

1. `config.build_entity_types_guidance()` is injected into LightRAG's entity-extraction
   prompt via `addon_params["entity_types_guidance"]`, biasing the main pipeline towards the
   types above.
2. `ingestion.py` runs a dedicated JSON profiling prompt per document and appends the result
   through `LightRAG.ainsert_custom_kg()` **before** the standard `ainsert()`, so the
   bibliographic nodes are guaranteed to exist and are then enriched by extraction.

## Entity reconciliation and cross-paper citation linking

LightRAG merges graph nodes only on exact name equality, so `reconciliation.py` keeps a
persistent registry (`<WORKING_DIR>/litgraph_registry.json`) of canonical names and resolves
every profiled record against it before anything is written:

* **Authors**: `"E. Marchetti"`, `"Marchetti, Elena"` and `"Elena Marchetti"` resolve to one
  node; when a fuller spelling arrives later the old node is merged into it. Initials that
  match several known authors (`"J. Smith"` vs. Jane and John) are deliberately left apart.
* **Works**: titles are matched by DOI, by normalised text, then fuzzily. A reference to an
  already ingested paper links straight to that paper's node (`cites` edge) instead of
  creating a `CitedWork` twin, and a paper ingested after being cited upgrades its
  `CitedWork` node to a `Paper`.
* **Venues / organisations** collapse on normalised names.
* Entities the registry already knows receive relationships only, so an existing node's
  description is never overwritten by a later document.

After each ingest run a graph-wide pass also merges near-duplicates produced by LightRAG's
own extraction (via `LightRAG.amerge_entities`, which redirects edges and updates the vector
store). Disable it with `RECONCILE_GRAPH=0`, or run it on demand:

```bash
uv run python main.py reconcile
```

## JSON Canvas output

* Every LightRAG entity becomes a `text` node with a Markdown body
  (`# name`, `**Type:** ...`, description, source file).
* `x`, `y`, `width`, `height` are strict integers; ids are unique 16-character hex strings.
* Layout: per-component `networkx.spring_layout` (Fruchterman-Reingold, seeded) -> landscape
  normalisation (tall components are transposed) -> isotropic pixel scaling -> deterministic
  overlap removal with padded boxes -> shelf packing of components -> grid snapping.
  `--algorithm forceatlas2` is available on NetworkX >= 3.4.
* Open the result in Obsidian by copying the `.canvas` file into any vault (the demo file was
  placed at `Vault/Graph RAG/knowledge_graph.canvas`).
* `--max-nodes N` keeps the N highest-degree entities for very large graphs.

## Verification

```bash
uv run pytest -q          # unit tests + offline end-to-end integration test
uv run ruff check .       # lint
uv run python verify_integrity.py output/knowledge_graph.canvas
```

The integration test constructs a *real* LightRAG instance with the stub backend, ingests
the sample paper, queries in both modes, exports the canvas and verifies it. The live
OpenAI path shares all code except the LLM/embedding adapters and has not been exercised
against a live model in this repository.

## Notes on lightrag-hku 1.5.7

* `PROMPTS["entity_extraction"]`, `DEFAULT_ENTITY_TYPES` and `addon_params["entity_types"]`
  from older documentation no longer exist; `entity_types_guidance` is the supported hook.
* `QueryParam` has no per-query model override; per-stage routing uses `role_llm_configs`.
* LightRAG binds `hashing_kv` into every `llm_model_func` call, so adapters accept `**kwargs`.
* `NetworkXStorage` is single-writer: never run two processes against the same
  `WORKING_DIR`, and do not keep the GraphML file open in another program during ingestion.
* `ainsert_custom_kg` sits outside LightRAG's document-level crash recovery.
