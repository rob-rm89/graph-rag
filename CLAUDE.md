Project Overview
We are building a Unified Graph-RAG System bridging semantic text retrieval with visual knowledge mapping.

Tech Stack & Dependencies
Package Manager: uv

Core RAG Engine: lightrag-hku (Python)

Graph Storage: networkx

Vector Storage: NanoVectorDB

LLM Integration: OpenAI compatible (gpt-4o-mini for fast extraction, gpt-4o for high-level retrieval)

Data Serialization: Python json module adhering strictly to jsoncanvas.org spec 1.0.

Architectural Rules
Storage Initialization: You must instantiate LightRAG with explicit backend definitions and immediately await rag.initialize_storages() and await initialize_pipeline_status() to prevent async context errors.

Separation of Concerns: Ingestion logic, query processing, and visualization export must exist in isolated, modular Python files.

Canvas Coordinate Generation: The exporter MUST apply a force-directed layout algorithm (via NetworkX) to calculate absolute x and y integer coordinates for the JSON Canvas file. Nodes must have padded bounding boxes to prevent overlapping.

Testing
Ensure unique 16-character hex IDs for all JSON Canvas nodes.

Validate that all fromNode and toNode edge values reference existing entities.