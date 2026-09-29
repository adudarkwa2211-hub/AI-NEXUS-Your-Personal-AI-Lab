# RAG Chatbot

Campus assistant module: **multilingual MiniLM → FAISS → Qwen2.5**, served by **FastAPI** and a **React** chat UI.

```
User question
  → multilingual MiniLM embedding
  → FAISS similarity search
  → retrieve relevant context
  → Qwen2.5
  → FastAPI JSON
  → React chat UI
```

## Requirements

- Python **3.11 or 3.12** (3.14 is not recommended: PyTorch/FAISS wheels are often missing)
- Node.js 18+
- Several GB of disk for Hugging Face model files (`paraphrase-multilingual-MiniLM-L12-v2` and `Qwen/Qwen2.5-0.5B-Instruct`)

## Configuration

Copy `.env.example` to `.env`. Paths and model names are read from that file, not hardcoded in request handlers.

| Variable | Meaning |
|---|---|
| `EMBEDDING_MODEL` | Must be multilingual MiniLM (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) |
| `LLM_MODEL` | Qwen2.5 Instruct checkpoint (default `Qwen/Qwen2.5-0.5B-Instruct` for CPU) |
| `DOCUMENTS_DIR` | Knowledge-base folder (`.txt` / `.md`) |
| `INDEX_DIR` | Where `index.faiss` and `metadata.json` are stored |
| `TOP_K` | How many FAISS neighbours to retrieve |

To use a larger Qwen2.5 model on a GPU machine, set `LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct` (or `3B` / `7B-Instruct`) in `.env`. Do not switch to a different model family.

## Backend setup

From the repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

On CUDA, install the GPU build of `torch` instead of the CPU index URL.

### Build the FAISS index

```powershell
cd backend
python scripts\ingest.py
```

This loads campus documents, chunks them, embeds them with MiniLM, and writes `backend/data/indexes/index.faiss` plus `metadata.json`.

### Check that the index loads

```powershell
python scripts\load_index.py
```

### Run tests

```powershell
cd backend
python -m pytest -q
```

Retrieval tests use FAISS with a deterministic encoder so they do not download Hugging Face models. API tests use FastAPI `TestClient` and a fake pipeline for validation and error handling.

### Run the API

Models are loaded **once** at process start (`RAGPipeline.load`), not on every chat request.

```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- `GET /api/health`
- `POST /api/chat` body: `{ "message": "When is the library open?", "history": [] }`
- `POST /api/detect` multipart form: image field `file`, optional `confidence` from 0.01 to 1.0. Uses YOLO11n pretrained on COCO; first detection request downloads `yolo11n.pt`.

### Object detection and evaluation

Open the **Object detection** tab in the React app, choose an image (up to 8 MB), and set the confidence threshold. The result includes an annotated image, detected COCO labels, confidence scores, and bounding boxes. Model weights are loaded once on the first request and cached for the process.

Evaluate the pretrained model on COCO128 and report both mAP metrics:

```powershell
cd backend
python scripts\evaluate_detector.py
```

Ultralytics downloads COCO128 when needed. Results are printed and saved as `backend/data/metrics/detector_metrics.json`.

## Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 . Vite proxies `/api` to FastAPI on port 8000.

## Project layout

- `backend/app/rag/` — MiniLM embedder, FAISS store, retriever, RAG prompt, Qwen2.5, pipeline
- `backend/app/detector/` — YOLO11n inference wrapper
- `backend/app/api/detect.py` — image upload and detection endpoint
- `backend/scripts/evaluate_detector.py` — COCO128 mAP50 / mAP50-95 evaluation
- `backend/scripts/ingest.py` — build the index
- `backend/scripts/load_index.py` — load the index
- `backend/data/documents/` — knowledge base
- `frontend/src/` — React chat UI

## Other AI modules

The repo contains a RAG chatbot and a YOLO11n object detector. Additional FastAPI routers can be mounted next to `/api/chat` and `/api/detect` in `backend/app/main.py` without changing either pipeline.
