# MalariaDetect AI — FastAPI Backend

Ensemble ML API for malaria detection. Deploy to **Render.com**.

## Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/predict` | Run diagnosis |
| GET | `/metrics` | Model metrics |
| GET | `/docs` | Swagger UI |

## Deploy to Render
1. Push this `backend/` folder to a GitHub repo
2. Go to render.com → New Web Service → connect repo
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Copy the URL (e.g. `https://malaria-detect-api.onrender.com`)
6. Paste it into `API_BASE` in `frontend/index.html`

## Local Development
```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Docs at: http://localhost:8000/docs
```
