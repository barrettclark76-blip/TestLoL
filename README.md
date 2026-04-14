# Draft Oracle — LoL Champion Recommender

A web app that recommends the best League of Legends champion to pick given your 4 teammates' picks and all 5 enemy picks, ranked by win-score from a Random Forest classifier trained on ranked match data.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Frontend — GitHub Pages web app |
| `server.py` | Flask backend — serves the RF model via REST API |
| `lol_champion_classifier.ipynb` | Jupyter notebook — trains and saves the model |

## Setup

### 1. Train the model

Open `lol_champion_classifier.ipynb` and run all cells with your `MATCHDATA.CSV` in the same directory. This produces two files:

```
rf_draft_model.pkl
champ_enc.pkl
```

### 2. Run the backend

```bash
pip install flask flask-cors joblib scikit-learn numpy
python server.py
```

The server starts at `http://localhost:5000`.

### 3. Open the frontend

Open `index.html` via a local server (not `file://` — the fetch call needs HTTP):

```bash
# Python 3
python -m http.server 8080
# then open http://localhost:8080
```

The app runs in **demo mode** (placeholder results) if no backend is reachable.

## Deploying the backend

To make the app fully live on GitHub Pages, deploy `server.py` to a public host and update `API_URL` in `index.html`:

| Platform | Notes |
|----------|-------|
| [Render](https://render.com) | Free tier, easy Flask deploy |
| [Railway](https://railway.app) | Fast deploy from GitHub |
| [Hugging Face Spaces](https://huggingface.co/spaces) | Good for ML models |

After deploying, change this line in `index.html`:

```js
const API_URL = 'https://your-app.onrender.com/recommend';
```

## GitHub Pages

Push `index.html` to your repo and enable GitHub Pages (Settings → Pages → Deploy from branch `main`, folder `/root`). The frontend is fully static — only `server.py` needs a separate host.

## API

`POST /recommend`

```json
{
  "allies":   ["Jinx", "Thresh", "Camille", "Vi"],
  "enemies":  ["Ahri", "Zed", "Jinx", "Leona", "Garen"],
  "position": "MIDDLE",
  "top_k":    5
}
```

Response:

```json
{
  "recommendations": [
    { "champion": "Orianna", "win_score": 0.082 },
    { "champion": "Viktor",  "win_score": 0.069 }
  ]
}
```

`GET /health` — model status  
`GET /champions` — full champion list for autocomplete
