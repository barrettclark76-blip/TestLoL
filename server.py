"""
server.py — Flask backend for Draft Oracle
Serves the Random Forest champion recommender via a simple REST API.

Usage:
    pip install flask flask-cors joblib scikit-learn numpy
    python server.py

The server listens on http://localhost:5000
Update API_URL in index.html if you deploy this elsewhere.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from GitHub Pages or local file

# ── Load model & encoder ───────────────────────────────────────────────────────
# These files are produced by the Jupyter notebook (cells 14 + encode step)
try:
    rf        = joblib.load('rf_draft_model.pkl')
    champ_enc = joblib.load('champ_enc.pkl')
    NUM_CHAMPS = len(champ_enc.classes_)
    print(f'Model loaded. Vocabulary: {NUM_CHAMPS} champions.')
    print(f'RF classes: {len(rf.classes_)}')
except FileNotFoundError as e:
    print(f'ERROR: Could not load model files — {e}')
    print('Run the Jupyter notebook first to generate rf_draft_model.pkl and champ_enc.pkl')
    rf = None
    champ_enc = None


@app.route('/recommend', methods=['POST'])
def recommend():
    if rf is None or champ_enc is None:
        return jsonify({'error': 'Model not loaded. Run the notebook first.'}), 503

    data = request.get_json(force=True)

    allies   = data.get('allies',   [])
    enemies  = data.get('enemies',  [])
    top_k    = int(data.get('top_k', 5))

    # Validate
    if len(allies) != 4:
        return jsonify({'error': f'Expected 4 allies, got {len(allies)}'}), 400
    if len(enemies) != 5:
        return jsonify({'error': f'Expected 5 enemies, got {len(enemies)}'}), 400

    # Encode — allies sorted, enemies sorted (order-invariant, matches training)
    try:
        encoded = champ_enc.transform(sorted(allies) + sorted(enemies))
    except ValueError as e:
        unknown = str(e).split(': ')[-1]
        return jsonify({'error': f'Unknown champion name: {unknown}. Check spelling.'}), 400

    X_input = encoded.reshape(1, -1)

    # Get probabilities
    proba_compressed = rf.predict_proba(X_input).squeeze()

    # Map RF's compressed class list back to full vocabulary
    full_proba = np.zeros(NUM_CHAMPS)
    for idx, cls in enumerate(rf.classes_):
        full_proba[cls] = proba_compressed[idx]

    # Zero out already-drafted champions
    for idx in encoded:
        full_proba[idx] = 0.0

    # Top-k
    top_ids   = np.argsort(full_proba)[::-1][:top_k]
    top_names = champ_enc.inverse_transform(top_ids)

    recommendations = [
        {'champion': str(name), 'win_score': round(float(full_proba[i]), 6)}
        for name, i in zip(top_names, top_ids)
    ]

    return jsonify({
        'recommendations': recommendations,
        'position':        data.get('position', 'UNKNOWN'),
        'allies':          allies,
        'enemies':         enemies,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':     'ok' if rf is not None else 'model_missing',
        'champions':  NUM_CHAMPS if champ_enc else 0,
    })


@app.route('/champions', methods=['GET'])
def champions():
    """Return full champion list for autocomplete sync with frontend."""
    if champ_enc is None:
        return jsonify({'error': 'Model not loaded'}), 503
    return jsonify({'champions': champ_enc.classes_.tolist()})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
