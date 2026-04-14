"""
app.py  —  K-Drama Face Matcher
================================
Run:   python app.py
Open:  http://127.0.0.1:5000
"""

import os, sys, base64, pickle
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# ── make sure imports work from same folder ────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)          # always run relative to app.py location

from preprocessing       import ImagePreprocessor
from feature_extraction  import FeatureExtractor, build_feature_database
from similarity_matching import FaceSimilarityMatcher

app = Flask(__name__, static_folder=os.path.join(ROOT, "static"))
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

DB_PATH       = os.path.join(ROOT, "models", "feature_db.pkl")
DATASET_DIR   = os.path.join(ROOT, "dataset", "actors")
PROCESSED_DIR = os.path.join(ROOT, "dataset", "processed")

preprocessor = ImagePreprocessor()
extractor    = FeatureExtractor(method="HOG")
matcher      = None


def load_matcher():
    global matcher
    if os.path.exists(DB_PATH):
        matcher = FaceSimilarityMatcher(db_path=DB_PATH)
        print("✅ Feature database loaded.")
    else:
        matcher = None
        print("⚠️  No database yet — go to Setup tab after adding actor images.")


def img_to_b64(bgr):
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def actor_thumb(actor_key, size=(160, 160)):
    for base in [PROCESSED_DIR, DATASET_DIR]:
        folder = os.path.join(base, actor_key)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if files:
            img = cv2.imread(os.path.join(folder, files[0]))
            if img is not None:
                return img_to_b64(cv2.resize(img, size))
    return None


# ── routes ─────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/status")
def status():
    actors = 0
    if os.path.isdir(DATASET_DIR):
        actors = sum(1 for d in os.listdir(DATASET_DIR)
                     if os.path.isdir(os.path.join(DATASET_DIR, d)))
    return jsonify({"db_ready": os.path.exists(DB_PATH), "actor_count": actors})


@app.route("/setup", methods=["POST"])
def setup():
    from preprocessing      import preprocess_dataset
    from feature_extraction import build_feature_database
    if not os.path.isdir(DATASET_DIR) or not any(
        os.path.isdir(os.path.join(DATASET_DIR, d)) for d in os.listdir(DATASET_DIR)
    ):
        return jsonify({"error": "No actor folders found in dataset/actors/"}), 400
    try:
        preprocess_dataset(DATASET_DIR, PROCESSED_DIR)
        build_feature_database(PROCESSED_DIR, DB_PATH, method="HOG")
        load_matcher()
        return jsonify({"ok": True, "message": "Database built! Ready to match."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/match", methods=["POST"])
def match():
    if matcher is None:
        return jsonify({"error": "Database not ready. Go to Setup tab first."}), 503

    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image received."}), 400

    arr = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read image."}), 400

    face_vec, face_rect = preprocessor.process(img)

    # annotate
    ann = img.copy()
    if face_rect is not None:
        x, y, w, h = face_rect
        cv2.rectangle(ann, (x, y), (x+w, y+h), (40, 200, 100), 3)
        cv2.putText(ann, "You", (x, max(y-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 200, 100), 2)
    ah, aw = ann.shape[:2]
    if aw > 420:
        ann = cv2.resize(ann, (420, int(ah * 420 / aw)))

    feat    = extractor.extract(
        (face_vec * 255).astype(np.uint8) if face_vec.max() <= 1.0 else face_vec)
    raw     = matcher.match(feat, top_k=5, method="BOTH")
    hits    = raw.get("cosine") or raw.get("knn") or []

    results = []
    for actor_key, score in hits[:5]:
        results.append({
            "actor": actor_key.replace("_", " "),
            "score": round(float(score) * 100, 1),
            "thumb": actor_thumb(actor_key),
        })

    return jsonify({
        "user_image":    img_to_b64(ann),
        "face_detected": face_rect is not None,
        "results":       results,
    })


@app.route("/webcam_frame", methods=["POST"])
def webcam_frame():
    if matcher is None:
        return jsonify({"error": "Database not ready."}), 503
    data = request.get_json(silent=True) or {}
    raw  = base64.b64decode(data.get("frame", "").split(",")[-1])
    img  = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Bad frame."}), 400

    face_vec, _ = preprocessor.process(img)
    feat = extractor.extract(
        (face_vec * 255).astype(np.uint8) if face_vec.max() <= 1.0 else face_vec)
    hits = matcher.match(feat, top_k=3, method="cosine").get("cosine", [])

    return jsonify({"results": [
        {"actor": k.replace("_", " "),
         "score": round(float(s) * 100, 1),
         "thumb": actor_thumb(k, (100, 100))}
        for k, s in hits
    ]})


# ── start ──────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "models"),  exist_ok=True)
    os.makedirs(os.path.join(ROOT, "static"),  exist_ok=True)
    os.makedirs(DATASET_DIR,  exist_ok=True)
    load_matcher()
    print("\n🎬  K-Drama Face Matcher — http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
