"""Flask backend. Run: python backend/app.py  ->  http://127.0.0.1:5000"""
import base64, io, json, os, time
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory
from preprocess import CLASSES, normalize_char, split_characters
from calculator import calculate
from cnn import CNN

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT = os.path.join(ROOT, "outputs")
LOG = os.path.join(ROOT, "data", "my_samples.json")
if not os.path.exists(os.path.join(OUT, "cnn_weights.npz")):
    raise SystemExit("Run `python backend/train.py` first.")

app = Flask(__name__, static_folder=os.path.join(ROOT, "frontend"), static_url_path="")
NET = CNN.load(os.path.join(OUT, "cnn_weights.npz"))
METRICS = json.load(open(os.path.join(OUT, "metrics.json")))


def decode(data_url):
    img = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, "white"); bg.paste(img, mask=img.split()[-1]); img = bg
    return 255 - np.asarray(img.convert("L"), dtype=np.uint8)          # pen = 255, paper = 0


def classify(x):
    p = NET.predict(x.reshape(1, 1, 28, 28).astype(np.float32))[0]
    order = np.argsort(-p)
    return p, [{"char": CLASSES[i], "prob": round(float(p[i]), 4)} for i in order[:3]]


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/info")
def info():
    return jsonify(METRICS)


@app.route("/api/read", methods=["POST"])
def read():
    ink = decode(request.get_json(force=True)["image"])
    segs = split_characters(ink)
    if not segs:
        return jsonify({"error": "I can't see any drawing yet."}), 400
    if len(segs) > 30:
        return jsonify({"error": "Too many characters. Draw fewer than 30."}), 400
    chars = []
    for a, b in segs:
        cols = ink[:, a:b + 1]
        rows = np.where((cols > 60).any(axis=1))[0]
        x = normalize_char(cols[rows.min():rows.max() + 1])
        p, top = classify(x)
        chars.append({"char": top[0]["char"], "confidence": top[0]["prob"], "top": top, "box": [int(a), int(b)],
                      "pixels": (x * 255).astype(np.uint8).reshape(-1).tolist(), "uncertain": top[0]["prob"] < 0.75})
    out = {"chars": chars}
    try:
        out["calc"] = calculate([c["char"] for c in chars])
    except ValueError as e:
        out["calc_error"] = str(e)
    return jsonify(out)


@app.route("/api/calculate", methods=["POST"])
def calc():
    try:
        return jsonify(calculate(request.get_json(force=True)["chars"]))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/inspect", methods=["POST"])
def inspect():
    x = np.array(request.get_json(force=True)["pixels"], np.float32).reshape(1, 1, 28, 28) / 255
    p = NET.predict(x)[0]
    a1, a2 = NET.feature_maps(x)
    q = lambda m: [(255 * (f / max(f.max(), 1e-6))).astype(np.uint8).tolist() for f in m]
    return jsonify({"probs": {c: round(float(v), 4) for c, v in zip(CLASSES, p)}, "conv1": q(a1), "conv2": q(a2)})


def tally():
    S = json.load(open(LOG)) if os.path.exists(LOG) else []
    per = {}
    for s in S:
        d = per.setdefault(s["label"], [0, 0]); d[0] += 1; d[1] += s["label"] == s["pred"]
    ok = sum(s["label"] == s["pred"] for s in S)
    return {"total": len(S), "correct": int(ok), "accuracy": round(ok / len(S), 4) if S else None,
            "per_char": {k: {"n": v[0], "correct": int(v[1])} for k, v in sorted(per.items())},
            "recent": [{"label": s["label"], "pred": s["pred"], "conf": s["conf"]} for s in S[-8:]][::-1]}


@app.route("/api/selftest", methods=["GET", "POST"])
def selftest():
    if request.method == "GET":
        return jsonify(tally())
    d = request.get_json(force=True)
    if d["label"] not in CLASSES:
        return jsonify({"error": "bad label"}), 400
    ink = decode(d["image"])
    if (ink > 60).sum() < 30:
        return jsonify({"error": "Draw the character first."}), 400
    x = normalize_char(ink)
    p, top = classify(x)
    S = json.load(open(LOG)) if os.path.exists(LOG) else []
    S.append({"label": d["label"], "pred": top[0]["char"], "conf": top[0]["prob"], "ts": int(time.time()),
              "pixels": (x * 255).astype(np.uint8).reshape(-1).tolist()})
    json.dump(S, open(LOG, "w"))
    return jsonify({"pred": top[0]["char"], "conf": top[0]["prob"], "top": top, "right": top[0]["char"] == d["label"], "tally": tally()})


if __name__ == "__main__":
    app.run(debug=False)
