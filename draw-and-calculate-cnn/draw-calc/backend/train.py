"""
Train the CNN.   python backend/train.py            (generated + real digits)
                 python backend/train.py --with-mine (also use drawings saved from the 'My handwriting' page)
Creates outputs/cnn_weights.npz and outputs/metrics.json
"""
import json, os, sys, time
import numpy as np
from sklearn.datasets import load_digits
from PIL import Image
from preprocess import normalize_char, CLASSES
import make_dataset
from cnn import CNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
K = len(CLASSES)

# 1) generated characters: digits and operators drawn with a virtual pen
Xs, ys = make_dataset.build(per_class=700)
print("Generated samples:", len(Xs))

# 2) real handwritten digits (1797 images from scikit-learn, written by 43 people)
d = load_digits()
Xr = np.array([normalize_char(np.asarray(Image.fromarray((im / 16 * 255).astype(np.uint8)).resize((56, 56), Image.BICUBIC))) for im in d.images], np.float32)[:, None]
yr = d.target.astype(np.int64)
print("Real handwritten digits:", len(Xr))


def split(n, a, b):
    idx = rng.permutation(n)
    return idx[:int(n * a)], idx[int(n * a):int(n * (a + b))], idx[int(n * (a + b)):]


s_tr, s_va, s_te = split(len(Xs), 0.8, 0.1)
r_tr, r_va, r_te = split(len(Xr), 0.6, 0.1)
Xtr = np.concatenate([Xs[s_tr], Xr[r_tr], Xr[r_tr]]); ytr = np.concatenate([ys[s_tr], yr[r_tr], yr[r_tr]])
Xva = np.concatenate([Xs[s_va], Xr[r_va]]); yva = np.concatenate([ys[s_va], yr[r_va]])

mine = os.path.join(ROOT, "data", "my_samples.json")
if "--with-mine" in sys.argv and os.path.exists(mine):
    S = json.load(open(mine))
    Xm = np.array([np.array(s["pixels"], np.float32).reshape(28, 28) / 255 for s in S])[:, None]
    ym = np.array([CLASSES.index(s["label"]) for s in S])
    Xtr = np.concatenate([Xtr] + [Xm] * 6); ytr = np.concatenate([ytr] + [ym] * 6)
    print("Added", len(S), "of my own drawings (x6)")


def augment(x):
    """random shift up to 2 px and a little noise, new every epoch"""
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.roll(x[i], (rng.integers(-2, 3), rng.integers(-2, 3)), axis=(1, 2))
    return np.clip(out + rng.normal(0, 0.015, x.shape).astype(np.float32), 0, 1)


def evaluate(net, X, y):
    P = np.concatenate([net.predict(X[i:i + 512]) for i in range(0, len(X), 512)])
    return P, float(np.mean(P.argmax(1) == y)), float(-np.log(P[np.arange(len(y)), y] + 1e-9).mean())


net = CNN(K)
print("Parameters:", net.n_params())
EPOCHS, BATCH, LR = 14, 64, 0.003
hist = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
best = (1e9, None, 0)
for ep in range(1, EPOCHS + 1):
    t0 = time.time()
    order = rng.permutation(len(Xtr))
    for i in range(0, len(order), BATCH):
        b = order[i:i + BATCH]
        net.train_step(augment(Xtr[b]), ytr[b], lr=LR * (0.85 ** ep))
    _, a, l = evaluate(net, Xtr[:3000], ytr[:3000])
    _, va, vl = evaluate(net, Xva, yva)
    for k, v in zip(hist, (l, vl, a, va)):
        hist[k].append(round(v, 4))
    if vl < best[0]:
        best = (vl, {k: v.copy() for k, v in net.P.items()}, ep)
    print(f"epoch {ep:2d}  loss={l:.4f}  val_loss={vl:.4f}  acc={a:.4f}  val_acc={va:.4f}  ({time.time() - t0:.0f}s)")

net.P = best[1]
print("Best epoch (lowest validation loss):", best[2])

Ps, acc_s, _ = evaluate(net, Xs[s_te], ys[s_te])
Pr, acc_r, _ = evaluate(net, Xr[r_te], yr[r_te])
Xte = np.concatenate([Xs[s_te], Xr[r_te]]); yte = np.concatenate([ys[s_te], yr[r_te]])
Pt = np.concatenate([Ps, Pr]); pred = Pt.argmax(1)
cm = np.zeros((K, K), int)
for t, p in zip(yte, pred):
    cm[t, p] += 1
per_class = {CLASSES[i]: round(float(cm[i, i] / max(cm[i].sum(), 1)), 4) for i in range(K)}
pairs = sorted([(int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(K) for j in range(K) if i != j and cm[i, j] > 0], reverse=True)[:6]
# evaluation extra: how well does it still do on shifted + noisy test images?
Xn = augment(Xs[s_te]); _, acc_noisy, _ = evaluate(net, Xn, ys[s_te])

# keep a small gallery of training images for the dashboard
gallery = {c: [(Xs[ys == i][j] * 255).astype(np.uint8).reshape(-1).tolist() for j in range(8)] for i, c in enumerate(CLASSES)}
metrics = {"classes": CLASSES, "params": net.n_params(), "epochs": EPOCHS, "batch_size": BATCH, "best_epoch": best[2], "history": hist,
           "counts": {"generated": int(len(Xs)), "real_digits": int(len(Xr)), "train": int(len(Xtr)), "val": int(len(Xva)), "test": int(len(Xte))},
           "test": {"overall_acc": round(float(np.mean(pred == yte)), 4), "generated_acc": round(acc_s, 4), "real_digits_acc": round(acc_r, 4),
                    "real_digits_n": int(len(r_te)), "shifted_noisy_acc": round(acc_noisy, 4)},
           "per_class": per_class, "confusion": cm.tolist(), "confused_pairs": [{"count": c, "true": a, "pred": b} for c, a, b in pairs],
           "gallery": gallery}
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"))
net.save(os.path.join(OUT, "cnn_weights.npz"))
print("\nTEST RESULTS")
for k, v in metrics["test"].items():
    print(f"  {k}: {v}")
print("  most confused:", metrics["confused_pairs"][:4])
