"""
A Convolutional Neural Network written from scratch with NumPy (no deep learning library).

  Input 1x28x28
  -> Conv2D(8 filters, 3x3) + ReLU        -> 8x26x26
  -> MaxPool 2x2                           -> 8x13x13
  -> Conv2D(16 filters, 3x3) + ReLU       -> 16x11x11
  -> MaxPool 2x2                           -> 16x5x5
  -> Flatten (400) -> Dense(64) + ReLU -> Dropout(0.3)
  -> Dense(15) + Softmax                   -> one probability per character
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv


def conv_forward(x, W, b):
    win = swv(x, W.shape[2:], axis=(2, 3))                       # N,C,H',W',kh,kw
    out = np.tensordot(win, W, axes=([1, 4, 5], [1, 2, 3]))     # N,H',W',F
    return out.transpose(0, 3, 1, 2) + b.reshape(1, -1, 1, 1), win


def conv_backward(dout, win, W, x_shape):
    kh, kw = W.shape[2:]
    dW = np.tensordot(dout, win, axes=([0, 2, 3], [0, 2, 3]))   # F,C,kh,kw
    db = dout.sum(axis=(0, 2, 3))
    dp = np.pad(dout, ((0, 0), (0, 0), (kh - 1, kh - 1), (kw - 1, kw - 1)))
    dx = np.tensordot(swv(dp, (kh, kw), axis=(2, 3)), W[:, :, ::-1, ::-1], axes=([1, 4, 5], [0, 2, 3])).transpose(0, 3, 1, 2)
    return dx, dW, db


def pool_forward(x):
    N, C, H, W = x.shape
    h2, w2 = H // 2, W // 2
    xr = x[:, :, :h2 * 2, :w2 * 2].reshape(N, C, h2, 2, w2, 2)
    out = xr.max(axis=(3, 5))
    return out, (xr, out, x.shape)


def pool_backward(dout, cache):
    xr, out, shape = cache
    mask = (xr == out[:, :, :, None, :, None]).astype(np.float32)
    mask /= mask.sum(axis=(3, 5), keepdims=True)
    dxr = mask * dout[:, :, :, None, :, None]
    N, C, h2, _, w2, _ = dxr.shape
    dx = np.zeros(shape, np.float32)
    dx[:, :, :h2 * 2, :w2 * 2] = dxr.reshape(N, C, h2 * 2, w2 * 2)
    return dx


class CNN:
    def __init__(self, n_classes=15, seed=0):
        r = np.random.default_rng(seed)
        he = lambda shape, fan: (r.normal(0, np.sqrt(2 / fan), shape)).astype(np.float32)
        self.P = {"W1": he((8, 1, 3, 3), 9), "b1": np.zeros(8, np.float32),
                  "W2": he((16, 8, 3, 3), 72), "b2": np.zeros(16, np.float32),
                  "W3": he((400, 64), 400), "b3": np.zeros(64, np.float32),
                  "W4": he((64, n_classes), 64), "b4": np.zeros(n_classes, np.float32)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0
        self.rng = r

    def n_params(self):
        return int(sum(v.size for v in self.P.values()))

    def forward(self, x, train=False):
        P = self.P
        z1, w1 = conv_forward(x, P["W1"], P["b1"]); a1 = np.maximum(z1, 0)
        p1, c1 = pool_forward(a1)
        z2, w2 = conv_forward(p1, P["W2"], P["b2"]); a2 = np.maximum(z2, 0)
        p2, c2 = pool_forward(a2)
        f = p2.reshape(len(x), -1)
        z3 = f @ P["W3"] + P["b3"]; a3 = np.maximum(z3, 0)
        drop = (self.rng.random(a3.shape) > 0.3).astype(np.float32) / 0.7 if train else 1.0
        a3d = a3 * drop
        z4 = a3d @ P["W4"] + P["b4"]
        e = np.exp(z4 - z4.max(1, keepdims=True))
        probs = e / e.sum(1, keepdims=True)
        return probs, dict(x=x, a1=a1, w1=w1, c1=c1, p1=p1, a2=a2, w2=w2, c2=c2, p2=p2, f=f, a3=a3, drop=drop, a3d=a3d)

    def predict(self, x):
        return self.forward(x)[0]

    def train_step(self, x, y, lr=0.002, b1=0.9, b2=0.999, eps=1e-8):
        P = self.P
        probs, c = self.forward(x, train=True)
        n = len(x)
        loss = float(-np.log(probs[np.arange(n), y] + 1e-9).mean())
        d4 = probs.copy(); d4[np.arange(n), y] -= 1; d4 /= n               # softmax + cross-entropy
        g = {"W4": c["a3d"].T @ d4, "b4": d4.sum(0)}
        d3 = (d4 @ P["W4"].T) * c["drop"] * (c["a3"] > 0)
        g["W3"], g["b3"] = c["f"].T @ d3, d3.sum(0)
        dp2 = (d3 @ P["W3"].T).reshape(c["p2"].shape)
        da2 = pool_backward(dp2, c["c2"]) * (c["a2"] > 0)
        dp1, g["W2"], g["b2"] = conv_backward(da2, c["w2"], P["W2"], c["p1"].shape)
        da1 = pool_backward(dp1, c["c1"]) * (c["a1"] > 0)
        _, g["W1"], g["b1"] = conv_backward(da1, c["w1"], P["W1"], c["x"].shape)
        self.t += 1
        for k in P:                                                           # Adam
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            P[k] -= lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + eps)
        return loss

    def feature_maps(self, x):
        """Activations of both conv layers for one image (used by the 'inside the CNN' panel)."""
        _, c = self.forward(x)
        return c["a1"][0], c["a2"][0]

    def save(self, path):
        np.savez(path, **self.P)

    @classmethod
    def load(cls, path):
        net = cls()
        d = np.load(path)
        net.P = {k: d[k] for k in net.P}
        return net
