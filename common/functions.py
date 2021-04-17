"""各ノードで使用する関数
"""
from common.np import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    if x.ndim == 2:
        # オーバーフローしないように最大値を各値から引く
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    
    return x

def cross_entropy_error(y, t):
    if y.ndim == 1: # if 1-dim vector
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size: # (N,one-hot-dim)
        t = t.argmax(axis=1) # (N,)

    batch_size = y.shape[0] # N

    # 1e-7はnp.log()の中身がゼロになって、-∞になるのを防ぐ
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

