import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP")
sys.path.append("/home/inoue/Desktop/DeepLearning2_NLP")
import os
from common.np import *
from common.layers import *
from common.functions import softmax, sigmoid


class TimeEmbedding:

    def __init__(self, W):
        # Timeレイヤの枠組みではEmbeddingレイヤの重みWは共通
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype=np.float32)
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W) # 各レイヤで重みは共通
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        
        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]
        
        self.grads[0][...] = grad
    
        return None


class TimeAffine:

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N * T, -1) # (N*T, D)
        out = np.dot(rx, W) + b # (N*T, D) * (D, H) + (, H)
        self.x = x

        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0) # データ方向に集約

        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:

    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3: # 教師ラベルがone-hotベクトルの場合
            ts = np.argmax(ts, axis=2) # ts(N, T, V) -> ts(N, T)
        
        mask = (ts != self.ignore_label) # コーパスの値として-1(無効値)以外を抽出するマスク

        # バッチ分と時系列分をまとめる
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T) # (True, True, False, True, ....)

        ys = softmax(xs) # (N*T, V)
        ls = np.log(ys[np.arange(N*T), ts]) # (N*T,) 教師ラベルに該当するsoftmaxの値(確率)だけ抜き出して、LOGを取る
        ls *= mask # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys # softmaxの出力(確率)
        dx[np.arange(N*T), ts] -= 1 # softmaxの出力(確率)と確率1.0(理想値)との差分=ソフトマックスの勾配
        dx *= dout
        dx /= mask.sum() # 有効値のみで
        dx *= mask[:, np.newaxis] # (N*T, 1), ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx

    



class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next
    
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2) # tanhの微分
        db = np.sum(dt, axis=0) # 順伝搬でブロードキャスト(Repeatノード)が働いているので、データ方向に集約
        dWh = np.dot(h_prev.T, dt) # (H, N) * (N, H) = (H, H)
        dh_prev = np.dot(dt, Wh.T) # (N, H) * (H, H) = (N, H)
        dWx = np.dot(x.T, dt) # (H, N) * (N, H) = (H, H)
        dx = np.dot(dt, Wx.T) # (N, H) * (H, H) = (N, H)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, statefull=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.statefull = statefull

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        # xs (N, T, D)
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype=np.float32)

        if not self.statefull or self.h is None:
            self.h = np.zeros((N, H), dtype=np.float32)
        
        for t in range(T):
            # 各RNNレイヤで同じ重みを使用している
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        
        return hs

    def backward(self, dhs):
        # 各RNNレイヤで同じ重みを使用している
        Wx, Wh, b = self.params
        N, T, D = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype=np.float32)
        dh = 0
        grads = [0, 0, 0] # 各RNNレイヤで使っている同じ重みの勾配は加算される
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh) # 合算した勾配
            dxs[:, t, :] = dx

            # 各RNNレイヤで同じ重み(Wx, Wh, b)を使用しているので
            # 対応する勾配は各レイヤでの逆誤差伝搬時に加算されていく
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
            
        # TimeRNNレイヤの勾配
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad # [...(三点ドット)]で上書きする(numpy)

        return dxs

    
