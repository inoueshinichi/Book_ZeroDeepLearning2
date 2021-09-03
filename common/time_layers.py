import sys
sys.path.append("/Users/inoueshinichi/Desktop/MyGithub/DeepLearning2_NLP")
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
        loss /= mask.sum() # 無効idを除いた総数で割る

        self.cache = (ts, ys, mask, (N, T, V))
        
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys # softmaxの出力(確率)
        dx[np.arange(N*T), ts] -= 1 # softmaxの出力(確率)と確率1.0(理想値)との差分=ソフトマックスの勾配
        dx *= dout # (N*T, V)
        dx /= mask.sum() # 有効値のみで???? 有効なラベル数で割る意味がわからない
        dx *= mask[:, np.newaxis] # (N*T, 1), ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx

    


# RNNセル
class RNNCell:
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
            # 各RNNセルで同じ重みを使用している(勾配は各々のセルで異なる)
            layer = RNNCell(*self.params)
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
        grads = [0, 0, 0] # dWx, dWh, db
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh) # 合算した勾配
            dxs[:, t, :] = dx

            # 各RNNセルで同じ重み(Wx, Wh, b)を使用しているが、
            # 各セルでの教師ラベルは異なるので勾配が変わってくる.
            # 対応する勾配は各セルでの逆誤差伝搬時に加算されていく
            for i, grad in enumerate(layer.grads):
                grads[i] += grad # 各RNNセルでの勾配は加算される
            
        # TimeRNNレイヤの勾配
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad # [...(三点ドット)]で上書きする(numpy)

        return dxs

    

class LSTMCell:

    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), 
                      np.zeros_like(Wh),
                      np.zeros_like(b)]
        self.cache = None
    
    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # slice
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)
        
        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = np.sum(dA, axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:

    def __init__(self, Wx, Wh, b, statefull=False):
        self.params = [Wx, Wh, b]
        self.grads = [
            np.zeros_like(Wx), 
            np.zeros_like(Wh), 
            np.zeros_like(b)
        ]
        self.layers = None
        self.h, self.c = None, None
        self.dh = None
        self.statefull = statefull

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.statefull or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.statefull or self.c is None:
            self.c = np.zeros((N, H), dtype='f')
        
        for t in range(T):
            layer = LSTMCell(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0] # dWx, dWh, db
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            # [Wx, Wh, b]は各LSTMセルで共通な重みであり、
            # 各セルの勾配の加算値で重みを更新する
            for i, grad in enumerate(layer.grads):
                grads[i] += grad # 各LSTMセルでの勾配は加算される
            
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh # 最終セルの隠れベクトルの勾配を取得
        return dxs

    def set_state(self, h, c=None):
        # 最初のセルへの隠れベクトルを設定
        self.h, self.c = h, c
    
    def reset_state(self):
        self.h, self.c = None, None
    


class GRUCell:

    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        # x (N, D) = (バッチサイズ, 1時刻あたりの特徴ベクトルの次元)
        # h_prev (N, H) = (バッチサイズ, 1時刻あたりの隠れベクトルの次元)
        # Wx (D, 3H)
        # Wh (H, 3H)
        # Wx_z (D, H)
        # Wx_r (D, H)
        # Wx_h (D, H)
        # Wh_z (H, H)
        # Wh_r (H, H)
        # Wh_h (H, H)
        Wx, Wh, b = self.params
        H = Wh.shape[0]

        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2*H], Wx[:, 2*H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2*H], Wh[:, 2*H:]
        bz, br, bh = b[:, :H], b[:, H:2*H], b[:, 2*H:]

        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bz)
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr) + br)
        h_hat = np.tanh(np.dot(x, Wxh) + np.dot(r*h_prev, Whh) + bh)
        h_next = (1 - z) * h_prev + z * h_hat

        self.cache = (x, h_prev, r, z, h_hat)
        return h_next


    def backward(self, dh_next):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        x, h_prev, r, z, h_hat = self.cache

        dh_hat = dh_next * z
        dh_prev = dh_next * (1 - z)

        # tanh
        dt = dh_hat * (1 - h_hat ** 2)
        dbh = np.sum(dt, axis=0)
        dWhh = np.dot((r * h_prev).T, dt) # (H, N) @ (N, H) = (H, H)
        dWxh = np.dot(x.T, dt) # (D, N) @ (N, H) = (D, H)
        dx = np.dot(dt, Wxh.T) # (N, H) @ (H, D) = (N, D)
        dhr = np.dot(dt, Whh.T) # (N, H) @ (H, H) = (N, H)
        dr = dhr * h_prev # (N, H) * (N, H) = (N, H)
        dh_prev += r * dhr

        # update gate(z)
        dz = dh_next * dh_hat - dh_prev * dh_next
        dt = dz * z * (1 - z)
        dbz = np.sum(dt, axis=0)
        dWhz = np.dot(h_prev.T, dt) # (H, N) @ (N, H) = (H, H)
        dh_prev += np.dot(dt, Whz.T) # (N, H) @ (H, H) = (N, H)
        dWxz = np.dot(x.T, dt) # (D, N) @ (N, H) = (D, H)(
        dx += np.dot(dt, Wxz.T) # (N, H) @ (H, D) = (N, D)

        # reset gate(r)
        dt = dr * r * (1 - r)
        dbr = np.sum(dt, axis=0)
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)

        self.dWx = np.hstack((dWxz, dWxr, dWxh))
        self.dWh = np.hstack((dWhz, dWhr, dWhh))
        self.db = np.hstack((dbz, dbr, dbh))

        self.grads[0][...] = self.dWx
        self.grads[1][...] = self.dWh
        self.grads[2][...] = self.db

        return dx, dh_prev


class GRUTime:

    def __init__(self, Wx, Wh, b, statefull=True):
        self.params = [Wx, Wh, b]
        self.grads = [
            np.zeros_like(Wx), 
            np.zeros_like(Wh),
            np.zeros_like(b)
        ]
        self.layers = None
        self.h = None
        self.statefull = statefull
    
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0] # 隠れベクトルの次元

        self.layers = []
        hs = np.empty((N, T, H), dtype=np.float32)
        
        if not self.statefull or self.h is None:
            self.h = np.zeros((N, H), dtype=np.float32)

        for t in range(T):
            layer = GRUCell(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype=np.float32)
        dh = 0
        grads = [0, 0, 0] # dWx, dWh, db
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            # [Wx, Wh, b]は各GRUセルで共通な重みであり、
            # 各セルの勾配の加算値で重みを更新する
            for i, grad in enumerate(layer.grads):
                grads[i] += grad # 各GRUセルでの勾配は加算される
            
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh # 最終セルの隠れベクトルの勾配を取得
        return dxs

    
class TimeDropout:

    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True
    
    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1.0 / (1.0 - self.dropout_ratio) 
            self.mask = flg.astype(np.float32) * scale
            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask
    
