import sys
sys.path.append("/Users/inoueshinichi/Desktop/MyGithub/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

from common.layers import Softmax

import numpy as np

# Encoderから計算したhs(N, T, H)に対して重要度a(N, T)を内積計算してコンテキストベクトルc(N, H)を計算する
class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        

    def forward(self, hs, a):
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1) # (N, H)

        self.cache = (hs, ar)
        return c


    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape

        dt = dc.reshape(N, 1, H).repeat(T, axis=1) # (N, H) -> (N, 1, H) -> (N, T, H)
        dar = dt * hs # (N, T, H)
        dhs = dt * ar # (N, T, H)
        da = np.sum(dar, axis=2) # (N, T, H) -> (N, T)

        return dhs, da



# 重みa(N, T)をhs(N, T, H)とh(N, H)の内積から求める
class AttentionWeight:
    def __init__(self):
        self.params, self.gards = [], []
        self.cache = None
        self.softmax = Softmax()


    def forward(self, hs, h):
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H).repeat(T, axis=1) # (N, H) -> (N, 1, H) -> (N, T, H)
        t = hs * hr # (N, T, H)
        s = np.sum(t, axis=2) # (N, T)
        a = self.softmax.forward(s)
        
        self.cache = (hs, hr)
        return a


    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr # (N, T, H)
        dhr = dt * hs # (N, T, H)
        dh = np.sum(dhr, axis=1) # (N, H)

        return dhs, dh


# Attentionレイヤ-
class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight = None
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()

    def forward(self, hs, h):
        # print('h', h.shape)
        # print('hs', hs.shape)
        a = self.attention_weight_layer.forward(hs, h)
        # print('a', a.shape)
        out = self.weight_sum_layer.forward(hs, a) # (N, H) コンテキストベクトル
        self.attention_weight = a

        # print('out', out.shape)
        return out

    def backward(self, dout):
        dh0, da = self.weight_sum_layer.backward(dout)
        dh1, dh = self.attention_weight_layer.backward(da)
        dhs = dh0 + dh1
        return dhs, dh
    

class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec
    
