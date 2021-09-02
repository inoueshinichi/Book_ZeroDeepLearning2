import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
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

        hr = h.reshape(N, 1, h).repeat(T, axis=1) # (N, H) -> (N, 1, H) -> (N, T, H)
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

