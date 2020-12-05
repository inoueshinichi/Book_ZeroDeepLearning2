"""Attention機構の実装
"""

# 標準
import sys
sys.path.append((".."))
import os

# サードパーティ
import numpy as np

# 自作
from common.layers import Softmax

# コンテキストベクトルの算出
class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        """順伝搬

        Args:
            hs (ndarray): Encorderで取得した各単語ベクトルを連結した隠れベクトル
            a (ndarray): 各単語ベクトルを連結した隠れベクトルの注目度を表す重みベクトル

        Returns:
            c (ndarray): コンテキストベクトル
        """
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar # 要素毎の積
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c
    
    def backward(self, dc):
        """逆伝搬

        Args:
            dc (ndarray): コンテキストベクトルの誤差微分

        Returns:
            dhs (ndarray): 各単語ベクトルを連結した隠れベクトルに対応する誤差微分
            da (ndarray): 注目度を表す重みベクトルに対応する誤差微分 
        """
        hs, ar = self.chache
        N, T, H = hs.shape

        dt = dc.reshape(N, 1, H).repeat(T, axis=1) # Sumの逆伝搬はRepeat
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2) # Repeatの逆伝搬はSum

        return dhs, da


# Attention用重み計算
class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        """順伝搬

        Args:
            hs (ndarray): Encorderで取得した各単語ベクトルを連結した隠れベクトル
            h (ndarray): hsの最終単語に対応するベクトル

        Returns:
            a (ndarray): Attention用重みベクトル
        """
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        t = hs * hr
        s = np.sum(t, axis=2) # スコア(重み付き和)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)
        return a

    def backward(self, da):
        """逆伝搬

        Args:
            da (ndarray): Attention用重みベクトルの誤差微分

        Returns:
            dhs: Encorderで取得した各単語ベクトルを連結した隠れベクトルの誤差微分
            dh: hsの最終単語に対応するベクトルの誤差微分
        """
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2) # Sumの逆伝搬はRepeat
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis=1) # Repeatの逆伝搬はSum
        
        return dhs, dh

# Attention機構
class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out
    
    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh

    
# TimeAttention機構
class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_enc.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.atteintion_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.atteintion_weights.append(layer.attention_weight)
        
        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t:, :] = dh

        return dhs_enc, dhs_dec

    

