"""計算グラフにつかう各ノードクラス
"""
from common.np import *
from common.config import GPU
from common.functions import softmax, cross_entropy_error

# 行列積ノード
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


# シグモイドノード
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


# Affineノード
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b # bはブロードキャストがRepeatノードとみなせる
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0) # Repeatノードの逆伝搬
        self.grads[0][...] = dW # 深いコピー
        self.grads[1][...] = db # 浅いコピー
        return dx


# Softmax
class Softmax:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout # (N,K) * (N,K) = (N,K)
        sumdx = np.sum(dx, axis=1, keepdims=True) # (N,1)
        dx -= self.out * sumdx # (N,K) - (N,1) = (N,K)
        return dx # (N,K) = y_k * dE/dy_k - y_k * sum(y_i * dE/dy_i)


# SoftmaxWithLoss
class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.y = None # softmaxの出力
        self.t = None # 教師ラベル
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0] # N

        dx = self.y.copy() # (N,K)
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx


# SigmoidWithLoss
class SigmoidWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.loss = None
        self.y = None # sigmoidの出力　
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t # (N,1) or (N,)
        self.y = 1 / (1 + np.exp(-x)) # (N,1) or (N,) 

        # input: (N,2), (N,)
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss # scalar
    
    def backward(self, dout=1):
            batch_size = self.t.shape[0]
            dx = (self.y - self.t) * dout / batch_size
            return dx

    

# Ebmeddingレイヤ: 重みW(N,W)から特定行を抜き出す作業
class Embedding:

    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        '''悪い例：idxにダブリがあった場合、指定行に代入されてダブリIDの情報が反映されない'''
        # dW, = self.grads
        # dW[...] = 0 # 形状を保ったまま各要素を0で上書きする
        # dW[self.idx] = dout # 実は悪い例

        '''idxにダブリがあった場合でも、大丈夫なプログラム'''
        dW, = self.grads
        dW[...] = 0

        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
            # もしくは
            # np.add.at(dW, self.idx, dout)

        return None