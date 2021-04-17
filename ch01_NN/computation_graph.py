"""計算グラフの各ノードの特徴
"""
import numpy as np

# 加算ノード
# z = x + y
# point: 上流(z側)の勾配を下流(x,y)にそのまま流すだけ
class Add:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.params = [x, y]
        self.grads = [0, 0]

    def forward(self):
        out = self.x + self.y
        return out
    
    def backward(self, dout):
        dx = dout
        dy = dout
        self.grads[0] = dx
        self.grads[1] = dy
        return dx, dy
    
# 乗算ノード
# z = x * y
# point: 上流(z側)から伝わる勾配に順伝搬時の入力を入れ替えた値を乗算する
class Mul:
    def __init__(self, x, y):
        self.params = [x, y]
        self.grads = [0, 0]
    
    def forward(self, x, y):
        out = np.dot(x, y)
        return out
    
    def backward(self, dout):
        dx = dout * self.params[1] # dout * y
        dy = dout * self.parmas[0] # dout * x
        self.grads[0] = dx
        self.grads[1] = dy
        return dx, dy


# Repeatノード(分岐ノード, コピーノード)
# point: N個の分岐ノードとみなせる
# x : N -> x, x,..., x (N個)
class Repeat:
    def __init__(self, x, N):
        self.N = N
        self.params = [x] # 多分バグ
        self.grads = [np.zeros_like(x)]
    
    def forward(self, x):
        out = np.repeat(x, self.N, axis=0) # 行(axis=0)がデータ点数
        return out
    
    def backward(self, dout):
        dx = np.sum(dout, axis=0, keepdims=True) 
        # keepdims=Trueは, dout(N,D)の時、sum(axis=0)で(1,D)という2次元配列を維持する.
        # keepdims=Falseのときは、(D,)となる
        self.grads[0] = dx
        return dx
    

# Sumノード(Repeatノードの逆ver)
# point: 順伝搬：sum(), 逆伝搬：repeat()
class Sum:
    def __init__(self, x):
        self.parmas = [x] # x: (N,D)
        self.grads = [np.zeros_like(x)] # 多分バグ
        self.N = None

    def forward(self, x):
        self.N = x.shape[0]
        out = np.sum(x, axis=0)
        return out
    
    def backward(self, dout):
        dx = np.repeat(dout, self.N, axis=0)
        self.grads[0] = dx
        return dx

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



# # SoftmaxWithLoss
# class SoftmaxWithLoss:
#     def __init__(self):
#         self.params = []
#         self.grads = []
#         self.y = None # softmaxの出力
#         self.t = None # 教師ラベル
    
#     def forward(self, x, t):
#         self.t = t
#         self.y = 

def main():
    pass


if __name__ == "__main__":
    main()