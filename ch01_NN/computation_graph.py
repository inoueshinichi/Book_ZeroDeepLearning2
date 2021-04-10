"""計算グラフの各ノードの特徴
"""

# 加算ノード
# z = x + y
# point: 上流(z側)の勾配を下流(x,y)にそのまま流すだけ
class Add:
    def __init__(self, x, y):
        self.params = [x, y]
        self.grads = [0, 0]

    def forward(self):
        out = x + y
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
        self.params = [x] # 多分バグ
        self.grads = [np.zeros_like(x)]
    
    def forward(self, x):
        out = np.repeat(x, N, axis=0) # 行(axis=0)がデータ点数
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

    



def main():
    pass


if __name__ = "__main__":
    main()