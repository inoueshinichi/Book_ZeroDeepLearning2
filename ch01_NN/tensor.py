"""ベクトル・行列の復習
"""

import numpy as np

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    # ベクトル・行列の復習
    x = np.array([1,2,3])
    print(x.__class__)
    print(x.shape)
    print(x.ndim)
    W = np.array([[1,2,3],[4,5,6]])
    print(W.shape)
    print(W.ndim)

    # 行列の要素ごとの演算
    W = np.array([[1,2,3],[4,5,6]])
    X = np.array([[0,1,2],[3,4,5]])
    print(W + X)
    print(W * X)

    # ブロードキャスト
    A = np.array([[1,2],[3,4]])
    print(A*10)
    A = np.array([[1,2],[3,4]])
    b = np.array([10,20])
    print(A*b)

    # ベクトルの内積と行列の積
    # ベクトルの内積
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    print(np.dot(a,b))

    # 行列の積
    A = np.array([[1,2],[3,4]])
    B = np.array([[5,6],[7,8]])
    print(np.dot(A,B))
    
    # ミニバッチ版の全結合層による変換
    # X・W = H XとHは行ベクトル
    # (N,2)・(2,4) = (N,4)     <- こっちのほうがわかりやすい
    # W^T・X = H XとHは列ベクトル
    # (4,2)・(2,N) = (4,N)
    W1 = np.random.randn(2,4) # 重み
    b1 = np.random.randn(4)   # バイアス
    x = np.random.randn(10, 2)# 入力
    h = np.dot(x, W1) + b1
    print(f"h: \n{h}")
    print(f"h shape: {h.shape}")

    W1_T = W1.T
    b1_T = b1.T
    x_T = x.T
    h_T = np.dot(W1_T, x_T) + b1_T[:, None]
    print(f"h_T: \n{h_T}")
    print(f"h_T shape: {h_T.shape}")

    # シグモイド関数を用いて線形変換したX(out=h)を非線形変換する
    a = sigmoid(h) # a: アクティベーション
    print(f"a: \n{a}")

    # a:アクティベーションを更に線形変換
    W2 = np.random.randn(4,3)
    b2 = np.random.randn(3)
    s = np.dot(a, W2) + b2
    print(f"s: \n{s}")

if __name__ == "__main__":
    main()