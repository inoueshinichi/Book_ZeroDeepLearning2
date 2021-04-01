import numpy as np

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
    



if __name__ == "__main__":
    main()