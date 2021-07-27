import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

import numpy as np
from matplotlib import pyplot as plt

N = 2 # ミニバッチサイズ
H = 3 # 隠れ状態ベクトルの次元数
T = 20 # 時系列データの長さ

def main():
    
    dh = np.ones((N, H))
    np.random.seed(3) # 再現性のため乱数のシードを固定
    # Wh = np.random.randn(H, H) # 勾配爆発
    Wh = np.random.randn(H, H) * 0.5 # 勾配消失

    norm_list = []

    for t in range(T):
        dh = np.dot(dh, Wh.T)
        norm = np.sqrt(np.sum(dh**2)) / N
        norm_list.append(norm)
    
    plt.plot(np.arange(len(norm_list)), np.array(norm_list))
    plt.show()


if __name__ == "__main__":
    main()