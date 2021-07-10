import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

import numpy as np


def test_embedding():
    """Embeddingレイヤを作る
    Embeddingレイヤ: (N, M)で格納されている重みから単語IDの行を抜き出す
    """

    # 重みの抽出
    W = np.arange(21).reshape(7,3)
    print("W: ", W)
    print("W[2]: ", W[2])
    print("W[5]: ", W[5])

    # 複数行を一度に抽出
    idx = np.array([1,0,3,0])
    print("W[idx]: ", W[idx])




if __name__ == "__main__":
    test_embedding()