import sys
sys.path.append("/Users/inoueshinichi/Desktop/MyGithub/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

import numpy as np

from negative_sampling_layer import UnigramSampler


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



def test_negative_sampling():
    """ネガティブサンプリングのテスト
    """

    corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
    power = 0.75
    sample_size = 2

    sampler = UnigramSampler(corpus, power, sample_size)
    target = np.array([1, 3, 0]) # ミニバッチ
    print("target: ", target)
    negative_sample = sampler.get_negative_sample(target) # ミニバッチ対応
    print("negative_sample: ", negative_sample)

if __name__ == "__main__":
    test_embedding()
    test_negative_sampling()