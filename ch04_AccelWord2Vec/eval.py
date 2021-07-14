from common.utils import most_similar
import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

import pickle
import numpy as np


def main():
    """学習済みのCBOWモデルを用いて
    単語の分散表現を評価する
    """

    pkl_file = "cbow_params.pkl"

    with open(pkl_file, 'rb') as f:
        params = pickle.load(f)
        word_vecs = params['word_vecs']
        word_to_id = params['word_to_id']
        id_to_word = params['id_to_word']

    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


if __name__ == "__main__":
    main()
