import sys
sys.path.append("/Users/inoueshinichi/Desktop/MyGithub/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")
from common.utils import np, preprocess, create_co_matrix, cos_similarity, most_similar, ppmi


def test_ppmi():
    """正の相互情報量を値とする共起行列を計算
    """

    text = 'You say goodbye and I say hello.'
    print(text)
    corpus, word_to_id, id_to_word = preprocess(text)
    print(f"corpus: {corpus}")
    print(f"word_to_id: {word_to_id}")
    print(f"id_to_word: {id_to_word}")
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    # 正の相互情報量
    W = ppmi(C)

    np.set_printoptions(precision=3)
    print("covariance matrix")
    print(C)
    print('-'*50)
    print("PPMI")
    print(W)

if __name__ == "__main__":
    test_ppmi()


