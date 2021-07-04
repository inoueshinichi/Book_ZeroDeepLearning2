import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/Desktop/DeepLearning2_NLP")
from common.utils import preprocess, create_co_matrix, cos_similarity

def test_similarity():
    """コサイン類似度の計算
    """

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print(f"corpus: {corpus}")
    print(f"word_to_id: {word_to_id}")
    print(f"id_to_word: {id_to_word}")
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    c0 = C[word_to_id['you']] # `you`の単語ベクトル
    c1 = C[word_to_id['i']]   # `i`の単語ベクトル
    print(f"cos_similarity: {cos_similarity(c0, c1)}")

if __name__ == "__main__":
    test_similarity()