import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/Desktop/DeepLearning2_NLP")
from common.utils import preprocess, create_co_matrix, cos_similarity, most_similar


def test_most_similar():
    """クエリ単語に対する類似度のランキングを表示
    """
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    most_similar('you', word_to_id, id_to_word, C, top=5)



if __name__ == "__main__":
    test_most_similar()