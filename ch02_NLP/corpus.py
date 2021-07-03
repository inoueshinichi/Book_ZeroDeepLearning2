"""カウントベースによる単語の表現(コーパス)
"""

def corpus():
    """コーパスの作り方
    """

    # ------------
    text = 'You say goodbye and I say hello.'
    text = text.lower()
    text = text.replace('.', ' .')
    print(f"text: {text}")

    words = text.split(' ')
    print(f"words: {words}")

    # ------------
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    print(f"word_to_id: {word_to_id}")
    print(f"id_to_word: {id_to_word}")

    import numpy as np
    corpus = [word_to_id[w] for w in words]
    corpus = np.array(corpus)
    print(f"corpus: {corpus}")


def test_corpus():
    """utils.pyのpreprocess関数を実行
    """
    import sys
    sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
    sys.path.append("/home/inoue/Desktop/DeepLearning2_NLP")
    from common.utils import preprocess

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    print(f"corpus: {corpus}")
    print(f"word_to_id: {word_to_id}")
    print(f"id_to_word: {id_to_word}")


if __name__ == "__main__":
    # corpus()
    test_corpus()