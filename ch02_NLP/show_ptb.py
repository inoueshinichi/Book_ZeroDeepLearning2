import sys
sys.path.append("/Users/inoueshinichi/Desktop/MyGithub/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

from dataset import ptb


def test_show_ptb():
    """Penn Treebank(ペン・ツリー・バンク)の一部を表示する.
    手頃なサイズのコーパス
    Word2Vecの作者(Tomas Mikolov)のWebページからダウンロードできる.
    """

    corpus, word_to_id, id_to_word = ptb.load_data('train')

    print('corpus size: ', len(corpus))
    print('corpus[:30]: ', corpus[:30])
    print()
    print('id_to_word[0]: ', id_to_word[0])
    print('id_to_word[1]: ', id_to_word[1])
    print('id_to_word[2]: ', id_to_word[2])
    print()
    print("word_to_id['car']: ", word_to_id['car'])
    print("word_to_id['happy']: ", word_to_id['happy'])
    print("word_to_id['lexus']: ", word_to_id['lexus'])


if __name__ == "__main__":
    test_show_ptb()



