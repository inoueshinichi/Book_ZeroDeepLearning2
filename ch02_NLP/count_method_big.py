import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")
from common.utils import np, preprocess, create_co_matrix, cos_similarity, most_similar, ppmi

from dataset import ptb

import matplotlib.pyplot as plt


def test_count_method_big():
    """PTBコーパスデータセットに対して、
    共起行列、正の相互情報量行列、SVDによる次元削減を行う.

    SVDによる次元削減はO(N^3)なので、高速なscikit-learn版を使う.
    """
    window_size = 2
    wordvec_size = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    print('counting co-occurence...')
    C = create_co_matrix(corpus, vocab_size, window_size)
    print('calculating PPMI...')
    W = ppmi(C, verbose=True)

    print('calcurating SVD ...')
    try:
        # truncated SVD (fast!)
        from sklearn.utils.extmath import randomized_svd
        U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
    except ImportError:
        # SVD (slow)
        U, S, V = np.linalg.svd(W)
    
    word_vecs = U[:, :wordvec_size]

    querys = ['you', 'year' 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    

    
if __name__ == "__main__":
    test_count_method_big()

