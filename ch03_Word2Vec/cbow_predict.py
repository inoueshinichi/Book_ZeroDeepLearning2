import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

import numpy as np

from common.layers import MatMul


def test_affine():

    # 簡素な全結合層
    c = np.array([[1, 0, 0, 0, 0, 0, 0]]) # 入力
    W = np.random.randn(7, 3) # 重み
    h = np.dot(c, W) # 中間ノード @Note 単語ベクトルはone-hotなので、やっていることは重みWからある行の要素を抜き出す事に等しい
    print(f"h: {h}")

    # 1章で作成したMatMulレイヤーで処理する
    layer = MatMul(W)
    h2 = layer.forward(c)
    print(f"h2: {h2}")

    print("-"*10)


def test_word2vec():
    """word2vecにつかうNNモデルとして、
    continuous bag-of-wards(CBOW)が有名.

    有名なword2vec用NNモデル
    ①CBOWモデル
    ②skip-gramモデル

    ポイント
    ・分散表現を獲得したい単語周辺のコンテキストの数だけ、入力層がある
    e.g 注目単語の前後1単語を使って中間の注目単語を推論する場合、入力層は2つある.

    `goodbye`の分散表現ベクトルを獲得したい場合
    corpus = [`you`, `say`, `goodbye`, `and`, `I`, `hello`, `.`]
    単語ベクトル(one-hot) = [○, ○, ○, ○, ○, ○, ○]
    You say [goodbye] and I say hello .
    コンテキスト=[`say`, `and`]
    invec = [[0, 1, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 1, 0, 0, 0]]

    中間層の出力 = invec @ W_73 = out_23 = ([[h1],[h2]]) (2, 3) --- 行方向に平均する ---> (1, 3) = 0.5 * (h1 + h2)

    出力層 = 中間層の出力 @ W_37 (1, 7) = 各単語(コーパス)の出現確率(スコア) // これは注目単語に対応している (※単語の分散表現ではない)
    教師データは`goodbye`の単語ベクトル(one-hot) = [0, 1, 0, 0, 0, 0, 0]

    注目単語に対応する重みW_73の行ベクトルが分散表現ベクトルとなる
    invec = [[0, 1, 0, 0, 0, 0, 0],  -> `say`
             [ 0, 0, 0, 1, 0, 0, 0]] -> `and`

    W_73 = [[○, ○, ○],
            [□, □, □],
            [△, △, △],
            [☆, ☆, ☆],
            [※, ※, ※],
            [◎, ◎, ◎],
            [✕, ✕, ✕]]

    コーパスに存在する単語が人間にとって理解できないベクトルに変換された(エンコード)
    `say`の分散表現ベクトル = [□, □, □]
    `and`の分散表現ベクトル = [☆, ☆, ☆]

    中間層の出力から出力層の表現(単語ベクトル)に変換する作業もある(デコード)

    出力側の重みも単語の意味をエンコードしてると考える
    W_37 = [[○, □, △, ☆, ※, ◎, ✕],
            [○, □, △, ☆, ※, ◎, ✕],
            [○, □, △, ☆, ※, ◎, ✕]]

    単語の分散表現重みを使うパターン
    ①W_73(入力側重み)のみ使う -> skip-gramモデル, CBOWモデル
    ②W_37(出力側重み)のみ使う 
    ③W_73とW_37両方使う
    """

    # サンプルのコンテキストデータ(注目単語の周辺2語)
    c0 = np.array([1, 0, 0, 0, 0, 0, 0])
    c1 = np.array([0, 0, 1, 0, 0, 0, 0])

    # 重みの初期化
    W_in = np.random.randn(7, 3)
    W_out = np.random.randn(3, 7)

    # レイヤの生成
    in_layer0 = MatMul(W_in)
    in_layer1 = MatMul(W_in)
    out_layer = MatMul(W_out)

    # 順伝搬
    h0 = in_layer0.forward(c0)
    h1 = in_layer1.forward(c1)
    h = 0.5 * (h0 + h1)
    s = out_layer.forward(h)

    print(f"s: {s}")

    


if __name__ == "__main__":
    test_affine()
    test_word2vec()
