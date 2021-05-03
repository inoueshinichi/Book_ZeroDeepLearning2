# Neural Language Took Kit
import nltk
from nltk.corpus import wordnet

def main():
    # wordnetのダウンロード
    # nltk.download('wordnet')

    # `car`の同義語を取得
    print(wordnet.synsets('car'))

    # [car.n.01]とう見出し語で指定される同義語について
    # 意味を確認する
    car = wordnet.synset('car.n.01') # 同義語グループ
    print(car.definition())

    # 同義語グループに存在する単語一覧を表示
    print(car.lemma_names())

    # carグループの上位語を取得
    print(car.hypernym_paths()[0])

    # [car(自動車)]という単語に対して'
    # [novel(小説)], [dog(犬)], [motorcycle(オートバイ)]
    # の3単語の類似度を求める
    car = wordnet.synset('car.n.01')
    novel = wordnet.synset('novel.n.01')
    dog = wordnet.synset('dog.n.01')
    motorcycle = wordnet.synset('motorcycle.n.01')
    print(car.path_similarity(novel))
    print(car.path_similarity(dog))
    print(car.path_similarity(motorcycle))





if __name__ == "__main__":
    main()