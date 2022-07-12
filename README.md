# [判例の個人情報の自動マスキング](https://www.nishika.com/competitions/7/summary) 優勝ソリューション

## ソリューション概要

本リポジトリ同梱の[pdf](https://github.com/nishika-inc/cpt-hanrei-1st-refactor/blob/main/%E3%82%BD%E3%83%AA%E3%83%A5%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E8%A7%A3%E8%AA%AC.pdf)に記載

## ディレクトリ構成

- srcディレクトリ配下に必要なコード、データ全て格納する想定

## インプットデータ

### コンペ配布データ

- src/data/input配下
    - train.zip
        - 固有表現にラベリングし出力したデータ
        - ラベリングは[doccano](https://github.com/doccano/doccano)で実施
    - test_token.csv
        - 固有表現にラベリングする前のデータ
        - [spacy](https://spacy.io/)(==2.3.2), [ginza](https://megagonlabs.github.io/ginza/)(==3.1.2)によりtokenizeしている
    - sample_submission.csv
        - 投稿データフォーマット（test_token.csvからtokenカラムを除いたデータ）

#### train.zipに含まれる固有表現ラベリングデータ例

```json
{
    "id": 43,
    "text": "主 文 被告人松村好利を懲役3年に,被告人石橋忠博を懲役2年に処する。 この裁判確定の日から,被告人松村好利に対し5年間,被告人石橋忠博に対し4年間,それぞれその刑の執行を猶予し,その猶予の期間中被告人両名を保護観察に付する。 理 由 (犯罪事実) 第1 (平成31年4月4日付け訴因並びに罪名及び罰条の変更請求書(以下「訴因等変更請求書」という。)記載の公訴事実1の別表番号1関係) 被告人松村好利は,常習として,平成30年12月29日午前6時頃から同日午前6時45分頃までの間,福岡県筑紫野市丸井町組沢1丁目2番3号東洋ビル204号の当時の被告人両名方において,被告人石橋忠博の実子である福島佐千雄(当時7歳。以下「被害者」という。)に対し,後ろ手にさせた両手首及び両足首をビニールテープで縛った上,その体を抱え上げて浴槽に張った冷水の中に入れるなどの暴行を加え,更に被告人石橋忠博は,同日午前6時45分頃に起床しシャワーを浴びるために浴室に入り,その頃,被告人松村好利との間で共謀を遂げ,常習として,その頃から同日午前7時15分頃までの間,同所において,引き続き被害者を前記浴槽に張った冷水の中に入れるなどの暴行を加えた。 ...",
    "meta": {
        "filename": "089226_hanrei.txt",
        "category": "下級裁裁判例"
    },
    "annotation_approver": null,
    "labels": [
        [
            7,
            11,
            "PERSON"
        ],
        [
            21,
            25,
            "PERSON"
        ],
        [
            50,
            54,
            "PERSON"
        ],
        [
            64,
            68,
            "PERSON"
        ],
        [
            196,
            200,
            "PERSON"
        ],
        [
            248,
            268,
            "LOCATION"
        ],
        ...
    ]
}
```

### 外部データ

- src/data/preprocessed配下
    - orgs_df.csv
        - インタネットから収集した3,003件の企業名リスト
        - https://chosa.itmedia.co.jp/providers
    - pi_df.csv
        - インタネットから収集した18,678件の人名リスト。Wikipediaから実在する人物の名前を3,678件収集し、さらに疑似個人情報生成のサイト2から人名を15,000件生成
        - https://ja.wikipedia.org/wiki/%E6%9D%B1%E4%BA%AC%E5%A4%A7%E5%AD%A6%E3%81%AE%E4%BA%BA%E7%89%A9%E4%B8%80%E8%A6%A7
        - https://hogehoge.tk/personal/generator/

## 事前学習済みモデル

以下6つの事前学習済みモデルを利用
- Hugging Faceが提供した4つのモデル
    - cl-tohoku/bert-base-Japanese
    - cl-tohoku/bert-base-japanese-whole-word-masking
    - cl-tohoku/bert-base-japanese-char
    - cl-tohoku/bert-base-japanese-char-whole-word-masking
- 情報通信研究機構が公開した2つのモデル
    - NICT_BERT-base_JapaneseWikipedia_100K
    - NICT_BERT-base_JapaneseWikipedia_32K_BPE

## 実行環境
- Google Colaboratory Pro
    - OS：Ubuntu
    - GPU：16GB
    - メモリ：32GB

## 実行手順

### 学習から推論まで実行

- 以下2つの事前学習済みモデルをダウンロード、解凍してsrc/data/model配下に格納
    - https://alaginrc.nict.go.jp/nict-bert/NICT_BERT-base_JapaneseWikipedia_100K.zip
    - https://alaginrc.nict.go.jp/nict-bert/NICT_BERT-base_JapaneseWikipedia_32K_BPE.zip
- srcディレクトリにて、以下の順でnotebookを実行
    - 0_create_flair_embedding.ipynb
    - 1_augment_data.ipynb
    - 2_generate_dataset.ipynb
    - 3_train_basemodel.ipynb
    - 4_create_stacking_data.ipynb
    - 5_train_stacking.ipynb

### 学習済みモデルから推論のみ実行

- 以下2つの事前学習済みモデルをダウンロード、解凍してsrc/data/model配下に格納
    - https://alaginrc.nict.go.jp/nict-bert/NICT_BERT-base_JapaneseWikipedia_100K.zip
    - https://alaginrc.nict.go.jp/nict-bert/NICT_BERT-base_JapaneseWikipedia_32K_BPE.zip
- 学習したモデル（.ptファイル）を格納
    - シングルモデル：src/save/train/{model_name}/model配下に格納
    - スタッキングモデル：src/save/stacking配下に格納
- srcディレクトリにて、6_infer.ipynbを実行
