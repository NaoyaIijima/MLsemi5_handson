## はじめに
これはGMMによって仮想的に生成したデータをLLGMNとGMMで識別するプログラムです．
ただ，識別時のGMMのパラメータは生成時のものを使用しています．

## Requirements
- Python3系が使える環境である
- Kerasがimportできる

## 解析手順
1. make_datasets.pyを実行して仮想データを生成します．
2. GMMで識別したいときはdiscriminate_GMMprosterior.pyを実行してください．
3. LLGMNで識別したいときはdiscriminate_LLGMN.pyを実行してください．

## フォルダの説明
- dataフォルダ：仮想データが保存されます
- graphフォルダ：決定境界のプロットが保存されます

