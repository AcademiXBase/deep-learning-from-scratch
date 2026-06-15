ゼロから作る Deep Learning + my_notebooks
==========================================

[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/images/deep-learning-from-scratch.png" width="200px">](https://www.oreilly.co.jp/books/9784873117584/)

書籍『[ゼロから作る Deep Learning](https://www.oreilly.co.jp/books/9784873117584/)』(オライリー・ジャパン発行)の公式サンプルコードに、学習用の `my_notebooks` を追加したリポジトリです。

公式コードの解説は書籍をご覧ください。このリポジトリでは、章ごとの Python スクリプトに加えて、Jupyter Notebook で確認しながら進められる学習ノートを置いています。

## このリポジトリについて

- `ch01` から `ch08`、`common`、`dataset` は公式リポジトリ由来のソースコードです。
- `notebooks` は公式サンプルコードを Notebook 形式で実行するためのファイルです。
- `my_notebooks` は個人学習用に追加した Notebook です。章ごとの補足、実験、図解、途中計算の確認などを含みます。

## ファイル構成

| パス | 説明 |
|:--|:--|
| `ch01` | 1章で使用するソースコード |
| `ch02` | 2章で使用するソースコード |
| `ch03` | 3章で使用するソースコード |
| `ch04` | 4章で使用するソースコード |
| `ch05` | 5章で使用するソースコード |
| `ch06` | 6章で使用するソースコード |
| `ch07` | 7章で使用するソースコード |
| `ch08` | 8章で使用するソースコード |
| `common` | 共通で使用するソースコード |
| `dataset` | データセット用のソースコード |
| `notebooks` | 公式サンプルコードの Notebook 版 |
| `my_notebooks` | 学習用に追加した Notebook |
| `pyproject.toml` | uv で管理する依存関係 |
| `uv.lock` | uv が生成するロックファイル |

## 必要な環境

ソースコードを実行するには、次のソフトウェアとライブラリが必要です。

- Python 3.x
- uv
- NumPy
- Matplotlib
- Jupyter Notebook または JupyterLab (`my_notebooks` を使う場合)

このリポジトリでは、Python 環境を uv で管理します。uv が未インストールの場合は、先にインストールしてください。

```bash
python3 -m pip install uv
```

その後、リポジトリのルートディレクトリで依存関係を同期します。

```bash
uv sync
```

Notebook から使うカーネルを登録する場合は、次のコマンドを実行します。

```bash
uv run python3 -m ipykernel install --user --name deep-learning-from-scratch --display-name "Python (uv: deep-learning-from-scratch)"
```

Conda を使いたい場合は、互換用に残している `environment.yml` から環境を作成できます。

```bash
conda env create -f environment.yml
conda activate dezero
```

## Pythonスクリプトの実行方法

各章のフォルダへ移動して、`python3` コマンドで実行します。

```bash
cd ch01
uv run python3 man.py

cd ../ch05
uv run python3 train_neuralnet.py
```

MNIST などのデータセットは、必要に応じて `dataset` 配下のコードから自動で取得されます。

## Notebookの実行方法

リポジトリのルートディレクトリで Jupyter を起動します。

```bash
uv run jupyter notebook
```

JupyterLab を使う場合は、次のコマンドで起動できます。

```bash
uv run jupyter lab
```

起動後、ブラウザから次のどちらかを開きます。

- `notebooks/`: 公式サンプルコードを Notebook で確認したい場合
- `my_notebooks/`: 追加した学習ノートを確認したい場合

`my_notebooks` には、章ごとの内容を自分で確認し直すための Notebook や補足用のファイルがあります。公式コードを参照しながら、途中式、図解、実験結果を確認できるようにしています。

## クラウドサービスでの実行

公式 Notebook は、次のボタンから [Amazon SageMaker Studio Lab](https://studiolab.sagemaker.aws/) 上で実行できます。利用には事前に [メールアドレスによる登録](https://studiolab.sagemaker.aws/requestAccount) が必要です。

| フォルダ名 | Amazon SageMaker Studio Lab |
|:--|:--|
| `ch01` | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch01.ipynb) |
| `ch02` | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch02.ipynb) |
| `ch03` | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch03.ipynb) |
| `ch04` | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch04.ipynb) |
| `ch05` | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch05.ipynb) |
| `ch06` | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch06.ipynb) |
| `ch07` | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch07.ipynb) |
| `ch08` | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch08.ipynb) |
| `common` | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/common.ipynb) |

## 公式リポジトリ

公式リポジトリはこちらです。

https://github.com/oreilly-japan/deep-learning-from-scratch

## ライセンス

公式リポジトリのソースコードは [MITライセンス](https://opensource.org/licenses/MIT) です。
商用・非商用問わず、自由に利用できます。

## 正誤表

本書の正誤情報は以下のページで公開されています。

https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata

本ページに掲載されていない誤植など間違いを見つけた場合は、[japan@oreilly.co.jp](mailto:japan@oreilly.co.jp) までお知らせください。
