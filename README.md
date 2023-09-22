# 1. 環境構築

```
$ python3 -m venv boenv
$ source boenv/bin/activate
$ pip install -r requirements.txt
```

# 2. 実験の実行

実験の設定を辞書として与えて， `main.py` を実行．(`main.py` を参照)

```
$ python3 main.py
```

# 3. 実験結果の可視化

`expts_result.ipynb` を参照．


# メモ

SingleTaskGP_laplacian は SingleTaskGP(_Matern) と比較して，fitting が異常に高速に感じたので，もしかしたら fitting をしていないかもしれない．

