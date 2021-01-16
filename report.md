<p style="text-align:right;">
姓名:李韋宗<br>
學號:B10615024<br>
日期:2020/11/2<br>
</p>

<h1 style="text-align:center;"> Homework 2: Best Match Model

## 建置環境與套件
* Python 3.6.9, numpy, collections.Counter, tqdm.tqdm, datetime.datetime, timezone, timedelta, functools.reduce, argparse.ArgumentParser, ArgumentDefaultsHelpFormatter, pickle, scipy.sparse

## 資料前處理
* 按照 doc_list.txt/query_list.txt 的順序讀入文章及 query
* 將所有文章及 query 存成 list of string 方便後續操作
    * 並且集中 file I/O 的時間
    * 同時用 `Counter` 計算字典型態的 TF

## 模型參數調整
* 用 BM model 做第 1 次 retrieval
* BM model 參數用上次 BM 作業的參數 (也剛好同樣是老師提供的)
    * beta=0.75, K1=3.5, 不使用 K3 項
* Rocchio 參數
    * alpha=1, beta=0.5, gamma=0.15, r_doc=5, nr_doc=1, iter=5

## 模型運作原理
* 使用 scipy.sparse 儲存所有 tf, tfidf 陣列節省資源
* 參考 sklearn 中對 tfidf 做 l2 normalization 後再進行相似度計算

## 心得
* 依舊是無法突破 baseline 與一直在修 bug 的作業，看來功力還是很需要加強。最開始嘗試實作老師上課推薦的 SMM，但分數都只有 0.00 多，經歷了一個禮拜多修改，才來到 0.26 左右，不確定跟篩選 index trem 到 lexicon 剩 1 萬個字有無關係。只好做同學已經突破 baseline 的方法— Rocchio，希望用跟同學相同的參數能得到好的結果，但一開始還沒做 normalization 時也事與願違，分數卡在 0.49 左右，最後細看 sklearn 的文件才知道預設會做 l2 的正規化，加完才終於突破，也在微調各種參數後得到最後結果。

## 參考資料
> tfidf normalization
>> https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/feature_extraction/text.py#L1498

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
<!-- <img src="https://latex.codecogs.com/gif.latex?[formula]"/> -->