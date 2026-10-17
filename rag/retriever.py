# -*- coding: utf-8 -*-
"""混合检索：BM25 关键词通道 + FAISS 向量通道，用 RRF 融合。

RRF（Reciprocal Rank Fusion）
-----------------------------
两条通道的分数**尺度完全不同**（BM25 是 0~20 的无界分，余弦是 -1~1），
直接加权相加需要调参且不稳定。RRF 只看**名次**：

    score(d) = Σ_channels  w_c / (k + rank_c(d))

k 取 60（原论文推荐值）起平滑作用，让头部名次之间不要拉得太开。
好处是无需归一化、对单通道的异常高分天然鲁棒；代价是名次几乎不拉开，
所以两条通道的相对权重必须实测（见 config.RRF_BM25_WEIGHT）。

用法
----
    python -m rag.retriever --query "高血压的诊断标准是什么"
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import config
from indexing.embed import Embedder
from indexing.tokenize import tokenize


def rrf_fuse(rankings, k=None, weights=None):
    """把多路有序索引融合成一路。rankings 是若干个"下标列表"，按名次排好序。

    ``weights`` 与 ``rankings`` 一一对应，缺省等权。
    为什么权重必须可调：k=60 时 1/(60+1)=0.0164 与 1/(60+50)=0.0091 几乎没差，
    "两路都召回但都排在末尾"（0.0182）会赢过"只在向量通道排第一"（0.0164）。
    BM25 明显弱于向量通道，它带进来的噪声候选就这样把正确答案挤下去 ——
    实测等权融合的 Recall@1（0.714）反而低于纯向量（0.818）。扫权重的表见 eval/report.md。
    """
    k = k or config.RRF_K
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = {}
    for ranking, w in zip(rankings, weights):
        if not w:
            continue
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class HybridRetriever(object):

    def __init__(self, index_dir=None, load_bm25=True, embed_model=None):
        # embed_model 只在消融实验里用：向量模型必须和当初建索引时的一致，
        # 所以换模型要连索引目录一起换（见 eval/run_ablation.py）
        self.embed_model = embed_model
        index_dir = index_dir or config.INDEX_DIR
        self.index_dir = index_dir

        import faiss
        faiss_path = os.path.join(index_dir, "dense.faiss")
        meta_path = os.path.join(index_dir, "chunks_meta.jsonl")
        bm25_path = os.path.join(index_dir, "bm25.json")

        for p in (faiss_path, meta_path):
            if not os.path.exists(p):
                raise RuntimeError(
                    "索引不完整，缺少 %s。先跑：\n"
                    "  python -m indexing.chunk && python -m indexing.build_index" % p)

        self.index = faiss.read_index(faiss_path)

        self.meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.meta.append(json.loads(line))

        if len(self.meta) != self.index.ntotal:
            raise RuntimeError("元数据条数 %d 与向量条数 %d 不一致，索引可能损坏"
                               % (len(self.meta), self.index.ntotal))

        self.bm25 = None
        if load_bm25 and os.path.exists(bm25_path):
            from rank_bm25 import BM25Okapi
            with open(bm25_path, "r", encoding="utf-8") as f:
                self.bm25 = BM25Okapi(json.load(f)["tokens"])

        self._embedder = None

    @property
    def embedder(self):
        # 懒加载：只用 BM25 调试时不必依赖 API key
        if self._embedder is None:
            self._embedder = Embedder(model=self.embed_model)
        return self._embedder

    def close(self):
        if self._embedder is not None:
            self._embedder.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ 单通道

    def search_dense(self, query, top_k=None):
        top_k = top_k or config.TOP_K_DENSE
        qv = self.embedder.embed_one(query).reshape(1, -1)
        scores, idxs = self.index.search(qv, top_k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    def search_bm25(self, query, top_k=None):
        if self.bm25 is None:
            return []
        top_k = top_k or config.TOP_K_BM25
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        order = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    # ------------------------------------------------------------ 融合

    def search(self, query, top_k=None, use_bm25=None):
        """混合检索，返回带 rrf 分数的候选列表（已按融合分排序）。

        每条结果包含 meta 里的全部字段，外加：
        ``rrf_score`` / ``dense_rank`` / ``bm25_rank`` / ``dense_score`` / ``bm25_score``
        保留两路名次是为了方便离线评测时定位是哪条通道召回的。
        """
        top_k = top_k or config.RRF_CANDIDATES
        use_bm25 = (self.bm25 is not None) if use_bm25 is None else use_bm25

        dense = self.search_dense(query, config.TOP_K_DENSE)
        sparse = self.search_bm25(query, config.TOP_K_BM25) if use_bm25 else []

        rankings = [[i for i, _ in dense]]
        weights = [config.RRF_DENSE_WEIGHT]
        if sparse:
            rankings.append([i for i, _ in sparse])
            weights.append(config.RRF_BM25_WEIGHT)
        fused = rrf_fuse(rankings, config.RRF_K, weights)

        d_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
        d_score = {i: s for i, s in dense}
        s_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}
        s_score = {i: s for i, s in sparse}
        # 向量已 L2 归一化，FAISS 内积就是余弦。取召回的**最高**那条：
        # 拒答要问的是「语料里有没有像的问题」，不是「重排后第一名像不像」。
        query_cosine = dense[0][1] if dense else 0.0

        out = []
        for idx, rrf in fused[:top_k]:
            item = dict(self.meta[idx])
            item.update({
                "idx": idx,
                "rrf_score": rrf,
                "query_cosine": query_cosine,
                "dense_rank": d_rank.get(idx),
                "dense_score": d_score.get(idx),
                "bm25_rank": s_rank.get(idx),
                "bm25_score": s_score.get(idx),
            })
            out.append(item)
        return out


def main():
    ap = argparse.ArgumentParser(description="混合检索自检")
    ap.add_argument("--query", "-q", required=True)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--no-bm25", action="store_true")
    args = ap.parse_args()

    with HybridRetriever() as r:
        print("索引: %d 个块，BM25 %s"
              % (r.index.ntotal, "开启" if r.bm25 else "关闭"))
        print("查询: %s" % args.query)
        print()
        for n, item in enumerate(r.search(args.query, top_k=args.top,
                                          use_bm25=not args.no_bm25), 1):
            print("%d. [rrf %.5f | 向量 #%s %.3f | BM25 #%s %.1f] %s > %s"
                  % (n, item["rrf_score"], item["dense_rank"],
                     item["dense_score"] or 0, item["bm25_rank"],
                     item["bm25_score"] or 0, item["title"], item["section"]))
            print("   %s" % item["text"][:110].replace("\n", " "))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
