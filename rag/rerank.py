# -*- coding: utf-8 -*-
"""重排：用 cross-encoder 把"召回"和"排序"分开做。

两段式检索的标准做法：

- **召回阶段**（retriever）：双塔/BM25，快但粗，只能看"像不像"；
- **重排阶段**（这里）：把 (问题, 文档) 拼在一起喂给 cross-encoder，
  模型能逐词比对，精度高但慢 —— 所以只对前几十条候选做。

硅基流动上走 ``/v1/rerank``，模型 ``BAAI/bge-reranker-v2-m3``。
接口不可用时**优雅降级**：退回 RRF 的顺序，并把分数标记为 None，
让上层用 RRF 阈值决定要不要拒答，而不是直接报错挂掉。
"""

from __future__ import print_function

import sys

import httpx

import config


class Reranker(object):

    def __init__(self, base_url=None, api_key=None, model=None, timeout=30.0):
        self.base_url = (base_url or config.RERANK_BASE_URL).rstrip("/")
        self.api_key = api_key or config.RERANK_API_KEY
        self.model = model or config.RERANK_MODEL
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": "Bearer " + self.api_key},
            timeout=timeout,
        )

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def doc_text(item):
        """给重排模型的文档文本：带上标题和章节路径，信息更全。"""
        head = item.get("title", "")
        sec = item.get("section", "")
        if sec and sec != head:
            head = "%s > %s" % (head, sec)
        return ("%s\n%s" % (head, item.get("text", ""))).strip()

    def rerank(self, query, items, top_n=None):
        """对候选重排，返回 [(item, score)]，按分数降序。

        失败时返回 ``[(item, None)]``（保持原序），由调用方降级处理。
        """
        top_n = top_n or config.TOP_N_RERANK
        if not items:
            return []

        docs = [self.doc_text(it) for it in items]
        try:
            r = self._client.post("/rerank", json={
                "model": self.model,
                "query": query,
                "documents": docs,
                "top_n": min(top_n, len(docs)),
            })
            if r.status_code != 200:
                raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:200]))
            results = r.json()["results"]
            ranked = sorted(results, key=lambda d: -d["relevance_score"])
            return [(items[d["index"]], float(d["relevance_score"]))
                    for d in ranked]
        except Exception as e:
            print("[重排不可用，退回 RRF 顺序] %s" % e, file=sys.stderr)
            return [(it, None) for it in items[:top_n]]


def should_refuse(scored):
    """判断该不该拒答。

    资料库里没有相关内容时，与其让大模型硬编，不如直说不知道 ——
    这在健康领域尤其重要。

    判据是**查询向量与语料的最高余弦**，不是重排分：重排分在域内和域外
    之间交叠（见 config.MIN_QUERY_COSINE 的说明），余弦能干净切开。
    只有取不到余弦时才退回重排分，再退回 RRF 分。
    """
    if not scored:
        return True
    top_item, top_score = scored[0]
    qcos = top_item.get("query_cosine")
    if qcos is not None:
        return qcos < config.MIN_QUERY_COSINE
    if top_score is None:
        return top_item.get("rrf_score", 0) < config.MIN_RRF_SCORE
    return top_score < config.MIN_RERANK_SCORE
