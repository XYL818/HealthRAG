# -*- coding: utf-8 -*-
"""把检索 → 重排 → 生成串成一条链路。

单独抽出来的原因：``app/main.py`` 的流式接口和命令行工具
（``python -m rag.generator``）需要同一套流程，写在两处迟早会走样。
这里只负责编排，不关心是 HTTP 还是命令行在调。

``astream_events`` 产出的 event 与 SSE 协议一一对应：
``("sources", {...})`` / ``("token", {...})`` / ``("done", {...})``。
"""

from __future__ import print_function

import asyncio
import time

import config
from rag.generator import Generator
from rag.rerank import Reranker, should_refuse
from rag.retriever import HybridRetriever


class RagPipeline(object):

    def __init__(self, retriever=None, reranker=None, generator=None):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.generator = generator or Generator()

    def close(self):
        self.retriever.close()
        self.reranker.close()
        self.generator.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ 检索

    def retrieve(self, question):
        """返回 (选中片段, 是否应拒答)。"""
        candidates = self.retriever.search(question)
        scored = self.reranker.rerank(question, candidates, config.TOP_N_RERANK)
        return scored, should_refuse(scored)

    @staticmethod
    def pack_sources(scored):
        """给前端的来源列表，编号与答案里的 [n] 对应。"""
        return [{
            "n": i,
            "chunk_id": it["chunk_id"],
            "title": it["title"],
            "section": it["section"],
            "url": it["url"],
            "score": s,
            "rrf_score": it.get("rrf_score"),
            "dense_rank": it.get("dense_rank"),
            "bm25_rank": it.get("bm25_rank"),
            "text": it["text"],
        } for i, (it, s) in enumerate(scored, 1)]

    # ------------------------------------------------------------ 问答

    def answer(self, question):
        """同步一次性问答（命令行、评测用）。"""
        t0 = time.time()
        scored, refused = self.retrieve(question)
        if refused:
            from rag.generator import REFUSAL_MESSAGE
            text = REFUSAL_MESSAGE
        else:
            text = self.generator.answer(question, [it for it, _ in scored])
        return {
            "question": question,
            "answer": text,
            "refused": refused,
            "sources": [] if refused else self.pack_sources(scored),
            "elapsed": round(time.time() - t0, 2),
        }

    async def astream_events(self, question):
        """异步产出事件流，供 SSE 直接转发。

        注意：检索与重排都是阻塞调用（含同步 HTTP 请求），
        必须丢到线程里跑，否则会把事件循环卡住。
        """
        from rag.generator import REFUSAL_MESSAGE

        t0 = time.time()
        scored, refused = await asyncio.to_thread(self.retrieve, question)

        sources = [] if refused else self.pack_sources(scored)
        yield "sources", {"sources": sources, "refused": refused}

        if refused:
            yield "token", {"t": REFUSAL_MESSAGE}
        else:
            async for piece in self.generator.astream(
                    question, [it for it, _ in scored]):
                yield "token", {"t": piece}

        yield "done", {"elapsed": round(time.time() - t0, 2)}
