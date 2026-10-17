# -*- coding: utf-8 -*-
"""FastAPI 服务：把 pipeline 的事件流转成 SSE 推给前端。

流式协议
--------
    event: sources   data: {"sources": [...], "refused": false}   # 先给来源
    event: token     data: {"t": "高血压"}                        # 增量文本
    event: done      data: {"elapsed": 3.2}
    event: error     data: {"message": "..."}                     # 出错时替代 done

先推 sources 是有意的：引用卡片不必等模型生成完，感知延迟明显更低。

启动
----
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import print_function

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from rag.pipeline import RagPipeline

WEB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

state = {}


@asynccontextmanager
async def lifespan(app):
    print("[启动] 加载索引 ...")
    t0 = time.time()
    state["pipeline"] = RagPipeline()
    print("[启动] %d 个片段，用时 %.1fs"
          % (state["pipeline"].retriever.index.ntotal, time.time() - t0))
    yield
    state["pipeline"].close()


app = FastAPI(
    title="知健 HealthRAG",
    description="生活与健康领域的检索增强问答服务。"
                "答案逐条标注来源，资料不足时明确拒答。仅供参考，不构成医疗建议。",
    version="1.0.0",
    lifespan=lifespan,
)


def _sse(event, payload):
    return "event: %s\ndata: %s\n\n" % (
        event, json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------- 问答

@app.get("/api/stream", summary="流式问答（SSE）")
async def api_stream(q: str = Query(..., min_length=2, description="问题")):
    pipeline = state["pipeline"]

    async def gen():
        try:
            async for event, payload in pipeline.astream_events(q):
                yield _sse(event, payload)
        except Exception as e:          # noqa: BLE001 — 让前端看到失败原因
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",      # 关掉反代缓冲，否则流式会攒成一坨
    })


@app.post("/api/ask", summary="一次性问答（非流式）")
async def api_ask(payload: dict):
    question = (payload or {}).get("question", "").strip()
    if len(question) < 2:
        raise HTTPException(status_code=400, detail="问题太短")
    return await asyncio.to_thread(state["pipeline"].answer, question)


# ---------------------------------------------------------------- 检索

@app.get("/api/search", summary="只检索，不生成")
async def api_search(q: str = Query(..., min_length=2), top: int = 8):
    """单独观察检索质量；评测脚本也走这条路径。"""
    pipeline = state["pipeline"]

    def run():
        scored, refused = pipeline.retrieve(q)
        return refused, pipeline.pack_sources(scored)[:top]

    refused, sources = await asyncio.to_thread(run)
    return {"query": q, "refused": refused, "results": sources}


# ---------------------------------------------------------------- 运维

@app.get("/health", summary="健康检查")
async def health():
    p = state.get("pipeline")
    if not p:
        return {"status": "loading"}
    return {
        "status": "healthy",
        "chunks": p.retriever.index.ntotal,
        "bm25": p.retriever.bm25 is not None,
        "rerank": config.ENABLE_RERANK,
        "embed_model": config.EMBED_MODEL,
        "llm_model": config.LLM_MODEL,
    }


@app.get("/api/stats", summary="索引统计")
async def api_stats():
    try:
        with open(config.INDEX_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@app.get("/", include_in_schema=False)
async def index_page():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
