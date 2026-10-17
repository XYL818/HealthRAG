# -*- coding: utf-8 -*-
"""构建双索引：FAISS 稠密向量 + BM25 关键词。

为什么要两条索引
----------------
纯向量检索对**专有名词**和**精确术语**不敏感："bge-m3" "BMI" "幽门螺杆菌"
这类词，语义模型容易把它们和相近概念混在一起；反过来纯 BM25 又抓不住
"血压高吃什么好" 和 "高血压的饮食建议" 这种换了说法的提问。
两条通道各召回一批，再用 RRF 融合，是工业界的常规做法。

产物（都在 indexing/store/）
---------------------------
- ``dense.faiss``      : IndexFlatIP，向量已 L2 归一化，内积即余弦
- ``bm25.json``        : 分好词的语料，加载时现建 BM25Okapi（比存 pickle 干净）
- ``chunks_meta.jsonl``: 与向量**逐行同序**的元数据，靠下标对齐
- ``index_meta.json``  : 构建信息（模型、维度、条数、耗时）

用法
----
    python -m indexing.build_index --chunks data/chunks.jsonl
"""

from __future__ import print_function

import argparse
import gc
import json
import os
import sys
import time

import config
from indexing.embed import Embedder
from indexing.tokenize import tokenize


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    ap = argparse.ArgumentParser(description="构建 FAISS + BM25 双索引")
    ap.add_argument("--chunks", default=config.CHUNKS_PATH)
    ap.add_argument("--index-dir", default=config.INDEX_DIR)
    ap.add_argument("--limit", type=int, default=0, help="只索引前 N 个块（调试用）")
    args = ap.parse_args()

    if not os.path.exists(args.chunks):
        print("找不到切分产物 %s，先跑 python -m indexing.chunk" % args.chunks,
              file=sys.stderr)
        return 1

    os.makedirs(args.index_dir, exist_ok=True)

    t0 = time.time()
    chunks = load_chunks(args.chunks)
    if args.limit:
        chunks = chunks[:args.limit]
    if not chunks:
        print("切分产物为空", file=sys.stderr)
        return 1
    print("载入 %d 个块" % len(chunks))

    # ---------------------------------------------------------- 稠密索引
    texts = [c["embed_text"] for c in chunks]
    print("[1/3] 向量化（模型 %s）" % config.EMBED_MODEL)
    with Embedder() as emb:
        vecs = emb.embed(texts, progress=True, desc="向量化")

    print("[2/3] 构建 FAISS 索引")
    import faiss
    dim = int(vecs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss_path = os.path.join(args.index_dir, "dense.faiss")
    faiss.write_index(index, faiss_path)
    print("  维度 %d，向量 %d 条 → %s" % (dim, index.ntotal, faiss_path))
    # faiss 已经持有自己的副本，这份矩阵（14k 块时约 58MB）可以放掉了
    del vecs, texts
    gc.collect()

    # ---------------------------------------------------------- BM25 索引
    # 逐块分词、边分边写：整份语料的 token 列表在内存里要几十 MB，
    # 再叠一个完整的 JSON 字符串，收尾时很容易撞上容器的内存上限
    # （实测 700m 的容器就是这么被撑爆的）
    print("[3/3] 分词并保存 BM25 语料")
    bm25_path = os.path.join(args.index_dir, "bm25.json")
    vocab = set()
    total = len(chunks)
    with open(bm25_path, "w", encoding="utf-8") as f:
        f.write('{"tokens": [')
        for i, c in enumerate(chunks):
            # 别名也进 BM25 词表：正文里写的是「重性抑郁障碍」，
            # 用户问的是「抑郁症」，不给别名，关键词通道直接空手而归
            aliases = c.get("aliases") or []
            toks = tokenize(c["text"] + " " + " ".join(aliases))
            vocab.update(toks)
            f.write(("" if i == 0 else ",") + json.dumps(toks, ensure_ascii=False))
            if (i + 1) % 500 == 0:
                sys.stderr.write("\r  分词 %d/%d (%.0f%%)"
                                 % (i + 1, total, 100.0 * (i + 1) / total))
                sys.stderr.flush()
        f.write("]}")
    sys.stderr.write("\r  分词 %d/%d 完成\n" % (total, total))
    print("  词表大小 %d → %s" % (len(vocab), bm25_path))

    # ---------------------------------------------------------- 元数据
    meta_path = os.path.join(args.index_dir, "chunks_meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "title": c["title"],
                # 别名要进元数据：评测按标题判定命中，用户也常用俗称提问，
                # 少了它「抑郁症」会被判成未命中（正主在语料里叫「重性抑郁障碍」）
                "aliases": c.get("aliases") or [],
                "url": c["url"],
                "section": c["section"],
                "text": c["text"],
            }, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    info = {
        "num_chunks": len(chunks),
        "num_docs": len({c["doc_id"] for c in chunks}),
        "dim": dim,
        "embed_model": config.EMBED_MODEL,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(elapsed, 1),
        "avg_chars": round(sum(c["chars"] for c in chunks) / float(len(chunks)), 1),
    }
    with open(os.path.join(args.index_dir, "index_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print()
    print("完成，用时 %.1fs" % elapsed)
    for k, v in info.items():
        print("  %-14s %s" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
