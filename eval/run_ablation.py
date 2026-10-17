# -*- coding: utf-8 -*-
"""消融实验：块长 / 嵌入模型 / 是否重排，各自值多少分。

为什么要单独一个脚本
--------------------
``run_eval.py`` 回答"这套流水线好不好"，它假设索引已经建好了。
但面试里更常被问的是**"你怎么知道该这么设计"** —— 那需要换掉某个变量、
其他不动，再测一遍。这个脚本就干这件事：

============  ==========  ==========================  ==============
配置          块长        嵌入模型                     说明
============  ==========  ==========================  ==============
``cs300``       300        BAAI/bge-m3                 块切小
``cs450``       450        BAAI/bge-m3                 **基线**（线上用的）
``cs700``       700        BAAI/bge-m3                 块切大
``qwen3-0.6b``  450        Qwen/Qwen3-Embedding-0.6B   换一个家族的向量模型
============  ==========  ==========================  ==============

**换模型的候选本来选的是 ``BAAI/bge-large-zh-v1.5``，实测建不起来**：它的输入上限是
512 token，而块长 450 切出来的块有 **8.2%** 超过 512 字（最长 980 字），接口直接回
``HTTP 400 code 20015``。换嵌入模型时**输入长度上限是硬约束**，不是改个模型名就行 ——
这件事本身值得记一笔。最后换成 ``Qwen/Qwen3-Embedding-0.6B``：同样 1024 维、
参数量级与 bge-m3 相当、且是完全不同的一族，替换得不含糊。

**是否重排不需要单独建索引** —— 重排只在检索时发生，所以四个配置都在
同一批索引上跑，重排开关直接取自 ``run_eval`` 里的四组检索配置。

成本
----
只有"建索引"这一步花钱（调向量接口）。块长和模型都要重嵌入整份语料，
所以 4 个配置 ≈ 5 万次嵌入调用。先跑 ``--plan`` 看确切数字再决定。
已经建好的变体（``index_meta.json`` 里块数与模型都对得上）会被跳过，
所以中断后重跑只补缺的那些。

用法
----
    python -m eval.run_ablation --plan     # 只切分 + 估成本，不调接口
    python -m eval.run_ablation --build    # 建各配置的索引（花钱）
    python -m eval.run_ablation --eval     # 跑评测，汇总成一张表
    python -m eval.run_ablation --all      # 上面三步全做
"""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from eval.run_eval import (CONFIGS, coverage_gaps, doc_names,  # noqa: E402
                           evaluate, load_questions, summarize)
from rag.rerank import Reranker                               # noqa: E402
from rag.retriever import HybridRetriever                     # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(_ROOT, "indexing", "store")
QUESTIONS = os.path.join(_ROOT, "eval", "questions.jsonl")
REPORT = os.path.join(_ROOT, "eval", "ablation.md")

BASE_TAG = "cs450"

# (标签, 块长, 重叠, 嵌入模型)
VARIANTS = [
    ("cs300",      300,  60, "BAAI/bge-m3"),
    ("cs450",      450,  80, "BAAI/bge-m3"),
    ("cs700",      700, 120, "BAAI/bge-m3"),
    ("qwen3-0.6b", 450,  80, "Qwen/Qwen3-Embedding-0.6B"),
]


def chunks_path(tag):
    return os.path.join(_ROOT, "data", "chunks.%s.jsonl" % tag)


def index_dir(tag):
    return os.path.join(STORE, tag)


def run(cmd, env=None):
    """跑一条子命令，失败就抛。子进程的输出直接透传，方便看进度。"""
    e = dict(os.environ)
    if env:
        e.update(env)
    print("  $ %s" % " ".join(cmd))
    rc = subprocess.call(cmd, cwd=_ROOT, env=e)
    if rc != 0:
        raise SystemExit("子命令失败（退出码 %d）: %s" % (rc, " ".join(cmd)))


def count_lines(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# ---------------------------------------------------------------- 三步

def do_chunk():
    print("[1/3] 切分（不花钱，只跑本地 CPU）")
    counts = {}
    for tag, target, overlap, _ in VARIANTS:
        out = chunks_path(tag)
        run([sys.executable, "-m", "indexing.chunk",
             "--target", str(target), "--overlap", str(overlap), "--out", out])
        counts[tag] = count_lines(out)
        print("  %-10s 块长 %-4d → %d 个块" % (tag, target, counts[tag]))
    return counts


def built_already(tag, n_chunks, model):
    """已有同参数的索引就返回它的构建时间，否则返回 None。

    这一步要过几万次向量接口，中断一次就白烧一半；没有断点续跑的话
    重跑必然从头再来。``index_meta.json`` 里记了块数和模型名，
    **块数对上说明 chunk 文件没变、模型对上说明向量口径没变**，
    两者都满足时复用已有索引与重建一遍等价。
    （只比块数不比内容：改内容又恰好不改块数的情形认不出来，
    真要换切分方式就先把对应目录删掉。）
    """
    path = os.path.join(index_dir(tag), "index_meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except ValueError:
        return None
    if info.get("num_chunks") != n_chunks or info.get("embed_model") != model:
        return None
    return info.get("built_at") or "?"


def do_build(counts):
    todo = []
    print()
    print("[2/3] 建索引（每个块一次嵌入调用）")
    for tag, target, overlap, model in VARIANTS:
        done = built_already(tag, counts[tag], model)
        if done:
            print("- %s（模型 %s，%d 个块）→ 已有同参数索引，跳过（建于 %s）"
                  % (tag, model, counts[tag], done))
            continue
        todo.append((tag, target, overlap, model))

    print("  待建 %d 个变体，共约 %d 次嵌入调用"
          % (len(todo), sum(counts[t] for t, _, _, _ in todo)))
    for tag, _target, _overlap, model in todo:
        print("- %s（模型 %s，%d 个块）" % (tag, model, counts[tag]))
        run([sys.executable, "-m", "indexing.build_index",
             "--chunks", chunks_path(tag), "--index-dir", index_dir(tag)],
            env={"EMBED_MODEL": model, "EMBED_BATCH": "16"})


def eval_variants():
    print()
    print("[3/3] 评测（每个配置跑一遍检索，不调用大模型）")
    questions = load_questions(QUESTIONS)
    print("  评测题 %d 道" % len(questions))

    all_rows = []
    notes = []
    for tag, target, _overlap, model in VARIANTS:
        d = index_dir(tag)
        if not os.path.exists(os.path.join(d, "dense.faiss")):
            print("  跳过 %s：索引不存在，先跑 --build" % tag)
            continue

        print("  → %s" % tag)
        retriever = HybridRetriever(index_dir=d, embed_model=model)
        reranker = Reranker()
        try:
            names = doc_names(retriever.meta)
            gaps = coverage_gaps(questions, names)
            # 查询向量缓存必须按**变体的模型**分开存。这里不能靠 evaluate()
            # 的默认路径 —— 那个是按 config.EMBED_MODEL 起名的，而消融实验里
            # 四个变体有两个不同的模型，共用一份缓存会让 bge-large 的索引
            # 拿到 bge-m3 的查询向量，指标全错且很难看出来。
            results = evaluate(
                retriever, reranker, questions, CONFIGS, gaps, names,
                qvec_path=os.path.join(_ROOT, "eval",
                                       "qvec-%s.json" % tag))
            n_doc = len({m["doc_id"] for m in retriever.meta})
            n_chunk = retriever.index.ntotal
        finally:
            retriever.close()
            reranker.close()

        if gaps:
            notes.append((tag, len(gaps)))
        for c in CONFIGS:
            row = summarize(results[c]["ranks"])
            row.update({"tag": tag, "target": target, "model": model,
                        "config": c, "docs": n_doc, "chunks": n_chunk,
                        "refuse_ok": results[c]["refuse_ok"]})
            if c == "hybrid+rerank":
                # 余弦和检索配置无关，每个变体记一次就够
                row["cosine"] = results.get("_cosine") or []
            all_rows.append(row)
    return all_rows, notes


# ---------------------------------------------------------------- 建表

def render(rows, notes):
    lines = []
    lines.append("# 消融实验：块长 / 嵌入模型 / 重排各值多少分")
    lines.append("")
    lines.append("同一份语料（%d 篇）与同一套评测题，每次只改一个变量。"
                 % (rows[0]["docs"] if rows else 0))
    lines.append("")
    lines.append("块长与嵌入模型决定**索引怎么建**，是否重排只在**检索时**生效，"
                 "所以四组检索配置共用同一批索引。")
    lines.append("")
    lines.append("| 索引配置 | 块长 | 嵌入模型 | 块数 | 检索 | Recall@1 | "
                 "Recall@3 | Recall@5 | Recall@10 | MRR@10 |")
    lines.append("|----------|-----:|----------|-----:|------|---------:|"
                 "---------:|---------:|----------:|-------:|")
    for r in rows:
        mark = " **←**" if (r["tag"] == BASE_TAG
                            and r["config"] == "hybrid+rerank") else ""
        lines.append("| `%s`%s | %d | `%s` | %d | `%s` | %.3f | %.3f | %.3f | %.3f | %.3f |"
                     % (r["tag"], mark, r["target"], r["model"], r["chunks"],
                        r["config"], r["recall@1"], r["recall@3"],
                        r["recall@5"], r["recall@10"], r["mrr@10"]))
    lines.append("")
    lines.append("**←** 标记的是线上实际采用的那一行。")
    lines.append("")

    # 摘要：每个块长/模型取"线上实际用的检索方式"，单独列出来看趋势
    served = [r for r in rows if r["config"] == "hybrid+rerank"]
    if served:
        lines.append("## 只看线上用的那条链路（混合检索 + 重排）")
        lines.append("")
        lines.append("| 索引配置 | 块长 | 嵌入模型 | 块数 | Recall@5 | MRR@10 |")
        lines.append("|----------|-----:|----------|-----:|---------:|-------:|")
        for r in served:
            lines.append("| `%s` | %d | `%s` | %d | %.3f | %.3f |"
                         % (r["tag"], r["target"], r["model"], r["chunks"],
                            r["recall@5"], r["mrr@10"]))
        lines.append("")

        # 重排到底值多少：同一索引内，hybrid vs hybrid+rerank
        lines.append("## 重排的增量（同一索引内对比）")
        lines.append("")
        lines.append("| 索引配置 | hybrid MRR@10 | +重排 MRR@10 | 提升 |")
        lines.append("|----------|--------------:|-------------:|-----:|")
        by_tag = {}
        for r in rows:
            by_tag.setdefault(r["tag"], {})[r["config"]] = r
        for tag in [v[0] for v in VARIANTS]:
            d = by_tag.get(tag, {})
            a, b = d.get("hybrid"), d.get("hybrid+rerank")
            if not a or not b:
                continue
            delta = b["mrr@10"] - a["mrr@10"]
            lines.append("| `%s` | %.3f | %.3f | %+.3f |"
                         % (tag, a["mrr@10"], b["mrr@10"], delta))
        lines.append("")

    lines.append("## 拒答判定")
    lines.append("")
    base = next((r for r in rows
                 if r["tag"] == BASE_TAG and r["config"] == "hybrid+rerank"),
                None)
    cos = (base or {}).get("cosine") or []
    if cos:
        ood = [s for s in cos if s["out_of_domain"]]
        ok = [s for s in ood if s["cos"] < config.MIN_QUERY_COSINE]
        in_dom = [s for s in cos
                  if not s["out_of_domain"] and not s["uncoverable"]]
        false_ref = [s for s in in_dom if s["cos"] < config.MIN_QUERY_COSINE]
        lines.append("基线配置下（判据：查询向量余弦 < `MIN_QUERY_COSINE` = %.2f），"
                     "%d 道域外问题有 %d 道被正确拒答（%.0f%%）；"
                     "%d 道域内可答题误拒 %d 道（%.1f%%）。"
                     % (config.MIN_QUERY_COSINE, len(ood), len(ok),
                        100.0 * len(ok) / len(ood) if ood else 0.0,
                        len(in_dom), len(false_ref),
                        100.0 * len(false_ref) / len(in_dom) if in_dom else 0.0))
    else:
        ro = (base or {}).get("refuse_ok") or []
        if ro:
            lines.append("基线配置下，%d 道域外问题有 %d 道被正确拒答（%.0f%%）。"
                         % (len(ro), sum(ro), 100.0 * sum(ro) / len(ro)))
    lines.append("")

    if notes:
        lines.append("> 标准答案不在语料中的题（测的是语料覆盖，已从召回率剔除）：")
        for tag, n in notes:
            lines.append("> - `%s`：%d 题" % (tag, n))
        lines.append("")

    lines.append("> 由 `python -m eval.run_ablation --eval` 生成。"
                 "评测只跑检索，不调用大模型。")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="块长 / 嵌入模型 / 重排 消融")
    ap.add_argument("--plan", action="store_true", help="只切分并估成本")
    ap.add_argument("--build", action="store_true", help="建索引（花钱）")
    ap.add_argument("--eval", dest="do_eval", action="store_true", help="跑评测出表")
    ap.add_argument("--all", action="store_true", help="三步全做")
    args = ap.parse_args()

    if not (args.plan or args.build or args.do_eval or args.all):
        ap.print_help()
        return 0

    need_chunks = args.build or args.all
    if need_chunks:
        counts = do_chunk()
    else:
        # 没切分就从已有产物数行数；产物不全时退回兜底估算
        counts = {tag: (count_lines(chunks_path(tag))
                        if os.path.exists(chunks_path(tag)) else 0)
                  for tag, _, _, _ in VARIANTS}

    if args.plan:
        print()
        print("计划：")
        total = 0
        for tag, target, overlap, model in VARIANTS:
            n = counts[tag]
            total += n
            print("  %-10s 块长 %-4d 重叠 %-4d 模型 %-22s 约 %6d 次嵌入调用"
                  % (tag, target, overlap, model, n))
        print()
        print("合计约 %d 次嵌入调用（未切分的配置显示 0，先跑 --build）。" % total)
        print("只有这一步花钱；评测本身不调用任何模型。")
        return 0

    if args.build or args.all:
        do_build(counts)

    if args.do_eval or args.all:
        t0 = time.time()
        rows, notes = eval_variants()
        if not rows:
            print("没有任何配置跑成功，检查索引是否已构建", file=sys.stderr)
            return 1
        report = render(rows, notes)
        print()
        print(report)
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write(report)
        print("报告已写入 %s（耗时 %.1fs）" % (REPORT, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
