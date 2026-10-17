# -*- coding: utf-8 -*-
"""检索质量离线评测：跑消融，看每个模块到底贡献了多少。

为什么要做这个
--------------
RAG 项目最容易糊弄的地方就是"看起来能答"。面试官没法验证，你自己也说不清
换成纯向量会不会一样好。所以这里做一件很朴素的事：**准备一批有标准答案的
问题，然后逐项关掉模块看指标掉多少**。

四组配置构成消融：

===============  ==================================================
配置              说明
===============  ==================================================
``bm25``         只用关键词通道，作为下界基线
``dense``        只用向量通道
``hybrid``       双通道 + RRF 融合
``hybrid+rerank``在 hybrid 基础上加 cross-encoder 重排（线上实际用的）
===============  ==================================================

指标
----
- **Recall@k**：前 k 条里至少命中一个标准答案的问题占比
- **MRR@10**  ：第一个命中项的排名倒数均值（衡量"命中得靠不靠前"）

评测**不调用大模型**，只跑检索，所以又快又省钱，可以随时重跑。

用法
----
    python -m eval.run_eval
    python -m eval.run_eval --questions eval/questions.jsonl --out eval/report.md
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from indexing.tokenize import tokenize
from rag.rerank import Reranker
from rag.retriever import HybridRetriever, rrf_fuse

_HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = ["bm25", "dense", "hybrid", "hybrid+rerank"]
KS = [1, 3, 5, 10]
POOL = 50           # 每个通道先各取这么多，再做融合/重排

# RRF 的两个参数扫描。等权融合（w=1.0）时 hybrid 的 Recall@1 反而低于纯向量，
# 原因见 rag/retriever.py 的 rrf_fuse 注释；**k 和 w 互相纠缠** ——
# k 越大名次差异被压得越平，w 的影响就被放得越大，所以两个一起扫而不是各扫各的。
# 候选已经算好，融合是纯 CPU 计算，多扫几组不额外花钱。
K_SWEEP = [10, 20, 30, 60]
W_SWEEP = [0.0, 0.1, 0.3, 0.5, 1.0]


def load_questions(path):
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                qs.append(json.loads(line))
    return qs


def _norm(title):
    """标题归一化。维基标题里常有空格和全角括号的差异，直接比字符串会漏配。"""
    return "".join(title.split()).lower()


def doc_names(meta):
    """每个块的「可接受名字」集合（已归一化）：规范标题 + 别名。

    只比规范标题是不够的：维基的规范标题常常不是日常说法
    （正主在语料里叫「重性抑郁障碍」，没人会这么问，俗称是「抑郁症」）。
    文档其实**能被检索到**，但按标题字面比对会把它判成未命中 ——
    那是评测口径的问题，不是检索的问题。
    """
    out = []
    for m in meta:
        s = {_norm(m.get("title", ""))}
        for a in m.get("aliases") or []:
            s.add(_norm(a))
        out.append(s)
    return out


def coverage_gaps(questions, names):
    """找出标准答案根本不在语料里的题。

    这类题测的是**语料覆盖**，不是检索质量 —— 语料里没有这篇文档，
再怎么调检索也召不回来，混进 Recall 分母只会让指标失真。
    单独报出来，提示该换题或补语料。
    """
    have = set()
    for s in names:
        have |= s
    gaps = []
    for item in questions:
        if item.get("should_refuse"):
            continue
        rel = item.get("relevant") or []
        if rel and not any(_norm(r) in have for r in rel):
            gaps.append((item["q"], rel))
    return gaps


def hit_rank(ranking_names, relevant):
    """返回第一个命中标准答案的名次（1 起），没命中返回 None。

    ``ranking_names`` 是每条的「可接受名字」集合，命中任意一个别名就算命中。
    """
    rel = {_norm(r) for r in relevant}
    for rank, names in enumerate(ranking_names, 1):
        if names & rel:
            return rank
    return None


def hybrid_label(k, w):
    """融合参数扫描里一档配置的名字。"""
    return "hybrid(k=%g,w=%g)" % (k, w)


def default_qvec_path(model=None):
    """按向量模型名给缓存文件命名 —— 换模型必须换缓存。

    消融实验里四个变体用了两个不同的向量模型。缓存如果不按模型分开，
    「bge-large 的索引」会拿到「bge-m3 的查询向量」，指标全错且很难看出来
    （两边维度都是 1024，不会报错，只会静默算出错的排名）。
    """
    name = (model or config.EMBED_MODEL).replace("/", "_")
    return os.path.join(_HERE, "qvec-%s.json" % name)


def load_qvec(path):
    """读查询向量缓存。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except ValueError:
        return {}


def save_qvec(path, cache):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def query_vector(embedder, question, cache):
    """取查询向量，优先用缓存。返回 (向量, 是否命中缓存)。

    **为什么必须缓存**：不缓存的话同一份索引跑两遍，指标会差 1 道题
    （实测 dense 的 MRR@10 在 0.908 与 0.910 之间跳，hybrid 的 Recall@1
    在 0.846 与 0.855 之间跳）。索引、分词、RRF 都是确定性的，
    唯一的变数就是向量接口 —— 同一个 query 不同时刻返回的向量在
    末位上有微小差异，足以让一道临界题的排名整体挪一格。
    评测报告要能被人复现，就不能带着这个抖动，所以把向量落盘：
    第一次跑写缓存，之后重跑逐位一致，顺带省掉重复的接口调用。
    缓存按向量模型分开存 —— 消融实验要换模型，混在一起会串味。
    """
    v = cache.get(question)
    if v is None:
        v = [float(x) for x in embedder.embed_one(question)]
        cache[question] = v
        return v, False
    return v, True


def evaluate(retriever, reranker, questions, configs, gaps=(), names=None,
             ksweep=(), wsweep=(), qvec_path=None):
    grid = [(k, w) for k in ksweep for w in wsweep]
    keys = list(configs) + [hybrid_label(k, w) for k, w in grid]
    results = {c: {"ranks": [], "refuse_ok": [], "details": [],
                   "scores": []} for c in keys}
    # 拒答判定的原始观测，和检索配置无关，单独存一份
    results["_cosine"] = []
    n = len(questions)
    gap_qs = {q for q, _ in gaps}
    names = names if names is not None else doc_names(retriever.meta)
    depth = max(KS)      # 所有配置都截到同样深度，否则「未命中」这一列不可比

    qvec_path = qvec_path or default_qvec_path()
    qcache = load_qvec(qvec_path)
    n_cached = 0

    for qi, item in enumerate(questions, 1):
        q = item["q"]
        relevant = item.get("relevant") or []
        expect_refuse = bool(item.get("should_refuse"))
        # 标准答案不在语料里的题：检索不可能命中，算进召回率只会失真
        uncoverable = q in gap_qs

        sys.stderr.write("\r  评测中 %d/%d" % (qi, n))
        sys.stderr.flush()

        # ---- 两条通道各召回一次，后面所有配置复用，避免重复调向量接口
        qv, cached = query_vector(retriever.embedder, q, qcache)
        n_cached += 1 if cached else 0
        qv = np.asarray(qv, dtype="float32").reshape(1, -1)
        dense_scores, dense_idx = retriever.index.search(qv, POOL)
        dense_idx = [int(i) for i in dense_idx[0] if i >= 0]
        # 向量已归一化，内积即余弦。拒答判据用的就是这条（见 config.MIN_QUERY_COSINE）
        query_cos = float(dense_scores[0][0]) if dense_idx else 0.0

        q_tokens = tokenize(q)
        if retriever.bm25 is not None and q_tokens:
            scores = retriever.bm25.get_scores(q_tokens)
            order = scores.argsort()[::-1][:POOL]
            bm25_idx = [int(i) for i in order if scores[i] > 0]
        else:
            bm25_idx = []

        fused_pairs = rrf_fuse([dense_idx, bm25_idx], config.RRF_K,
                               [config.RRF_DENSE_WEIGHT, config.RRF_BM25_WEIGHT])
        fused = [i for i, _ in fused_pairs]
        # 融合后第一名的分数 —— 重排不可用时的拒答兜底就靠它（见 MIN_RRF_SCORE）
        top_rrf = fused_pairs[0][1] if fused_pairs else 0.0

        rankings = {
            "bm25": bm25_idx,
            "dense": dense_idx,
            "hybrid": fused,
            "hybrid+rerank": None,     # 见下
        }
        # 参数扫描：同一批候选换组参数再融合一次，纯 CPU，不额外调接口
        for k, w in grid:
            rankings[hybrid_label(k, w)] = [i for i, _ in rrf_fuse(
                [dense_idx, bm25_idx], k, [1.0, w])]

        # ---- 重排配置：对 RRF 前若干个候选做 cross-encoder
        # top_n 取到 KS 的最大值，否则重排配置只返回 6 条，
        # 跟其他配置的 Recall@10 不在一个口径上
        reranked_idx, top_score = None, None
        if "hybrid+rerank" in configs:
            cand = [dict(retriever.meta[i], idx=i, rrf_score=1.0 / (config.RRF_K + r + 1))
                    for r, i in enumerate(fused[:config.RRF_CANDIDATES])]
            scored = reranker.rerank(q, cand, top_n=max(max(KS), config.TOP_N_RERANK))
            if scored:
                top_score = scored[0][1]
                reranked_idx = [it["idx"] for it, _ in scored]
            else:
                reranked_idx = []
            rankings["hybrid+rerank"] = reranked_idx

        # 逐题记下重排分。原有的 refuse_ok 只统计域外题，也就是**拒答判定的正面**；
        # 只报这一面的话，一个「见谁都拒」的系统能拿满分，必须同时报它误伤了多少
        # 合法问题（反面）。顺带记下正确答案在**融合阶段**的名次，用来区分
        # 「语料里没有」和「有、但被重排扔了」这两种拒答。
        fused_rank = (hit_rank([names[i] for i in fused[:config.RRF_CANDIDATES]],
                               relevant) if relevant else None)
        results["_cosine"].append({
            "q": q,
            "cos": query_cos,
            "out_of_domain": expect_refuse,
            "uncoverable": uncoverable,
            "fused_rank": fused_rank,
        })
        if top_score is not None:
            results["hybrid+rerank"]["scores"].append({
                "q": q,
                "score": top_score,
                "out_of_domain": expect_refuse,
                "uncoverable": uncoverable,
                "fused_rank": fused_rank,
            })

        for c in keys:
            idxs = (rankings[c] or [])[:depth]
            titles = [retriever.meta[i]["title"] for i in idxs]
            if uncoverable:
                continue
            if expect_refuse:
                # 域外题没有标准文档，算进召回率分母只会把指标压低，
                # 它们只参与拒答判定
                if c == "hybrid+rerank" and top_score is not None:
                    results[c]["refuse_ok"].append(
                        top_score < config.MIN_RERANK_SCORE)
                elif c == "hybrid":
                    # 顺带记下「不重排」那条兜底路径的拒答准确率。重排在本语料上
                    # 排序是净亏（见报告里的重排增量），所以必须回答「那还留着它干
                    # 什么」—— 如果它的拒答判定也不比 RRF 阈值强，就没有理由留。
                    results[c]["refuse_ok"].append(
                        top_rrf < config.MIN_RRF_SCORE)
                continue
            r = hit_rank([names[i] for i in idxs], relevant)
            results[c]["ranks"].append(r)
            results[c]["details"].append((q, r, titles[:3]))

    save_qvec(qvec_path, qcache)
    sys.stderr.write("\n")
    sys.stderr.write("查询向量：%d 条命中缓存 / 共 %d 条（缓存 %s）\n"
                     % (n_cached, n, os.path.relpath(qvec_path)))
    results["_qvec"] = {"cached": n_cached, "total": n,
                        "path": os.path.relpath(qvec_path)}
    return results


def summarize(ranks, ks=KS):
    out = {}
    for k in ks:
        out["recall@%d" % k] = (sum(1 for r in ranks if r and r <= k)
                                / float(len(ranks))) if ranks else 0.0
    rr = [1.0 / r if r else 0.0 for r in ranks]
    out["mrr@10"] = sum(rr) / float(len(rr)) if rr else 0.0
    out["miss"] = sum(1 for r in ranks if not r)
    return out


def main():
    ap = argparse.ArgumentParser(description="检索质量消融评测")
    ap.add_argument("--questions", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "questions.jsonl"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "report.md"))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--index-dir", default=None,
                    help="要评测的索引目录（默认 config.INDEX_DIR）。"
                         "消融实验用它在同一套题上比不同的索引")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        print("评测集为空", file=sys.stderr)
        return 1
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    print("载入 %d 道评测题" % len(questions))
    index_dir = args.index_dir or config.INDEX_DIR
    print("加载索引 %s ..." % index_dir)
    retriever = HybridRetriever(index_dir=index_dir)
    reranker = Reranker() if "hybrid+rerank" in configs else None

    names = doc_names(retriever.meta)
    gaps = coverage_gaps(questions, names)
    if gaps:
        print("有 %d 题的标准答案不在语料中，本次不计入召回率" % len(gaps))

    t0 = time.time()
    try:
        results = evaluate(retriever, reranker, questions, configs, gaps, names,
                           ksweep=K_SWEEP, wsweep=W_SWEEP)
    finally:
        retriever.close()
        if reranker:
            reranker.close()

    # ---------------------------------------------------------- 汇总输出
    rows = []
    for c in configs:
        s = summarize(results[c]["ranks"])
        s["config"] = c
        rows.append(s)

    grid = [(k, w) for k in K_SWEEP for w in W_SWEEP]
    grid_rows = {}
    for k, w in grid:
        grid_rows[(k, w)] = summarize(results[hybrid_label(k, w)]["ranks"])

    lines = []
    lines.append("# 检索质量评测报告")
    lines.append("")
    n_in = sum(1 for q in questions if not q.get("should_refuse"))
    n_out = len(questions) - n_in
    lines.append("语料 %d 篇文档 / %d 个片段；评测集 %d 题（域内 %d 题计入召回，"
                 "域外 %d 题只验拒答）。"
                 % (len({m["doc_id"] for m in retriever.meta}),
                    retriever.index.ntotal, len(questions), n_in, n_out))
    lines.append("")
    # 写明测的是哪套索引，否则几张消融表摆在一起会分不清口径
    lines.append("> 索引 `%s`，向量模型 `%s`。"
                 % (os.path.relpath(index_dir), config.EMBED_MODEL))
    lines.append("")

    # 标准答案不在语料里的题：测的是语料覆盖，不是检索质量，单独报出来
    if gaps:
        lines.append("> 注意：有 %d 题的标准答案不在语料中，检索**不可能**命中，"
                     "已从召回率统计里剔除：" % len(gaps))
        for q, rel in gaps[:20]:
            lines.append("> - %s（期望来源：%s）" % (q, "、".join(rel)))
        if len(gaps) > 20:
            lines.append("> - …另有 %d 题" % (len(gaps) - 20))
        lines.append("")
    lines.append("| 配置 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | 未命中 |")
    lines.append("|------|---------:|---------:|---------:|----------:|-------:|-------:|")
    for s in rows:
        lines.append("| `%s` | %.3f | %.3f | %.3f | %.3f | %.3f | %d |"
                     % (s["config"], s["recall@1"], s["recall@3"], s["recall@5"],
                        s["recall@10"], s["mrr@10"], s["miss"]))
    lines.append("")
    lines.append("> 所有配置都只取前 %d 条对比，「未命中」= 前 %d 条里一条都不是来源文档，"
                 "这样各列才可比（重排配置本来只返回 %d 条）。"
                 % (max(KS), max(KS), max(KS)))
    lines.append("")
    lines.append("`hybrid` 用的是默认融合参数（k=%d，向量 %.1f : BM25 %.1f），"
                 "这个默认值是下面两张扫描表里选出来的。"
                 % (config.RRF_K, config.RRF_DENSE_WEIGHT,
                    config.RRF_BM25_WEIGHT))
    lines.append("")

    # 融合参数扫描：解释「等权 hybrid 为什么打不过纯向量」
    # 三个指标一起看，因为它们的用途不同：R@1/MRR 是直接展示给用户的顺序，
    # R@3 才决定**重排候选池**里有没有正确答案 —— 线上过重排，所以 R@3 不能不管。
    for metric, title in (("recall@1", "Recall@1"), ("recall@3", "Recall@3"),
                          ("mrr@10", "MRR@10")):
        lines.append("**融合参数扫描 · %s**（行 = RRF 的 k，列 = BM25 权重）" % title)
        lines.append("")
        lines.append("| k \\ w |" + "".join(" %.1f |" % w for w in W_SWEEP))
        lines.append("|------:|" + "-----:|" * len(W_SWEEP))
        for k in K_SWEEP:
            cells = []
            for w in W_SWEEP:
                s = grid_rows[(k, w)]
                mark = "**%.3f**" % s[metric] if (
                    k == config.RRF_K
                    and abs(w - config.RRF_BM25_WEIGHT) < 1e-9) else "%.3f" % s[metric]
                cells.append(" " + mark + " |")
            lines.append("| %d |" % k + "".join(cells))
        lines.append("")
    lines.append("> 加粗格是线上默认参数。BM25 权重 0.0 那一列等于「纯向量通道 + "
                 "RRF 空转」，把它和 `dense` 行对照可以看出融合有没有帮上忙。")
    lines.append(">")
    lines.append("> **默认参数是怎么定的**：三个表里 w=0.0 在 R@1 和 MRR 上都是最优，"
                 "说明**本语料上向量通道占绝对主导，BM25 对排序几乎不贡献**。"
                 "但 R@3 一行在 w=0.1 处高出一档 —— 而线上链路是"
                 "「融合取 %d 条候选 → 重排取 %d 条」，R@3 决定的正是重排候选池里"
                 "有没有正确答案，这个位置比 R@1 更值钱。所以最终取 k=%d、w=%.1f："
                 "用 R@1/MRR 上 1 题以内（%.3f）的代价，换 R@3 上多命中 2 题。"
                 % (config.RRF_CANDIDATES, config.TOP_N_RERANK, config.RRF_K,
                    config.RRF_BM25_WEIGHT, 1.0 / len(questions)))
    lines.append(">")
    lines.append("> 这张表解释本报告里一个反直觉的结果：**等权融合（w=1.0）的 "
                 "Recall@1 低于纯向量**。RRF 的 k 把名次差异压平 —— k=60 时 "
                 "1/(60+1)=0.0164 与 1/(60+50)=0.0091 只差一倍，于是"
                 "「两路都召回但都排在末尾」的块（0.0091×2=0.0182）会赢过"
                 "「只在向量通道排第一」的块（0.0164）。BM25 召回质量明显更差，"
                 "它带进来的噪声候选就这样把正确答案挤下去了。"
                 "把 k 调小、或把 w 调小，都能消除这个反常。")
    lines.append(">")
    lines.append("> 相邻格子的差距大多只有 1~2 题（%d 题里 1 题 = %.3f），"
                 "不要过度解读单格；要看的是整体趋势。" % (len(questions),
                                                     1.0 / len(questions)))

    # 两通道各自的独占命中 —— 「那还要 BM25 干什么」这个问题的唯一证据。
    # 只看整体指标的话 BM25 是净负担，但它可能在**某类**问题上不可替代。
    if "bm25" in configs and "dense" in configs:
        b = {q: r for q, r, _ in results["bm25"]["details"]}
        d = {q: r for q, r, _ in results["dense"]["details"]}
        only_b = [q for q in b if b[q] and not d.get(q)]
        only_d = [q for q in d if d[q] and not b.get(q)]
        lines.append("")
        lines.append("**两通道的互补性**（前 %d 条内是否命中）" % max(KS))
        lines.append("")
        lines.append("| 只有 BM25 命中 | 只有向量命中 | 都命中 | 都没命中 |")
        lines.append("|---------------:|-------------:|-------:|---------:|")
        both = sum(1 for q in b if b[q] and d.get(q))
        neither = sum(1 for q in b if not b[q] and not d.get(q))
        lines.append("| %d | %d | %d | %d |"
                     % (len(only_b), len(only_d), both, neither))
        lines.append("")
        if only_b:
            lines.append("只有 BM25 命中的题（关键词通道不可替代的地方）：")
            for q in only_b[:6]:
                lines.append("- %s" % q)
            lines.append("")
        if only_d:
            lines.append("> 反向只有向量命中的题共 %d 道，这里不逐一列出；"
                         "两边数量上的差距就是「本语料上向量通道更强」的直接体现。"
                         % len(only_d))
            lines.append("")

    # ---- 重排的净增量：这是「还要不要留重排」的唯一证据。
    # 能逐格直接比的前提：所有配置都截到前 max(KS)=10 条，而重排的候选池是融合
    # 后的前 RRF_CANDIDATES 条（=24）。10 < 24，所以非重排配置的前 10 条本来就
    # 落在重排的候选池内 —— 两者**同池、同深度**，不存在候选池大小的口径差。
    if "hybrid" in configs and "hybrid+rerank" in configs:
        h = {q: r for q, r, _ in results["hybrid"]["details"]}
        rr = {q: r for q, r, _ in results["hybrid+rerank"]["details"]}
        demoted = [q for q in h if h[q] and not rr.get(q)]
        summ = {c: summarize(results[c]["ranks"])
                for c in ("hybrid", "hybrid+rerank")}
        nq = len(results["hybrid"]["ranks"])
        lines.append("")
        lines.append("**重排的净增量**（同一批候选、同一输出深度，逐格可比）")
        lines.append("")
        lines.append("| 指标 | `hybrid`（不重排） | `hybrid+rerank` | 增量 |")
        lines.append("|------|------------------:|----------------:|-----:|")
        for k in KS:
            a = summ["hybrid"]["recall@%d" % k]
            b = summ["hybrid+rerank"]["recall@%d" % k]
            lines.append("| Recall@%d | %.3f | %.3f | %+d 题 |"
                         % (k, a, b, int(round((b - a) * nq))))
        a, b = summ["hybrid"]["mrr@10"], summ["hybrid+rerank"]["mrr@10"]
        lines.append("| MRR@10 | %.3f | %.3f | %+.3f |" % (a, b, b - a))
        lines.append("| 未命中 | %d | %d | %+d 题 |"
                     % (summ["hybrid"]["miss"], summ["hybrid+rerank"]["miss"],
                        summ["hybrid+rerank"]["miss"] - summ["hybrid"]["miss"]))
        lines.append("")
        if demoted:
            lines.append("重排把原本命中的正确答案挤出前 %d 条的题（%d 道）："
                         % (max(KS), len(demoted)))
            for q in demoted[:8]:
                lines.append("- %s（不重排时命中在第 %d 条）" % (q, h[q]))
            lines.append("")

    # 拒答判定。两面都要报：只统计域外（该拒的拒掉了吗）的话，一个「见谁都拒」
    # 的系统能拿满分，所以必须同时报域内误伤了多少合法问题（不该拒的拒了吗）。
    #
    # 判据是查询向量余弦（config.MIN_QUERY_COSINE）。重排分那行留作对照 ——
    # 它分不开域内域外，正是被换掉的原因。
    cos = results.get("_cosine") or []
    ro = results.get("hybrid+rerank", {}).get("refuse_ok") or []
    fallback = results.get("hybrid", {}).get("refuse_ok") or []
    if cos or ro or fallback:
        lines.append("**拒答判定**（两面都要看）")
        lines.append("")
        lines.append("*正面 · 域外题（判成「该拒」才算对）*")
        lines.append("")
        lines.append("| 判定依据 | 正确拒答 | 准确率 |")
        lines.append("|----------|---------:|-------:|")
        if cos:
            ood = [s for s in cos if s["out_of_domain"]]
            ok = [s for s in ood if s["cos"] < config.MIN_QUERY_COSINE]
            lines.append("| 查询向量余弦 < `MIN_QUERY_COSINE`（%.2f，线上用的） | %d / %d | %.0f%% |"
                         % (config.MIN_QUERY_COSINE, len(ok), len(ood),
                            100.0 * len(ok) / len(ood)))
        if ro:
            lines.append("| 重排分 < `MIN_RERANK_SCORE`（旧判据，留作对照） | %d / %d | %.0f%% |"
                         % (sum(ro), len(ro), 100.0 * sum(ro) / len(ro)))
        if fallback:
            lines.append("| 兜底：RRF 分 < `MIN_RRF_SCORE` | %d / %d | %.0f%% |"
                         % (sum(fallback), len(fallback),
                            100.0 * sum(fallback) / len(fallback)))

    # 反面：域内误拒率。口径与召回率一致 —— 标注来源不在语料里的题剔掉，
    # 它们检索不到属于语料覆盖问题，不算阈值误伤。
    cos_in = [s for s in cos if not s["out_of_domain"] and not s["uncoverable"]]
    if cos_in:
        dropped = [s for s in cos if not s["out_of_domain"] and s["uncoverable"]]
        cos_false = [s for s in cos_in if s["cos"] < config.MIN_QUERY_COSINE]
        cos_vals = [s["cos"] for s in cos_in]
        ood_cos = [s["cos"] for s in cos if s["out_of_domain"]]
        lines.append("")
        lines.append("*反面 · 域内题（%d 道可答题，另 %d 道标注来源不在语料里的"
                     "已按召回率的同一口径剔除）*：**误拒 %d 道（%.1f%%）**。"
                     % (len(cos_in), len(dropped), len(cos_false),
                        100.0 * len(cos_false) / len(cos_in)))
        if cos_vals and ood_cos:
            gap = min(cos_vals) - max(ood_cos)
            lines.append("")
            lines.append("> 余弦分布：域内可答 **%.4f~%.4f**，域外 **%.4f~%.4f**，"
                         "中间空档 **%.4f**；阈值 %.2f 落在空档内%s。"
                         % (min(cos_vals), max(cos_vals), min(ood_cos), max(ood_cos),
                            gap, config.MIN_QUERY_COSINE,
                            "，两侧都还有余量" if gap > 0 else "，但两类已经交叠，需重新标定"))
        if cos_false:
            lines.append("")
            lines.append("| 被误拒的域内题 | 查询余弦 | 正确答案在融合阶段的名次 |")
            lines.append("|----------------|---------:|-------------------------:|")
            for s in sorted(cos_false, key=lambda x: x["cos"]):
                lines.append("| %s | %.4f | 第 %s 名 |"
                             % (s["q"], s["cos"],
                                s["fused_rank"] if s["fused_rank"] else "—"))

    # 对照：旧判据（重排分）在同一批题上的表现 —— 解释为什么换掉它。
    sc = results.get("hybrid+rerank", {}).get("scores") or []
    in_dom_r = [s for s in sc if not s["out_of_domain"] and not s["uncoverable"]]
    if in_dom_r:
        false_ref = [s for s in in_dom_r if s["score"] < config.MIN_RERANK_SCORE]
        kept = [s for s in in_dom_r if s["score"] >= config.MIN_RERANK_SCORE]
        ood_miss = [s for s in sc
                    if s["out_of_domain"] and s["score"] >= config.MIN_RERANK_SCORE]
        lines.append("")
        lines.append("> **对照 · 旧判据为什么被换掉**：同一批题上重排分对域内题"
                     "误拒 %d 道（%.1f%%）%s"
                     % (len(false_ref), 100.0 * len(false_ref) / len(in_dom_r),
                        ("：" + "、".join(
                            "%s（重排 %.3f，融合第 %s 名）"
                            % (s["q"], s["score"],
                               s["fused_rank"] if s["fused_rank"] else "—")
                            for s in sorted(false_ref, key=lambda x: x["score"])))
                        if false_ref else "。"))
        if kept and ood_miss:
            lo = min(kept, key=lambda x: x["score"])
            hi = max(ood_miss, key=lambda x: x["score"])
            lines.append("> 未被拒的域内题最低重排分 %.3f（%s），域外漏网题最高 %.3f（%s）—— "
                         "已经交叠，抬阈值必然误伤，所以换判据而不是调阈值。"
                         % (lo["score"], lo["q"], hi["score"], hi["q"]))

    lines.append("")
    qv = results.get("_qvec")
    if qv:
        lines.append("> **可复现性**：查询向量缓存在 `%s`（本次 %d/%d 条命中缓存）。"
                     % (qv["path"], qv["cached"], qv["total"]))
        lines.append("> 不缓存的话同一份索引跑两遍指标会差 1 道题 —— 索引、分词、RRF "
                     "都是确定性的，抖动来自向量接口在末位上的微小差异，"
                     "足以让临界题的排名整体挪一格。第一次跑写缓存，之后逐位一致。")
        lines.append("")
    lines.append("> 由 `python -m eval.run_eval` 生成，评测只跑检索、不调用大模型，"
                 "缓存落地后可重复执行。")

    report = "\n".join(lines) + "\n"
    print()
    print(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print("报告已写入 %s（耗时 %.1fs）" % (args.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
