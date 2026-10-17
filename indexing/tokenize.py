# -*- coding: utf-8 -*-
"""中文分词，供 BM25 通道使用。

索引时和查询时**必须用同一个分词器**，否则 BM25 的词表对不上，
关键词通道会静默失效（查询词全变成 OOV）。所以单独抽成一个模块，
两边都从这里 import。

用 ``jieba.cut_for_search``：它会把长词再切成子词（"高血压" → 高血压/高/血压），
细粒度对召回更友好，尤其适合专业术语。
"""

from __future__ import print_function

import re

import jieba

# 中文里的高频虚词。留着它们会让 BM25 偏向长文档，对检索没帮助。
STOPWORDS = set("""
的 了 和 是 在 有 就 不 也 都 与 及 或 而 但 对 从 到 被 把 让 使 为 于 以 之 其 该 此
这 那 这个 那个 这些 那些 这样 那样 一个 一些 一种 一样 什么 怎么 如何 哪些 哪个
可以 可能 需要 应该 必须 进行 通过 由于 因为 所以 如果 虽然 但是 并且 以及
我们 你们 他们 它们 自己 其中 之一 以上 以下 之间 同时 此外 另外 例如 比如
一般 通常 往往 主要 重要 很多 很少 非常 比较 更加 最 更 很 太 十分 特别
时候 情况 方面 问题 方式 方法 过程 结果 原因 影响 作用 关系 部分 不同 相同
of the and to in a is are for on with as by at be or an
""".split())

# 只由标点/空白组成的 token 直接丢
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)


def tokenize(text):
    """切词并去停用词，返回 token 列表。"""
    if not text:
        return []
    tokens = []
    for w in jieba.cut_for_search(text):
        w = w.strip().lower()
        if not w or w in STOPWORDS or _PUNCT_ONLY.match(w):
            continue
        tokens.append(w)
    return tokens


def tokenize_batch(texts, progress=False):
    import sys
    out = []
    n = len(texts)
    for i, t in enumerate(texts, 1):
        out.append(tokenize(t))
        if progress and i % 500 == 0:
            sys.stderr.write("\r  分词 %d/%d" % (i, n))
            sys.stderr.flush()
    if progress:
        sys.stderr.write("\r  分词 %d/%d 完成\n" % (n, n))
    return out
