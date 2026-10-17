# -*- coding: utf-8 -*-
"""单元测试：只测不依赖网络的部分（切分、分词、RRF、拒答判定）。

涉及向量接口和大模型的链路由 ``python -m eval.run_eval`` 与手工验证覆盖，
不放在单元测试里 —— 那会让测试依赖 API key，CI 上跑不起来。
"""

from __future__ import print_function

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing.chunk import (chunk_document, parse_sections,  # noqa: E402
                            split_sentences)
from indexing.embed import Embedder, EmbedError              # noqa: E402
from indexing.tokenize import tokenize                       # noqa: E402
from rag.rerank import should_refuse                         # noqa: E402
from rag.retriever import rrf_fuse                           # noqa: E402


DOC = {
    "doc_id": "wiki-1",
    "title": "高血压",
    "url": "https://example.org/高血压",
    "text": (
        "高血压是一种慢性病，指动脉血压长期持续偏高，是最常见的心血管疾病危险因素之一。"
        "多数患者早期没有明显症状，因此又被称为沉默的杀手，往往在体检时才被发现。"
        "\n\n## 诊断标准"
        "\n\n在未使用降压药物的情况下，非同日三次测量收缩压不低于140mmHg，"
        "或舒张压不低于90mmHg，即可诊断为高血压。家庭自测血压的判断标准略有不同，"
        "通常采用135/85mmHg作为参考界限，24小时动态血压监测则采用另一套阈值。"
        "单次测量结果偏高并不能直接确诊，需要在不同日期重复测量以排除白大衣高血压。"
        "\n\n### 血压测量"
        "\n\n测量前应当静坐五分钟以上，避免饮用含咖啡因的饮料，也不要刚做完运动就测。"
        "袖带需与心脏保持同一水平，袖带过松或过紧都会影响读数。"
        "首次测量应当两只手臂都测，之后以读数较高的一侧为准。"
        "\n\n## 参见"
        "\n\n- 低血压\n- 心血管疾病"
    ),
}


# ---------------------------------------------------------------- 切分

def test_parse_sections_建出标题路径():
    sections = parse_sections(DOC["text"], DOC["title"])
    paths = [p for p, _ in sections]
    assert "高血压" in paths
    assert "高血压 > 诊断标准" in paths
    # 三级标题要带上二级标题，形成完整路径
    assert "高血压 > 诊断标准 > 血压测量" in paths


def test_parse_sections_丢弃参见章节():
    sections = parse_sections(DOC["text"], DOC["title"])
    paths = [p for p, _ in sections]
    assert not any("参见" in p for p in paths)
    # 「参见」的正文也不该漏成无路径的野块
    assert "" not in paths


def test_parse_sections_噪声章节之后仍保留文档名():
    # 「参见」这类章节常常夹在中间。清栈清过头会让后面小节丢掉文档名，
    # 路径变成断头的「二级标题」，脱离上下文提问时就召回不到了。
    text = ("开头一段正文，讲的是这个东西是什么。\n\n"
            "## 参见\n\n- 别的条目\n\n"
            "## 治疗方法\n\n这里讲治疗方法的具体内容，句子要够长才不会被丢掉。")
    sections = dict(parse_sections(text, "测试条目"))
    assert "测试条目 > 治疗方法" in sections
    assert "治疗方法" not in sections
    assert not any("别的条目" in body for body in sections.values())


def test_chunk_每块都带来源信息():
    chunks = chunk_document(DOC)
    assert chunks, "应该至少切出一个块"
    for c in chunks:
        assert c["doc_id"] == DOC["doc_id"]
        assert c["title"] == DOC["title"]
        assert c["url"] == DOC["url"]
        assert c["text"].strip()
        # 送向量的文本必须带上标题路径，否则脱离上下文的提问召回不到
        assert c["section"] in c["embed_text"]


def test_chunk_正文不跨小节():
    chunks = chunk_document(DOC)
    diag = [c for c in chunks if "诊断标准" in c["section"]]
    assert diag, "应该切出诊断标准相关的块"
    for c in diag:
        # 「参见」的内容不应该混进诊断标准的块里
        assert "低血压" not in c["text"]


def test_split_sentences_在中文句末断开():
    sents = split_sentences("第一句。第二句！第三句？")
    assert len(sents) == 3


# ---------------------------------------------------------------- 分词

def test_tokenize_去掉停用词与标点():
    tokens = tokenize("的了和是高血压，需要治疗。")
    assert "高血压" in tokens
    assert "的" not in tokens
    assert "，" not in tokens
    assert "。" not in tokens


def test_tokenize_切出专业术语的子词():
    # cut_for_search 会把长词切成子词，细粒度对召回更友好
    tokens = tokenize("幽门螺杆菌")
    assert tokens, "不应切空"


def test_tokenize_空输入():
    assert tokenize("") == []
    assert tokenize("   ") == []


# ---------------------------------------------------------------- 向量化

def _fake_embedder(vectors_by_text, batch_size=2):
    """造一个不发网络请求的 Embedder：按文本查表返回向量。"""
    emb = Embedder(api_key="dummy", batch_size=batch_size)
    emb._embed_batch = lambda batch: [vectors_by_text[t] for t in batch]
    return emb


def _vec(seed, dim=None, scale=1.0):
    dim = dim or 4
    return [(seed + i) * scale for i in range(dim)]


def test_embed_跨批次按原顺序拼装(monkeypatch):
    # 分批请求时最容易出的错是把后一批写到前面去，
    # 那样向量和元数据就错位了（chunks_meta.jsonl 靠下标对齐）
    monkeypatch.setattr("config.EMBED_DIM", 4)
    texts = ["a", "b", "c", "d", "e"]
    vecs = {t: _vec(n) for n, t in enumerate(texts)}
    emb = _fake_embedder(vecs, batch_size=2)   # 5 条分 3 批，故意除不尽
    try:
        mat = emb.embed(texts)
    finally:
        emb.close()
    assert mat.shape == (5, 4)

    want = np.asarray([_vec(n) for n in range(5)], dtype=np.float32)
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    assert np.allclose(mat, want), "分批拼接后行序乱了"


def test_embed_做l2归一化(monkeypatch):
    monkeypatch.setattr("config.EMBED_DIM", 4)
    emb = _fake_embedder({"a": [3.0, 0.0, 0.0, 4.0]})
    try:
        mat = emb.embed(["a"])
    finally:
        emb.close()
    assert abs(float((mat[0] ** 2).sum()) - 1.0) < 1e-6


def test_embed_维度不符时报错(monkeypatch):
    monkeypatch.setattr("config.EMBED_DIM", 1024)
    emb = _fake_embedder({"a": [1.0, 2.0, 3.0, 4.0]})
    try:
        with pytest.raises(EmbedError):
            emb.embed(["a"])
    finally:
        emb.close()


def test_embed_空输入不报错(monkeypatch):
    monkeypatch.setattr("config.EMBED_DIM", 4)
    emb = _fake_embedder({})
    try:
        mat = emb.embed([])
    finally:
        emb.close()
    assert mat.shape == (0, 4)


# ---------------------------------------------------------------- RRF

def test_rrf_两路都命中的排在前面():
    dense = [10, 20, 30]
    bm25 = [30, 40, 50]
    fused = [i for i, _ in rrf_fuse([dense, bm25], k=60)]
    # 30 被两条通道分别排在第 3、第 1，融合后应该排第一
    assert fused[0] == 30


def test_rrf_只有一路召回也能保留():
    fused = [i for i, _ in rrf_fuse([[1, 2, 3]], k=60)]
    assert fused == [1, 2, 3]


def test_rrf_空输入():
    assert rrf_fuse([[], []], k=60) == []


def test_rrf_权重压住弱通道带进来的噪声():
    # 块 999 在向量通道排第一；块 0 在两路都排最后一名。
    # k=60 把名次差异压得很平（1/61 与 1/110 只差一倍），等权融合时
    # 「两路都垫底」的块 0 会反超「向量第一」的块 999 ——
    # 这正是等权 hybrid 的 Recall@1 反而低于纯向量的原因（见 eval/report.md）。
    dense = [999] + list(range(1, 49)) + [0]     # 999 第 1，0 第 50
    bm25 = list(range(100, 149)) + [0]           # 0 第 50，其余全是噪声
    assert len(dense) == 50 and len(bm25) == 50

    equal = [i for i, _ in rrf_fuse([dense, bm25], k=60, weights=[1.0, 1.0])]
    assert equal[0] == 0, "等权时弱通道的噪声候选会挤掉向量第一"

    tuned = [i for i, _ in rrf_fuse([dense, bm25], k=60, weights=[1.0, 0.3])]
    assert tuned[0] == 999, "压低 BM25 权重后向量第一应该回到首位"


# ---------------------------------------------------------------- 拒答

def test_should_refuse_没有候选时拒答():
    assert should_refuse([]) is True


def test_should_refuse_按查询向量余弦判定():
    # 余弦是主判据：域外题即使重排分很高也要拒，域内题即使重排分很低也放行
    out_of_domain = [({"query_cosine": 0.50}, 0.90)]
    in_domain = [({"query_cosine": 0.70}, 0.05)]
    assert should_refuse(out_of_domain) is True
    assert should_refuse(in_domain) is False


def test_should_refuse_低分拒答高分放行():
    # 没有余弦时退回重排分
    low = [({"rrf_score": 0.01}, 0.02)]
    high = [({"rrf_score": 0.03}, 0.85)]
    assert should_refuse(low) is True
    assert should_refuse(high) is False


def test_should_refuse_重排降级时改用rrf阈值():
    # 重排不可用时分数为 None，此时应该退回用 rrf_score 判断
    ok = [({"rrf_score": 0.05}, None)]
    bad = [({"rrf_score": 0.0001}, None)]
    assert should_refuse(ok) is False
    assert should_refuse(bad) is True


# ---------------------------------------------------------------- 查询向量缓存

class _CountingEmbedder(object):
    """假的向量客户端：只数调用次数，不碰网络。"""

    def __init__(self):
        self.calls = 0

    def embed_one(self, question):
        self.calls += 1
        rng = np.random.RandomState(len(question) * 7919 % (2 ** 31))
        return rng.randn(1024).astype("float32")


def test_查询向量缓存第二次不再调接口():
    from eval.run_eval import query_vector

    emb = _CountingEmbedder()
    cache = {}
    _v1, hit1 = query_vector(emb, "高血压吃什么药", cache)
    _v2, hit2 = query_vector(emb, "高血压吃什么药", cache)
    assert hit1 is False, "第一次必然未命中，要回源"
    assert hit2 is True, "第二次必须命中缓存"
    assert emb.calls == 1, "命中缓存时不该再调向量接口"


def test_查询向量缓存落盘往返逐位一致(tmp_path):
    # 缓存把 float32 序列化成 JSON 文本，读回来必须逐位相同 ——
    # 只要末位有差异，临界题的排名就会挪一格，「重跑结果一致」就成了空话。
    from eval.run_eval import load_qvec, query_vector, save_qvec

    emb = _CountingEmbedder()
    cache = {}
    v1, _ = query_vector(emb, "每天应该喝多少水", cache)
    path = str(tmp_path / "qvec.json")
    save_qvec(path, cache)

    reloaded = load_qvec(path)
    v2, hit = query_vector(emb, "每天应该喝多少水", reloaded)
    assert hit is True
    assert emb.calls == 1, "从磁盘读回的缓存也必须生效"
    assert np.array_equal(np.asarray(v1, dtype="float32"),
                          np.asarray(v2, dtype="float32"))


def test_查询向量缓存按模型分开存():
    # 消融实验里四个变体用了两个不同的向量模型，两者维度都是 1024 ——
    # 共用一份缓存不会报错，只会静默算出错的排名，所以这里必须守住。
    from eval.run_eval import default_qvec_path

    a = default_qvec_path("BAAI/bge-m3")
    b = default_qvec_path("BAAI/bge-large-zh-v1.5")
    assert a != b, "不同向量模型的缓存必须是不同文件"
    assert "/" not in os.path.basename(a), "模型名里的斜杠不能带进文件名"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
