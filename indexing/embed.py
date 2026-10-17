# -*- coding: utf-8 -*-
"""把文本批量转成向量（调用 OpenAI 兼容的 /v1/embeddings 接口）。

默认用硅基流动的 ``BAAI/bge-m3``（1024 维，中文效果好，有免费额度）。
换成别的 OpenAI 兼容服务只需改 ``EMBED_BASE_URL`` / ``EMBED_MODEL``。

向量做 **L2 归一化** 后存成 float32：这样 FAISS 的内积就等于余弦相似度，
省掉一次开方，也让归一化这件事只在一个地方发生。

用法
----
    python -m indexing.embed --test          # 自检，确认 key 和网络通
"""

from __future__ import print_function

import argparse
import sys
import time

import httpx
import numpy as np

import config


class EmbedError(RuntimeError):
    pass


def _l2_normalize(mat):
    # 原地归一化：整份语料建索引时矩阵有几十 MB，
    # 再复制一份会让峰值内存翻倍（实测 14152 个块时把 700m 的容器撑爆过）
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    mat /= norm
    return mat


class Embedder(object):
    """OpenAI 兼容的向量化客户端。"""

    def __init__(self, base_url=None, api_key=None, model=None,
                 batch_size=None, timeout=60.0):
        self.base_url = (base_url or config.EMBED_BASE_URL).rstrip("/")
        self.api_key = api_key or config.EMBED_API_KEY
        self.model = model or config.EMBED_MODEL
        self.batch_size = batch_size or config.EMBED_BATCH
        self.timeout = timeout

        if not self.api_key:
            raise EmbedError(
                "没有配置 EMBED_API_KEY。请在仓库根目录建一个 .env，写入：\n"
                "    EMBED_API_KEY=sk-xxxx\n"
                "（.env 已在 .gitignore 里，不会被提交）"
            )

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

    def _embed_batch(self, batch, max_retry=5):
        payload = {"model": self.model, "input": batch}
        last = None
        for attempt in range(max_retry):
            try:
                r = self._client.post("/embeddings", json=payload)
                if r.status_code == 200:
                    data = r.json()["data"]
                    # 接口不保证返回顺序，按 index 排回去
                    data = sorted(data, key=lambda d: d.get("index", 0))
                    return [d["embedding"] for d in data]
                # 429 / 5xx 退避重试；4xx 其它情况直接报错
                if r.status_code in (429, 500, 502, 503, 504):
                    last = "HTTP %d: %s" % (r.status_code, r.text[:200])
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                raise EmbedError("向量化失败 HTTP %d: %s"
                                 % (r.status_code, r.text[:400]))
            except httpx.HTTPError as e:
                last = repr(e)
                time.sleep(1.5 * (2 ** attempt))

        # 整批重试都失败：如果批里不止一条，拆小再试，避免一条坏数据拖垮整批
        if len(batch) > 1:
            mid = len(batch) // 2
            return (self._embed_batch(batch[:mid], max_retry)
                    + self._embed_batch(batch[mid:], max_retry))
        raise EmbedError("向量化重试 %d 次仍失败: %s" % (max_retry, last))

    def embed(self, texts, progress=False, desc=""):
        """把一批文本转成 (N, dim) 的归一化 float32 矩阵。

        边取边写进预分配的 numpy 数组，**不要**先攒成 Python 列表再转 ——
        每个 float 在 CPython 里是个独立对象，14152×1024 攒下来光对象开销
        就有 400MB 上下，整份语料建索引时会把内存撑爆。
        """
        if not texts:
            return np.zeros((0, config.EMBED_DIM), dtype=np.float32)

        total = len(texts)
        mat = None
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            vecs = self._embed_batch(batch)
            if len(vecs) != len(batch):
                raise EmbedError("返回条数不符：请求 %d 条，返回 %d 条"
                                 % (len(batch), len(vecs)))
            arr = np.asarray(vecs, dtype=np.float32)
            if arr.shape[1] != config.EMBED_DIM:
                raise EmbedError("向量维度不符：期望 %d，实际 %d"
                                 % (config.EMBED_DIM, arr.shape[1]))
            if mat is None:
                mat = np.zeros((total, arr.shape[1]), dtype=np.float32)
            mat[i:i + arr.shape[0]] = arr

            if progress:
                done = min(i + self.batch_size, total)
                sys.stderr.write("\r  %s %d/%d (%.0f%%)"
                                 % (desc or "向量化", done, total,
                                    100.0 * done / total))
                sys.stderr.flush()

        if progress:
            sys.stderr.write("\n")

        return _l2_normalize(mat)

    def embed_one(self, text):
        return self.embed([text])[0]

    # 检索时复用同一个 client，别每条查询都新建连接
    embed_query = embed_one


def main():
    ap = argparse.ArgumentParser(description="向量化自检")
    ap.add_argument("--test", action="store_true", help="跑一个自检")
    args = ap.parse_args()

    if not args.test:
        ap.print_help()
        return 0

    print("BASE_URL = %s" % config.EMBED_BASE_URL)
    print("MODEL    = %s" % config.EMBED_MODEL)

    with Embedder() as emb:
        texts = ["高血压的诊断标准是什么？", "每天应该喝多少水？", "高血压怎么判断"]
        vecs = emb.embed(texts)
        print("向量形状: %s" % (vecs.shape,))
        # 归一化后，向量点积即余弦相似度
        sim_related = float(vecs[0] @ vecs[2])
        sim_unrelated = float(vecs[0] @ vecs[1])
        print("「高血压诊断标准」vs「高血压怎么判断」 余弦 = %.4f" % sim_related)
        print("「高血压诊断标准」vs「每天喝多少水」   余弦 = %.4f" % sim_unrelated)
        ok = sim_related > sim_unrelated
        print("语义区分正常" if ok else "警告：相近句子的相似度没有更高，检查模型")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
