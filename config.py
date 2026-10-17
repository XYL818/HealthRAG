# -*- coding: utf-8 -*-
"""集中配置。

所有可调项都从环境变量读，部署时用 Docker 的 environment 覆盖即可，
不用改代码。

本地开发把密钥写在仓库根目录的 ``.env``（已加入 .gitignore）：

    EMBED_API_KEY=sk-xxxx
    LLM_API_KEY=sk-xxxx

``.env`` 只在变量尚未设置时生效，不会覆盖已有的环境变量。
**千万不要把 .env 提交进仓库**，仓库里给的是 .env.example。
"""

from __future__ import print_function

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_dotenv(path=None):
    """极简 .env 读取 —— 为一个小功能引入 python-dotenv 不划算。"""
    path = path or os.path.join(_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()


def env(name, default=""):
    return os.environ.get(name, default)


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- 路径

DATA_DIR = env("DATA_DIR", os.path.join(_ROOT, "data"))
INDEX_DIR = env("INDEX_DIR", os.path.join(_ROOT, "indexing", "store"))
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")

FAISS_PATH = os.path.join(INDEX_DIR, "dense.faiss")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.json")
META_PATH = os.path.join(INDEX_DIR, "chunks_meta.jsonl")
INDEX_META_PATH = os.path.join(INDEX_DIR, "index_meta.json")

# ---------------------------------------------------------------- 向量模型

# 默认走硅基流动（OpenAI 兼容协议）。换成任何 OpenAI 兼容的服务都只需改这三个变量。
EMBED_BASE_URL = env("EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
EMBED_API_KEY = env("EMBED_API_KEY")
EMBED_MODEL = env("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = env_int("EMBED_DIM", 1024)
EMBED_BATCH = env_int("EMBED_BATCH", 16)

# ---------------------------------------------------------------- 生成模型

LLM_BASE_URL = env("LLM_BASE_URL", EMBED_BASE_URL)
LLM_API_KEY = env("LLM_API_KEY") or EMBED_API_KEY
LLM_MODEL = env("LLM_MODEL", "deepseek-ai/DeepSeek-V3")
LLM_TEMPERATURE = env_float("LLM_TEMPERATURE", 0.2)
LLM_MAX_TOKENS = env_int("LLM_MAX_TOKENS", 1024)
LLM_TIMEOUT = env_float("LLM_TIMEOUT", 90.0)

# ---------------------------------------------------------------- 重排

# bge-reranker 与 bge-m3 同源，硅基流动上走 /v1/rerank
RERANK_BASE_URL = env("RERANK_BASE_URL", EMBED_BASE_URL)
RERANK_API_KEY = env("RERANK_API_KEY") or EMBED_API_KEY
RERANK_MODEL = env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
ENABLE_RERANK = env("ENABLE_RERANK", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------- 检索参数

TOP_K_DENSE = env_int("TOP_K_DENSE", 20)     # 向量通道召回数
TOP_K_BM25 = env_int("TOP_K_BM25", 20)       # 关键词通道召回数
RRF_K = env_int("RRF_K", 60)                 # RRF 融合平滑常数
TOP_N_RERANK = env_int("TOP_N_RERANK", 6)    # 重排后喂给大模型的条数
RRF_CANDIDATES = env_int("RRF_CANDIDATES", 24)   # 融合后进入重排的候选数

# RRF 融合权重（向量通道 / 关键词通道）。默认 1.0 : 0.1 是**实测扫出来的**，
# 不是拍脑袋，扫参表见 eval/report.md 的「融合参数扫描」。
#
# 两条实测结论：
# 1. 等权 1:1 时 hybrid 的 Recall@1 反而低于纯向量（0.803 vs 0.855）。
#    k=60 把名次差异压得很平（1/61=0.0164 对 1/110=0.0091 只差一倍），
#    于是「两路都召回但都垫底」的块（0.0182）会赢过「只在向量通道排第一」的块。
# 2. 本语料上向量通道占绝对主导：BM25 权重给 0.0（等于关掉关键词通道）
#    在 Recall@1 和 MRR 上都是最优，说明 BM25 带进来的候选基本是噪声。
#
# 那为什么还留 0.1 而不是 0？因为线上链路是「融合取 RRF_CANDIDATES 条 → 重排取
# TOP_N_RERANK 条」，真正决定成败的是**候选池里有没有正确答案**（看 R@3），
# 而 R@3 在 w=0.1 处比 w=0.0 高（0.957 vs 0.940）。用 R@1 上 1 题的代价
# 换 R@3 上多命中 2 题，划算。改这两个值之前先重跑评测。
RRF_DENSE_WEIGHT = env_float("RRF_DENSE_WEIGHT", 1.0)
RRF_BM25_WEIGHT = env_float("RRF_BM25_WEIGHT", 0.1)

# 拒答阈值：查询向量与语料的最高余弦低于此值，就认为资料库里没有相关内容。
#
# 为什么用它而不是重排分：实测重排分分不开域内和域外（域内可答题最低 0.423，
# 域外漏网题最高 0.490，已经交叠）；而查询向量余弦在 129 道评测题上干净分开
# ——域内可答 0.6408~0.8193，域外 0.4594~0.5780，中间 0.0628 的空档。
# 取 0.60，两侧各留约 0.02 的余量。换向量模型后必须重新标定，详见 eval/report.md。
MIN_QUERY_COSINE = env_float("MIN_QUERY_COSINE", 0.60)
# 下面是兜底判据，只在取不到查询向量时才轮到。
# 重排分低于此值拒答；不同 rerank 模型分值尺度不同，这里是 bge-reranker-v2-m3 的经验值。
MIN_RERANK_SCORE = env_float("MIN_RERANK_SCORE", 0.15)
# 重排也不可用时改用 RRF 分数阈值（名次信息，区分度很弱）。
MIN_RRF_SCORE = env_float("MIN_RRF_SCORE", 0.008)
