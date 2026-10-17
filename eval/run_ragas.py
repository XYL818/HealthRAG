# -*- coding: utf-8 -*-
"""生成侧评测：用 RAGAS 量「接地约束」值多少分。

为什么不直接报一个 faithfulness 数字
-------------------------------------
RAGAS 是 LLM-as-judge，绝对分数受裁判模型影响很大，单看一个 0.93 说明不了什么
（换个裁判模型就变）。所以这里不比绝对值，而是**做对照**：

两组配置**共用同一批检索结果**，system prompt 是唯一的自变量 ——

============  ==========================================================
配置            system prompt
============  ==========================================================
``grounded``   线上实际用的那个：只依据资料作答，每个事实标 [编号]
``ungrounded`` 去掉接地约束：允许模型用自己的知识把资料补全
============  ==========================================================

于是两行的差就是「接地约束」本身的贡献。这比一个孤零零的分数有说服力得多：
它回答的是「你为什么不直接把问题丢给大模型」这个一定会被问到的问题。

测哪两个指标
------------
- ``faithfulness``      回答里的每个论断是否都能从**检索到的资料**里找到依据
- ``answer_relevancy``  回答是否切题（用问题反推回答，看语义相似度）

预期 ungrounded 的 faithfulness 明显更低；如果两者的 relevancy 差不多甚至
ungrounded 更高，那正好说明：**接地的代价不是答得差，而是答得更保守** ——
这正是健康领域愿意付的代价。

用法
----
    python -m eval.run_ragas --limit 40

生成结果会缓存在 ``eval/ragas_samples.jsonl``，重跑只补缺失的题，
不会重复烧钱。删掉那个文件即可全部重来。
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from eval.run_eval import load_questions
from rag.generator import UNGROUNDED_PROMPT, Generator
from rag.pipeline import RagPipeline

_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_PATH = os.path.join(_HERE, "ragas_samples.jsonl")

# 裁判模型必须**不同于**被评的生成模型，否则等于自己判自己。
# 这里用通义千问：硅基流动上价格低、中文好，且和 DeepSeek-V3 不同家族。
JUDGE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


# ---------------------------------------------------------------- 生成答案

def load_samples():
    """读已缓存的问题 → 回答，用来断点续跑。"""
    out = {}
    if not os.path.exists(SAMPLES_PATH):
        return out
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            out[item["question"]] = item
    return out


def collect(questions, limit):
    """跑检索 + 两遍生成，产出评测样本（带缓存）。

    ``limit`` 是**参与评测的样本总数**，不是「这次新生成几题」。按后者理解的话，
    缓存里已有的题不计入，重跑时调小 limit 反而会继续往后多生成 ——
    想缩小规模却多烧一遍钱，正好反了。所以这里先算还差几题。
    """
    cached = load_samples()
    if cached:
        print("已有 %d 条缓存 ← %s" % (len(cached), os.path.relpath(SAMPLES_PATH)))

    have = [q for q in questions if q in cached]
    todo = [q for q in questions if q not in cached]
    if limit:
        todo = todo[:max(0, limit - len(have))]
    if not todo:
        print("缓存已够（%d 条），不再生成" % len(have))
        return [cached[q] for q in questions if q in cached][:limit or None]

    print("待生成 %d 题（每题生成两遍：接地 / 不接地）" % len(todo))

    samples = []
    with RagPipeline() as pipe, \
            Generator() as grounded, \
            Generator(system_prompt=UNGROUNDED_PROMPT) as ungrounded, \
            open(SAMPLES_PATH, "a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            t0 = time.time()
            # 两组共用同一批检索结果，检索就不是自变量了
            scored, refused = pipe.retrieve(q)
            items = [it for it, _ in scored]
            contexts = [it["text"] for it in items]

            if refused:
                # 拒答的题不进生成侧评测：没有回答可评。它的对错由
                # eval/report.md 的拒答判定那一项负责。
                rec = {"question": q, "refused": True, "contexts": contexts,
                       "grounded": "", "ungrounded": ""}
            else:
                rec = {
                    "question": q,
                    "refused": False,
                    "contexts": contexts,
                    "grounded": grounded.answer(q, items),
                    "ungrounded": ungrounded.answer(q, items),
                }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            samples.append(rec)
            print("  [%d/%d] %s（%.1fs）"
                  % (i, len(todo), q[:24], time.time() - t0))

    # 缓存里可能有更早的题（这次 --limit 没覆盖到），一并带上，再按总数截断
    all_samples = load_samples()
    return [all_samples[q] for q in questions if q in all_samples][:limit or None]


# ---------------------------------------------------------------- RAGAS

def build_judge():
    """构造裁判 LLM 与向量模型。

    RAGAS 是通过 LangChain 的 wrapper 去调模型的 —— langchain 是它拉进来的，
    不是我们主动选的框架，所以只出现在这个评测脚本里，运行镜像不含它。

    向量这一环**必须**用 ``LangchainEmbeddingsWrapper`` 包一层：ragas 0.4.3 自带的
    ``ragas.embeddings.OpenAIEmbeddings`` 只实现了新版接口（``embed_text``），
    而 ``evaluate()`` 认的旧版答案相关性指标调的是 ``embed_query`` —— 直接用它
    会抛 AttributeError，而 RAGAS 把该异常吞成 NaN，最终表现为
    ``answer_relevancy`` 恒为 0、且不报任何错。这个坑是冒烟测试才暴露出来的。
    """
    from langchain_openai import OpenAIEmbeddings as LangchainEmbeddings
    from openai import OpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import llm_factory

    # 裁判和向量都走同一个 OpenAI 兼容端点（硅基流动），只是换模型名
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
                    timeout=120.0)
    # max_tokens 必须显式给大一点。faithfulness 要把回答拆成逐条论断再逐一验证，
    # 输出比一般问答长得多；用默认值时长回答会被截断，抛
    # IncompleteOutputException，该样本直接记 NaN —— 实测 76 个子任务里中招好几个，
    # 而它不报错、只是悄悄把有效样本数变小。宁可慢一点也要把它调大。
    llm = llm_factory(JUDGE_MODEL, provider="openai", client=client,
                      max_tokens=4096)
    emb = LangchainEmbeddingsWrapper(LangchainEmbeddings(
        model=config.EMBED_MODEL, api_key=config.EMBED_API_KEY,
        base_url=config.EMBED_BASE_URL))
    return llm, emb


def score(rows, key, llm, emb):
    """对一列回答跑 RAGAS，返回 {指标: 均值} 与有效样本数。"""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    # 必须从 ragas.metrics 取（旧版类），**不能**从 ragas.metrics.collections 取。
    # 0.4.3 里 evaluate() 校验的是 Metric 基类，而 collections 里的新版对象继承
    # BaseMetric，两者不兼容 —— 用 collections 会抛
    # TypeError: All metrics must be initialised metric objects，
    # 尽管 ragas 自己的弃用提示正是让你改用 collections（版本内部没对齐）。
    # 导入 ragas.metrics 会有 DeprecationWarning，但它是唯一 evaluate() 认的来源。
    from ragas.metrics import AnswerRelevancy, Faithfulness

    samples = [SingleTurnSample(user_input=r["question"],
                                retrieved_contexts=r["contexts"],
                                response=r[key])
               for r in rows]
    # strictness 默认是 3：RAGAS 会向接口要 3 个候选问题，而硅基流动只返回 1 个，
    # 于是它按「返回数量不符合要求」重试，白等好几轮（日志里那句
    # "LLM returned 1 generations instead of requested 3" 就是它）。语义上只用得上
    # 1 个，直接声明成 1，把这串重试整个省掉。
    metrics = [Faithfulness(llm=llm),
               AnswerRelevancy(llm=llm, embeddings=emb, strictness=1)]
    # batch_size 是同时在飞的子任务数。**不要往上调**：试过 8，结果整轮卡死 ——
    # 接口在 8 路并发下触发限流，而 RAGAS 的重试策略是 max_retries=10、退避上限
    # 60 秒，于是一个子任务能闷头重试十几分钟，表现为进度条一直停在 0/40。
    # 4 是实测能稳定推进的值；想再快只能减少题量，不能加并发。
    result = evaluate(EvaluationDataset(samples=samples), metrics=metrics,
                      llm=llm, embeddings=emb, batch_size=4,
                      show_progress=True)
    frame = result.to_pandas()
    out = {}
    for m in ("faithfulness", "answer_relevancy"):
        if m in frame.columns:
            vals = [v for v in frame[m].tolist() if v == v]   # 去掉 NaN
            out[m] = (sum(vals) / len(vals) if vals else 0.0, len(vals))
        else:
            out[m] = (0.0, 0)
    return out


def main():
    ap = argparse.ArgumentParser(description="RAGAS 生成侧评测（接地 vs 不接地）")
    ap.add_argument("--questions", default=os.path.join(_HERE, "questions.jsonl"))
    ap.add_argument("--limit", type=int, default=40,
                    help="评测多少题（**总数**，含缓存里已有的；每题两遍生成 "
                         "+ 若干次裁判调用，都是钱）")
    ap.add_argument("--out", default=os.path.join(_HERE, "ragas_report.md"))
    args = ap.parse_args()

    questions = [q["q"] for q in load_questions(args.questions)
                 if not q.get("should_refuse")]
    if not questions:
        print("评测集为空", file=sys.stderr)
        return 1

    t0 = time.time()
    rows = collect(questions, args.limit)
    # 拒答的题没有回答可评，剔除
    rows = [r for r in rows if r.get("grounded") and r.get("ungrounded")]
    if not rows:
        print("没有可评测的样本", file=sys.stderr)
        return 1
    print("参与评测 %d 题（拒答的题不计）" % len(rows))

    print("构造裁判（%s）..." % JUDGE_MODEL)
    llm, emb = build_judge()

    table = []
    for key, label in (("grounded", "`grounded`（线上 prompt）"),
                       ("ungrounded", "`ungrounded`（去掉接地约束）")):
        print("评测 %s ..." % key)
        table.append((label, score(rows, key, llm, emb)))

    # ---------------------------------------------------------- 报告
    lines = []
    lines.append("# 生成侧评测（RAGAS）")
    lines.append("")
    lines.append("抽取 %d 道域内问题。两组配置**共用同一批检索结果**，"
                 "唯一的差别是 system prompt 里有没有「只依据资料作答」这条约束，"
                 "所以两行之差就是接地约束本身的贡献。" % len(rows))
    lines.append("")
    lines.append("> 评测的是生成质量，不是检索质量 —— 检索指标见 `eval/report.md`。")
    lines.append("")
    lines.append("| 配置 | faithfulness | answer_relevancy | 有效样本 |")
    lines.append("|------|-------------:|-----------------:|---------:|")
    for label, m in table:
        n = max(m["faithfulness"][1], m["answer_relevancy"][1])
        lines.append("| %s | %.3f | %.3f | %d |"
                     % (label, m["faithfulness"][0], m["answer_relevancy"][0], n))
    lines.append("")
    lines.append("> 裁判模型 `%s`，向量模型 `%s`。裁判与被评的生成模型"
                 "（`%s`）不同家族，避免自己判自己。"
                 % (JUDGE_MODEL, config.EMBED_MODEL, config.LLM_MODEL))
    lines.append(">")
    lines.append("> RAGAS 是 LLM-as-judge，**绝对分数受裁判模型影响很大**，"
                 "不要单独引用某一行；有意义的是同一裁判下两行的差。")
    lines.append("")
    lines.append("> 由 `python -m eval.run_ragas` 生成，逐题回答缓存在 "
                 "`eval/ragas_samples.jsonl`，重跑只补缺失的题。")

    report = "\n".join(lines) + "\n"
    print()
    print(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print("报告已写入 %s（总耗时 %.1f 分钟）" % (args.out, (time.time() - t0) / 60.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
