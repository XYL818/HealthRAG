# 生成侧评测（RAGAS）

抽取 20 道域内问题。两组配置**共用同一批检索结果**，唯一的差别是 system prompt 里有没有「只依据资料作答」这条约束，所以两行之差就是接地约束本身的贡献。

> 评测的是生成质量，不是检索质量 —— 检索指标见 `eval/report.md`。

| 配置 | faithfulness | answer_relevancy | 有效样本 |
|------|-------------:|-----------------:|---------:|
| `grounded`（线上 prompt） | 0.976 | 0.662 | 20 |
| `ungrounded`（去掉接地约束） | 0.925 | 0.664 | 20 |

> 裁判模型 `Qwen/Qwen3-30B-A3B-Instruct-2507`，向量模型 `BAAI/bge-m3`。裁判与被评的生成模型（`deepseek-ai/DeepSeek-V3`）不同家族，避免自己判自己。
>
> RAGAS 是 LLM-as-judge，**绝对分数受裁判模型影响很大**，不要单独引用某一行；有意义的是同一裁判下两行的差。

> 由 `python -m eval.run_ragas` 生成，逐题回答缓存在 `eval/ragas_samples.jsonl`，重跑只补缺失的题。
