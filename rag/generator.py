# -*- coding: utf-8 -*-
"""生成：把检索到的资料拼进 prompt，让大模型带着引用回答。

两个关键约束
------------
1. **只依据资料回答**。open-book 提问，资料里没有的就不许编 —— 这是 RAG
   区别于直接问大模型的全部意义。prompt 里反复强调，且要求逐条标注 [编号]。
2. **健康领域的安全边界**。不做诊断、不开药、不给剂量建议；
   涉及这些时只做一般性科普并提示就医。这是产品要求，不是免责话术 ——
   一个健康问答如果敢直接说"你这是高血压，吃 XX 药"，本身就是不可用的。

回答走 OpenAI 兼容的 /chat/completions（默认硅基流动的 DeepSeek-V3），
支持流式输出。
"""

from __future__ import print_function

import json
import sys

import httpx

import config

REFUSAL_MESSAGE = (
    "资料库里没有找到能回答这个问题的内容。\n\n"
    "你可以换个说法再问，或者咨询医生、营养师等专业人士。"
)

SYSTEM_PROMPT = """你是「知健」，一个生活与健康知识的检索问答助手。你的回答必须**严格基于**用户提供的资料片段。

规则：
1. 只使用资料片段中的信息作答。资料里没有的，一律不要补充，即使你自己知道。
2. 每个事实性陈述后面用 [编号] 标注来源，编号对应资料片段的编号。一句里引用多个来源就写 [1][3]。
3. 如果资料片段不足以回答问题，直接说「资料库里没有找到相关内容」，并建议咨询专业人士。**绝不允许编造**。
4. 遇到涉及疾病诊断、用药、剂量、治疗方案的问题：只做一般性知识说明，不要给出针对个人的诊断或用药建议，
   并在结尾提示这不构成医疗建议、请及时就医。
5. 用简体中文，简洁分点作答，不要复述资料原文，用你自己的话概括。
6. 不要写"根据资料1"这类话，直接写结论再标注编号即可。"""

# 评测对照组专用的 prompt：**故意**不加「只依据资料」这条约束。
# 对照组看到的是同一批资料片段，唯一的差别是允不允许模型自由发挥，
# 于是两者之差就是「接地约束」本身的贡献（见 eval/run_ragas.py）。
UNGROUNDED_PROMPT = """你是「知健」，一个生活与健康知识的问答助手。

规则：
1. 回答用户的问题。可以使用下面提供的资料片段，也可以使用你自己掌握的知识。
2. 资料片段不完整时，用你自己的知识把它补全。
3. 用简体中文，简洁分点作答。
4. 不要标注来源编号。"""


def build_context_block(items):
    """把检索结果拼成给模型看的资料块。"""
    parts = []
    for n, it in enumerate(items, 1):
        head = it.get("title", "")
        sec = it.get("section", "")
        if sec and sec != head:
            head = "%s > %s" % (head, sec)
        parts.append("[%d] 《%s》\n%s" % (n, head, it.get("text", "").strip()))
    return "\n\n".join(parts)


def build_messages(question, items, system_prompt=None):
    user = (
        "资料片段：\n\n%s\n\n"
        "----\n"
        "请根据以上资料回答问题：%s" % (build_context_block(items), question)
    )
    return [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _headers():
    return {
        "Authorization": "Bearer " + config.LLM_API_KEY,
        "Content-Type": "application/json",
    }


class Generator(object):

    def __init__(self, base_url=None, api_key=None, model=None,
                 system_prompt=None):
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or config.LLM_API_KEY
        self.model = model or config.LLM_MODEL
        # 换 system prompt 只为了做「接地约束」的对照实验（见 eval/run_ragas.py），
        # 线上永远用默认的 SYSTEM_PROMPT
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self._client = httpx.Client(
            base_url=self.base_url, headers=_headers(), timeout=config.LLM_TIMEOUT)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _payload(self, messages, stream):
        return {
            "model": self.model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
            "stream": stream,
        }

    def answer(self, question, items):
        """非流式，返回完整回答文本（评测用）。"""
        if not items:
            return REFUSAL_MESSAGE
        r = self._client.post(
            "/chat/completions",
            json=self._payload(build_messages(question, items, self.system_prompt),
                               False))
        if r.status_code != 200:
            raise RuntimeError("生成失败 HTTP %d: %s" % (r.status_code, r.text[:300]))
        return r.json()["choices"][0]["message"]["content"].strip()

    async def astream(self, question, items):
        """异步流式，逐段产出文本。"""
        if not items:
            yield REFUSAL_MESSAGE
            return

        payload = self._payload(
            build_messages(question, items, self.system_prompt), True)
        async with httpx.AsyncClient(
                base_url=self.base_url, headers=_headers(),
                timeout=config.LLM_TIMEOUT) as client:
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise RuntimeError("生成失败 HTTP %d: %s"
                                       % (resp.status_code, body[:300]))
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {})
                    except (ValueError, KeyError, IndexError):
                        continue
                    piece = delta.get("content")
                    if piece:
                        yield piece


def main():
    """命令行问答，方便不启动服务时调试整条链路。"""
    import argparse
    from rag.retriever import HybridRetriever
    from rag.rerank import Reranker, should_refuse

    ap = argparse.ArgumentParser(description="命令行 RAG 问答")
    ap.add_argument("question")
    ap.add_argument("--top", type=int, default=config.TOP_N_RERANK)
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    with HybridRetriever() as r:
        cands = r.search(args.question)
        with Reranker() as rk:
            scored = rk.rerank(args.question, cands, top_n=args.top)

        if should_refuse(scored):
            print(REFUSAL_MESSAGE)
            return 0

        items = [it for it, _ in scored]
        if args.show_context:
            for n, (it, s) in enumerate(scored, 1):
                print("--- [%d] %s > %s (score=%s)"
                      % (n, it["title"], it["section"], s))

        with Generator() as g:
            print(g.answer(args.question, items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
