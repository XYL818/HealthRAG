# -*- coding: utf-8 -*-
"""结构感知切分：把 corpus.jsonl 拆成 chunks.jsonl。

为什么不用"每 N 个字切一刀"
--------------------------
朴素定长切分会把一个小标题和它下面的内容切散，检索时召回半句话，
生成的答案也就断章取义。这里利用爬虫保留的 ``## / ### / ####`` 章节标记：

1. **先按章节切**，每段自带完整的小标题路径（如 ``高血压 > 诊断 > 血压测量``）；
2. **章节内再按长度切**，且只在句子边界断开，保留少量重叠防止答案跨块丢失；
3. **送去做向量的文本带上标题路径**（``embed_text``），
   而返回给用户看的仍是纯正文（``text``）—— 这样"诊断标准是什么"这类
   脱离上下文的提问也能命中。

用法
----
    python -m indexing.chunk --corpus data/corpus.jsonl --out data/chunks.jsonl
"""

from __future__ import print_function

import argparse
import json
import re
import sys

# 章节标记：## / ### / ####
HEADING_RE = re.compile(r"^(#{2,4})\s*(.+?)\s*$")

# 中文句末标点。只在断句处切，避免把 "舒张压90mmHg。" 之类拦腰截断。
# 注意 `\s*` 不能省：去掉它就变成零宽匹配，re.split 在零宽匹配后会跳过一个字符，
# 相邻的两句会被并成一句（实测块数从 14152 掉到 9 千多）。那条 FutureWarning 是噪音，
# 忍了。
SENT_END = re.compile(r"(?<=[。！？；!?;])\s*|\n{2,}")

# 丢弃这些明显的噪声章节
NOISE_HEADING = re.compile(r"^(参见|參見|参考文献|參考文獻|外部链接|外部連結|注释|註釋|"
                           r"延伸阅读|延伸閱讀|资料来源|資料來源|引用|参考|參考)$")

MIN_CHUNK = 80       # 短于此长度的块直接丢弃（低于此值基本只有一句定义，检索价值低）
TARGET = 450         # 目标块长（字符）
OVERLAP = 80         # 相邻块重叠
HARD_MAX = 900       # 单块上限，超了强制切断

# 垃圾行：参考文献残留、纯符号、孤立的外链编号
JUNK_LINE = re.compile(r"^[\s\|⋅·•\-\*\.\d,，、。;；:：()（）\[\]【】]+$")


def split_sentences(text):
    """按中文句末标点断句，保留标点。"""
    parts = [p for p in SENT_END.split(text) if p]
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def pack_sentences(sentences, target=TARGET, overlap=OVERLAP, hard_max=HARD_MAX):
    """把句子打包成 ~target 长度的块，块间留 overlap 个字符的重叠。"""
    if not sentences:
        return []

    chunks = []
    cur = []
    cur_len = 0

    for s in sentences:
        # 单句就超长（表格残留之类）→ 单独硬切
        while len(s) > hard_max:
            if cur:
                chunks.append("".join(cur))
                cur, cur_len = [], 0
            chunks.append(s[:hard_max])
            s = s[hard_max:]

        if cur_len + len(s) > target and cur:
            chunks.append("".join(cur))
            # 用尾部若干字符做重叠，找回被切断的上下文
            tail = "".join(cur)[-overlap:] if overlap else ""
            cur = [tail, s] if tail else [s]
            cur_len = len(tail) + len(s)
        else:
            cur.append(s)
            cur_len += len(s)

    if cur:
        chunks.append("".join(cur))

    return [c.strip() for c in chunks if c.strip()]


def parse_sections(text, root_title=None):
    """把带章节标记的正文解析成 [(标题路径, 正文)] 列表。

    ``root_title`` 会作为层级 1 的根压进路径栈，这样每个小节的路径都是
    完整的「文档 > 二级 > 三级」，而不是断头的「二级 > 三级」——
    脱离上下文提问时（"诊断标准是什么"），完整路径对召回很关键。
    """
    stack = [(1, root_title)] if root_title else []   # [(level, title)]
    buf = []
    sections = []
    skipping = False

    def flush():
        # skipping 为真时，buf 里装的是噪声章节的正文，整段丢掉
        body = "\n".join(buf).strip()
        if body and not skipping:
            path = " > ".join(t for _, t in stack)
            sections.append((path, body))

    for line in text.split("\n"):
        m = HEADING_RE.match(line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            if NOISE_HEADING.match(title):
                # 只保留根节点：噪声章节的层级关系没有意义，
                # 但整栈清空会让后面小节丢掉文档名，路径变成断头的「二级 > 三级」
                stack = stack[:1]
                skipping = True
                continue
            skipping = False
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            if JUNK_LINE.match(line):
                continue
            buf.append(line)

    flush()
    return sections


def clean_body(body):
    """去掉正文里的残留噪声行。"""
    lines = []
    for ln in body.split("\n"):
        ln = ln.strip()
        if not ln or JUNK_LINE.match(ln):
            continue
        lines.append(ln)
    return "\n".join(lines)


def chunk_document(doc, target=TARGET, overlap=OVERLAP):
    """把一篇文档切成若干 chunk。

    ``target`` / ``overlap`` 可覆盖，是为了做"块长对检索效果影响"的消融 ——
    评测脚本要在同一批文档上比不同块长，不能只有一套写死的参数。
    """
    out = []
    aliases = doc.get("aliases") or []
    sections = parse_sections(doc["text"], doc["title"])
    if not sections:
        sections = [(doc["title"], doc["text"])]

    for si, (path, body) in enumerate(sections):
        body = clean_body(body)
        if len(body) < MIN_CHUNK:
            continue

        pieces = pack_sentences(split_sentences(body), target=target,
                                overlap=overlap)
        for ci, piece in enumerate(pieces):
            if len(piece) < MIN_CHUNK:
                continue
            cid = "%s-s%02d-c%02d" % (doc["doc_id"], si, ci)
            # 送向量的文本带上完整标题路径，增强脱离上下文的提问的召回。
            # 别名一并写进路径行：维基的规范标题常常不是日常说法
            # （正主叫「重性抑郁障碍」，没人会这么问），不写进去的话
            # 「抑郁症吃什么药」这类提问在向量通道上就对不上。
            head = "%s（%s）" % (path, "、".join(aliases)) if (
                path and aliases) else path
            embed_text = "%s\n%s" % (head, piece) if head else piece
            out.append({
                "chunk_id": cid,
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "aliases": aliases,
                "url": doc["url"],
                "section": path,
                "text": piece,
                "embed_text": embed_text,
                "chars": len(piece),
            })
    return out


def main():
    ap = argparse.ArgumentParser(description="结构感知切分")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default="data/chunks.jsonl")
    ap.add_argument("--target", type=int, default=TARGET,
                    help="目标块长，默认 %d。消融实验用" % TARGET)
    ap.add_argument("--overlap", type=int, default=OVERLAP,
                    help="相邻块重叠字符数，默认 %d" % OVERLAP)
    args = ap.parse_args()

    if args.target < MIN_CHUNK:
        print("--target 不能小于 MIN_CHUNK=%d" % MIN_CHUNK, file=sys.stderr)
        return 1

    total_docs = total_chunks = 0
    sizes = []
    kept_docs = 0

    with open(args.corpus, "r", encoding="utf-8") as fin, \
            open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            total_docs += 1
            cs = chunk_document(doc, target=args.target, overlap=args.overlap)
            if cs:
                kept_docs += 1
            for c in cs:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                sizes.append(c["chars"])
            total_chunks += len(cs)

    if not sizes:
        print("没有产出任何 chunk，检查语料是否为空", file=sys.stderr)
        return 1

    sizes.sort()
    n = len(sizes)
    print("文档 %d 篇（有效 %d）→ chunk %d 个" % (total_docs, kept_docs, total_chunks))
    print("块长 中位数 %d / 均值 %d / 最小 %d / 最大 %d"
          % (sizes[n // 2], sum(sizes) // n, sizes[0], sizes[-1]))
    print("平均每篇 %.1f 块 → %s" % (total_chunks / max(kept_docs, 1), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
