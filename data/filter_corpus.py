# -*- coding: utf-8 -*-
"""语料清洗：先去重，再做域过滤，剔除采集时混进来的无关条目。

为什么需要这一步
----------------
``fetch_wiki.py`` 靠搜索命中来收集条目，而搜索词本身有歧义
（搜"运动"会带回体育人物，搜"压力"会带回物理条目）。一千多篇里混进
几十篇无关条目，平时不影响问答，但**它们可能被检索到并当成引用来源展示**，
在演示时很难看。

所以用最朴素的办法：数一篇条目里健康相关术语的出现情况，
低于阈值的丢掉。术语表刻意选**多字、低歧义**的词，避免误杀。

用法
----
    python data/filter_corpus.py --report          # 只看分布，不改文件
    python data/filter_corpus.py --in-place        # 就地过滤，备份成 .bak
"""

from __future__ import print_function

import argparse
import json
import os
import sys

# 多字词，歧义低。单字词（"病""症"）刻意不用 —— 它们会匹配到"毛病""特征"之类
STRONG_TERMS = [
    "疾病", "症状", "治疗", "疗法", "诊断", "患者", "医学", "医疗", "医生",
    "医师", "健康", "营养", "营养素", "预防", "药物", "用药", "剂量", "疫苗",
    "接种", "病毒", "细菌", "感染", "传染", "免疫", "抗体", "血液", "细胞",
    "手术", "临床", "疗效", "风险因素", "发病率", "死亡率", "病因", "症候群",
    "综合征", "膳食", "饮食", "卡路里", "维生素", "蛋白质", "脂肪", "碳水化合物",
    "运动", "锻炼", "康复", "护理", "卫生", "公共卫生", "心理", "精神",
    "抑郁", "焦虑", "睡眠", "血压", "血糖", "血脂", "激素", "代谢",
    "癌", "肿瘤", "炎症", "过敏", "中毒", "急救", "消毒", "饮食建议",
    "膳食指南", "保健", "体检", "筛查", "慢性病", "传染病", "医师建议",
]

# 两个条件满足其一即保留。只看命中数会误杀**短条目** ——
# 「母乳喂养」全文才一千多字，术语出现次数天然就少，但它显然属于本领域。
# 所以补一个密度条件：命中的术语总次数 / 千字。
MIN_HITS = 5          # 绝对命中数（长条目走这条）
MIN_DENSITY = 1.2     # 每千字命中次数（短条目走这条）


def score_doc(doc):
    """返回 (命中的不同术语数, 每千字命中次数)。"""
    text = doc.get("text", "")
    if not text:
        return 0, 0.0
    hits = 0
    total = 0
    for t in STRONG_TERMS:
        if t in text:
            hits += 1
            total += text.count(t)
    density = total / float(max(len(text), 1)) * 1000.0
    return hits, density


def keep_doc(hits, density):
    return hits >= MIN_HITS or density >= MIN_DENSITY


def main():
    ap = argparse.ArgumentParser(description="语料域过滤")
    ap.add_argument("--corpus", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "corpus.jsonl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--report", action="store_true", help="只报告分布，不写文件")
    args = ap.parse_args()

    docs = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    if not docs:
        print("语料为空", file=sys.stderr)
        return 1

    # 同一条目可能被抓进来两次：搜索命中一次、--titles-file 点名一次
    # （实测「钠」和「钠=鈉」就是这样）。doc_id 相同但分属两行，
    # 不合并的话同一篇文档会在索引里占两份，重复条目会霸占召回列表。
    # 别名取并集，否则后写的那条会丢掉前一条记下的别名。
    seen, uniq = {}, []
    for d in docs:
        key = d.get("doc_id") or d.get("title")
        if key in seen:
            prev = seen[key]
            cur = prev.setdefault("aliases", [])
            for a in d.get("aliases") or []:
                if a not in cur:
                    cur.append(a)
            continue
        seen[key] = d
        uniq.append(d)
    if len(uniq) != len(docs):
        print("合并重复条目 %d 篇" % (len(docs) - len(uniq)))
    docs = uniq

    scored = [(score_doc(d), d) for d in docs]

    print("共 %d 篇" % len(docs))

    print("\n命中术语数分布：")
    hits_list = sorted(h for (h, _), _ in scored)
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 999)]:
        n = sum(1 for h in hits_list if lo <= h < hi)
        print("  %3d~%-3s : %5d 篇 (%4.1f%%)"
              % (lo, "∞" if hi == 999 else hi - 1, n, 100.0 * n / len(docs)))

    print("\n密度分布（每千字命中次数）：")
    dens_list = sorted(round(dn, 1) for (_, dn), _ in scored)
    for lo, hi in [(0, 0.5), (0.5, 1.2), (1.2, 2), (2, 4), (4, 8), (8, 999)]:
        n = sum(1 for x in dens_list if lo <= x < hi)
        print("  %.1f~%-4s : %5d 篇 (%4.1f%%)"
              % (lo, "∞" if hi == 999 else hi, n, 100.0 * n / len(docs)))

    dropped = [(h, dn, d) for (h, dn), d in scored if not keep_doc(h, dn)]
    kept = [(h, dn, d) for (h, dn), d in scored if keep_doc(h, dn)]

    print("\n规则「命中≥%d 或 密度≥%.1f」：保留 %d 篇，丢弃 %d 篇 (%.1f%%)"
          % (MIN_HITS, MIN_DENSITY, len(kept), len(dropped),
             100.0 * len(dropped) / len(docs)))

    if dropped:
        print("\n丢弃样例（最弱的 20 篇，用来人工确认没误杀）：")
        for h, dn, d in sorted(dropped, key=lambda x: (x[0], x[1]))[:20]:
            print("  [命中 %2d 密度 %4.1f] %s" % (h, dn, d.get("title", "")[:60]))

    if args.report:
        return 0

    out = args.out
    if args.in_place:
        out = args.corpus + ".bak"
        os.replace(args.corpus, out)
        out = args.corpus
    elif not out:
        out = os.path.splitext(args.corpus)[0] + ".filtered.jsonl"

    with open(out, "w", encoding="utf-8") as f:
        for _, _, d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("\n已写入 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
