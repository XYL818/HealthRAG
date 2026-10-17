# -*- coding: utf-8 -*-
"""从中文维基百科采集**生活与健康**领域语料，产出 corpus.jsonl。

设计要点
--------
1. **只用标准库**（urllib / html.parser / json），因此可以在任何一台
   裸 Python 3.6 机器上直接跑，不需要 pip install 任何东西。
2. **输出简体中文**。zh.wikipedia 的规范标题是繁体，简体标题多为重定向，
   所以先用搜索 API 拿到规范标题，再用 ``action=parse&variant=zh-cn``
   取正文，这样章节名和正文都是简体。
3. **保留章节结构**。正文按 ``<h2>/<h3>/<h4>`` 转成 ``## / ### / ####``
   标记，供下游做结构感知切分（见 indexing/chunk.py）。
4. **对维基友好**。带可识别 User-Agent、请求间隔 1.2s、遇 429/403 指数退避。
   请不要把 --delay 调得太小。

用法
----
    python3 data/crawl/fetch_wiki.py --out data/corpus.jsonl
    python3 data/crawl/fetch_wiki.py --limit 20 --out /tmp/sample.jsonl   # 试跑
"""

from __future__ import print_function

import argparse
import html.parser
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://zh.wikipedia.org/w/api.php"
USER_AGENT = (
    "HealthRAG-CorpusBot/0.1 "
    "(https://github.com/Null-Phnix/HealthRAG; contact: 3455812936@qq.com)"
)

# ---------------------------------------------------------------- 采集范围

# 类目树根节点，递归收集子类目下的条目。
# zh.wikipedia 的规范类目名多为繁体，简繁两版都列上，找不到的会被自动跳过。
#
# 注意：**不要往里加「食品」「飲食」「生活方式」这类宽泛根**。实测「食品」往下
# 会一路钻到「各地食品 / 節日食品 / 肉類 / 食品擬人化」，收上来的全是饮食文化条目，
# 和健康无关，还会把「食品小作品」这种 499 条的倾倒场整个吃进来。
# 吃的部分交给搜索词（营养素、膳食指南、各类食物）更准。
CATEGORY_ROOTS = [
    "健康", "健康促進", "公共衛生", "衛生", "營養學",
    "疾病", "医学", "醫學",
    "心理學", "精神衛生",
    "運動", "體適能", "睡眠", "睡眠障礙",
    "急救", "藥物", "疫苗", "傳染病", "慢性病",
    "人體", "解剖學",
]

# 不往下递归的子分类名。命中即跳过，避免类目树向饮食文化、维基维护类扩散。
SUBCAT_BLACKLIST = re.compile(
    r"(小作品|列表|模板|分類|索引|消歧義|消歧义|各地|各族|各國|各国|各年|節日|节日"
    r"|擬人|维基|維基|文化|历史|歷史|人物|公司|組織|组织|獎|奖|媒體|媒体|事件"
    r"|年表|時間軸|時間線|作品|虚构|虛構"
    # 职业/从业者类目：深度一放开就会一路钻到"健康教育工作者""医学学者"这类
    r"|工作者|學者|学家|學家|醫師|医生|医生|護理師|护士|藥師|技师|技師|研究員"
    r"|教育家|企业家|企業家|政治人物|傳記|传记)"
)

MAX_CAT_MEMBERS = 400   # 单类目条目数超过此值，视为倾倒场/跨领域大类，整类丢弃

# 搜索词，用于补足类目树覆盖不到的条目（搜索 API 接受简体）
SEARCH_TERMS = [
    # 营养素与膳食
    "蛋白质", "碳水化合物", "脂肪", "膳食纤维", "维生素", "维生素C", "维生素D",
    "矿物质", "钙营养", "铁缺乏症", "锌缺乏症", "叶酸", "钠", "钾", "热量", "卡路里",
    "基础代谢率", "身体质量指数", "营养素密度", "反式脂肪", "添加糖",
    "食品添加剂", "膳食补充剂", "益生菌", "膳食指南", "地中海饮食",
    "得舒饮食", "生酮饮食", "间歇性断食", "素食主义", "升糖指数", "食物过敏",
    "乳糖不耐症", "麸质", "饮用水", "膳食纤维", "胆固醇",
    # 常见疾病与代谢
    "高血压", "糖尿病", "2型糖尿病", "高脂血症", "肥胖", "代谢综合征",
    "冠心病", "脑卒中", "动脉粥样硬化", "脂肪肝", "痛风", "贫血",
    "缺铁性贫血", "甲状腺功能减退", "甲亢", "骨质疏松", "关节炎",
    "胃食管反流", "胃炎", "消化性溃疡", "幽门螺杆菌", "肠易激综合征",
    "便秘", "痔疮", "过敏性鼻炎", "哮喘", "慢性阻塞性肺病", "肺炎",
    "流感", "普通感冒", "带状疱疹", "颈椎病", "腰椎间盘突出", "近视",
    "白内障", "牙周炎", "龋齿", "痤疮", "湿疹", "银屑病", "荨麻疹",
    # 心理健康
    "抑郁症", "焦虑症", "双相情感障碍", "强迫症", "创伤后应激障碍",
    "失眠", "睡眠呼吸暂停", "嗜睡症", "倦怠", "孤独", "正念", "冥想",
    "认知行为疗法", "压力管理", "情绪调节", "心理韧性",
    # 运动与体适能
    "有氧运动", "无氧运动", "力量训练", "柔韧性", "核心肌群", "热身",
    "肌肉酸痛", "运动损伤", "跑步", "游泳", "瑜伽", "久坐", "步数",
    "最大心率", "体力活动",
    # 生活方式与预防
    "吸烟", "二手烟", "饮酒", "酒精", "咖啡因", "熬夜", "昼夜节律",
    "睡眠卫生", "屏幕时间", "久坐行为", "防晒", "紫外线", "洗手",
    "口腔卫生", "健康体检", "癌症筛查", "疫苗接种", "健康预期寿命",
    # 急救与安全
    "心肺复苏", "自动体外除颤器", "海姆立克急救法", "中暑", "热射病",
    "烧伤", "骨折", "止血", "溺水", "电击伤", "一氧化碳中毒",
    "食物中毒", "农药中毒", "蛇咬伤", "狂犬病", "破伤风", "异物窒息",
    # 特殊人群
    "孕期营养", "妊娠", "母乳喂养", "婴幼儿喂养", "辅食", "儿童生长发育",
    "生长曲线", "青春期", "更年期", "老年人营养", "肌肉减少症", "跌倒预防",
    # 医疗与循证
    "循证医学", "临床试验", "安慰剂效应", "随机对照试验", "系统综述",
    "抗生素", "抗生素耐药性", "非处方药", "处方药", "药物不良反应",
    "家庭医生", "分级诊疗", "基本医疗保险", "公共卫生", "流行病学",
    "疾病预防控制", "健康素养", "中医", "针灸",
]

# 明显不属于语料范围的标题（列表、消歧义、小作品等）
TITLE_BLACKLIST = re.compile(
    r"(列表|小作品|消歧义|模板|分类|索引|时间线|年表|Category:|Wikipedia:|Template:)"
)

# 年份/日期开头的条目基本都是事件类（「2012年某某案」「1854年宽街霍乱爆发」），
# 是新闻不是知识，检索时只会干扰。
TITLE_JUNK_PREFIX = re.compile(r"^(\d{3,4}\s*年|\d{1,2}\s*月\s*\d{1,2}\s*日)")

# 纯数字/符号标题（搜索「20」「100」这类词会带回这种条目）
TITLE_PURE_NUM = re.compile(r"^[\d\s\W]+$")

# 消歧义页的标志词。**只在正文末尾找**，不要在开头找 ——
# 消歧义页的「这是一个消歧义页…」是模板固定在页面底部的；而页面**开头**
# 常见的是 hatnote（「此条目介绍的是吸食烟草。关于所有吸食行为，请见…」
# 「关于与「睡眠」标题相近或相同的条目，请见「睡眠 (消歧义)」」）。
# 在开头匹配「可以指／消歧义」会误杀真正的条目：实测 29KB 的「吸烟」、
# 19KB 的「睡眠」、「体育运动」、「食盐」就是这么整篇被丢掉的。
DISAMBIG_MARK = re.compile(r"(消歧義|消歧义)")

MIN_CHARS = 1200        # 正文短于此长度视为无效条目
MAX_CHARS = 60000       # 过长的条目截断，避免单条撑爆检索粒度


# ---------------------------------------------------------------- HTTP

def api_get(params, retries=5, delay=1.2):
    """调 MediaWiki API，带节流与退避。"""
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("formatversion", "1")
    url = API + "?" + urllib.parse.urlencode(params)

    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            body = urllib.request.urlopen(req, timeout=30).read()
            time.sleep(delay)
            data = json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            # 429/403 是限流，退避重试；404 之类直接抛
            if e.code in (429, 403, 503):
                backoff = delay * (2 ** attempt) + 2
                print("    [限流 %d] 等待 %.1fs 后重试" % (e.code, backoff),
                      file=sys.stderr)
                time.sleep(backoff)
                continue
            raise
        except Exception as e:              # 网络抖动
            last_err = e
            time.sleep(delay * (2 ** attempt))
            continue
        # 限流有时是以 HTTP 200 + error 字段返回的。不当成错误的话，
        # 调用方看到的就是「没有 parse 字段」，会误判成「条目不存在」而静默跳过 ——
        # 语料就这么悄悄缺一块，事后完全看不出来（实测因此丢了 14 个核心条目）。
        err = data.get("error")
        if err:
            code = err.get("code", "")
            last_err = RuntimeError("API 错误 %s: %s" % (code, err.get("info", "")))
            if code in ("ratelimited", "maxlag", "readonly",
                        "internal_api_error_DBConnectionError"):
                backoff = delay * (2 ** attempt) + 3
                print("    [限流 %s] 等待 %.1fs 后重试" % (code, backoff),
                      file=sys.stderr)
                time.sleep(backoff)
                continue
            raise last_err
        return data
    raise RuntimeError("API 请求失败: %s" % last_err)


# ---------------------------------------------------------------- HTML → 文本

class _Extractor(html.parser.HTMLParser):
    """把维基正文 HTML 转成带章节标记的纯文本。

    只做语料需要的事：保留标题层级、段落、列表项；
    丢弃 style/script、引用角标、编辑链接、表格与信息框。
    """

    SKIP_TAGS = {"style", "script", "sup", "table", "figure", "figcaption"}

    def __init__(self):
        html.parser.HTMLParser.__init__(self)
        self.parts = []
        self.skip_depth = 0
        self.heading = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in ("h2", "h3", "h4"):
            self.heading = tag
        elif tag == "p":
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "tr":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in ("h2", "h3", "h4"):
            self.parts.append("\n")
            self.heading = None
        elif tag == "p":
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.heading:
            level = {"h2": "## ", "h3": "### ", "h4": "#### "}[self.heading]
            self.parts.append("\n\n" + level + data.strip())
        else:
            self.parts.append(data)

    def text(self):
        raw = "".join(self.parts)
        raw = raw.replace("\xa0", " ")
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r" *\n *", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html_str):
    p = _Extractor()
    p.feed(html_str)
    p.close()
    return p.text()


# ---------------------------------------------------------------- 标题收集

def collect_categories(roots, max_depth, delay):
    """递归收集类目树下的条目标题。"""
    seen_cats = set()
    titles = set()
    frontier = [(c, 0) for c in roots]

    while frontier:
        cat, depth = frontier.pop(0)
        if cat in seen_cats or depth > max_depth:
            continue
        seen_cats.add(cat)

        try:
            r = api_get({"action": "query", "list": "categorymembers",
                         "cmtitle": "Category:" + cat, "cmlimit": "500"}, delay=delay)
        except Exception as e:
            print("  [跳过类目 %s] %s" % (cat, e), file=sys.stderr)
            continue

        members = r.get("query", {}).get("categorymembers", [])
        if len(members) > MAX_CAT_MEMBERS:
            print("  类目 %-22s 条目 %-4d 过大，整类跳过" % (cat, len(members)))
            continue

        n_art = n_sub = 0
        for m in members:
            if m.get("ns") == 0:
                titles.add(m["title"])
                n_art += 1
            elif m.get("ns") == 14 and depth < max_depth:
                sub = m["title"].split(":", 1)[-1]
                if SUBCAT_BLACKLIST.search(sub):
                    continue
                frontier.append((sub, depth + 1))
                n_sub += 1
        print("  类目 %-22s 条目 %-4d 子类 %d" % (cat, n_art, n_sub))

    return titles


def collect_search(terms, per_term, delay):
    """用搜索 API 补条目（顺便帮我们把简体词映射到规范标题）。"""
    titles = set()
    for t in terms:
        try:
            r = api_get({"action": "query", "list": "search", "srsearch": t,
                         "srlimit": str(per_term), "srnamespace": "0"}, delay=delay)
        except Exception as e:
            print("  [跳过搜索词 %s] %s" % (t, e), file=sys.stderr)
            continue
        hits = [s["title"] for s in r.get("query", {}).get("search", [])]
        titles.update(hits)
        print("  搜索 %-16s → %d 条" % (t, len(hits)))
    return titles


# ---------------------------------------------------------------- 抓正文

def simplified_title(displaytitle_html, fallback):
    """从 parse 的 displaytitle 里取简体标题。

    zh.wikipedia 的规范标题是繁体，``converttitles`` 参数不生效，
    但在 ``variant=zh-cn`` 下 ``displaytitle`` 里会带一个
    ``<span class="mw-page-title-main">高血压</span>``，剥掉标签就是简体标题。
    这样不必为了转个标题名引入 opencc 之类的依赖。
    """
    if displaytitle_html:
        txt = re.sub(r"<[^>]+>", "", displaytitle_html).strip()
        if txt:
            return txt
    return fallback


def fetch_article(title, delay, extra_aliases=()):
    """取单篇正文，返回 doc dict 或 None。

    ``redirects=1`` 让 API 自动跟随重定向，所以传「抑郁症」会拿到
    「重性抑郁障碍」的正文。这时把用户实际会说的那个名字记进 ``aliases`` ——
    否则「抑郁症吃什么药」这类提问在检索时对不上标题，
    评测按标题比对也会把它算成未命中。

    ``extra_aliases`` 是调用方额外指定的别名，用于「常用名根本没建重定向」的情况：
    「体力活动」在中文维基上连重定向都没有，正主叫「身体活动」，
    只能靠 ``--titles-file`` 里的 ``体力活动=身體活動`` 这种写法把两者挂上。
    """
    r = api_get({"action": "parse", "page": title, "prop": "text|displaytitle",
                 "variant": "zh-cn", "disableeditsection": "1",
                 "disabletoc": "1", "redirects": "1"}, delay=delay)

    if "parse" not in r:
        return None
    node = r["parse"]
    page_id = node.get("pageid")
    canonical = simplified_title(node.get("displaytitle"), node.get("title", title))

    text = html_to_text(node["text"]["*"])
    if len(text) < MIN_CHARS or DISAMBIG_MARK.search(text[-300:]):
        return None
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    aliases = []
    for a in [title.strip()] + [x.strip() for x in extra_aliases]:
        if a and a != canonical and a not in aliases:
            aliases.append(a)

    return {
        "doc_id": "wiki-%s" % page_id,
        "title": canonical,
        "aliases": aliases,
        "url": "https://zh.wikipedia.org/wiki/" + urllib.parse.quote(canonical),
        "source": "zh.wikipedia.org",
        "license": "CC BY-SA 4.0",
        "chars": len(text),
        "text": text,
    }


def backfill_aliases(path, hints):
    """把 hints（{实际标题: [常用名, ...]}）补进已存在的语料记录。

    返回改动的篇数。逐行读、逐行写临时文件，最后整体替换 ——
    语料有几万行，整份读进内存没必要。
    """
    tmp = path + ".tmp"
    n = 0
    with open(path, "r", encoding="utf-8") as fin, \
            open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            want = hints.get(doc.get("title"))
            if want:
                cur = doc.get("aliases") or []
                add = [a for a in want if a and a not in cur]
                if add:
                    doc["aliases"] = cur + add
                    n += 1
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return n


def main():
    ap = argparse.ArgumentParser(description="采集中文维基生活健康领域语料")
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="最多抓多少篇（0=不限）")
    ap.add_argument("--delay", type=float, default=1.2, help="请求间隔秒数")
    ap.add_argument("--max-depth", type=int, default=1,
                    help="类目递归深度。默认 1（根类目 + 其一级子类目的条目）。"
                         "调到 2 会一路钻到「健康教育工作者」「医学学者」这类职业分类，"
                         "语料质量明显变差，不建议。")
    ap.add_argument("--search-limit", type=int, default=8, help="每个搜索词取几条")
    ap.add_argument("--titles-file", default="",
                    help="额外按标题补采（一行一个）。搜索 API 返回的常是"
                         "「兄弟条目」而不是正主（搜「抑郁症」拿到的是产后抑郁症、"
                         "季节性抑郁症），核心条目要靠这个文件点名补上。")
    ap.add_argument("--with-categories", action="store_true",
                    help="额外用类目树补条目。**默认关闭**：实测类目树会带进大量"
                         "无关条目（人物、事件、法令、赛事），信噪比远低于搜索词。")
    ap.add_argument("--no-search", action="store_true", help="只用类目树（不推荐）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只收集标题并打印，不抓正文（用来先检查语料范围对不对）")
    args = ap.parse_args()

    print("=== 1/3 收集标题 ===")
    print("类目递归深度 = %d，搜索词 %d 个" % (args.max_depth, len(SEARCH_TERMS)))
    titles = set()
    if args.with_categories:
        print("[类目树]")
        titles |= collect_categories(CATEGORY_ROOTS, args.max_depth, args.delay)
    if not args.no_search:
        print("[搜索词]")
        titles |= collect_search(SEARCH_TERMS, args.search_limit, args.delay)
    alias_hints = {}
    explicit = set()          # 用户在 titles 文件里点名的标题，绕过后面的启发式过滤
    if args.titles_file:
        print("[指定标题]")
        with open(args.titles_file, "r", encoding="utf-8") as f:
            extra = [ln.strip() for ln in f
                     if ln.strip() and not ln.startswith("#")]
        # 支持「常用名=实际标题」：常用名在中文维基上可能连重定向都没有
        # （「体力活动」→「身体活动」、「自动体外除颤器」→「自动体外去颤器」），
        # 这时只能点名实际标题，同时把常用名记成别名
        n_map = 0
        for ln in extra:
            if "=" in ln:
                shown, real = ln.split("=", 1)
                shown, real = shown.strip(), real.strip()
                if shown and real:
                    titles.add(real)
                    explicit.add(real)
                    alias_hints.setdefault(real, []).append(shown)
                    n_map += 1
                    continue
            titles.add(ln)
            explicit.add(ln)
        print("  %d 行 ← %s（其中 %d 行是「常用名=实际标题」）"
              % (len(extra), args.titles_file, n_map))

    # 显式点名的标题不受启发式过滤约束 —— 那些规则是为搜索 API 带回来的
    # 上千个标题做粗筛用的，误伤代价很低；但点名的是几十个，误伤就是真的缺条目。
    # 实测「钠」（单字，被 len>=2 拦掉）就是这么漏的。
    # 消歧义页不怕放过：fetch_article 会看正文末尾的标记把它挡掉。
    titles = sorted(t for t in titles
                    if t in explicit or (
                        len(t) >= 2
                        and not TITLE_BLACKLIST.search(t)
                        and not TITLE_JUNK_PREFIX.match(t)
                        and not TITLE_PURE_NUM.match(t)))
    print("候选标题合计: %d" % len(titles))

    if args.dry_run:
        print()
        print("=== dry-run：候选标题如下 ===")
        for t in titles:
            print("  " + t)
        return

    if args.limit:
        titles = titles[:args.limit]

    print()
    print("=== 2/3 抓取正文（约 %.1f 分钟）===" % (len(titles) * args.delay / 60.0))

    # 断点续抓：已存在于 --out 的 doc 不再重复请求
    done = set()
    n_done = 0
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                except Exception:
                    continue
                # 别名也算「已抓过」。否则「抑郁症」每次都会重抓一遍 ——
                # 它拿回来的正文标题是规范名「重性抑郁障碍」，只记 title
                # 的话下次仍然认为它没抓过
                done.add(doc["title"])
                done.update(doc.get("aliases") or [])
                n_done += 1
        if n_done:
            # 按篇数报，不报 len(done)：别名也进了 done，那样会多报
            print("已存在 %d 篇，跳过" % n_done)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    kept = skipped = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i, t in enumerate(titles, 1):
            if t in done:
                continue
            try:
                doc = fetch_article(t, args.delay, alias_hints.get(t, ()))
            except Exception as e:
                print("  [%d/%d] %-24s 失败: %s" % (i, len(titles), t, e),
                      file=sys.stderr)
                skipped += 1
                continue

            if doc is None:
                # 一定要出声。这条分支以前是静默的，结果限流被当成
                # 「条目不存在」，一次就悄悄少了 14 个核心条目，
                # 事后翻日志什么都看不出来。
                print("  [%d/%d] %-24s 跳过（条目不存在/过短/消歧义页）"
                      % (i, len(titles), t), file=sys.stderr)
                skipped += 1
                continue

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            f.flush()
            kept += 1
            print("  [%d/%d] %-24s %6d 字" % (i, len(titles), doc["title"], doc["chars"]))

    # ---------------------------------------------------------- 别名回填
    # 「常用名=实际标题」里那个实际标题如果**早就在语料里**，上面会整篇跳过，
    # 别名于是永远补不上 —— 而别名恰恰是这条映射存在的唯一理由。
    # 非做不可：不然评测会把本来检索得到的问题一直报成未命中。
    if alias_hints:
        n_patch = backfill_aliases(args.out, alias_hints)
        if n_patch:
            print("别名回填 %d 篇 → %s" % (n_patch, args.out))

    print()
    print("=== 3/3 完成 ===")
    print("入库 %d 篇，跳过 %d 篇 → %s" % (kept, skipped, args.out))


if __name__ == "__main__":
    main()
