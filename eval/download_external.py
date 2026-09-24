"""Download the external evaluation corpora into eval/external/ (git-ignored).

    python eval/download_external.py            # all three
    python eval/download_external.py shushan    # or: webnovelbench / detectiveqa

Standard library only. Already downloaded files are skipped, so re-running is cheap.
The texts are for local evaluation only: never commit them, and never use them in prompt
examples or test stories (same rule as eval/private/).

Sources and licenses:
- webnovelbench: huggingface.co/datasets/Oedon42/webnovelbench
  CC BY-NC-SA 4.0 (non-commercial; the novels' own copyright status is unclear)
- detectiveqa: huggingface.co/datasets/Phospheneser/DetectiveQA
  Apache 2.0; the Chinese novels are mostly translations still under copyright
- shushan: zh.wikisource.org 蜀山劍俠傳 by 还珠楼主 (d. 1961)
  public domain in China, not in the US; Wikisource only has 第1–12回
"""

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

EXTERNAL = Path(__file__).resolve().parent / "external"
UA = {"User-Agent": "webfic-eval/0.1 (https://github.com/kkkkkkatherine01/Webfic-Checker)"}
HF = "https://huggingface.co"

SHUSHAN_CHAPTERS = 12  # zh.wikisource only has 第1–12回 (the book has 329)


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _save(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _get(url)
    path.with_suffix(path.suffix + ".part").write_bytes(data)
    path.with_suffix(path.suffix + ".part").replace(path)
    print(f"  {path.relative_to(EXTERNAL)}  {len(data) / 1024:.0f} KB")


def _hf_file(repo: str, path: str) -> str:
    return f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def _hf_tree(repo: str, folder: str) -> list[str]:
    url = f"{HF}/api/datasets/{repo}/tree/main/{urllib.parse.quote(folder)}"
    return [f["path"] for f in json.loads(_get(url)) if f["type"] == "file"]


def webnovelbench() -> None:
    """100 web novels, 10 consecutive full chapters each (the 578 MB full file is not
    needed: it holds the same kind of data for 4000 novels)."""
    print("WebNovelBench")
    repo, path = "Oedon42/webnovelbench", "novel_data/novel_data_subset_d_100.json"
    _save(EXTERNAL / "webnovelbench" / "novel_data_subset_d_100.json", _hf_file(repo, path))


def detectiveqa() -> None:
    """The Chinese novels that have human annotations, plus those annotations. Novels
    are numbered paragraphs ("[12] ..."); annotations point at clue paragraphs."""
    print("DetectiveQA")
    repo = "Phospheneser/DetectiveQA"
    target = EXTERNAL / "detectiveqa"
    annotations = _hf_tree(repo, "anno_data_zh/human_anno")
    novels = {p.split("/")[-1].split("-")[0]: p for p in _hf_tree(repo, "novel_data_zh")}
    for anno in annotations:
        novel_id = Path(anno).stem
        _save(target / "human_anno" / f"{novel_id}.json", _hf_file(repo, anno))
        novel = novels.get(novel_id)
        if novel is None:
            print(f"  ! no novel text for annotation {novel_id}")
            continue
        _save(target / "novel_data_zh" / novel.split("/")[-1], _hf_file(repo, novel))


def _plain(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment))


def _wikisource_chapter(number: int) -> str:
    """One 回 as plain text: its heading line, then one line per paragraph.

    Pages come in two layouts: most carry the heading in a header table and one <p> per
    paragraph; some (e.g. 第011囘) have no table, put the heading in the first <p> and
    separate paragraphs with newlines and full-width indents inside a <p>."""
    page = f"蜀山劍俠傳/第{number:03d}囘"
    query = urllib.parse.urlencode({
        "action": "parse", "page": page, "prop": "text", "variant": "zh-hans",
        "format": "json", "formatversion": 2,
    })  # fmt: skip
    page_html = json.loads(_get(f"https://zh.wikisource.org/w/api.php?{query}"))["parse"]["text"]

    lines = [
        line.strip(" \t　")
        for p in re.findall(r"<p>(.*?)</p>", page_html, flags=re.S)
        for line in _plain(p).splitlines()
    ]
    lines = [line for line in lines if line]
    heading = re.search(
        r'<td style="width:70%; text-align:center">(第.+?回.*?)</td>', page_html, flags=re.S
    )
    if heading is not None:
        title = _plain(heading.group(1))
    elif lines and re.match(r"第.+?回", lines[0]):
        title = lines.pop(0)
    else:
        raise ValueError(f"{page}: heading not found")
    title = " ".join(title.replace("　", " ").split())
    return f"{title}\n" + "\n".join(lines) + "\n"


def shushan() -> None:
    print(f"蜀山剑侠传（第 1–{SHUSHAN_CHAPTERS} 回）")
    folder = EXTERNAL / "shushan"
    chapters = []
    for number in range(1, SHUSHAN_CHAPTERS + 1):
        cached = folder / "chapters" / f"{number:03d}.txt"
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(_wikisource_chapter(number), "utf-8")
            time.sleep(1)  # be polite to Wikimedia: at most one request per second
        chapters.append(cached.read_text("utf-8"))
    path = folder / f"shushan_001-{SHUSHAN_CHAPTERS:03d}.txt"
    text = "\n".join(chapters)
    path.write_text(text, "utf-8")
    print(f"  {path.relative_to(EXTERNAL)}  {len(text)} 字")


CORPORA = {"webnovelbench": webnovelbench, "detectiveqa": detectiveqa, "shushan": shushan}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(CORPORA):
        CORPORA[name]()
