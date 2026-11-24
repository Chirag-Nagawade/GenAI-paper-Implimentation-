"""
Build hc3.csv (text,label,group) from the HC3 dataset so it can be used as extra training data.

    python -m ai_detect.prepare_hc3                    # downloads 4 small domains, writes hc3.csv
    python -m ai_detect.prepare_hc3 --domains open_qa finance medicine wiki_csai reddit_eli5
    python -m ai_detect.prepare_hc3 --local path\\to\\finance.jsonl path\\to\\medicine.jsonl

Then train:
    python -c "from ai_detect.train import train_ai_detector as t; t(extra_csv=['hc3.csv'])"
"""
import argparse
import csv
import json
import os
import random
import urllib.request

BASE = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/"
DEFAULT_DOMAINS = ["open_qa", "finance", "medicine", "wiki_csai"]   # reddit_eli5 is 55 MB: opt-in


def _download(domain: str, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{domain}.jsonl")
    if not os.path.isfile(path):
        print(f"Downloading {domain}.jsonl ...")
        urllib.request.urlretrieve(BASE + f"{domain}.jsonl", path)
    return path


def _read(path: str, tag: str):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            gid = f"{tag}-{i}"          # human + ChatGPT answers to one question share a group
            for a in d.get("human_answers", []) or []:
                if isinstance(a, str) and len(a.split()) >= 20:
                    rows.append((a.strip(), "human", gid))
            for a in d.get("chatgpt_answers", []) or []:
                if isinstance(a, str) and len(a.split()) >= 20:
                    rows.append((a.strip(), "ai", gid))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS)
    ap.add_argument("--local", nargs="*", help="use already-downloaded .jsonl files instead")
    ap.add_argument("--per_class", type=int, default=1500, help="max rows per class")
    ap.add_argument("--out", default="hc3.csv")
    ap.add_argument("--raw_dir", default="hc3_raw")
    a = ap.parse_args()

    files = [(os.path.splitext(os.path.basename(p))[0], p) for p in a.local] if a.local else \
            [(d, _download(d, a.raw_dir)) for d in a.domains]
    rows = []
    for tag, p in files:
        r = _read(p, tag)
        print(f"{tag}: {len(r)} answers")
        rows += r

    random.seed(42)
    by = {"human": [r for r in rows if r[1] == "human"], "ai": [r for r in rows if r[1] == "ai"]}
    n = min(a.per_class, len(by["human"]), len(by["ai"]))
    if n == 0:
        raise SystemExit("No usable rows found - check the input files.")
    sample = random.sample(by["human"], n) + random.sample(by["ai"], n)   # balanced classes
    random.shuffle(sample)

    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "group"])
        w.writerows(sample)
    print(f"Wrote {a.out}: {len(sample)} rows ({n} human, {n} ai)")


if __name__ == "__main__":
    main()
