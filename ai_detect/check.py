"""Quick manual test:  python -m ai_detect.check "some text"   or   python -m ai_detect.check --file x.txt"""
import sys
from ai_detect.predict import predict_document

def main():
    args = sys.argv[1:]
    text = open(args[1], encoding="utf-8").read() if args[:1] == ["--file"] else " ".join(args)
    r = predict_document(text)
    s = r["summary"]
    print(f"\nOverall: {s['overall_classification']}  (mean P(AI)={s.get('mean_prob_ai')})")
    print(f"AI {s['ai_percentage']}% | review {s['review_percentage']}% | human {s['human_percentage']}%\n")
    seen = set()
    for x in r["sentences"]:
        if x["paragraph_index"] in seen:
            continue
        seen.add(x["paragraph_index"])
        print(f"[chunk {x['paragraph_index']}] P(AI)={x['prob_ai']}  {x['category']}  {x['warning']}")

if __name__ == "__main__":
    main()
