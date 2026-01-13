import argparse
import os
from typing import List, Dict, Any
import json
from Crawl_test01 import ingest_url_to_cards



def load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--schema_path", required=True, help="Your full habit-card-mvp.json")
    parser.add_argument("--out_jsonl", default="cards.jsonl")
    parser.add_argument("--locale", default="en_US")
    parser.add_argument("--no_llm", action="store_true")
    args = parser.parse_args()

    full_schema = load_schema(args.schema_path)

    stats = ingest_url_to_cards(
        url=args.url,
        full_schema=full_schema,
        out_jsonl=args.out_jsonl,
        use_llm_fallback=(not args.no_llm),
        locale=args.locale,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
    


