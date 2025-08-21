import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict


def typeof(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def summarize_jsonl(path: Path, sample_size: int, show_examples: int, truncate: int) -> None:
    key_counts: Counter = Counter()
    key_type_counts: Dict[str, Counter] = defaultdict(Counter)
    key_examples: Dict[str, list] = defaultdict(list)
    total_lines = 0
    errors = 0

    def maybe_store_example(key: str, value: Any):
        if len(key_examples[key]) < show_examples:
            v = value
            if isinstance(v, str) and truncate and len(v) > truncate:
                v = v[:truncate] + "..."
            key_examples[key].append(v)

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total_lines += 1
            if sample_size and i >= sample_size:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            if not isinstance(obj, dict):
                errors += 1
                continue
            for k, v in obj.items():
                key_counts[k] += 1
                t = typeof(v)
                key_type_counts[k][t] += 1
                maybe_store_example(k, v)

    print(f"File: {path}")
    print(f"Sampled lines: {min(sample_size, total_lines) if sample_size else total_lines}")
    print(f"Parse errors: {errors}")
    print("\nSchema (keys, presence, types):")
    for k in sorted(key_counts.keys()):
        presence = key_counts[k]
        types_str = ", ".join(f"{t}:{c}" for t, c in key_type_counts[k].most_common())
        print(f"- {k}: present in {presence} lines; types -> {types_str}")
        if key_examples[k]:
            for ex in key_examples[k]:
                ex_preview = ex
                if isinstance(ex_preview, (dict, list)):
                    ex_preview = json.dumps(ex_preview)[:truncate] + ("..." if truncate else "")
                print(f"  example: {ex_preview}")


def print_examples(path: Path, n: int, truncate: int) -> None:
    if n <= 0:
        return
    print("\nSample entries:")
    shown = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if shown >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            shown += 1
            pretty = json.dumps(obj, ensure_ascii=False, indent=2)
            if truncate and len(pretty) > truncate:
                pretty = pretty[:truncate] + "..."
            print(f"--- example {shown} ---")
            print(pretty)


def main():
    parser = argparse.ArgumentParser(description="Inspect JSONL structure: keys, types, and examples")
    parser.add_argument("path", type=str, help="Path to .jsonl file (e.g., datasets_downloads/prompts_testing_t1.jsonl)")
    parser.add_argument("--samples", type=int, default=200, help="Number of lines to sample for schema summary")
    parser.add_argument("--examples", type=int, default=2, help="Number of full entries to print after summary")
    parser.add_argument("--truncate", type=int, default=400, help="Max characters for long fields/examples (0 = no truncate)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    summarize_jsonl(path, sample_size=args.samples, show_examples=1, truncate=args.truncate)
    print_examples(path, n=args.examples, truncate=args.truncate)


if __name__ == "__main__":
    main()



