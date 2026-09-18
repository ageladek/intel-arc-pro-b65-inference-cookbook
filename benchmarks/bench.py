#!/usr/bin/env python3
"""Small synthetic OpenAI-compatible vLLM benchmark; no response text is saved."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def make_payload(model: str, output_tokens: int, prompt_repeat: int) -> dict:
    marker = uuid4().hex
    filler = "alpha " * prompt_repeat
    return {
        "model": model,
        "messages": [{"role": "user", "content": (
            f"{filler}Benchmark marker {marker}. Explain how a hash table works "
            "in detail; keep writing until the response limit."
        )}],
        "temperature": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def request_once(url: str, model: str, output_tokens: int, prompt_repeat: int,
                 api_key: str | None, timeout: int) -> dict:
    payload = make_payload(model, output_tokens, prompt_repeat)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"), headers=headers,
    )
    started = time.perf_counter()
    first_content = None
    last_content = None
    usage = {}
    status = None
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    if (delta.get("content") or delta.get("reasoning_content")
                            or delta.get("reasoning")):
                        now = time.perf_counter()
                        if first_content is None:
                            first_content = now
                        last_content = now
    except HTTPError as error:
        status = error.code
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        pass
    ended = time.perf_counter()
    completion_tokens = usage.get("completion_tokens")
    span = (last_content - first_content
            if first_content is not None and last_content is not None else None)
    return {
        "http_status": status,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "ttft_s": first_content - started if first_content is not None else None,
        "wall_s": ended - started,
        "decode_tok_s": completion_tokens / span if completion_tokens and span else None,
        "e2e_tok_s": completion_tokens / (ended - started) if completion_tokens else None,
    }


def median(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.8-27B-FP8")
    parser.add_argument("--single-runs", type=int, default=3)
    parser.add_argument("--pair-runs", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--prompt-repeat", type=int, default=0,
                        help="synthetic 'alpha' words added before each request")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--json-output", help="optional path for metrics-only JSON")
    args = parser.parse_args()
    if min(args.single_runs, args.pair_runs, args.output_tokens, args.timeout) < 1:
        parser.error("runs, output-tokens, and timeout must be positive")
    if args.prompt_repeat < 0:
        parser.error("prompt-repeat must be nonnegative")
    api_key = os.environ.get("VLLM_API_KEY") or None

    call = lambda: request_once(args.url, args.model, args.output_tokens,
                                args.prompt_repeat, api_key, args.timeout)
    warmup = call()
    singles = [call() for _ in range(args.single_runs)]
    pairs = []
    for _ in range(args.pair_runs):
        pair_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: call(), range(2)))
        pair_wall = time.perf_counter() - pair_start
        tokens = sum(row.get("completion_tokens") or 0 for row in results)
        pairs.append({"wall_s": pair_wall, "requests": results,
                      "aggregate_e2e_tok_s": tokens / pair_wall if tokens else None})

    rows = singles + [item for pair in pairs for item in pair["requests"]]
    summary = {
        "single_decode_median_tok_s": median(singles, "decode_tok_s"),
        "single_ttft_median_s": median(singles, "ttft_s"),
        "pair_aggregate_median_tok_s": median(pairs, "aggregate_e2e_tok_s"),
        "successful_requests": sum(row["http_status"] == 200 for row in rows),
        "total_requests": len(rows),
    }
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model, "output_tokens": args.output_tokens,
        "prompt_repeat": args.prompt_repeat, "warmup": warmup,
        "singles": singles, "pairs": pairs, "summary": summary,
        "metric_definition": "decode tokens / first-to-last content delta; pair sum tokens / pair wall",
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if summary["successful_requests"] == summary["total_requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
