import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


def read_urls(path: Path) -> List[str]:
    """Read URLs from a text file, skipping empty lines."""
    if not path.exists():
        raise FileNotFoundError(f"URL 文件不存在: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def parse_env(env_items: List[str]) -> Dict[str, str]:
    """Parse KEY=VALUE strings into a dict."""
    env_map: Dict[str, str] = {}
    for item in env_items:
        if "=" not in item:
            raise ValueError(f"环境变量格式错误，应为 KEY=VALUE，收到: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"环境变量键为空，收到: {item}")
        env_map[k] = v
    return env_map


def main() -> None:
    parser = argparse.ArgumentParser(description="批量调用 main_crawl.py 爬取 url.txt 中的链接")
    parser.add_argument(
        "--url_file", default=Path(__file__).parent.parent / "url.txt", type=Path, help="存放待爬 URL 的 txt 文件路径"
    )
    parser.add_argument(
        "--schema_path", default="cards_schama.json", help="传给 main_crawl.py 的 schema 路径"
    )
    parser.add_argument(
        "--out_dir", default=Path(__file__).parent / "outputs", type=Path, help="输出 jsonl 目录"
    )
    parser.add_argument("--locale", default="en_US", help="传给 main_crawl.py 的 locale")
    parser.add_argument(
        "--no_llm", action="store_true", help="传递给 main_crawl.py 的 --no_llm，默认使用 LLM"
    )
    parser.add_argument(
        "--env",
        action="append",
        default=["OPENAI_API_KEY="],
        help="以 KEY=VALUE 形式追加环境变量，可重复多次传入",
    )
    args = parser.parse_args()

    urls = read_urls(args.url_file)
    if not urls:
        print(f"{args.url_file} 中没有可用链接")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        extra_env = parse_env(args.env)
    except ValueError as exc:
        print(f"环境变量解析失败: {exc}")
        return
    merged_env = os.environ.copy()
    merged_env.update(extra_env)

    for idx, url in enumerate(urls, start=1):
        out_file = args.out_dir / f"cards_{idx}.jsonl"
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "main_crawl.py"),
            "--url",
            url,
            "--schema_path",
            args.schema_path,
            "--out_jsonl",
            str(out_file),
            "--locale",
            args.locale,
        ]
        if args.no_llm:
            cmd.append("--no_llm")

        print(f"[{idx}/{len(urls)}] 抓取 {url} -> {out_file}")
        subprocess.run(cmd, check=True, env=merged_env)

    print("全部抓取完成。")


if __name__ == "__main__":
    main()
