"""端到端评测：真机测试 → 数据采集 → 评测 → HTML 报告

一键跑完 collect → evaluate → report 完整流程。

用法:
    python scripts/run_eval.py
    python scripts/run_eval.py --yaml examples/toutiao_comment.yaml
    python scripts/run_eval.py --manifest tests/fixtures/eval_data/20260412_224610/eval_manifest.json

前置条件:
    1. 连接 Android 设备（adb devices 可见）— 跳过采集时不需要
    2. .env 配置好 API Key
"""

import argparse
import logging
import os
import sys

# 确保 auto_qa 在 Python path 中
_script_dir = os.path.dirname(os.path.abspath(__file__))
_auto_qa_dir = os.path.dirname(_script_dir)
if _auto_qa_dir not in sys.path:
    sys.path.insert(0, _auto_qa_dir)
os.chdir(_auto_qa_dir)


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(name)-25s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.WARNING,
    )
    for topic in ("autoqa:executor", "autoqa:optimizer"):
        logging.getLogger(topic).setLevel(logging.INFO)
    for module in ("executor", "runner", "asserter", "planner", "screenshot", "config", "device", "cache", "monitor"):
        logging.getLogger(module).setLevel(logging.WARNING)


def collect(yaml_path: str, output_dir: str):
    """阶段 1: 真机运行测试 + 采集数据"""
    from dotenv import load_dotenv
    load_dotenv()

    from planner import parse_yaml
    from config.loader import load_global_config
    from main import _build_components
    from eval.collector import ScreenshotCollector

    print(f"\n{'='*60}")
    print(f"  阶段 1: 采集测试数据")
    print(f"  用例: {yaml_path}")
    print(f"{'='*60}\n")

    suite, device_config, model_config, vlm_config, llm_config = parse_yaml(yaml_path)
    _, _, _, _, cache_config = load_global_config()

    runner, executor, device, monitor = _build_components(
        device_config, model_config, vlm_config, llm_config,
    )

    collector = ScreenshotCollector(
        runner=runner,
        output_dir=output_dir,
        yaml_path=yaml_path,
    )

    manifest = collector.collect(suite)

    print(f"\n  采集完成: {manifest.run_id}")
    for case in manifest.cases:
        action_count = sum(1 for s in case.steps if hasattr(s, 'instruction'))
        assert_count = sum(1 for s in case.steps if hasattr(s, 'expectation'))
        print(f"    {case.case_name}: {action_count} actions, {assert_count} asserts, status={case.status}")

    manifest_path = os.path.join(output_dir, manifest.run_id, "eval_manifest.json")
    return manifest_path


def evaluate(manifest_path: str, report_path: str):
    """阶段 2: 评测 + 生成报告"""
    from dotenv import load_dotenv
    load_dotenv()

    from config.loader import load_global_config
    from eval.models import EvalManifest
    from eval.pipeline import EvalPipeline
    from eval.reporter import Reporter
    from providers import create_provider

    print(f"\n{'='*60}")
    print(f"  阶段 2: 评测")
    print(f"  数据: {manifest_path}")
    print(f"{'='*60}\n")

    _, _, vlm_config, _, _ = load_global_config()
    provider = create_provider(
        provider=vlm_config.provider,
        api_key=vlm_config.api_key,
        model=vlm_config.model,
        base_url=vlm_config.base_url,
    )

    manifest = EvalManifest.load(manifest_path)
    pipeline = EvalPipeline(provider)
    result = pipeline.run(manifest)

    # 生成报告
    reporter = Reporter()
    reporter.generate(result, report_path)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"  评测完成!")
    print(f"  可靠性评分: {result['reliability_score']}")
    a = result["asserter_eval"]["summary"]
    print(f"  断言: {a['total_assertions']} 个")
    for k in ("confirmed_pass", "confirmed_fail", "false_pass", "false_fail", "ambiguous"):
        if a[k] > 0:
            print(f"    {k}: {a[k]}")
    e = result["executor_eval"]["summary"]
    print(f"  执行: {e['action_correct']}/{e['total_actions']} 操作正确, {e['state_correct']}/{e['total_actions']} 状态正确")
    t = result["token_usage"]
    total_tokens = sum(v["prompt"] + v["completion"] for v in t.values())
    print(f"  Token 总量: {total_tokens}")
    print(f"\n  报告: {report_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="AutoQA 端到端评测")
    parser.add_argument("--yaml", default="examples/toutiao_task.yaml", help="YAML 测试用例路径")
    parser.add_argument("--manifest", default=None, help="已有的 eval_manifest.json 路径（跳过采集，直接评测）")
    parser.add_argument("--output", default="eval_data", help="输出目录")
    args = parser.parse_args()

    setup_logging()

    if args.manifest:
        # 跳过采集，直接评测已有数据
        manifest_path = args.manifest
        report_path = os.path.join(os.path.dirname(manifest_path), "eval_report.html")
    else:
        # 完整流程：采集 + 评测
        try:
            manifest_path = collect(args.yaml, args.output)
        except Exception as e:
            print(f"\n[ERROR] 采集失败: {e}")
            sys.exit(1)
        report_path = os.path.join(os.path.dirname(manifest_path), "eval_report.html")

    try:
        evaluate(manifest_path, report_path)
    except Exception as e:
        print(f"\n[ERROR] 评测失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()