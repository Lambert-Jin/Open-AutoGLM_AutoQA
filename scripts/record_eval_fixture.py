"""预录评测 fixture 数据

在真机上运行一次 AutoQA 测试，通过 ScreenshotCollector 采集完整数据，
保存到 tests/fixtures/eval_data/ 供后续评测测试使用。

用法:
    cd auto_qa
    python scripts/record_eval_fixture.py

前置条件:
    1. 连接 Android 设备（adb devices 可见）
    2. .env 配置好 API Key
"""

import logging
import os
import shutil
import sys

# 确保 auto_qa 在 Python path 中
_script_dir = os.path.dirname(os.path.abspath(__file__))
_auto_qa_dir = os.path.dirname(_script_dir)
if _auto_qa_dir not in sys.path:
    sys.path.insert(0, _auto_qa_dir)
os.chdir(_auto_qa_dir)

# ── 配置 ──────────────────────────────────────────
YAML_PATH = "examples/toutiao_task.yaml"       # 使用的测试用例
OUTPUT_DIR = "tests/fixtures/eval_data"            # fixture 输出目录
VERBOSE = True                                     # 详细日志
# ─────────────────────────────────────────────────


def main():
    # 设置日志：只显示步骤执行、模型响应、指令优化
    logging.basicConfig(
        format="%(asctime)s %(name)-25s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.WARNING,
    )
    # 只开启需要的 topic logger
    for topic in ("autoqa:executor", "autoqa:optimizer"):
        logging.getLogger(topic).setLevel(logging.INFO)
    # 静默其他所有模块
    for module in ("executor", "runner", "asserter", "planner", "screenshot", "config", "device", "cache", "monitor"):
        logging.getLogger(module).setLevel(logging.WARNING)

    from dotenv import load_dotenv
    load_dotenv()

    from planner import parse_yaml
    from config.loader import load_global_config
    from main import _build_components
    from eval.collector import ScreenshotCollector

    # 1. 解析 YAML + 加载配置
    print(f"\n{'='*60}")
    print(f"  预录评测 Fixture 数据")
    print(f"  用例: {YAML_PATH}")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    suite, device_config, model_config, vlm_config, llm_config = parse_yaml(YAML_PATH)
    _, _, _, _, cache_config = load_global_config()

    # 2. 构建组件
    runner, executor, device, monitor = _build_components(
        device_config, model_config, vlm_config, llm_config,
    )

    # 3. 运行测试 + 采集数据
    collector = ScreenshotCollector(
        runner=runner,
        output_dir=OUTPUT_DIR,
        yaml_path=YAML_PATH,
    )

    try:
        manifest = collector.collect(suite)
    except Exception as e:
        print(f"\n[ERROR] 测试执行失败: {e}")
        sys.exit(1)

    # 4. 打印结果摘要
    print(f"\n{'='*60}")
    print(f"  预录完成!")
    print(f"  run_id: {manifest.run_id}")
    print(f"  用例数: {len(manifest.cases)}")
    for case in manifest.cases:
        action_count = sum(1 for s in case.steps if hasattr(s, 'instruction'))
        assert_count = sum(1 for s in case.steps if hasattr(s, 'expectation'))
        print(f"    {case.case_name}: {action_count} actions, {assert_count} asserts, status={case.status}")
    print(f"\n  数据目录: {os.path.join(OUTPUT_DIR, manifest.run_id)}")
    print(f"  manifest: {os.path.join(OUTPUT_DIR, manifest.run_id, 'eval_manifest.json')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
