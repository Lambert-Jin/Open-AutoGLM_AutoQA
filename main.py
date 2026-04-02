"""AutoQA CLI 入口"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="autoqa",
        description="AutoQA - 基于 VLM 的移动端自动化测试框架",
    )
    subparsers = parser.add_subparsers(dest="command")

    # run 子命令
    run_parser = subparsers.add_parser("run", help="运行 YAML 测试用例")
    run_parser.add_argument("yaml_path", help="YAML 测试用例文件路径")
    run_parser.add_argument("--device-type", default=None, help="设备类型: adb")
    run_parser.add_argument("--device-id", default=None, help="设备 ID")
    run_parser.add_argument("--no-cache", action="store_true", help="禁用 Action 缓存")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="详细日志输出")
    _add_monitor_args(run_parser)

    # generate 子命令
    gen_parser = subparsers.add_parser("generate", help="自然语言生成 YAML 测试用例")
    gen_parser.add_argument("description", help="自然语言测试描述")
    gen_parser.add_argument("-o", "--output", default=None, help="输出 YAML 文件路径（不指定则输出到终端）")
    gen_parser.add_argument("--device-type", default="android", help="设备类型: android | harmony | ios")
    gen_parser.add_argument("--verbose", "-v", action="store_true", help="详细日志输出")

    # interactive 子命令
    int_parser = subparsers.add_parser("interactive", help="交互式测试模式")
    int_parser.add_argument("--device-type", default=None, help="设备类型: adb")
    int_parser.add_argument("--device-id", default=None, help="设备 ID")
    int_parser.add_argument("--verbose", "-v", action="store_true", help="详细日志输出")
    _add_monitor_args(int_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        run_test(args)
    elif args.command == "generate":
        generate_test(args)
    elif args.command == "interactive":
        interactive_test(args)


def _setup_logging(verbose: bool):
    """配置日志：只对项目模块开 DEBUG，第三方库保持 WARNING"""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    level = logging.DEBUG if verbose else logging.INFO
    for module in ("executor", "runner", "asserter", "planner", "screenshot", "config", "device", "cache", "describer", "optimizer", "monitor"):
        logging.getLogger(module).setLevel(level)


def run_test(args):
    """模式 1：执行 YAML 测试用例"""
    _setup_logging(args.verbose)

    from planner import parse_yaml
    from config.loader import load_global_config

    suite, device_config, model_config, vlm_config, llm_config = parse_yaml(args.yaml_path)
    _, _, _, _, cache_config = load_global_config()
    monitor_config = _build_monitor_config(args)

    action_cache = None
    if cache_config.enabled and not args.no_cache:
        action_cache = _create_action_cache(cache_config)

    runner, _, _, monitor = _build_components(
        device_config, model_config, vlm_config, llm_config,
        device_type_override=args.device_type,
        device_id_override=args.device_id,
        action_cache=action_cache,
        monitor_config=monitor_config,
    )
    try:
        if monitor:
            monitor.start()
            if monitor.url:
                print(f"监控页: {monitor.url}")
        result = runner.run_suite(suite)
    finally:
        if monitor:
            monitor.stop()
    sys.exit(0 if result.failed == 0 else 1)


def _create_action_cache(cache_config):
    """创建 ActionCache 实例"""
    from cache import ActionCache, CacheStore, Embedder

    store = CacheStore(cache_config.db_path)
    embedder = Embedder()
    return ActionCache(
        embedder=embedder,
        store=store,
        similarity_threshold=cache_config.similarity_threshold,
        region_similarity_threshold=cache_config.region_similarity_threshold,
        ttl_days=cache_config.ttl_days,
    )


def _build_components(
    device_config,
    model_config,
    vlm_config,
    llm_config,
    device_type_override: str | None = None,
    device_id_override: str | None = None,
    action_cache=None,
    monitor_config=None,
):
    """创建核心组件：device, executor, asserter, screenshot_mgr, runner"""
    from device import DeviceType, create_device
    from executor.models import create_action_model
    from executor import TestExecutor
    from asserter import Asserter
    from runner import TestRunner
    from screenshot import ScreenshotManager
    from describer import PageDescriber
    from optimizer import ActionOptimizer
    from monitor import MonitorRuntime

    device_type_str = device_type_override or device_config.device_type
    device_id = device_id_override or device_config.device_id
    device = create_device(DeviceType(device_type_str), device_id)

    action_model = create_action_model(
        provider=model_config.provider,
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        temperature=model_config.temperature,
        lang=model_config.lang,
        custom_rules=model_config.custom_rules,
    )

    page_describer = PageDescriber(vlm_config) if vlm_config else None
    action_optimizer = ActionOptimizer(llm_config) if llm_config else None
    monitor = MonitorRuntime(monitor_config, device_id=device.device_id) if monitor_config and monitor_config.enabled else None

    executor = TestExecutor(
        model=action_model, device=device, action_cache=action_cache,
        page_describer=page_describer, action_optimizer=action_optimizer,
        monitor=monitor,
    )
    asserter = Asserter(vlm_config)
    screenshot_mgr = ScreenshotManager(device=device)
    runner = TestRunner(executor, asserter, screenshot_mgr, monitor=monitor)

    return runner, executor, device, monitor


def generate_test(args):
    """模式 2：自然语言 → 生成 YAML 文件"""
    _setup_logging(args.verbose)

    from config.loader import load_global_config
    from planner import plan_test_case, generate_yaml_content, append_to_yaml

    _, _, _, llm_config, _ = load_global_config()

    print(f"\n规划中: {args.description}\n")

    try:
        test_case = plan_test_case(args.description, llm_config)
    except ValueError as e:
        print(f"规划失败: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output and os.path.exists(args.output):
        # 追加模式：向已有 YAML 文件追加新 task
        append_to_yaml(args.output, test_case)
        print(f"已追加到: {args.output}")
        print(f"  新用例: {test_case.name}")
        print(f"  步骤数: {len(test_case.steps)}")
        print(f"\n可通过以下命令执行:")
        print(f"  python main.py run {args.output}")
    elif args.output:
        # 新建模式
        yaml_content = generate_yaml_content(
            test_case,
            device_type=args.device_type,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"已生成: {args.output}")
        print(f"  用例名: {test_case.name}")
        print(f"  步骤数: {len(test_case.steps)}")
        print(f"\n可通过以下命令执行:")
        print(f"  python main.py run {args.output}")
    else:
        yaml_content = generate_yaml_content(
            test_case,
            device_type=args.device_type,
        )
        print("--- 生成的 YAML ---\n")
        print(yaml_content)

    # 打印步骤预览
    _print_steps_preview(test_case)


def interactive_test(args):
    """模式 3：交互式测试"""
    _setup_logging(args.verbose)

    from config.loader import load_global_config
    from planner import plan_test_case
    from suite import TestSuite

    device_config, model_config, vlm_config, llm_config, _ = load_global_config()
    monitor_config = _build_monitor_config(args)

    runner, _, _, monitor = _build_components(
        device_config, model_config, vlm_config, llm_config,
        device_type_override=args.device_type,
        device_id_override=args.device_id,
        monitor_config=monitor_config,
    )
    if monitor:
        monitor.start()
        if monitor.url:
            print(f"监控页: {monitor.url}")

    print("\n" + "=" * 60)
    print("  AutoQA 交互式测试模式")
    print("  输入自然语言描述测试步骤和预期结果")
    print("  输入 quit 或 exit 退出")
    print("=" * 60)

    round_num = 0
    while True:
        round_num += 1
        print(f"\n── 第 {round_num} 轮 ──")

        try:
            description = input("\n请描述测试场景: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not description or description.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        # 规划
        print(f"\n规划中...\n")
        try:
            test_case = plan_test_case(description, llm_config)
        except ValueError as e:
            print(f"规划失败: {e}")
            continue

        _print_steps_preview(test_case)

        # 确认执行
        try:
            confirm = input("\n是否执行? (Y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if confirm in ("n", "no"):
            print("已跳过")
            continue

        # 执行
        suite = TestSuite(name=f"交互测试-{round_num}", test_cases=[test_case])
        result = runner.run_suite(suite)

        if result.failed == 0:
            print("\n所有测试通过!")
        else:
            print(f"\n{result.failed}/{result.total} 个步骤失败")

    if monitor:
        monitor.stop()


def _print_steps_preview(test_case):
    """打印步骤预览"""
    print(f"\n  用例: {test_case.name}")
    print(f"  步骤:")
    for i, step in enumerate(test_case.steps, 1):
        from suite import ActionStep, AssertStep
        if isinstance(step, ActionStep):
            print(f"    {i}. [操作] {step.description}")
        elif isinstance(step, AssertStep):
            sev = f" ({step.severity})" if step.severity != "critical" else ""
            print(f"    {i}. [断言] {step.expectation}{sev}")


def _add_monitor_args(parser):
    parser.add_argument("--live-monitor", action="store_true", help="启动本地监控页（scrcpy + 事件时间线）")
    parser.add_argument("--monitor-host", default="127.0.0.1", help="监控页监听地址")
    parser.add_argument("--monitor-port", type=int, default=8765, help="监控页端口")
    parser.add_argument("--artifact-dir", default=".artifacts/monitor", help="监控截图与产物目录")
    parser.add_argument("--scrcpy-path", default=None, help="scrcpy 可执行文件路径")
    parser.add_argument("--scrcpy-max-fps", type=int, default=15, help="scrcpy 最大帧率")
    parser.add_argument("--scrcpy-bit-rate", default="6M", help="scrcpy 视频码率，如 6M")
    parser.add_argument("--scrcpy-max-size", type=int, default=1080, help="scrcpy 最大边长，0 表示不限制")
    parser.add_argument("--show-input-text", action="store_true", help="监控图中显示输入文本原文（默认脱敏）")


def _build_monitor_config(args):
    if not getattr(args, "live_monitor", False):
        return None

    from monitor import MonitorConfig

    return MonitorConfig(
        enabled=True,
        host=args.monitor_host,
        port=args.monitor_port,
        artifact_dir=args.artifact_dir,
        scrcpy_path=args.scrcpy_path,
        scrcpy_max_fps=args.scrcpy_max_fps,
        scrcpy_video_bit_rate=args.scrcpy_bit_rate,
        scrcpy_max_size=args.scrcpy_max_size,
        mask_input_text=not args.show_input_text,
    )


if __name__ == "__main__":
    main()
