"""AutoGLM 适配器：自主实现，不依赖 phone_agent"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from executor.actions import ActionType, ModelOutput, UnifiedAction
from executor.models.base_client import BaseModelClient, ClientConfig

logger = logging.getLogger(__name__)

# ── 动作名 → ActionType 映射 ──

_ACTION_MAP = {
    "Tap": ActionType.TAP,
    "Double Tap": ActionType.DOUBLE_TAP,
    "Long Press": ActionType.LONG_PRESS,
    "Swipe": ActionType.SWIPE,
    "Type": ActionType.TYPE,
    "Type_Name": ActionType.TYPE,
    "Back": ActionType.BACK,
    "Home": ActionType.HOME,
    "Launch": ActionType.LAUNCH,
    "Wait": ActionType.WAIT,
    "Take_over": ActionType.TAKE_OVER,
    "Note": ActionType.NOTE,
    "Call_API": ActionType.CALL_API,
    "Interact": ActionType.TAKE_OVER,  # Interact 也需要用户介入
}


# ── System Prompt ──

def _build_system_prompt_cn(custom_rules: list[str] | None = None) -> str:
    """构建中文 system prompt，从 phone_agent 独立维护"""
    today = datetime.today()
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekday_names[today.weekday()]
    formatted_date = today.strftime("%Y年%m月%d日") + " " + weekday

    prompt = """\
今天的日期是: {date}
你是一个智能体分析专家，可以根据操作历史和当前状态图执行一系列操作来完成任务。
你必须严格按照要求输出以下格式：
<think>{{think}}</think>
<answer>{{action}}</answer>

其中：
- {{think}} 是对你为什么选择这个操作的简短推理说明。
- {{action}} 是本次执行的具体操作指令，必须严格遵循下方定义的指令格式。

重要：每次回复只能输出一个 <answer> 和一个操作指令。执行后你会收到新的截图，再决定下一步操作。严禁在一次回复中输出多个操作、预测屏幕变化或自行模拟后续步骤。

操作指令及其作用如下：
- do(action="Launch", app="xxx")
    Launch是启动目标app的操作，这比通过主屏幕导航更快。此操作完成后，您将自动收到结果状态的截图。
- do(action="Tap", element=[x,y])
    Tap是点击操作，点击屏幕上的特定点。可用此操作点击按钮、选择项目、从主屏幕打开应用程序，或与任何可点击的用户界面元素进行交互。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的截图。
- do(action="Tap", element=[x,y], message="重要操作")
    基本功能同Tap，点击涉及财产、支付、隐私等敏感按钮时触发。
- do(action="Type", text="xxx")
    Type是输入操作，在当前聚焦的输入框中输入文本。使用此操作前，请确保输入框已被聚焦（先点击它）。输入的文本将像使用键盘输入一样输入。重要提示：手机可能正在使用 ADB 键盘，该键盘不会像普通键盘那样占用屏幕空间。要确认键盘已激活，请查看屏幕底部是否显示 'ADB Keyboard {{ON}}' 类似的文本，或者检查输入框是否处于激活/高亮状态。不要仅仅依赖视觉上的键盘显示。自动清除文本：当你使用输入操作时，输入框中现有的任何文本（包括占位符文本和实际输入）都会在输入新文本前自动清除。你无需在输入前手动清除文本——直接使用输入操作输入所需文本即可。操作完成后，你将自动收到结果状态的截图。
- do(action="Type_Name", text="xxx")
    Type_Name是输入人名的操作，基本功能同Type。
- do(action="Interact")
    Interact是当有多个满足条件的选项时而触发的交互操作，询问用户如何选择。
- do(action="Swipe", start=[x1,y1], end=[x2,y2])
    Swipe是滑动操作，通过从起始坐标拖动到结束坐标来执行滑动手势。可用于滚动内容、在屏幕之间导航、下拉通知栏以及项目栏或进行基于手势的导航。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。滑动持续时间会自动调整以实现自然的移动。此操作完成后，您将自动收到结果状态的截图。
- do(action="Note", message="True")
    记录当前页面内容以便后续总结。
- do(action="Call_API", instruction="xxx")
    总结或评论当前页面或已记录的内容。
- do(action="Long Press", element=[x,y])
    Long Press是长按操作，在屏幕上的特定点长按指定时间。可用于触发上下文菜单、选择文本或激活长按交互。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的屏幕截图。
- do(action="Double Tap", element=[x,y])
    Double Tap在屏幕上的特定点快速连续点按两次。使用此操作可以激活双击交互，如缩放、选择文本或打开项目。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的截图。
- do(action="Take_over", message="xxx")
    Take_over是接管操作，表示在登录和验证阶段需要用户协助。
- do(action="Back")
    导航返回到上一个屏幕或关闭当前对话框。相当于按下 Android 的返回按钮。使用此操作可以从更深的屏幕返回、关闭弹出窗口或退出当前上下文。此操作完成后，您将自动收到结果状态的截图。
- do(action="Home")
    Home是回到系统桌面的操作，相当于按下 Android 主屏幕按钮。使用此操作可退出当前应用并返回启动器，或从已知状态启动新任务。此操作完成后，您将自动收到结果状态的截图。
- do(action="Wait", duration="x seconds")
    等待页面加载，x为需要等待多少秒。
- finish(message="xxx")
    finish是结束任务的操作，表示准确完整完成任务，message是终止信息。

必须遵循的规则：
1. 在执行任何操作前，先检查当前app是否是目标app，如果不是，先执行 Launch。
2. 如果进入到了无关页面，先执行 Back。如果执行Back后页面没有变化，请点击页面左上角的返回键进行返回，或者右上角的X号关闭。
3. 如果页面未加载出内容，最多连续 Wait 三次，否则执行 Back重新进入。
4. 如果页面显示网络问题，需要重新加载，请点击重新加载。
5. 如果当前页面找不到目标联系人、商品、店铺等信息，可以尝试 Swipe 滑动查找。
6. 遇到价格区间、时间区间等筛选条件，如果没有完全符合的，可以放宽要求。
7. 在做小红书总结类任务时一定要筛选图文笔记。
8. 购物车全选后再点击全选可以把状态设为全不选，在做购物车任务时，如果购物车里已经有商品被选中时，你需要点击全选后再点击取消全选，再去找需要购买或者删除的商品。
9. 在做外卖任务时，如果相应店铺购物车里已经有其他商品你需要先把购物车清空再去购买用户指定的外卖。
10. 在做点外卖任务时，如果用户需要点多个外卖，请尽量在同一店铺进行购买，如果无法找到可以下单，并说明某个商品未找到。
11. 请严格遵循用户意图执行任务，用户的特殊要求可以执行多次搜索，滑动查找。比如（i）用户要求点一杯咖啡，要咸的，你可以直接搜索咸咖啡，或者搜索咖啡后滑动查找咸的咖啡，比如海盐咖啡。（ii）用户要找到XX群，发一条消息，你可以先搜索XX群，找不到结果后，将\u201c群\u201d字去掉，搜索XX重试。（iii）用户要找到宠物友好的餐厅，你可以搜索餐厅，找到筛选，找到设施，选择可带宠物，或者直接搜索可带宠物，必要时可以使用AI搜索。
12. 在选择日期时，如果原滑动方向与预期日期越来越远，请向反方向滑动查找。
13. 执行任务过程中如果有多个可选择的项目栏，请逐个查找每个项目栏，直到完成任务，一定不要在同一项目栏多次查找，从而陷入死循环。
14. 在执行下一步操作前请一定要检查上一步的操作是否生效，如果点击没生效，可能因为app反应较慢，请先稍微等待一下，如果还是不生效请调整一下点击位置重试，如果仍然不生效请跳过这一步继续任务，并在finish message说明点击不生效。
15. 在执行任务中如果遇到滑动不生效的情况，请调整一下起始点位置，增大滑动距离重试，如果还是不生效，有可能是已经滑到底了，请继续向反方向滑动，直到顶部或底部，如果仍然没有符合要求的结果，请跳过这一步继续任务，并在finish message说明但没找到要求的项目。
16. 在做游戏任务时如果在战斗页面如果有自动战斗一定要开启自动战斗，如果多轮历史状态相似要检查自动战斗是否开启。
17. 如果没有合适的搜索结果，可能是因为搜索页面不对，请返回到搜索页面的上一级尝试重新搜索，如果尝试三次返回上一级搜索后仍然没有符合要求的结果，执行 finish(message="原因")。
18. 在结束任务前请一定要仔细检查任务是否完整准确的完成，如果出现错选、漏选、多选的情况，请返回之前的步骤进行纠正。
19. 严格只执行指令要求的操作，不要自行推断下一步或多做额外操作。例如指令要求"找到某个按钮"，你只需要确认该按钮在屏幕上可见即可，不要自动点击它。完成指令要求的操作后立即 finish。
""".format(date=formatted_date)

    if custom_rules:
        rules = "\n".join("- " + r for r in custom_rules)
        prompt += "\n额外规则：\n" + rules + "\n"

    return prompt


def _build_system_prompt_en(custom_rules: list[str] | None = None) -> str:
    """构建英文 system prompt"""
    today = datetime.today()
    formatted_date = today.strftime("%Y-%m-%d, %A")

    prompt = """\
The current date: {date}
# Setup
You are a professional Android operation agent assistant that can fulfill the user's high-level instructions. \
Given a screenshot of the Android interface at each step, you first analyze the situation, then plan the best \
course of action using Python-style pseudo-code.

# More details about the code
Your response format must be structured as follows:

Think first: Use <think>...</think> to analyze the current screen, identify key elements, and determine the most efficient action.
Provide the action: Use <answer>...</answer> to return a single line of pseudo-code representing the operation.

Your output should STRICTLY follow the format:
<think>
[Your thought]
</think>
<answer>
[Your operation code]
</answer>

- **Tap**
  Perform a tap action on a specified screen area.
  do(action="Tap", element=[x,y])
- **Type**
  Enter text into the currently focused input field.
  do(action="Type", text="Hello World")
- **Swipe**
  Perform a swipe action with start point and end point.
  do(action="Swipe", start=[x1,y1], end=[x2,y2])
- **Long Press**
  Perform a long press action on a specified screen area.
  do(action="Long Press", element=[x,y])
- **Launch**
  Launch an app.
  do(action="Launch", app="Settings")
- **Back**
  Press the Back button to navigate to the previous screen.
  do(action="Back")
- **Finish**
  Terminate the program and optionally print a message.
  finish(message="Task completed.")

REMEMBER:
- Think before you act: Always analyze the current UI and the best course of action before executing any step.
- Only ONE LINE of action in <answer> part per response.
- Generate execution code strictly according to format requirements.
""".format(date=formatted_date)

    if custom_rules:
        rules = "\n".join("- " + r for r in custom_rules)
        prompt += "\nAdditional rules:\n" + rules + "\n"

    return prompt


# ── AutoGLM 适配器 ──

@dataclass
class AutoGLMConfig:
    """AutoGLM 模型配置"""

    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key: str = ""
    model: str = "autoglm-phone"
    max_tokens: int = 3000
    temperature: float = 0.0
    lang: str = "cn"
    custom_rules: list[str] = field(default_factory=list)


class AutoGLMModel:
    """
    AutoGLM 适配器。

    改进（对比 phone_agent）：
    - system prompt 独立维护，支持通过 custom_rules 注入自定义规则
    - 不 print，全部走 logging
    - 坐标转换使用 round() + clamp，精度更高
    - 用正则分离 thinking/action，更健壮
    """

    def __init__(self, config: AutoGLMConfig):
        self._config = config
        self._client = BaseModelClient(ClientConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        ))

    def get_system_prompt(self) -> str:
        if self._config.lang == "cn":
            return _build_system_prompt_cn(self._config.custom_rules or None)
        return _build_system_prompt_en(self._config.custom_rules or None)

    @staticmethod
    def build_screen_info(current_app: str, **extra_info) -> str:
        """构建 screen_info JSON（与 phone_agent.MessageBuilder.build_screen_info 兼容）"""
        info = {"current_app": current_app, **extra_info}
        return json.dumps(info, ensure_ascii=False)

    def build_user_message(
        self,
        text: str,
        image_base64: str | None = None,
        screen_width: int = 0,
        screen_height: int = 0,
    ) -> dict[str, Any]:
        if image_base64:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                ],
            }
        return {"role": "user", "content": text}

    def build_assistant_message(self, raw_content: str) -> dict[str, Any]:
        return {"role": "assistant", "content": raw_content}

    def remove_images(self, message: dict) -> dict:
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [c for c in content if c.get("type") != "image_url"]
            return {**message, "content": text_parts or ""}
        return message

    def call(self, messages: list[dict]) -> ModelOutput:
        output = self._client.request(messages, stream=True)
        thinking, action = self._split_response(output.raw_content)
        return ModelOutput(
            thinking=thinking,
            action_text=action,
            raw_content=output.raw_content,
            time_to_first_token=output.time_to_first_token,
            total_time=output.total_time,
        )

    def parse(
        self,
        output: ModelOutput,
        screen_width: int,
        screen_height: int,
    ) -> UnifiedAction:
        action_text = output.action_text.strip()

        # finish(message="...")
        finish_match = re.match(r'finish\(message="(.*)"\)', action_text, re.DOTALL)
        if finish_match:
            return UnifiedAction(
                type=ActionType.FINISH,
                text=finish_match.group(1),
                thinking=output.thinking,
                raw_response=output.raw_content,
            )

        # do(action=..., ...)
        do_match = re.match(r"do\((.+)\)", action_text, re.DOTALL)
        if not do_match:
            # 无法解析则视为 finish
            return UnifiedAction(
                type=ActionType.FINISH,
                text=action_text,
                thinking=output.thinking,
                raw_response=output.raw_content,
            )

        params = self._parse_do_params(do_match.group(1))
        action_name = params.get("action", "")
        action_type = _ACTION_MAP.get(action_name, ActionType.FINISH)

        # 坐标：0-999 相对坐标 → 绝对像素（round + clamp）
        def _to_px(rel: int, screen_dim: int) -> int:
            return max(0, min(screen_dim - 1, round(rel / 1000 * screen_dim)))

        x, y, end_x, end_y = None, None, None, None

        for key in ("position", "element"):
            val = params.get(key)
            if isinstance(val, list) and len(val) >= 2:
                x = _to_px(val[0], screen_width)
                y = _to_px(val[1], screen_height)
                break

        start = params.get("start")
        if isinstance(start, list) and len(start) >= 2:
            x = _to_px(start[0], screen_width)
            y = _to_px(start[1], screen_height)

        end = params.get("end")
        if isinstance(end, list) and len(end) >= 2:
            end_x = _to_px(end[0], screen_width)
            end_y = _to_px(end[1], screen_height)

        # duration（Wait 的 "x seconds" 格式）
        duration_ms = params.get("duration")
        if duration_ms is None and action_type == ActionType.WAIT:
            dur_str = params.get("duration_str", "")
            dur_match = re.search(r"(\d+)", dur_str)
            if dur_match:
                duration_ms = int(dur_match.group(1)) * 1000

        return UnifiedAction(
            type=action_type,
            x=x,
            y=y,
            end_x=end_x,
            end_y=end_y,
            text=params.get("text") or params.get("app_name") or params.get("app")
                 or params.get("message") or params.get("instruction"),
            duration_ms=duration_ms,
            thinking=output.thinking,
            raw_response=output.raw_content,
        )

    @staticmethod
    def _split_response(raw: str) -> tuple[str, str]:
        """
        分离 thinking 和 action。

        改进（对比 phone_agent）：用正则替代字符串搜索，更健壮。
        """
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        if think_match and answer_match:
            return think_match.group(1).strip(), answer_match.group(1).strip()

        # 回退：从后往前查找最后一个 do(...) 或 finish(...)
        # 优先找 do(action=，因为模型有时在 thinking 中提到 finish 但实际要执行 do
        for marker in ("do(action=", "finish(message="):
            idx = raw.rfind(marker)
            if idx >= 0:
                return raw[:idx].strip(), raw[idx:].strip()

        return "", raw

    @staticmethod
    def _parse_do_params(params_str: str) -> dict:
        """
        解析 do(...) 内部参数。

        改进（对比 phone_agent.parse_action）：
        用正则逐个提取参数，不用 ast.literal_eval，更容错。
        """
        result = {}

        # action="Tap" 或 action=Tap
        m = re.search(r'action\s*=\s*"?([^",\)]+)"?', params_str)
        if m:
            result["action"] = m.group(1).strip()

        # position=[500, 300] 或 element=[500, 300]
        for key in ("position", "element", "start", "end"):
            m = re.search(rf"{key}\s*=\s*\[([^\]]+)\]", params_str)
            if m:
                try:
                    nums = [int(float(x.strip())) for x in m.group(1).split(",")]
                    result[key] = nums
                except ValueError:
                    pass

        # text="..." / app_name="..." / app="..." / message="..." / instruction="..."
        for key in ("text", "app_name", "app", "message", "instruction"):
            m = re.search(rf'{key}\s*=\s*"((?:[^"\\]|\\.)*)"', params_str)
            if m:
                result[key] = m.group(1).replace('\\"', '"')

        # duration=N (int)
        m = re.search(r'duration\s*=\s*(\d+)', params_str)
        if m:
            result["duration"] = int(m.group(1))

        # duration="x seconds" (string)
        m = re.search(r'duration\s*=\s*"([^"]*)"', params_str)
        if m:
            result["duration_str"] = m.group(1)

        return result