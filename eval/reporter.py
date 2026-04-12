"""Reporter: 生成 HTML 诊断报告"""
from __future__ import annotations

import logging
import os
from typing import Any

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


class Reporter:
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=True,
        )

    def generate(self, eval_result: dict[str, Any], output_path: str):
        template = self.env.get_template("report.html")
        html = template.render(**eval_result)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("评测报告已生成: %s", output_path)
