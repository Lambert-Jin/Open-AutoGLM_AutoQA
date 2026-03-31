"""utils/text 工具函数测试"""
from __future__ import annotations

import os
import pytest
from utils.text import resolve_env_vars, resolve_dict, strip_markdown_json


class TestResolveEnvVars:
    def test_basic_substitution(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret123")
        assert resolve_env_vars("${MY_KEY}") == "secret123"

    def test_missing_var_strict(self):
        with pytest.raises(ValueError, match="not set"):
            resolve_env_vars("${NONEXISTENT_VAR_12345}", strict=True)

    def test_missing_var_lenient(self):
        assert resolve_env_vars("${NONEXISTENT_VAR_12345}", strict=False) == "${NONEXISTENT_VAR_12345}"

    def test_no_placeholder(self):
        assert resolve_env_vars("plain text") == "plain text"

    def test_mixed(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        assert resolve_env_vars("http://${HOST}:8080") == "http://localhost:8080"


class TestResolveDict:
    def test_nested(self, monkeypatch):
        monkeypatch.setenv("KEY", "val")
        data = {"a": "${KEY}", "b": {"c": "${KEY}"}, "d": [{"e": "${KEY}"}]}
        result = resolve_dict(data, strict=False)
        assert result == {"a": "val", "b": {"c": "val"}, "d": [{"e": "val"}]}

    def test_non_string_values(self):
        data = {"a": 123, "b": True, "c": None}
        assert resolve_dict(data) == {"a": 123, "b": True, "c": None}


class TestStripMarkdownJson:
    def test_plain_json(self):
        assert strip_markdown_json('{"key": "value"}') == '{"key": "value"}'

    def test_with_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_markdown_json(text) == '{"key": "value"}'

    def test_with_code_block_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        assert strip_markdown_json(text) == '{"key": "value"}'

    def test_with_surrounding_text(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        assert strip_markdown_json(text) == '{"key": "value"}'

    def test_no_code_block(self):
        text = '{"key": "value"}'
        assert strip_markdown_json(text) == '{"key": "value"}'