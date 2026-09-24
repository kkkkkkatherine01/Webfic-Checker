"""Smoke tests: both command-line entry points import and show their help."""

import pytest
from typer.testing import CliRunner

from webfic.cli import app as webfic_app
from webfic.evaluation.cli import app as eval_app


@pytest.mark.parametrize("app", [webfic_app, eval_app])
def test_help(app):
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output


def test_failure_summary():
    from webfic.cli import _failure_summary

    assert _failure_summary({}) == ""
    assert _failure_summary({"content_filter": 2}) == "（被服务商内容审核拒绝 2 章）"
