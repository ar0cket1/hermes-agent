"""Tests for /reasoning slash command in HermesCLI."""

from unittest.mock import patch


def _make_cli(**kwargs):
    """Create a HermesCLI instance with minimal mocking."""
    import cli as _cli_mod
    from cli import HermesCLI

    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {"reasoning_effort": "medium"},
        "terminal": {"env_type": "local"},
    }

    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}

    with (
        patch("cli.get_tool_definitions", return_value=[]),
        patch.dict("os.environ", clean_env, clear=False),
        patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": clean_config}),
    ):
        return HermesCLI(**kwargs)


def test_reasoning_command_sets_high_and_persists():
    cli_obj = _make_cli()
    cli_obj.agent = object()  # ensure command forces re-init

    with patch("cli.save_config_value", return_value=True) as mock_save:
        keep_running = cli_obj.process_command("/reasoning high")

    assert keep_running is True
    assert cli_obj.reasoning_config == {"enabled": True, "effort": "high"}
    assert cli_obj.agent is None
    mock_save.assert_called_once_with("agent.reasoning_effort", "high")


def test_reasoning_command_rejects_invalid_level(capsys):
    cli_obj = _make_cli()
    before = cli_obj.reasoning_config

    with patch("cli.save_config_value", return_value=True) as mock_save:
        keep_running = cli_obj.process_command("/reasoning ultra")

    out = capsys.readouterr().out
    assert keep_running is True
    assert "Invalid reasoning level" in out
    assert cli_obj.reasoning_config == before
    mock_save.assert_not_called()


def test_reasoning_command_without_args_shows_current(capsys):
    cli_obj = _make_cli()
    cli_obj.reasoning_config = {"enabled": True, "effort": "medium"}

    keep_running = cli_obj.process_command("/reasoning")

    out = capsys.readouterr().out
    assert keep_running is True
    assert "Current reasoning effort: medium" in out
