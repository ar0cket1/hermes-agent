"""Tests for non-interactive setup and first-run headless behavior."""

from argparse import Namespace
from unittest.mock import patch

import pytest


def _make_setup_args(**overrides):
    return Namespace(
        non_interactive=overrides.get("non_interactive", False),
        section=overrides.get("section", None),
        reset=overrides.get("reset", False),
    )


def _make_chat_args(**overrides):
    return Namespace(
        continue_last=overrides.get("continue_last", None),
        resume=overrides.get("resume", None),
        model=overrides.get("model", None),
        provider=overrides.get("provider", None),
        toolsets=overrides.get("toolsets", None),
        verbose=overrides.get("verbose", False),
        query=overrides.get("query", None),
        worktree=overrides.get("worktree", False),
        yolo=overrides.get("yolo", False),
        pass_session_id=overrides.get("pass_session_id", False),
        quiet=overrides.get("quiet", False),
        checkpoints=overrides.get("checkpoints", False),
    )


class TestNonInteractiveSetup:
    """Verify setup paths exit cleanly in headless/non-interactive environments."""

    def test_non_interactive_flag_skips_wizard(self, capsys):
        """--non-interactive should print guidance and not enter the wizard."""
        from hermes_cli.setup import run_setup_wizard

        args = _make_setup_args(non_interactive=True)

        with (
            patch("hermes_cli.setup.ensure_hermes_home"),
            patch("hermes_cli.setup.load_config", return_value={}),
            patch("hermes_cli.setup.get_hermes_home", return_value="/tmp/.hermes"),
            patch("hermes_cli.auth.get_active_provider", side_effect=AssertionError("wizard continued")),
            patch("builtins.input", side_effect=AssertionError("input should not be called")),
        ):
            run_setup_wizard(args)

        out = capsys.readouterr().out
        assert "hermes config set model.provider custom" in out

    def test_no_tty_skips_wizard(self, capsys):
        """When stdin has no TTY, the setup wizard should print guidance and return."""
        from hermes_cli.setup import run_setup_wizard

        args = _make_setup_args(non_interactive=False)

        with (
            patch("hermes_cli.setup.ensure_hermes_home"),
            patch("hermes_cli.setup.load_config", return_value={}),
            patch("hermes_cli.setup.get_hermes_home", return_value="/tmp/.hermes"),
            patch("hermes_cli.auth.get_active_provider", side_effect=AssertionError("wizard continued")),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=AssertionError("input should not be called")),
        ):
            mock_stdin.isatty.return_value = False
            run_setup_wizard(args)

        out = capsys.readouterr().out
        assert "hermes config set model.provider custom" in out

    def test_chat_first_run_headless_skips_setup_prompt(self, capsys):
        """Bare `hermes` should not prompt for input when no provider exists and stdin is headless."""
        from hermes_cli.main import cmd_chat

        args = _make_chat_args()

        with (
            patch("hermes_cli.main._has_any_provider_configured", return_value=False),
            patch("hermes_cli.main.cmd_setup") as mock_setup,
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=AssertionError("input should not be called")),
        ):
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc:
                cmd_chat(args)

        assert exc.value.code == 1
        mock_setup.assert_not_called()
        out = capsys.readouterr().out
        assert "hermes config set model.provider custom" in out


class TestReturningSetupMenu:
    """Verify the returning-user setup menu routes to the intended section."""

    def test_online_rl_menu_choice_opens_online_rl_setup(self, tmp_path):
        from hermes_cli import setup as setup_mod

        args = _make_setup_args(non_interactive=False)
        config = {"online_rl": {"enabled": True, "backend": "tinker", "algorithm": "sdpo"}}

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", return_value=config),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch.object(
                setup_mod,
                "get_env_value",
                side_effect=lambda key: "sk-test" if key == "OPENROUTER_API_KEY" else "",
            ),
            patch.object(setup_mod, "prompt_choice", return_value=8),
            patch.object(setup_mod, "setup_online_rl") as mock_setup_online_rl,
            patch.object(setup_mod, "setup_agent_settings") as mock_setup_agent_settings,
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        mock_setup_online_rl.assert_called_once_with(config)
        mock_setup_agent_settings.assert_not_called()

    def test_returning_setup_defaults_to_online_rl_when_configured(self, tmp_path):
        from hermes_cli import setup as setup_mod

        args = _make_setup_args(non_interactive=False)
        config = {"online_rl": {"enabled": True, "backend": "tinker", "algorithm": "sdpo"}}

        def _pick_default(question, choices, default=0, **kwargs):
            assert default == 8
            assert kwargs.get("treat_default_as_skip") is False
            return default

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", return_value=config),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch.object(
                setup_mod,
                "get_env_value",
                side_effect=lambda key: "sk-test" if key == "OPENROUTER_API_KEY" else "",
            ),
            patch.object(setup_mod, "prompt_choice", side_effect=_pick_default),
            patch.object(setup_mod, "setup_online_rl") as mock_setup_online_rl,
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        mock_setup_online_rl.assert_called_once_with(config)


class TestTinkerOnlineRLSetup:
    def test_tinker_online_rl_reconfigures_primary_runtime(self, tmp_path, monkeypatch):
        from hermes_cli import setup as setup_mod
        from hermes_cli.config import get_env_path

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        config = {"online_rl": {}, "model": {"provider": "openrouter", "default": "old/model"}}

        prompt_values = iter(["test-tinker-key"])

        def fake_prompt(question, default=None, password=False):
            return next(prompt_values)

        def fake_prompt_choice(question, choices, default=0, **kwargs):
            if question == "Which model to train?":
                return 0
            if question == "Which online RL mode do you want?":
                return 0
            raise AssertionError(f"Unexpected prompt_choice: {question}")

        monkeypatch.setattr(setup_mod.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(setup_mod, "prompt", fake_prompt)
        monkeypatch.setattr(setup_mod, "prompt_choice", fake_prompt_choice)
        monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *args, **kwargs: True)
        monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)

        setup_mod._setup_online_rl_tinker(config, config["online_rl"])

        assert config["online_rl"]["backend"] == "tinker"
        assert config["online_rl"]["algorithm"] == "sdpo"
        assert config["online_rl"]["training_base_model"] == "moonshotai/Kimi-K2.5"
        assert config["online_rl"]["learning_rate"] == 1e-6
        assert config["online_rl"]["min_batch_size"] == 4
        assert config["online_rl"]["sdpo_topk"] == 20
        assert config["online_rl"]["train_steps"] == 0
        assert config["online_rl"]["gradient_accumulation_steps"] == 4

        model_cfg = config["model"]
        assert model_cfg["provider"] == "custom"
        assert model_cfg["default"] == "moonshotai/Kimi-K2.5"
        assert model_cfg["base_url"] == "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
        assert model_cfg["api_mode"] == "chat_completions"

        env_path = get_env_path()
        assert env_path.exists()
        assert "TINKER_API_KEY=test-tinker-key" in env_path.read_text()

    def test_quick_setup_offers_online_rl_reconfigure_when_nothing_missing(self, tmp_path):
        from hermes_cli import setup as setup_mod

        config = {"online_rl": {"enabled": True, "backend": "tinker", "algorithm": "sdpo"}}

        with (
            patch.object(setup_mod, "prompt_yes_no", return_value=True),
            patch.object(setup_mod, "setup_online_rl") as mock_setup_online_rl,
            patch.object(setup_mod, "save_config") as mock_save_config,
            patch.object(setup_mod, "_print_setup_summary") as mock_print_summary,
            patch("hermes_cli.config.get_missing_env_vars", return_value=[]),
            patch("hermes_cli.config.get_missing_config_fields", return_value=[]),
            patch("hermes_cli.config.check_config_version", return_value=(1, 1)),
        ):
            setup_mod._run_quick_setup(config, tmp_path)

        mock_setup_online_rl.assert_called_once_with(config)
        mock_save_config.assert_called_once_with(config)
        mock_print_summary.assert_called_once_with(config, tmp_path)
