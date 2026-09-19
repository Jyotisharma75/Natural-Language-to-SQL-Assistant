"""Configuration layering and the versioned prompt registry."""

from __future__ import annotations

import pytest
import yaml

from nl2sql.config.loader import deep_merge, load_config_mapping, load_settings
from nl2sql.config.secrets import require_secret, resolve_secret
from nl2sql.core.exceptions import ConfigurationError, SecretNotFoundError
from nl2sql.llm.prompts import PromptRegistry

pytestmark = pytest.mark.unit


def test_deep_merge_replaces_lists_and_merges_mappings():
    """Nested mappings merge; a list in the override replaces the base list."""
    base = {"a": {"b": 1, "c": 2}, "list": [1, 2, 3]}
    override = {"a": {"c": 3}, "list": [9]}
    assert deep_merge(base, override) == {"a": {"b": 1, "c": 3}, "list": [9]}


def test_environment_layer_overrides_the_base(repo_root):
    """The environment file wins over base.yaml."""
    mapping = load_config_mapping(env="test", config_dir=repo_root / "configs")
    assert mapping["env"] == "test"
    assert mapping["api"]["auth_mode"] == "none"
    assert mapping["security"]["allowed_statements"] == ["SELECT"]


def test_environment_variables_override_the_files(repo_root, monkeypatch):
    """An environment variable wins over both YAML layers."""
    monkeypatch.setenv("NL2SQL_LIMITS__MAX_ROWS", "7")
    settings = load_settings(env="test", config_dir=repo_root / "configs", load_dotenv_file=False)
    assert settings.limits.max_rows == 7


def test_missing_environment_file_is_an_error(repo_root):
    """An unknown environment fails loudly rather than falling back."""
    with pytest.raises(ConfigurationError):
        load_config_mapping(env="does-not-exist", config_dir=repo_root / "configs")


def test_invalid_value_reports_the_field_path(repo_root):
    """A validation failure names the offending key."""
    with pytest.raises(ConfigurationError) as caught:
        load_settings(
            env="test",
            config_dir=repo_root / "configs",
            overrides={"limits": {"max_rows": -5}},
            load_dotenv_file=False,
        )
    assert "limits.max_rows" in str(caught.value)


def test_routing_thresholds_must_be_consistent(repo_root):
    """A local threshold above the verify threshold is refused."""
    with pytest.raises(ConfigurationError):
        load_settings(
            env="test",
            config_dir=repo_root / "configs",
            overrides={
                "routing": {"local_max_complexity": 0.9, "always_verify_above_complexity": 0.5}
            },
            load_dotenv_file=False,
        )


def test_tenancy_requires_column_patterns(repo_root):
    """Enabling tenant isolation without telling it which column is refused."""
    with pytest.raises(ConfigurationError):
        load_settings(
            env="test",
            config_dir=repo_root / "configs",
            overrides={"tenancy": {"enabled": True, "tenant_column_patterns": []}},
            load_dotenv_file=False,
        )


def test_settings_summary_holds_no_credentials(settings):
    """The diagnostic summary carries no secret material."""
    summary = settings.public_summary()
    assert "password" not in str(summary).lower()
    assert "api_key" not in str(summary).lower()


def test_secret_resolution(monkeypatch):
    """Secrets come from the environment variable the settings name."""
    monkeypatch.setenv("NL2SQL_TEST_SECRET", "value")
    assert resolve_secret("NL2SQL_TEST_SECRET") == "value"
    assert resolve_secret("NL2SQL_ABSENT") is None
    assert require_secret("NL2SQL_TEST_SECRET", purpose="testing") == "value"
    with pytest.raises(SecretNotFoundError):
        require_secret("NL2SQL_ABSENT", purpose="testing")


# -- prompts ---------------------------------------------------------------
def test_every_required_prompt_is_present(repo_root, settings):
    """The prompts the pipeline needs exist on disk at the configured versions."""
    registry = PromptRegistry(
        repo_root / settings.prompts.directory,
        active_versions=settings.prompts.versions,
        untrusted_delimiters=tuple(settings.prompts.untrusted_delimiters),
    )
    from nl2sql.container import REQUIRED_PROMPTS

    registry.require(REQUIRED_PROMPTS)
    for name in REQUIRED_PROMPTS:
        assert registry.get(name).version == settings.prompts.versions[name]


def test_prompt_version_is_recorded_on_what_it_renders(repo_root, settings):
    """A rendered prompt carries the reference recorded in the audit trail."""
    registry = PromptRegistry(repo_root / settings.prompts.directory)
    rendered = registry.render(
        "sql_verification", dialect="SQLite", schema="TABLE t", question="q", sql="SELECT 1"
    )
    assert rendered.ref == "sql_verification@v1"


def test_missing_prompt_variable_is_an_error(repo_root, settings):
    """A prompt is never sent with an unfilled placeholder."""
    registry = PromptRegistry(repo_root / settings.prompts.directory)
    with pytest.raises(ConfigurationError):
        registry.render("sql_generation", dialect="SQLite")


def test_unknown_prompt_names_the_available_ones(repo_root):
    """An unknown prompt name fails with something actionable."""
    registry = PromptRegistry(repo_root / "prompts")
    with pytest.raises(ConfigurationError) as caught:
        registry.get("no_such_prompt")
    assert "sql_generation" in str(caught.value)


def test_a_second_version_can_be_selected(tmp_path):
    """Two versions can coexist and configuration chooses between them."""
    directory = tmp_path / "prompts"
    for version, instruction in (("v1", "First."), ("v2", "Second.")):
        folder = directory / "demo"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{version}.yaml").write_text(
            yaml.safe_dump(
                {"name": "demo", "version": version, "system": instruction, "user": "$value"}
            ),
            encoding="utf-8",
        )
    assert PromptRegistry(directory).get("demo").version == "v2"
    pinned = PromptRegistry(directory, active_versions={"demo": "v1"})
    assert pinned.render("demo", value="x").system == "First."


def test_prompt_file_must_match_its_location(tmp_path):
    """A prompt whose declared name does not match its path is refused."""
    folder = tmp_path / "prompts" / "demo"
    folder.mkdir(parents=True)
    (folder / "v1.yaml").write_text(
        yaml.safe_dump({"name": "other", "version": "v1", "system": "s", "user": "u"}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        PromptRegistry(tmp_path / "prompts")
