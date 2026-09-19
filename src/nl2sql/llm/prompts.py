"""Versioned prompt templates loaded from disk.

Prompts are data, not code. They live in ``prompts/<name>/<version>.yaml`` so
that changing an instruction is a reviewable change to a file rather than an
edit inside business logic, and so the version that produced an answer can be
recorded next to it. Without that record, a change in answer quality after a
prompt edit is indistinguishable from a change in the model.

Substitution uses ``string.Template``, so the JSON braces that appear in
output examples need no escaping, and a missing variable is an error at render
time rather than a literal placeholder sent to a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import yaml

from nl2sql.core.exceptions import ConfigurationError

_VERSION_PATTERN = re.compile(r"^v(\d+)$")


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A prompt with its variables filled in, ready to send."""

    name: str
    version: str
    system: str
    user: str

    @property
    def ref(self) -> str:
        """Return the identifier recorded against anything this prompt produced."""
        return f"{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One version of one prompt."""

    name: str
    version: str
    system: str
    user: str
    description: str = ""

    @property
    def ref(self) -> str:
        """Return the name and version."""
        return f"{self.name}@{self.version}"

    def render(self, **values: Any) -> RenderedPrompt:
        """Render both messages, raising when a placeholder has no value."""
        as_text = {key: "" if value is None else str(value) for key, value in values.items()}
        try:
            system = Template(self.system).substitute(as_text)
            user = Template(self.user).substitute(as_text)
        except KeyError as exc:
            raise ConfigurationError(f"Prompt {self.ref} is missing a value for {exc}.") from exc
        except ValueError as exc:
            raise ConfigurationError(
                f"Prompt {self.ref} contains an invalid placeholder: {exc}. "
                "Write a literal dollar sign as two dollar signs."
            ) from exc
        return RenderedPrompt(name=self.name, version=self.version, system=system, user=user)


class PromptRegistry:
    """Loads every prompt version from disk and serves the active one."""

    def __init__(
        self,
        directory: Path | str,
        *,
        active_versions: dict[str, str] | None = None,
        untrusted_delimiters: tuple[str, ...] = (),
    ) -> None:
        self._directory = Path(directory)
        self._active = dict(active_versions or {})
        self._untrusted_delimiters = untrusted_delimiters
        self._templates: dict[str, dict[str, PromptTemplate]] = {}
        self._load()

    @property
    def untrusted_delimiters(self) -> tuple[str, ...]:
        """Return the delimiter names that untrusted text must not contain."""
        return self._untrusted_delimiters

    def _load(self) -> None:
        if not self._directory.is_dir():
            raise ConfigurationError(
                f"The prompt directory {self._directory} does not exist. "
                "Set prompts.directory to the folder holding the prompt files."
            )
        for path in sorted(self._directory.glob("*/*.yaml")):
            template = self._parse(path)
            self._templates.setdefault(template.name, {})[template.version] = template
        if not self._templates:
            raise ConfigurationError(f"No prompt files were found under {self._directory}.")

    @staticmethod
    def _parse(path: Path) -> PromptTemplate:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"Prompt file {path} could not be read: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigurationError(f"Prompt file {path} must contain a mapping.")
        name = str(data.get("name") or path.parent.name)
        version = str(data.get("version") or path.stem)
        system = data.get("system")
        user = data.get("user")
        if not isinstance(system, str) or not isinstance(user, str):
            raise ConfigurationError(
                f"Prompt file {path} must define string 'system' and 'user' fields."
            )
        if name != path.parent.name or version != path.stem:
            raise ConfigurationError(
                f"Prompt file {path} declares {name}@{version}, which does not match its "
                "directory and file name."
            )
        return PromptTemplate(
            name=name,
            version=version,
            system=system.strip(),
            user=user.strip(),
            description=str(data.get("description") or ""),
        )

    def names(self) -> tuple[str, ...]:
        """Return every prompt name that was loaded."""
        return tuple(sorted(self._templates))

    def versions_of(self, name: str) -> tuple[str, ...]:
        """Return the versions available for one prompt, oldest first."""
        available = self._templates.get(name, {})
        return tuple(sorted(available, key=self._version_sort_key))

    @staticmethod
    def _version_sort_key(version: str) -> tuple[int, str]:
        match = _VERSION_PATTERN.match(version)
        return (int(match.group(1)), version) if match else (0, version)

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """Return one prompt, defaulting to the configured active version."""
        available = self._templates.get(name)
        if not available:
            raise ConfigurationError(
                f"No prompt named {name} was found in {self._directory}. "
                f"Available prompts: {', '.join(self.names())}."
            )
        wanted = version or self._active.get(name)
        if wanted is None:
            wanted = self.versions_of(name)[-1]
        template = available.get(wanted)
        if template is None:
            raise ConfigurationError(
                f"Prompt {name} has no version {wanted}. "
                f"Available versions: {', '.join(self.versions_of(name))}."
            )
        return template

    def render(self, name: str, *, version: str | None = None, **values: Any) -> RenderedPrompt:
        """Render the active version of a prompt."""
        return self.get(name, version).render(**values)

    def active_versions(self) -> dict[str, str]:
        """Return the version in force for every loaded prompt."""
        return {name: self.get(name).version for name in self.names()}

    def require(self, names: tuple[str, ...]) -> None:
        """Fail fast at startup when a prompt the pipeline needs is absent."""
        missing = [name for name in names if name not in self._templates]
        if missing:
            raise ConfigurationError(
                f"These prompts are required but were not found in {self._directory}: "
                f"{', '.join(missing)}."
            )
