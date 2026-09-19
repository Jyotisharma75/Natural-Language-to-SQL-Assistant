"""Prompt injection screening.

Screening a question for injection patterns is the weakest of the three
defences here, and it is deliberately the least relied upon. The other two do
the real work:

* untrusted text is always placed inside a delimited block in a prompt, and
  anything resembling a closing delimiter is neutralised first, so a question
  cannot end the block and start issuing instructions
* model output is never trusted. Whatever the model returns is parsed,
  guarded, checked against the access policy and rewritten before it can reach
  the database, so a successful injection still cannot produce a statement the
  policy forbids

The patterns are configuration, and the action taken, reject or warn, is too.
"""

from __future__ import annotations

import re

from nl2sql.config.settings import PromptInjectionSettings
from nl2sql.core.exceptions import PromptInjectionError


class PromptInjectionDetector:
    """Matches a question against the configured injection patterns."""

    def __init__(self, settings: PromptInjectionSettings) -> None:
        self._settings = settings
        self._rules: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
            (name, re.compile(pattern)) for name, pattern in settings.patterns.items()
        )

    @property
    def enabled(self) -> bool:
        """Return whether screening is switched on."""
        return self._settings.enabled

    def scan(self, text: str) -> tuple[str, ...]:
        """Return the names of every rule that matched."""
        if not self._settings.enabled or not text:
            return ()
        return tuple(name for name, pattern in self._rules if pattern.search(text))

    def enforce(self, text: str) -> tuple[str, ...]:
        """Screen ``text``, raising when the configured action is to reject.

        Returns the matched rule names so the caller can record them as
        warnings when the action is to warn.
        """
        findings = self.scan(text)
        if findings and self._settings.action == "reject":
            raise PromptInjectionError(
                "The question was refused by prompt injection screening.",
                details={"rules": list(findings)},
            )
        return findings


def neutralise_delimiters(text: str, delimiters: tuple[str, ...]) -> str:
    """Remove anything that looks like one of the prompt's own block delimiters.

    Untrusted text is wrapped in tags such as ``<user_question>``. If the text
    itself contained ``</user_question>`` it could appear to close the block
    and have what follows read as instructions. Stripping the tag shapes costs
    nothing, because no real question contains them.
    """
    if not text or not delimiters:
        return text
    names = "|".join(re.escape(name) for name in delimiters)
    pattern = re.compile(rf"</?\s*(?:{names})\s*/?>", re.IGNORECASE)
    return pattern.sub(" ", text)
