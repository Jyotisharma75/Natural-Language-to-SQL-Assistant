"""Command line interface.

Five commands, each one something an operator actually needs: start the
service, check that the configuration is coherent before a deploy, see what
the assistant can see, ask a question without an HTTP client, and run the
evaluation suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from nl2sql.config.loader import load_settings
from nl2sql.container import Container
from nl2sql.core.exceptions import AppError
from nl2sql.evaluation.dataset import load_dataset
from nl2sql.evaluation.report import write_report
from nl2sql.evaluation.runner import EvaluationRunner
from nl2sql.pipeline.orchestrator import QueryCommand
from nl2sql.security.auth import ROLE_ADMIN, ROLE_READER, Principal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nl2sql", description=__doc__)
    parser.add_argument("--env", default=None, help="Environment name, for example production.")
    parser.add_argument("--config-dir", default=None, help="Directory holding the YAML layers.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    sub.add_parser("check-config", help="Validate the configuration and report what it enables.")

    schema = sub.add_parser("schema", help="Show the tables the assistant can see.")
    schema.add_argument("--schema", default=None, help="Restrict to one schema.")
    schema.add_argument("--json", action="store_true", help="Print JSON instead of text.")

    ask = sub.add_parser("ask", help="Ask one question.")
    ask.add_argument("question")
    ask.add_argument("--validate-only", action="store_true")
    ask.add_argument("--tenant", default=None)

    evaluate = sub.add_parser("evaluate", help="Run an evaluation dataset.")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--reports-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface."""
    args = _parser().parse_args(argv)
    try:
        settings = load_settings(env=args.env, config_dir=args.config_dir)
    except AppError as exc:
        print(f"Configuration error: {exc.message}", file=sys.stderr)
        return 2

    if args.command == "serve":
        return _serve(args, settings)

    try:
        if args.command == "check-config":
            return _check_config(settings)
        if args.command == "schema":
            return _schema(args, settings)
        if args.command == "ask":
            return asyncio.run(_ask(args, settings))
        if args.command == "evaluate":
            return asyncio.run(_evaluate(args, settings))
    except AppError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    return 0


def _serve(args: argparse.Namespace, settings: Any) -> int:
    import uvicorn

    uvicorn.run(
        "nl2sql.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,
    )
    return 0


def _check_config(settings: Any) -> int:
    """Report what the configuration enables, without connecting anywhere."""
    print(json.dumps(settings.public_summary(), indent=2))
    container = Container.build(settings, configure_logs=False)
    print(f"prompts loaded: {', '.join(container.prompts.names())}")
    print(f"prompt versions: {json.dumps(container.prompts.active_versions())}")
    print(f"model providers configured: {', '.join(p.name for p in container.providers.all())}")
    print(f"model providers usable: {', '.join(container.providers.available_names()) or 'none'}")
    print(f"allowed statements: {', '.join(sorted(container.policy.allowed_statements))}")
    container.dispose()
    return 0


def _schema(args: argparse.Namespace, settings: Any) -> int:
    """Print the catalogue the assistant can see."""
    container = Container.build(settings, configure_logs=False)
    try:
        tables = container.metadata.list_tables(args.schema)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "schema": table.schema_name,
                            "name": table.name,
                            "kind": table.kind,
                            "description": table.comment,
                            "columns": [
                                {"name": c.name, "type": c.data_type} for c in table.columns
                            ],
                        }
                        for table in tables
                    ],
                    indent=2,
                )
            )
        else:
            for table in tables:
                print(f"{table.qualified_name} ({table.kind}, {len(table.columns)} columns)")
                if table.comment:
                    print(f"    {table.comment}")
        print(f"\n{len(tables)} table(s) visible.", file=sys.stderr)
    finally:
        container.dispose()
    return 0


async def _ask(args: argparse.Namespace, settings: Any) -> int:
    """Answer one question from the command line."""
    container = Container.build(settings, configure_logs=True)
    try:
        outcome = await container.pipeline.run(
            QueryCommand(
                question=args.question,
                principal=Principal(
                    id="cli", tenant_id=args.tenant, roles=(ROLE_READER, ROLE_ADMIN)
                ),
                validate_only=args.validate_only,
            )
        )
    finally:
        container.dispose()

    print(f"\nSQL:\n{outcome.sql}\n")
    if outcome.answer:
        print(f"Answer:\n{outcome.answer}\n")
    print(f"Confidence: {outcome.confidence}")
    if outcome.warnings:
        print("Warnings:")
        for warning in outcome.warnings:
            print(f"  - {warning}")
    if outcome.rows:
        header = " | ".join(column["name"] for column in outcome.columns)
        print(f"\n{header}")
        print("-" * len(header))
        for row in outcome.rows[:20]:
            print(" | ".join("" if value is None else str(value) for value in row))
        if len(outcome.rows) > 20:
            print(f"... {len(outcome.rows) - 20} more rows")
    return 0


async def _evaluate(args: argparse.Namespace, settings: Any) -> int:
    """Run an evaluation dataset and write the report."""
    container = Container.build(settings, configure_logs=True)
    try:
        cases = load_dataset(args.dataset)
        runner = EvaluationRunner(
            pipeline=container.pipeline,
            validator=container.validator,
            executor=container.executor,
            metadata=container.metadata,
            settings=settings.evaluation,
        )
        report = await runner.run(cases, dataset_name=str(Path(args.dataset).name))
        json_path, markdown_path = write_report(
            report, args.reports_dir or settings.evaluation.reports_dir
        )
    finally:
        container.dispose()

    print(json.dumps(report.aggregates, indent=2))
    print(f"\nReports written to {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
