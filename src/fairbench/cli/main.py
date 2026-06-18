"""Main CLI entry point for FAIRBench."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="fairbench",
    help="FAIRBench: A fairness benchmarking framework for generative AI",
    no_args_is_help=True,
)

console = Console()


@app.command()
def run(
    scenario: str = typer.Argument(
        ...,
        help=(
            "Scenario set name (built-in), path to a scenario YAML, "
            "or path to a benchmark spec YAML (detected by 'model_under_test' key)"
        ),
    ),
    model: str = typer.Option(
        "anthropic",
        "--model", "-m",
        help="Model adapter to use — ignored when a benchmark spec is provided",
    ),
    metrics: Optional[str] = typer.Option(
        None, "--metrics", help="Comma-separated list of metrics (default: all)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output directory for scorecard files (defaults to ./reports)",
    ),
    output_format: str = typer.Option(
        "all",
        "--output_format",
        help="Scorecard format: json | md | all  (default: all)",
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", "-c", help="Max concurrent API calls"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
) -> None:
    """Run a fairness evaluation on a model.

    Pass a benchmark spec YAML (containing a 'model_under_test' key) to drive
    the full evaluation in a single file.  Pass a scenario name or scenario YAML
    for the classic two-argument flow where --model selects the adapter.
    """
    if output_format not in ("json", "md", "all"):
        console.print(f"[red]--output_format must be one of: json, md, all[/red]")
        raise typer.Exit(1)

    asyncio.run(
        _run_evaluation(
            scenario=scenario,
            model=model,
            metrics=metrics,
            output=output,
            output_format=output_format,
            concurrency=concurrency,
            verbose=verbose,
        )
    )


async def _run_evaluation(
    scenario: str,
    model: str,
    metrics: Optional[str],
    output: Optional[Path],
    output_format: str,
    concurrency: int,
    verbose: bool,
) -> None:
    """Execute the evaluation — handles both benchmark specs and plain scenario runs."""
    from fairbench.adapters.anthropic import AnthropicAdapter
    from fairbench.adapters.openai import OpenAIAdapter
    from fairbench.core.benchmark import is_benchmark_spec, load_benchmark_spec
    from fairbench.core.engine import FairBenchEngine

    # -----------------------------------------------------------------------
    # Determine whether this is a benchmark spec or a plain scenario run
    # -----------------------------------------------------------------------
    spec = None
    benchmark_name = "FAIRBench Audit"

    if Path(scenario).exists() and is_benchmark_spec(scenario):
        spec = load_benchmark_spec(scenario)
        benchmark_name = spec.benchmark.name
        console.print(f"[bold blue]FAIRBench Evaluation[/bold blue]  (benchmark spec)")
        console.print(f"  Benchmark : {spec.benchmark.name}")
        console.print(f"  Model     : {spec.model_under_test.model} ({spec.model_under_test.provider})")
        console.print(f"  Scenarios : {', '.join(spec.scenarios)}")
        console.print()
    else:
        console.print(f"[bold blue]FAIRBench Evaluation[/bold blue]")
        console.print(f"  Scenario  : {scenario}")
        console.print(f"  Model     : {model}")
        console.print()

    # -----------------------------------------------------------------------
    # Resolve effective parameters from spec or CLI args
    # -----------------------------------------------------------------------
    engine = FairBenchEngine()

    if spec is not None:
        # --- Model adapter from spec ---
        mut = spec.model_under_test
        try:
            adapter = _build_adapter(mut.provider, mut.model, mut.api_key, mut.base_url)
        except Exception as e:
            console.print(f"[red]Error setting up model adapter from spec: {e}[/red]")
            raise typer.Exit(1)

        effective_model_key = f"{mut.provider}/{mut.model}"
        engine.register_adapter(effective_model_key, adapter)

        # --- Scenarios from spec ---
        resolved_scenarios = spec.resolve_scenario_paths()
        scenario_names: list[str] = []
        for entry in resolved_scenarios:
            if Path(entry).exists():
                loaded = engine.scenario_registry.load_file(entry)
                scenario_names.append(loaded.name)
            else:
                scenario_names.append(entry)

        # --- Metrics from spec (CLI --metrics overrides) ---
        metric_list = (
            [m.strip() for m in metrics.split(",")]
            if metrics
            else spec.metrics
        )

        # --- Output directory from spec (CLI --output overrides) ---
        effective_output_dir = output or Path(spec.output.path)
        # Output format: CLI flag overrides spec default
        effective_format = output_format if output_format != "all" else spec.output.format

        # Store benchmark name in run config for scorecard header
        engine.config = engine.config.model_copy(update={})  # keep frozen; pass via snapshot below
        extra_snapshot = {"benchmark_name": spec.benchmark.name}

    else:
        # --- Classic flow: model from --model flag, scenario from argument ---
        effective_model_key = model
        try:
            if model == "anthropic" or model.startswith("claude"):
                adapter = AnthropicAdapter(
                    model=model if model.startswith("claude") else None
                )
            elif model == "openai" or model.startswith("gpt"):
                adapter = OpenAIAdapter(
                    model=model if model.startswith("gpt") else None
                )
            else:
                adapter = engine.adapter_registry.get(model)
            engine.register_adapter(model, adapter)
        except Exception as e:
            console.print(f"[red]Error setting up model adapter: {e}[/red]")
            raise typer.Exit(1)

        if Path(scenario).exists():
            loaded = engine.scenario_registry.load_file(scenario)
            scenario_names = [loaded.name]
        else:
            scenario_names = [scenario]

        metric_list = [m.strip() for m in metrics.split(",")] if metrics else None
        effective_output_dir = output or Path("./reports")
        effective_format = output_format
        extra_snapshot = {}

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    console.print("[yellow]Running evaluation...[/yellow]")
    try:
        with console.status("[bold green]Generating and evaluating outputs..."):
            result = await engine.evaluate(
                model=effective_model_key,
                scenarios=scenario_names,
                metrics=metric_list,
                concurrency=concurrency,
                save_run=True,
            )
        # Attach extra snapshot info (benchmark name etc.) post-hoc
        if extra_snapshot:
            updated_snapshot = {**result.config_snapshot, **extra_snapshot}
            from fairbench.core.types import EvaluationRun
            result = EvaluationRun(
                **{**result.model_dump(), "config_snapshot": updated_snapshot}
            )
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

    # -----------------------------------------------------------------------
    # Display summary in terminal
    # -----------------------------------------------------------------------
    console.print()
    console.print("[bold green]Evaluation Complete![/bold green]")
    console.print(f"  Run ID  : {result.id}")
    console.print(f"  Status  : {result.status.value}")
    console.print(f"  Outputs : {len(result.outputs)}")
    console.print()

    if result.metric_results:
        table = Table(title="Fairness Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Samples", style="green")
        table.add_column("Band / Interpretation", style="yellow")
        for mr in result.metric_results:
            table.add_row(
                mr.metric_name,
                f"{mr.value:.4f}",
                str(mr.n_samples),
                mr.interpretation or "",
            )
        console.print(table)

    # -----------------------------------------------------------------------
    # Write scorecard file(s)
    # -----------------------------------------------------------------------
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = benchmark_name.lower().replace(" ", "_").replace("/", "-")[:60]
    date_str = __import__("datetime").date.today().isoformat()
    base_stem = f"{safe_name}_{date_str}"

    _write_scorecards(
        result=result,
        output_dir=effective_output_dir,
        base_stem=base_stem,
        fmt=effective_format,
        benchmark_name=benchmark_name,
    )

    await engine.close()


def _build_adapter(provider: str, model_name: str, api_key: Optional[str], base_url: Optional[str]):
    """Instantiate a model adapter from benchmark spec parameters."""
    from fairbench.adapters.anthropic import AnthropicAdapter
    from fairbench.adapters.dalle import DallE3Adapter
    from fairbench.adapters.http_webhook import HttpWebhookAdapter
    from fairbench.adapters.openai import OpenAIAdapter
    from fairbench.adapters.openai_compatible import OpenAICompatibleAdapter

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key

    if provider == "anthropic":
        return AnthropicAdapter(model=model_name, **kwargs)
    elif provider == "openai":
        return OpenAIAdapter(model=model_name, **kwargs)
    elif provider in ("dalle", "openai_image"):
        # DALL-E 3 + GPT-4o Vision captioning pipeline
        return DallE3Adapter(model=model_name, **kwargs)
    elif provider == "openai_compatible":
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAICompatibleAdapter(model=model_name, **kwargs)
    elif provider == "http_webhook":
        if base_url:
            kwargs["url"] = base_url
        return HttpWebhookAdapter(model=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider in benchmark spec: '{provider}'")


def _write_scorecards(
    result,
    output_dir: Path,
    base_stem: str,
    fmt: str,
    benchmark_name: str,
) -> None:
    """Write JSON and/or Markdown scorecards to output_dir."""
    from fairbench.reporting.markdown import generate_markdown_scorecard
    from fairbench.reporting.scorecard import generate_scorecard

    wrote: list[str] = []

    if fmt in ("json", "all"):
        sc = generate_scorecard(result)
        path = output_dir / f"{base_stem}.json"
        path.write_text(json.dumps(sc, indent=2))
        wrote.append(str(path))

    if fmt in ("md", "all"):
        md = generate_markdown_scorecard(result, benchmark_name=benchmark_name)
        path = output_dir / f"{base_stem}.md"
        path.write_text(md)
        wrote.append(str(path))

    if wrote:
        console.print()
        for p in wrote:
            console.print(f"[green]Scorecard written:[/green] {p}")


@app.command()
def scenarios(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show scenario details"),
) -> None:
    """List available scenario sets."""
    from fairbench.scenarios.registry import get_registry

    registry = get_registry()
    sets = registry.list_sets()

    if not sets:
        console.print("[yellow]No scenario sets loaded.[/yellow]")
        console.print("Load scenarios with: fairbench run <path-to-scenario.yaml>")
        return

    table = Table(title="Available Scenario Sets")
    table.add_column("Name", style="cyan")
    table.add_column("Scenarios", style="green")
    table.add_column("Dimensions", style="yellow")

    for name in sets:
        scenario_set = registry.get_set(name)
        dims = ", ".join(d.value for d in scenario_set.dimensions)
        table.add_row(name, str(len(scenario_set.scenarios)), dims)

    console.print(table)

    if verbose:
        console.print()
        for name in sets:
            scenario_set = registry.get_set(name)
            console.print(f"[bold]{name}[/bold]: {scenario_set.description or 'No description'}")
            for s in scenario_set.scenarios:
                console.print(f"  - {s.id}: {s.prompt[:50]}...")


@app.command()
def metrics() -> None:
    """List available fairness metrics."""
    from fairbench.core.engine import FairBenchEngine

    engine = FairBenchEngine()
    available = engine.get_available_metrics()

    table = Table(title="Available Fairness Metrics")
    table.add_column("Metric", style="cyan", width=10)
    table.add_column("Description", style="white")

    for name, description in available.items():
        table.add_row(name, description)

    console.print(table)


@app.command()
def runs(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of runs to show"),
) -> None:
    """List recent evaluation runs."""
    asyncio.run(_list_runs(limit))


async def _list_runs(limit: int) -> None:
    """List recent runs."""
    from fairbench.core.engine import FairBenchEngine

    engine = FairBenchEngine()
    run_list = await engine.list_runs(limit=limit)

    if not run_list:
        console.print("[yellow]No evaluation runs found.[/yellow]")
        await engine.close()
        return

    table = Table(title="Recent Evaluation Runs")
    table.add_column("Run ID", style="cyan", width=36)
    table.add_column("Status", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Created", style="white")

    for run in run_list:
        table.add_row(
            run.run_id[:36],
            run.status.value,
            run.model_name,
            run.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    await engine.close()


@app.command()
def show(
    run_id: str = typer.Argument(..., help="Run ID to display"),
) -> None:
    """Show details of an evaluation run."""
    asyncio.run(_show_run(run_id))


async def _show_run(run_id: str) -> None:
    """Display run details."""
    from fairbench.core.engine import FairBenchEngine

    engine = FairBenchEngine()
    run = await engine.get_run(run_id)

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        await engine.close()
        return

    console.print(f"[bold]Run ID:[/bold] {run.id}")
    console.print(f"[bold]Status:[/bold] {run.status.value}")
    console.print(f"[bold]Model:[/bold] {run.model_info.name} ({run.model_info.provider})")
    console.print(f"[bold]Scenarios:[/bold] {', '.join(run.scenario_sets)}")
    console.print(f"[bold]Created:[/bold] {run.created_at}")
    if run.completed_at:
        console.print(f"[bold]Completed:[/bold] {run.completed_at}")
    console.print(f"[bold]Outputs:[/bold] {len(run.outputs)}")
    console.print()

    if run.metric_results:
        table = Table(title="Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Interpretation", style="yellow")

        for metric in run.metric_results:
            table.add_row(
                metric.metric_name,
                f"{metric.value:.4f}",
                metric.interpretation or "",
            )

        console.print(table)

    if run.error_message:
        console.print(f"\n[red]Error:[/red] {run.error_message}")

    await engine.close()


@app.command()
def scorecard(
    run_id: str = typer.Argument(..., help="Run ID to generate scorecard for"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (JSON). Prints to stdout if omitted."
    ),
) -> None:
    """Generate a JSON scorecard for a completed evaluation run."""
    asyncio.run(_generate_scorecard(run_id, output))


async def _generate_scorecard(run_id: str, output: Optional[Path]) -> None:
    """Load run and produce scorecard."""
    import json

    from fairbench.core.engine import FairBenchEngine
    from fairbench.reporting.scorecard import generate_scorecard

    engine = FairBenchEngine()
    run = await engine.get_run(run_id)
    await engine.close()

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    card = generate_scorecard(run)
    json_str = json.dumps(card, indent=2)

    if output:
        output.write_text(json_str)
        console.print(f"[green]Scorecard saved to: {output}[/green]")
    else:
        console.print(json_str)


@app.command()
def init(
    path: Path = typer.Option(
        Path("fairbench.yaml"), "--output", "-o", help="Output path for config"
    ),
) -> None:
    """Initialize a FAIRBench configuration file."""
    if path.exists():
        overwrite = typer.confirm(f"{path} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit(0)

    config_content = '''# FAIRBench Configuration
fairbench:
  version: "1.0"
  log_level: INFO

  storage:
    backend: sqlite
    sqlite_path: ~/.fairbench/fairbench.db

  evaluation:
    concurrency: 10
    timeout_seconds: 60
    retry_attempts: 3

  metrics:
    default_baseline: uniform

  models:
    anthropic:
      model: claude-sonnet-4-20250514
      max_tokens: 1024
    openai:
      model: gpt-4o
      max_tokens: 1024

  reporting:
    output_dir: ./reports
    formats:
      - json
      - html
'''

    path.write_text(config_content)
    console.print(f"[green]Configuration file created: {path}[/green]")


if __name__ == "__main__":
    app()
