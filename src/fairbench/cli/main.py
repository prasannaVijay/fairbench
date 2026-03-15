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
        ..., help="Scenario set name or path to scenario file"
    ),
    model: str = typer.Option(
        "anthropic", "--model", "-m", help="Model adapter to use"
    ),
    metrics: Optional[str] = typer.Option(
        None, "--metrics", help="Comma-separated list of metrics (default: all)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for results (JSON)"
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", "-c", help="Max concurrent API calls"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
) -> None:
    """Run a fairness evaluation on a model."""
    asyncio.run(_run_evaluation(scenario, model, metrics, output, concurrency, verbose))


async def _run_evaluation(
    scenario: str,
    model: str,
    metrics: Optional[str],
    output: Optional[Path],
    concurrency: int,
    verbose: bool,
) -> None:
    """Execute the evaluation."""
    from fairbench.adapters.anthropic import AnthropicAdapter
    from fairbench.adapters.openai import OpenAIAdapter
    from fairbench.core.engine import FairBenchEngine

    console.print(f"[bold blue]FAIRBench Evaluation[/bold blue]")
    console.print(f"  Scenario: {scenario}")
    console.print(f"  Model: {model}")
    console.print()

    # Initialize engine
    engine = FairBenchEngine()

    # Load scenarios
    if Path(scenario).exists():
        # Load from file
        engine.scenario_registry.load_file(scenario)
        scenario_name = Path(scenario).stem
    else:
        # Use built-in or already loaded
        scenario_name = scenario

    # Set up model adapter
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
            # Try to get from registry
            adapter = engine.adapter_registry.get(model)

        engine.register_adapter(model, adapter)
    except Exception as e:
        console.print(f"[red]Error setting up model adapter: {e}[/red]")
        raise typer.Exit(1)

    # Parse metrics
    metric_list = None
    if metrics:
        metric_list = [m.strip() for m in metrics.split(",")]

    # Run evaluation
    console.print("[yellow]Running evaluation...[/yellow]")

    try:
        with console.status("[bold green]Generating and evaluating outputs..."):
            result = await engine.evaluate(
                model=model,
                scenarios=[scenario_name],
                metrics=metric_list,
                concurrency=concurrency,
                save_run=True,
            )
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

    # Display results
    console.print()
    console.print("[bold green]Evaluation Complete![/bold green]")
    console.print(f"  Run ID: {result.id}")
    console.print(f"  Status: {result.status.value}")
    console.print(f"  Outputs evaluated: {len(result.outputs)}")
    console.print()

    # Metrics table
    if result.metric_results:
        table = Table(title="Fairness Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Samples", style="green")
        table.add_column("Interpretation", style="yellow")

        for metric in result.metric_results:
            table.add_row(
                metric.metric_name,
                f"{metric.value:.4f}",
                str(metric.n_samples),
                metric.interpretation or "",
            )

        console.print(table)

    # Save output
    if output:
        output_data = {
            "run_id": str(result.id),
            "model": result.model_info.model_dump(),
            "scenarios": result.scenario_sets,
            "metrics": [m.model_dump(mode="json") for m in result.metric_results],
        }
        output.write_text(json.dumps(output_data, indent=2))
        console.print(f"\nResults saved to: {output}")

    await engine.close()


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
