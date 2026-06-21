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
        "anthropic", "--model", "-m",
        help="Text: 'anthropic'|'openai'|'claude-*'|'gpt-*'. Image: 'gpt-image-1'|'sd:<hf-id>'",
    ),
    modality: str = typer.Option(
        "text", "--modality",
        help="'text' (default) or 'image'. Selects the evaluation pipeline.",
    ),
    metrics: Optional[str] = typer.Option(
        None, "--metrics", help="Comma-separated list of metrics (default: all)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save results to this JSON file"
    ),
    html: Optional[Path] = typer.Option(
        None, "--html", help="Also render an HTML report to this path"
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", "-c", help="Max concurrent API calls"
    ),
    # Image-only options (ignored for text)
    vision_model: str = typer.Option(
        "claude-sonnet-4-6", "--vision-model",
        help="[image] Claude model for VisionAnalyzer",
    ),
    size: str = typer.Option(
        "1024x1024", "--size", help="[image] Image size"
    ),
    quality: str = typer.Option(
        "auto", "--quality", help="[image] gpt-image-1 quality: low|medium|high|auto"
    ),
    no_clip: bool = typer.Option(
        False, "--no-clip", help="[image] Skip CLIP evaluation"
    ),
    save_images: Optional[Path] = typer.Option(
        None, "--save-images", help="[image] Directory to save generated images"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
) -> None:
    """Run a fairness evaluation — text or image generation.

    Text examples:
        fairbench run gender_occupation --model anthropic
        fairbench run gender_occupation --model openai --output results.json --html report.html

    Image examples:
        fairbench run soccer_player --modality image --model gpt-image-1
        fairbench run soccer_player --modality image --model gpt-image-1 --quality high --html report.html
        fairbench run soccer_player --modality image --model sd:stabilityai/stable-diffusion-xl-base-1.0
    """
    if modality == "image":
        asyncio.run(
            _run_image_evaluation(
                scenario, model, vision_model, None, size, quality,
                save_images, output, html, concurrency, no_clip, verbose,
            )
        )
    else:
        asyncio.run(_run_evaluation(scenario, model, metrics, output, html, concurrency, verbose))


async def _run_evaluation(
    scenario: str,
    model: str,
    metrics: Optional[str],
    output: Optional[Path],
    html: Optional[Path],
    concurrency: int,
    verbose: bool,
) -> None:
    """Execute the text evaluation."""
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

        # --- Output settings from spec (CLI --output / --html override) ---
        spec_output_dir = Path(spec.output.path)
        spec_output_format = spec.output.format  # "json" | "md" | "all"
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
        spec_output_dir = None
        spec_output_format = "json"  # unused for non-spec runs; output written via --output / --html
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

    # If a benchmark spec was used, write scorecards to the spec's output directory
    if spec is not None and spec_output_dir is not None:
        spec_output_dir.mkdir(parents=True, exist_ok=True)
        stem = result.id
        _write_scorecards(result, spec_output_dir, str(stem), spec_output_format, benchmark_name)

    # CLI --output flag: write JSON to the specified path
    if output:
        output_data = {
            "run_id": str(result.id),
            "model": result.model_info.model_dump(),
            "scenarios": result.scenario_sets,
            "metrics": [m.model_dump(mode="json") for m in result.metric_results],
        }
        output.write_text(json.dumps(output_data, indent=2))
        console.print(f"\nResults saved to: {output}")

    # CLI --html flag: render HTML report
    if html:
        from fairbench.reporting.html_report import generate_html_report
        from fairbench.reporting.scorecard import generate_scorecard
        scorecard_data = generate_scorecard(result)
        html.write_text(generate_html_report(scorecard_data))
        console.print(f"HTML report saved to: {html}")

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
    html: Optional[Path] = typer.Option(
        None, "--html", help="Also render an HTML report to this path."
    ),
) -> None:
    """Generate a scorecard for a completed evaluation run (JSON + optional HTML)."""
    asyncio.run(_generate_scorecard(run_id, output, html))


async def _generate_scorecard(run_id: str, output: Optional[Path], html: Optional[Path]) -> None:
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

    if html:
        from fairbench.reporting.html_report import generate_html_report
        html.write_text(generate_html_report(card))
        console.print(f"[green]HTML report saved to: {html}[/green]")


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


@app.command(name="image-run")
def image_run(
    scenario: str = typer.Argument(
        ..., help="Scenario set name or path to a scenario YAML file"
    ),
    model: str = typer.Option(
        "gpt-image-1", "--model", "-m",
        help="Image model: 'gpt-image-1' | 'sd:<hf-model-id>' | 'sd-local:<hf-model-id>'"
    ),
    vision_model: str = typer.Option(
        "claude-sonnet-4-6", "--vision-model",
        help="Claude model for VisionAnalyzer analysis"
    ),
    size: str = typer.Option(
        "1024x1024", "--size", help="Image size"
    ),
    quality: str = typer.Option(
        "auto", "--quality", help="gpt-image-1 quality: low|medium|high|auto"
    ),
    save_images: Optional[Path] = typer.Option(
        None, "--save-images", help="Directory to save generated images"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save scorecard JSON to this path"
    ),
    html: Optional[Path] = typer.Option(
        None, "--html", help="Also render an HTML report to this path"
    ),
    concurrency: int = typer.Option(
        3, "--concurrency", "-c", help="Max concurrent image generation calls"
    ),
    no_clip: bool = typer.Option(
        False, "--no-clip", help="Skip CLIP evaluation (faster, no openai-clip dependency)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tracebacks on error"),
) -> None:
    """Run an image generation fairness benchmark.

    Examples:
        fairbench image-run soccer_player --model gpt-image-1
        fairbench image-run soccer_player --model gpt-image-1 --html report.html
        fairbench image-run soccer_player --model sd:stabilityai/stable-diffusion-xl-base-1.0
    """
    asyncio.run(
        _run_image_evaluation(
            scenario, model, vision_model, None, size, quality,
            save_images, output, html, concurrency, no_clip, verbose,
        )
    )


async def _run_image_evaluation(
    scenario: str,
    model_name: str,
    vision_model: str,
    clip_model: str | None,
    size: str,
    quality: str,
    save_images: Optional[Path],
    output: Optional[Path],
    html: Optional[Path],
    concurrency: int,
    no_clip: bool,
    verbose: bool,
) -> None:
    """Execute the image evaluation."""
    import json
    from pathlib import Path as P

    from fairbench.adapters.image.dalle import DALLEAdapter
    from fairbench.adapters.image.stable_diffusion import StableDiffusionAdapter
    from fairbench.core.image_engine import ImageBenchEngine
    from fairbench.core.image_types import ImageGenerationConfig
    from fairbench.evaluation.image.clip_evaluator import CLIPEvaluator
    from fairbench.evaluation.image.vision_analyzer import VisionAnalyzer

    console.print("[bold blue]FAIRBench Image Evaluation[/bold blue]")
    console.print(f"  Scenario : {scenario}")
    console.print(f"  Model    : {model_name}")
    console.print(f"  Vision   : {vision_model}")
    console.print(f"  CLIP     : {'disabled' if no_clip else clip_model}")
    console.print()

    # ── Build image adapter ──────────────────────────────────────────────────
    try:
        save_dir = str(save_images) if save_images else None
        if model_name in ("dalle3", "dall-e-3", "gpt-image-1"):
            adapter = DALLEAdapter(model="gpt-image-1", save_dir=save_dir)
        elif model_name in ("dalle2", "dall-e-2"):
            adapter = DALLEAdapter(model="dall-e-2", save_dir=save_dir)
        elif model_name.startswith("sd-local:"):
            hf_id = model_name.split("sd-local:", 1)[1]
            adapter = StableDiffusionAdapter(model=hf_id, backend="local", save_dir=save_dir)
        elif model_name.startswith("sd:"):
            hf_id = model_name.split("sd:", 1)[1]
            adapter = StableDiffusionAdapter(model=hf_id, backend="hf_api", save_dir=save_dir)
        else:
            console.print(f"[red]Unknown model: {model_name!r}[/red]")
            console.print("Valid options: dalle3, dalle2, sd:<hf-model-id>, sd-local:<hf-model-id>")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error building adapter: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

    # ── Build evaluators ─────────────────────────────────────────────────────
    try:
        vision_analyzer = VisionAnalyzer(model=vision_model)
        clip_evaluator = None if no_clip else CLIPEvaluator(model_name=clip_model)
    except Exception as e:
        console.print(f"[red]Error building evaluators: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

    # ── Load scenarios ───────────────────────────────────────────────────────
    engine = ImageBenchEngine()
    scenario_path = P(scenario)
    if scenario_path.exists():
        engine.scenario_registry.load_file(str(scenario_path))
        scenario_name = scenario_path.stem
    else:
        # Try built-in image scenarios first
        builtin_image = (
            P(__file__).parent.parent / "scenarios" / "image" / f"{scenario}.yaml"
        )
        if builtin_image.exists():
            engine.scenario_registry.load_file(str(builtin_image))
            scenario_name = scenario
        else:
            scenario_name = scenario  # Hope it's already registered

    metric_list = [m.strip() for m in metrics.split(",")] if metrics else None
    gen_config = ImageGenerationConfig(size=size, quality=quality)

    # ── Run ──────────────────────────────────────────────────────────────────
    console.print("[yellow]Running image generation and analysis…[/yellow]")
    try:
        with console.status("[bold green]Generating images and evaluating fairness…"):
            run = await engine.evaluate(
                model=adapter,
                scenarios=[scenario_name],
                vision_analyzer=vision_analyzer,
                clip_evaluator=clip_evaluator,
                metrics=metric_list,
                generation_config=gen_config,
                concurrency=concurrency,
            )
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

    # ── Results ──────────────────────────────────────────────────────────────
    console.print()
    console.print("[bold green]Image Evaluation Complete![/bold green]")
    console.print(f"  Run ID        : {run.id}")
    console.print(f"  Total images  : {run.total_images()}")
    console.print(f"  Refused       : {run.refused_count()}")
    console.print()

    if run.metric_results:
        table = Table(title="Image Fairness Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Samples", style="green")
        table.add_column("Interpretation", style="yellow")

        for mr in run.metric_results:
            table.add_row(
                mr.metric_name,
                f"{mr.value:.4f}" if mr.value == mr.value else "N/A",
                str(mr.n_samples),
                mr.interpretation or "",
            )
        console.print(table)

    # Gender breakdown across all images
    gender_counts: dict[str, int] = {}
    for ei in run.evaluated_images:
        if ei.vision_analysis:
            g = ei.vision_analysis.perceived_gender
            gender_counts[g] = gender_counts.get(g, 0) + 1

    if gender_counts:
        total = sum(gender_counts.values())
        console.print()
        console.print("[bold]Gender representation (across all generated images):[/bold]")
        for g, count in sorted(gender_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total
            console.print(f"  {g:15s}: {count:3d} ({pct:.1f}%)")

    # ── Scorecard ────────────────────────────────────────────────────────────
    scorecard = engine.generate_scorecard(run)

    if output:
        output.write_text(json.dumps(scorecard, indent=2))
        console.print(f"\n[green]Scorecard saved to: {output}[/green]")

    if html:
        from fairbench.reporting.html_report import generate_html_report
        html.write_text(generate_html_report(scorecard))
        console.print(f"[green]HTML report saved to: {html}[/green]")


if __name__ == "__main__":
    app()
