# Image Benchmark Walkthrough

The image pipeline generates images from text prompts, analyses each image with Claude Vision and CLIP, then scores the run with the same six fairness metrics.

## 1. Run the built-in soccer benchmark

```bash
# Quick run — 1 scenario (~9 images), validates the whole pipeline
fairbench run soccer_player --modality image --model gpt-image-1 --html report.html

# Full run — all 8 scenarios (~60 images)
fairbench run soccer_player --modality image --model gpt-image-1 --html report.html --save-images ./images
```

Or with the dedicated `image-run` command (identical, just shorter):

```bash
fairbench image-run soccer_player --model gpt-image-1 --html report.html
```

## 2. What the soccer benchmark measures

The `soccer_player` scenario set has 8 scenarios covering players, coaches, referees, youth soccer, and team photos. Each scenario has counterfactuals across gender, race, nationality, socioeconomic setting, and age.

Known biases this benchmark surfaces:

- **Gender default** — neutral prompt "a soccer player" generates male images at 70–80%
- **Kit quality disparity** — female player images show unbranded kits in professional stadiums more often than male player images
- **Setting disparity** — African/South American players are depicted in informal settings more than European players
- **Non-binary erasure** — "non-binary player" prompts map to binary gender presentations

## 3. Use a different image model

```bash
# Stable Diffusion XL via HuggingFace Inference API (requires HF_API_TOKEN)
fairbench image-run soccer_player \
  --model sd:stabilityai/stable-diffusion-xl-base-1.0 \
  --html report.html

# Local Stable Diffusion (requires: pip install diffusers torch)
fairbench image-run soccer_player \
  --model sd-local:stabilityai/stable-diffusion-xl-base-1.0 \
  --html report.html
```

## 4. Python API for image evaluation

```python
import asyncio
from fairbench.adapters.image.dalle import DALLEAdapter
from fairbench.core.image_engine import ImageBenchEngine
from fairbench.evaluation.image.vision_analyzer import VisionAnalyzer
from fairbench.evaluation.image.clip_evaluator import CLIPEvaluator
from fairbench.reporting.html_report import generate_html_report

async def main():
    engine = ImageBenchEngine()
    engine.scenario_registry.load_file("src/fairbench/scenarios/image/soccer_player.yaml")

    run = await engine.evaluate(
        model=DALLEAdapter(model="gpt-image-1"),
        scenarios=["soccer_player"],
        vision_analyzer=VisionAnalyzer(model="claude-sonnet-4-6"),
        clip_evaluator=CLIPEvaluator(model_name="ViT-B/32"),
        concurrency=3,
    )
    scorecard = engine.generate_scorecard(run)
    open("report.html", "w").write(generate_html_report(scorecard))

asyncio.run(main())
```

---

## CLI reference — image-specific flags

```
fairbench image-run <scenario> [OPTIONS]

  --model          'gpt-image-1' | 'sd:<hf-id>' | 'sd-local:<hf-id>'
  --html           Render HTML report to this path
  --save-images    Directory to save generated images
  --vision-model   Claude model for VisionAnalyzer (default: claude-sonnet-4-6)
  --size           Image size (default: 1024x1024)
  --quality        gpt-image-1: low|medium|high|auto (default: auto)
  --no-clip        Skip CLIP evaluation
  --concurrency    Max concurrent API calls (default: 3)
```
