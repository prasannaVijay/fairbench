# Installation

FAIRBench requires **Python 3.11+**.

## Install the package

```bash
pip install -e ".[dev]"
```

## Set API keys

Set keys for the services you want to use. The easiest approach is a `.env` file in the project root — FAIRBench auto-loads it.

```bash
# Required for text benchmarks (Claude)
export ANTHROPIC_API_KEY=sk-ant-...

# Required for image generation (DALL-E / gpt-image-1)
export OPENAI_API_KEY=sk-...

# VisionAnalyzer (Claude Vision) is always needed for image runs;
# it uses ANTHROPIC_API_KEY — already covered above.

# Optional: Stable Diffusion via HuggingFace Inference API
export HF_API_TOKEN=hf_...
```

Or in `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
HF_API_TOKEN=hf_...
```

## Verify the install

```bash
fairbench metrics        # lists the six metrics
fairbench scenarios      # lists built-in scenario sets
```

If both commands print output without errors, you are ready to run your first benchmark.
