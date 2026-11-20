# Contributing to experai

A small language-model training toolkit in Rust using the Candle ML framework, with hardware-aware training (CUDA/Metal/CPU), data loading from any format, AT Protocol ingestion (Bluesky REST + Jetstream), and one-command packaging to GGUF/Ollama/OpenCode. Thanks for wanting to help!

## Getting started

```bash
# NVIDIA + CUDA (default)
cargo build --release
# Apple Silicon
cargo build --release --no-default-features --features metal
# CPU only
cargo build --release --no-default-features
cargo test
```

macOS note: the default `cuda` feature will not build there — use `--no-default-features --features metal`.

The tokenizer (`models/tokenizer.json`) is required for every command and must be a valid HuggingFace `tokenizers` JSON. `models/` is gitignored, so run `experai fetch --model gpt2 --what all` on your machine.

## Notes for this project

- **Hardware-awareness is the core value**: auto-detection of GPU/CPU, automatic batch-size/precision/memory tuning, and bf16/fp16/fp32 selection based on hardware. Changes must not silently assume one backend; keep the CUDA, Metal and CPU paths compiling and behaving.
- The models directory is deliberately un-tracked: never commit tokenizers, checkpoints or other fetched model assets.
- Provider data comes in through auto-detection (JSONL/JSON/CSV/TSV/text/directories/stdin) — keep auto-detection cheap and safe, with `--data-format` as the escape hatch.
- Jetstream backfill semantics (cursor handling, clamped replay windows, `--max-duration-secs`) are documented behaviour: change them deliberately and update the README in the same PR.

## Before opening a PR

- The documented build and tests pass, and you ran them yourself.
- Conventional commits; one focused PR per change.
- AI-assisted PRs are welcome; stand behind every line.

## Licence

AGPL-3.0 (see [LICENSE](LICENSE)). Contributions are accepted under the same terms.