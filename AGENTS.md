# AGENTS.md

# Experai Project Guide

## Project Overview
Experai is a small language model training toolkit built in Rust using the Candle ML framework with CUDA acceleration. It provides CLI commands for training, preprocessing, and text generation.

## Repository Structure
```
experai
├── Cargo.toml
├── AGENTS.md
├── README.md
├── LICENSE
├── .gitignore
└── src
    ├── main.rs           # CLI entry point with clap subcommands
    ├── lib.rs            # Module declarations and public exports
    ├── data.rs           # Dataset loading, tokenization, batching
    ├── model.rs          # Model architecture definitions (transformer, embeddings, heads)
    ├── preprocessing.rs  # Text cleaning, tokenization pipeline, data collation
    ├── training.rs       # Training loop, loss functions, optimization, checkpointing
    ├── utils.rs          # Shared utilities: logging, metrics, device management
    └── logging
        └── mod.rs        # Structured logging configuration
```

## Agent Tasks

### 1. Model Development (`model.rs`)
- Implement transformer-based architectures using `candle-transformers`
- Define model configs: hidden size, layers, heads, vocab size, max seq length
- Support loading pretrained weights from Hugging Face Hub (`hf-hub` crate)
- Implement custom layers for niche NLP tasks (e.g., rotary embeddings, ALiBi)
- Ensure all operations use CUDA-accelerated `candle-nn` backend
- Add model serialization/deserialization via `safetensors`

### 2. Data Pipeline (`data.rs`, `preprocessing.rs`)
- Implement dataset loading from multiple formats: JSONL, CSV, Hugging Face datasets
- Build tokenization pipeline using `tokenizers` crate (BPE/WordPiece)
- Handle dynamic padding with `DataCollatorForLanguageModeling` equivalent
- Implement train/validation/test splits with reproducible shuffling
- Support streaming large datasets without full memory load
- Add data validation: sequence length checks, vocab coverage, duplicate detection

### 3. Training Workflow (`training.rs`, `main.rs` CLI)
- Implement training loop with gradient accumulation, mixed precision (bf16/fp16)
- Support learning rate scheduling: cosine, linear warmup, constant
- Add gradient clipping, weight decay, optimizer configuration (AdamW)
- Implement checkpointing: save best model, periodic saves, resume from checkpoint
- Add evaluation loop with perplexity, loss metrics
- Support distributed training basics (single GPU, data parallel)
- CLI flags in `main.rs`:
  - `train`: `--model`, `--data`, `--epochs`, `--lr`, `--batch-size`, `--grad-accum`, `--precision`, `--output-dir`
  - `preprocess`: `--input`, `--output`, `--tokenizer`, `--max-length`, `--split-ratio`
  - `generate`: `--model`, `--prompt`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`

### 4. CLI Operations (`main.rs`)
- Use `clap` derive API with subcommands and nested arg groups
- Validate all inputs: file existence, numeric ranges, compatible flag combinations
- Provide helpful error messages with suggested fixes
- Support config file (TOML) for complex training runs
- Add `--dry-run` flag for configuration validation without execution

### 5. Utilities & Infrastructure (`utils.rs`, `logging/`)
- Structured logging with `tracing` + `tracing-subscriber` (JSON for production, pretty for dev)
- Device management: auto-detect CUDA/Metal/CPU, fallback logic
- Metrics collection: loss curves, throughput, memory usage
- Random seed management for reproducibility
- Helper functions: tensor utilities, shape validation, memory profiling

## Implementation Invariants

### Memory Safety
- Validate all tensor shapes before operations; use `candle-core`'s shape checking
- Prevent OOM: implement gradient checkpointing for large models, streaming dataloaders
- Use `half::bf16`/`half::f16` explicitly for mixed precision; avoid accidental `f32` promotion
- Free intermediate tensors promptly; avoid retaining computation graphs longer than needed

### Numerical Stability
- Gradient clipping at 1.0 norm before optimizer step
- Loss scaling for fp16: dynamic loss scale with overflow detection
- Initialize weights with proper schemes (Xavier, Kaiming) per layer type
- Use `LayerNorm`/`RMSNorm` with epsilon ≥ 1e-6

### Reproducibility
- Seed all RNGs: `rand`, `candle`, CUDA (via `CUBLASLT_LOG_LEVEL=0`)
- Deterministic algorithms where available (`candle` CUDA kernels)
- Log exact software versions: `candle-core`, `candle-nn`, `candle-transformers`, CUDA driver, Rust toolchain
- Store training config (hyperparameters, data splits) alongside checkpoints

### Data Integrity
- Validate tokenized sequences: no OOV tokens beyond special tokens, length ≤ model max
- Check dataset for empty sequences, duplicate examples, label leakage
- Verify checksum of downloaded pretrained weights
- Atomic writes for checkpoints (write to temp, rename)

### Concurrency & Performance
- Use `rayon` for CPU-bound preprocessing parallelism
- Async I/O for dataset loading where beneficial
- Pin memory for GPU transfers
- Profile with `cargo flamegraph` and `nvprof`/`nsys` regularly

## Configuration Management

### Cargo Features
- Default: `cuda` (requires NVIDIA driver + CUDA toolkit)
- Optional: `metal` (Apple Silicon), `mkl` (Intel MKL BLAS), `accelerate` (Apple Accelerate)
- Test all feature combinations in CI

### Environment Variables
- `EXPERAI_DEVICE`: override device selection (`cuda:0`, `metal`, `cpu`)
- `EXPERAI_LOG_LEVEL`: `trace`|`debug`|`info`|`warn`|`error`
- `EXPERAI_SEED`: global random seed (default: 42)
- `HF_HUB_CACHE`: cache directory for Hugging Face models
- `CANDLE_CUDA_ARCH`: target CUDA compute capability (e.g., `80` for Ampere)

## Testing Strategy

### Unit Tests (in-module `#[cfg(test)]`)
- Tokenizer round-trip: encode → decode preserves text
- Data collator: batch shapes, padding correctness, mask alignment
- Model forward pass: output shapes, no NaN, gradient flow
- Loss functions: cross-entropy matches reference implementation
- Learning rate schedules: values at step 0, warmup end, decay end

### Integration Tests (`tests/`)
- Full training run on tiny dataset (1-2 epochs, 100 samples) → loss decreases
- Checkpoint save/load → identical model weights
- Generation: deterministic output with fixed seed + greedy decoding
- CLI: all subcommands exit 0 with valid args, non-zero with invalid args

### Property-Based Tests (`proptest`)
- Tokenizer: round-trip for arbitrary UTF-8 strings
- Data collator: batch sizes 1-128, sequence lengths 1-4096
- Model: random configs within valid ranges produce valid forward pass

### CI Validation (GitHub Actions)
```yaml
- cargo fmt --check
- cargo clippy --all-targets --all-features -- -D warnings
- cargo test --all-features
- cargo build --release --features cuda
- cargo test --features cuda (if runner has GPU)
```

## Security & Privacy

- No telemetry, no network calls except explicit `hf-hub` downloads
- Never log raw training data, prompts, or generated text at `info` level
- Sanitize file paths in logs (no absolute paths to user directories)
- Validate all external inputs: model weights, tokenizer files, dataset files
- Use `safetensors` exclusively; never load arbitrary `pickle`/`pt` files

## Documentation Standards

- All public items: `///` doc comments with examples
- `README.md`: quickstart, CLI reference, architecture overview
- `ARCHITECTURE.md`: data flow, module dependencies, extension points
- Update docs alongside code changes; treat outdated docs as bugs
- Generate API docs: `cargo doc --no-deps --all-features`

## Error Handling

- Use `anyhow::Result` for fallible operations; `thiserror` for library errors
- Error variants: `Io`, `ModelLoad`, `Tokenization`, `Training`, `Config`, `Cuda`
- Include actionable suggestions in error messages
- Never `unwrap()`/`expect()` in production paths; only in tests/examples

## Release Process

1. Update `Cargo.toml` version (semver)
2. `cargo test --all-features` passes
3. `cargo build --release --features cuda` produces binary
4. Tag: `git tag v{version}`
5. Publish: `cargo publish` (if on crates.io)
6. GitHub Release with binary artifacts and changelog

## Troubleshooting Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `CUDA error: invalid device ordinal` | `EXPERAI_DEVICE` points to non-existent GPU | `nvidia-smi` to list devices; set valid ordinal |
| `OutOfMemory` during training | Batch size too large, no gradient accumulation | Reduce `--batch-size`, increase `--grad-accum` |
| Loss = NaN after N steps | fp16 overflow, no loss scaling | Enable dynamic loss scaling, check gradient clipping |
| Tokenizer `encode` panics | Input contains chars not in vocab | Add `[UNK]` handling, validate data preprocessing |
| `hf-hub` download fails | Network, auth, or model not found | Check `HF_TOKEN`, model ID, connectivity |
| Checkpoint resume fails | Config mismatch (vocab size, layers) | Verify `model_config.json` matches checkpoint |

## Extension Points

- **Custom Models**: Implement `candle_nn::Module` in `model.rs`, register in `lib.rs`
- **Custom Datasets**: Add loader in `data.rs` implementing `Dataset` trait
- **Custom Loss**: Add function in `training.rs`, wire via CLI flag
- **Custom Callbacks**: Training hooks (on_epoch_end, on_step) via trait objects
- **New CLI Commands**: Add variant to `Commands` enum, implement handler

## Validation Checklist (Pre-Commit)

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test --all-features`
- [ ] `cargo build --release --features cuda`
- [ ] CLI manual test: `./target/release/experai train --help`
- [ ] CLI manual test: `./target/release/experai preprocess --help`
- [ ] CLI manual test: `./target/release/experai generate --help`
- [ ] Edge cases: zero epochs, huge batch, invalid paths, missing files
- [ ] Docs updated: `README.md`, `ARCHITECTURE.md`, inline comments
- [ ] No `unwrap()`/`expect()` in `src/` (except tests)
- [ ] No hardcoded paths; all configurable via CLI/env
- [ ] Secrets check: `git diff --check` + `git secrets --scan` (if configured)