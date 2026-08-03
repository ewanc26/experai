# Experai

A small language model training toolkit built in Rust using the Candle ML framework with CUDA/Metal acceleration.

## Features

- **Hardware-Aware Training** - Automatically detects GPU/CPU and optimizes batch size, precision, and memory usage
- **AT Protocol Integration** - Train on data from Bluesky via REST API or real-time Jetstream streaming
- **CLI Interface** - Full-featured command line with train, preprocess, generate, and data loading commands
- **Mixed Precision** - Automatic bf16/fp16/fp32 selection based on hardware capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/experai.git
cd experai

# Build with CUDA support (default, requires NVIDIA driver + CUDA toolkit)
cargo build --release

# Build with Metal support (Apple Silicon / macOS)
cargo build --release --no-default-features --features metal

# Build with CPU only (no GPU acceleration)
cargo build --release --no-default-features
```

> **Note for macOS users:** The default `cuda` feature will fail to build on macOS. Use `--no-default-features --features metal` instead.

### Tokenizer

The tokenizer file (`models/tokenizer.json`) is required for all commands. It must be a valid HuggingFace `tokenizers` format JSON file (not a custom vocab/merges file). Download the GPT-2 tokenizer:

```bash
curl -sL "https://huggingface.co/gpt2/resolve/main/tokenizer.json" -o models/tokenizer.json
```

The `models/` directory is gitignored, so each machine needs to fetch the tokenizer separately.

## Quick Start

### Train a model on local data

```bash
./target/release/experai train \
  --model gpt2 \
  --data data/training.jsonl \
  --output-dir output \
  --auto-tune
```

### Train from Bluesky (AT Protocol)

```bash
# Train from a specific user's posts
./target/release/experai at-protocol \
  --handle bsky.app \
  --max-samples 1000 \
  --output-dir output \
  --auto-tune

# Train from the live Jetstream firehose
./target/release/experai jetstream-train \
  --auto-tune \
  --max-samples 10000 \
  --output-dir output
```

### Generate text

```bash
./target/release/experai generate \
  --model output \
  --prompt "The future of AI is" \
  --max-tokens 100
```

## Commands

| Command | Description |
|---------|-------------|
| `train` | Train a model on a JSONL dataset |
| `preprocess` | Clean and tokenize raw text data |
| `generate` | Generate text from a trained model |
| `at-protocol` | Load training data from a Bluesky user |
| `jetstream-train` | Stream and train from the AT Protocol firehose |

Global options:

```bash
experai --support   # Print sponsor links and exit
experai             # No subcommand: show help
```

### Train

```bash
experai train [OPTIONS]

Options:
  -m, --model <MODEL>           Model name
  -d, --data <DATA>             Path to JSONL training data
  -o, --output-dir <OUTPUT_DIR> Output directory
  -e, --epochs <EPOCHS>         Number of epochs [default: 3]
  -b, --batch-size <BATCH_SIZE> Batch size [default: 4]
  -g, --grad-accum <GRAD_ACCUM> Gradient accumulation steps [default: 8]
  -l, --lr <LR>                 Learning rate [default: 0.0005]
  -t, --tokenizer <TOKENIZER>   Tokenizer path [default: models/tokenizer.json]
      --auto-tune               Auto-detect hardware and optimize params
```

### Jetstream Train

```bash
experai jetstream-train [OPTIONS]

Options:
      --jetstream-host <HOST>   Jetstream host [default: jetstream2.us-east.bsky.network]
  -c, --collections <COLLECTIONS>  Collections to subscribe to [default: app.bsky.feed.post]
  -d, --dids <DIDS>             DIDs to filter (empty = all users)
  -n, --max-samples <MAX>       Maximum samples to collect [default: 10000]
      --max-duration-secs <SECS> Maximum collection time [default: 3600]
  -o, --output-dir <OUTPUT_DIR> Output directory
      --auto-tune               Auto-detect hardware and optimize params
```

## Auto-Tuning

The `--auto-tune` flag enables automatic hardware detection and parameter optimization:

| VRAM | Batch Size | Grad Accum | Seq Len | Precision |
|------|------------|------------|---------|-----------|
| 24 GB | 32 | 1 | 2048 | bf16 |
| 16 GB | 16 | 2 | 2048 | bf16 |
| 8 GB | 4 | 8 | 1024 | bf16 |
| 4 GB | 1 | 32 | 512 | bf16/fp16/fp32 |

## Memory Governance

Experai is designed to be a good system citizen. Training dynamically adapts to system load to ensure the machine never stutters and never dips into swap memory.

### Swap avoidance

The system load monitor tracks swap usage alongside CPU and memory. The default policy is **zero tolerance** for swap:

- If any swap is in use, training throttles hard (batch size scaled to 25%, learning rate halved, precision forced to f32).
- If swap exceeds 2% of total swap capacity **or** free memory drops below 2 GB, training pauses for 2 seconds to let the OS reclaim memory before continuing.
- The `max_swap_usage` config field (default `0.0`) controls the swap warning threshold.
- The `min_free_mem_mb` config field (default `2048`) sets the minimum free memory floor.

### Anti-stutter thresholds

CPU and memory thresholds are set conservatively to keep the system responsive:

| Metric | Warning | Critical | Response |
|--------|---------|----------|----------|
| CPU usage | 50% | 75% | Scale batch 1.0→0.5, LR 1.0→0.7 |
| Memory usage | 60% | 75% | Scale batch 1.0→0.5, LR 1.0→0.7 |
| Swap usage | 0% | 2% | Heavy throttle or pause |
| Free memory | — | < 2 GB | Pause 2s |

At critical levels, batch size is scaled to 25%, learning rate to 50%, and precision is forced to f32 to reduce compute intensity. All metrics are smoothed with an exponential moving average (alpha=0.3) to avoid reacting to transient spikes.

## Data Format

Training data should be in JSONL format with a `text` field:

```json
{"text": "The quick brown fox jumps over the lazy dog"}
{"text": "Another training example here"}
```

Optional `label` field for classification tasks:

```json
{"text": "This is positive sentiment", "label": "positive"}
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `EXPERAI_DEVICE` | Force device (`cuda:0`, `metal`, `cpu`) | Auto-detect |
| `EXPERAI_LOG_LEVEL` | Log level (`trace`, `debug`, `info`, `warn`, `error`) | `info` |
| `EXPERAI_SEED` | Random seed | `42` |

### Training config fields

| Field | Description | Default |
|-------|-------------|---------|
| `max_swap_usage` | Swap fraction that triggers throttling (0.0 = any swap) | `0.0` |
| `min_free_mem_mb` | Minimum free memory in MB before training pauses | `2048` |
| `enable_load_monitoring` | Dynamically scale batch/LR based on system load | `true` |
| `max_cpu_load` | CPU fraction that triggers batch scaling | `0.80` |
| `load_check_interval` | Poll system load every N steps | `10` |

## Architecture

```
src/
├── main.rs               # CLI entry point
├── lib.rs                # Module declarations
├── model.rs              # Transformer architecture
├── training/
│   ├── config.rs         # Hyperparameters and runtime settings
│   ├── trainer.rs        # Training loop, checkpointing, LR scheduling
│   └── loss.rs           # Loss and perplexity computation
├── data/                 # Dataset loading, collation, tokenization
├── commands/             # CLI command implementations (train, generate, etc.)
├── utils/
│   ├── device.rs         # GPU/CPU detection and selection
│   ├── hardware.rs       # Hardware profiling and auto-tuning presets
│   ├── monitor.rs        # System load monitor (CPU, memory, swap)
│   └── helpers.rs        # Misc utilities (gradient accumulator, etc.)
├── jetstream.rs          # AT Protocol Jetstream streaming
├── at_protocol.rs        # AT Protocol REST API client
├── preprocessing.rs      # Text cleaning pipeline
└── logging/              # Structured logging configuration
```

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_auto_tuner
```

## opencode integration

The repo ships with opencode (https://opencode.ai) integrations that let you drive
the toolkit from an agent or the TUI:

- **Plugin** — `.opencode/plugin/experai.ts` registers `experai_*` tools
  (`experai_support`, `experai_generate`, `experai_train`, `experai_preprocess`,
  `experai_package`, `experai_atprotocol`, `experai_jetstream`,
  `experai_commands`) that shell out to the compiled binary.
- **Slash command** — `/experai <args>` runs the CLI and summarizes the result.
- **MCP server** — `mcp/experai-server.mjs` exposes the same capabilities as
  standard MCP tools (usable by opencode and any other MCP client). It is wired
  up in `opencode.json`; the plugin tools are auto-discovered from
  `.opencode/plugin/`.

The integrations call `./target/release/experai` (falling back to
`target/debug/experai`, then `experai` on `PATH`). Set `EXPERAI_BIN` to point
at a specific binary. The binary is resolved at runtime, so
`cargo build --release --no-default-features --features metal` must have been
run at least once.

Set up the MCP server dependencies once:

```bash
cd mcp && npm install
```

Machine-readable output is available for tooling:

```bash
experai --support --json   # structured sponsor links
experai --json             # CLI metadata: name, version, subcommands
experai generate --model output --prompt "..." --json   # {"text": "...", "tokens": N}
```

## Support

If you find this project useful, consider supporting its development:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/ewancroft)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-30363D?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sponsors/ewanc26)

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
