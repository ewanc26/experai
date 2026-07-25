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

## Architecture

```
src/
├── main.rs           # CLI entry point
├── lib.rs            # Module declarations
├── model.rs          # Transformer architecture
├── training.rs       # Training loop and optimization
├── data.rs           # Dataset loading and batching
├── jetstream.rs      # AT Protocol Jetstream streaming
├── at_protocol.rs    # AT Protocol REST API client
├── preprocessing.rs  # Text cleaning pipeline
├── utils.rs          # Hardware detection and auto-tuning
└── logging/          # Structured logging configuration
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

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
