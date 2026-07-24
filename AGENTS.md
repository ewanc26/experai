# AGENTS.md

# Experai Project Guide

## Project Structure
```
experai
├── Cargo.toml
└── src
    ├── main.rs
    ├── lib.rs
    ├── data.rs
    ├── model.rs
    ├── preprocessing.rs
    ├── training.rs
    ├── utils.rs
    └── logging
```

## Agent Tasks
1. **Model Development**
   - Create/extend models in `model.rs` using candle-transformers
   - Add custom layers for niche NLP tasks

2. **Data Pipeline**
   - Implement preprocessing in `preprocessing.rs`
   - Handle dataset loading/format in `data.rs`

3. **Training Workflow**
   - Configure CLI flags in `main.rs` for training
   - Implement loss functions in `training.rs`

4. **CLI Operations**
   - Support 3 subcommands: train/preprocess/generate
   - Add new flags via clap parser updates

## Implementation Notes
- All models must use CUDA-accelerated candle-nn
- Validate with `cargo test` after changes
- Logging handled in dedicated `logging` module

## Validation Checklist
- `cargo fmt --check`
- `cargo clippy --all-targets`
- `cargo build --release`
- Test CLI edge cases (zero/large values, invalid paths)