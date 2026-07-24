use anyhow::Result;
use candle_core::IndexOp;
use candle_nn::Module;
use clap::{Parser, Subcommand};
use experai::at_protocol::ATProtocolConfig;
use experai::data::{load_tokenizer, validate_dataset, Dataset};
use experai::logging::init_logger;
use experai::model::ModelConfig;
use experai::preprocessing::Preprocessor;
use experai::training::{Trainer, TrainingConfig};
use experai::utils;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "experai")]
#[command(version = "0.1.0")]
#[command(about = "Small language model training toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model on a dataset
    Train {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        data: String,
        #[arg(short = 'l', long, default_value = "0.0005")]
        lr: f64,
        #[arg(short = 'e', long, default_value = "3")]
        epochs: usize,
        #[arg(short = 'b', long, default_value = "4")]
        batch_size: usize,
        #[arg(short = 'g', long, default_value = "8")]
        grad_accum: usize,
        #[arg(short = 'o', long)]
        output_dir: String,
        #[arg(short = 't', long, default_value = "models/tokenizer.json")]
        tokenizer: String,
        /// Auto-detect hardware and optimize training params
        #[arg(long)]
        auto_tune: bool,
    },
    /// Preprocess text data for training
    Preprocess {
        #[arg(short = 'i', long)]
        input: String,
        #[arg(short = 'o', long)]
        output: String,
        #[arg(short = 't', long)]
        tokenizer: String,
        #[arg(short = 'l', long, default_value = "512")]
        max_length: usize,
    },
    /// Generate text from a trained model
    Generate {
        #[arg(short = 'm', long)]
        model: String,
        #[arg(short = 'p', long)]
        prompt: String,
        #[arg(short = 'n', long, default_value = "50")]
        max_tokens: usize,
        #[arg(short = 't', long, default_value = "0.7")]
        temperature: f64,
        #[arg(short = 'k', long, default_value = "50")]
        top_k: usize,
        #[arg(short = 'P', long, default_value = "0.95")]
        top_p: f64,
        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer: String,
    },
    /// Load training data from an AT Protocol PDS
    AtProtocol {
        #[arg(short = 'u', long, default_value = "https://bsky.social")]
        pds_url: String,
        #[arg(short = 'h', long, default_value = "public")]
        handle: String,
        #[arg(short = 'n', long, default_value = "1000")]
        max_samples: usize,
        #[arg(short = 'o', long)]
        output_dir: String,
        #[arg(short = 'm', long, default_value = "gpt2")]
        model_name: String,
        #[arg(short = 'l', long, default_value = "0.0005")]
        lr: f64,
        #[arg(short = 'e', long, default_value = "3")]
        epochs: usize,
        #[arg(short = 'b', long, default_value = "4")]
        batch_size: usize,
        #[arg(short = 'g', long, default_value = "8")]
        grad_accum: usize,
        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer: String,
    },
    /// Stream from AT Protocol Jetstream and train
    JetstreamTrain {
        /// Jetstream host (default: jetstream2.us-east.bsky.network)
        #[arg(long, default_value = "jetstream2.us-east.bsky.network")]
        jetstream_host: String,
        /// Collections to subscribe to (e.g., app.bsky.feed.post)
        #[arg(short = 'c', long, default_value = "app.bsky.feed.post")]
        collections: Vec<String>,
        /// DIDs to filter (empty = all users)
        #[arg(short = 'd', long)]
        dids: Vec<String>,
        /// Maximum number of samples to collect
        #[arg(short = 'n', long, default_value = "10000")]
        max_samples: usize,
        /// Maximum collection time in seconds
        #[arg(long, default_value = "3600")]
        max_duration_secs: u64,
        /// Output directory for model and data
        #[arg(short = 'o', long)]
        output_dir: String,
        /// Model name
        #[arg(short = 'm', long, default_value = "gpt2")]
        model_name: String,
        /// Learning rate
        #[arg(short = 'l', long, default_value = "0.0005")]
        lr: f64,
        /// Number of epochs
        #[arg(short = 'e', long, default_value = "3")]
        epochs: usize,
        /// Batch size
        #[arg(short = 'b', long, default_value = "4")]
        batch_size: usize,
        /// Gradient accumulation steps
        #[arg(short = 'g', long, default_value = "8")]
        grad_accum: usize,
        /// Tokenizer path
        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer: String,
        /// Auto-detect hardware and optimize training params
        #[arg(long)]
        auto_tune: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let log_level = std::env::var("EXPERAI_LOG_LEVEL").ok();
    init_logger(log_level, false)?;

    match cli.command {
        Commands::Train {
            model,
            data,
            lr,
            epochs,
            batch_size,
            grad_accum,
            output_dir,
            tokenizer,
            auto_tune,
        } => {
            info!(
                "Starting training: model={}, data={}, lr={}, epochs={}",
                model, data, lr, epochs
            );

            // Apply auto-tuning if requested
            let mut final_config = TrainingConfig {
                model_name: model.clone(),
                data_path: data.clone(),
                output_dir: output_dir.clone(),
                epochs,
                learning_rate: lr,
                batch_size,
                gradient_accumulation_steps: grad_accum,
                max_seq_len: 512,
                precision: "bf16".to_string(),
                warmup_steps: 500,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
                seed: 42,
                enable_load_monitoring: true,
                max_cpu_load: 0.80,
                load_check_interval: 10,
            };

            if auto_tune {
                let profile = utils::HardwareProfile::detect();
                let tuned = utils::AutoTuner::recommend(&profile);
                info!("Auto-tune: {:?}", tuned);

                final_config.batch_size = tuned.batch_size;
                final_config.gradient_accumulation_steps = tuned.gradient_accumulation_steps;
                final_config.precision = tuned.precision;
                final_config.max_seq_len = tuned.max_seq_len;
                final_config.learning_rate = tuned.learning_rate;
                final_config.warmup_steps = tuned.warmup_steps;
                final_config.weight_decay = tuned.weight_decay;
                final_config.max_grad_norm = tuned.max_grad_norm;
            }

            std::fs::create_dir_all(&output_dir)?;

            // Save training config
            let config_path = format!("{}/config.json", output_dir);
            std::fs::write(&config_path, serde_json::to_string_pretty(&final_config)?)?;
            info!("Saved training config to {}", config_path);

            // Load tokenizer
            let tok = load_tokenizer(&tokenizer)?;
            info!("Loaded tokenizer from {}", tokenizer);

            // Load dataset
            let dataset = Dataset::from_jsonl(&data, tok, final_config.max_seq_len)?;
            info!("Loaded {} samples from {}", dataset.len(), data);

            // Validate dataset
            validate_dataset(&dataset)?;

            // Build and train model
            let model_config = ModelConfig::default();
            let mut trainer = Trainer::new(final_config, model_config)?;
            info!("Trainer initialised on {:?}", trainer.device);

            let losses = trainer.train(&dataset)?;
            info!("Training complete. Losses: {:?}", losses);
        }
        Commands::Preprocess {
            input,
            output,
            tokenizer,
            max_length,
        } => {
            info!(
                "Starting preprocessing: input={}, output={}, tokenizer={}, max_length={}",
                input, output, tokenizer, max_length
            );
            let tok = load_tokenizer(&tokenizer)?;
            let preprocessor = Preprocessor::new(tok, None);
            preprocessor.preprocess_file(&input, &output)?;
            info!("Preprocessing complete: {} -> {}", input, output);
        }
        Commands::Generate {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            tokenizer,
        } => {
            info!("Starting generation: model={}, prompt={}", model, prompt);
            let tok = load_tokenizer(&tokenizer)?;

            // Load model weights and generate
            let model_config = ModelConfig::default();
            let device = utils::device_from_env().to_candle()?;
            let mut var_map = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);

            let model_arch = experai::model::build_model(&model_config, vb)?;

            // Try to load weights
            let weights_path = format!("{}/best.safetensors", model);
            if std::path::Path::new(&weights_path).exists() {
                var_map.load(&weights_path)?;
                info!("Loaded weights from {}", weights_path);
            } else {
                warn!(
                    "No weights found at {}, using random initialisation",
                    weights_path
                );
            }

            // Tokenise prompt
            let encoding = tok
                .encode(prompt.to_string(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
            let input_ids: Vec<u32> = encoding.get_ids().iter().copied().collect();
            info!("Prompt tokenised to {} tokens", input_ids.len());

            // Generate tokens
            let mut generated = input_ids.clone();
            for _ in 0..max_tokens {
                let input_tensor =
                    candle_core::Tensor::new(generated.clone(), &device)?.unsqueeze(0)?;

                let logits = model_arch.forward(&input_tensor)?;
                let seq_len = logits.dim(1)?;
                let next_logits = logits.i((0, seq_len - 1))?;

                // Apply temperature
                let next_logits = (next_logits / temperature)?;

                // Apply top-k and top-p filtering
                let next_token = sample_top_k_top_p(&next_logits, top_k, top_p)?;

                generated.push(next_token as u32);

                // Stop on EOS
                if next_token == model_config.eos_token_id {
                    info!("Generated EOS token, stopping");
                    break;
                }
            }

            // Decode
            let output_text = tok
                .decode(&generated, true)
                .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
            info!("Generated {} tokens", generated.len() - input_ids.len());
            println!("{}", output_text);
        }
        Commands::AtProtocol {
            pds_url,
            handle,
            max_samples,
            output_dir,
            model_name,
            lr,
            epochs,
            batch_size,
            grad_accum,
            tokenizer,
        } => {
            info!(
                "Starting AT Protocol dataset loading from PDS: {} handle: {} max_samples: {}",
                pds_url, handle, max_samples
            );

            let at_config = ATProtocolConfig {
                pds_url: pds_url.clone(),
                handle: handle.clone(),
                max_samples,
            };

            let tok = load_tokenizer(&tokenizer)?;

            // Use tokio runtime for async AT Protocol calls
            let rt = tokio::runtime::Runtime::new()?;
            let dataset = rt.block_on(Dataset::from_at_protocol(
                &at_config.pds_url,
                &at_config.handle,
                at_config.max_samples,
                tok,
            ))?;
            info!("Loaded {} samples from AT Protocol", dataset.len());

            // Save dataset to JSONL
            std::fs::create_dir_all(&output_dir)?;
            let data_path = format!("{}/at_protocol_data.jsonl", output_dir);
            let mut file = std::fs::File::create(&data_path)?;
            for sample in &dataset.samples {
                use std::io::Write;
                let json = serde_json::to_string(&serde_json::json!({
                    "text": sample.text,
                    "label": sample.label,
                }))?;
                writeln!(file, "{}", json)?;
            }
            info!("Saved {} samples to {}", dataset.len(), data_path);

            // Train on the loaded data
            let mut final_at_config = TrainingConfig {
                model_name: model_name.clone(),
                data_path: data_path,
                output_dir: output_dir.clone(),
                epochs,
                learning_rate: lr,
                batch_size,
                gradient_accumulation_steps: grad_accum,
                max_seq_len: 512,
                precision: "bf16".to_string(),
                warmup_steps: 500,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
                seed: 42,
                enable_load_monitoring: true,
                max_cpu_load: 0.80,
                load_check_interval: 10,
            };

            // Auto-tune for AT Protocol training too
            let profile = utils::HardwareProfile::detect();
            let tuned = utils::AutoTuner::recommend(&profile);
            final_at_config.batch_size = tuned.batch_size;
            final_at_config.gradient_accumulation_steps = tuned.gradient_accumulation_steps;
            final_at_config.precision = tuned.precision;
            final_at_config.max_seq_len = tuned.max_seq_len;

            let model_config = ModelConfig::default();
            let mut trainer = Trainer::new(final_at_config, model_config)?;
            info!("Trainer initialised on {:?}", trainer.device);

            let losses = trainer.train(&dataset)?;
            info!("AT Protocol training complete. Losses: {:?}", losses);
        }
        Commands::JetstreamTrain {
            jetstream_host,
            collections,
            dids,
            max_samples,
            max_duration_secs,
            output_dir,
            model_name,
            lr,
            epochs,
            batch_size,
            grad_accum,
            tokenizer,
            auto_tune,
        } => {
            info!(
                "Starting Jetstream training: host={}, collections={:?}, max_samples={}",
                jetstream_host, collections, max_samples
            );

            let tok = load_tokenizer(&tokenizer)?;

            // Build Jetstream config
            let jetstream_config = experai::jetstream::JetstreamConfig {
                host: jetstream_host.clone(),
                collections: collections.clone(),
                dids: dids.clone(),
                max_samples,
                batch_size: 100,
                max_duration_secs,
                compression: true,
            };

            // Collect data from Jetstream
            let rt = tokio::runtime::Runtime::new()?;
            let (dataset, stats) = rt.block_on(experai::jetstream::collect_from_jetstream(
                &jetstream_config,
                tok,
            ))?;

            info!(
                "Collected {} posts from Jetstream ({:.1}s, {:.1} posts/s)",
                stats.valid_posts, stats.duration_secs, stats.posts_per_second
            );

            // Save collected data
            std::fs::create_dir_all(&output_dir)?;
            let data_path = format!("{}/jetstream_data.jsonl", output_dir);
            let mut file = std::fs::File::create(&data_path)?;
            for sample in &dataset.samples {
                use std::io::Write;
                let json = serde_json::to_string(&serde_json::json!({
                    "text": sample.text,
                    "label": sample.label,
                }))?;
                writeln!(file, "{}", json)?;
            }
            info!("Saved {} samples to {}", dataset.len(), data_path);

            // Build training config
            let mut final_config = TrainingConfig {
                model_name: model_name.clone(),
                data_path: data_path.clone(),
                output_dir: output_dir.clone(),
                epochs,
                learning_rate: lr,
                batch_size,
                gradient_accumulation_steps: grad_accum,
                max_seq_len: 512,
                precision: "bf16".to_string(),
                warmup_steps: 500,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
                seed: 42,
            };

            // Apply auto-tuning if requested
            if auto_tune {
                let profile = utils::HardwareProfile::detect();
                let tuned = utils::AutoTuner::recommend(&profile);
                info!("Auto-tune: {:?}", tuned);

                final_config.batch_size = tuned.batch_size;
                final_config.gradient_accumulation_steps = tuned.gradient_accumulation_steps;
                final_config.precision = tuned.precision;
                final_config.max_seq_len = tuned.max_seq_len;
                final_config.learning_rate = tuned.learning_rate;
                final_config.warmup_steps = tuned.warmup_steps;
                final_config.weight_decay = tuned.weight_decay;
                final_config.max_grad_norm = tuned.max_grad_norm;
            }

            // Save training config
            let config_path = format!("{}/config.json", output_dir);
            std::fs::write(&config_path, serde_json::to_string_pretty(&final_config)?)?;
            info!("Saved training config to {}", config_path);

            // Train
            let model_config = ModelConfig::default();
            let mut trainer = Trainer::new(final_config, model_config)?;
            info!("Trainer initialised on {:?}", trainer.device);

            let losses = trainer.train(&dataset)?;
            info!("Jetstream training complete. Losses: {:?}", losses);
        }
    }

    Ok(())
}

fn sample_top_k_top_p(logits: &candle_core::Tensor, top_k: usize, top_p: f64) -> Result<usize> {
    let mut logits_vec = logits.to_vec1::<f32>()?;

    // Apply temperature (caller has already divided by temperature)

    // Top-k filtering
    if top_k > 0 && top_k < logits_vec.len() {
        let mut indices: Vec<usize> = (0..logits_vec.len()).collect();
        indices.sort_by(|&a, &b| {
            logits_vec[b]
                .partial_cmp(&logits_vec[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(top_k);
        let top_k_set: std::collections::HashSet<usize> = indices.iter().copied().collect();
        for (i, v) in logits_vec.iter_mut().enumerate() {
            if !top_k_set.contains(&i) {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Top-p (nucleus) filtering
    if top_p > 0.0 && top_p < 1.0 {
        let max_val = logits_vec
            .iter()
            .filter(|v| v.is_finite())
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits_vec.iter().map(|v| (v - max_val).exp()).sum();
        let probs: Vec<f32> = logits_vec
            .iter()
            .map(|v| (v - max_val).exp() / exp_sum)
            .collect();

        let mut sorted_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        sorted_probs
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cum = 0.0_f32;
        let mut cutoff_idx = sorted_probs.len();
        for &(idx, prob) in &sorted_probs {
            cum += prob;
            if cum >= top_p as f32 {
                cutoff_idx = idx;
                break;
            }
        }

        let top_p_set: std::collections::HashSet<usize> = sorted_probs
            .iter()
            .take(cutoff_idx + 1)
            .map(|&(i, _)| i)
            .collect();
        for (i, v) in logits_vec.iter_mut().enumerate() {
            if !top_p_set.contains(&i) {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Renormalize and sample
    let max_val = logits_vec
        .iter()
        .filter(|v| v.is_finite())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits_vec.iter().map(|v| (v - max_val).exp()).sum();
    let probs: Vec<f32> = logits_vec
        .iter()
        .map(|v| (v - max_val).exp() / exp_sum)
        .collect();

    let mut rng = rand::thread_rng();
    let r: f32 = rand::Rng::gen_range(&mut rng, 0.0..1.0);
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return Ok(i);
        }
    }
    Ok(probs.len().saturating_sub(1))
}
