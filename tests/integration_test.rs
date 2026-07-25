use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;
use tokenizers::models::bpe::BPE;
use tokenizers::Tokenizer;

use experai::data::Dataset;
use experai::model::ModelConfig;
use experai::training::{Trainer, TrainingConfig};

fn create_test_tokenizer() -> tokenizers::Tokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();

    vocab.insert("[PAD]".to_string(), 0);
    vocab.insert("[UNK]".to_string(), 100);
    vocab.insert("[CLS]".to_string(), 101);
    vocab.insert("[SEP]".to_string(), 102);
    vocab.insert("[MASK]".to_string(), 103);

    let test_words = [
        "the",
        "a",
        "cat",
        "dog",
        "sat",
        "ran",
        "on",
        "in",
        "at",
        "big",
        "small",
        "fast",
        "slow",
        "mat",
        "floor",
        "house",
        "is",
        "was",
        "are",
        "this",
        "that",
        "it",
        "to",
        "and",
        "of",
        "for",
        "with",
        "by",
        "from",
        "or",
        "but",
        "not",
        "you",
        "i",
        "we",
        "he",
        "she",
        "they",
        "my",
        "his",
        "her",
        "good",
        "bad",
        "red",
        "blue",
        "one",
        "two",
        "three",
        "hello",
        "world",
        "how",
        "what",
        "when",
        "where",
        "do",
        "have",
        "can",
        "will",
        "would",
        "should",
        "like",
        "love",
        "need",
        "want",
        "go",
        "come",
        "see",
        "look",
        "make",
        "take",
        "give",
        "get",
        "use",
        "find",
        "tell",
        "ask",
        "work",
        "try",
        "call",
        "help",
        "start",
        "play",
        "move",
        "live",
        "become",
        "leave",
        "put",
        "mean",
        "keep",
        "let",
        "begin",
        "seem",
        "show",
        "hear",
        "turn",
        "night",
        "city",
        "tree",
        "road",
        "car",
        "book",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "jumped",
        "very",
        "quickly",
        "each",
        "other",
        "learning",
        "models",
        "require",
        "large",
        "datasets",
        "training",
        "rust",
        "systems",
        "programming",
        "language",
        "focused",
        "safety",
        "natural",
        "processing",
        "enables",
        "computers",
        "understand",
        "text",
        "deep",
        "revolutionized",
        "artificial",
        "intelligence",
    ];

    let mut id = 104u32;
    for word in &test_words {
        if !vocab.contains_key(*word) {
            vocab.insert(word.to_string(), id);
            id += 1;
        }
    }

    for c in 'a'..='z' {
        let s = c.to_string();
        if !vocab.contains_key(&s) && id < 30522 {
            vocab.insert(s, id);
            id += 1;
        }
    }

    let bpe = BPE::new(vocab, Vec::<(String, String)>::new());
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(tokenizers::pre_tokenizers::whitespace::Whitespace);
    tokenizer
}

#[test]
fn test_full_training_pipeline() {
    let tmp_dir = tempdir().unwrap();
    let data_path = tmp_dir.path().join("train.jsonl");
    let output_dir = tmp_dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    let samples = vec![
        r#"{"text": "the cat sat on the mat"}"#,
        r#"{"text": "the dog ran fast on the floor"}"#,
        r#"{"text": "a big red fox jumps over the lazy dog"}"#,
        r#"{"text": "hello world this is a test"}"#,
        r#"{"text": "the quick brown fox jumps"}"#,
        r#"{"text": "she loves to read good books"}"#,
        r#"{"text": "he went to the city today"}"#,
        r#"{"text": "the small bird sat on a tree"}"#,
    ];

    let mut file = File::create(&data_path).unwrap();
    for sample in &samples {
        writeln!(file, "{}", sample).unwrap();
    }

    let tokenizer = create_test_tokenizer();

    let dataset = Dataset::from_jsonl(data_path.to_str().unwrap(), tokenizer, 128).unwrap();
    assert_eq!(dataset.len(), 8);

    let model_config = ModelConfig {
        vocab_size: 30522,
        hidden_size: 64,
        num_layers: 2,
        num_heads: 4,
        intermediate_size: 128,
        max_seq_len: 128,
        hidden_act: "gelu".to_string(),
        initializer_range: 0.02,
        layer_norm_eps: 1e-5,
        pad_token_id: 0,
        bos_token_id: 101,
        eos_token_id: 102,
    };

    let train_config = TrainingConfig {
        model_name: "test".to_string(),
        data_path: data_path.to_str().unwrap().to_string(),
        output_dir: output_dir.to_str().unwrap().to_string(),
        epochs: 1,
        learning_rate: 5e-4,
        batch_size: 4,
        gradient_accumulation_steps: 1,
        max_seq_len: 128,
        precision: "f32".to_string(),
        warmup_steps: 0,
        weight_decay: 0.01,
        max_grad_norm: 1.0,
        seed: 42,
        enable_load_monitoring: false,
        max_cpu_load: 0.8,
        load_check_interval: 10,
        max_swap_usage: 0.0,
        min_free_mem_mb: 2048,
        min_lr_ratio: 0.1,
    };

    let vocab_size = model_config.vocab_size;
    let mut trainer = Trainer::new(train_config, model_config).unwrap();
    let losses = trainer.train(&dataset).unwrap();

    assert!(!losses.is_empty(), "Should have at least one loss value");
    let final_loss = losses.last().unwrap();
    assert!(
        final_loss.is_finite(),
        "Loss should be finite, got {}",
        final_loss
    );
    // Cross-entropy is a *negative* log-likelihood, so it must be positive. An
    // untrained model predicts roughly uniformly, putting the loss near
    // ln(vocab_size); allow generous slack, but catch a sign flip or a blow-up.
    let uniform_baseline = (vocab_size as f64).ln();
    assert!(
        *final_loss > 0.0,
        "Cross-entropy loss should be positive, got {}",
        final_loss
    );
    assert!(
        *final_loss < uniform_baseline * 2.0,
        "Loss {} is implausibly far above the uniform-prediction baseline {:.2}",
        final_loss,
        uniform_baseline
    );

    let checkpoint_path = output_dir.join("best.json");
    assert!(
        checkpoint_path.exists(),
        "Checkpoint metadata should exist at {:?}",
        checkpoint_path
    );

    let weights_path = output_dir.join("best.safetensors");
    assert!(
        weights_path.exists(),
        "Checkpoint weights should exist at {:?}",
        weights_path
    );

    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&checkpoint_path).unwrap()).unwrap();
    assert!(
        metadata.get("global_step").is_some(),
        "Checkpoint metadata should contain global_step"
    );
    assert!(
        metadata.get("best_loss").is_some(),
        "Checkpoint metadata should contain best_loss"
    );
}

/// Flatten every variable in a trainer's `VarMap` into a comparable, deterministically
/// ordered form.
fn weight_snapshot(trainer: &Trainer) -> Vec<(String, Vec<f32>)> {
    let data = trainer.var_map.data().lock().unwrap();
    let mut snapshot: Vec<(String, Vec<f32>)> = data
        .iter()
        .map(|(name, var)| {
            let values = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            (name.clone(), values)
        })
        .collect();
    snapshot.sort_by(|a, b| a.0.cmp(&b.0));
    snapshot
}

fn small_model_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 512,
        hidden_size: 32,
        num_layers: 2,
        num_heads: 4,
        intermediate_size: 64,
        max_seq_len: 32,
        hidden_act: "gelu".to_string(),
        initializer_range: 0.02,
        layer_norm_eps: 1e-5,
        pad_token_id: 0,
        bos_token_id: 101,
        eos_token_id: 102,
    }
}

/// A saved checkpoint must restore bit-identical weights into a fresh trainer.
#[test]
fn test_checkpoint_roundtrip_restores_weights() {
    let tmp_dir = tempdir().unwrap();
    let output_dir = tmp_dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    let config_for = |seed: u64| TrainingConfig {
        model_name: "test".to_string(),
        data_path: String::new(),
        output_dir: output_dir.to_str().unwrap().to_string(),
        epochs: 1,
        learning_rate: 1e-3,
        batch_size: 4,
        gradient_accumulation_steps: 1,
        max_seq_len: 32,
        precision: "f32".to_string(),
        warmup_steps: 0,
        weight_decay: 0.0,
        max_grad_norm: 1.0,
        seed,
        enable_load_monitoring: false,
        max_cpu_load: 0.8,
        load_check_interval: 10,
        max_swap_usage: 0.0,
        min_free_mem_mb: 2048,
        min_lr_ratio: 0.1,
    };

    let saved = Trainer::new(config_for(42), small_model_config()).unwrap();
    let checkpoint = output_dir.join("roundtrip");
    saved.save_checkpoint(checkpoint.to_str().unwrap()).unwrap();
    let expected = weight_snapshot(&saved);

    // Staging files must not survive a successful save.
    let leftovers: Vec<_> = std::fs::read_dir(&output_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "atomic save left temp files behind: {:?}",
        leftovers
    );

    // A different seed gives different initial weights, so the comparison below is
    // only satisfiable by actually loading the checkpoint.
    let mut restored = Trainer::new(config_for(7), small_model_config()).unwrap();
    assert_ne!(
        weight_snapshot(&restored),
        expected,
        "a differently-seeded trainer should not start with the saved weights"
    );

    restored
        .load_checkpoint(checkpoint.to_str().unwrap())
        .unwrap();
    assert_eq!(
        weight_snapshot(&restored),
        expected,
        "loaded weights should be identical to the saved ones"
    );
}

/// Training on a tiny, highly repetitive dataset must actually reduce the loss.
///
/// This is the end-to-end guard that the optimization objective has the right sign
/// and that gradients reach the weights: a model minimizing cross-entropy on eight
/// repeated sentences should fit them quickly.
#[test]
fn test_training_reduces_loss() {
    let tmp_dir = tempdir().unwrap();
    let data_path = tmp_dir.path().join("train.jsonl");
    let output_dir = tmp_dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    let mut file = File::create(&data_path).unwrap();
    for _ in 0..4 {
        for sample in [
            r#"{"text": "the cat sat on the mat"}"#,
            r#"{"text": "the dog ran on the floor"}"#,
        ] {
            writeln!(file, "{}", sample).unwrap();
        }
    }
    drop(file);

    let dataset =
        Dataset::from_jsonl(data_path.to_str().unwrap(), create_test_tokenizer(), 32).unwrap();

    // A small vocabulary keeps the softmax cheap enough to converge in a few epochs.
    let model_config = small_model_config();

    let train_config = TrainingConfig {
        model_name: "test".to_string(),
        data_path: data_path.to_str().unwrap().to_string(),
        output_dir: output_dir.to_str().unwrap().to_string(),
        epochs: 12,
        learning_rate: 1e-3,
        batch_size: 4,
        gradient_accumulation_steps: 1,
        max_seq_len: 32,
        precision: "f32".to_string(),
        warmup_steps: 0,
        weight_decay: 0.0,
        max_grad_norm: 1.0,
        seed: 42,
        enable_load_monitoring: false,
        max_cpu_load: 0.8,
        load_check_interval: 10,
        max_swap_usage: 0.0,
        min_free_mem_mb: 2048,
        min_lr_ratio: 0.1,
    };

    let mut trainer = Trainer::new(train_config, model_config).unwrap();
    let losses = trainer.train(&dataset).unwrap();

    assert!(
        losses.len() >= 2,
        "Need at least two epochs to compare loss, got {}",
        losses.len()
    );
    assert!(
        losses.iter().all(|l| l.is_finite() && *l > 0.0),
        "All epoch losses should be positive and finite, got {:?}",
        losses
    );

    // Require a real margin, not just noise. Observed decrease is ~6% over 12 epochs
    // and monotone, so a 1% floor leaves ample headroom.
    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        last < first * 0.99,
        "Loss should decrease over training: first epoch {:.4}, last epoch {:.4} (all: {:?})",
        first,
        last,
        losses
    );
}
