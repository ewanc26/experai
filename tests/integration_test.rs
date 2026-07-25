use std::fs::File;
use std::io::Write;
use std::collections::HashMap;
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
        "the", "a", "cat", "dog", "sat", "ran", "on", "in", "at",
        "big", "small", "fast", "slow", "mat", "floor", "house",
        "is", "was", "are", "this", "that", "it", "to", "and",
        "of", "for", "with", "by", "from", "or", "but", "not",
        "you", "i", "we", "he", "she", "they", "my", "his", "her",
        "good", "bad", "red", "blue", "one", "two", "three",
        "hello", "world", "how", "what", "when", "where",
        "do", "have", "can", "will", "would", "should",
        "like", "love", "need", "want", "go", "come", "see", "look",
        "make", "take", "give", "get", "use", "find", "tell", "ask",
        "work", "try", "call", "help", "start", "play", "move",
        "live", "become", "leave", "put", "mean", "keep",
        "let", "begin", "seem", "show", "hear", "turn",
        "night", "city", "tree", "road", "car", "book",
        "quick", "brown", "fox", "jumps", "over", "lazy",
        "jumped", "very", "quickly", "each", "other",
        "learning", "models", "require", "large", "datasets",
        "training", "rust", "systems", "programming", "language",
        "focused", "safety", "natural", "processing", "enables",
        "computers", "understand", "text", "deep", "revolutionized",
        "artificial", "intelligence",
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

    let dataset =
        Dataset::from_jsonl(data_path.to_str().unwrap(), tokenizer, 128).unwrap();
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
    };

    let mut trainer = Trainer::new(train_config, model_config).unwrap();
    let losses = trainer.train(&dataset).unwrap();

    assert!(!losses.is_empty(), "Should have at least one loss value");
    let final_loss = losses.last().unwrap();
    assert!(
        final_loss.is_finite(),
        "Loss should be finite, got {}",
        final_loss
    );
    assert!(
        *final_loss < 0.0,
        "Loss (NLL) should be negative, got {}",
        final_loss
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
