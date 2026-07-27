use crate::data::{DataCollator, DatasetSample};
use crate::model::{apply_rope, build_model, causal_mask, init_weights, rope_cos_sin, ModelConfig};
use crate::training::TrainingConfig;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use tokenizers::Tokenizer;

fn small_model_config_for_seed_test() -> ModelConfig {
    ModelConfig {
        vocab_size: 64,
        hidden_size: 16,
        num_layers: 2,
        num_heads: 4,
        intermediate_size: 32,
        max_seq_len: 16,
        hidden_act: "gelu".to_string(),
        initializer_range: 0.02,
        layer_norm_eps: 1e-5,
        pad_token_id: 0,
        bos_token_id: 1,
        eos_token_id: 2,
    }
}

fn init_and_snapshot(seed: u64) -> Vec<(String, Vec<f32>)> {
    let device = Device::Cpu;
    let config = small_model_config_for_seed_test();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    build_model(&config, vb).expect("failed to build model");
    init_weights(&varmap, &config, &device, seed).expect("init_weights failed");

    let data = varmap.data().lock().unwrap();
    let mut snapshot: Vec<(String, Vec<f32>)> = data
        .iter()
        .map(|(name, var)| {
            (
                name.clone(),
                var.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            )
        })
        .collect();
    snapshot.sort_by(|a, b| a.0.cmp(&b.0));
    snapshot
}

#[test]
fn test_init_weights_same_seed_is_reproducible() {
    assert_eq!(
        init_and_snapshot(42),
        init_and_snapshot(42),
        "same seed should produce identical weight init"
    );
}

#[test]
fn test_init_weights_different_seeds_diverge() {
    assert_ne!(
        init_and_snapshot(42),
        init_and_snapshot(7),
        "different seeds should produce different weight init"
    );
}

#[test]
fn test_init_weights_norm_is_ones() {
    let device = Device::Cpu;
    let config = small_model_config_for_seed_test();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    build_model(&config, vb).expect("failed to build model");
    init_weights(&varmap, &config, &device, 42).expect("init_weights failed");

    let data = varmap.data().lock().unwrap();
    // Regression check for the final RMSNorm's parameter path ("norm.weight"),
    // which used to fall through the role-matching logic (`.ends_with("norm")`
    // doesn't match a path ending in ".weight") into the generic Xavier/normal
    // fallback instead of the ones-init RMSNorm expects.
    let norm_weight = data
        .get("norm.weight")
        .expect("norm.weight should exist")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        norm_weight.iter().all(|&v| v == 1.0),
        "final RMSNorm weight should be initialised to ones, got {:?}",
        norm_weight
    );
}

#[test]
fn test_training_config_defaults() {
    let config = TrainingConfig::default();
    assert_eq!(config.epochs, 3);
    assert_eq!(config.learning_rate, 5e-5);
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.gradient_accumulation_steps, 8);
    assert_eq!(config.seed, 42);
}

#[test]
fn test_model_config_defaults() {
    let config = ModelConfig::default();
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_layers, 12);
    assert_eq!(config.num_heads, 12);
}

#[test]
fn test_data_collator() {
    let collator = DataCollator::new(0, 64);
    let samples = vec![
        DatasetSample {
            text: "hello world".to_string(),
            tokens: vec![5, 10, 2],
            label: None,
            did: None,
        },
        DatasetSample {
            text: "foo bar".to_string(),
            tokens: vec![8, 12],
            label: None,
            did: None,
        },
    ];
    let (input_ids, _attention_mask) = collator.collate(&samples).unwrap();
    let (batch, seq) = input_ids.dims2().unwrap();
    assert_eq!(batch, 2);
    assert_eq!(seq, 3);
}

#[test]
fn test_gradient_accumulator() {
    let mut acc = crate::utils::GradientAccumulator::new(4);
    assert!(!acc.step());
    assert!(!acc.step());
    assert!(!acc.step());
    assert!(acc.step());
    assert!(!acc.step());
}

#[test]
fn test_check_value_in_range() {
    assert!(crate::utils::check_value_in_range(5, 0, 10));
    assert!(!crate::utils::check_value_in_range(15, 0, 10));
    assert!(crate::utils::check_value_in_range(0, 0, 10));
    assert!(crate::utils::check_value_in_range(10, 0, 10));
}

#[test]
fn test_model_forward_pass() {
    let config = ModelConfig {
        vocab_size: 100,
        hidden_size: 64,
        num_layers: 2,
        num_heads: 4,
        intermediate_size: 128,
        max_seq_len: 64,
        hidden_act: "gelu".to_string(),
        initializer_range: 0.02,
        layer_norm_eps: 1e-5,
        pad_token_id: 0,
        bos_token_id: 1,
        eos_token_id: 2,
    };

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = build_model(&config, vb).expect("failed to build model");

    let input_ids = Tensor::rand(0f32, 100f32, (1, 10), &device)
        .expect("failed to create input")
        .to_dtype(DType::U32)
        .expect("failed to cast to u32");
    let output = model.forward(&input_ids).expect("forward pass failed");

    assert_eq!(output.dims(), &[1, 10, 100]);
    let has_nan = output
        .to_vec3::<f32>()
        .unwrap()
        .iter()
        .flatten()
        .flatten()
        .any(|v| v.is_nan());
    assert!(!has_nan, "output contains NaN values");
}

#[test]
fn test_causal_mask() {
    let device = Device::Cpu;
    let seq_len = 5;
    let mask = causal_mask(seq_len, &device).expect("causal_mask failed");

    assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);

    let mask_2d = mask.reshape((seq_len, seq_len)).expect("reshape failed");
    let mask_vec = mask_2d.to_vec2::<f32>().unwrap();
    for (i, row) in mask_vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if j > i {
                assert!(
                    val.is_infinite() && val < 0.0,
                    "expected -inf at ({},{}) but got {}",
                    i,
                    j,
                    val
                );
            } else {
                assert_eq!(val, 0.0, "expected 0.0 at ({},{}) but got {}", i, j, val);
            }
        }
    }
}

#[test]
fn test_rope_matches_candle_reference() {
    let device = Device::Cpu;
    let (seq_len, head_dim) = (6, 8);
    let (cos, sin) = rope_cos_sin(seq_len, head_dim, &device).expect("rope_cos_sin failed");
    let x = Tensor::randn(0f32, 1f32, (2, 3, seq_len, head_dim), &device)
        .expect("failed to create input");

    let ours = apply_rope(&x, &cos, &sin).expect("apply_rope failed");
    // rope_slow is candle_nn's differentiable reference implementation (as opposed to
    // `rope`/`rope_i`, which use a backward-less custom kernel); our hand-rolled
    // version should agree with it bit-for-bit up to float error.
    let reference =
        candle_nn::rotary_emb::rope_slow(&x, &cos, &sin).expect("reference rope_slow failed");

    let max_diff = (ours - reference)
        .expect("subtraction failed")
        .abs()
        .expect("abs failed")
        .max_all()
        .expect("max_all failed")
        .to_scalar::<f32>()
        .expect("to_scalar failed");
    assert!(max_diff < 1e-5, "rope diverges from reference: {max_diff}");
}

#[test]
fn test_rope_preserves_vector_norm() {
    // Rotation is norm-preserving: ||RoPE(x)|| == ||x|| for every (batch, head, position).
    let device = Device::Cpu;
    let (seq_len, head_dim) = (10, 16);
    let (cos, sin) = rope_cos_sin(seq_len, head_dim, &device).expect("rope_cos_sin failed");
    let x = Tensor::randn(0f32, 1f32, (2, 4, seq_len, head_dim), &device)
        .expect("failed to create input");

    let rotated = apply_rope(&x, &cos, &sin).expect("apply_rope failed");

    let norm_before = x.sqr().unwrap().sum(3).unwrap();
    let norm_after = rotated.sqr().unwrap().sum(3).unwrap();
    let max_diff = (norm_before - norm_after)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        max_diff < 1e-4,
        "RoPE should preserve per-vector norm, diff: {max_diff}"
    );
}

#[test]
fn test_rope_gradients_flow_to_input() {
    // Regression guard for the bug this replaces: `candle_nn::rotary_emb::rope`/`rope_i`
    // register no backward pass, which would silently zero gradients to q/k projections
    // during training. Confirm gradients actually reach `x` through `apply_rope`.
    let device = Device::Cpu;
    let (seq_len, head_dim) = (4, 8);
    let (cos, sin) = rope_cos_sin(seq_len, head_dim, &device).expect("rope_cos_sin failed");

    let x = Tensor::randn(0f32, 1f32, (1, 1, seq_len, head_dim), &device)
        .expect("failed to create input")
        .to_dtype(DType::F32)
        .unwrap();
    let x = candle_core::Var::from_tensor(&x).unwrap();

    let out = apply_rope(x.as_tensor(), &cos, &sin).expect("apply_rope failed");
    let loss = out.sqr().unwrap().sum_all().unwrap();
    let grads = loss.backward().expect("backward failed");

    let grad = grads
        .get(x.as_tensor())
        .expect("no gradient recorded for rope input");
    let grad_norm = grad
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(grad_norm > 0.0, "expected nonzero gradient through RoPE");
}

// Preprocessing unit tests

fn make_preprocessor(
    config: crate::preprocessing::PreprocessConfig,
) -> crate::preprocessing::Preprocessor {
    use tokenizers::models::bpe::BPE;
    let tokenizer = Tokenizer::new(BPE::default());
    crate::preprocessing::Preprocessor::new(tokenizer, Some(config))
}

fn preprocessor_for(
    cleanup_fn: impl FnOnce(&mut crate::preprocessing::PreprocessConfig),
) -> crate::preprocessing::Preprocessor {
    let mut config = crate::preprocessing::PreprocessConfig {
        lowercase: false,
        remove_extra_whitespace: false,
        remove_html_tags: false,
        remove_urls: false,
        remove_emails: false,
        min_length: 0,
        max_length: 1024,
        dedupe: false,
        clean: false,
    };
    cleanup_fn(&mut config);
    make_preprocessor(config)
}

#[test]
fn test_clean_text_lowercase() {
    let p = preprocessor_for(|c| c.lowercase = true);
    assert_eq!(p.clean_text("Hello WORLD"), "hello world");
    assert_eq!(p.clean_text("FOO"), "foo");
    assert_eq!(p.clean_text("MiXeD"), "mixed");
}

#[test]
fn test_clean_text_remove_html() {
    let p = preprocessor_for(|c| c.remove_html_tags = true);
    assert_eq!(p.clean_text("Hello <b>world</b>!"), "Hello world!");
    assert_eq!(
        p.clean_text("A <div class=\"x\">nested</div> test."),
        "A nested test."
    );
    assert_eq!(
        p.clean_text("<p>Paragraph</p><br/>Line break"),
        "ParagraphLine break"
    );
}

#[test]
fn test_clean_text_remove_urls() {
    let p = preprocessor_for(|c| c.remove_urls = true);
    assert_eq!(
        p.clean_text("Visit https://example.com for more"),
        "Visit [URL] for more"
    );
    assert_eq!(
        p.clean_text("http://a.com and https://b.org/path?q=1"),
        "[URL] and [URL]"
    );
    assert_eq!(p.clean_text("no urls here"), "no urls here");
}

#[test]
fn test_clean_text_remove_emails() {
    let p = preprocessor_for(|c| c.remove_emails = true);
    assert_eq!(
        p.clean_text("Contact user@example.com for info"),
        "Contact [EMAIL] for info"
    );
    assert_eq!(p.clean_text("a@b.com and x@y.org"), "[EMAIL] and [EMAIL]");
    assert_eq!(p.clean_text("no email here"), "no email here");
}

#[test]
fn test_clean_text_remove_whitespace() {
    let p = preprocessor_for(|c| c.remove_extra_whitespace = true);
    assert_eq!(p.clean_text("hello   world"), "hello world");
    assert_eq!(
        p.clean_text("  leading and trailing  "),
        "leading and trailing"
    );
    assert_eq!(p.clean_text("tabs\tand\nnewlines"), "tabs and newlines");
    assert_eq!(p.clean_text("one"), "one");
}

// Property-based tests live in `tests/property_tests.rs`, which exercises the
// same helpers through the public API.
