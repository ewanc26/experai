use candle_core::{Device, Tensor, DType};
use proptest::prelude::*;

use experai::data::{DatasetSample, DataCollator};
use experai::preprocessing::{PreprocessConfig, Preprocessor};
use experai::training::{compute_loss, compute_perplexity, TrainingConfig};
use experai::utils::{GradientAccumulator, check_value_in_range};

use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::Tokenizer;

fn make_preprocessor(config: PreprocessConfig) -> Preprocessor {
    let tokenizer = Tokenizer::new(BPE::default());
    Preprocessor::new(tokenizer, Some(config))
}

/// A self-contained byte-level BPE tokenizer with no merges.
///
/// The vocabulary is the full byte-level alphabet, so every possible input byte
/// maps to exactly one token and `encode` → `decode` is lossless. `BPE::default()`
/// has an *empty* vocabulary and encodes everything to zero tokens, which makes it
/// useless for round-trip assertions.
fn roundtrip_tokenizer() -> Tokenizer {
    let vocab: std::collections::HashMap<String, u32> = ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .expect("byte-level BPE should build");

    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(ByteLevel::new(false, false, false));
    tokenizer.with_decoder(ByteLevel::new(false, false, false));
    tokenizer
}

fn default_clean_config() -> PreprocessConfig {
    PreprocessConfig {
        lowercase: true,
        remove_extra_whitespace: true,
        remove_html_tags: true,
        remove_urls: true,
        remove_emails: true,
        min_length: 0,
        max_length: 1024,
        dedupe: false,
        clean: true,
    }
}

proptest! {
    #[test]
    fn tokenizer_roundtrip_ascii(text in "[a-zA-Z0-9 .,!?\\n]{1,200}") {
        let tokenizer = roundtrip_tokenizer();
        let encoding = tokenizer.encode(text.as_str(), true)
            .expect("encode should succeed");
        let decoded = tokenizer.decode(encoding.get_ids(), true)
            .expect("decode should succeed");
        // A byte-level vocabulary covers every input byte, so the round-trip is exact.
        prop_assert_eq!(&decoded, &text);
    }

    #[test]
    fn tokenizer_roundtrip_arbitrary_utf8(text in "\\PC{1,100}") {
        let tokenizer = roundtrip_tokenizer();
        let encoding = tokenizer.encode(text.as_str(), true)
            .expect("encode should succeed");
        let decoded = tokenizer.decode(encoding.get_ids(), true)
            .expect("decode should succeed");
        prop_assert_eq!(&decoded, &text);
    }

    #[test]
    fn collator_shape_invariants(
        batch_size in 1usize..32,
        seq_len in 1usize..256,
        pad_token_id in 0u32..1000,
    ) {
        let collator = DataCollator::new(pad_token_id, seq_len);
        let mut samples = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let len = (i % seq_len) + 1;
            let tokens: Vec<u32> = (0..len as u32).map(|x| x + 1).collect();
            samples.push(DatasetSample {
                text: format!("sample_{}", i),
                tokens,
                label: None,
                did: None,
            });
        }
        let (input_ids, attention_mask) = collator.collate(&samples).unwrap();

        let dims = input_ids.dims();
        prop_assert_eq!(dims.len(), 2, "input_ids should be 2D, got {:?}", dims);
        prop_assert_eq!(dims[0], batch_size);
        prop_assert!(dims[1] <= seq_len, "seq dim {} exceeds max_length {}", dims[1], seq_len);

        let mask_dims = attention_mask.dims();
        prop_assert_eq!(mask_dims, dims, "mask shape must match input_ids shape");

        let input_vec = input_ids.to_vec2::<u32>().unwrap();
        let mask_vec = attention_mask.to_vec2::<u32>().unwrap();

        for row_idx in 0..batch_size {
            let input_row = &input_vec[row_idx];
            let mask_row = &mask_vec[row_idx];
            let sample_tokens = &samples[row_idx].tokens;

            // Count real tokens (mask == 1)
            let real_count = mask_row.iter().filter(|&&v| v == 1).count();
            let pad_count = mask_row.iter().filter(|&&v| v == 0).count();
            prop_assert_eq!(real_count + pad_count, input_row.len());

            // Real tokens should match original (truncated to max_len)
            let max_len = dims[1];
            let expected_real = sample_tokens.len().min(max_len);
            prop_assert_eq!(real_count, expected_real);

            // All padding positions should have pad_token_id
            for &v in input_row.iter().skip(real_count) {
                prop_assert_eq!(v, pad_token_id, "padding mismatch");
            }

            // All real token positions should have mask=1, all padding mask=0
            for (i, &m) in mask_row.iter().take(real_count).enumerate() {
                prop_assert_eq!(m, 1u32, "real token at {} has mask 0", i);
            }
            for (i, &m) in mask_row.iter().enumerate().skip(real_count) {
                prop_assert_eq!(m, 0u32, "padding at {} has mask 1", i);
            }
        }
    }

    #[test]
    fn loss_is_nonnegative_finite(
        batch in 1usize..8,
        seq in 1usize..32,
        vocab in 2usize..64,
    ) {
        let logits = Tensor::rand(-1.0f32, 1.0, (batch, seq, vocab), &Device::Cpu).unwrap();
        let labels = Tensor::rand(0f32, vocab as f32 - 1.0, (batch, seq), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap();

        let loss = compute_loss(&logits, &labels).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap() as f64;

        prop_assert!(loss_val >= 0.0, "loss {} should be >= 0", loss_val);
        prop_assert!(loss_val.is_finite(), "loss {} should be finite", loss_val);
    }

    #[test]
    fn perplexity_properties(
        loss in 0.0f64..20.0,
    ) {
        let ppl = compute_perplexity(loss);
        prop_assert!(ppl >= 1.0, "perplexity {} should be >= 1.0", ppl);
        prop_assert!(ppl.is_finite(), "perplexity {} should be finite", ppl);
    }

    #[test]
    fn perplexity_monotonicity(
        loss_a in 0.0f64..10.0,
        loss_b in 0.0f64..10.0,
    ) {
        prop_assume!(loss_a < loss_b);
        let ppl_a = compute_perplexity(loss_a);
        let ppl_b = compute_perplexity(loss_b);
        prop_assert!(ppl_a < ppl_b,
            "perplexity should be monotonically increasing: loss_a={}, loss_b={}, ppl_a={}, ppl_b={}",
            loss_a, loss_b, ppl_a, ppl_b);
    }

    #[test]
    fn preprocess_never_panics(
        text in "[a-zA-Z0-9 .,!?\\n<>@/]{0,500}",
    ) {
        let p = make_preprocessor(default_clean_config());
        let _ = p.clean_text(&text);
    }

    #[test]
    fn preprocess_lowercase(
        text in "[A-Za-z0-9 .,!?]{1,200}",
    ) {
        let config = PreprocessConfig {
            lowercase: true,
            remove_extra_whitespace: false,
            remove_html_tags: false,
            remove_urls: false,
            remove_emails: false,
            min_length: 0,
            max_length: 1024,
            dedupe: false,
            clean: true,
        };
        let p = make_preprocessor(config);
        let result = p.clean_text(&text);
        let lower = result.to_lowercase();
        prop_assert_eq!(result, lower,
            "result should be lowercase");
    }

    #[test]
    fn preprocess_no_growth(
        text in "[a-zA-Z0-9 .,!?\\n]{1,200}",
    ) {
        let p = make_preprocessor(default_clean_config());
        let result = p.clean_text(&text);
        prop_assert!(result.len() <= text.len(),
            "cleaned text ({} chars) should not be longer than input ({} chars)",
            result.len(), text.len());
    }

    #[test]
    fn gradient_accumulator_step_count(
        steps_per_accum in 1usize..16,
        total_steps in 1usize..100,
    ) {
        let mut acc = GradientAccumulator::new(steps_per_accum);
        let mut fire_count = 0usize;
        for _ in 0..total_steps {
            if acc.step() {
                fire_count += 1;
            }
        }
        prop_assert_eq!(fire_count, total_steps / steps_per_accum,
            "should fire exactly every {} steps out of {} total (got {} fires)",
            steps_per_accum, total_steps, fire_count);
    }

    #[test]
    fn default_config_valid(
        epochs in 1usize..100,
        lr in 1e-7f64..1e-1,
        batch in 1usize..256,
        grad_accum in 1usize..64,
        warmup in 0usize..10000,
    ) {
        let config = TrainingConfig {
            epochs,
            learning_rate: lr,
            batch_size: batch,
            gradient_accumulation_steps: grad_accum,
            warmup_steps: warmup,
            ..TrainingConfig::default()
        };

        // Default config should always be valid
        prop_assert!(config.epochs > 0);
        prop_assert!(config.learning_rate > 0.0);
        prop_assert!(config.batch_size > 0);
        prop_assert!(config.gradient_accumulation_steps > 0);
        prop_assert!(config.max_grad_norm > 0.0);

        // Perplexity relationship
        let ppl = compute_perplexity(config.learning_rate);
        prop_assert!(ppl >= 1.0);
    }

    #[test]
    fn check_value_in_range_true(
        value in 0i64..100,
        min in 0i64..50,
        max in 50i64..100,
    ) {
        prop_assume!(min <= value && value <= max);
        prop_assert!(check_value_in_range(value, min, max),
            "check_value_in_range({}, {}, {}) should be true", value, min, max);
    }

    #[test]
    fn check_value_in_range_false(
        value in 100i64..200,
        min in 0i64..50,
        max in 50i64..100,
    ) {
        prop_assert!(!check_value_in_range(value, min, max),
            "check_value_in_range({}, {}, {}) should be false", value, min, max);
    }
}
