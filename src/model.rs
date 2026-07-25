use crate::errors::ExperaiError;
use candle_core::{Device, DType, Tensor};
use candle_nn::{linear_no_bias, Dropout, Embedding, Linear, Module, VarBuilder, VarMap};
use tracing::{debug, info, trace};

/// Configuration for a transformer language model.
///
/// Defines all hyperparameters needed to construct a GPT-2-style transformer
/// including vocabulary size, hidden dimensions, layer count, and special
/// token identifiers.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub hidden_act: String,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            max_seq_len: 2048,
            hidden_act: "gelu".to_string(),
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }
}

/// A GPT-2-style transformer language model.
///
/// Composed of token embeddings, a stack of [`TransformerLayer`] blocks,
/// a final RMSNorm, and a linear projection to vocabulary logits.
pub struct TransformerModel {
    pub config: ModelConfig,
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub norm: candle_nn::RmsNorm,
    pub lm_head: Linear,
    pub dropout: Dropout,
}

/// A single transformer block with pre-norm self-attention and an MLP.
///
/// Uses pre-LayerNorm ordering: norm → attention → residual →
/// norm → MLP → residual.
pub struct TransformerLayer {
    self_attn: MultiHeadAttention,
    mlp: Mlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

/// Multi-head self-attention with separate Q/K/V/O projections.
struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

/// SwiGLU-style MLP with gated linear units.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

/// Construct a [`TransformerModel`] from a [`ModelConfig`] and weight [`VarBuilder`].
///
/// Allocates embeddings, transformer layers, final norm, and LM head
/// in the order expected by the weight file layout. After construction,
/// weights are reinitialised using the config's `initializer_range`.
pub fn build_model(config: &ModelConfig, vb: VarBuilder) -> Result<TransformerModel, ExperaiError> {
    info!(
        "Building model: vocab={}, hidden={}, layers={}, heads={}, intermediate={}, max_seq={}",
        config.vocab_size, config.hidden_size, config.num_layers, config.num_heads, config.intermediate_size, config.max_seq_len
    );

    let embedding =
        candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embedding"))?;
    debug!("Created embedding: {}x{}", config.vocab_size, config.hidden_size);

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = TransformerLayer::new(config, vb.pp(format!("layers.{}", i)))?;
        layers.push(layer);
    }
    debug!("Created {} transformer layers", config.num_layers);

    let norm = candle_nn::rms_norm(config.hidden_size, config.layer_norm_eps, vb.pp("norm"))?;
    let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
    let dropout = Dropout::new(0.1);

    let head_dim = config.hidden_size / config.num_heads;
    let params = (config.hidden_size * config.vocab_size)
        + config.num_layers
            * (4 * config.hidden_size * config.hidden_size
                + 4 * config.hidden_size
                + 2 * config.intermediate_size * config.hidden_size)
        + config.hidden_size * config.vocab_size;
    info!(
        "Model built: {} params ({:.1}M), head_dim={}",
        params,
        params as f64 / 1e6,
        head_dim
    );

    Ok(TransformerModel {
        config: config.clone(),
        embedding,
        layers,
        norm,
        lm_head,
        dropout,
    })
}

/// Reinitialise all named parameters in `varmap` according to their role.
///
/// - Embeddings and lm_head: normal distribution with `initializer_range` std dev.
/// - Attention projections (q/k/v/o): Xavier with scale `sqrt(2 / (hidden * 2))`.
/// - MLP projections (gate/up/down): Xavier with scale `sqrt(2 / (hidden + intermediate))`.
/// - RMSNorm weights: ones.
pub fn init_weights(varmap: &VarMap, config: &ModelConfig, device: &Device) -> Result<(), ExperaiError> {
    let dtype = DType::F32;
    let h = config.hidden_size as f64;
    let inter = config.intermediate_size as f64;
    let std = config.initializer_range;

    let attn_scale = (2.0 / (h + h)).sqrt();
    let mlp_scale = (2.0 / (h + inter)).sqrt();

    let data = varmap.data();
    let tensor_data = data.lock().map_err(|e| {
        ExperaiError::ModelLoad(format!("variable map lock was poisoned by another thread: {e}"))
    })?;

    for (name, var) in tensor_data.iter() {
        let shape = var.shape().dims().to_vec();

        let new_tensor = if name.contains("layernorm") || name.ends_with("norm") {
            // RMSNorm: weight = ones
            Tensor::ones(&*shape, dtype, device)?
        } else if name.starts_with("embedding") {
            // Embedding: normal with initializer_range
            Tensor::randn(0.0, std, &*shape, device)?
        } else if name.contains("self_attn")
            && (name.contains("q_proj") || name.contains("k_proj")
                || name.contains("v_proj") || name.contains("o_proj"))
        {
            // Attention projections: Xavier
            Tensor::randn(0.0, attn_scale, &*shape, device)?
        } else if name.contains("mlp")
            && (name.contains("gate_proj") || name.contains("up_proj")
                || name.contains("down_proj"))
        {
            // MLP projections: Xavier
            Tensor::randn(0.0, mlp_scale, &*shape, device)?
        } else if name.starts_with("lm_head") {
            // LM head: normal with initializer_range
            Tensor::randn(0.0, std, &*shape, device)?
        } else {
            // Fallback: normal with initializer_range
            Tensor::randn(0.0, std, &*shape, device)?
        };

        var.set(&new_tensor)?;
    }

    info!("Applied custom weight initialisation (initializer_range={})", config.initializer_range);
    Ok(())
}

impl TransformerLayer {
    fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self, ExperaiError> {
        let head_dim = config.hidden_size / config.num_heads;
        trace!("Creating transformer layer: head_dim={}", head_dim);
        let self_attn = MultiHeadAttention::new(config, head_dim, vb.pp("self_attn"))?;
        let mlp = Mlp::new(config, vb.pp("mlp"))?;
        let input_layernorm = candle_nn::rms_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = candle_nn::rms_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Pre-norm transformer block forward pass.
    ///
    /// Applies two residual sub-layers:
    /// 1. `x + attn(norm(x))`
    /// 2. `h + mlp(norm(h))`
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h)?;
        let h = (h + residual)?;

        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        let h = (h + residual)?;

        Ok(h)
    }
}

impl MultiHeadAttention {
    fn new(config: &ModelConfig, head_dim: usize, vb: VarBuilder) -> Result<Self, ExperaiError> {
        let q_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_heads,
            head_dim,
        })
    }

    /// Multi-head self-attention forward pass.
    ///
    /// Projects input to Q/K/V, reshapes to per-head dimensions, computes
    /// scaled dot-product attention with a causal mask, then projects back.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = x.dims3()?;
        trace!("Attention forward: input shape {:?}", x.shape());

        // Linear projections: [B, T, D] → [B, T, D]
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, heads, seq, head_dim)
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?;

        // Causal mask: upper triangle filled with -inf to prevent attending to future tokens
        let mask = causal_mask(seq_len, x.device())?;
        let scores = (q
            .contiguous()?
            .matmul(&k_t.contiguous()?)?
            .broadcast_add(&mask)?
            * scale)?;

        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let out = attn.matmul(&v.contiguous()?)?;

        // Merge heads: (batch, heads, seq, head_dim) → (batch, seq, hidden)
        let out = out
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, hidden_size))?;

        let out = self.o_proj.forward(&out)?;
        Ok(out)
    }
}

/// Build a causal attention mask of shape `[1, 1, seq_len, seq_len]`.
///
/// Upper-triangular positions (future tokens) are filled with `-inf` so
/// they are zeroed out after softmax. Lower-triangular and diagonal
/// positions are `0.0`.
pub(crate) fn causal_mask(seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
    // Create an upper triangular mask of -inf
    let mask_vals: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
        })
        .collect();
    let mask = Tensor::from_slice(&mask_vals, (seq_len, seq_len), device)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    Ok(mask)
}

impl Mlp {
    fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self, ExperaiError> {
        trace!(
            "Creating MLP: hidden={}, intermediate={}",
            config.hidden_size,
            config.intermediate_size
        );
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

/// SwiGLU MLP: `down_proj(gelu(gate(x)) * up(x))`.
///
/// The gate branch applies GELU activation; the result is element-wise
/// multiplied with the up projection before the down projection.
impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = gate.gelu_erf()?;
        let up = self.up_proj.forward(x)?;
        // SwiGLU gating: element-wise product of activated gate and up path
        let gate_up = (gate * up)?;
        let out = self.down_proj.forward(&gate_up)?;
        Ok(out)
    }
}

impl Module for TransformerModel {
    /// Full forward pass: embed tokens → transformer stack → norm → logits.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        trace!("TransformerModel forward: input shape {:?}", x.shape());
        let mut h = self.embedding.forward(x)?;
        h = self.dropout.forward(&h, false)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            trace!("Layer {} output shape: {:?}", i, h.shape());
        }
        h = self.norm.forward(&h)?;
        let logits = self.lm_head.forward(&h)?;
        trace!("TransformerModel forward: output shape {:?}", logits.shape());
        Ok(logits)
    }
}
