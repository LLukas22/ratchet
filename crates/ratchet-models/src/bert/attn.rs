use std::io::{BufRead, Seek};

use ratchet::{prelude::shape, rvec, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{LayerNorm, Linear, Module};

#[cfg(target_arch = "wasm32")]
use crate::{ratchet_from_gguf_web, TensorMap};

pub struct BertAttentionInput {
    pub x: Tensor,
    pub mask: Option<Tensor>,
}

#[derive(Debug)]
pub struct BertSelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear, //layer.wo + layer.bo
    norm: LayerNorm,
    n_heads: u32,
    attention_head_size: u32, // = config.hidden_size / config.num_attention_heads;
    softmax_scale: Tensor,
}

impl BertSelfAttention {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Header,
        reader: &mut R,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            disk_model.tensor(reader, &key, device)
        };
        Self::load_inner(disk_model, lt, device)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_web(
        header: &Header,
        tensors: &mut TensorMap,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            let tensor = tensors
                .remove(&key)
                .ok_or_else(|| anyhow::anyhow!("missing tensor"))?;
            ratchet_from_gguf_web(tensor, device)
        };
        Self::load_inner(header, lt, device)
    }

    fn load_inner<F>(header: &Header, mut lt: F, device: &Device) -> anyhow::Result<Self>
    where
        F: FnMut(&str) -> anyhow::Result<Tensor>,
    {
        let q = Linear::new(lt("attn_q.weight")?, Some(lt("attn_q.bias")?));
        let k = Linear::new(lt("attn_k.weight")?, Some(lt("attn_k.bias")?));
        let v = Linear::new(lt("attn_v.weight")?, Some(lt("attn_v.bias")?));
        let o = Linear::new(lt("attn_output.weight")?, Some(lt("attn_output.bias")?));

        let eps = header
            .metadata
            .get("bert.attention.layer_norm_epsilon")
            .unwrap()
            .to_f32()?;

        let norm = LayerNorm::new(
            lt("attn_output_norm.weight")?,
            Some(lt("attn_output_norm.bias")?),
            eps,
        );

        let n_heads = header
            .metadata
            .get("bert.attention.head_count")
            .unwrap()
            .to_u32()?;

        let embedding_length = header
            .metadata
            .get("bert.embedding_length")
            .unwrap()
            .to_u32()?;

        let attention_head_size = embedding_length / n_heads;

        let scale_val = 1.0 / (attention_head_size as f32).sqrt();
        let softmax_scale = Tensor::from_data([scale_val], shape![1], device.clone());

        Ok(Self {
            q,
            k,
            v,
            o,
            norm,
            n_heads,
            attention_head_size,
            softmax_scale,
        })
    }
}

impl Module for BertSelfAttention {
    type Input = BertAttentionInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let BertAttentionInput { x, mask } = input;

        let [batch_size, seq_len, embedding_dim]: [usize; 3] = x.shape().try_into()?;

        let residual = x.clone();
        let q = self.q.schedule(x.clone())?;
        let k = self.k.schedule(x.clone())?;
        let v = self.v.schedule(x.clone())?;

        let attention_shapes = shape![
            batch_size as _,
            seq_len,
            self.n_heads as _,
            self.attention_head_size as _
        ];
        let query_states = q.view(attention_shapes.clone())?.permute(&[0, 2, 1, 3])?;
        let key_states = k.view(attention_shapes.clone())?.permute(&[0, 2, 1, 3])?;
        let value_states = v.view(attention_shapes)?.permute(&[0, 2, 1, 3])?;

        let scores = query_states.matmul(key_states, false, true)?;
        let mut attention = scores.mul(self.softmax_scale.clone())?;
        if let Some(m) = mask {
            attention = attention.add(m)?;
        }
        attention = attention.softmax(3)?;

        let output = attention
            .matmul(value_states, false, false)?
            .permute(&[0, 2, 1, 3])?;
        let output = output.view(shape![batch_size as _, seq_len, embedding_dim])?;

        let output = self.o.schedule(output)?;

        // Add residual
        let output = output.add(residual)?;

        // attention layer norm
        let output = self.norm.schedule(output)?;

        Ok(output)
    }
}
