use super::{attn::BertSelfAttention, mlp::MLP};

use std::io::{BufRead, Seek};

use ratchet::{Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{LayerNorm, Linear, Module};

#[cfg(target_arch = "wasm32")]
use crate::{ratchet_from_gguf_web, TensorMap};

pub struct EncoderLayer {
    attention: BertSelfAttention,
    mlp: MLP,
    norm: LayerNorm,
}

impl EncoderLayer {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Header,
        reader: &mut R,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let attention = BertSelfAttention::load(disk_model, reader, layer_index, device)?;
        let lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            disk_model.tensor(reader, &key, device)
        };
        Self::load_inner(attention, disk_model, lt, device)
    }

    fn load_inner<F>(
        attention: BertSelfAttention,
        header: &Header,
        mut lt: F,
        _: &Device,
    ) -> anyhow::Result<Self>
    where
        F: FnMut(&str) -> anyhow::Result<Tensor>,
    {
        let eps = header
            .metadata
            .get("bert.attention.layer_norm_epsilon")
            .unwrap()
            .to_f32()?;

        let mlp = MLP::new(
            Linear::new(lt("ffn_up.weight")?, Some(lt("ffn_up.bias")?)),
            Linear::new(lt("ffn_up.weight")?, Some(lt("ffn_up.bias")?)),
        );

        let norm = LayerNorm::new(
            lt("layer_output_norm.weight")?,
            Some(lt("layer_output_norm.bias")?),
            eps,
        );
        Ok(Self {
            attention,
            mlp,
            norm,
        })
    }
}

impl Module for EncoderLayer {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let attention = self.attention.schedule(input)?;
        let mlp = self.mlp.schedule(attention.clone())?;
        //Bypass and norm
        let norm = self.norm.schedule(mlp.add(attention)?)?;
        Ok(norm)
    }
}
