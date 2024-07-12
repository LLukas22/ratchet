use ratchet::{shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{Embedding, LayerNorm, Module};
use std::io::{BufRead, Seek};

#[derive(Clone)]
pub struct EmbeddingInput {
    pub input_ids: Tensor,
    pub token_type_ids: Tensor,
}

#[cfg(target_arch = "wasm32")]
use {crate::ratchet_from_gguf_web, crate::TensorMap};

pub struct BertEmbedding {
    token_type_embedding: Embedding,
    position_embedding: Embedding,
    word_embedding: Embedding,
    norm: LayerNorm,
}

impl Module for BertEmbedding {
    type Input = EmbeddingInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let [_, seq_len]: [usize; 2] = input.input_ids.shape().try_into()?;

        let input_embeddings = self.word_embedding.schedule(input.input_ids)?;
        let token_type_embeddings = self.token_type_embedding.schedule(input.token_type_ids)?;

        let embeddings = input_embeddings.add(token_type_embeddings)?;

        let position_ids = (0..seq_len as i32).collect::<Vec<_>>();
        let position_ids =
            Tensor::from_data(position_ids, shape![seq_len], embeddings.device().clone());
        let position_embeddings = self.position_embedding.schedule(position_ids)?;

        let embeddings = embeddings.add(position_embeddings)?;

        let embeddings = self.norm.schedule(embeddings)?;
        Ok(embeddings)
    }
}

impl BertEmbedding {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Header,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| disk_model.tensor(reader, &name, device);
        Self::load_inner(disk_model, lt)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_web(
        header: &Header,
        tensors: &mut TensorMap,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let tensor = tensors
                .remove(&name)
                .ok_or_else(|| anyhow::anyhow!("missing tensor"))?;
            ratchet_from_gguf_web(tensor, device)
        };
        Self::load_inner(header, lt)
    }

    fn load_inner<F>(header: &Header, mut lt: F) -> anyhow::Result<Self>
    where
        F: FnMut(&str) -> anyhow::Result<Tensor>,
    {
        let eps: f32 = header
            .metadata
            .get("bert.attention.layer_norm_epsilon")
            .unwrap()
            .to_f32()?;
        let token_type_embedding = Embedding::new(lt("token_types.weight")?);
        let position_embedding = Embedding::new(lt("position_embd.weight")?);
        let word_embedding = Embedding::new(lt("token_embd.weight")?);
        let norm = LayerNorm::new(
            lt("token_embd_norm.weight")?,
            Some(lt("token_embd_norm.bias")?),
            eps,
        );
        Ok(Self {
            token_type_embedding,
            position_embedding,
            word_embedding,
            norm,
        })
    }
}
