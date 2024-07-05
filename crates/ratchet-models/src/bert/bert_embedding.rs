
use ratchet::{shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{Embedding,LayerNorm, Module};
use std::io::{BufRead, Seek};

pub struct EmbeddingInput{
    input_ids: Tensor,
    token_type_ids: Tensor
}

#[cfg(target_arch = "wasm32")]
use {crate::ratchet_from_gguf_web, crate::TensorMap};

pub struct BertEmbedding{
    token_type_embedding: Embedding,
    position_embedding: Embedding,
    word_embedding: Embedding,
    norm:LayerNorm
}

impl Module for BertEmbedding{
    type Input = EmbeddingInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let [_, seq_len]: [usize; 2] = input.input_ids.shape().try_into()?;

        let input_embeddings = self.word_embedding.schedule(input.input_ids)?;
        let token_type_embeddings = self.token_type_embedding.schedule(input.token_type_ids)?;

        let embeddings = input_embeddings.add(token_type_embeddings)?;

        let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
        let position_ids = Tensor::from_data(
            position_ids,
            shape![seq_len],
            embeddings.device().clone(),
        );
        let position_embeddings = self.position_embedding.schedule(position_ids)?;

        // Will `add` broadcast here?
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

        let eps: f32 = disk_model.metadata.get("bert.attention.layer_norm_epsilon").unwrap().to_f32()?;
        let token_type_embedding = Embedding::new(disk_model.tensor(reader, "token_types.weight", device)?);
        let position_embedding = Embedding::new(disk_model.tensor(reader, "position_embd.weight", device)?);
        let word_embedding = Embedding::new(disk_model.tensor(reader, "token_embd.weight", device)?);
        let norm = LayerNorm::new(
            disk_model.tensor(reader, "token_embd_norm.weight", device)?,
            Some(disk_model.tensor(reader, "token_embd_norm.bias", device)?),
            eps
        );
        Ok(Self{
            token_type_embedding,
            position_embedding,
            word_embedding,
            norm
        })
    }
}