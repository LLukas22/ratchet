
use ratchet::{shape, DType, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{Embedding, KVCache, KVEntry, LayerNorm, Linear, Module};


pub struct EmbeddingInput{
    input_ids: Tensor,
    token_type_ids: Tensor
}

pub struct BertEmbedding{
    token_type_embedding: Embedding,
    position_embedding: Embedding,
    word_embedding: Embedding
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

        let embeddings = embeddings.add(position_embeddings)?;

        Ok(embeddings)
    }

}