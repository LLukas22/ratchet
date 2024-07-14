mod attn;
mod embedding;
mod encoder;
mod mlp;
mod model;
mod pooler;

use model::BertInput;
pub use model::BERT;

use pooler::PoolerInput;
use ratchet::{prelude::shape, Device, Tensor};
use ratchet_nn::Module;
use tokenizers::Tokenizer;

fn tokenize_batched(
    tokenizer: Tokenizer,
    inputs: Vec<&str>,
    device: Device,
) -> anyhow::Result<(Tensor, Vec<Vec<i32>>)> {
    let batch_encoding = tokenizer.encode_batch(inputs, true).unwrap();
    // TODO: limit batch_size and validate padding
    let mut tokens = Vec::new();
    let mut attention_mask = Vec::new();

    for encoding in batch_encoding {
        tokens.push(
            encoding
                .get_ids()
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>(),
        );

        attention_mask.push(
            encoding
                .get_attention_mask()
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>(),
        );
    }
    let batchsize = tokens.len();
    let sequence_length = tokens[0].len();

    let flat_ids: Vec<i32> = tokens.into_iter().flatten().collect();
    let input_ids = Tensor::from_data(flat_ids, shape![batchsize, sequence_length], device);
    Ok((input_ids, attention_mask))
}

pub fn embed(
    model: &BERT,
    tokenizer: Tokenizer,
    inputs: Vec<&str>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let (input_ids, attention_mask) = tokenize_batched(tokenizer, inputs, model.device.clone())?;

    let input = BertInput {
        input_ids: input_ids,
        token_type_ids: None,
        attention_mask: Some(attention_mask.clone()),
    };
    let hidden_state = model.schedule(input)?;
    let pooler_input = PoolerInput {
        last_hidden_state: hidden_state,
        attention_mask: Some(attention_mask),
    };
    let pooled = model.pooler.forward(pooler_input)?;

    let mut embeddings = Vec::with_capacity(pooled.nrows());
    for row in pooled.rows() {
        let vec: Vec<f32> = row.to_vec();
        embeddings.push(vec);
    }
    Ok(embeddings)
}
