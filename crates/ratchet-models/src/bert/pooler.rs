use anyhow::{anyhow, Error};
use ndarray::{Array2, Array3, Axis};
#[repr(u32)]
/// GGUF pooling types:
/// - None: No pooling.
/// - Mean: Mean pooling.
/// - CLS: CLS token pooling.
/// - Last: Last value pooling.
pub enum PoolingType {
    None = 0,
    Mean = 1,
    CLS = 2,
    Last = 3,
}

impl TryFrom<u32> for PoolingType {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(PoolingType::None),
            1 => Ok(PoolingType::Mean),
            2 => Ok(PoolingType::CLS),
            3 => Ok(PoolingType::Last),
            _ => Err(anyhow!("Invalid value for PoolingType: {}", value)),
        }
    }
}

pub struct PoolerInput {
    pub last_hidden_state: Array3<f32>,
    pub attention_mask: Option<Vec<Vec<i32>>>, // Mask in the form [batch_size, 1, 1, sequence_length]
}
pub trait Pooler {
    fn forward(&self, input: PoolerInput) -> anyhow::Result<Array2<f32>>;
}

pub struct MeanPooling;

impl Pooler for MeanPooling {
    /// Implements mean pooling.
    /// ```python
    /// def mean_pooling(model_output, attention_mask):
    ///     token_embeddings = model_output[0]
    ///     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    ///     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    /// ```
    fn forward(&self, input: PoolerInput) -> anyhow::Result<Array2<f32>> {
        let PoolerInput {
            last_hidden_state,
            attention_mask,
        } = input;

        if let Some(attention_mask) = attention_mask {
            let [batch_size, seq_len, embedding_dim]: [usize; 3] =
                last_hidden_state.shape().try_into()?;

            // Build a [batch_size, seq_len, embedding_dim] mask.
            let blackout_mask = vec![0_f32; embedding_dim];
            let keep_mask = vec![1_f32; embedding_dim];

            let flat_mask: Vec<f32> = attention_mask
                .into_iter()
                .flatten()
                .map(|x| {
                    if x == 0 {
                        blackout_mask.clone()
                    } else {
                        keep_mask.clone()
                    }
                })
                .flatten()
                .collect();
            let mask = Array3::from_shape_vec((batch_size, seq_len, embedding_dim), flat_mask)?;

            let weighted_embeddings = (last_hidden_state * &mask).sum_axis(Axis(1))
                / mask
                    .sum_axis(Axis(1))
                    .mapv(|v| num::clamp(v, 1e-9, f32::INFINITY));

            return Ok(weighted_embeddings);
        }

        let weighted_embeddings = last_hidden_state.mean_axis(Axis(1)).unwrap();

        Ok(weighted_embeddings)
    }
}
