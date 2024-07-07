use super::{
    embedding::{BertEmbedding, EmbeddingInput},
    encoder::EncoderLayer,
};

use ratchet::{Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::Module;
use std::io::{BufRead, Seek};

pub struct BERT {
    pub embedding: BertEmbedding,
    pub layers: Vec<EncoderLayer>,
    pub device: Device,
}

impl Module for BERT {
    type Input = EmbeddingInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut x = self.embedding.schedule(input)?;
        for layer in self.layers.iter() {
            x = layer.schedule(x)?;
        }
        Ok(x)
    }
}

impl BERT {
    pub fn load<R: BufRead + Seek>(
        header: Header,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let embedding = BertEmbedding::load(&header, reader, device)?;

        let n_layers = header.metadata.get("bert.block_count").unwrap().to_u32()? as i32;

        let layers = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(EncoderLayer::load(&header, reader, i as _, device));
                blocks
            })
            .into_iter()
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(BERT {
            embedding: embedding,
            layers: layers,
            device: device.clone(),
        })
    }
}

#[cfg(test)]
mod bert_tests {
    use hf_hub::api::sync::Api;
    use ndarray::Axis;
    use ndarray_stats::QuantileExt;
    use numpy::PyArrayDyn;
    use pyo3::{types::PyModule, Python};
    use ratchet::{prelude::shape, Device, DeviceRequest, Tensor};
    use ratchet_loader::gguf;
    use ratchet_nn::Module;
    use tokenizers::Tokenizer;

    use crate::bert::{embedding::EmbeddingInput, BERT};

    #[test]
    fn load_bert() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let api = Api::new().unwrap();
        let model_repo = api.model("LLukas22/all-MiniLM-L6-v2-GGUF".to_string());
        let model_path = model_repo.get("all-minilm-l6-v2-f32.gguf").unwrap();
        println!("MODEL PATH: {}", model_path.display());

        let tokenizer_repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        let prompt = "Why did the crab cross the road?";
        println!("Prompt: '{}'", prompt);
        let encoding = tokenizer.encode(prompt, true).unwrap();
        let tokens = encoding
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();

        let token_types = vec![0_i32; tokens.len()];

        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = gguf::gguf::Header::read(&mut reader)?;
        let model = BERT::load(content, &mut reader, &device)?;

        let input_ids = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
        let token_type_ids = Tensor::from_data(token_types, shape![1, tokens.len()], device);

        let input = EmbeddingInput {
            input_ids,
            token_type_ids,
        };

        let result = model.schedule(input)?.resolve()?;
        let result = result.to(&Device::CPU)?;
        println!("{:?}", result);
        Ok(())
    }
}
