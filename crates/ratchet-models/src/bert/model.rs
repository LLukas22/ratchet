use super::{
    attn::BertAttentionInput,
    embedding::{BertEmbedding, EmbeddingInput},
    encoder::EncoderLayer,
    pooler::MeanPooling,
};

use ratchet::{prelude::shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::Module;
use std::io::{BufRead, Seek};

pub struct BertInput {
    pub input_ids: Tensor,
    pub token_type_ids: Option<Tensor>,
    pub attention_mask: Option<Vec<Vec<i32>>>,
}
pub struct BERT {
    pub embedding: BertEmbedding,
    pub layers: Vec<EncoderLayer>,
    pub device: Device,
}

impl Module for BERT {
    type Input = BertInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let BertInput {
            input_ids,
            token_type_ids,
            attention_mask,
        } = input;

        let mask = self.create_mask(&input_ids, attention_mask)?;

        let mut x = self.embedding.schedule(EmbeddingInput {
            input_ids,
            token_type_ids,
        })?;

        for layer in self.layers.iter() {
            x = layer.schedule(BertAttentionInput {
                x,
                mask: mask.clone(),
            })?;
        }
        Ok(x)
    }
}

impl BERT {
    fn create_mask(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<Vec<Vec<i32>>>,
    ) -> anyhow::Result<Option<Tensor>> {
        let [batch_size, seq_len]: [usize; 2] = input_ids.shape().try_into()?;
        if let Some(mask) = attention_mask {
            let mut batched_attention_masks: Vec<Vec<f32>> = Vec::new();

            for batch_mask in &mask {
                let attention_mask: Vec<f32> = batch_mask
                    .iter()
                    .map(|&x| if x == 0 { f32::NEG_INFINITY } else { 0.0 })
                    .collect();
                batched_attention_masks.push(attention_mask);
            }
            let flat_mask: Vec<f32> = batched_attention_masks.into_iter().flatten().collect();

            let mut tensor_mask = Tensor::from_data(
                flat_mask,
                shape![batch_size, seq_len],
                input_ids.device().clone(),
            );
            tensor_mask = tensor_mask.view(shape![batch_size, 1, 1, seq_len])?;
            return Ok(Some(tensor_mask));
        }

        Ok(None)
    }

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

    #[cfg(target_arch = "wasm32")]
    pub async fn from_web(header: Header, mut tensors: TensorMap) -> anyhow::Result<Self> {
        let device = Device::request_device(ratchet::DeviceRequest::GPU).await?;
        let embedding = BertEmbedding::from_web(&header, reader, &device)?;

        let n_layers = header.metadata.get("bert.block_count").unwrap().to_u32()? as i32;

        let layers = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(EncoderLayer::from_web(&header, reader, i as _, &device));
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod bert_tests {
    use hf_hub::api::sync::Api;
    use numpy::PyArrayDyn;
    use pyo3::{types::PyModule, Python};
    use ratchet::{prelude::shape, Device, DeviceRequest, Tensor};
    use ratchet_loader::gguf;
    use ratchet_nn::Module;
    use tokenizers::{
        DecoderWrapper, ModelWrapper, NormalizerWrapper, PaddingParams, PostProcessorWrapper,
        PreTokenizerWrapper, Tokenizer, TokenizerImpl,
    };

    use crate::bert::{
        attn::BertAttentionInput, embedding::EmbeddingInput, model::BertInput, pooler::Pooler, BERT,
    };
    const PRG: &str = r#"
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertModel
import torch 

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load()->tuple[AutoTokenizer,BertModel]:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model:BertModel = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = model.eval()
    return tokenizer,model

def embedding():
    tokenizer, model = load()
    
    input_sentence = "Why did the crab cross the road?"
    inputs = tokenizer(input_sentence, return_tensors="pt")
    
    with torch.no_grad():
        embeddings = model.embeddings(input_ids=inputs.input_ids,token_type_ids=inputs.token_type_ids)
        
    return embeddings.numpy()


def encoder_outputs():
    tokenizer, model = load()
    
    input_sentence = "Why did the crab cross the road?"
    inputs = tokenizer(input_sentence, return_tensors="pt")
    
    outputs = []
    with torch.no_grad():
        embeddings = model.embeddings(input_ids=inputs.input_ids,token_type_ids=inputs.token_type_ids)
        hidden_state:torch.Tensor = embeddings
        for layer in model.encoder.layer:
            hidden_state:torch.Tensor = layer(hidden_state)[0]
            outputs.append(hidden_state.clone().numpy())
    
    return outputs


def pooled_output():
    tokenizer, model = load()
    
    input_sentence = "Why did the crab cross the road?"
    input_sentence2 = "Another unrelated prompt"
    inputs = tokenizer([input_sentence, input_sentence2], return_tensors="pt", padding=True)

    with torch.no_grad():
        results = model(**inputs)
        pooled = mean_pooling(results, inputs["attention_mask"])

    return pooled.numpy()
"#;

    fn gt_embeddings() -> anyhow::Result<Tensor> {
        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, PRG, "e.py", "e")?;
            let py_result: &PyArrayDyn<f32> = prg.getattr("embedding")?.call0()?.extract()?;
            Ok(Tensor::from(py_result))
        })
    }
    fn gt_layers() -> anyhow::Result<Vec<Tensor>> {
        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, PRG, "l.py", "l")?;
            let py_result: Vec<&PyArrayDyn<f32>> =
                prg.getattr("encoder_outputs")?.call0()?.extract()?;
            Ok(py_result.into_iter().map(Tensor::from).collect::<_>())
        })
    }

    fn gt_pooling() -> anyhow::Result<Tensor> {
        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, PRG, "p.py", "p")?;
            let py_result: &PyArrayDyn<f32> = prg.getattr("pooled_output")?.call0()?.extract()?;
            Ok(Tensor::from(py_result))
        })
    }

    type BertTokenizer = TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >;

    fn load() -> anyhow::Result<(BERT, BertTokenizer)> {
        let _ = env_logger::builder().is_test(true).try_init();
        let api = Api::new().unwrap();
        let model_repo = api.model("LLukas22/all-MiniLM-L6-v2-GGUF".to_string());
        let model_path = model_repo.get("all-minilm-l6-v2-f32.gguf").unwrap();
        println!("MODEL PATH: {}", model_path.display());

        let tokenizer_repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").unwrap();
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        //Force padding to test masking
        let tokenizer = tokenizer.with_padding(Some(PaddingParams {
            pad_to_multiple_of: Some(128),
            ..Default::default()
        }));

        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = gguf::gguf::Header::read(&mut reader)?;
        let model = BERT::load(content, &mut reader, &device)?;

        Ok((model, tokenizer.clone()))
    }

    fn tokenize(
        tokenizer: BertTokenizer,
        inputs: Vec<&str>,
        device: Device,
    ) -> anyhow::Result<(Tensor, Vec<Vec<i32>>)> {
        let batch_encoding = tokenizer.encode_batch(inputs, true).unwrap();
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

        let flat_ids: Vec<i32> = tokens.into_iter().flatten().collect();
        let input_ids = Tensor::from_data(flat_ids.clone(), shape![2, 128], device);
        Ok((input_ids, attention_mask))
    }

    #[test]
    fn embedding() -> anyhow::Result<()> {
        let (model, tokenizer) = load()?;

        let prompt = "Why did the crab cross the road?"; // [101, 2339, 2106, 1996, 18081, 2892, 1996, 2346, 1029, 102]
        let prompt2 = "Another unrelated prompt";

        let (input_ids, _) = tokenize(tokenizer, vec![prompt, prompt2], model.device.clone())?;
        println!("Prompt: '{}'", prompt);

        let input = EmbeddingInput {
            input_ids,
            token_type_ids: None,
        };

        //Extract embeddings for the first sequence
        let ratchet_embeddings = model
            .embedding
            .schedule(input.clone())?
            .slice(&[0..1, 0..10, 0..384])?
            .resolve()?
            .to(&Device::CPU)?;
        let pytorch_embeddings = gt_embeddings()?;

        pytorch_embeddings.all_close(&ratchet_embeddings, 1e-3, 1e-3)?;

        Ok(())
    }

    #[test]
    fn hidden_states() -> anyhow::Result<()> {
        let (model, tokenizer) = load()?;

        let prompt = "Why did the crab cross the road?"; // [101, 2339, 2106, 1996, 18081, 2892, 1996, 2346, 1029, 102]
        let prompt2 = "Another unrelated prompt";

        let (input_ids, attention_mask) =
            tokenize(tokenizer, vec![prompt, prompt2], model.device.clone())?;
        println!("Prompt: '{}'", prompt);

        let mask = model.create_mask(&input_ids, Some(attention_mask.clone()))?;

        if let Some(m) = mask.clone() {
            let cloned_mask = m.clone().resolve()?.to(&Device::CPU)?;
            println!("Mask: '{:?}'", cloned_mask);
        }

        let input = EmbeddingInput {
            input_ids,
            token_type_ids: None,
        };

        let pytorch_layers = gt_layers()?;

        let mut x = model.embedding.schedule(input.clone())?;
        for (i, layer) in model.layers.iter().enumerate() {
            x = layer.schedule(BertAttentionInput {
                x,
                mask: mask.clone(),
            })?;

            //Check if the output correctly matches the pytorch implementation without the mask.
            let result = x
                .clone()
                .slice(&[0..1, 0..10, 0..384])?
                .resolve()?
                .to(&Device::CPU)?;
            pytorch_layers[i].all_close(&result, 4e-3, 4e-3)?;
        }

        Ok(())
    }

    #[test]
    fn pooling() -> anyhow::Result<()> {
        let (model, tokenizer) = load()?;

        let prompt = "Why did the crab cross the road?"; // [101, 2339, 2106, 1996, 18081, 2892, 1996, 2346, 1029, 102]
        let prompt2 = "Another unrelated prompt";

        let (input_ids, attention_mask) =
            tokenize(tokenizer, vec![prompt, prompt2], model.device.clone())?;
        println!("Prompt: '{}'", prompt);

        let input = BertInput {
            input_ids,
            token_type_ids: None,
            attention_mask: Some(attention_mask.clone()),
        };

        let result = model.schedule(input)?.resolve()?;

        let pooler = crate::bert::pooler::MeanPooling;
        let pooler_input = crate::bert::pooler::PoolerInput {
            last_hidden_state: result
                .to(&Device::CPU)?
                .into_ndarray::<f32>()
                .into_dimensionality::<ndarray::Ix3>()?,
            attention_mask: Some(attention_mask),
        };
        let pooled = pooler.forward(pooler_input)?;

        let gt_pooled = gt_pooling()?;
        let gt_pooled = gt_pooled.to_ndarray_view::<f32>();
        assert!(pooled.shape() == gt_pooled.shape());

        Ok(())
    }
}
