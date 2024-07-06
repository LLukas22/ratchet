use ratchet::Tensor;
use ratchet_nn::{Linear, Module};

#[derive(Debug, derive_new::new)]
pub struct MLP {
    up: Linear,
    down: Linear,
}

impl Module for MLP {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<ratchet::Tensor> {
        self.down.schedule(self.up.schedule(input)?.gelu()?)
    }
}
