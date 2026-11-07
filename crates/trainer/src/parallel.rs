mod comm;
mod data;
mod plan;
mod split;
mod tensor;

pub use data::{DataParallel, StepOutput};
pub use tensor::TensorParallel;

use std::{collections::BTreeMap, sync::Arc};

use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    runtime::{Device, Gpu, Stream},
};

use crate::{
    model::{ModelDefinition, ModelWeights, TensorMap},
    optimiser::{AdditionalUpdate, OptimiserState, OptimiserUpdateResult},
    run::{PreparedBatchHost, TrainingError},
};

pub enum Executor<G: Gpu, S: OptimiserState<G>> {
    Data(Box<DataParallel<G, S>>),
    Tensor(Box<TensorParallel<G, S>>),
}

impl<G: Gpu, S: OptimiserState<G>> Executor<G, S> {
    pub fn device(&self) -> Arc<Device<G>> {
        match self {
            Self::Data(data) => data.device(),
            Self::Tensor(tensor) => tensor.device(),
        }
    }

    pub fn ranks(&self) -> usize {
        match self {
            Self::Data(data) => data.ranks(),
            Self::Tensor(tensor) => tensor.ranks(),
        }
    }

    pub fn weights(&self) -> &TensorMap<G> {
        match self {
            Self::Data(data) => data.weights(),
            Self::Tensor(tensor) => tensor.weights(),
        }
    }

    pub fn state(&self) -> &BTreeMap<String, S> {
        match self {
            Self::Data(data) => data.state(),
            Self::Tensor(tensor) => tensor.state(),
        }
    }

    pub fn states_mut(&mut self) -> Vec<&mut BTreeMap<String, S>> {
        match self {
            Self::Data(data) => data.states_mut().collect(),
            Self::Tensor(tensor) => tensor.states_mut().collect(),
        }
    }

    pub fn read_weights(&self, host: &mut ModelWeights) -> Result<(), G::Error> {
        match self {
            Self::Data(data) => data.read_weights(host),
            Self::Tensor(tensor) => tensor.read_weights(host),
        }
    }

    pub fn write_weights(&self, host: &ModelWeights) -> Result<(), G::Error> {
        match self {
            Self::Data(data) => data.write_weights(host),
            Self::Tensor(tensor) => tensor.write_weights(host),
        }
    }

    pub fn write_state(&self, path: &str) -> Result<(), G::Error> {
        match self {
            Self::Data(data) => data.write_state(path),
            Self::Tensor(tensor) => tensor.write_state(path),
        }
    }

    pub fn load_state(&mut self, path: &str) -> Result<(), G::Error> {
        match self {
            Self::Data(data) => data.load_state(path),
            Self::Tensor(tensor) => tensor.load_state(path),
        }
    }

    pub fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
        gradients: &TensorMap<G>,
        pre_update: &'a mut [Box<dyn AdditionalUpdate<G>>],
        post_update: &'a mut [Box<dyn AdditionalUpdate<G>>],
    ) -> OptimiserUpdateResult<'a, G> {
        match self {
            Self::Data(data) => data.update(stream, gradient_factor, learning_rate, gradients, pre_update, post_update),
            Self::Tensor(_) => unreachable!("tensor parallelism always spans several devices"),
        }
    }

    pub fn prepare(
        &mut self,
        defn: &ModelDefinition,
        batch_size: usize,
        grad_norms: bool,
    ) -> Result<(), TrainingError<G>> {
        match self {
            Self::Data(data) => data.prepare(defn, batch_size, grad_norms),
            Self::Tensor(tensor) => tensor.prepare(defn, batch_size, grad_norms),
        }
    }

    pub fn load_batch(&mut self, batch: &PreparedBatchHost<G>) -> Result<(), G::Error> {
        match self {
            Self::Data(data) => data.load_batch(batch),
            Self::Tensor(tensor) => tensor.load_batch(batch),
        }
    }

    pub fn step(
        &mut self,
        lr: f32,
        next: &mut dyn FnMut() -> Option<PreparedBatchHost<G>>,
        pre_update: &mut [Box<dyn AdditionalUpdate<G>>],
        post_update: &mut [Box<dyn AdditionalUpdate<G>>],
    ) -> Result<StepOutput, TrainingError<G>> {
        match self {
            Self::Data(data) => data.step(lr, next, pre_update, post_update),
            Self::Tensor(tensor) => tensor.step(lr, next, pre_update, post_update),
        }
    }

    pub fn finish(&mut self) {
        match self {
            Self::Data(data) => data.finish(),
            Self::Tensor(tensor) => tensor.finish(),
        }
    }
}

fn sync_all<G: Gpu, T>(handles: Vec<SyncOnValue<G, T>>) -> Result<(), G::Error> {
    let mut result = Ok(());

    for handle in handles {
        result = result.and(handle.value().map(drop));
    }

    result
}

unsafe fn release_all<G: Gpu, T>(handles: Vec<SyncOnValue<G, T>>) {
    for handle in handles {
        unsafe { handle.release() }
    }
}

fn drain<G: Gpu, T, E>(result: Result<T, E>, streams: &[Arc<Stream<G>>]) -> Result<T, E> {
    if result.is_err() {
        for stream in streams {
            let _ = stream.sync();
        }
    }

    result
}

fn write_state<G: Gpu, S: OptimiserState<G>>(state: &BTreeMap<String, S>, path: &str) -> Result<(), G::Error> {
    let map = state.iter().map(|(id, single)| (id.clone(), single)).collect();
    S::write_to_checkpoint(&map, path)
}

fn load_state<G: Gpu, S: OptimiserState<G>>(state: &mut BTreeMap<String, S>, path: &str) -> Result<(), G::Error> {
    let mut map = state.iter_mut().map(|(id, single)| (id.clone(), single)).collect();
    S::load_from_checkpoint(&mut map, path)
}
