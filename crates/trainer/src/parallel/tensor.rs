use std::{
    collections::BTreeMap,
    env::temp_dir,
    fs, io, process,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use bullet_compiler::{
    ir::NodeId,
    model::split::{WeightRole, split_model},
    tensor::{DType, TValue, transform::partition::split_at_boundary},
};
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    function::Function,
    runtime::{Device, Gpu, Stream},
};

use crate::{
    model::{ModelDefinition, ModelWeights, TensorMap},
    optimiser::{AdditionalUpdate, OptimiserState, OptimiserUpdateSync},
    run::{PreparedBatchHost, TrainingError},
};

use super::{
    StepOutput,
    comm::Comm,
    data::{Bound, Replica, Upload, check_devices, compile, swap_batches, update_weights},
    drain, load_state, release_all,
    split::{gather_rows, merge_state_dirs, scatter_rows, split_state_dir},
    sync_all, write_state,
};

const NOT_PREPARED: &str = "Tensor-parallel training not prepared!";
const CHECK_BATCH: usize = 4096;

pub struct TensorParallel<G: Gpu, S: OptimiserState<G>> {
    replicas: Vec<Replica<G, S>>,
    locals: Vec<Local>,
    roles: BTreeMap<String, WeightRole>,
    split_comm: Comm<G>,
    head_comm: Comm<G>,
    mirror: OnceLock<TensorMap<G>>,
    stale: AtomicBool,
    run: Option<RunState<G>>,
    check: Option<Check<G>>,
    checked: bool,
}

struct Local {
    definition: ModelDefinition,
    partial_loss: Option<NodeId>,
}

struct RunState<G: Gpu> {
    ranks: Vec<RankRun<G>>,
    comm: Comm<G>,
    names: Option<Vec<String>>,
    sqnorms: Vec<Arc<Buffer<G>>>,
}

struct RankRun<G: Gpu> {
    before: Function<G>,
    after: Function<G>,
    bound: Bound<G>,
    partial: Arc<Buffer<G>>,
    partial_loss: Option<Arc<Buffer<G>>>,
    tgf: Arc<Buffer<G>>,
    tlr: Arc<Buffer<G>>,
}

struct Check<G: Gpu> {
    run: RunState<G>,
    reference: Function<G>,
    bound: Bound<G>,
    weights: TensorMap<G>,
}

impl<G: Gpu, S: OptimiserState<G>> TensorParallel<G, S> {
    pub fn new(
        definition: &ModelDefinition,
        weights: &ModelWeights,
        devices: &[Arc<Device<G>>],
        params: S::Params,
    ) -> Result<Self, G::Error> {
        if devices.len() != 2 {
            return Err("Tensor parallelism needs exactly 2 devices!".to_string().into());
        }

        check_devices(devices)?;

        let loss = definition.loss().ok_or_else(|| "Loss node not defined!".to_string())?;
        let split = split_model(definition.ir(), loss, devices.len())?;
        let roles = split.roles;

        let mut replicas = Vec::new();
        let mut locals = Vec::new();

        for (rank, (device, local)) in devices.iter().zip(split.ranks).enumerate() {
            let shards = weights.iter().map(|(name, value)| {
                let value = shard(&roles[name], &value.values, devices.len(), rank);
                Buffer::from_host(device, &value).map(|buffer| (name.clone(), buffer))
            });

            replicas.push(Replica::new(device, shards.collect::<Result<_, _>>()?, params.clone())?);

            let mut local_definition = ModelDefinition::new(local.ir, Some(local.loss), []);
            local_definition.debug = definition.debug;
            locals.push(Local { definition: local_definition, partial_loss: local.partial_loss });
        }

        let sizes = |role: WeightRole| {
            let names = roles.iter().filter(|(_, other)| **other == role);
            names.map(|(name, _)| (DType::F32, weights.get(name).values.size())).collect::<Vec<_>>()
        };

        let split_comm = Comm::new(devices, &sizes(WeightRole::SplitK))?;
        let head_comm = Comm::new(devices, &sizes(WeightRole::Head))?;

        Ok(Self {
            replicas,
            locals,
            roles,
            split_comm,
            head_comm,
            mirror: OnceLock::new(),
            stale: AtomicBool::new(true),
            run: None,
            check: None,
            checked: false,
        })
    }

    pub fn device(&self) -> Arc<Device<G>> {
        self.replicas[0].device.clone()
    }

    pub fn ranks(&self) -> usize {
        self.replicas.len()
    }

    pub fn state(&self) -> &BTreeMap<String, S> {
        &self.replicas[0].state
    }

    pub fn states_mut(&mut self) -> impl Iterator<Item = &mut BTreeMap<String, S>> {
        self.replicas.iter_mut().map(|replica| &mut replica.state)
    }

    pub fn weights(&self) -> &TensorMap<G> {
        if self.stale.swap(false, Ordering::Relaxed) {
            self.refresh().unwrap_or_else(|e| panic!("Tensor parallelism could not gather weights: {e:?}"));
        }

        self.mirror.get().unwrap()
    }

    pub fn read_weights(&self, host: &mut ModelWeights) -> Result<(), G::Error> {
        for (name, value) in self.gather()? {
            host.set(&name, value);
        }

        Ok(())
    }

    pub fn write_weights(&self, host: &ModelWeights) -> Result<(), G::Error> {
        self.stale.store(true, Ordering::Relaxed);

        for (rank, replica) in self.replicas.iter().enumerate() {
            for (name, value) in host.iter() {
                let value = shard(&self.roles[name], &value.values, self.replicas.len(), rank);
                replica.weights[name].copy_from_host(&value)?;
            }
        }

        Ok(())
    }

    pub fn write_state(&self, path: &str) -> Result<(), G::Error> {
        let dirs = state_dirs(self.replicas.len()).map_err(|e| e.to_string())?;

        let written = self.replicas.iter().zip(&dirs).try_for_each(|(replica, dir)| write_state(&replica.state, dir));
        let merged = written
            .and_then(|_| merge_state_dirs(&self.roles, &self.sizes(), &dirs, path).map_err(|e| e.to_string().into()));

        remove_dirs(&dirs);
        merged
    }

    pub fn load_state(&mut self, path: &str) -> Result<(), G::Error> {
        let dirs = state_dirs(self.replicas.len()).map_err(|e| e.to_string())?;

        let loaded = split_state_dir(&self.roles, &self.sizes(), path, &dirs).map_err(|e| e.to_string().into());
        let loaded = loaded.and_then(|_| {
            let mut replicas = self.replicas.iter_mut().zip(&dirs);
            replicas.try_for_each(|(replica, dir)| load_state(&mut replica.state, dir))
        });

        remove_dirs(&dirs);
        loaded
    }

    pub fn prepare(
        &mut self,
        defn: &ModelDefinition,
        batch_size: usize,
        grad_norms: bool,
    ) -> Result<(), TrainingError<G>> {
        self.run = None;
        self.check = None;

        for (name, role) in &self.roles {
            if matches!(role, WeightRole::Sharded { .. }) && !self.replicas[0].state[name].is_elementwise() {
                let msg = format!("Tensor parallelism cannot split '{name}': its optimiser is not elementwise!");
                return Err(TrainingError::OptimiserUpdateError(msg.into()));
            }
        }

        if !self.checked {
            let check = Check::new(defn, &self.replicas, &self.locals, batch_size.min(CHECK_BATCH), batch_size)?;
            self.check = Some(check);
        }

        self.run = Some(RunState::new(&self.replicas, &self.locals, batch_size, batch_size, grad_norms)?);

        Ok(())
    }

    pub fn load_batch(&mut self, batch: &PreparedBatchHost<G>) -> Result<(), G::Error> {
        if let Some(check) = &self.check {
            self.self_check(check, batch)?;
            self.check = None;
            self.checked = true;
        }

        let run = self.run.as_ref().ok_or_else(|| NOT_PREPARED.to_string())?;
        run.upload(&self.streams(|replica| &replica.copy), batch)?.sync()
    }

    pub fn step(
        &mut self,
        lr: f32,
        next: &mut dyn FnMut() -> Option<PreparedBatchHost<G>>,
        pre_update: &mut [Box<dyn AdditionalUpdate<G>>],
        post_update: &mut [Box<dyn AdditionalUpdate<G>>],
    ) -> Result<StepOutput, TrainingError<G>> {
        if !pre_update.is_empty() || !post_update.is_empty() {
            let msg = "Tensor parallelism does not support additional updates!";
            return Err(TrainingError::OptimiserUpdateError(msg.to_string().into()));
        }

        let (compute, copy) = (self.streams(|replica| &replica.compute), self.streams(|replica| &replica.copy));
        let Self { replicas, roles, split_comm, head_comm, run, stale, .. } = self;
        let run = run.as_mut().ok_or_else(|| TrainingError::Unexpected(NOT_PREPARED.to_string().into()))?;

        let lr = TValue::F32(vec![lr]);
        let mut lrs = Vec::new();
        for (rank, compute) in run.ranks.iter().zip(&compute) {
            lrs.push(rank.tlr.copy_from_host_async(compute, &lr).map_err(TrainingError::Unexpected)?);
        }

        let before = run.before(&compute).map_err(TrainingError::GradientCalculationError)?;
        unsafe { release_all(before) };

        stale.store(true, Ordering::Relaxed);
        let updates = drain(exchange_and_update(run, replicas, roles, split_comm, head_comm, &compute), &compute)?;
        run.comm.wait_for(&copy).map_err(TrainingError::Unexpected)?;

        let host = next();
        let uploads = match &host {
            Some(host) => run.upload_next(&copy, host).map_err(TrainingError::Unexpected)?,
            None => Upload::default(),
        };

        updates.sync().map_err(TrainingError::Unexpected)?;
        sync_all(lrs).and(uploads.sync()).map_err(TrainingError::Unexpected)?;

        if host.is_some() {
            swap_batches(run.ranks.iter_mut().map(|rank| &mut rank.bound));
        }

        let loss = run.loss(&copy).map_err(TrainingError::Unexpected)?;
        let grad_sqnorms = run.sqnorms(roles, &copy).map_err(TrainingError::Unexpected)?;

        Ok(StepOutput { loss, grad_sqnorms, next_queued: host.is_some() })
    }

    pub fn finish(&mut self) {
        self.run = None;
        self.check = None;
    }

    fn self_check(&self, check: &Check<G>, batch: &PreparedBatchHost<G>) -> Result<(), G::Error> {
        let (compute, copy) = (self.streams(|replica| &replica.compute), self.streams(|replica| &replica.copy));

        check.run.upload(&copy, batch)?.sync()?;
        sync_all(check.run.before(&compute)?)?;
        check.run.comm.all_reduce(&compute, &check.run.partials())?;
        sync_all(check.run.after(&compute)?)?;
        self.split_comm.all_reduce(&compute, &check.run.grads(&self.roles, WeightRole::SplitK))?;

        for (name, value) in self.gather()? {
            check.weights[&name].copy_from_host(&value)?;
        }

        let reference = &check.bound;
        reference.upload(&copy[0], batch, 0)?.sync()?;
        check.reference.execute(compute[0].clone(), &reference.tensors)?.value()?;

        let loss = reference.loss.to_host()?.f32()[0];
        check_close("loss", &[check.run.loss(&copy)?], &[loss])?;

        for (name, role) in &self.roles {
            let grads = check.run.ranks.iter().map(|rank| rank.bound.grads[name].to_host().map(|x| x.f32().to_vec()));
            let grads = grads.collect::<Result<Vec<_>, _>>()?;

            let ours = match role {
                WeightRole::Sharded { rows, segments } => gather_rows(&grads, *rows, segments),
                _ => grads[0].clone(),
            };

            check_close(name, &ours, reference.grads[name].to_host()?.f32())?;
        }

        Ok(())
    }

    fn refresh(&self) -> Result<(), G::Error> {
        let full = self.gather()?;

        match self.mirror.get() {
            Some(mirror) => full.iter().try_for_each(|(name, value)| mirror[name].copy_from_host(value)),
            None => {
                let device = &self.replicas[0].device;
                let buffers = full.iter().map(|(name, value)| Ok((name.clone(), Buffer::from_host(device, value)?)));
                let buffers = buffers.collect::<Result<_, G::Error>>()?;
                self.mirror.get_or_init(|| buffers);
                Ok(())
            }
        }
    }

    fn gather(&self) -> Result<BTreeMap<String, TValue>, G::Error> {
        let mut full = BTreeMap::new();

        for (name, role) in &self.roles {
            let value = match role {
                WeightRole::Sharded { rows, segments } => {
                    let shards =
                        self.replicas.iter().map(|replica| replica.weights[name].to_host().map(|x| x.f32().to_vec()));
                    TValue::F32(gather_rows(&shards.collect::<Result<Vec<_>, _>>()?, *rows, segments))
                }
                _ => self.replicas[0].weights[name].to_host()?,
            };

            full.insert(name.clone(), value);
        }

        Ok(full)
    }

    fn sizes(&self) -> BTreeMap<String, usize> {
        self.replicas[0].weights.iter().map(|(name, buffer)| (name.clone(), buffer.size())).collect()
    }

    fn streams(&self, stream: fn(&Replica<G, S>) -> &Arc<Stream<G>>) -> Vec<Arc<Stream<G>>> {
        self.replicas.iter().map(|replica| stream(replica).clone()).collect()
    }
}

impl<G: Gpu> RunState<G> {
    fn new<S>(
        replicas: &[Replica<G, S>],
        locals: &[Local],
        batch_size: usize,
        host_batch: usize,
        grad_norms: bool,
    ) -> Result<Self, TrainingError<G>> {
        let mut runs = Vec::new();
        let (mut names, mut sqnorms) = (None, Vec::new());

        for (replica, local) in replicas.iter().zip(locals) {
            let device = &replica.device;
            let extra = local.partial_loss.into_iter().collect::<Vec<_>>();
            let lowering =
                local.definition.lower_backward_with(&Default::default(), batch_size, grad_norms, false, &extra);
            let (lowered, gmap, grad_norms) = lowering.map_err(TrainingError::CompilingBackwards)?;
            let split = split_at_boundary(lowered.ir()).map_err(TrainingError::CompilingBackwards)?;

            let (before, after) = (compile(device, split.before)?, compile(device, split.after)?);

            let bound =
                Bound::new(&local.definition, &lowered, &gmap, &replica.weights, device, batch_size, host_batch);
            let mut bound = bound.map_err(TrainingError::Unexpected)?;
            let zeroed = |dtype, size| Buffer::zeroed(device, dtype, size).map_err(TrainingError::Unexpected);
            let scalar =
                |value| Buffer::from_host(device, &TValue::F32(vec![value])).map_err(TrainingError::Unexpected);

            for id in split.cross.iter().chain([&split.partial]) {
                let ty = lowered.ir().get_node(*id).map_err(TrainingError::CompilingBackwards)?.ty();
                bound.tensors.insert(*id, zeroed(ty.dtype(), ty.size().get())?);
            }

            let partial = bound.tensors[&split.partial].clone();
            bound.tensors.insert(split.reduced, partial.clone());

            let partial_loss = match local.partial_loss {
                Some(id) => {
                    let buffer = scalar(0.0)?;
                    bound.tensors.insert(lowered.map()[&id], buffer.clone());
                    Some(buffer)
                }
                None => None,
            };

            if let Some(grad_norms) = grad_norms {
                let buffer = zeroed(DType::F32, grad_norms.names.len())?;
                bound.tensors.insert(grad_norms.acc, buffer.clone());
                sqnorms.push(buffer);
                names = Some(grad_norms.names);
            }

            let (tgf, tlr) = (scalar(1.0 / host_batch as f32)?, scalar(0.0)?);
            runs.push(RankRun { before, after, bound, partial, partial_loss, tgf, tlr });
        }

        let devices = replicas.iter().map(|replica| replica.device.clone()).collect::<Vec<_>>();
        let comm = Comm::new(&devices, &[(DType::F32, runs[0].partial.size())]).map_err(TrainingError::Unexpected)?;

        Ok(Self { ranks: runs, comm, names, sqnorms })
    }

    fn upload<'a>(
        &'a self,
        copy: &[Arc<Stream<G>>],
        host: &'a PreparedBatchHost<G>,
    ) -> Result<Upload<'a, G>, G::Error> {
        let mut uploads = Upload::default();
        for (rank, stream) in self.ranks.iter().zip(copy) {
            uploads.extend(rank.bound.upload(stream, host, 0)?);
        }

        Ok(uploads)
    }

    fn upload_next<'a>(
        &'a self,
        copy: &[Arc<Stream<G>>],
        host: &'a PreparedBatchHost<G>,
    ) -> Result<Upload<'a, G>, G::Error> {
        let mut uploads = Upload::default();
        for (rank, stream) in self.ranks.iter().zip(copy) {
            uploads.extend(rank.bound.upload_next(stream, host, 0)?);
        }

        Ok(uploads)
    }

    fn before(&self, compute: &[Arc<Stream<G>>]) -> Result<Vec<SyncOnValue<G, &Function<G>>>, G::Error> {
        self.ranks
            .iter()
            .zip(compute)
            .map(|(rank, stream)| rank.before.execute(stream.clone(), &rank.bound.tensors))
            .collect()
    }

    fn after(&self, compute: &[Arc<Stream<G>>]) -> Result<Vec<SyncOnValue<G, &Function<G>>>, G::Error> {
        self.ranks
            .iter()
            .zip(compute)
            .map(|(rank, stream)| rank.after.execute(stream.clone(), &rank.bound.tensors))
            .collect()
    }

    fn partials(&self) -> Vec<Vec<Arc<Buffer<G>>>> {
        self.ranks.iter().map(|rank| vec![rank.partial.clone()]).collect()
    }

    fn grads(&self, roles: &BTreeMap<String, WeightRole>, role: WeightRole) -> Vec<Vec<Arc<Buffer<G>>>> {
        let names = roles.iter().filter(|(_, other)| **other == role).map(|(name, _)| name).collect::<Vec<_>>();
        self.ranks.iter().map(|rank| names.iter().map(|&name| rank.bound.grads[name].clone()).collect()).collect()
    }

    fn loss(&self, copy: &[Arc<Stream<G>>]) -> Result<f32, G::Error> {
        let mut loss = 0.0;

        for (i, (rank, stream)) in self.ranks.iter().zip(copy).enumerate() {
            let buffer = if i == 0 { Some(&rank.bound.loss) } else { rank.partial_loss.as_ref() };

            if let Some(buffer) = buffer {
                loss += buffer.to_host_async(stream)?.value()?.f32()[0];
            }
        }

        Ok(loss)
    }

    fn sqnorms(
        &self,
        roles: &BTreeMap<String, WeightRole>,
        copy: &[Arc<Stream<G>>],
    ) -> Result<Option<Vec<(String, f32)>>, G::Error> {
        let Some(names) = &self.names else { return Ok(None) };

        let values = self.sqnorms.iter().zip(copy).map(|(buffer, stream)| buffer.to_host_async(stream)?.value());
        let values = values.collect::<Result<Vec<_>, _>>()?;

        let sqnorm = |i: usize, name: &String| match roles[name] {
            WeightRole::Head => values[0].f32()[i],
            _ => values.iter().map(|value| value.f32()[i]).sum(),
        };

        Ok(Some(names.iter().enumerate().map(|(i, name)| (name.clone(), sqnorm(i, name))).collect()))
    }
}

impl<G: Gpu> Check<G> {
    fn new<S>(
        defn: &ModelDefinition,
        replicas: &[Replica<G, S>],
        locals: &[Local],
        batch_size: usize,
        host_batch: usize,
    ) -> Result<Self, TrainingError<G>> {
        let run = RunState::new(replicas, locals, batch_size, host_batch, false)?;
        let device = &replicas[0].device;

        let lowering = defn.lower_backward(&Default::default(), batch_size, false, false);
        let (lowered, gmap, _) = lowering.map_err(TrainingError::CompilingBackwards)?;
        let reference = compile(device, lowered.ir().clone())?;

        let weights = ModelWeights::zeroed(defn).to_device(device).map_err(TrainingError::Unexpected)?;
        let bound = Bound::new(defn, &lowered, &gmap, &weights, device, batch_size, host_batch);

        Ok(Self { run, reference, bound: bound.map_err(TrainingError::Unexpected)?, weights })
    }
}

fn exchange_and_update<'a, G: Gpu, S: OptimiserState<G>>(
    run: &RunState<G>,
    replicas: &'a mut [Replica<G, S>],
    roles: &BTreeMap<String, WeightRole>,
    split_comm: &Comm<G>,
    head_comm: &Comm<G>,
    compute: &[Arc<Stream<G>>],
) -> Result<OptimiserUpdateSync<'a, G>, TrainingError<G>> {
    run.comm.all_reduce(compute, &run.partials()).map_err(TrainingError::Unexpected)?;

    let after = run.after(compute).map_err(TrainingError::GradientCalculationError)?;
    unsafe { release_all(after) };

    split_comm.all_reduce(compute, &run.grads(roles, WeightRole::SplitK)).map_err(TrainingError::Unexpected)?;
    head_comm.broadcast(0, compute, &run.grads(roles, WeightRole::Head)).map_err(TrainingError::Unexpected)?;

    let mut updates = OptimiserUpdateSync::default();
    for ((replica, rank), stream) in replicas.iter_mut().zip(&run.ranks).zip(compute) {
        let (tgf, tlr) = (rank.tgf.clone(), rank.tlr.clone());
        let update = update_weights(&mut replica.state, &replica.weights, stream, tgf, tlr, &rank.bound.grads);
        updates.extend_by(update.map_err(TrainingError::OptimiserUpdateError)?);
    }

    Ok(updates)
}

fn shard(role: &WeightRole, full: &TValue, ranks: usize, rank: usize) -> TValue {
    match role {
        WeightRole::Sharded { rows, segments } => TValue::F32(scatter_rows(full.f32(), *rows, segments, ranks, rank)),
        _ => full.clone(),
    }
}

fn state_dirs(ranks: usize) -> io::Result<Vec<String>> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let dirs = (0..ranks).map(|rank| format!("{}/bullet-tp-{}-{id}-{rank}", temp_dir().display(), process::id()));
    let dirs = dirs.collect::<Vec<_>>();

    for dir in &dirs {
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir)?;
    }

    Ok(dirs)
}

fn remove_dirs(dirs: &[String]) {
    for dir in dirs {
        let _ = fs::remove_dir_all(dir);
    }
}

fn check_close(name: &str, ours: &[f32], reference: &[f32]) -> Result<(), String> {
    let diff = ours.iter().zip(reference).map(|(x, y)| f64::from(x - y).powi(2)).sum::<f64>().sqrt();
    let norm = reference.iter().map(|y| f64::from(*y).powi(2)).sum::<f64>().sqrt();
    let err = if diff == 0.0 { 0.0 } else { diff / norm };

    if err <= 1e-4 {
        Ok(())
    } else {
        Err(format!("Tensor-parallel self-check failed for '{name}': relative error {err:.2e}!"))
    }
}
