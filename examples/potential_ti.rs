#[path = "advanced/filter.rs"]
mod filter;
#[path = "ti_inputs.rs"]
mod inputs;

use std::sync::Arc;
use bullet_lib::{
    game::{
        formats::{
            bulletformat::ChessBoard,
            montyformat::chess::{Piece, Side},
        },
        inputs::{ChessBucketsMirrored, SparseInputType},
        outputs::{MaterialCount, OutputBuckets},
    },
    trainer::schedule::{
        lr::{self, LrScheduler},
        wdl,
    },
    value::{
        loader::{ViriBinpackLoader, viribinpack::ViriFilter},
        save::save_to_checkpoint,
    },
    wdl::WdlScheduler,
};
use bullet_trainer::{
    model::{DenseInput, InitSettings, ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat, SparseInput},
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    reader::ReadMapLoader,
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, train},
};

const HIDDEN_SIZE: usize = 1024;
const NUM_OUTPUT_BUCKETS: usize = 8;
const QA: i16 = 255;
const QB: i16 = 64;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];
const NUM_INPUT_BUCKETS: usize = 4;

fn build_bbs(pos: &ChessBoard) -> [u64; 8] {
    let mut bbs = [0u64; 8];
    for (pc, sq) in pos.into_iter() {
        let bit = 1 << sq;
        bbs[usize::from(pc & 8 > 0)] |= bit;
        bbs[2 + usize::from(pc & 7)] |= bit;
    }
    bbs
}

#[derive(Clone)]
struct ThreatInputs {
    threats: Arc<inputs::Threats>,
}

impl ThreatInputs {
    fn new() -> Self {
        Self { threats: Arc::new(inputs::Threats::new()) }
    }
    fn num_inputs(&self) -> usize {
        self.threats.num_inputs()
    }
    fn max_active(&self) -> usize {
        self.threats.max_active()
    }
    fn map_features(&self, pos: &ChessBoard, on_stm: impl FnMut(usize), on_ntm: impl FnMut(usize)) {
        let bbs = build_bbs(pos);
        self.threats.map(bbs, on_stm, on_ntm);
    }
}

pub type InputTy = ((((SparseInput, SparseInput), SparseInput), SparseInput), DenseInput<f32>);

fn main() {
    let is_kaggle = std::path::Path::new("/kaggle").exists();

    let data_path = if is_kaggle {
        "/kaggle/input/datasets/kirill020708/1024hl-dataset/combined-1024hl.vf"
    } else {
        "../combined-1024hl.vf"
    };

    let (loader_threads, batch_queue) = if is_kaggle {
        (36, 256) 
    } else {
        (12, 64)
    };
    
    let buffer_size_mb = if is_kaggle { 16384 } else { 4096 };

    let psqt = ChessBucketsMirrored::new(BUCKET_LAYOUT);
    let threats = ThreatInputs::new();
    let output_buckets = MaterialCount::<NUM_OUTPUT_BUCKETS>;

    let model_inputs = ModelInputs::default()
        .add_sparse("stm/threats", (threats.num_inputs(), 1), threats.max_active())
        .add_sparse("ntm/threats", (threats.num_inputs(), 1), threats.max_active())
        .add_sparse("stm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("ntm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("buckets", (NUM_OUTPUT_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(
        &model_inputs,
        |builder, (((((stm_threats, ntm_threats), stm_psqt), ntm_psqt), output_buckets), target)| {
            let mut l0_psqt = builder.new_weights("l0/psqt", (HIDDEN_SIZE, psqt.num_inputs()), InitSettings::Normal { mean: 0.0, stdev: (2f32 / 32.0).sqrt() });
            let l0_threats = builder.new_affine("l0/threats", threats.num_inputs(), HIDDEN_SIZE);

            let l0f = builder.new_weights("l0/fac", (HIDDEN_SIZE, 768), InitSettings::Zeroed);
            l0_psqt = l0_psqt + l0f.repeat(NUM_INPUT_BUCKETS);

            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

            let stm_hidden = (l0_psqt.matmul(stm_psqt) + l0_threats.forward(stm_threats)).screlu();
            let ntm_hidden = (l0_psqt.matmul(ntm_psqt) + l0_threats.forward(ntm_threats)).screlu();
            
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            
            let output = l1.forward(hidden_layer).select(output_buckets);
            
            let loss = output.sigmoid().squared_error(target);

            (Some(loss.reduce_sum_batch()), vec![("output".to_string(), output)])
        },
    );

    let device = DefaultDevice::new(0).unwrap();
    let weights = ModelWeights::new(&defn, 12412421);
    let params = AdamWParams::default();
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device, params).unwrap();

    let saved_format = vec![
        SavedFormat::id("l0/psqt")
            .transform(|store, weights| {
                let factoriser = store.get("l0/fac").values.f32().repeat(NUM_INPUT_BUCKETS);
                weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
            })
            .round()
            .quantise::<i16>(QA),
        SavedFormat::id("l0/threats/w").round().quantise::<i16>(QA),
        SavedFormat::id("l0/threats/b").round().quantise::<i16>(QA),
        SavedFormat::id("l1/w").round().quantise::<i16>(QB).transpose(),
        SavedFormat::id("l1/b").round().quantise::<i16>(QA * QB).transpose(),
    ];

    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    optimiser.set_params_for_weight("l0/psqt", stricter_clipping);
    optimiser.set_params_for_weight("l0/fac", stricter_clipping);
    optimiser.set_params_for_weight("l0/threats/w", stricter_clipping);

    let superbatches = 480;

    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        lr_schedule: lr::Warmup {
            inner: lr::CosineDecayLR { 
                initial_lr: 0.001, 
                final_lr: 0.001 * 0.3f32.powi(5), 
                final_superbatch: superbatches 
            },
            warmup_batches: 800,
        }.boxed(),
        log_rate: 128,
    };

    let reader = ViriBinpackLoader::new(
        data_path, 
        buffer_size_mb, 
        loader_threads, 
        ViriFilter::Custom(filter::should_keep)
    );

    let mapper = ModelInputsMapper::build(
        &model_inputs,
        move |pos, step, (((((stm_threats, ntm_threats), stm_psqt), ntm_psqt), buckets), target)| {
            let mut cnt = 0;
            psqt.map_features(pos, |stm, ntm| {
                stm_psqt[cnt] = stm.try_into().unwrap();
                ntm_psqt[cnt] = ntm.try_into().unwrap();
                cnt += 1;
            });
            if cnt < psqt.max_active() {
                stm_psqt[cnt] = -1;
                ntm_psqt[cnt] = -1;
            }

            let mut stm_cnt = 0;
            let mut ntm_cnt = 0;
            threats.map_features(
                pos,
                |stm| {
                    stm_threats[stm_cnt] = stm.try_into().unwrap();
                    stm_cnt += 1;
                },
                |ntm| {
                    ntm_threats[ntm_cnt] = ntm.try_into().unwrap();
                    ntm_cnt += 1;
                },
            );
            if stm_cnt < threats.max_active() {
                stm_threats[stm_cnt] = -1;
            }
            if ntm_cnt < threats.max_active() {
                ntm_threats[ntm_cnt] = -1;
            }

            let bucket = output_buckets.bucket(pos);
            buckets[0] = bucket as i32;

            let result = f32::from(pos.result) / 2.0;
            let score = 1.0 / (1.0 + (f32::from(-pos.score) / 400.0).exp());
            let wdl_scheduler = wdl::LinearWDL { start: 0.2, end: 0.5 };
            let lambda = wdl_scheduler.blend(step.batch(), step.superbatch(), step.final_superbatch());
            target[0] = lambda * result + (1. - lambda) * score;
        }
    );

    let net_id = "potential-ti-1024hl";

    train(
        &mut optimiser,
        schedule,
        ReadMapLoader::new(reader, mapper, batch_queue as u8),
        |_, _, _| {},
        |optimiser, step| {
            let superbatch = step.superbatch();
            if superbatch.is_multiple_of(25) || superbatch == step.final_superbatch() {
                let name = format!("{net_id}-{superbatch}");
                save_to_checkpoint(optimiser, &saved_format, &format!("checkpoints/{name}"));
                println!("Saved [{name}]");
            }
        },
    ).unwrap();
}