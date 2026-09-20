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

const L1_SIZE: usize = 384;
const L2_SIZE: usize = 16;
const L3_SIZE: usize = 32;
const NUM_OUTPUT_BUCKETS: usize = 8;

const SCALE: f32 = 400.0;
const Q0: i16 = 255;
const Q1: i16 = 128;
const Q: i16 = 64;

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

const FT_SHIFT: usize = 8;
const FT_SHIFT_SCALE: f32 = Q0 as f32 / ((1 << FT_SHIFT) as f32);
const L1_RANGE: f32 = (i8::MAX as f32 / Q1 as f32) * FT_SHIFT_SCALE * FT_SHIFT_SCALE;

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
        "/kaggle/input/datasets/kirill020708/1024hl-multilayer-15m-games/1024hl-multilayer-15m-games.vf"
    } else {
        "../1024hl-multilayer-15m-games.vf"
    };

    let (loader_threads, mapper_threads) = if is_kaggle {
        (36, 16) 
    } else {
        (12, 4)
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
            let mut l0_psqt = builder.new_weights("l0/psqt", (L1_SIZE, psqt.num_inputs()), InitSettings::Normal { mean: 0.0, stdev: (2f32 / 32.0).sqrt() });
            let l0_threats = builder.new_affine("l0/threats", threats.num_inputs(), L1_SIZE);

            let l0f = builder.new_weights("l0/fac", (L1_SIZE, 768), InitSettings::Zeroed);
            l0_psqt = l0_psqt + l0f.repeat(NUM_INPUT_BUCKETS);

            let l1 = builder.new_affine("l1", 2 * L1_SIZE, NUM_OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE, NUM_OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, NUM_OUTPUT_BUCKETS);

            let stm_hidden = (l0_psqt.matmul(stm_psqt) + l0_threats.forward(stm_threats)).screlu();
            let ntm_hidden = (l0_psqt.matmul(ntm_psqt) + l0_threats.forward(ntm_threats)).screlu();
            
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            
            let l1_out = l1.forward(hidden_layer).select(output_buckets).screlu();
            let l2_out = l2.forward(l1_out).select(output_buckets).crelu();
            let output = l3.forward(l2_out).select(output_buckets);
            
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
            .quantise::<i16>(Q0),
        SavedFormat::id("l0/threatsw").round().quantise::<i16>(Q0),
        SavedFormat::id("l0/threatsb").round().quantise::<i16>(Q0),
        SavedFormat::id("l1w")
            .transform(|_, values| {
                values.iter().map(|f| f / (FT_SHIFT_SCALE * FT_SHIFT_SCALE)).collect()
            })
            .round()
            .quantise::<i8>(Q1 as i8),
        SavedFormat::id("l1b").round().quantise::<i32>(i32::from(Q) * 256),
        SavedFormat::id("l2w").round().quantise::<i32>(i32::from(Q)),
        SavedFormat::id("l2b").round().quantise::<i32>(i32::from(Q).pow(3)),
        SavedFormat::id("l3w").round().quantise::<i32>(i32::from(Q)),
        SavedFormat::id("l3b").round().quantise::<i32>(i32::from(Q).pow(4)),
    ];

    let l0_clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    optimiser.set_params_for_weight("l0/psqt", l0_clip);
    optimiser.set_params_for_weight("l0/fac", l0_clip);
    optimiser.set_params_for_weight("l0/threatsw", l0_clip);

    let l1_clip = AdamWParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    optimiser.set_params_for_weight("l1w", l1_clip);

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
            let score = 1.0 / (1.0 + (f32::from(-pos.score) / SCALE).exp());
            let wdl_scheduler = wdl::LinearWDL { start: 0.2, end: 0.5 };
            let lambda = wdl_scheduler.blend(step.batch(), step.superbatch(), step.final_superbatch());
            target[0] = lambda * result + (1. - lambda) * score;
        }
    );

    let net_id = "potential-ti-384hl-ml";

    train(
        &mut optimiser,
        schedule,
        ReadMapLoader::new(reader, mapper, mapper_threads as u8),
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

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/P2P2PP/rq2Q1R1K w kq - 0 2",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQka - 0 1",
        "3N4/b2R2p1/3q3r/6P1/4k1nQ/7B/8/K7 w - - 0 1",
        "k2B1Q1q/8/b7/4p3/3Pr3/1N5R/2n5/1K6 w - - 0 1",
        "1B3q2/8/r5n1/8/Rp1N1PQ1/8/4bk2/2K5 w - - 0 1",
        "8/5NR1/5q1b/8/7p/3P2B1/6Q1/1k1K1n1r w - - 0 1",
        "8/8/6r1/4B3/3Q3p/N1nq4/5RP1/b3K2k b - - 0 1",
        "3qn2Q/1R6/8/1N3b1p/4B3/1kP5/r7/5K2 b - - 0 1",
        "3rBR2/2qQ1p2/N7/2P2b2/6n1/k7/8/6K1 b - - 0 1",
        "k7/8/p1rB1q2/7Q/4R3/2N2n2/7P/6bK b - - 0 1",
        "2n2Rr1/Bk5p/N7/2Q3q1/b7/8/KP6/8 w - - 0 1",
        "8/Q6r/3qR1P1/b4p2/k7/3B4/1KN2n2/8 b - - 0 1",
        "2nR4/1qB5/2p5/7r/4bQ2/1P1N4/2K1k3/8 w - - 0 1",
        "8/2Q1B3/n3qR1r/bk1p4/1P6/8/3K4/7N w - - 0 1",
        "7r/4b3/4k1N1/2q4n/1Q2B3/R5p1/1P2K3/8 b - - 0 1",
        "2r1n1k1/NbR5/6B1/2p1P3/8/8/5K2/q6Q b - - 0 1",
        "2Q2R2/P1pn4/q1N5/1b5k/1r6/B7/6K1/8 b - - 0 1",
        "1Nr2b2/R1p5/5q2/7B/2P5/3nk3/7K/1Q6 w - - 0 1",
        "4Q3/6P1/1k3p2/4N3/2r5/K6b/1n1B2Rq/8 b - - 0 1",
        "1B5Q/1n6/2p1rN2/3R4/3P4/1K3k2/3b4/6q1 w - - 0 1",
        "3n4/3q4/5Q2/4rP2/1N2p3/2K2B2/5k2/2b4R b - - 0 1",
        "6B1/2k5/2n1R3/1q2p3/2P4Q/3K4/r5b1/3N4 w - - 0 1",
        "8/8/b6N/R3pr1n/Q7/1Pk1K3/4B3/5q2 b - - 0 1",
        "3Q2r1/4P2R/1b6/8/8/1B3K2/4p2q/1k1n1N2 w - - 0 1",
        "bR5q/2r3B1/2Q1P3/8/2n5/1N1p2K1/k7/8 w - - 0 1",
        "1q1b2r1/8/8/2p5/4N3/3k1P1K/2nB1Q2/4R3 w - - 0 1",
        "5rRq/8/1Qn5/8/K7/P1B4b/1p2N3/7k w - - 0 1",
        "1n6/8/B3q3/5R2/1KPb2N1/7Q/r4p2/2k5 w - - 0 1",
        "q3N1R1/8/1B5n/2p5/2K2P2/7r/1b1k4/7Q w - - 0 1",
        "1B6/N6q/2b5/7R/P2K4/1Q1pr3/6n1/2k5 b - - 0 1",
        "1R3q2/p3Q1n1/4N3/6r1/4K1B1/2P5/7b/4k3 w - - 0 1",
        "1k6/2RQP3/1p6/b7/1B3K2/r1n5/3Nq3/8 b - - 0 1",
        "b7/k7/5P2/n2N4/5pK1/2q5/2B2R2/r4Q2 b - - 0 1",
        "1B6/P4q2/5r2/8/1k2n2K/5b2/1NR1p3/6Q1 w - - 0 1",
        "8/3Pk3/B2r4/K5N1/b7/3n1p1Q/2R5/5q2 w - - 0 1",
        "q2Q1R2/2p4N/1b1P4/1K6/1B3r2/8/8/n2k4 w - - 0 1",
        "n1k5/5pq1/R4b2/2K5/3N4/7P/4BrQ1/8 b - - 0 1",
        "8/4Q3/B7/3KN1P1/3b4/nk3p2/8/R4r1q w - - 0 1",
        "b6n/B1k5/8/4KN1r/1Q6/7R/6Pp/5q2 b - - 0 1",
        "6k1/7r/8/bB3K1N/1R1q4/4Q3/2nP1p2/8 w - - 0 1",
        "Q6R/8/2B1q3/3N1nK1/2kb4/P7/r6p/8 w - - 0 1",
        "8/p5r1/k7/6PK/3b4/2B5/n4qQ1/3N2R1 w - - 0 1",
        "4kb2/6r1/K7/p7/6n1/2N5/2BP1qR1/7Q w - - 0 1",
        "6q1/1B5/1K3P2/3br1np/3R4/Q7/8/5k2 w - - 0 1",
        "5n2/5q2/1NK5/k1P3r1/3p4/7Q/B6b/1R6 w - - 0 1",
        "B3r3/3p4/N2K2k1/1Q6/2R5/1bP5/1q5n/8 w - - 0 1",
        "BR2Q3/4N3/1n2K3/k7/1p1b1q2/8/5P2/7r b - - 0 1",
        "1k6/7R/5K1N/1pQ5/1n6/P4b2/1r6/6qB b - - 0 1",
        "8/3k4/3NnPK1/3QR3/3r2pB/8/4b3/q7 w - - 0 1",
        "1Q6/4q3/NB5K/1R1r4/3P4/bp1k4/6n1/8 w - - 0 1",
        "3Br3/K7/2q1N3/7n/8/4PbRQ/1p1k4/8 w - - 0 1",
        "R2r4/pK1b4/1n4NB/7P/8/3Q4/6k1/4q3 b - - 0 1",
        "3N2r1/2KP4/8/1B1p4/2b5/3RQq2/2k5/7n w - - 0 1",
        "5q2/1N1KB3/5b2/p4R2/4k3/P7/Q7/4n1r1 b - - 0 1",
        "NR6/4K3/1q3r2/3Q3P/3n2k1/8/7p/B5b1 b - - 0 1",
        "q7/1N1B1K2/1Q6/5b2/5pP1/6r1/n6k/R7 w - - 0 1",
        "2R5/2n1k1K1/5r2/3P4/2Q4p/2q5/6NB/7b w - - 0 1",
        "3n1Qr1/3p3K/8/3B4/R5b1/4P3/1qN4k/8 w - - 0 1",
        "K7/3k4/3n2b1/1P2r3/8/p2Bq3/3R4/3QN3 b - - 0 1",
        "1K6/8/3rRN2/1BP3b1/3p4/8/k2n2q1/5Q2 w - - 0 1",
        "2K5/6Bn/p4r2/2P1Q3/1qb5/8/2R5/3kN3 w - - 0 1",
        "3K4/8/2bP4/1qN5/2n3B1/3R4/4Qrp1/6k1 b - - 0 1",
        "1B2K1k1/P3b3/5q2/3R4/1pQ2r1n/8/8/6N1 b - - 0 1",
        "5K2/p4P1b/5QB1/4q3/6k1/8/4r3/R1n1N3 b - - 0 1",
        "6K1/8/b6R/N2p2P1/8/q1Q5/6r1/2Bk3n b - - 0 1",
        "7K/r2R3b/1Q6/8/2q5/1nPB2k1/N3p3/8 w - - 0 1",
    ] {
        let eval = optimiser.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", SCALE * eval);
    }
}
