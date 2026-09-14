use bullet_lib::{
    game::{inputs::{ChessBucketsMirrored, get_num_buckets}, outputs::MaterialCount},
    nn::{optimiser::{AdamW, AdamWParams}, InitSettings, Shape},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const HIDDEN_SIZE: usize = 1024;
const L2_SIZE: usize = 16;
const L3_SIZE: usize = 32;
const NUM_OUTPUT_BUCKETS: usize = 8;

const SCALE: i32 = 400;
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

const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

const FT_SHIFT: usize = 8;
const FT_SHIFT_SCALE: f32 = Q0 as f32 / ((1 << FT_SHIFT) as f32);
const L1_RANGE: f32 = (i8::MAX as f32 / Q1 as f32) * FT_SHIFT_SCALE * FT_SHIFT_SCALE;

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(Q0),
            SavedFormat::id("l0b").round().quantise::<i16>(Q0),
            SavedFormat::id("l1w")
                .transform(|_, values| {
                    values.iter().map(|f| f / (FT_SHIFT_SCALE * FT_SHIFT_SCALE)).collect()
                })
                .round()
                .quantise::<i8>(Q1),
            SavedFormat::id("l1b").round().quantise::<i32>(i32::from(Q) * 256),
            SavedFormat::id("l2w").round().quantise::<i32>(i32::from(Q)),
            SavedFormat::id("l2b").round().quantise::<i32>(i32::from(Q).pow(3)),
            SavedFormat::id("l3w").round().quantise::<i32>(i32::from(Q)),
            SavedFormat::id("l3b").round().quantise::<i32>(i32::from(Q).pow(4)),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HIDDEN_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", 3 * L2_SIZE, NUM_OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, NUM_OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);

            let l1_out = l1.forward(hidden_layer).select(buckets);
            
            let act1 = l1_out.crelu();
            let act2 = l1_out.screlu();
            // Third Activation: Asymmetric Clamp (-0.5)
            let act3 = l1_out.min(0.0).max(-0.5);
            let l1_out = act1.concat(act2).concat(act3);
            
            let l2_out = l2.forward(l1_out).select(buckets).crelu();
            l3.forward(l2_out).select(buckets)
        });

    let l0_clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", l0_clip);
    trainer.optimiser.set_params_for_weight("l0f", l0_clip);

    let l1_clip = AdamWParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l1w", l1_clip);

    let superbatches = 480;

    let mut schedule = TrainingSchedule {
        net_id: "potential-1024hl-triple-act".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.2, end: 0.5 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR {
                initial_lr: 0.001,
                final_lr: 0.001 * 0.3f32.powi(5),
                final_superbatch: superbatches,
            },
            warmup_batches: 800,
        },
        save_rate: 80,
    };

    let is_kaggle = std::path::Path::new("/kaggle").exists();

    let (trainer_threads, loader_threads, batch_queue) = if is_kaggle {
        (8, 36, 256)
    } else {
        (4, 12, 64)
    };

    let buffer_size_mb = if is_kaggle { 16384 } else { 4096 };

    let file_path = if is_kaggle {
        "/kaggle/input/datasets/kirill020708/1024hl-dataset/combined-1024hl.vf"
    } else {
        "../combined-1024hl.vf"
    };

    let settings = LocalSettings {
        threads: trainer_threads,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: batch_queue,
    };

    let data_loader = {
        use loader::viribinpack::{Filter, ViriBinpackLoader};

        let filter = Filter {
            ..Default::default()
        };

        ViriBinpackLoader::new(file_path, buffer_size_mb, loader_threads, filter)
    };

    trainer.run(&schedule, &settings, &data_loader);
}