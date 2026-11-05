use std::{
    collections::BTreeMap,
    fs,
    io::{self, Error},
    ops::Range,
};

use bullet_compiler::{model::split::WeightRole, tensor::TValue};

use crate::{model::utils::write_to_byte_buffer, optimiser::utils::load_weights_from_file};

fn local_rows(rows: usize, segments: &[usize], ranks: usize, rank: usize) -> Vec<Range<usize>> {
    assert_eq!(segments.iter().sum::<usize>(), rows);

    let mut start = 0;
    let mut ranges = Vec::new();

    for len in segments {
        let block = len / ranks;
        ranges.push(start + rank * block..start + (rank + 1) * block);
        start += len;
    }

    ranges
}

pub fn scatter_rows(full: &[f32], rows: usize, segments: &[usize], ranks: usize, rank: usize) -> Vec<f32> {
    let ranges = local_rows(rows, segments, ranks, rank);
    let columns = full.chunks_exact(rows);
    columns.flat_map(|column| ranges.iter().flat_map(|range| &column[range.clone()])).copied().collect()
}

pub fn gather_rows(shards: &[Vec<f32>], rows: usize, segments: &[usize]) -> Vec<f32> {
    let local = rows / shards.len();
    let mut full = vec![0.0; shards[0].len() * shards.len()];

    for (rank, shard) in shards.iter().enumerate() {
        let ranges = local_rows(rows, segments, shards.len(), rank);

        for (column, values) in full.chunks_exact_mut(rows).zip(shard.chunks_exact(local)) {
            let mut values = values.iter();
            for range in &ranges {
                column[range.clone()].iter_mut().zip(&mut values).for_each(|(x, y)| *x = *y);
            }
        }
    }

    full
}

pub fn merge_state_dirs(
    roles: &BTreeMap<String, WeightRole>,
    sizes: &BTreeMap<String, usize>,
    ranks: &[String],
    out: &str,
) -> io::Result<()> {
    for file in state_files(&ranks[0])? {
        if !file.ends_with(".bin") {
            fs::copy(format!("{}/{file}", ranks[0]), format!("{out}/{file}"))?;
            continue;
        }

        let records = ranks.iter().map(|dir| load_weights_from_file(&format!("{dir}/{file}"))).collect::<Vec<_>>();
        let mut bytes = Vec::new();

        for (i, (name, values)) in records[0].iter().enumerate() {
            let matches = |values: &Vec<f32>| sizes.get(name).is_none_or(|&size| values.len() == size);

            let values = match roles.get(name) {
                Some(WeightRole::Sharded { rows, segments }) => {
                    let shards = records.iter().map(|records| records.get(i).filter(|(id, _)| id == name));
                    let shards = shards.map(|record| record.map(|(_, values)| values.clone()));

                    match shards.collect::<Option<Vec<_>>>() {
                        Some(shards) if shards.iter().all(matches) => gather_rows(&shards, *rows, segments),
                        _ => return Err(bad_record(&file, name)),
                    }
                }
                Some(_) if !matches(values) => return Err(bad_record(&file, name)),
                _ => values.clone(),
            };

            bytes.extend(write_to_byte_buffer(&TValue::F32(values), name)?);
        }

        fs::write(format!("{out}/{file}"), bytes)?;
    }

    Ok(())
}

pub fn split_state_dir(
    roles: &BTreeMap<String, WeightRole>,
    sizes: &BTreeMap<String, usize>,
    dir: &str,
    out: &[String],
) -> io::Result<()> {
    for file in state_files(dir)?.into_iter().filter(|file| file != "weights.bin") {
        if !file.ends_with(".bin") {
            for path in out {
                fs::copy(format!("{dir}/{file}"), format!("{path}/{file}"))?;
            }
            continue;
        }

        let records = load_weights_from_file(&format!("{dir}/{file}"));

        for (rank, path) in out.iter().enumerate() {
            let mut bytes = Vec::new();

            for (name, values) in &records {
                let values = match (roles.get(name), sizes.get(name)) {
                    (Some(WeightRole::Sharded { rows, segments }), Some(size)) if values.len() == size * out.len() => {
                        scatter_rows(values, *rows, segments, out.len(), rank)
                    }
                    (Some(WeightRole::Sharded { .. }), _) => return Err(bad_record(&file, name)),
                    (_, Some(&size)) if values.len() != size => return Err(bad_record(&file, name)),
                    _ => values.clone(),
                };

                bytes.extend(write_to_byte_buffer(&TValue::F32(values), name)?);
            }

            fs::write(format!("{path}/{file}"), bytes)?;
        }
    }

    Ok(())
}

fn state_files(dir: &str) -> io::Result<Vec<String>> {
    let mut files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            files.push(entry.file_name().to_string_lossy().into_owned());
        }
    }

    files.sort();
    Ok(files)
}

fn bad_record(file: &str, name: &str) -> Error {
    Error::other(format!("Tensor parallelism cannot checkpoint '{file}': record '{name}' does not match its weight!"))
}
