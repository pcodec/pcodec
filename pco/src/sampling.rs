use std::cmp::max;
use std::collections::HashMap;

use crate::macros::match_latent_enum;
use crate::{constants::CLASSIC_MEMORIZABLE_BINS_LOG, metadata::DynLatents};
use rand_xoshiro::rand_core::{RngCore, SeedableRng};

use crate::data_types::Latent;

pub const MIN_SAMPLE: usize = 10;
// Int mults will be considered infrequent if they occur less than 1/this of
// the time.
const CLASSIC_MEMORIZABLE_BINS: f64 = (1 << CLASSIC_MEMORIZABLE_BINS_LOG) as f64;
const DELTA_MAX_GROUPS: usize = 16;
const DELTA_TARGET_GROUP_SIZE: usize = 200;

fn sample_ratio(compression_level: usize) -> usize {
  // we use approximately n/sample_ratio nums for the sample
  match compression_level {
    0 => 100,
    1 => 84,
    2 => 73,
    3 => 64,
    4 => 57,
    5 => 52,
    6 => 47,
    7 => 43,
    8 => 40,
    9 => 35,
    10 => 31,
    11 => 28,
    12 => 25,
    _ => unreachable!("impossible compression level"),
  }
}

fn calc_sample_n(n: usize, compression_level: usize) -> Option<usize> {
  if n >= MIN_SAMPLE {
    Some(MIN_SAMPLE + (n - MIN_SAMPLE) / sample_ratio(compression_level))
  } else {
    None
  }
}

// extracted for testing
fn choose_delta_sample_inner(
  primary_latents: &DynLatents,
  group_size: usize,
  n_groups: usize,
) -> DynLatents {
  let n = primary_latents.len();
  let nominal_sample_size = n_groups * group_size;
  let group_step = group_size + n.saturating_sub(nominal_sample_size) / (n_groups.max(2) - 1);

  match_latent_enum!(
    primary_latents,
    DynLatents<L>(primary_latents) => {
      let mut sample = Vec::<L>::with_capacity(nominal_sample_size);
      for group_i in 0..n_groups {
        let group_start = group_i * group_step;
        sample.extend(&primary_latents[group_start..group_start + group_size]);
      }
      DynLatents::new(sample)
    }
  )
}

pub(crate) fn choose_delta_sample(
  primary_latents: &DynLatents,
  compression_level: usize,
) -> Option<DynLatents> {
  // We select some large contiguous groups of latents so that delta encodings
  // can work both sample from a varienty of places from the chunk while
  // minimizing the number of unnatural jumps.
  let n = primary_latents.len();
  let target_sample_n = calc_sample_n(n, compression_level)?;
  let n_groups = ((target_sample_n as f32 / DELTA_TARGET_GROUP_SIZE as f32).ceil() as usize)
    .min(DELTA_MAX_GROUPS);
  let group_n = target_sample_n / n_groups;
  Some(choose_delta_sample_inner(
    primary_latents,
    group_n,
    n_groups,
  ))
}

#[inline]
fn is_visited(visited: &[u8], idx: usize) -> bool {
  visited[idx / 8] & (1 << (idx % 8)) != 0
}

#[inline]
fn mark_visited(visited: &mut [u8], idx: usize) {
  visited[idx / 8] |= 1 << (idx % 8);
}

#[inline(never)]
pub fn choose_mode_sample<T, S, Filter: Fn(&T) -> Option<S>>(
  nums: &[T],
  compression_level: usize,
  filter: Filter,
) -> Option<Vec<S>> {
  // We use Floyd's algorithm to draw a sample without replacement or modifying
  // nums. One nice property is that if we take a larger sample, it will begin
  // with the same subset as a smaller sample, keeping noise manageable.
  let n = nums.len();
  let target_sample_size = calc_sample_n(n, compression_level)?;

  let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0);
  let mut visited = vec![0_u8; n.div_ceil(8)];
  let mut res = Vec::with_capacity(target_sample_size);
  for j in (n - target_sample_size)..n {
    let t = (rng.next_u64() % (j as u64 + 1)) as usize;
    let idx = if is_visited(&visited, t) { j } else { t };
    mark_visited(&mut visited, idx);
    if let Some(x) = filter(&nums[idx]) {
      res.push(x);
    }
  }

  if res.len() >= MIN_SAMPLE {
    Some(res)
  } else {
    None
  }
}

pub struct PrimaryLatentAndSavings<L: Latent> {
  pub primary: L,
  pub bits_saved: f64,
}

#[inline(never)]
pub fn est_bits_saved_per_num<L: Latent, S: Copy, F: Fn(S) -> PrimaryLatentAndSavings<L>>(
  sample: &[S],
  primary_fn: F,
) -> f64 {
  let mut primary_counts_and_savings = HashMap::<L, (usize, f64)>::with_capacity(sample.len());
  for &x in sample {
    let PrimaryLatentAndSavings {
      primary: primary_latent,
      bits_saved,
    } = primary_fn(x);
    let entry = primary_counts_and_savings
      .entry(primary_latent)
      .or_default();
    entry.0 += 1;
    entry.1 += bits_saved;
  }

  let infrequent_cutoff = max(
    1,
    (sample.len() as f64 / CLASSIC_MEMORIZABLE_BINS) as usize,
  );

  // Maybe this should be made fuzzy instead of a hard cutoff because it's just
  // a sample.
  let sample_bits_saved = primary_counts_and_savings
    .values()
    .filter(|&&(count, _)| count <= infrequent_cutoff)
    .map(|&(_, bits_saved)| bits_saved)
    .sum::<f64>();
  sample_bits_saved / sample.len() as f64
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sample_ratio_monotonic() {
    for i in 1..12 {
      assert!(sample_ratio(i) <= sample_ratio(i - 1));
    }
  }

  #[test]
  fn test_sample_n() {
    assert_eq!(calc_sample_n(9, 8), None);
    assert_eq!(calc_sample_n(10, 8), Some(10));
    assert_eq!(calc_sample_n(100, 8), Some(12));
    assert_eq!(calc_sample_n(1000010, 8), Some(25010));
    assert_eq!(calc_sample_n(1000010, 0), Some(10010));
  }

  #[test]
  fn test_choose_delta_sample() {
    let latents = DynLatents::new(vec![0_u32, 1]);
    assert_eq!(
      choose_delta_sample_inner(&latents, 2, 1)
        .downcast::<u32>()
        .unwrap(),
      vec![0, 1]
    );
    assert_eq!(
      choose_delta_sample_inner(&latents, 1, 2)
        .downcast::<u32>()
        .unwrap(),
      vec![0, 1]
    );

    let latents = DynLatents::new((0..300).collect::<Vec<u32>>());
    let sample = choose_delta_sample_inner(&latents, 100, 2)
      .downcast::<u32>()
      .unwrap();
    assert_eq!(sample.len(), 200);
    assert_eq!(&sample[..3], &[0, 1, 2]);
    assert_eq!(&sample[197..], &[297, 298, 299]);

    let latents = DynLatents::new((0..8).collect::<Vec<u32>>());
    assert_eq!(
      choose_delta_sample_inner(&latents, 2, 3)
        .downcast::<u32>()
        .unwrap(),
      vec![0, 1, 3, 4, 6, 7]
    );
  }

  #[test]
  fn test_choose_mode_sample() {
    let mut nums = Vec::new();
    for i in 0..150 {
      nums.push(-i as f32);
    }
    let mut sample = choose_mode_sample(&nums, 8, |&num| {
      if num == 0.0 {
        None
      } else {
        Some(num)
      }
    })
    .unwrap();
    sample.sort_unstable_by(f32::total_cmp);
    assert_eq!(sample.len(), 13);
    assert_eq!(&sample[0..3], &[-135.0, -131.0, -114.0]);
  }
}
