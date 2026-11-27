use std::mem::MaybeUninit;
use std::{array, cmp};

use crate::constants::{Bitlen, DeltaLookback};
use crate::data_types::Latent;
use crate::metadata::DeltaLookbackConfig;
use crate::FULL_BATCH_N;

// there are 3 types of proposed lookbacks:
// * brute force: just try the most recent few latents
// * repeating: try the most recent lookbacks we actually used
// * hash: look up similar values by hash
const PROPOSED_LOOKBACKS: usize = 16;
const BRUTE_LOOKBACKS: usize = 6;
const REPEATING_LOOKBACKS: usize = 4;
// To help locate similar latents for lookback encoding, we hash each latent at
// different "coarsenesses" and write them into a vector. e.g. a coarseness
// of 8 means that (l >> 8) gets hashed, so we can lookup recent values by
// quotient by 256.
const COARSENESSES: [Bitlen; 2] = [0, 8];

fn hash_lookup(
  l: u64,
  i: usize,
  hash_table_n: usize,
  window_n: usize,
  idx_hash_table: &mut [usize],
  proposed_lookbacks: &mut [usize; PROPOSED_LOOKBACKS],
) {
  let hash_mask = hash_table_n - 1;
  // might be possible to improve this hash fn
  let hash_fn = |mut x: u64| {
    // constant is roughly 2**64 / phi
    x = (x ^ (x >> 32)).wrapping_mul(11400714819323197441);
    x = x ^ (x >> 32);
    x as usize & hash_mask
  };

  let mut proposal_idx = BRUTE_LOOKBACKS + REPEATING_LOOKBACKS;
  let mut offset = 0;
  for coarseness in COARSENESSES {
    let bucket = l >> coarseness;
    let buckets = [bucket.wrapping_sub(1), bucket, bucket.wrapping_add(1)];
    let hashes = buckets.map(hash_fn);
    for h in hashes {
      // SAFETY: `h = hash_fn(...) & hash_mask < hash_table_n`, and
      // `offset` is a multiple of `hash_table_n` with at most
      // `COARSENESSES.len() - 1` increments, so `offset + h < idx_hash_table.len()`.
      let lookback_to_last_instance = unsafe { i - *idx_hash_table.get_unchecked(offset + h) };
      proposed_lookbacks[proposal_idx] = if lookback_to_last_instance <= window_n {
        lookback_to_last_instance
      } else {
        proposal_idx.min(i)
      };
      proposal_idx += 1;
    }
    let h = hashes[1];
    // SAFETY: same bounds argument as the read above.
    unsafe {
      *idx_hash_table.get_unchecked_mut(offset + h) = i;
    }
    offset += hash_table_n;
  }
}

#[inline(never)]
fn find_best_lookback<L: Latent>(
  l: L,
  i: usize,
  latents: &[L],
  proposed_lookbacks: &[usize; PROPOSED_LOOKBACKS],
  lookback_goodnesses: &[u8],
) -> usize {
  let mut best_goodness = 0;
  let mut best_lookback: usize = 0;
  for &lookback in proposed_lookbacks {
    // SAFETY: each `lookback` comes from `proposed_lookbacks`, whose entries
    // are initialised to `(k+1).min(state_n) >= 1` and subsequently updated
    // only to values in `[1, window_n]`.  `window_n <= lookback_goodnesses.len()`,
    // so `lookback - 1 < lookback_goodnesses.len()`.  `i >= state_n >= lookback`,
    // so `i - lookback` doesn't underflow and stays in `[0, latents.len())`.
    let (lookback_goodness, other) = unsafe {
      (
        *lookback_goodnesses.get_unchecked(lookback - 1),
        *latents.get_unchecked(i - lookback),
      )
    };
    let delta = L::min(l.wrapping_sub(other), other.wrapping_sub(l));
    let delta_goodness = delta.leading_zeros();
    let goodness = lookback_goodness as Bitlen + delta_goodness;
    if goodness > best_goodness {
      best_goodness = goodness;
      best_lookback = lookback;
    }
  }
  best_lookback
}

#[inline(never)]
pub fn choose_lookbacks<L: Latent>(
  config: DeltaLookbackConfig,
  latents: &[L],
) -> Vec<DeltaLookback> {
  let state_n = config.state_n();

  if latents.len() <= state_n {
    return vec![];
  }

  let hash_table_n_log = config.window_n_log + 1;
  let hash_table_n = 1 << hash_table_n_log;
  let window_n = config.window_n();
  assert!(
    window_n >= PROPOSED_LOOKBACKS,
    "we do not support tiny windows during compression"
  );

  let n_lookbacks = window_n.min(latents.len());
  let mut lookback_counts = vec![1_u32; n_lookbacks];
  // `lookback_goodnesses[j]` is always the bit width of `lookback_counts[j]`,
  // which only changes when the count crosses a power of two. Maintaining it
  // there keeps a `leading_zeros` out of the hot loop below.
  let mut lookback_goodnesses = vec![1_u8; n_lookbacks];
  let mut lookbacks = Vec::with_capacity(latents.len() - state_n);
  let uninit_lookbacks = lookbacks.spare_capacity_mut();
  let mut idx_hash_table = vec![0_usize; COARSENESSES.len() * hash_table_n];
  let mut proposed_lookbacks = array::from_fn::<_, PROPOSED_LOOKBACKS, _>(|i| (i + 1).min(state_n));
  let mut best_lookback = 1;
  let mut repeating_lookback_idx: usize = 0;
  for i in state_n..latents.len() {
    let l = latents[i];

    let new_brute_lookback = i.min(PROPOSED_LOOKBACKS);
    proposed_lookbacks[new_brute_lookback - 1] = new_brute_lookback;

    hash_lookup(
      l.to_u64(),
      i,
      hash_table_n,
      window_n,
      &mut idx_hash_table,
      &mut proposed_lookbacks,
    );
    let new_best_lookback = find_best_lookback(
      l,
      i,
      latents,
      &proposed_lookbacks,
      &lookback_goodnesses,
    );
    if new_best_lookback != best_lookback {
      repeating_lookback_idx += 1;
    }
    proposed_lookbacks[BRUTE_LOOKBACKS + (repeating_lookback_idx) % REPEATING_LOOKBACKS] =
      new_best_lookback;
    best_lookback = new_best_lookback;
    uninit_lookbacks[i - state_n] = MaybeUninit::new(best_lookback as DeltaLookback);
    let count = &mut lookback_counts[best_lookback - 1];
    *count += 1;
    if count.is_power_of_two() {
      lookback_goodnesses[best_lookback - 1] += 1;
    }
  }

  unsafe { lookbacks.set_len(latents.len() - state_n) };
  lookbacks
}

// All encode in place functions leave junk data (`state_n` latents in this
// case) at the front of the latents.
// Using the front instead of the back is preferable because it means we don't
// need an extra copy of the latents in this case.
#[inline(never)]
pub fn encode_in_place<L: Latent>(
  config: DeltaLookbackConfig,
  lookbacks: &[DeltaLookback],
  latents: &mut [L],
) -> Vec<L> {
  let state_n = config.state_n();
  let real_state_n = cmp::min(latents.len(), state_n);
  // TODO make this fast
  for i in (real_state_n..latents.len()).rev() {
    let lookback = lookbacks[i - state_n] as usize;
    latents[i] = latents[i].wrapping_sub(latents[i - lookback])
  }

  let mut state = vec![L::ZERO; state_n];
  state[state_n - real_state_n..].copy_from_slice(&latents[..real_state_n]);

  super::toggle_center_in_place(latents);

  state
}

pub fn new_window_buffer_and_pos<L: Latent>(
  config: DeltaLookbackConfig,
  state: &[L],
) -> (Vec<L>, usize) {
  let window_n = config.window_n();
  let buffer_n = cmp::max(window_n, FULL_BATCH_N) * 2;
  // TODO better default window
  let mut res = vec![L::ZERO; buffer_n];
  res[window_n - state.len()..window_n].copy_from_slice(state);
  (res, window_n)
}

// Returns the decoded latents, which live in the window buffer, or None if
// the lookbacks were corrupt, in which case the window buffer and its position
// are left unchanged.
pub fn decode<'a, L: Latent>(
  config: DeltaLookbackConfig,
  lookbacks: &[DeltaLookback],
  window_buffer_pos: &mut usize,
  window_buffer: &'a mut [L],
  latents: &[L],
) -> Option<&'a [L]> {
  let (window_n, state_n) = (config.window_n(), config.state_n());
  // Lookbacks can be shorter than latents in the final batch,
  // but we always decompress latents.len() numbers
  let batch_n = latents.len();
  let lookbacks = &lookbacks[..cmp::min(batch_n, lookbacks.len())];

  // We report corrupt lookbacks rather than clamping them, which would silence
  // the problem. Validating all lookbacks in a single vectorized pass is
  // cheaper than checking each one in the serial loop below.
  let max_lookback = lookbacks.iter().copied().fold(0, DeltaLookback::max);
  if max_lookback as usize > window_n {
    return None;
  }

  let mut start_pos = *window_buffer_pos;
  if start_pos + batch_n > window_buffer.len() {
    // we need to cycle the buffer
    window_buffer.copy_within(start_pos - window_n..start_pos, 0);
    start_pos = window_n;
  }

  // SAFETY: the buffer was cycled so that start_pos + batch_n is in bounds,
  // and start_pos >= window_n >= lookback, so the lookback can't underflow.
  unsafe {
    let dst = window_buffer.as_mut_ptr().add(start_pos);
    for (i, &lookback) in lookbacks.iter().enumerate() {
      let latent = latents.get_unchecked(i).toggle_center();
      let dst = dst.add(i);
      *dst = latent.wrapping_add(*dst.sub(lookback as usize));
    }
  }

  let end_pos = start_pos + batch_n;
  *window_buffer_pos = end_pos;
  Some(&window_buffer[start_pos - state_n..end_pos - state_n])
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::data_types::latent_priv::LatentPriv;
  use crate::metadata::DeltaLookbackConfig;

  #[test]
  fn test_lookback_encode_decode() {
    let original_latents = {
      let mut res = vec![100_u32; 100];
      res[1] = 200;
      res[2] = 201;
      res[3] = 202;
      res[5] = 203;
      res[15] = 204;
      res[50] = 205;
      res
    };
    let config = DeltaLookbackConfig {
      window_n_log: 4,
      state_n_log: 1,
    };
    let window_n = config.window_n();
    assert_eq!(window_n, 16);
    let state_n = config.state_n();
    assert_eq!(state_n, 2);

    let mut deltas = original_latents.clone();
    let lookbacks = choose_lookbacks(config, &original_latents);
    assert_eq!(lookbacks[0], 1); // 201 -> 200
    assert_eq!(lookbacks[2], 4); // 0 -> 0
    assert_eq!(lookbacks[13], 10); // 204 -> 203
    assert_eq!(lookbacks[48], 1); // 205 -> 0; 204 was outside window

    let state = encode_in_place(config, &lookbacks, &mut deltas);
    assert_eq!(state, vec![100, 200]);

    // Encoding left junk deltas at the front,
    // but for decoding we need junk deltas at the end.
    let mut deltas_to_decode = Vec::<u32>::new();
    deltas_to_decode.extend(&deltas[state_n..]);
    for _ in 0..state_n {
      deltas_to_decode.push(1337);
    }

    let (mut window_buffer, mut pos) = new_window_buffer_and_pos(config, &state);
    assert_eq!(pos, window_n);
    let decoded = decode(
      config,
      &lookbacks,
      &mut pos,
      &mut window_buffer,
      &deltas_to_decode,
    );
    assert_eq!(decoded, Some(original_latents.as_slice()));
    assert_eq!(pos, window_n + original_latents.len());
  }

  #[test]
  fn test_decode_matches_naive() {
    let mut rng_state = 1_u64;
    let mut rng = || {
      rng_state = rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      rng_state >> 33
    };

    for (window_n_log, state_n_log) in [(4, 0), (4, 1), (10, 2)] {
      let config = DeltaLookbackConfig {
        window_n_log,
        state_n_log,
      };
      let (window_n, state_n) = (config.window_n(), config.state_n());
      let state = (0..state_n as u32).map(|i| 1000 + i).collect::<Vec<_>>();
      let (mut window_buffer, mut pos) = new_window_buffer_and_pos(config, &state);

      // the naive decoded sequence, including the initial window
      let mut expected = vec![0_u32; window_n - state_n];
      expected.extend(&state);

      // Enough batches to cycle the window buffer several times, with every
      // short constant lookback, some long ones, and some varied lookbacks.
      let mut lookback_options: Vec<Option<usize>> = (1..=window_n.min(20)).map(Some).collect();
      lookback_options.extend([Some(window_n), None]);
      for batch_idx in 0..40 {
        let batch_n = if batch_idx % 7 == 6 {
          FULL_BATCH_N - 3
        } else {
          FULL_BATCH_N
        };
        let lookbacks = match lookback_options[batch_idx % lookback_options.len()] {
          Some(lookback) => vec![lookback as DeltaLookback; batch_n],
          None => (0..batch_n)
            .map(|_| 1 + rng() as DeltaLookback % window_n as DeltaLookback)
            .collect(),
        };
        let latents = (0..batch_n).map(|_| rng() as u32).collect::<Vec<_>>();

        let start = expected.len();
        for (&latent, &lookback) in latents.iter().zip(&lookbacks) {
          let prev = expected[expected.len() - lookback as usize];
          expected.push(latent.toggle_center().wrapping_add(prev));
        }

        let decoded = decode(
          config,
          &lookbacks,
          &mut pos,
          &mut window_buffer,
          &latents,
        );
        assert_eq!(
          decoded,
          Some(&expected[start - state_n..start + batch_n - state_n]),
          "window_n_log={} state_n_log={} batch_idx={}",
          window_n_log,
          state_n_log,
          batch_idx,
        );
      }
    }
  }

  #[test]
  fn test_corrupt_lookbacks_do_not_panic() {
    let config = DeltaLookbackConfig {
      state_n_log: 0,
      window_n_log: 2,
    };
    let delta_state = vec![0_u32];
    let lookbacks = vec![5, 1, 1, 1];
    let latents = vec![1_u32, 2, 3, 4];
    let (mut window_buffer, mut pos) = new_window_buffer_and_pos(config, &delta_state);
    let decoded = decode(
      config,
      &lookbacks,
      &mut pos,
      &mut window_buffer,
      &latents,
    );
    assert!(decoded.is_none());
  }
}

#[cfg(feature = "bench")]
mod micro {
  use divan::Bencher;

  use super::*;
  use crate::bench_utils::{uniform_latents, BENCH_N};
  use crate::metadata::DeltaLookbackConfig;

  // Interleaved subsequences: each latent's predecessor sits this far back, so
  // every lookback reaches the same distance into the window.
  const N_SUBSEQS: DeltaLookback = 16;

  #[divan::bench(types = [u8, u16, u32, u64], name = crate::bench_utils::arch_bench_name!("decode_in_place"))]
  fn decode<L: Latent>(bencher: Bencher) {
    let config = DeltaLookbackConfig {
      window_n_log: 8,
      state_n_log: 0,
    };
    let lookbacks = vec![N_SUBSEQS; BENCH_N];
    let latents = uniform_latents::<L>(L::BITS);
    let (mut window_buffer, mut pos) = new_window_buffer_and_pos::<L>(config, &[]);

    bencher
      .counter(divan::counter::ItemsCount::new(BENCH_N))
      .bench_local(|| {
        for (batch_idx, batch) in latents.chunks(FULL_BATCH_N).enumerate() {
          let start = batch_idx * FULL_BATCH_N;
          super::decode(
            config,
            &lookbacks[start..start + batch.len()],
            &mut pos,
            &mut window_buffer,
            batch,
          );
        }
      });
  }
}
