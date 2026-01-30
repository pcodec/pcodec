use crate::compression_intermediates::Bid;
use crate::constants::{Bitlen, QUANT_REQUIRED_BITS_SAVED_PER_NUM};
use crate::data_types::float::Float;
use crate::data_types::latent_priv::LatentPriv;
use crate::data_types::SplitLatents;
use crate::dyn_slices::DynLatentSlice;
use crate::errors::PcoResult;
use crate::metadata::{DynLatents, Mode};
use crate::sampling::{self, PrimaryLatentAndSavings};
use std::cmp;

#[inline(never)]
pub(crate) fn join_latents<F: Float>(
  k: Bitlen,
  primary: DynLatentSlice,
  secondary: Option<DynLatentSlice>,
  dst: &mut [F],
) -> PcoResult<()> {
  let primary = primary.downcast::<F::L>().unwrap();
  let secondary = secondary.unwrap().downcast::<F::L>().unwrap();
  // For any float `num` such that `split_latents([num], k) == [[y], [m]]`, we have
  //     num.is_sign_positive() == (y >= sign_cutoff)
  let sign_cutoff = F::L::MID >> k;
  let lowest_k_bits_max = (F::L::ONE << k) - F::L::ONE;
  for ((&y, &m), dst) in primary.iter().zip(secondary.iter()).zip(dst.iter_mut()) {
    debug_assert!(
      m >> k == F::L::ZERO,
      "Invalid input to FloatQuant: m must be a k-bit integer"
    );
    let is_pos_as_float = y >= sign_cutoff;
    let lowest_k_bits = if is_pos_as_float {
      m
    } else {
      lowest_k_bits_max - m
    };
    *dst = F::from_latent_ordered((y << k) + lowest_k_bits);
  }

  Ok(())
}

pub(crate) fn split_latents<F: Float>(page_nums: &[F], k: Bitlen) -> SplitLatents {
  let n = page_nums.len();
  let uninit_vec = || unsafe {
    let mut res = Vec::<F::L>::with_capacity(n);
    res.set_len(n);
    res
  };
  let mut primary = uninit_vec();
  let mut secondary = uninit_vec();
  let lowest_k_bits_max = (F::L::ONE << k) - F::L::ONE;
  for (&num, (primary_dst, secondary_dst)) in page_nums
    .iter()
    .zip(primary.iter_mut().zip(secondary.iter_mut()))
  {
    let num_ = num.to_latent_ordered();
    *primary_dst = num_ >> k;
    let lowest_k_bits = num_ & lowest_k_bits_max;
    // Motivation for the sign-dependent logic below:
    // In the common case where `num` is exactly quantized, we want `*secondary_dst` to always be
    // zero.  But when `num` is negative, `lowest_k_bits == lowest_k_bits_max`.  So we manually
    // flip it here, and un-flip it in `join_latents`.
    *secondary_dst = if num.is_sign_positive_() {
      lowest_k_bits
    } else {
      lowest_k_bits_max - lowest_k_bits
    };
  }

  SplitLatents {
    primary: DynLatents::new(primary),
    secondary: Some(DynLatents::new(secondary)),
  }
}

pub(crate) fn compute_bid<F: Float>(sample: &[F]) -> Option<Bid<F>> {
  let (k, bits_saved_per_infrequent_primary) = estimate_best_k_and_bits_saved(sample);
  let bits_saved_per_num = sampling::est_bits_saved_per_num(sample, |x| {
    let primary = x.to_latent_bits() >> k;
    PrimaryLatentAndSavings {
      primary,
      bits_saved: bits_saved_per_infrequent_primary,
    }
  });
  if bits_saved_per_num > QUANT_REQUIRED_BITS_SAVED_PER_NUM {
    Some(Bid {
      mode: Mode::FloatQuant(k),
      bits_saved_per_num,
      split_fn: Box::new(move |nums| split_latents(nums, k)),
    })
  } else {
    None
  }
}

fn estimate_best_k_and_bits_saved_from_hist(
  cumulative_hist: &[u32],
  sample_len: usize,
) -> (Bitlen, f64) {
  let sample_len = sample_len as f64;
  let mut best_k = 0;
  let mut best_bits_saved = 0.0;

  // There may be multiple local maxima in the function from
  // k -> bits_saved_per_infrequent_primary.
  // We just want the first one, since later ones are likely to make the
  // distribution of (x >> k) degenerate and easily-memorizable.
  for (k, &occurrences) in cumulative_hist.iter().enumerate().skip(1) {
    // Here we borrow the worst case bits saved approach from int mult utils,
    // taking a lower confidence bound estimate for the number of occurrences.
    // And then we consider the worst case, where the probability distribution
    // of adjustments has a spike at 0 and is uniform elsewhere.
    if occurrences == 0 {
      continue;
    }

    let occurrences = occurrences as f64;
    let freq = occurrences / sample_len;
    let n_categories = (1_u64 << k) - 1;
    let worst_case_bits_per_infrequent_primary =
      super::worst_case_categorical_entropy(freq, n_categories as f64);
    let bits_saved_per_infrequent_primary = k as f64 - worst_case_bits_per_infrequent_primary;
    if bits_saved_per_infrequent_primary > best_bits_saved {
      best_k = k as Bitlen;
      best_bits_saved = bits_saved_per_infrequent_primary;
    } else {
      break;
    }
  }

  (best_k, best_bits_saved)
}

pub(crate) fn estimate_best_k_and_bits_saved<F: Float>(sample: &[F]) -> (Bitlen, f64) {
  let mut hist = vec![0; (F::PRECISION_BITS + 1) as usize];
  for x in sample {
    // Using the fact that significand bits come last in
    // the floating-point representations we care about
    let trailing_mantissa_zeros = cmp::min(F::PRECISION_BITS, x.trailing_zeros());
    hist[trailing_mantissa_zeros as usize] += 1
  }

  // turn it into a cumulative histogram from k -> # of samples with *at least*
  // k trailing bits
  let mut rev_csum = 0;
  for x in hist.iter_mut().rev() {
    rev_csum += *x;
    *x = rev_csum;
  }

  estimate_best_k_and_bits_saved_from_hist(&hist, sample.len())
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_estimate_best_k() {
    // all but the last of these have 21 out of 23 mantissa bits zeroed
    let mut sample = vec![1.0_f32, 1.25, -1.5, 1.75, -0.875, 0.75, 0.625].repeat(3);
    sample.push(f32::from_bits(1.0_f32.to_bits() + 1));
    let (k, bits_saved) = estimate_best_k_and_bits_saved(&sample);
    assert_eq!(k, 21);
    assert!(bits_saved < 21.0);
    assert!(bits_saved > 10.0);
  }

  #[test]
  fn test_estimate_best_k_full_precision() {
    // all elements have all 52 mantissa bits zeroed
    let sample = vec![1.0_f64; 20];
    let (k, bits_saved) = estimate_best_k_and_bits_saved(&sample);
    assert_eq!(k, 52);
    assert_eq!(bits_saved, 52.0);
  }

  #[test]
  fn test_split_latents_specific_values() {
    let expected = vec![
      (
        -f32::INFINITY,
        (0b00000000000000111111111111111111, 0b00000),
      ),
      (
        -1.0f32 - f32::EPSILON,
        (0b00000010000000111111111111111111, 0b00001),
      ),
      (
        -1.0f32,
        (0b00000010000000111111111111111111, 0b00000),
      ),
      (
        -0.0f32,
        (0b00000011111111111111111111111111, 0b00000),
      ),
      (
        0.0f32,
        (0b00000100000000000000000000000000, 0b00000),
      ),
      (
        1.0f32,
        (0b00000101111111000000000000000000, 0b00000),
      ),
      (
        1.0f32 + f32::EPSILON,
        (0b00000101111111000000000000000000, 0b00001),
      ),
      (
        f32::INFINITY,
        (0b00000111111111000000000000000000, 0b00000),
      ),
    ];
    let (nums, (_expected_ys, _expected_ms)): (Vec<_>, (Vec<_>, Vec<_>)) =
      expected.iter().cloned().unzip();
    let k: Bitlen = 5;
    let SplitLatents { primary, secondary } = split_latents(&nums, k);
    let primary = primary.downcast::<u32>().unwrap();
    let secondary = secondary.unwrap().downcast::<u32>().unwrap();
    let actual: Vec<_> = nums
      .iter()
      .cloned()
      .zip(primary.iter().cloned().zip(secondary.iter().cloned()))
      .collect();
    assert_eq!(expected, actual);
  }

  #[test]
  fn test_secondary_is_zero_for_exact_quantized() {
    let k: Bitlen = f64::MANTISSA_DIGITS - f32::MANTISSA_DIGITS;
    let nums: Vec<f64> = [-2.345f32, -1.234f32, -0.0f32, 0.0f32, 1.234f32, 2.345f32]
      .iter()
      .map(|&num| num as f64)
      .collect();
    let SplitLatents {
      primary: _primary,
      secondary,
    } = split_latents(&nums, k);
    let secondary = secondary.unwrap().downcast::<u64>().unwrap();
    assert!(secondary.iter().all(|&m| m == 0u64));
  }

  #[test]
  fn test_join_split_round_trip() {
    let nums = vec![1.234, -9999.999, f64::NAN, -f64::INFINITY];

    let k: Bitlen = 5;
    let SplitLatents { primary, secondary } = split_latents(&nums, k);
    let mut primary = primary.downcast::<u64>().unwrap();
    let mut secondary = secondary.unwrap().downcast::<u64>().unwrap();
    let mut dst = vec![0.0; nums.len()];
    join_latents::<f64>(
      k,
      DynLatentSlice::U64(&mut primary),
      Some(DynLatentSlice::U64(&mut secondary)),
      &mut dst,
    )
    .unwrap();

    assert_eq!(dst.len(), nums.len());
    for (a, b) in dst.iter().zip(&nums) {
      assert_eq!(a.to_bits(), b.to_bits());
    }
  }

  #[test]
  fn test_compute_bid() {
    // the larger numbers in this sample have 23 - 6 = 17 bits of quantization
    let sample = (0..100).map(|x| x as f32).collect::<Vec<_>>();
    let bid = compute_bid(&sample).unwrap();
    assert!(matches!(bid.mode, Mode::FloatQuant(17)));
    assert_eq!(bid.bits_saved_per_num, 17.0);

    // same as above, except not all perfectly quantized
    let mut sample = (0..100).map(|x| x as f32).collect::<Vec<_>>();
    sample[0] += 0.1;
    sample[37] -= 0.1;
    let bid = compute_bid(&sample).unwrap();
    assert!(matches!(bid.mode, Mode::FloatQuant(17)));
    assert!(bid.bits_saved_per_num < 17.0);
    assert!(bid.bits_saved_per_num > 15.0);

    // the primary latent in this dataset has too few values and would be easily memorizable
    let sample = [0.0_f32, 1.0].repeat(50);
    let bid = compute_bid(&sample);
    assert!(bid.is_none());
  }
}
