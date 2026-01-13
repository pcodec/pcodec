use std::cmp::{max, min};
use std::mem;

use crate::compression_intermediates::Bid;
use crate::constants::{Bitlen, MULT_REQUIRED_BITS_SAVED_PER_NUM};
use crate::data_types::SplitLatents;
use crate::data_types::{Float, Latent};
use crate::dyn_latent_slice::DynLatentSlice;
use crate::errors::PcoResult;
use crate::metadata::{DynLatents, Mode};
use crate::mode::int_mult;
use crate::sampling;
use crate::sampling::PrimaryLatentAndSavings;

#[inline(never)]
pub(crate) fn join_latents<F: Float>(
  base: F,
  primary: DynLatentSlice,
  secondary: Option<DynLatentSlice>,
  dst: &mut [F],
) -> PcoResult<()> {
  let primary = primary.downcast_unwrap::<F::L>();
  let secondary = secondary.unwrap().downcast_unwrap::<F::L>();
  for ((&mult, &adj), dst) in primary.iter().zip(secondary.iter()).zip(dst.iter_mut()) {
    let unadjusted = F::int_float_from_latent(mult) * base;
    *dst = F::from_latent_ordered(
      unadjusted
        .to_latent_ordered()
        .wrapping_add(adj)
        .toggle_center(),
    );
  }

  Ok(())
}

pub(crate) fn split_latents<F: Float>(page_nums: &[F], config: FloatMultConfig<F>) -> SplitLatents {
  let FloatMultConfig { base, inv_base } = config;
  let n = page_nums.len();
  let uninit_vec = || unsafe {
    let mut res = Vec::<F::L>::with_capacity(n);
    res.set_len(n);
    res
  };
  let mut primary = uninit_vec();
  let mut adjustments = uninit_vec();
  for (&num, (primary_dst, adj_dst)) in page_nums
    .iter()
    .zip(primary.iter_mut().zip(adjustments.iter_mut()))
  {
    let mult = (num * inv_base).round();
    *primary_dst = F::int_float_to_latent(mult);
    *adj_dst = num
      .to_latent_ordered()
      .wrapping_sub((mult * base).to_latent_ordered())
      // ULP adjustments are naturally signed quantities, so we toggle them so
      // that 0 is in the middle of the range
      .toggle_center();
  }

  SplitLatents {
    primary: DynLatents::new(primary),
    secondary: Some(DynLatents::new(adjustments)),
  }
}

// The rest of this file concerns automatically detecting the float `base`
// such that `x = mult * base + adj * ULP` usefully splits a delta `x` into
// latent variables `mult` and `adj` (if such a `base` exists).
//
// Somewhat different from int mult, we simplistically model that each `x` is
// a multiple of `base` with floating point errors; we would identify `base`
// for the numbers e, 2e, 3e; but if we add 1 to all the numbers, even
// though `base=e` would be just as useful in either case.
// As a result, we can think of the "loss" of an error from a multiple of base
// as O(ln|error|).
//
// I (Martin) thought about using an FFT here, but I'm not sure how to pull it
// off computationally efficiently when the frequency of interest could be in
// such a large range and must be determined so precisely.
// So instead we use an approximate Euclidean algorithm on pairs of floats.

const REQUIRED_PRECISION_BITS: Bitlen = 6;
const SNAP_THRESHOLD_ABSOLUTE: f64 = 0.02;
const SNAP_THRESHOLD_DECIMAL_RELATIVE: f64 = 0.01;
const INTERESTING_TRAILING_ZEROS: u32 = 5;
const REQUIRED_TRAILING_ZEROS_FREQUENCY: f64 = 0.5;
const REQUIRED_GCD_PAIR_FREQUENCY: f64 = 0.001;

fn insignificant_float_to<F: Float>(x: F) -> F {
  let spare_precision_bits = F::PRECISION_BITS.saturating_sub(REQUIRED_PRECISION_BITS) as i32;
  x * F::exp2(-spare_precision_bits)
}

fn is_approx_zero<F: Float>(small: F, big: F) -> bool {
  small <= insignificant_float_to(big)
}

fn is_small_remainder<F: Float>(remainder: F, original: F) -> bool {
  remainder <= original * F::exp2(-16)
}

fn is_imprecise<F: Float>(value: F, err: F) -> bool {
  value <= err * F::exp2(REQUIRED_PRECISION_BITS as i32)
}

fn approx_pair_gcd<F: Float>(greater: F, lesser: F) -> Option<F> {
  if is_approx_zero(lesser, greater) || lesser == greater {
    return None;
  }

  #[derive(Clone, Copy, Debug)]
  struct PairMult<F: Float> {
    value: F,
    err: F,
  }

  let machine_eps = F::exp2(-(F::PRECISION_BITS as i32));
  let rem_assign = |lhs: &mut PairMult<F>, rhs: &PairMult<F>| {
    let ratio = (lhs.value / rhs.value).round();
    lhs.err += ratio * rhs.err + lhs.value * machine_eps;
    lhs.value = (lhs.value - ratio * rhs.value).abs();
  };

  let mut p_greater = PairMult {
    value: greater,
    err: F::ZERO,
  };
  let mut p_lesser = PairMult {
    value: lesser,
    err: F::ZERO,
  };

  loop {
    let prev = p_greater.value;
    rem_assign(&mut p_greater, &p_lesser);
    if is_small_remainder(p_greater.value, prev) || p_greater.value <= p_greater.err {
      return Some(p_lesser.value);
    }

    if is_approx_zero(p_greater.value, greater) || is_imprecise(p_greater.value, p_greater.err) {
      return None;
    }

    mem::swap(&mut p_greater, &mut p_lesser);
  }
}

#[inline(never)]
fn choose_config_by_trailing_zeros<F: Float>(sample: &[F]) -> Option<FloatMultConfig<F>> {
  let precision_bits = F::PRECISION_BITS;
  let calc_power_of_2_divisor =
    |exponent, trailing_zeros| exponent - precision_bits.saturating_sub(trailing_zeros) as i32;

  // the greatest k such that 2^k divides all the floats exactly
  let mut k = i32::MAX;
  let mut count = 0;
  for x in sample {
    let trailing_zeros = x.trailing_zeros();
    if *x != F::ZERO && trailing_zeros >= INTERESTING_TRAILING_ZEROS {
      let k_prime = calc_power_of_2_divisor(x.exponent(), trailing_zeros);
      count += 1;
      k = min(k, k_prime);
    }
  }

  let required_samples = max(
    (sample.len() as f64 * REQUIRED_TRAILING_ZEROS_FREQUENCY).ceil() as usize,
    sampling::MIN_SAMPLE,
  );
  if count < required_samples {
    return None;
  }

  let mut int_sample = Vec::new();
  let lshift = F::L::BITS - precision_bits - 1;
  let explicit_mantissa = F::L::MID;
  for x in sample {
    let exponent = x.exponent();
    // the greatest k' such that 2^k' divides this float exactly
    let k_prime = calc_power_of_2_divisor(x.exponent(), x.trailing_zeros());
    if k_prime >= k && exponent < k + F::L::BITS as i32 {
      let rshift = F::L::BITS - 1 - (exponent - k) as u32;
      let lshifted_w_explicit_mantissa = (x.to_latent_bits() << lshift) | explicit_mantissa;
      let multiple_of_k = lshifted_w_explicit_mantissa >> rshift;
      int_sample.push(multiple_of_k);
    }
  }

  if int_sample.len() >= required_samples {
    let int_base = int_mult::choose_candidate_base(&mut int_sample)
      .map(|(base, _)| base)
      .unwrap_or(F::L::ONE);
    let base = F::from_latent_numerical(int_base) * F::exp2(k);
    Some(FloatMultConfig::from_base(base))
  } else {
    None
  }
}

#[inline(never)]
fn approx_sample_gcd_euclidean<F: Float>(sample: &[F]) -> Option<F> {
  let mut gcds = Vec::new();
  for i in (0..sample.len() - 1).step_by(2) {
    let a = sample[i];
    let b = sample[i + 1];
    if let Some(gcd) = approx_pair_gcd(F::max(a, b), F::min(a, b)) {
      gcds.push(gcd);
    }
  }

  let required_pairs_with_common_gcd =
    1 + (sample.len() as f64 * REQUIRED_GCD_PAIR_FREQUENCY).ceil() as usize;
  if gcds.len() < required_pairs_with_common_gcd {
    return None;
  }

  // safe because we filtered out poorly-behaved floats
  gcds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
  // we check a few GCDs in the middle and see if they show up frequently enough
  for percentile in [0.1, 0.3, 0.5] {
    let candidate = gcds[(percentile * gcds.len() as f64) as usize];
    let similar_gcd_count = gcds
      .iter()
      .filter(|&&gcd| (gcd - candidate).abs() < F::from_f64(0.01) * candidate)
      .count();

    if similar_gcd_count >= required_pairs_with_common_gcd {
      return Some(candidate);
    }
  }

  None
}

fn choose_config_by_euclidean<F: Float>(sample: &[F]) -> Option<FloatMultConfig<F>> {
  let base = approx_sample_gcd_euclidean(sample)?;
  let base = center_sample_base(base, sample);
  let config = snap_to_int_reciprocal(base);
  Some(config)
}

#[inline(never)]
fn center_sample_base<F: Float>(base: F, sample: &[F]) -> F {
  // Go back through the sample, holding all mults fixed, and adjust the gcd to
  // minimize the average deviation from mult * gcd, weighting by mult.
  // Ideally we would tweak by something between the weighted median and mode
  // of the individual tweaks, since we model loss as proportional to
  // sum[log|error|], but doing so would be computationally harder.
  let inv_base = base.inv();
  let mut tweak_sum = F::ZERO;
  let mut tweak_weight = F::ZERO;
  for &x in sample {
    let mult = (x * inv_base).round();
    let mult_exponent = mult.exponent() as Bitlen;
    if mult_exponent < F::PRECISION_BITS && mult != F::ZERO {
      let overshoot = (mult * base) - x;
      let weight = F::from_f64((F::PRECISION_BITS - mult_exponent) as f64);
      tweak_sum += weight * (overshoot / mult);
      tweak_weight += weight;
    }
  }
  base - tweak_sum / tweak_weight
}

fn snap_to_int_reciprocal<F: Float>(base: F) -> FloatMultConfig<F> {
  let inv_base = base.inv();
  let round_inv_base = inv_base.round();
  let decimal_inv_base = F::from_f64(10.0_f64.powf(inv_base.to_f64().log10().round()));
  // check if relative error is below a threshold
  if (inv_base - round_inv_base).abs() < F::from_f64(SNAP_THRESHOLD_ABSOLUTE) {
    FloatMultConfig::from_inv_base(round_inv_base)
  } else if (inv_base - decimal_inv_base).abs() / inv_base
    < F::from_f64(SNAP_THRESHOLD_DECIMAL_RELATIVE)
  {
    FloatMultConfig::from_inv_base(decimal_inv_base)
  } else {
    FloatMultConfig::from_base(base)
  }
}

fn bits_saved_per_num_over_classic<F: Float>(
  config: FloatMultConfig<F>,
  sample: &[F],
) -> Option<f64> {
  let bits_saved_per_num = sampling::est_bits_saved_per_num(sample, |x| {
    let mult = (x * config.inv_base).round();
    let primary = mult.int_float_to_latent();
    // We treat a mult of 0 as if there are only PRECISION_BITS bits between it and 1,
    // which is not true (there are actually BITS - 2), but this is more useful
    // for estimating bit savings.
    let inter_base_bits = F::PRECISION_BITS.saturating_sub(mult.exponent() as Bitlen);
    let approx_unsigned = (mult * config.base).to_latent_ordered();
    let x_as_unsigned = x.to_latent_ordered();
    let abs_adj = max(x_as_unsigned, approx_unsigned) - min(x_as_unsigned, approx_unsigned);
    // Estimating bit cost of adjustments, we conservatively assume the
    // probability distribution is
    //
    // P(adj) = 1/(2 * (adj^2 + 1)),
    //   -2^k < adj < 2^k where k = inter_base_bits - 1
    //
    // So adj should cost the log_2 of this, or approximately
    //
    // B(adj) = 1 + 2 * log_2|adj|
    //
    // We relax this slightly for 0 and let it slide.
    let adj_bits = 1 + 2 * (F::L::BITS - abs_adj.leading_zeros());
    PrimaryLatentAndSavings {
      primary,
      bits_saved: inter_base_bits as f64 - adj_bits as f64,
    }
  });

  if bits_saved_per_num >= MULT_REQUIRED_BITS_SAVED_PER_NUM {
    Some(bits_saved_per_num)
  } else {
    None
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FloatMultConfig<F: Float> {
  pub base: F,
  pub inv_base: F,
}

impl<F: Float> FloatMultConfig<F> {
  fn from_base(base: F) -> Self {
    Self {
      base,
      inv_base: base.inv(),
    }
  }

  fn from_inv_base(inv_base: F) -> Self {
    Self {
      base: inv_base.inv(),
      inv_base,
    }
  }
}

fn choose_config<F: Float>(sample: &[F]) -> Option<FloatMultConfig<F>> {
  choose_config_by_trailing_zeros(sample).or_else(|| choose_config_by_euclidean(sample))
}

pub(crate) fn compute_bid<F: Float>(sample: &[F]) -> Option<Bid<F>> {
  choose_config(sample).and_then(|config| {
    let bits_saved_per_num = bits_saved_per_num_over_classic(config, sample)?;
    Some(Bid {
      mode: Mode::float_mult(config.base),
      bits_saved_per_num,
      split_fn: Box::new(move |nums| split_latents(nums, config)),
    })
  })
}

#[cfg(test)]
mod test {
  use std::f32::consts::TAU;

  use rand::{Rng, SeedableRng};
  use rand_xoshiro::Xoroshiro128PlusPlus;

  use crate::data_types::Number;

  use super::*;

  fn assert_almost_equal_ulps(a: f32, b: f32, ulps_tolerance: u32, desc: &str) {
    let (a, b) = (a.to_latent_ordered(), b.to_latent_ordered());
    let udiff = max(a, b) - min(a, b);
    assert!(
      udiff <= ulps_tolerance,
      "{} far from {}; {}",
      a,
      b,
      desc,
    );
  }

  fn assert_almost_equal(a: f32, b: f32, abs_tolerance: f32, desc: &str) {
    let diff = (a - b).abs();
    assert!(
      diff <= abs_tolerance,
      "{} far from {}; {}",
      a,
      b,
      desc,
    );
  }

  fn plus_epsilons(a: f32, epsilons: i32) -> f32 {
    f32::from_latent_ordered(a.to_latent_ordered().wrapping_add(epsilons as u32))
  }

  fn better_compression_than_classic<F: Float>(config: FloatMultConfig<F>, sample: &[F]) -> bool {
    bits_saved_per_num_over_classic(config, sample)
      .is_some_and(|bits_saved| bits_saved >= MULT_REQUIRED_BITS_SAVED_PER_NUM)
  }

  #[test]
  fn test_near_zero() {
    assert_eq!(
      insignificant_float_to(1.0_f64),
      1.0 / ((1_u64 << 46) as f64)
    );
    assert_eq!(
      insignificant_float_to(1.0_f32),
      1.0 / ((1_u64 << 17) as f32)
    );
    assert_eq!(
      insignificant_float_to(32.0_f32),
      1.0 / ((1_u64 << 12) as f32)
    );
  }

  #[test]
  fn test_trailing_zeros() {
    assert_eq!(
      choose_config_by_trailing_zeros(&[0.0, 3.0, 6.0, 21.0, f32::exp2(100.0)].repeat(5)).unwrap(),
      FloatMultConfig::from_base(3.0),
    )
  }

  #[test]
  fn test_approx_pair_gcd() {
    assert_eq!(approx_pair_gcd(0.0, 0.0), None);
    assert_eq!(approx_pair_gcd(1.0, 0.0), None);
    assert_eq!(approx_pair_gcd(1.0, 1.0), None);
    assert_eq!(approx_pair_gcd(1.0, 2.0), Some(1.0));
    assert_eq!(approx_pair_gcd(6.0, 3.0), Some(3.0));
    assert_eq!(
      approx_pair_gcd(10.01_f64, 0.009999999999999787_f64),
      Some(0.009999999999999787)
    );
    assert_eq!(approx_pair_gcd(2.0_f32.powi(100), 3.0), None);
    assert_eq!(
      approx_pair_gcd(
        f32::MAX_FOR_SAMPLING,
        f32::MAX_FOR_SAMPLING * 0.6
      ),
      Some(f32::MAX_FOR_SAMPLING * 0.2)
    );
    assert_eq!(
      approx_pair_gcd(f32::MAX_FOR_SAMPLING, 0.0000000000001),
      None
    );
    assert_almost_equal_ulps(
      approx_pair_gcd(1.0 / 3.0, 1.0 / 4.0).unwrap(),
      1.0 / 12.0,
      1,
      "1/3 gcd 1/4",
    );
  }

  #[test]
  fn test_candidate_euclidean() {
    let mut nums = vec![0.0, 2.0_f32.powi(-100), 0.0037, 1.0001].repeat(5);
    nums.push(f32::MAX);
    assert_almost_equal(
      choose_config_by_euclidean(&nums).unwrap().base,
      1.0E-4,
      1.0E-6,
      "10^-4 adverse",
    );
  }

  #[test]
  fn test_gcd_euclidean() {
    let nums = vec![0.0, 2.0_f32.powi(-100), 0.0037, 1.0001, f32::MAX].repeat(5);
    assert_almost_equal(
      approx_sample_gcd_euclidean(&nums).unwrap(),
      1.0E-4,
      1.0E-6,
      "10^-4 adverse",
    );

    let nums = vec![0.0, 2.0_f32.powi(-100), 0.0037, 0.0049, 1.0001, f32::MAX].repeat(5);
    assert_almost_equal(
      approx_sample_gcd_euclidean(&nums).unwrap(),
      1.0E-4,
      1.0E-9,
      "10^-4",
    );

    let mut nums = Vec::new();
    let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
    for _ in 0..25 {
      nums.push(rng.gen_range(0.0..1.0_f32));
    }
    assert_eq!(approx_sample_gcd_euclidean(&nums), None);
  }

  #[test]
  fn test_center_gcd() {
    let nums = vec![6.0 / 7.0 - 1E-4, 16.0 / 7.0 + 1E-4, 18.0 / 7.0 - 1E-4];
    assert_almost_equal(
      center_sample_base(0.28, &nums),
      2.0 / 7.0,
      1E-4,
      "center",
    )
  }

  #[test]
  fn test_snap() {
    assert_eq!(
      snap_to_int_reciprocal(0.01000333),
      FloatMultConfig {
        base: 0.01,
        inv_base: 100.0
      }
    );
    assert_eq!(
      snap_to_int_reciprocal(0.009999666),
      FloatMultConfig {
        base: 0.01,
        inv_base: 100.0
      }
    );
    assert_eq!(
      snap_to_int_reciprocal(0.143),
      FloatMultConfig {
        base: 1.0 / 7.0,
        inv_base: 7.0,
      }
    );
    assert_eq!(
      snap_to_int_reciprocal(0.0105),
      FloatMultConfig {
        base: 0.0105,
        inv_base: 1.0 / 0.0105
      }
    );
    assert_eq!(snap_to_int_reciprocal(TAU).base, TAU);
  }

  #[test]
  fn test_float_mult_better_than_classic() {
    let config = FloatMultConfig::from_inv_base(10.0);
    let nums = vec![
      f32::NEG_INFINITY,
      -f32::NAN,
      -999.0,
      -0.3,
      0.0,
      0.1,
      0.2,
      0.3,
      0.3,
      0.4,
      0.5,
      0.6,
      0.7,
      f32::NAN,
      f32::INFINITY,
    ];
    assert!(better_compression_than_classic(
      config, &nums
    ));

    for n in [10, 1000] {
      let nums = (0..n)
        .map(|x| plus_epsilons((x as f32) * 0.1, x % 2))
        .collect::<Vec<_>>();
      assert!(
        better_compression_than_classic(config, &nums),
        "n={}",
        n
      );
    }
  }

  #[test]
  fn test_float_mult_worse_than_classic() {
    let config = FloatMultConfig::from_inv_base(10.0);
    for n in [10, 1000] {
      let nums = vec![0.1; n];
      assert!(
        !better_compression_than_classic(config, &nums),
        "n={}",
        n
      );

      let nums = (0..n).map(|x| (x as f32 + 1.0) * TAU).collect::<Vec<_>>();
      assert!(
        !better_compression_than_classic(config, &nums),
        "n={}",
        n
      );

      let nums = (0..n)
        // at this magnitude, each increment of `base` is only ~2 bits
        .map(|x| (x + 5_000_000) as f32 * 0.1)
        .collect::<Vec<_>>();
      assert!(
        !better_compression_than_classic(config, &nums),
        "n={}",
        n
      );
    }
  }

  #[test]
  fn test_float_mult_worse_than_classic_zeros() {
    let mut nums = vec![0.0_f32; 1000];
    let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0);
    let config = FloatMultConfig::from_inv_base(1E7);
    for _ in 0..1000 {
      nums.push(rng.gen_range(0.0..1.0));
    }
    assert!(!better_compression_than_classic(
      config, &nums
    ));
  }

  #[test]
  fn test_choose_config_and_bid() {
    let mut sevenths = Vec::new();
    let mut ones = Vec::new();
    let mut noisy_decimals = Vec::new();
    let mut junk = Vec::new();
    for i in 0..1000 {
      sevenths.push(((i % 50) - 20) as f32 * (1.0 / 7.0));
      ones.push(1.0);
      noisy_decimals.push(plus_epsilons(
        0.1 * ((i - 100) as f32),
        -7 + i % 15,
      ));
      junk.push((i as f32).sin());
    }

    let sevenths_sample = &sevenths[..50];

    assert_eq!(
      choose_config(&sevenths),
      Some(FloatMultConfig {
        base: 1.0 / 7.0,
        inv_base: 7.0,
      })
    );
    assert_eq!(
      choose_config(&ones),
      Some(FloatMultConfig {
        base: 1.0,
        inv_base: 1.0,
      })
    );
    assert_eq!(
      choose_config(&noisy_decimals),
      Some(FloatMultConfig {
        base: 1.0 / 10.0,
        inv_base: 10.0,
      })
    );
    // just check this last one terminates
    let mut big_nums = vec![f32::MAX; 10];
    big_nums.resize(20, f32::MAX * 0.6);
    choose_config(&big_nums);

    assert!(compute_bid(sevenths_sample).is_some());
    // not enough distinct mults
    assert!(compute_bid(&ones).is_none());
    assert!(compute_bid(&junk).is_none());
  }
}
