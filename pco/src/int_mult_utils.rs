use std::cmp::min;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::mem;

use crate::constants::MULT_REQUIRED_BITS_SAVED_PER_NUM;
use crate::data_types::SplitLatents;
use crate::data_types::{Latent, Number};
use crate::dyn_latent_slice::DynLatentSlice;
use crate::metadata::DynLatents;
use crate::sampling::{self, PrimaryLatentAndSavings};

// riemann zeta function
const ZETA_OF_2: f64 = PI * PI / 6.0;
const LCB_RATIO: f64 = 1.0;

#[inline(never)]
pub fn split_latents<T: Number>(nums: &[T], base: T::L) -> SplitLatents {
  let n = nums.len();
  let mut mults = Vec::with_capacity(n);
  let mut adjs = Vec::with_capacity(n);
  unsafe {
    mults.set_len(n);
    adjs.set_len(n);
  }
  for (&num, (mult_dst, adj_dst)) in nums.iter().zip(mults.iter_mut().zip(adjs.iter_mut())) {
    let u = num.to_latent_ordered();
    // Maybe one day we could do a libdivide approach for these
    *mult_dst = u / base;
    *adj_dst = u % base;
  }

  SplitLatents {
    primary: DynLatents::new(mults).unwrap(),
    secondary: Some(DynLatents::new(adjs).unwrap()),
  }
}

#[inline(never)]
pub(crate) fn join_latents<T: Number>(
  base: T::L,
  primary: DynLatentSlice,
  secondary: Option<DynLatentSlice>,
  dst: &mut [T],
) {
  let primary = primary.downcast::<T::L>().unwrap();
  let secondary = secondary.unwrap().downcast::<T::L>().unwrap();
  for ((&mult, &adj), dst) in primary.iter().zip(secondary.iter()).zip(dst.iter_mut()) {
    *dst = T::from_latent_ordered((mult * base).wrapping_add(adj));
  }
}

fn calc_gcd<L: Latent>(mut x: L, mut y: L) -> L {
  if x == L::ZERO {
    return y;
  }

  loop {
    if y == L::ZERO {
      return x;
    }

    x %= y;
    mem::swap(&mut x, &mut y);
  }
}

fn solve_root_by_false_position<F: Fn(f64) -> f64>(f: F, mut lb: f64, mut ub: f64) -> Option<f64> {
  const X_TOLERANCE: f64 = 1E-4;
  let mut flb = f(lb);
  let mut fub = f(ub);
  if flb > 0.0 || fub < 0.0 {
    return None;
  }

  while ub - lb > X_TOLERANCE && fub - flb > 0.0 {
    // Pure false position is a bit unsafe, since we haven't ruled out the
    // possibility that f(lb) == f(ub).
    // Instead we squeeze just slightly toward bisection.
    let lb_prop = 0.001 + 0.998 * fub / (fub - flb);
    let mid = lb_prop * lb + (1.0 - lb_prop) * ub;
    let fmid = f(mid);
    if fmid < 0.0 {
      lb = mid;
      flb = fmid;
    } else {
      ub = mid;
      fub = fmid;
    }
  }

  Some((lb + ub) / 2.0)
}

fn calc_triple_gcd<L: Latent>(triple: &[L]) -> L {
  let mut a = triple[0];
  let mut b = triple[1];
  let mut c = triple[2];
  // sort a, b, c
  if a > b {
    mem::swap(&mut a, &mut b);
  }
  if b > c {
    mem::swap(&mut b, &mut c);
  }
  if a > b {
    mem::swap(&mut a, &mut b);
  }

  calc_gcd(b - a, c - a)
}

fn single_category_entropy(p: f64) -> f64 {
  if p == 0.0 || p == 1.0 {
    0.0
  } else {
    -p * p.log2()
  }
}

pub(crate) fn worse_case_categorical_entropy(concentrated_p: f64, n_categories_m1: f64) -> f64 {
  single_category_entropy(concentrated_p)
    + n_categories_m1 * single_category_entropy((1.0 - concentrated_p) / n_categories_m1)
}

fn filter_score_triple_gcd(gcd: f64, triples_w_gcd: usize, total_triples: usize) -> Option<f64> {
  let triples_w_gcd = triples_w_gcd as f64;
  let total_triples = total_triples as f64;
  let prob_per_triple = triples_w_gcd / total_triples;

  // 1. Check if the GCD has statistical evidence.
  // We make the null hypothesis that "naturally" the numbers have a uniform
  // distribution modulo the GCD.
  // If so, subtracting out the low element from the triple still leaves us
  // with a uniform distribution over the other 2, so the probability of
  // observing this GCD exactly (and not a multiple of it) would be
  // 1/(gcd^2) / (1 + 1/2^2 + 1/3^2 + ...) = 1/(zeta(2) * gcd^2)
  // per triple.
  let natural_prob_per_triple = 1.0 / (ZETA_OF_2 * gcd * gcd);
  let stdev = (natural_prob_per_triple * (1.0 - natural_prob_per_triple) / total_triples).sqrt();
  // simple frequentist z test
  let z_score = (prob_per_triple - natural_prob_per_triple) / stdev;
  if z_score < 3.0 {
    return None;
  }

  // 2. Make a conservative estimate (Lower Confidence Bound) for the number of
  // triples that have congruence modulo this GCD.
  // Again, we correct by a factor of zeta(2) because (assuming there is a
  // certain distribution modulo GCD), there should be a 1/4 + 1/9 + 1/16 + ...
  // chance of having observed a multiple of this GCD instead.
  let triples_w_gcd_lcb = triples_w_gcd - LCB_RATIO * triples_w_gcd.sqrt();
  if triples_w_gcd_lcb <= 0.0 {
    return None;
  }
  let congruence_prob_per_triple_lcb = (ZETA_OF_2 * triples_w_gcd_lcb / total_triples).min(1.0);

  // 3. Measure and score by the number of bits saved by using this modulus,
  // as opposed to just a uniform distribution over [0, GCD).
  // If the distribution modulo the GCD has entropy H, this is
  // log_2(GCD) - H.
  // To calculate H, we consider the worst case: the probability is concentrated
  // in one particular value in [0, GCD), and the rest of the probability is
  // distributed uniformly.
  // This maximizes H subject to
  // \sum_{k\in [0, GCD)} P(k)^3 = our observed probability of triple congruence
  // You can use intuition or Lagrange multipliers to verify that.
  let gcd_m1 = gcd - 1.0;
  let gcd_m1_inv_sq = 1.0 / (gcd_m1 * gcd_m1);
  // This is the summation described above: one concentrated p value for a
  // single k, and (GCD - 1) dispersed probabilities of (1 - p) / (GCD - 1)
  let f = |p: f64| p.powi(3) + (1.0 - p).powi(3) * gcd_m1_inv_sq - congruence_prob_per_triple_lcb;
  // It's easy to show that f minimizes at 1/GCD, and that f>0 when
  // p>cbrt(congruence_prob_per_triple_lcb).
  // So if a root in [0, 1] exists at all, it has these lower and upper bounds.
  let lb = 1.0 / gcd;
  // + EPSILON because for large GCDs truncation error can make f(cbrt()) < 0.
  let ub = congruence_prob_per_triple_lcb.cbrt() + f64::EPSILON;
  // You might think
  // * We should apply the cubic formula directly! But no, it's horribly
  //   numerically unstable, hard to choose the root you want, and hard to
  //   avoid every possible NaN or inf.
  // * We should use Newton's method! This also has some tricky NaN,
  //   infinite loop, and divergent cases, and even if you get past those, it
  //   converges annoyingly slowly in some cases.
  // So instead we use the method of false position.
  let concentrated_p = solve_root_by_false_position(f, lb, ub)?;
  let worst_case_entropy_mod_gcd = worse_case_categorical_entropy(concentrated_p, gcd_m1);
  let worst_case_bits_saved = gcd.log2() - worst_case_entropy_mod_gcd;
  if worst_case_bits_saved < MULT_REQUIRED_BITS_SAVED_PER_NUM {
    return None;
  }

  Some(worst_case_bits_saved)
}

fn most_prominent_gcd<L: Latent>(triple_gcds: &[L], total_triples: usize) -> Option<(L, f64)> {
  let mut counts = HashMap::new();
  for &gcd in triple_gcds {
    *counts.entry(gcd).or_insert(0) += 1;
  }

  let gcd_and_score = counts //counts_accounting_for_small_multiples
    .iter()
    .filter_map(|(&gcd, &count)| {
      let gcd_f64 = min(gcd, L::from_u64(u64::MAX)).to_u64() as f64;
      let score = filter_score_triple_gcd(gcd_f64, count, total_triples)?;
      Some((gcd, score))
    })
    .max_by_key(|(_, score)| score.to_latent_ordered())?;

  Some(gcd_and_score)
}

pub fn choose_candidate_base<L: Latent>(sample: &mut [L]) -> Option<(L, f64)> {
  let triple_gcds = sample
    .chunks_exact(3)
    .map(calc_triple_gcd)
    .filter(|&gcd| gcd > L::ONE)
    .collect::<Vec<_>>();

  most_prominent_gcd(&triple_gcds, sample.len() / 3)
}

pub fn choose_base<T: Number>(nums: &[T]) -> Option<T::L> {
  let mut sample = sampling::choose_sample(nums, |num| Some(num.to_latent_ordered()))?;
  let (candidate, bits_saved_per_adj) = choose_candidate_base(&mut sample)?;

  if sampling::est_bits_saved_per_num(&sample, |x| PrimaryLatentAndSavings {
    primary: x / candidate,
    bits_saved: bits_saved_per_adj,
  }) > MULT_REQUIRED_BITS_SAVED_PER_NUM
  {
    Some(candidate)
  } else {
    None
  }
}

#[cfg(test)]
mod tests {
  use rand::Rng;
  use rand_xoshiro::rand_core::SeedableRng;

  use super::*;

  #[test]
  fn test_false_position() {
    fn assert_close(x: f64, y: f64) {
      assert!((x - y).abs() < 1E-4);
    }

    let x0 = solve_root_by_false_position(|x| x * x - 1.0, -0.9, 2.0).unwrap();
    assert_close(x0, 1.0);

    assert!(solve_root_by_false_position(|x| x * x, -0.9, 2.0).is_none());

    assert_eq!(
      solve_root_by_false_position(|_| 0.0, 0.0, 1.0),
      Some(0.5)
    );
  }

  #[test]
  fn test_split_join_latents() {
    // SPLIT
    let nums = vec![8_u32, 1, 5];
    let base = 4_u32;
    let latents = split_latents(&nums, base);
    let mut primary = latents.primary.downcast::<u32>().unwrap();
    let mut secondary = latents.secondary.unwrap().downcast::<u32>().unwrap();
    let mut dst = vec![0; nums.len()];
    assert_eq!(&primary, &vec![2_u32, 0, 1]);
    assert_eq!(&secondary, &vec![0_u32, 1, 1]);

    // JOIN
    join_latents(
      base,
      DynLatentSlice::U32(&mut primary),
      Some(DynLatentSlice::U32(&mut secondary)),
      &mut dst,
    );

    assert_eq!(dst, nums);
  }

  #[test]
  fn test_calc_gcd() {
    assert_eq!(calc_gcd(0_u32, 0), 0);
    assert_eq!(calc_gcd(0_u32, 1), 1);
    assert_eq!(calc_gcd(1_u32, 0), 1);
    assert_eq!(calc_gcd(2_u32, 0), 2);
    assert_eq!(calc_gcd(2_u32, 3), 1);
    assert_eq!(calc_gcd(6_u32, 3), 3);
    assert_eq!(calc_gcd(12_u32, 30), 6);
  }

  #[test]
  fn test_calc_triple_gcd() {
    assert_eq!(calc_triple_gcd(&[1_u32, 5, 9]), 4);
    assert_eq!(calc_triple_gcd(&[8_u32, 5, 2]), 3);
    assert_eq!(calc_triple_gcd(&[3_u32, 3, 3]), 0);
    assert_eq!(calc_triple_gcd(&[5_u32, 0, 10]), 5);
  }

  #[test]
  fn test_calc_candidate_gcd() {
    // not significant enough
    assert_eq!(
      choose_candidate_base(&mut [0_u32, 4, 8]),
      None,
    );
    assert_eq!(
      choose_candidate_base(&mut [0_u32, 4, 8, 10, 14, 18, 20, 24, 28])
        .unwrap()
        .0,
      4,
    );
    // 2 out of 3 triples have a rare congruency
    assert_eq!(
      choose_candidate_base(&mut [1_u32, 11, 21, 31, 41, 51, 61, 71, 82])
        .unwrap()
        .0,
      10,
    );
    // 1 out of 3 triples has a rare congruency
    assert_eq!(
      choose_candidate_base(&mut [1_u32, 11, 22, 31, 41, 51, 61, 71, 82]),
      None,
    );
    // even just evens can be useful if the signal is strong enough
    let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0);
    let mut twos = (0_u32..200)
      .map(|_| rng.gen_range(0_u32..1000) * 2)
      .collect::<Vec<_>>();
    assert_eq!(
      choose_candidate_base(&mut twos).unwrap().0,
      2
    );
  }
}
