use crate::{Flags, gcd_utils, Prefix};
use crate::bits::{avg_depth_bits, avg_offset_bits};
use crate::data_types::{NumberLike, UnsignedLike};
use crate::prefix::WeightedPrefix;

fn prefix_bit_cost<Diff: UnsignedLike>(
  base_meta_cost: f64,
  lower: Diff,
  upper: Diff,
  weight: usize,
  total_weight: usize,
  gcd: Diff,
) -> f64 {
  let offset_cost = avg_offset_bits(lower, upper, gcd);
  let huffman_cost = avg_depth_bits(weight, total_weight);
  let gcd_cost = if gcd > Diff::ONE {
    Diff::BITS as f64
  } else {
    0.0
  };
  base_meta_cost +
    gcd_cost +
    huffman_cost + // extra meta cost depending on length of huffman code
    (offset_cost + huffman_cost) * weight as f64 // body cost
}

// this is an exact optimal strategy
pub fn optimize_prefixes<T: NumberLike>(
  wprefixes: Vec<WeightedPrefix<T>>,
  flags: &Flags,
  n: usize,
) -> Vec<WeightedPrefix<T>> {
  let mut c = 0;
  let mut cum_weight = Vec::with_capacity(wprefixes.len() + 1);
  cum_weight.push(0);
  for wp in &wprefixes {
    c += wp.weight;
    cum_weight.push(c);
  }
  let gcds = wprefixes.iter()
    .map(|wp| wp.prefix.gcd)
    .collect::<Vec<_>>();
  let total_weight = cum_weight[cum_weight.len() - 1];
  let lower_unsigneds = wprefixes.iter()
    .map(|wp| wp.prefix.lower.to_unsigned())
    .collect::<Vec<_>>();

  let maybe_rep_idx = wprefixes.iter()
    .position(|wp| wp.prefix.run_len_jumpstart.is_some());

  let mut best_costs = Vec::with_capacity(wprefixes.len() + 1);
  let mut best_paths = Vec::with_capacity(wprefixes.len() + 1);
  best_costs.push(0.0);
  best_paths.push(Vec::new());

  let bits_to_encode_count = flags.bits_to_encode_count(n);
  let base_meta_cost = bits_to_encode_count as f64 +
    2.0 * T::PHYSICAL_BITS as f64 + // lower and upper bounds
    flags.bits_to_encode_code_len() as f64 +
    if flags.use_gcds { 1.0 } else { 0.0 } +
    1.0; // bit to say there is no run len jumpstart
  for i in 0..wprefixes.len() {
    let mut best_cost = f64::MAX;
    let mut best_j = usize::MAX;
    let upper = wprefixes[i].prefix.upper.to_unsigned();
    let cum_weight_i = cum_weight[i + 1];
    let start_j = match maybe_rep_idx {
      Some(ind) if ind < i => ind + 1,
      Some(ind) if ind == i => ind,
      _ => 0,
    };
    let mut gcd = gcds[i];
    for j in (start_j..i + 1).rev() {
      let lower = lower_unsigneds[j];
      // reducing GCD's involves taking their pairwise GCD, additionally
      // GCD'd with their modulo offset
      gcd = gcd_utils::pair_gcd(gcds[j], gcd);
      gcd = gcd_utils::pair_gcd(upper - lower, gcd);
      let cost = best_costs[j] + prefix_bit_cost::<T::Unsigned>(
        base_meta_cost,
        lower,
        upper,
        cum_weight_i - cum_weight[j],
        total_weight,
        gcd,
      );
      if cost < best_cost {
        best_cost = cost;
        best_j = j;
      }
    }

    best_costs.push(best_cost);
    let mut best_path = Vec::with_capacity(best_paths[best_j].len() + 1);
    best_path.extend(&best_paths[best_j]);
    best_path.push((best_j, i));
    best_paths.push(best_path);
  }

  let path = best_paths.last().unwrap();
  let mut res = Vec::with_capacity(path.len());
  for &(j, i) in path {
    let mut count = 0;
    let mut gcd = gcds[j];
    let lower = wprefixes[j].prefix.lower;
    for wp in wprefixes.iter().take(i + 1).skip(j) {
      count += wp.prefix.count;
      gcd = gcd_utils::pair_gcd(wp.prefix.gcd, gcd);
      gcd = gcd_utils::pair_gcd(wp.prefix.upper.to_unsigned() - lower.to_unsigned(), gcd);
    }
    let prefix = Prefix {
      count,
      code: Vec::new(),
      lower,
      upper: wprefixes[i].prefix.upper,
      run_len_jumpstart: wprefixes[i].prefix.run_len_jumpstart,
      gcd,
    };
    res.push(WeightedPrefix {
      weight: cum_weight[i + 1] - cum_weight[j],
      prefix,
    })
  }
  res
}

