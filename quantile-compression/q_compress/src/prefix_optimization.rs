use crate::bits::{avg_depth_bits, avg_offset_bits};
use crate::data_types::{NumberLike, UnsignedLike};
use crate::{gcd_utils, run_len_utils, Flags, Prefix};

fn prefix_bit_cost<U: UnsignedLike>(
  base_meta_cost: f64,
  lower: U,
  upper: U,
  count: usize,
  n: usize,
  gcd: U,
) -> f64 {
  let offset_cost = avg_offset_bits(lower, upper, gcd);
  let (weight, jumpstart_cost) = run_len_utils::weight_and_jumpstart_cost(count, n);
  let total_weight = n + weight - count;
  let huffman_cost = avg_depth_bits(weight, total_weight);
  let gcd_cost = if gcd > U::ONE {
    gcd_utils::gcd_bits_required(upper - lower) as f64
  } else {
    0.0
  };
  base_meta_cost +
    gcd_cost + // extra meta cost of storing GCD
    huffman_cost + // extra meta cost of storing Huffman code
    (huffman_cost + jumpstart_cost) * (weight as f64) +
    offset_cost * count as f64
}

// this is an exact optimal strategy
pub fn optimize_prefixes<T: NumberLike>(
  prefixes: Vec<Prefix<T>>,
  flags: &Flags,
  n: usize,
) -> Vec<Prefix<T>> {
  let mut c = 0;
  let mut cum_count = Vec::with_capacity(prefixes.len() + 1);
  cum_count.push(0);
  for p in &prefixes {
    c += p.count;
    cum_count.push(c);
  }
  let gcds = prefixes.iter().map(|p| p.gcd).collect::<Vec<_>>();
  let lower_unsigneds = prefixes
    .iter()
    .map(|p| p.lower.to_unsigned())
    .collect::<Vec<_>>();
  let upper_unsigneds = prefixes
    .iter()
    .map(|p| p.upper.to_unsigned())
    .collect::<Vec<_>>();

  let mut best_costs = Vec::with_capacity(prefixes.len() + 1);
  let mut best_paths = Vec::with_capacity(prefixes.len() + 1);
  best_costs.push(0.0);
  best_paths.push(Vec::new());

  let bits_to_encode_count = flags.bits_to_encode_count(n);
  let base_meta_cost = bits_to_encode_count as f64 +
    2.0 * T::PHYSICAL_BITS as f64 + // lower and upper bounds
    flags.bits_to_encode_code_len() as f64 +
    if flags.use_gcds { 1.0 } else { 0.0 } + // bit to say whether there is GCD or not
    1.0; // bit to say there is no run len jumpstart
         // determine whether we can skip GCD folding to improve performance in some cases
  let fold_gcd = gcd_utils::use_gcd_prefix_optimize(&prefixes, flags);

  for i in 0..prefixes.len() {
    let mut best_cost = f64::MAX;
    let mut best_j = usize::MAX;
    let upper = upper_unsigneds[i];
    let cum_count_i = cum_count[i + 1];
    let mut gcd_acc = None;
    for j in (0..i + 1).rev() {
      let lower = lower_unsigneds[j];
      if fold_gcd {
        gcd_utils::fold_prefix_gcds_left(
          lower,
          upper_unsigneds[j],
          gcds[j],
          upper,
          &mut gcd_acc,
        );
      }
      let cost = best_costs[j]
        + prefix_bit_cost::<T::Unsigned>(
          base_meta_cost,
          lower,
          upper,
          cum_count_i - cum_count[j],
          n,
          gcd_acc.unwrap_or(T::Unsigned::ONE),
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
    let mut gcd_acc = None;
    for (k, p) in prefixes.iter().enumerate().take(i + 1).skip(j).rev() {
      count += p.count;
      if fold_gcd {
        gcd_utils::fold_prefix_gcds_left(
          lower_unsigneds[k],
          upper_unsigneds[k],
          p.gcd,
          upper_unsigneds[i],
          &mut gcd_acc,
        );
      }
    }

    let prefix = Prefix {
      count,
      code: Vec::new(),
      lower: prefixes[j].lower,
      upper: prefixes[i].upper,
      run_len_jumpstart: run_len_utils::run_len_jumpstart(count, n),
      gcd: gcd_acc.unwrap_or(T::Unsigned::ONE),
    };
    res.push(prefix);
  }
  res
}

#[cfg(test)]
mod tests {
  use crate::prefix_optimization::optimize_prefixes;
  use crate::{Flags, Prefix};

  fn basic_flags() -> Flags {
    Flags {
      use_gcds: true,
      use_min_count_encoding: true,
      use_5_bit_code_len: true,
      delta_encoding_order: 0,
      use_wrapped_mode: false,
    }
  }

  fn make_prefix(count: usize, lower: i32, upper: i32, gcd: u32) -> Prefix<i32> {
    Prefix {
      count,
      code: vec![],
      lower,
      upper,
      run_len_jumpstart: None,
      gcd,
    }
  }

  #[test]
  fn test_optimize_trivial_ranges_gcd() {
    let prefs = vec![
      make_prefix(1, 1000, 1000, 1),
      make_prefix(1, 2000, 2000, 1),
      make_prefix(1, 3000, 3000, 1),
      make_prefix(1, 4000, 4000, 1),
    ];
    let res = optimize_prefixes(prefs, &basic_flags(), 4);
    let expected = vec![make_prefix(4, 1000, 4000, 1000)];
    assert_eq!(res, expected);
  }

  #[test]
  fn test_optimize_single_nontrivial_range_gcd() {
    let prefs = vec![
      make_prefix(100, 1000, 2000, 10),
      make_prefix(1, 2100, 2100, 1),
    ];
    let res = optimize_prefixes(prefs, &basic_flags(), 101);
    let expected = vec![make_prefix(101, 1000, 2100, 10)];
    assert_eq!(res, expected);
  }

  #[test]
  fn test_optimize_nontrivial_ranges_gcd() {
    let prefs = vec![
      make_prefix(5, 1000, 1100, 10),
      make_prefix(5, 1105, 1135, 15),
    ];
    let res = optimize_prefixes(prefs, &basic_flags(), 10);
    let expected = vec![make_prefix(10, 1000, 1135, 5)];
    assert_eq!(res, expected);
  }

  #[test]
  fn test_optimize_nontrivial_misaligned_ranges_gcd() {
    let prefs = vec![
      make_prefix(100, 1000, 1100, 10),
      make_prefix(100, 1101, 1201, 10),
    ];
    let res = optimize_prefixes(prefs, &basic_flags(), 200);
    let expected = vec![
      make_prefix(100, 1000, 1100, 10),
      make_prefix(100, 1101, 1201, 10),
    ];
    assert_eq!(res, expected);
  }
}
