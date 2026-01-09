use crate::constants::Bitlen;
use crate::data_types::{Latent, Signed};
use crate::metadata::DeltaConv1Config;
use crate::{delta, sort_utils};

// For now, using f32 ruins the precision of xtx and xty accumulations
type Real = f64;

const ENCODE_BATCH_SIZE: usize = 512;

// poor man's nalgebra so we don't need a whole new dep
#[derive(Clone, Debug)]
struct Matrix {
  data: Vec<Real>,
  h: usize,
  w: usize,
}

fn safe_sqrt(x: Real) -> Real {
  x.max(0.0).sqrt()
}

impl Matrix {
  fn constant(value: Real, h: usize, w: usize) -> Self {
    Self {
      data: vec![value; h * w],
      h,
      w,
    }
  }

  #[inline]
  fn physical_idx(&self, i: usize, j: usize) -> usize {
    // column-major is more efficient for our use cases
    i + j * self.h
  }

  #[inline]
  unsafe fn set(&mut self, i: usize, j: usize, value: Real) {
    let idx = self.physical_idx(i, j);
    *self.data.get_unchecked_mut(idx) = value;
  }

  #[inline]
  unsafe fn get(&self, i: usize, j: usize) -> Real {
    let idx = self.physical_idx(i, j);
    *self.data.get_unchecked(idx)
  }

  #[inline(never)]
  fn into_cholesky(mut self) -> Self {
    // returns L matrix from X = LL* assuming X is positive semi-definite
    // Cholesky-Crout algorithm
    assert_eq!(self.h, self.w);
    let h = self.h;
    for j in 0..h {
      unsafe {
        // top half of matrix is 0s
        for i in 0..j {
          self.set(i, j, 0.0);
        }

        // diagonal requires square root
        let mut s = 0.0;
        for k in 0..j {
          let value = self.get(j, k);
          s += value * value;
        }
        let diag_value = safe_sqrt(self.get(j, j) - s);
        self.set(j, j, diag_value);
        let scale = if diag_value == 0.0 {
          0.0
        } else {
          1.0 / diag_value
        };

        // bottom half
        for i in j + 1..h {
          let mut s = 0.0;
          for k in 0..j {
            s += self.get(i, k) * self.get(j, k);
          }
          self.set(i, j, scale * (self.get(i, j) - s));
        }
      }
    }

    self
  }

  #[inline(never)]
  fn transposed_backward_sub_into(&self, mut y: Matrix) -> Matrix {
    // assuming self is lower triangular, solves for x in
    //   Self^T * x = y
    let h = self.h;
    assert_eq!(h, self.w);
    assert_eq!(h, y.h);
    let w = y.w;
    for k in 0..w {
      for j in (0..h).rev() {
        unsafe {
          let diag_value = y.get(j, k) / self.get(j, j);
          y.set(j, k, diag_value);
          for i in 0..j {
            y.set(
              i,
              k,
              y.get(i, k) - diag_value * self.get(j, i),
            );
          }
        }
      }
    }
    y
  }

  #[inline(never)]
  fn forward_sub_into(&self, mut y: Matrix) -> Matrix {
    // assuming self is lower triangular, solves for x in
    //   Self * x = y
    let h = self.h;
    assert_eq!(h, self.w);
    assert_eq!(h, y.h);
    let w = y.w;
    for k in 0..w {
      for j in 0..h {
        unsafe {
          let diag_value = y.get(j, k) / self.get(j, j);
          y.set(j, k, diag_value);
          for i in j + 1..h {
            y.set(
              i,
              k,
              y.get(i, k) - diag_value * self.get(i, j),
            );
          }
        }
      }
    }
    y
  }
}

#[inline]
fn predict_one<L: Latent>(
  latents: &[L],
  weights: &[L::Conv],
  bias: L::Conv,
  quantization: Bitlen,
) -> L {
  let mut s = bias;
  for (&w, &l) in weights.iter().zip(latents) {
    s += w * l.to_conv();
  }
  L::from_conv(s.max(L::Conv::ZERO) >> quantization)
}

fn predict_into<L: Latent>(
  latents: &[L],
  weights: &[L::Conv],
  bias: L::Conv,
  quantization: Bitlen,
  preds: &mut [L],
) {
  // This should completely fill dst, leaving the results in the 2nd element of
  // each tuple, and reading from slightly more latents.
  // I.e. if there are 4 weights, then the 1st element of dst will be produced
  // by using the first 4 latents.

  // TODO: do passes over latents instead of weights so we can use SIMD?
  let order = weights.len();
  for (i, dst) in preds
    .iter_mut()
    .take(latents.len().saturating_sub(order) + 1)
    .enumerate()
  {
    *dst = predict_one(
      &latents[i..i + order],
      weights,
      bias,
      quantization,
    );
  }
}

fn decode_residuals<L: Latent>(
  weights: &[L::Conv],
  bias: L::Conv,
  quantization: Bitlen,
  residuals: &mut [L],
) {
  let order = weights.len();
  for i in order..residuals.len() {
    unsafe {
      let latent = residuals.get_unchecked(i).wrapping_add(predict_one(
        &residuals[i - order..i],
        weights,
        bias,
        quantization,
      ));
      *residuals.get_unchecked_mut(i) = latent;
    };
  }
}

#[inline(never)]
fn build_initial_autocov_dots(v: &[Real], order: usize) -> Vec<Real> {
  // Some annoying tricks to improve performance here:
  // * manually unroll the inner loop by 4 to get SIMD benefit in the tight loop
  //   and keep things in registers
  // * advance the outer loop in batches to get cache benefit across all dot
  //   products
  assert!(order <= 32);
  let n = v.len();
  let mut dots = vec![0.0; order + 1];
  let almost_n = (n - order) / ENCODE_BATCH_SIZE * ENCODE_BATCH_SIZE;
  for start in (0..almost_n).step_by(ENCODE_BATCH_SIZE) {
    for sep in 0..order + 1 {
      let mut dot0 = 0.0;
      let mut dot1 = 0.0;
      let mut dot2 = 0.0;
      let mut dot3 = 0.0;
      for i in (start..start + ENCODE_BATCH_SIZE).step_by(4) {
        dot0 += v[i] * v[i + sep];
        dot1 += v[i + 1] * v[i + sep + 1];
        dot2 += v[i + 2] * v[i + sep + 2];
        dot3 += v[i + 3] * v[i + sep + 3];
      }
      dots[sep] += (dot0 + dot1) + (dot2 + dot3);
    }
  }
  for i in almost_n..n - order {
    for sep in 0..order + 1 {
      dots[sep] += v[i] * v[i + sep];
    }
  }
  dots
}

#[inline(never)]
fn build_autocov_mats(v: &[Real], order: usize) -> (Matrix, Matrix) {
  // Here we take advantage of the structure of the problem to build the x^Tx
  // and x^Ty matrices with rolling dot products.
  // This is O(n * order + order^2) instead of the naive O(n * order^2)
  // approach.
  let n = v.len();
  let initial_sum: Real = v[..n - order].iter().sum();
  let initial_dots = build_initial_autocov_dots(v, order);

  let mut xtx = Matrix::constant(0.0, order + 1, order + 1);
  let mut xty = Matrix::constant(0.0, order + 1, 1);

  unsafe {
    // fill out the left column and top row
    for (i, &dot) in initial_dots.iter().enumerate().take(order) {
      xtx.set(i, 0, dot);
      xtx.set(0, i, dot);
    }
    xtx.set(order, 0, initial_sum);
    xtx.set(0, order, initial_sum);
    xty.set(0, 0, initial_dots[order]);

    for i in 1..order {
      // fill the main dot products
      for j in 1..=i {
        let last_dot = xtx.get(i - 1, j - 1);
        let dot = last_dot + (v[n - order + i - 1] * v[n - order + j - 1] - v[i - 1] * v[j - 1]);
        xtx.set(i, j, dot);
        xtx.set(j, i, dot);
      }
      // fill the product with 1s for bias term
      let last_sum = xtx.get(order, i - 1);
      let sum = last_sum + (v[n - order + i - 1] - v[i - 1]);
      xtx.set(order, i, sum);
      xtx.set(i, order, sum);
    }
    for i in 1..order {
      // fill the dot products in xty
      let last_dot = xtx.get(order - 1, i - 1);
      let dot = last_dot + (v[n - order + i - 1] * v[n - 1] - v[i - 1] * v[order - 1]);
      xty.set(i, 0, dot);
    }
    // bottom right corners
    xtx.set(order, order, (n - order) as Real);
    let last_sum = xtx.get(order, order - 1);
    let sum = last_sum + (v[n - 1] - v[order - 1]);
    xty.set(order, 0, sum);
  }
  (xtx, xty)
}

fn autocorr_least_squares(v: &[Real], order: usize) -> Matrix {
  // To choose weights and bias, we solve the least squares problem using x features
  //   v_i ~ [v_{i-order}, v_{i-order+1}, ... v_{i-1}, 1.0] @ beta
  // The solution to beta gives the `order` weight coefficients, and its last
  // element is the bias.
  // To do this efficiently, we use a couple tricks:
  // * build the xT^x and x^Ty matrices using rolling dot products, avoiding
  //   duplicate computation
  // * use the Cholesky decomposition and forward/back substitution
  let (xtx, xty) = build_autocov_mats(v, order);
  let cholesky = xtx.into_cholesky();
  let half_solved = cholesky.forward_sub_into(xty);
  cholesky.transposed_backward_sub_into(half_solved)
}

pub fn choose_config<L: Latent>(order: usize, latents: &[L]) -> Option<DeltaConv1Config> {
  let center = latents[latents.len() / 2];
  // let center = sort_utils::choose_pivot(latents);
  let v = latents
    .iter()
    .cloned()
    .map(|v| {
      if v < center {
        -((center - v).to_u64() as Real)
      } else {
        (v - center).to_u64() as Real
      }
    })
    .collect::<Vec<_>>();

  let float_weights_and_centered_bias = autocorr_least_squares(&v, order).data;
  let mut total_weight = 0.0;
  let mut total_abs_weight = 0.0;
  for &w in &float_weights_and_centered_bias[..order] {
    total_abs_weight += w.abs();
    total_weight += w;
  }
  if !total_weight.is_finite() || !total_abs_weight.is_finite() {
    // if we ever add logging, put a debug message here
    return None;
  }
  let float_bias =
    ((1.0 - total_weight) * center.to_u64() as Real) + float_weights_and_centered_bias[order];
  // This quantization should safely avoid overflow: each weight and bias can be rounded
  // up to at most double its value (0.5 -> 1.0), so the largest possible
  // quantized value we could obtain during arithmetic is at most
  //   2^quantization * 2 * (total_abs_weight * L::MAX + abs_bias).
  // Therefore we require
  //   2^quantization * 2 * (total_abs_weight * L::MAX + abs_bias) <= L::Conv::MAX
  //   quantization <= log2(L::Conv::MAX / (total_abs_weight * L::MAX + abs_bias)) - 1
  let quantization = (L::Conv::MAX.to_f64()
    / (total_abs_weight * L::MAX.to_u64() as f64 + float_bias.abs() + 1.0))
    .log2()
    .floor() as i32
    - 1;
  if quantization < 0 {
    return None;
  }
  let quantize_factor = (2.0 as Real).powi(quantization);
  let weights = float_weights_and_centered_bias
    .iter()
    .take(order)
    .map(|x| (x * quantize_factor).round() as i64)
    .collect::<Vec<_>>();
  let bias = (float_bias * quantize_factor) as i64;

  let config = DeltaConv1Config::new(quantization as Bitlen, bias, weights);
  Some(config)
}

pub fn encode_in_place<L: Latent>(config: &DeltaConv1Config, latents: &mut [L]) -> Vec<L> {
  let bias = config.bias::<L::Conv>();
  let weights = config.weights::<L::Conv>();
  let initial_state = latents[..weights.len()].to_vec();
  // Like all delta encode in place functions, we fill the first few (order
  // in this case) latents with junk and properly delta encode the rest.
  let order = weights.len();
  let mut predictions = vec![L::ZERO; ENCODE_BATCH_SIZE + order];
  let mut start = 0;
  while start < latents.len() {
    let end = (start + ENCODE_BATCH_SIZE).min(latents.len());
    // 1. Compute predictions based on this batch and slightly further.
    let dst = &mut predictions[order..];
    predict_into(
      &latents[start..],
      &weights,
      bias,
      config.quantization,
      dst,
    );

    // 2. Use predictions from the end of last batch and most of this batch and
    // take residuals. Don't apply the entirety of this batch yet because we
    // still need those latents to compute the next predictions.
    for (&prediction, latent) in predictions[..ENCODE_BATCH_SIZE]
      .iter()
      .zip(latents[start..end].iter_mut())
    {
      *latent = latent.wrapping_sub(prediction).wrapping_add(L::MID);
    }

    // 3. Copy the predictions from the end of this batch to the start of the
    // next batch's predictions.
    for i in 0..order {
      predictions[i] = predictions[ENCODE_BATCH_SIZE + i];
    }
    start = end;
  }
  initial_state
}

pub fn decode_in_place<L: Latent>(config: &DeltaConv1Config, state: &mut [L], latents: &mut [L]) {
  let weights = &config.weights::<L::Conv>();
  let bias = config.bias::<L::Conv>();
  let order = weights.len();
  assert_eq!(order, state.len());

  delta::toggle_center_in_place(latents);
  let mut residuals = vec![L::ZERO; latents.len() + order];
  residuals[..order].copy_from_slice(&state[..order]);
  residuals[order..order + latents.len()].copy_from_slice(latents);

  decode_residuals(
    weights,
    bias,
    config.quantization,
    &mut residuals,
  );
  latents.copy_from_slice(&residuals[..latents.len()]);
  state.copy_from_slice(&residuals[latents.len()..]);
}

#[cfg(test)]
mod tests {
  use super::*;

  fn matrix_from_rows(rows: Vec<Vec<Real>>) -> Matrix {
    let h = rows.len();
    let w = rows[0].len();
    let mut m = Matrix::constant(0.0, h, w);
    for i in 0..h {
      for j in 0..w {
        unsafe {
          m.set(i, j, rows[i][j]);
        }
      }
    }
    m
  }

  #[test]
  fn build_autocorr_mats() {
    let x = [1.0, 2.0, -1.0, 5.0, -3.0];
    let order = 2;
    let (xtx, xty) = build_autocov_mats(&x, order);

    assert_eq!(xtx.h, 3);
    assert_eq!(xtx.w, 3);
    assert_eq!(
      xtx.data,
      vec![
        6.0, -5.0, 2.0, //
        -5.0, 30.0, 6.0, //
        2.0, 6.0, 3.0, //
      ]
    );

    assert_eq!(xty.h, 3);
    assert_eq!(xty.w, 1);
    assert_eq!(
      xty.data,
      vec![
        12.0,  //
        -22.0, //
        1.0,   //
      ]
    );
  }

  #[test]
  fn cholesky() {
    // here A = LL^T
    //  0.1  0   0
    //  -2   3   0
    //  -4   5   6
    let l = matrix_from_rows(vec![
      vec![0.1, 0.0, 0.0],
      vec![-2.0, 3.0, 0.0],
      vec![-4.0, 5.0, 6.0],
    ]);
    let a = matrix_from_rows(vec![
      vec![0.01, -0.2, -0.4],
      vec![-0.2, 13.0, 23.0],
      vec![-0.4, 23.0, 77.0],
    ]);
    let cholesky = a.into_cholesky();
    assert_eq!(l.data, cholesky.data);
  }

  #[test]
  fn forward_sub() {
    let a = matrix_from_rows(vec![
      vec![2.0, 0.0],  //
      vec![3.0, -4.0], //
    ]);
    let y = matrix_from_rows(vec![
      vec![1.0], //
      vec![2.0], //
    ]);
    let x = a.forward_sub_into(y);
    let expected = vec![0.5, -0.125];
    for i in 0..expected.len() {
      assert!((x.data[i] - expected[i]).abs() < 1E-6);
    }
  }

  #[test]
  fn transpose_backward_sub() {
    let a = matrix_from_rows(vec![
      vec![2.0, 0.0],  //
      vec![3.0, -4.0], //
    ]);
    let y = matrix_from_rows(vec![
      vec![1.0], //
      vec![2.0], //
    ]);
    let x = a.transposed_backward_sub_into(y);
    let expected = vec![1.25, -0.5];
    for i in 0..expected.len() {
      assert!((x.data[i] - expected[i]).abs() < 1E-6);
    }
  }
}
