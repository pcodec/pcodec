use std::cmp;

use crate::constants::Bitlen;
use crate::data_types::{Latent, Signed};
use crate::metadata::DeltaIntConv1Config;
use crate::{delta, sort_utils};

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

  // TODO make this column-major
  #[inline]
  fn physical_idx(&self, i: usize, j: usize) -> usize {
    i * self.w + j
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
  weights: &[L::IntConv],
  bias: L::IntConv,
  quantization: Bitlen,
) -> L {
  let mut s = bias;
  for (&w, &l) in weights.iter().zip(latents) {
    s += w * l.to_int_conv();
  }
  L::from_int_conv(s.max(L::IntConv::ZERO) >> quantization)
}

fn predict_into<L: Latent>(
  latents: &[L],
  weights: &[L::IntConv],
  bias: L::IntConv,
  quantization: Bitlen,
  preds: &mut [L],
) {
  // This should completely fill dst, leaving the results in the 2nd element of
  // each tuple, and reading from slightly more latents.
  // I.e. if there are 4 weights, then the 1st element of dst will be produced
  // by using the first 4 latents.

  // do passes over latents instead of weights so we can use SIMD? TODO
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
  weights: &[L::IntConv],
  bias: L::IntConv,
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

fn autocorr_build_xtx_xty(x: &[Real], order: usize) -> (Matrix, Matrix) {
  // TODO explain
  let n = x.len();
  let mut full_dot_prods = vec![0.0; order + 1];
  fn dot(xi: &[Real], xj: &[Real]) -> Real {
    xi.into_iter().zip(xj).map(|(&xi, &xj)| xi * xj).sum()
  }
  for sep in 0..order + 1 {
    full_dot_prods[sep] = dot(x, &x[sep..]);
  }

  let dot_prod = |i: usize, j: usize| {
    let lo = i.min(j);
    let hi = i.max(j);
    let sep = hi - lo;
    let mut res = full_dot_prods[sep];
    for k in 0..lo {
      res -= x[k] * x[k + sep];
    }
    for k in hi..order {
      res -= x[n - order + k - sep] * x[n - order + k];
    }
    res
  };
  // corr with 1s so we can calculate bias term
  // let full_sum: Real = x.into_iter().sum();
  // let corr_11 = (x.len() - order) as Real;
  // let mut xtx = Matrix::constant(0.0, order + 1, order + 1);
  // let mut xty = Matrix::constant(0.0, order + 1, 1);
  let mut xtx = Matrix::constant(0.0, order, order);
  let mut xty = Matrix::constant(0.0, order, 1);
  for i in 0..order {
    unsafe {
      for j in 0..order {
        xtx.set(i, j, dot_prod(i, j));
      }
      // xtx.set(i, order, full_sum);
      // xtx.set(order, i, full_sum);
      xty.set(i, 0, dot_prod(i, order));
    }
  }
  // unsafe {
  //   xtx.set(order, order, corr_11);
  //   xty.set(order, 0, full_sum);
  // }
  // println!("xtx {:?}", xtx);
  // println!("xty {:?}", xty);
  // let xty = x.transpose_mul(&y);
  (xtx, xty)
}

fn autocorr_least_squares(x: &[Real], order: usize) -> Matrix {
  let (xtx, xty) = autocorr_build_xtx_xty(x, order);
  let cholesky = xtx.into_cholesky();
  // println!("CHOL {:?}", cholesky.data);
  let half_solved = cholesky.forward_sub_into(xty);
  // println!("HALF {:?}", half_solved.data);
  cholesky.transposed_backward_sub_into(half_solved)
}

pub fn choose_config<L: Latent>(
  order: usize,
  latents: &[L],
  ranges: &[(usize, usize)],
) -> Option<DeltaIntConv1Config> {
  // if true {
  //   return Some(DeltaIntConv1Config::new(
  //     9,
  //     0,
  //     vec![176, -676, 1392, -1915, 1529],
  //   ));
  // }
  let center = L::MID;
  let x = latents
    .iter()
    .cloned()
    .map(|x| {
      if x < center {
        -((center - x).to_u64() as f64)
      } else {
        (x - center).to_u64() as f64
      }
    })
    .collect::<Vec<_>>();

  // TODO regularize
  // TODO find a better weight quantization
  let float_weights_and_centered_bias = autocorr_least_squares(&x, order).data;
  // println!("fwcb {:?}", float_weights_and_centered_bias);
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
  // TODO explain why this quantization safely avoids overflow
  let quantization = L::BITS as i32 - 2 - (total_abs_weight + 0.0001).log2().round() as i32;
  if quantization < 0 {
    return None;
  }
  let quantize_factor = (2.0 as Real).powi(quantization);
  let weights = float_weights_and_centered_bias
    .iter()
    .take(order)
    .map(|x| (x * quantize_factor).round() as i64)
    .collect::<Vec<_>>();
  // let float_bias = ((1.0 - total_weight) * center.to_u64() as f64)
  //   + *float_weights_and_centered_bias.last().unwrap();
  let float_bias = (1.0 - total_weight) * center.to_u64() as f64;
  let bias = float_bias as i64;

  let config = DeltaIntConv1Config::new(quantization as Bitlen, bias, weights);
  // println!("config {:?}", config);
  Some(config)
}

pub fn encode_in_place<L: Latent>(config: &DeltaIntConv1Config, latents: &mut [L]) -> Vec<L> {
  let bias = config.bias::<L::IntConv>();
  let weights = config.weights::<L::IntConv>();
  let initial_state = latents[..weights.len()].to_vec();
  // Like all delta encode in place functions, we fill the first few (order
  // in this case) latents with junk and properly delta encode the rest.
  let order = weights.len();
  let mut predictions = vec![L::ZERO; ENCODE_BATCH_SIZE + order];
  let mut start = 0;
  while start < latents.len() {
    let end = cmp::min(start + ENCODE_BATCH_SIZE, latents.len());
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

pub fn decode_in_place<L: Latent>(
  config: &DeltaIntConv1Config,
  state: &mut [L],
  latents: &mut [L],
) {
  let weights = &config.weights::<L::IntConv>();
  let bias = config.bias::<L::IntConv>();
  let order = weights.len();
  assert_eq!(order, state.len());

  delta::toggle_center_in_place(latents);
  let mut residuals = vec![L::ZERO; latents.len() + order];
  residuals[..order].copy_from_slice(&state[..order]);
  residuals[order..order + latents.len()].copy_from_slice(&latents);

  decode_residuals(
    &weights,
    bias,
    config.quantization,
    &mut residuals,
  );
  latents.copy_from_slice(&residuals[..latents.len()]);
  // println!("decoded latents {:?}", latents);

  state.copy_from_slice(&residuals[latents.len()..]);
  // println!("new state {:?}", state);
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
    let (xtx, xty) = autocorr_build_xtx_xty(&x, order);

    assert_eq!(xtx.h, 2);
    assert_eq!(xtx.w, 2);
    assert_eq!(xtx.data, vec![6.0, -5.0, -5.0, 30.0,]);

    assert_eq!(xty.h, 2);
    assert_eq!(xty.w, 1);
    assert_eq!(
      xty.data,
      vec![
        12.0,  //
        -22.0, //
      ]
    );
  }
  // #[test]
  // fn build_autocorr_mats() {
  //   let x = [1.0, 2.0, -1.0, 5.0, -3.0];
  //   let order = 2;
  //   let (xtx, xty) = autocorr_build_xtx_xty(&x, order);

  //   assert_eq!(xtx.h, 3);
  //   assert_eq!(xtx.w, 3);
  //   assert_eq!(
  //     xtx.data,
  //     vec![
  //       6.0, -5.0, 2.0, //
  //       -5.0, 6.0, 2.0, //
  //       2.0, 2.0, 3.0, //
  //     ]
  //   );

  //   assert_eq!(xty.h, 3);
  //   assert_eq!(xty.w, 1);
  //   assert_eq!(
  //     xty.data,
  //     vec![
  //       12.0, //
  //       -5.0, //
  //       2.0,  //
  //     ]
  //   );
  // }

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

  // #[test]
  // fn least_squares() {
  //   let n_features = 2;
  //   let n = 4;
  //   let true_beta = Matrix {
  //     data: vec![2.0, -3.0],
  //     h: n_features,
  //     w: 1,
  //   };

  //   let x = matrix_from_rows(vec![
  //     vec![10.0, 11.0],
  //     vec![12.0, 13.0],
  //     vec![14.0, 15.0],
  //     vec![16.0, 17.0],
  //   ]);
  //   let mut x_transpose = Matrix::constant(0.0, n_features, n);
  //   for i in 0..n {
  //     for j in 0..n_features {
  //       unsafe {
  //         x_transpose.set(j, i, x.get(i, j));
  //       }
  //     }
  //   }
  //   let y = x_transpose.transpose_mul(&true_beta);

  //   let beta = ordinary_least_squares(x, y);
  //   assert_eq!(beta.data.len(), n_features);
  //   for i in 0..n_features {
  //     unsafe { assert!((beta.get(i, 0) - true_beta.get(i, 0)).abs() < 1E-3) }
  //   }
  // }

  // #[test]
  // fn test_predict_one() {
  //   let latents = vec![32747_u16, 32774];
  //   let weights = vec![65427_u16, 359];
  //   let bias = 796_u16;
  //   let quantization = 8;
  //   let predicted = predict_one(&latents, &weights, bias, quantization);
  //   assert_eq!(predicted, 32813);
  // }

  // #[test]
  // fn test_predict_into_from() {
  //   // these latents are equivalent to [10, -20, 30, -40]
  //   let latents = vec![10_u32, u32::MAX - 19, 30_u32, u32::MAX - 39];
  //   let weights = vec![4, -3];
  //   let bias = -5;
  //   let quantization = 3;
  //   let mut preds = vec![0; latents.len() - weights.len() + 1];

  //   // INTO
  //   predict_into(
  //     &latents,
  //     &weights,
  //     bias,
  //     quantization,
  //     &mut preds,
  //   );
  //   // the last prediction here is useless to us, but just to show that it can
  //   // predict the next one:
  //   assert_eq!(preds, vec![11, u32::MAX - 21, 29]);

  //   // FROM
  //   let mut recovered = vec![0; latents.len()];
  //   recovered[..2].copy_from_slice(&latents[..2]);
  //   recovered[2..].copy_from_slice(
  //     &latents
  //       .iter()
  //       .skip(2)
  //       .zip(&preds)
  //       .map(|(l, &p)| l.wrapping_sub(p))
  //       .collect::<Vec<_>>(),
  //   );
  //   decode_residuals(&weights, bias, quantization, &mut recovered);
  //   assert_eq!(recovered, latents);
  // }
}
