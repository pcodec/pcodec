use std::cmp;

use crate::constants::Bitlen;
use crate::data_types::{Latent, Signed};
use crate::metadata::DeltaIntConv1Config;
use crate::{delta, sort_utils};

type Real = f64;

const ENCODE_BATCH_SIZE: usize = 512;
const QUANTIZATION: Bitlen = 11;

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
    // backward substitution into y, but where self is transposed
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

fn autocorr_least_squares(x: &[Real], order: usize) -> Matrix {
  let n = x.len();
  let x0 = &x[..n - order - 1];
  let mut autocorrs = vec![0.0; order + 1];
  fn dot(xi: &[Real], xj: &[Real]) -> Real {
    xi.into_iter().zip(xj).map(|(&xi, &xj)| xi * xj).sum()
  }
  for sep in 0..order + 1 {
    autocorrs[sep] = dot(x0, &x[sep..n - order - 1 + sep]);
  }
  let mut xtx = Matrix::constant(0.0, order, order);
  let mut xty = Matrix::constant(0.0, order, 1);
  for i in 0..order {
    for j in 0..order {
      unsafe {
        xtx.set(
          i,
          j,
          autocorrs[(i as i32 - j as i32).abs() as usize],
        );
      }
    }
    unsafe {
      xty.set(i, 0, autocorrs[order - i]);
    }
  }
  // let xty = x.transpose_mul(&y);
  let cholesky = xtx.into_cholesky();
  let half_solved = cholesky.forward_sub_into(xty);
  cholesky.transposed_backward_sub_into(half_solved)
}

pub fn choose_config<L: Latent>(
  order: usize,
  latents: &[L],
  ranges: &[(usize, usize)],
) -> Option<DeltaIntConv1Config> {
  // TODO
  let center = sort_utils::choose_pivot(latents);
  // TODO
  let centered = latents
    .iter()
    .map(|&l| {
      if l >= center {
        (l - center).to_u64() as Real
      } else {
        -((center - l).to_u64() as Real)
      }
    })
    .collect::<Vec<_>>();
  // println!("center {}", center);

  let _n_pts = ranges
    .iter()
    .map(|(start, end)| *end - *start - order)
    .sum::<usize>();

  // TODO regularize
  let quantization = QUANTIZATION;
  let quantize_factor = (2.0 as Real).powi(quantization as i32);
  // TODO find a better weight quantization
  let float_weights_and_centered_bias = autocorr_least_squares(&centered, order).data;
  if float_weights_and_centered_bias
    .iter()
    .any(|weight| !weight.is_finite())
  {
    // if we ever add logging, put a debug message here
    return None;
  }

  let mut total_weight = 0.0;
  for &w in &float_weights_and_centered_bias[..order] {
    total_weight += w;
  }
  let weights = float_weights_and_centered_bias
    .iter()
    .take(order)
    .map(|x| (x * quantize_factor).round() as i64)
    .collect::<Vec<_>>();
  let float_bias = (1.0 - total_weight) * center.to_u64() as Real;
  let bias = (float_bias * quantize_factor) as i64;

  let config = DeltaIntConv1Config::new(quantization, bias, weights);
  println!("config {:?}", config);
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
  fn forward_sub() {
    let a = matrix_from_rows(vec![vec![2.0, 0.0], vec![3.0, -4.0]]);
    let y = matrix_from_rows(vec![vec![1.0], vec![2.0]]);
    let x = a.forward_sub_into(y);
    let expected = vec![0.5, -0.125];
    for i in 0..expected.len() {
      assert!((x.data[i] - expected[i]).abs() < 1E-6);
    }
  }

  #[test]
  fn transpose_backward_sub() {
    let a = matrix_from_rows(vec![vec![2.0, 0.0], vec![3.0, -4.0]]);
    let y = matrix_from_rows(vec![vec![1.0], vec![2.0]]);
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
