use core::panic;

use pco::experimental::{Matrix, solve_least_squares};

use crate::img::{ImgView, ImgViewMut};

#[derive(Clone, Debug)]
pub struct ConvFit {
  pub quantization: u32,
  // Per output channel: 4 positions × c channels = 4*c weights
  // Layout: [weights for ch0, weights for ch1, ...]
  // Position order: (-1,-1), (-1,0), (0,-1), (0,0)
  pub weights: Vec<i32>,
  pub biases: Vec<i32>,
}

/// Fit a 2x2 causal convolution including the current pixel.
/// For output channel k, the current pixel's channels k..c are treated as 0.
pub fn fit(chunk: &ImgView) -> ConvFit {
  let (h, w, c) = chunk.hwc();
  let n_weights = 4 * c; // 4 positions × c channels
  let n_coeffs = n_weights + 1; // + bias

  // We fit a separate regression per output channel since each has different causal mask
  let mut all_coeffs = Vec::with_capacity(n_coeffs * c);

  for out_k in 0..c {
    let mut xtx = Matrix::constant(0.0, n_coeffs, n_coeffs);
    let mut xty = Matrix::constant(0.0, n_coeffs, 1);
    let mut x = vec![0.0f64; n_coeffs];
    x[n_weights] = 1.0; // bias term

    for i in 0..h {
      for j in 0..w {
        let pi = i.saturating_sub(1);
        let pj = j.saturating_sub(1);

        // When on boundary, use (pi, pj) for neighbor positions to stay causal
        let (i_above, j_above, i_left, j_left) = if i == 0 || j == 0 {
          (pi, pj, pi, pj)
        } else {
          (pi, j, i, pj)
        };

        // Position 0: (-1, -1) diagonal above-left
        for k in 0..c {
          x[k] = chunk.get(pi, pj, k) as f64;
        }
        // Position 1: (-1, 0) directly above
        for k in 0..c {
          x[c + k] = chunk.get(i_above, j_above, k) as f64;
        }
        // Position 2: (0, -1) directly left
        for k in 0..c {
          x[2 * c + k] = chunk.get(i_left, j_left, k) as f64;
        }
        // Position 3: (0, 0) current pixel - only channels 0..out_k are causal
        for k in 0..c {
          if k < out_k {
            x[3 * c + k] = chunk.get(i, j, k) as f64;
          } else {
            x[3 * c + k] = 0.0; // causally unknown, treat as 0
          }
        }

        // Accumulate X^T X
        for r in 0..n_coeffs {
          for s in 0..n_coeffs {
            unsafe {
              xtx.set(r, s, xtx.get(r, s) + x[r] * x[s]);
            }
          }
        }

        // Accumulate X^T y
        let y = chunk.get(i, j, out_k) as f64;
        for r in 0..n_coeffs {
          unsafe {
            xty.set(r, 0, xty.get(r, 0) + x[r] * y);
          }
        }
      }
    }

    let coeffs = solve_least_squares(xtx, xty).data;
    all_coeffs.extend_from_slice(&coeffs);
  }

  // println!("{:?}", all_coeffs);
  // Quantize weights
  let mut max_magnitude = 0.0f64;
  for out_k in 0..c {
    let mut sum_abs_weights = 0.0f64;
    for r in 0..n_weights {
      sum_abs_weights += all_coeffs[out_k * n_coeffs + r].abs();
    }
    let bias = all_coeffs[out_k * n_coeffs + n_weights];
    let magnitude = sum_abs_weights * 255.0 + bias.abs();
    if !magnitude.is_finite() {
      panic!("Non-finite max magnitude in convolution fit");
    }
    max_magnitude = max_magnitude.max(magnitude);
  }

  let quantization = if max_magnitude > 0.0 {
    ((i32::MAX as f64 / max_magnitude).log2().floor() as u32).min(31) - 1
  } else {
    16
  };
  let scale = (1u64 << quantization) as f64;

  let mut weights = Vec::with_capacity(n_weights * c);
  let mut biases = Vec::with_capacity(c);

  for out_k in 0..c {
    for r in 0..n_weights {
      let coeff = all_coeffs[out_k * n_coeffs + r];
      weights.push((coeff * scale).round() as i32);
    }
  }

  for out_k in 0..c {
    // +0.5, taking into account that our right shift rounds down during prediction
    let coeff = all_coeffs[out_k * n_coeffs + n_weights] + 0.5;
    biases.push((coeff * scale).round() as i32 + 0);
  }
  println!(
    "{} {:?} {:?}",
    quantization, &weights, &biases
  );

  ConvFit {
    quantization,
    weights,
    biases,
  }
}

/// Apply the fitted convolution to predict values
pub fn predict(fit: &ConvFit, chunk: &ImgView, dst: &ImgViewMut) {
  let (h, w, c) = chunk.hwc();
  let n_weights = 4 * c;

  for i in 0..h {
    for j in 0..w {
      let pi = i.saturating_sub(1);
      let pj = j.saturating_sub(1);

      let (i_above, j_above, i_left, j_left) = if i == 0 || j == 0 {
        (pi, pj, pi, pj)
      } else {
        (pi, j, i, pj)
      };

      for out_k in 0..c {
        let mut sum: i32 = fit.biases[out_k];
        let w_base = out_k * n_weights;

        // Position 0: (-1, -1)
        for k in 0..c {
          sum += fit.weights[w_base + k].wrapping_mul(chunk.get(pi, pj, k) as i32);
        }
        // Position 1: (-1, 0)
        for k in 0..c {
          sum += fit.weights[w_base + c + k].wrapping_mul(chunk.get(i_above, j_above, k) as i32);
        }
        // Position 2: (0, -1)
        for k in 0..c {
          sum += fit.weights[w_base + 2 * c + k].wrapping_mul(chunk.get(i_left, j_left, k) as i32);
        }
        // Position 3: (0, 0) - only channels 0..out_k are causal
        for k in 0..out_k {
          // Use original values from chunk (same as what fit was trained on)
          sum += fit.weights[w_base + 3 * c + k].wrapping_mul(chunk.get(i, j, k) as i32);
        }
        // Channels out_k..c at (0,0) are 0, so no contribution

        let pred = (sum >> fit.quantization).clamp(0, 255) as u8;
        unsafe {
          dst.set(i, j, out_k, pred);
        }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::img::Img;

  #[test]
  fn test_fit_predict_basic() {
    let h = 4;
    let w = 4;
    let c = 3;
    let mut img = Img::empty(h, w, c);
    // Fill with gradient
    for i in 0..h {
      for j in 0..w {
        for k in 0..c {
          img.values[c * (i * w + j) + k] = ((i + j + k) * 20) as u8;
        }
      }
    }

    let chunk = img.iter_chunks(h, w).next().unwrap();
    let fit_result = fit(&chunk);

    assert_eq!(fit_result.weights.len(), 4 * c * c);
    assert_eq!(fit_result.biases.len(), c);

    // Verify predict doesn't crash
    let mut out = Img::empty(h, w, c);
    let out_view = out.as_view_mut(h, w);
    predict(&fit_result, &chunk, &out_view);
  }
}
