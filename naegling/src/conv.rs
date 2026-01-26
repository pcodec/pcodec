use core::panic;

use pco::experimental::{Matrix, solve_least_squares};

use crate::img::{ImgView, ImgViewMut};

#[derive(Clone, Debug)]
pub struct ConvFit {
  pub top_left: u8,
  pub quantization: u32,
  // Per output channel: 4 positions × c channels = 4*c weights
  // Position order: (-1,-1), (-1,0), (0,-1), (0,0)
  pub weights: Vec<i32>,
  pub bias: i32,
}

impl ConvFit {
  pub fn weights_i16(&self) -> Vec<i16> {
    self.weights.iter().map(|&w| w as i16).collect()
  }
}

/// Fit a 2x2 causal convolution including the current pixel.
/// For output channel k, the current pixel's channels k..c are treated as 0.
pub fn fit_conv(chunk: &ImgView) -> Vec<ConvFit> {
  let (h, w, c) = chunk.hwc();
  let n_weights = 4 * c; // 4 positions × c channels
  let n_coeffs = n_weights + 1; // + bias
  let mut fits = Vec::with_capacity(c);

  for out_k in 0..c {
    // We fit a separate regression per output channel since each has different causal mask
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

    // Quantize weights
    let mut max_abs_weight = 0.0_f64;
    let mut sum_abs_weights = 0.0_f64;
    for r in 0..n_weights {
      let abs_weight = coeffs[r].abs();
      sum_abs_weights += abs_weight;
      max_abs_weight = max_abs_weight.max(abs_weight);
    }
    let bias = coeffs[n_weights];
    let max_abs_sum = sum_abs_weights * 255.0 + bias.abs();
    if !max_abs_sum.is_finite() {
      panic!("Non-finite max magnitude in convolution fit");
    }

    let mut quantization_bounds = vec![31];
    if max_abs_weight > 0.0 {
      // bound to avoid arithmetic overflow
      quantization_bounds.push((i32::MAX as f64 / max_abs_sum).log2().floor() as u32 - 1);
      // bound to fit weights in i16
      quantization_bounds.push((i16::MAX as f64 / max_abs_weight).log2().floor() as u32 - 1);
    };
    let quantization = *quantization_bounds.iter().min().unwrap();
    let scale = (1u64 << quantization) as f64;

    let weights = coeffs
      .iter()
      .take(n_weights)
      .map(|&coeff| (coeff * scale).round() as i32)
      .collect::<Vec<_>>();
    // +0.5, taking into account that our right shift rounds down during prediction
    let bias = ((coeffs[n_weights] + 0.5) * scale).round() as i32;

    fits.push(ConvFit {
      top_left: chunk.get(0, 0, out_k),
      quantization,
      weights,
      bias,
    });
  }

  fits
}

/// Apply the fitted convolution to predict values
pub fn predict(fits: &[ConvFit], chunk: &ImgView, dst: &ImgViewMut) {
  let (h, w, c) = chunk.hwc();

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
        let fit = &fits[out_k];
        let weights = &fit.weights;
        let mut sum: i32 = fit.bias;

        // Position 0: (-1, -1)
        for k in 0..c {
          sum += weights[k].wrapping_mul(chunk.get(pi, pj, k) as i32);
        }
        // Position 1: (-1, 0)
        for k in 0..c {
          sum += weights[c + k].wrapping_mul(chunk.get(i_above, j_above, k) as i32);
        }
        // Position 2: (0, -1)
        for k in 0..c {
          sum += weights[2 * c + k].wrapping_mul(chunk.get(i_left, j_left, k) as i32);
        }
        // Position 3: (0, 0) - only channels 0..out_k are causal
        for k in 0..out_k {
          // Use original values from chunk (same as what was trained on)
          sum += weights[3 * c + k].wrapping_mul(chunk.get(i, j, k) as i32);
        }
        // Channels out_k..c at (0,0) are 0, so no contribution

        let pred = (sum >> fit.quantization).clamp(0, 255) as u8;
        dst.set(i, j, out_k, pred);
      }
    }
  }

  for out_k in 0..c {
    dst.set(0, 0, out_k, fits[out_k].top_left);
  }
}

/// Decode one pixel channel at a time, reconstructing original values from residuals.
/// Uses top_left from each ConvFit for the initial pixel and computes predictions
/// from already-decoded values in dst.
pub fn decode(fits: &[ConvFit], dst: &ImgViewMut) {
  let (h, w, c) = dst.hwc();

  // Initialize top-left corner for all channels (residuals assumed to be 0)
  for out_k in 0..c {
    dst.set(0, 0, out_k, fits[out_k].top_left);
  }

  for i in 0..h {
    let jmin = if i == 0 { 1 } else { 0 };
    for j in jmin..w {
      let pi = i.saturating_sub(1);
      let pj = j.saturating_sub(1);

      let (i_above, j_above, i_left, j_left) = if i == 0 || j == 0 {
        (pi, pj, pi, pj)
      } else {
        (pi, j, i, pj)
      };

      for out_k in 0..c {
        let fit = &fits[out_k];
        let weights = &fit.weights;
        let mut sum: i32 = fit.bias;

        // Position 0: (-1, -1) - use already-decoded values from dst
        for k in 0..c {
          sum += weights[k].wrapping_mul(dst.get(pi, pj, k) as i32);
        }
        // Position 1: (-1, 0) - use already-decoded values from dst
        for k in 0..c {
          sum += weights[c + k].wrapping_mul(dst.get(i_above, j_above, k) as i32);
        }
        // Position 2: (0, -1) - use already-decoded values from dst
        for k in 0..c {
          sum += weights[2 * c + k].wrapping_mul(dst.get(i_left, j_left, k) as i32);
        }
        // Position 3: (0, 0) - only channels 0..out_k are causal (already decoded)
        for k in 0..out_k {
          sum += weights[3 * c + k].wrapping_mul(dst.get(i, j, k) as i32);
        }

        let pred = (sum >> fit.quantization).clamp(0, 255) as u8;
        let decoded = pred.wrapping_add(dst.get(i, j, out_k));
        dst.set(i, j, out_k, decoded);
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
    let fits = fit_conv(&chunk);

    assert_eq!(fits.len(), c);
    assert_eq!(fits[0].weights.len(), 4 * c);

    // Verify predict doesn't crash
    let mut out = Img::empty(h, w, c);
    let out_view = out.as_view_mut(h, w);
    predict(&fits, &chunk, &out_view);
  }
}
