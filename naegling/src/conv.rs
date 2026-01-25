use pco::experimental::{Matrix, solve_least_squares};

use crate::img::{Img, ImgView, ImgViewMut};

pub struct Conv2X2Fit {
  top_left: Vec<u8>,
  quantization: u32,
  weights: Vec<i16>,
  biases: Vec<i16>,
}

pub fn fit_2x2<'a>(chunk: &ImgView<'a>) -> Conv2X2Fit {
  let (h, w, c) = chunk.hwc();
  let n_coeffs = 3 * c + 1;
  let mut xtx = Matrix::constant(0.0, n_coeffs, n_coeffs);
  let mut xty = Matrix::constant(0.0, n_coeffs, c);

  // Iterate over all pixels, padding the top-left edge by copying nearest valid pixel
  // Build feature vector x with layout:
  // [c channels at (-1,-1), c channels at (-1,0), c channels at (0,-1), bias]
  let mut x = vec![0.0f64; n_coeffs];
  x[3 * c] = 1.0;

  for i in 0..h {
    for j in 0..w {
      let pi = i.saturating_sub(1); // padded i-1
      let pj = j.saturating_sub(1); // padded j-1

      // Offset (-1, -1): diagonal above-left
      for k in 0..c {
        x[k] = chunk.get(pi, pj, k) as f64;
      }
      // Offset (-1, 0): directly above
      for k in 0..c {
        x[c + k] = chunk.get(pi, j, k) as f64;
      }
      // Offset (0, -1): directly left
      for k in 0..c {
        x[2 * c + k] = chunk.get(i, pj, k) as f64;
      }

      // Accumulate X^T X (outer product x * x^T)
      for r in 0..n_coeffs {
        for s in 0..n_coeffs {
          unsafe {
            xtx.set(r, s, xtx.get(r, s) + x[r] * x[s]);
          }
        }
      }

      // Accumulate X^T y for each output channel
      for out_k in 0..c {
        let y = chunk.get(i, j, out_k) as f64;
        for r in 0..n_coeffs {
          unsafe {
            xty.set(r, out_k, xty.get(r, out_k) + x[r] * y);
          }
        }
      }
    }
  }

  let coeffs = solve_least_squares(xtx, xty).data;

  // Quantize weights to i16
  // The prediction will be: (sum(w_i * x_i) + bias) >> quantization
  // where x_i are u8 [0, 255] and w_i, bias are i16
  // We need the worst case sum(|w_i| * 255) + |bias| to fit in i16
  let n_weights = 3 * c;

  // Find max possible magnitude of (w . x + b) across all output channels
  let mut max_magnitude = 0.0f64;
  for out_k in 0..c {
    let mut sum_abs_weights = 0.0f64;
    for r in 0..n_weights {
      sum_abs_weights += coeffs[r + out_k * n_coeffs].abs();
    }
    let bias = coeffs[n_weights + out_k * n_coeffs];
    let magnitude = sum_abs_weights * 255.0 + bias.abs();
    max_magnitude = max_magnitude.max(magnitude);
  }

  // Choose quantization such that max_magnitude * 2^quantization <= 32767
  let quantization = if max_magnitude > 0.0 {
    ((32767.0 / max_magnitude).log2().floor() as u32).min(15)
  } else {
    8
  };
  let scale = (1u32 << quantization) as f64;

  let mut weights = Vec::with_capacity(n_weights * c);
  let mut biases = Vec::with_capacity(c);

  // Extract and scale weights (rows 0..n_weights for each output channel)
  for out_k in 0..c {
    for r in 0..n_weights {
      let coeff = coeffs[r + out_k * n_coeffs];
      weights.push((coeff * scale).round() as i16);
    }
  }

  // Extract and scale biases (row n_weights = 3*c for each output channel)
  for out_k in 0..c {
    let coeff = coeffs[n_weights + out_k * n_coeffs];
    biases.push((coeff * scale).round() as i16);
  }

  Conv2X2Fit {
    top_left: (0..c).map(|k| chunk.get(0, 0, k)).collect(),
    quantization,
    weights,
    biases,
  }
}

pub fn predict<'a>(fit: &Conv2X2Fit, chunk: &ImgView, dst: &ImgViewMut<'a>) {
  let (h, w, c) = chunk.hwc();
  let n_weights = 3 * c;

  for i in 0..h {
    for j in 0..w {
      let pi = i.saturating_sub(1);
      let pj = j.saturating_sub(1);

      for out_k in 0..c {
        let mut sum: i16 = 0;

        // Offset (-1, -1): diagonal above-left
        for k in 0..c {
          sum += fit.weights[out_k * n_weights + k].wrapping_mul(chunk.get(pi, pj, k) as i16);
        }
        // Offset (-1, 0): directly above
        for k in 0..c {
          sum += fit.weights[out_k * n_weights + c + k].wrapping_mul(chunk.get(pi, j, k) as i16);
        }
        // Offset (0, -1): directly left
        for k in 0..c {
          sum +=
            fit.weights[out_k * n_weights + 2 * c + k].wrapping_mul(chunk.get(i, pj, k) as i16);
        }

        sum += fit.biases[out_k];
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

  fn apply_fit(fit: &Conv2X2Fit, chunk: &ImgView, i: usize, j: usize, out_k: usize) -> i16 {
    let (_, _, c) = chunk.hwc();
    let n_weights = 3 * c;
    let mut sum: i32 = 0;

    // Offset (-1, -1)
    for k in 0..c {
      let w = fit.weights[out_k * n_weights + k] as i32;
      let x = chunk.get(i - 1, j - 1, k) as i32;
      sum += w * x;
    }
    // Offset (-1, 0)
    for k in 0..c {
      let w = fit.weights[out_k * n_weights + c + k] as i32;
      let x = chunk.get(i - 1, j, k) as i32;
      sum += w * x;
    }
    // Offset (0, -1)
    for k in 0..c {
      let w = fit.weights[out_k * n_weights + 2 * c + k] as i32;
      let x = chunk.get(i, j - 1, k) as i32;
      sum += w * x;
    }
    sum += fit.biases[out_k] as i32;

    // Verify it fits in i16 before shift
    assert!(
      sum >= i16::MIN as i32 && sum <= i16::MAX as i32,
      "sum {} overflows i16",
      sum
    );

    sum as i16
  }

  #[test]
  fn test_fit_2x2_grayscale_gradient() {
    // Create a simple grayscale gradient image
    let h = 4;
    let w = 4;
    let c = 1;
    let mut img = Img::empty(h, w, c);
    // Fill with a gradient: value = i + j
    for i in 0..h {
      for j in 0..w {
        img.values[i * w + j] = (i + j) as u8;
      }
    }

    let chunk = img.iter_chunks(h, w).next().unwrap();
    let fit = fit_2x2(&chunk);

    assert_eq!(fit.weights.len(), 3 * c);
    assert_eq!(fit.biases.len(), c);

    // Verify predictions stay in i16 bounds
    let chunk = img.iter_chunks(h, w).next().unwrap();
    for i in 1..h {
      for j in 1..w {
        let pred = apply_fit(&fit, &chunk, i, j, 0);
        // Just verify no overflow occurred (assert in apply_fit)
        let _ = pred;
      }
    }
  }

  #[test]
  fn test_fit_2x2_rgb_constant() {
    // Create a constant RGB image - should learn zero weights and constant bias
    let h = 4;
    let w = 4;
    let c = 3;
    let mut img = Img::empty(h, w, c);
    for i in 0..h {
      for j in 0..w {
        img.values[c * (i * w + j) + 0] = 100; // R
        img.values[c * (i * w + j) + 1] = 150; // G
        img.values[c * (i * w + j) + 2] = 200; // B
      }
    }

    let chunk = img.iter_chunks(h, w).next().unwrap();
    let fit = fit_2x2(&chunk);

    assert_eq!(fit.weights.len(), 3 * c * c); // 9 weights per output channel, 3 channels
    assert_eq!(fit.biases.len(), c);

    // Verify predictions stay in i16 bounds for all channels
    let chunk = img.iter_chunks(h, w).next().unwrap();
    for i in 1..h {
      for j in 1..w {
        for out_k in 0..c {
          let pred = apply_fit(&fit, &chunk, i, j, out_k);
          let _ = pred;
        }
      }
    }
  }

  #[test]
  fn test_fit_2x2_extreme_values() {
    // Test with extreme values to verify i16 bounds
    let h = 4;
    let w = 4;
    let c = 1;
    let mut img = Img::empty(h, w, c);
    // Checkerboard of 0 and 255
    for i in 0..h {
      for j in 0..w {
        img.values[i * w + j] = if (i + j) % 2 == 0 { 0 } else { 255 };
      }
    }

    let chunk = img.iter_chunks(h, w).next().unwrap();
    let fit = fit_2x2(&chunk);

    // Verify predictions stay in i16 bounds
    let chunk = img.iter_chunks(h, w).next().unwrap();
    for i in 1..h {
      for j in 1..w {
        let pred = apply_fit(&fit, &chunk, i, j, 0);
        let _ = pred;
      }
    }
  }
}
