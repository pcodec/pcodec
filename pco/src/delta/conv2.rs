use crate::{data_types::Latent, metadata::delta_encoding::DeltaConv2Config};

// validation: grid size > kernel size, overflow,

pub fn new_buffer_and_pos<L: Latent>(
  config: &DeltaConv2Config,
  stored_state: Vec<L>,
  page_n: usize,
) -> (Vec<L>, usize) {
  // initializing this with 0s is a bit of a waste, but keeping it simple for now
  let mut state = vec![L::ZERO; page_n];
  state[..stored_state.len()].copy_from_slice(&stored_state);
  (state, config.kw)
}

fn predict_one<L: Latent>(
  weights: &[L::Conv],
  bias: L::Conv,
  w: usize,
  kw: usize,
  kh: usize,
  i: usize,
  j: usize,
  grid: &[L],
) -> L {
  let khm1 = kh - 1;
  let kwm1 = kw - 1;
  let mut s = bias;
  let default = grid[(i.saturating_sub(khm1)) * w + j.saturating_sub(kwm1)].to_conv();
  for di in 0..kh {
    for dj in 0..kw {
      let v = if i + di >= khm1 && j + dj >= kwm1 {
        grid[(i + di - khm1) * w + j + dj - kwm1].to_conv()
      } else {
        default
      };
      s += weights[di * kw + dj] * v;
    }
  }
  L::from_conv(s)
}

pub fn encode_in_place<L: Latent>(config: &DeltaConv2Config, latents: &mut [L]) -> Vec<L> {
  let initial_state = latents[..config.kw].to_vec();

  let weights = config.weights::<L::Conv>();
  let bias = config.bias::<L::Conv>();
  let (residuals, latents) = {
    let orig_latents = latents.to_vec();
    (latents, orig_latents)
  };

  let &DeltaConv2Config { w, kh, kw, .. } = config;
  let h = latents.len() / w;

  for i in 0..h {
    let start = if i == 0 { kw } else { 0 };
    for j in start..w {
      let pred = predict_one(&weights, bias, w, kw, kh, i, j, &latents);
      residuals[i * w + j] = latents[i * w + j].wrapping_sub(pred);
    }
  }
  initial_state
}

pub fn decode_in_place<L: Latent>(
  config: &DeltaConv2Config,
  state: &mut [L],
  state_pos: &mut usize,
  latents: &mut [L],
) {
}
