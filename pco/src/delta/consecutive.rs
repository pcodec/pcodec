use crate::data_types::Latent;

fn first_order_encode_consecutive_in_place<L: Latent>(latents: &mut [L]) {
  if latents.is_empty() {
    return;
  }

  for i in (1..latents.len()).rev() {
    latents[i] = latents[i].wrapping_sub(latents[i - 1]);
  }
}

// Used for a single page, so we return the delta moments.
// All encode in place functions leave junk data (`order`
// latents in this case) at the front of the latents.
// Using the front instead of the back is preferable because it makes the lookback
// encode function simpler and faster.
#[inline(never)]
pub fn encode_in_place<L: Latent>(order: usize, mut latents: &mut [L]) -> Vec<L> {
  // TODO this function could be made faster by doing all steps on mini batches
  // of ~512 at a time
  let mut page_moments = Vec::with_capacity(order);
  for _ in 0..order {
    page_moments.push(latents.first().copied().unwrap_or(L::ZERO));

    first_order_encode_consecutive_in_place(latents);
    let truncated_start = latents.len().min(1);
    latents = &mut latents[truncated_start..];
  }
  super::toggle_center_in_place(latents);

  page_moments
}

fn first_order_decode_consecutive_in_place<L: Latent>(moment: &mut L, latents: &mut [L]) {
  for delta in latents.iter_mut() {
    let tmp = *delta;
    *delta = *moment;
    *moment = moment.wrapping_add(tmp);
  }
}

// used for a single batch, so we mutate the delta moments
#[inline(never)]
pub fn decode_in_place<L: Latent>(delta_moments: &mut [L], latents: &mut [L]) {
  super::toggle_center_in_place(latents);
  for moment in delta_moments.iter_mut().rev() {
    first_order_decode_consecutive_in_place(moment, latents);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_consecutive_encode_decode() {
    let orig_latents: Vec<u32> = vec![2, 2, 1, u32::MAX, 0];
    let mut deltas = orig_latents.clone();
    let order = 2;
    let mut moments = encode_in_place(order, &mut deltas);

    // Encoding left junk deltas at the front,
    // but for decoding we need junk deltas at the end.
    let mut deltas_to_decode = Vec::new();
    deltas_to_decode.extend(&deltas[order..]);
    for _ in 0..order {
      deltas_to_decode.push(1337);
    }
    let mut deltas = deltas_to_decode;

    // decode in two parts to show we keep state properly
    decode_in_place(&mut moments, &mut deltas[..3]);
    assert_eq!(&deltas[..3], &orig_latents[..3]);

    decode_in_place(&mut moments, &mut deltas[3..]);
    assert_eq!(&deltas[3..5], &orig_latents[3..5]);
  }
}
