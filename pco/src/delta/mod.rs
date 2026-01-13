mod consecutive;
mod conv1;
mod lookback;

use crate::bits;
use crate::constants::{Bitlen, DeltaLookback};
use crate::data_types::Latent;
use crate::dyn_latent_slice::DynLatentSlice;
use crate::errors::{PcoError, PcoResult};
use crate::macros::match_latent_enum;
use crate::metadata::delta_encoding::LatentVarDeltaEncoding;
use crate::metadata::dyn_latents::DynLatents;
use crate::metadata::{DeltaEncoding, DeltaLookbackConfig};
use std::ops::Range;

const LOOKBACK_MAX_WINDOW_N_LOG: Bitlen = 15;
const LOOKBACK_MIN_WINDOW_N_LOG: Bitlen = 4;

pub type DeltaState = DynLatents;

// Without this, deltas in, say, [-5, 5] would be split out of order into
// [U::MAX - 4, U::MAX] and [0, 5].
// This can be used to convert from
// * unsigned deltas -> (effectively) signed deltas; encoding
// * signed deltas -> unsigned deltas; decoding
#[inline(never)]
fn toggle_center_in_place<L: Latent>(latents: &mut [L]) {
  for l in latents.iter_mut() {
    *l = l.toggle_center();
  }
}

// TODO taking deltas of secondary latents has been proven to help slightly
// in some cases, so we should consider it in the future
pub fn new_lookback(n: usize) -> DeltaEncoding {
  DeltaEncoding::Lookback {
    config: DeltaLookbackConfig {
      window_n_log: bits::bits_to_encode_offset(n as u32 - 1).clamp(
        LOOKBACK_MIN_WINDOW_N_LOG,
        LOOKBACK_MAX_WINDOW_N_LOG,
      ),
      state_n_log: 0,
    },
    secondary_uses_delta: false,
  }
}

pub fn new_conv1(order: usize, latents: &DynLatents) -> PcoResult<Option<DeltaEncoding>> {
  match latents {
    DynLatents::U16(_) | DynLatents::U32(_) => (),
    DynLatents::U64(_) => {
      // we don't support u64 int conv because of lack of a large enough
      // efficient accumulator type on regular CPUs
      return Err(PcoError::invalid_argument(
        "Conv1 delta encoding cannot be used with 64-bit latents",
      ));
    }
  }

  let delta_encoding = match_latent_enum!(
    latents,
    DynLatents<L>(latents) => {
      conv1::choose_config(order, latents)
    }
  )
  .map(DeltaEncoding::Conv1);
  Ok(delta_encoding)
}

pub fn new_buffer_and_pos<L: Latent>(
  delta_encoding: &LatentVarDeltaEncoding,
  stored_state: Vec<L>,
) -> (Vec<L>, usize) {
  match delta_encoding {
    LatentVarDeltaEncoding::NoOp
    | LatentVarDeltaEncoding::Consecutive(_)
    | LatentVarDeltaEncoding::Conv1(_) => (stored_state, 0),
    LatentVarDeltaEncoding::Lookback(config) => {
      lookback::new_window_buffer_and_pos(*config, &stored_state)
    }
  }
}

pub fn compute_delta_latent_var(
  delta_encoding: &DeltaEncoding,
  primary_latents: &mut DynLatents,
  range: Range<usize>,
) -> Option<DynLatents> {
  match delta_encoding {
    DeltaEncoding::NoOp | DeltaEncoding::Consecutive { .. } | DeltaEncoding::Conv1(_) => None,
    DeltaEncoding::Lookback { config, .. } => {
      let res = match_latent_enum!(
        primary_latents,
        DynLatents<L>(inner) => {
          let latents = &mut inner[range];
          DynLatents::new(lookback::choose_lookbacks(*config, latents))
        }
      );
      Some(res)
    }
  }
}

pub fn encode_in_place<L: Latent>(
  delta_encoding: &LatentVarDeltaEncoding,
  delta_latents: Option<&DynLatents>,
  latents: &mut [L],
) -> Vec<L> {
  match delta_encoding {
    LatentVarDeltaEncoding::NoOp => vec![],
    LatentVarDeltaEncoding::Consecutive(order) => consecutive::encode_in_place(*order, latents),
    LatentVarDeltaEncoding::Lookback(config) => {
      let lookbacks = delta_latents
        .unwrap()
        .downcast_ref::<DeltaLookback>()
        .unwrap();
      lookback::encode_in_place(*config, lookbacks, latents)
    }
    LatentVarDeltaEncoding::Conv1(config) => conv1::encode_in_place(config, latents),
  }
}

pub fn decode_in_place<L: Latent>(
  delta_encoding: &LatentVarDeltaEncoding,
  delta_latents: Option<DynLatentSlice>,
  state_pos: &mut usize,
  state: &mut [L],
  latents: &mut [L],
) -> PcoResult<()> {
  match delta_encoding {
    LatentVarDeltaEncoding::NoOp => Ok(()),
    LatentVarDeltaEncoding::Consecutive(_) => {
      consecutive::decode_in_place(state, latents);
      Ok(())
    }
    LatentVarDeltaEncoding::Lookback(config) => {
      let has_oob_lookbacks = lookback::decode_in_place(
        *config,
        delta_latents.unwrap().downcast_unwrap::<DeltaLookback>(),
        state_pos,
        state,
        latents,
      );
      if has_oob_lookbacks {
        Err(PcoError::corruption(
          "delta lookback exceeded window n",
        ))
      } else {
        Ok(())
      }
    }
    LatentVarDeltaEncoding::Conv1(config) => {
      conv1::decode_in_place(config, state, latents);
      Ok(())
    }
  }
}
