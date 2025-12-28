mod consecutive;
mod lookback;

use crate::constants::DeltaLookback;
use crate::data_types::Latent;
use crate::dyn_latent_slice::DynLatentSlice;
use crate::errors::{PcoError, PcoResult};
use crate::macros::match_latent_enum;
use crate::metadata::delta_encoding::LatentVarDeltaEncoding;
use crate::metadata::dyn_latents::DynLatents;
use crate::metadata::DeltaEncoding;
use std::ops::Range;

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

pub fn new_buffer_and_pos<L: Latent>(
  delta_encoding: &LatentVarDeltaEncoding,
  stored_delta_state: Vec<L>,
) -> (Vec<L>, usize) {
  match delta_encoding {
    LatentVarDeltaEncoding::NoOp | LatentVarDeltaEncoding::Consecutive(_) => {
      (stored_delta_state, 0)
    }
    LatentVarDeltaEncoding::Lookback(config) => {
      lookback::new_window_buffer_and_pos(*config, &stored_delta_state)
    }
  }
}

pub fn compute_delta_latent_var(
  delta_encoding: &DeltaEncoding,
  primary_latents: &mut DynLatents,
  range: Range<usize>,
) -> Option<DynLatents> {
  match delta_encoding {
    DeltaEncoding::NoOp | DeltaEncoding::Consecutive { .. } => None,
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

pub fn encode_in_place(
  delta_encoding: &LatentVarDeltaEncoding,
  delta_latents: Option<&DynLatents>,
  range: Range<usize>,
  latents: &mut DynLatents,
) -> DeltaState {
  match_latent_enum!(
    latents,
    DynLatents<L>(inner) => {
      let delta_state = match delta_encoding {
        LatentVarDeltaEncoding::NoOp => Vec::<L>::new(),
        LatentVarDeltaEncoding::Consecutive(order) => {
          consecutive::encode_in_place(*order, &mut inner[range])
        }
        LatentVarDeltaEncoding::Lookback(config) => {
          let lookbacks = delta_latents.unwrap().downcast_ref::<DeltaLookback>().unwrap();
          lookback::encode_in_place(*config, lookbacks, &mut inner[range])
        }
      };
      DynLatents::new(delta_state)
    }
  )
}

pub fn decode_in_place<L: Latent>(
  delta_encoding: &LatentVarDeltaEncoding,
  delta_latents: Option<DynLatentSlice>,
  delta_state_pos: &mut usize,
  delta_state: &mut [L],
  latents: &mut [L],
) -> PcoResult<()> {
  match delta_encoding {
    LatentVarDeltaEncoding::NoOp => Ok(()),
    LatentVarDeltaEncoding::Consecutive(_) => {
      consecutive::decode_in_place(delta_state, latents);
      Ok(())
    }
    LatentVarDeltaEncoding::Lookback(config) => {
      let has_oob_lookbacks = lookback::decode_in_place(
        *config,
        delta_latents.unwrap().downcast_unwrap::<DeltaLookback>(),
        delta_state_pos,
        delta_state,
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
  }
}
