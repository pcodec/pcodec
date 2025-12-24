
use crate::ans::AnsState;
use crate::bit_reader::BitReader;
use crate::bit_writer::BitWriter;
use crate::constants::{Bitlen, ANS_INTERLEAVING};
use crate::data_types::LatentType;
use crate::delta::DeltaState;
use crate::errors::PcoResult;
use crate::macros::match_latent_enum;
use crate::metadata::dyn_latents::DynLatents;
use std::io::Write;

#[derive(Clone, Debug)]
pub struct PageLatentVarMeta {
  pub delta_state: DeltaState,
  pub ans_final_state_idxs: [AnsState; ANS_INTERLEAVING],
}

impl PageLatentVarMeta {
  pub unsafe fn write_to<W: Write>(&self, ans_size_log: Bitlen, writer: &mut BitWriter<W>) {
    self.delta_state.write_uncompressed_to(writer);

    // write the final ANS state, moving it down the range [0, table_size)
    for state_idx in self.ans_final_state_idxs {
      writer.write_uint(state_idx, ans_size_log);
    }
  }

  pub unsafe fn read_from(
    reader: &mut BitReader,
    latent_type: LatentType,
    n_latents_per_delta_state: usize,
    ans_size_log: Bitlen,
  ) -> PcoResult<Self> {
    let delta_state = match_latent_enum!(
      latent_type,
      LatentType<L> => {
        DynLatents::read_short_uncompressed_from::<L>(reader, n_latents_per_delta_state)
      }
    )?;
    let mut ans_final_state_idxs = [0; ANS_INTERLEAVING];
    for state in &mut ans_final_state_idxs {
      *state = reader.read_uint::<AnsState>(ans_size_log);
    }

    Ok(Self {
      delta_state,
      ans_final_state_idxs,
    })
  }
}
