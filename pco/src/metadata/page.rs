use std::io::Write;

use crate::bit_reader::BitReader;
use crate::bit_writer::BitWriter;
use crate::constants::Bitlen;
use crate::errors::PcoResult;
use crate::metadata::page_latent_var::PageLatentVarMeta;
use crate::metadata::per_latent_var::PerLatentVar;
use crate::metadata::ChunkMeta;

// Data page metadata is slightly semantically different from chunk metadata,
// so it gets its own type.
// Importantly, `n` and `compressed_body_size` might come from either the
// chunk metadata parsing step (standalone mode) OR from the wrapping format
// (wrapped mode).
#[derive(Clone, Debug)]
pub struct PageMeta {
  pub per_latent_var: PerLatentVar<PageLatentVarMeta>,
}

impl PageMeta {
  pub unsafe fn write_to<W: Write>(
    &self,
    ans_size_logs: PerLatentVar<Bitlen>,
    writer: &mut BitWriter<W>,
  ) {
    for (_, (ans_size_log, latent_var_meta)) in ans_size_logs
      .zip_exact(self.per_latent_var.as_ref())
      .enumerated()
    {
      latent_var_meta.write_to(ans_size_log, writer);
    }
    writer.finish_byte();
  }

  pub unsafe fn read_from(reader: &mut BitReader, chunk_meta: &ChunkMeta) -> PcoResult<Self> {
    let per_latent_var = chunk_meta
      .per_latent_var
      .as_ref()
      .map(|key, chunk_latent_var_meta| {
        let n_latents_per_state = chunk_meta
          .delta_encoding
          .for_latent_var(key)
          .n_latents_per_state();
        PageLatentVarMeta::read_from(
          reader,
          chunk_latent_var_meta.latent_type(),
          n_latents_per_state,
          chunk_latent_var_meta.ans_size_log,
        )
      });
    reader.drain_empty_byte("non-zero bits at end of data page metadata")?;

    Ok(Self { per_latent_var })
  }
}
