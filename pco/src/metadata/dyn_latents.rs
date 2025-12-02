use crate::bit_reader::BitReader;
use crate::bit_writer::BitWriter;
use crate::data_types::Latent;
use crate::macros::{define_latent_enum, match_latent_enum};
use std::io::Write;

define_latent_enum!(
  #[derive(Clone, Debug, PartialEq, Eq)]
  pub DynLatents(Vec)
);

impl DynLatents {
  pub(crate) fn len(&self) -> usize {
    match_latent_enum!(
      self,
      DynLatents<T>(inner) => { inner.len() }
    )
  }

  pub(crate) unsafe fn read_uncompressed_from<L: Latent>(
    reader: &mut BitReader,
    len: usize,
  ) -> Self {
    let mut latents = Vec::with_capacity(len);
    for _ in 0..len {
      latents.push(reader.read_uint::<L>(L::BITS));
    }
    DynLatents::new(latents).unwrap()
  }

  pub(crate) unsafe fn write_uncompressed_to<W: Write>(&self, writer: &mut BitWriter<W>) {
    match_latent_enum!(
      &self,
      DynLatents<L>(inner) => {
        for &latent in inner {
          writer.write_uint(latent, L::BITS);
        }
      }
    );
  }
}
