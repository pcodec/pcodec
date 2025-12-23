use crate::bit_reader::BitReader;
use crate::bit_writer::BitWriter;
use crate::data_types::Latent;
use crate::macros::{define_latent_enum, match_latent_enum};
use std::io::Write;

type Single<L> = L;

define_latent_enum!(
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub DynLatent(Single)
);

impl DynLatent {
  pub(crate) unsafe fn read_uncompressed_from<L: Latent>(reader: &mut BitReader) -> Self {
    DynLatent::new(reader.read_uint::<L>(L::BITS)).unwrap()
  }

  pub(crate) unsafe fn write_uncompressed_to<W: Write>(&self, writer: &mut BitWriter<W>) {
    match_latent_enum!(
      &self,
      DynLatent<L>(latent) => {
        writer.write_uint(*latent, L::BITS);
      }
    );
  }
}
