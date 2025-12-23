use better_io::BetterBufRead;

use crate::bit_reader::{BitReader, BitReaderBuilder};
use crate::bit_writer::BitWriter;
use crate::data_types::Latent;
use crate::errors::PcoResult;
use crate::macros::{define_latent_enum, match_latent_enum};
use std::io::Write;

define_latent_enum!(
  #[derive(Clone, Debug, PartialEq, Eq)]
  pub DynLatents(Vec)
);

unsafe fn read<L: Latent>(reader: &mut BitReader, dst: &mut [L]) {
  for x in dst.iter_mut() {
    *x = reader.read_uint::<L>(L::BITS);
  }
}

impl DynLatents {
  pub(crate) fn len(&self) -> usize {
    match_latent_enum!(
      self,
      DynLatents<T>(inner) => { inner.len() }
    )
  }

  pub(crate) fn bit_size(&self) -> usize {
    match_latent_enum!(
      self,
      DynLatents<T>(inner) => { inner.len() * T::BITS as usize}
    )
  }

  pub(crate) fn read_uncompressed_in_place<R: BetterBufRead>(
    &mut self,
    reader_builder: &mut BitReaderBuilder<R>,
    n_per_reader: usize,
  ) -> PcoResult<()> {
    match_latent_enum!(
      self,
      DynLatents<L>(inner) => {
        let n = inner.len();
        for start in (0..n).step_by(n_per_reader) {
          reader_builder.with_reader(|reader| unsafe {
            read(reader, &mut inner[start..(start + n_per_reader).min(n)]);
            Ok(())
          })?;
        }
      }
    );

    Ok(())
  }

  pub(crate) unsafe fn read_uncompressed_from<L: Latent>(reader: &mut BitReader, n: usize) -> Self {
    let mut uninitialized = Vec::<L>::with_capacity(n);
    uninitialized.set_len(n);
    read(reader, &mut uninitialized);
    Self::new(uninitialized)
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
