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

  pub(crate) unsafe fn read_uncompressed_in_place<R: BetterBufRead>(
    &mut self,
    reader_builder: &mut BitReaderBuilder<R>,
    n_per_reader: usize,
  ) -> PcoResult<()> {
    match_latent_enum!(
      self,
      DynLatents<L>(inner) => {
        let n = inner.len();
        for reader_i in (0..n).step_by(n_per_reader) {
          reader_builder.with_reader(|reader| {
            for i in reader_i * n_per_reader..((reader_i + 1) * n_per_reader).min(n) {
              inner[i] = reader.read_uint::<L>(L::BITS);
            }
            Ok(())
          })?;
        }
      }
    );

    Ok(())
  }

  pub(crate) unsafe fn read_uncompressed_from<L: Latent, R: BetterBufRead>(
    &mut self,
    reader_builder: &mut BitReaderBuilder<R>,
    n: usize,
    n_per_reader: usize,
  ) -> PcoResult<Self> {
    let mut uninitialized = Vec::<L>::with_capacity(n);
    uninitialized.set_len(n);
    let mut res = Self::new(uninitialized);
    res.read_uncompressed_in_place(reader_builder, n_per_reader)?;
    Ok(res)
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
