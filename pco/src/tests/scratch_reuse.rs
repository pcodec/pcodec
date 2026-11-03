use crate::{
  constants::Bitlen,
  data_types::Number,
  errors::PcoResult,
  metadata::{ChunkMeta, DynBins},
  wrapped::{FileCompressor, FileDecompressor},
  ChunkConfig, DecompressorScratch, DeltaSpec, ModeSpec,
};

use crate::macros::match_latent_enum;

fn round_trip<T: Number>(
  fc: &FileCompressor,
  fd: &FileDecompressor,
  chunk: Vec<T>,
  scratch: &mut DecompressorScratch,
) -> PcoResult<ChunkMeta> {
  let config = ChunkConfig::default()
    .with_delta_spec(DeltaSpec::NoOp)
    .with_mode_spec(ModeSpec::Classic);
  let mut cc = fc.chunk_compressor(&chunk, &config)?;
  let mut metas = vec![];
  let mut pages = vec![];

  let mut meta = vec![];
  let mut page = vec![];
  cc.write_meta(&mut meta)?;
  cc.write_page(0, &mut page)?;
  metas.push(meta);
  pages.push(page);

  let (cd, _) = fd.chunk_decompressor::<T, _>(metas[0].as_slice())?;
  let mut pd = cd.page_decompressor_with_scratch(pages[0].as_slice(), chunk.len(), scratch)?;
  let mut dst = vec![T::default(); chunk.len()];
  pd.read(&mut dst)?;
  assert_eq!(dst, chunk);

  Ok(cc.meta().clone())
}

fn assert_single_bin_offset_bits(meta: &ChunkMeta, expected: Option<Bitlen>) {
  let single_bin_offset_bits = match_latent_enum!(
    &meta.per_latent_var.primary.bins,
    DynBins<L>(bins) => {
      if bins.len() == 1 {
        Some(bins[0].offset_bits)
      } else {
        None
      }
    }
  );
  assert_eq!(single_bin_offset_bits, expected);
}

#[test]
fn test_scratch_reuse_for_different_data_types_and_bins() -> PcoResult<()> {
  let fc = FileCompressor::default();
  let mut scratch = DecompressorScratch::default();

  let mut header = vec![];
  fc.write_header(&mut header)?;
  let (fd, _) = FileDecompressor::new(header.as_slice())?;

  let chunk = (0_u64..8).collect::<Vec<_>>().repeat(100);
  let meta = round_trip(&fc, &fd, chunk, &mut scratch)?;
  assert_single_bin_offset_bits(&meta, Some(3));

  let chunk = (70_u32..86).collect::<Vec<_>>().repeat(50);
  let meta = round_trip(&fc, &fd, chunk, &mut scratch)?;
  assert_single_bin_offset_bits(&meta, Some(4));

  let chunk = vec![0_u16, 999].repeat(400);
  let meta = round_trip(&fc, &fd, chunk, &mut scratch)?;
  assert_single_bin_offset_bits(&meta, None);

  let chunk = (70_u32..86).collect::<Vec<_>>().repeat(50);
  let meta = round_trip(&fc, &fd, chunk, &mut scratch)?;
  assert_single_bin_offset_bits(&meta, Some(4));

  let chunk = vec![0_u16, 999].repeat(400);
  let meta = round_trip(&fc, &fd, chunk, &mut scratch)?;
  assert_single_bin_offset_bits(&meta, None);

  let chunk = vec![101_u64, 102].repeat(400);
  let meta = round_trip(&fc, &fd, chunk, &mut scratch)?;
  assert_single_bin_offset_bits(&meta, Some(1));

  let chunk = vec![101_u64].repeat(800);
  let meta = round_trip(&fc, &fd, chunk, &mut scratch)?;
  assert_single_bin_offset_bits(&meta, Some(0));
  Ok(())
}
