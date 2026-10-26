use crate::data_types::Number;
use crate::errors::PcoResult;
use crate::metadata::Mode;
use crate::wrapped::{ChunkDecompressor, FileCompressor, FileDecompressor};
use crate::{ChunkConfig, DeltaSpec, ModeSpec, PagingSpec};

fn compressed_page<T: Number>(
  values: &[T],
  mode_spec: ModeSpec,
) -> PcoResult<(ChunkDecompressor<T>, Vec<u8>)> {
  let fc = FileCompressor::default();
  let config = ChunkConfig {
    mode_spec,
    delta_spec: DeltaSpec::NoOp,
    paging_spec: PagingSpec::Exact(vec![values.len()]),
    ..Default::default()
  };
  let mut cc = fc.chunk_compressor(values, &config)?;
  let header = fc.write_header(Vec::new())?;
  let metadata = cc.write_meta(Vec::new())?;
  let page = cc.write_page(0, Vec::new())?;
  let (fd, _) = FileDecompressor::new(header.as_slice())?;
  let (cd, _) = fd.chunk_decompressor(metadata.as_slice())?;
  Ok((cd, page))
}

#[test]
fn test_read_uninit_batches_and_tail() -> PcoResult<()> {
  for mode in [
    ModeSpec::Classic,
    ModeSpec::TryIntMult(7),
    ModeSpec::TryDict,
  ] {
    let values: Vec<i32> = (0..257).map(|i| (i % 17) * 7).collect();
    let (mut cd, page) = compressed_page(&values, mode)?;
    let mut pd = cd.page_decompressor(page.as_slice(), values.len())?;
    let mut output = Vec::<i32>::with_capacity(300);

    let progress = pd.read_uninit(&mut output.spare_capacity_mut()[..256])?;
    assert_eq!(progress.n_processed, 256);
    assert!(!progress.finished);
    // SAFETY: read_uninit initialized this prefix on success.
    unsafe { output.set_len(progress.n_processed) };

    let old_len = output.len();
    let capacity = output.spare_capacity_mut().len();
    let progress = pd.read_uninit(output.spare_capacity_mut())?;
    assert_eq!(progress.n_processed, 1);
    assert!(progress.n_processed <= capacity);
    assert!(progress.finished);
    // SAFETY: only the successfully initialized prefix is exposed.
    unsafe { output.set_len(old_len + progress.n_processed) };
    assert_eq!(output, values);
  }
  Ok(())
}

#[test]
fn test_read_uninit_oversized_destination_preserves_prefix() -> PcoResult<()> {
  let (mut cd, page) = compressed_page(&[11i32, 22, 33], ModeSpec::Classic)?;
  let mut pd = cd.page_decompressor(page.as_slice(), 3)?;
  let mut output = Vec::with_capacity(32);
  output.push(-1);
  let old_len = output.len();
  let progress = pd.read_uninit(output.spare_capacity_mut())?;
  assert_eq!(progress.n_processed, 3);
  assert!(progress.finished);
  // SAFETY: the successful call initialized exactly the appended prefix.
  unsafe { output.set_len(old_len + progress.n_processed) };
  assert_eq!(output, [-1, 11, 22, 33]);
  Ok(())
}

#[test]
fn test_read_uninit_float_bits() -> PcoResult<()> {
  let values = [
    0.0f64,
    -0.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::from_bits(0x7ff8000000000001),
  ];
  let (mut cd, page) = compressed_page(&values, ModeSpec::Classic)?;
  let mut pd = cd.page_decompressor(page.as_slice(), values.len())?;
  let mut output = Vec::<f64>::with_capacity(values.len());
  let progress = pd.read_uninit(output.spare_capacity_mut())?;
  assert_eq!(progress.n_processed, values.len());
  assert!(progress.finished);
  // SAFETY: all returned elements were initialized by the successful read.
  unsafe { output.set_len(progress.n_processed) };
  assert_eq!(
    output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    values.map(f64::to_bits)
  );
  Ok(())
}

#[test]
fn test_read_uninit_empty_destination() -> PcoResult<()> {
  let (mut cd, page) = compressed_page(&[19i32], ModeSpec::Classic)?;
  let mut pd = cd.page_decompressor(page.as_slice(), 1)?;
  let progress = pd.read_uninit(&mut [])?;
  assert_eq!(progress.n_processed, 0);
  assert!(!progress.finished);
  let mut output = Vec::<i32>::with_capacity(1);
  let progress = pd.read_uninit(output.spare_capacity_mut())?;
  assert_eq!(progress.n_processed, 1);
  assert!(progress.finished);
  // SAFETY: the successful read initialized the returned prefix.
  unsafe { output.set_len(progress.n_processed) };
  assert_eq!(output, [19]);
  let progress = pd.read_uninit(&mut [])?;
  assert_eq!(progress.n_processed, 0);
  assert!(progress.finished);
  Ok(())
}

#[test]
fn test_read_uninit_error_keeps_existing_prefix() -> PcoResult<()> {
  let values: Vec<u32> = (0..1024u32).map(|i| i.wrapping_mul(0x9e3779b9)).collect();
  let (mut cd, page) = compressed_page(&values, ModeSpec::Classic)?;
  let mut pd = cd.page_decompressor(&page[..page.len() - 1], values.len())?;
  let mut output = Vec::with_capacity(values.len() + 1);
  output.push(123u32);
  let old_len = output.len();
  let result = pd.read_uninit(output.spare_capacity_mut());
  assert!(result.is_err());
  // No returned initialized prefix: do not change the vector's length.
  assert_eq!(output.len(), old_len);
  assert_eq!(output, [123]);
  Ok(())
}

#[test]
fn test_read_uninit_float_modes() -> PcoResult<()> {
  let values: Vec<f64> = (0..2048u32)
    .map(|i| f64::from(i.wrapping_mul(2654435761) % 1000000) * 0.25)
    .collect();
  let expected_bits: Vec<u64> = values.iter().map(|v| v.to_bits()).collect();
  for mode in [ModeSpec::TryFloatMult(0.25), ModeSpec::TryFloatQuant(20)] {
    let mult = matches!(mode, ModeSpec::TryFloatMult(_));
    let (mut cd, page) = compressed_page(&values, mode)?;
    // These public metadata checks ensure both output modes are exercised.
    assert!(if mult {
      matches!(&cd.meta().mode, Mode::FloatMult(_))
    } else {
      matches!(&cd.meta().mode, Mode::FloatQuant(_))
    });
    let mut pd = cd.page_decompressor(page.as_slice(), values.len())?;
    let mut output = Vec::<f64>::with_capacity(values.len());
    let progress = pd.read_uninit(output.spare_capacity_mut())?;
    assert_eq!(progress.n_processed, values.len());
    assert!(progress.finished);
    // SAFETY: the successful read initialized exactly this prefix.
    unsafe { output.set_len(progress.n_processed) };
    assert_eq!(
      output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
      expected_bits
    );
  }
  Ok(())
}
