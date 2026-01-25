mod color;
mod conv;
mod format_generated;
mod img;

use crate::{
  format_generated::pcodec::naegling::{self, Nae, NaeArgs, NaeChunk, NaeChunkArgs},
  img::Img,
};
use anyhow::Result;
use clap::Parser;
use flatbuffers::FlatBufferBuilder;
use image::{DynamicImage, ImageFormat, ImageReader};
use pco::{
  ChunkConfig, DeltaSpec, ModeSpec,
  wrapped::{FileCompressor, FileDecompressor},
};
use std::{
  env,
  fs::{self, OpenOptions},
  io::Write,
  path::PathBuf,
};

#[derive(Parser)]
struct Opt {
  src: PathBuf,
  dst: PathBuf,
  #[clap(long, default_value = "128")]
  chunk_h: usize,
  #[clap(long, default_value = "128")]
  chunk_w: usize,
  #[clap(long, default_value_t = pco::DEFAULT_COMPRESSION_LEVEL)]
  level: usize,
}

fn write_naegling_into<W: Write>(img: &Img, opt: &Opt, mut dst: W) -> Result<()> {
  let h = img.h;
  let w = img.w;
  let c = img.c;
  let &Opt {
    chunk_h,
    chunk_w,
    level,
    ..
  } = opt;
  let config = ChunkConfig::default()
    .with_enable_8_bit(true)
    .with_mode_spec(ModeSpec::Classic)
    .with_compression_level(level)
    .with_delta_spec(DeltaSpec::NoOp);

  let fc = FileCompressor::default();
  let mut pco_header = Vec::new();
  fc.write_header(&mut pco_header)?;
  let chunk_n = (h * w) as usize;
  let nh = h.div_ceil(chunk_h);
  let nw = w.div_ceil(chunk_w);
  let mut chunks = Vec::with_capacity(nh * nw);
  let mut fbb = FlatBufferBuilder::with_capacity(1024);
  let mut buf = vec![0; chunk_n];
  let mut scratch_a = Img::empty(chunk_h, chunk_w, c);
  let mut scratch_b = Img::empty(chunk_h, chunk_w, c);
  for chunk in img.iter_chunks(chunk_h, chunk_w) {
    // 1. Convert RGB to YCbCr
    // let scratch_ycbcr_view = scratch_a.as_view_mut(chunk.h, chunk.w);
    // color::view_rgb_to_ycbcr(&chunk, &scratch_ycbcr_view);
    // let scratch_ycbcr_view = scratch_a.as_view(chunk.h, chunk.w);

    // 2. 2x2 causal convolution (predict from neighbors + previous channels at current pixel)
    // let fit = conv::fit(&scratch_ycbcr_view);
    // let scratch_pred_ycbcr_view = scratch_b.as_view_mut(chunk.h, chunk.w);
    // conv::predict(
    //   &fit,
    //   &scratch_ycbcr_view,
    //   &scratch_pred_ycbcr_view,
    // );
    let fit = conv::fit(&chunk);
    let scratch_pred_rgb_view = scratch_b.as_view_mut(chunk.h, chunk.w);
    conv::predict(&fit, &chunk, &scratch_pred_rgb_view);

    // 3. Convert predicted YCbCr back to RGB
    // let scratch_pred_ycbcr_view = scratch_b.as_view(chunk.h, chunk.w);
    // let scratch_pred_rgb_view = scratch_a.as_view_mut(chunk.h, chunk.w);
    // color::view_ycbcr_to_rgb(
    //   &scratch_pred_ycbcr_view,
    //   &scratch_pred_rgb_view,
    // );

    // 4. Compute residuals in RGB space: original - predicted + 128
    scratch_pred_rgb_view.binary_op_in_place(&chunk, |pred, orig| {
      orig.wrapping_sub(pred).wrapping_add(128)
    });
    let residual_view = scratch_pred_rgb_view;
    let chunk_n = chunk.n();
    for k in 0..c {
      residual_view.write_flat(k, &mut buf[..chunk_n]);
      let mut cc = fc.chunk_compressor(&buf[..chunk_n], &config)?;
      assert!(cc.n_per_page().len() == 1);
      let mut meta = Vec::with_capacity(cc.meta_size_hint());
      cc.write_meta(&mut meta)?;
      let mut page = Vec::with_capacity(cc.page_size_hint(0));
      cc.write_page(0, &mut page)?;
      println!(
        "Writing chunk {}/{} channel {}/{} size {}",
        chunks.len(),
        nh * nw * c,
        k,
        c,
        page.len()
      );

      let chunk_args = NaeChunkArgs {
        quantization: 0,
        weights: None,
        biases: None,
        pco_meta: Some(fbb.create_vector(&meta)),
        pco_page: Some(fbb.create_vector(&page)),
      };
      chunks.push(NaeChunk::create(&mut fbb, &chunk_args));
    }
  }

  let args = NaeArgs {
    h: h as u32,
    w: w as u32,
    c: c as u32,
    chunk_h: chunk_h as u32,
    chunk_w: chunk_w as u32,
    pco_header: Some(fbb.create_vector(&pco_header)),
    chunks: Some(fbb.create_vector(&chunks)),
  };
  let root = Nae::create(&mut fbb, &args);
  fbb.finish(root, None);
  dst.write(fbb.finished_data())?;

  Ok(())
}

fn read_naegling_from(src: &[u8]) -> Result<DynamicImage> {
  let header = naegling::root_as_nae(src).unwrap();
  let h = header.h() as usize;
  let w = header.w() as usize;
  let c = header.c() as usize;
  let chunk_h = header.chunk_h() as usize;
  let chunk_w = header.chunk_w() as usize;
  let (fd, _) = FileDecompressor::new(header.pco_header().unwrap().bytes())?;
  let mut buf = vec![0; chunk_h * chunk_w];
  let mut img = Img::empty(h, w, c);
  let mut channel_chunk_idx = 0;
  let compressed_chunks = header.chunks().unwrap();
  for img_chunk in img.iter_chunks_mut(chunk_h, chunk_w) {
    let chunk_n = img_chunk.n();
    for k in 0..c {
      let compressed_chunk = &compressed_chunks.get(channel_chunk_idx);
      let (mut cd, _) = fd.chunk_decompressor(compressed_chunk.pco_meta().unwrap().bytes())?;
      let page_bytes = compressed_chunk.pco_page().unwrap().bytes();
      let mut pd = cd.page_decompressor(page_bytes, chunk_n)?;
      let progress = pd.read(&mut buf[..chunk_n])?;
      assert!(progress.finished);
      img_chunk.read_flat(k, &buf[..chunk_n]);
      channel_chunk_idx += 1;
    }
  }

  Ok(img.into())
}

fn main() -> Result<()> {
  let opt = Opt::parse();
  let src_path = &opt.src;
  let dst_path = &opt.dst;
  println!(
    "Reading from {:?}, writing to {:?}",
    src_path, dst_path
  );

  let ext = src_path.extension().and_then(|e| e.to_str());
  let img = if matches!(ext, Some("nae")) {
    let src = fs::read(src_path)?;
    read_naegling_from(&src)?
  } else {
    ImageReader::open(src_path)?.decode()?
  };

  let dst = OpenOptions::new()
    .create(true)
    .truncate(true)
    .write(true)
    .open(dst_path)?;

  let ext = dst_path.extension().and_then(|e| e.to_str());
  if matches!(ext, Some("nae")) {
    let img: Img = match img {
      DynamicImage::ImageRgb8(img) => img.into(),
      _ => panic!("not prepared for this input image format"),
    };
    write_naegling_into(&img.into(), &opt, dst)?;
  } else if matches!(ext, Some("webp")) {
    img.write_to(dst, ImageFormat::WebP)?;
  } else {
    panic!();
  }

  Ok(())
}
