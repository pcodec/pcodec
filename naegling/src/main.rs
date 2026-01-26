mod conv;
mod format_generated;
mod img;

use crate::{
  conv::ConvFit,
  format_generated::pcodec::naegling::{self, Nae, NaeArgs, NaeChunkChannel, NaeChunkChannelArgs},
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
  let mut scratch_img = Img::empty(chunk_h, chunk_w, c);
  let mut meta = vec![];
  let mut page = vec![];
  for chunk in img.iter_chunks(chunk_h, chunk_w) {
    // 1. Compute causal 2D conv fit
    let fits = conv::fit_conv(&chunk);
    let pred_view = scratch_img.as_view_mut(chunk.h, chunk.w);
    conv::predict(&fits, &chunk, &pred_view);

    // 2. Compute residuals in RGB space
    pred_view.binary_op_in_place(&chunk, |pred, orig| {
      orig.wrapping_sub(pred).wrapping_add(128)
    });
    let centered_residual_view = pred_view;
    let chunk_n = chunk.n();
    for k in 0..c {
      centered_residual_view.write_channel_flat(k, &mut buf[..chunk_n]);
      let mut cc = fc.chunk_compressor(&buf[..chunk_n], &config)?;
      assert!(cc.n_per_page().len() == 1);
      cc.write_meta(&mut meta)?;
      cc.write_page(0, &mut page)?;

      let fit = &fits[k];
      let chunk_args = NaeChunkChannelArgs {
        top_left: fit.top_left,
        quantization: fit.quantization,
        weights: Some(fbb.create_vector(&fit.weights_i16())),
        bias: fit.bias,
        pco_meta: Some(fbb.create_vector(&meta)),
        pco_page: Some(fbb.create_vector(&page)),
      };
      chunks.push(NaeChunkChannel::create(
        &mut fbb,
        &chunk_args,
      ));
      meta.clear();
      page.clear();
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
  let nae = naegling::root_as_nae(src).unwrap();
  let h = nae.h() as usize;
  let w = nae.w() as usize;
  let c = nae.c() as usize;
  let chunk_h = nae.chunk_h() as usize;
  let chunk_w = nae.chunk_w() as usize;
  let (fd, _) = FileDecompressor::new(nae.pco_header().unwrap().bytes())?;
  let mut buf = vec![0_u8; chunk_h * chunk_w];
  let mut img = Img::empty(h, w, c);
  let mut channel_chunk_idx = 0;
  let compressed_chunks = nae.chunks().unwrap();
  for dst_view in img.iter_chunks_mut(chunk_h, chunk_w) {
    let chunk_n = dst_view.n();
    let mut fits = Vec::with_capacity(c);
    for k in 0..c {
      let compressed_chunk = &compressed_chunks.get(channel_chunk_idx);
      let (mut cd, _) = fd.chunk_decompressor(compressed_chunk.pco_meta().unwrap().bytes())?;
      let page_bytes = compressed_chunk.pco_page().unwrap().bytes();
      let mut pd = cd.page_decompressor(page_bytes, chunk_n)?;
      let progress = pd.read(&mut buf[..chunk_n])?;
      assert!(progress.finished);
      for centered_residual in buf[..chunk_n].iter_mut() {
        *centered_residual = centered_residual.wrapping_sub(128);
      }
      dst_view.read_channel_flat(k, &buf[..chunk_n]);
      fits.push(ConvFit {
        top_left: compressed_chunk.top_left(),
        quantization: compressed_chunk.quantization(),
        weights: compressed_chunk
          .weights()
          .unwrap()
          .iter()
          .map(|x| x as i32)
          .collect(),
        bias: compressed_chunk.bias(),
      });
      channel_chunk_idx += 1;
    }

    // now dst_view is residuals
    conv::decode(&fits, &dst_view);
  }

  Ok(img.into())
}

fn main() -> Result<()> {
  let opt = Opt::parse();
  let src_path = &opt.src;
  let dst_path = &opt.dst;

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
