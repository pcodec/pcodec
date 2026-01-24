use anyhow::anyhow;
use std::{
  env,
  fs::{self, OpenOptions},
  io::{self, Cursor, Read, Write},
  ops::Range,
};

mod format_generated;
mod img;
use anyhow::Result;
use better_io::{BetterBufRead, BetterBufReader};
use flatbuffers::{FlatBufferBuilder, SIZE_SIZEPREFIX};
use image::{DynamicImage, ImageBuffer, ImageFormat, ImageReader, Rgb, RgbImage, RgbaImage};
use pco::{
  ChunkConfig, ModeSpec,
  wrapped::{FileCompressor, FileDecompressor},
};

use crate::{
  format_generated::pcodec::naegling::{
    self, Nae, NaeArgs, NaeBuilder, NaeChunk, NaeChunkArgs, NaeChunkBuilder,
  },
  img::Img,
};

// channel-major
struct ColorSpaceFit {
  quantization: u32,
  weights: Vec<i16>,
  biases: Vec<i16>,
}

// fn decorrelate(s: &ImgSlice) -> ColorSpaceFit {
//   let c = s.img.c;
//   let mut weights = vec![0; (c * (c - 1)) / 2];
//   let mut biases = vec![0; c - 1];

//   let mut weight_idx = 0;
//   for resp in 1..c {
//     for pred in 0..resp {
//       weights[weight_idx] = fit[pred];
//       weight_idx += 1;
//     }
//   }
//   (weights, biases)
// }

struct Config {
  chunk_h: usize,
  chunk_w: usize,
}

fn write_naegling_into<W: Write>(img: &Img, config: &Config, mut dst: W) -> Result<()> {
  let h = img.h;
  let w = img.w;
  let c = img.c;
  let &Config { chunk_h, chunk_w } = config;
  let config = ChunkConfig::default()
    .with_enable_8_bit(true)
    .with_mode_spec(ModeSpec::Classic);

  let fc = FileCompressor::default();
  let mut pco_header = Vec::new();
  fc.write_header(&mut pco_header)?;
  let chunk_n = (h * w) as usize;
  let nh = h.div_ceil(chunk_h);
  let nw = w.div_ceil(chunk_w);
  let mut chunks = Vec::with_capacity(nh * nw);
  let mut fbb = FlatBufferBuilder::with_capacity(1024);
  let mut buf = vec![0; chunk_n];
  for chunk in img.iter_chunks(chunk_h, chunk_w) {
    let chunk_n = chunk.n();
    for k in 0..c {
      chunk.write_flat(k, &mut buf[..chunk_n]);
      let mut cc = fc.chunk_compressor(&buf[..chunk_n], &config)?;
      assert!(cc.n_per_page().len() == 1);
      let mut meta = Vec::with_capacity(cc.meta_size_hint());
      cc.write_meta(&mut meta)?;
      let mut page = Vec::with_capacity(cc.page_size_hint(0));
      cc.write_page(0, &mut page)?;

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
  let args = env::args().collect::<Vec<_>>();
  let src_path = &args[1];
  let dst_path = &args[2];
  let config = &Config {
    chunk_h: 128,
    chunk_w: 128,
  };

  let img = if src_path.ends_with(".nae") {
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

  if dst_path.ends_with(".nae") {
    let img: Img = match img {
      DynamicImage::ImageRgb8(img) => img.into(),
      _ => panic!("not prepared for this input image format"),
    };
    write_naegling_into(&img.into(), config, dst)?;
  } else if dst_path.ends_with(".webp") {
    img.write_to(dst, ImageFormat::WebP)?;
  } else {
    panic!();
  }

  Ok(())
}
