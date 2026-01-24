use anyhow::anyhow;
use std::{
  env,
  fs::{self, OpenOptions},
  io::{self, Cursor, Read, Write},
};

mod header_generated;
use anyhow::Result;
use better_io::{BetterBufRead, BetterBufReader};
use flatbuffers::{FlatBufferBuilder, SIZE_SIZEPREFIX};
use image::{DynamicImage, ImageBuffer, ImageFormat, ImageReader, Rgb, RgbImage, RgbaImage};
use pco::{
  ChunkConfig, ModeSpec,
  standalone::{DecompressorItem, FileCompressor, FileDecompressor},
};

use crate::header_generated::pcodec::naegling::{self, Header, HeaderArgs, HeaderBuilder};

struct Img {
  h: u32,
  w: u32,
  c: u32,
  values: Vec<u8>,
}

impl From<ImageBuffer<Rgb<u8>, Vec<u8>>> for Img {
  fn from(value: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Self {
    Img {
      h: value.height(),
      w: value.width(),
      c: 3,
      values: value.into_raw(),
    }
  }
}

struct Config {
  chunk_h: u32,
  chunk_w: u32,
}

fn write_naegling_into<W: Write>(img: &Img, config: &Config, mut dst: W) -> Result<()> {
  let h = img.h;
  let w = img.w;
  let c = img.c;
  let &Config { chunk_h, chunk_w } = config;
  let mut fbb = FlatBufferBuilder::with_capacity(1024);
  let header = Header::create(
    &mut fbb,
    &HeaderArgs {
      h,
      w,
      c,
      chunk_h,
      chunk_w,
    },
  );
  fbb.finish_size_prefixed(header, None);
  dst.write(fbb.finished_data())?;

  let config = ChunkConfig::default()
    .with_enable_8_bit(true)
    .with_mode_spec(ModeSpec::Classic);

  let fc = FileCompressor::default();
  fc.write_header(&mut dst)?;
  let chunk_n = (h * w) as usize;
  let mut buf = vec![0; chunk_n];
  for chunk_i in 0..h.div_ceil(chunk_h) {
    for chunk_j in 0..w.div_ceil(chunk_w) {
      let i0 = chunk_i * chunk_h;
      let j0 = chunk_j * chunk_w;
      let i1 = (i0 + chunk_h).min(h);
      let j1 = (j0 + chunk_w).min(w);
      let real_w = j1 - j0;
      for k in 0..c {
        for i in i0..i1 {
          for j in j0..j1 {
            buf[((i - i0) * real_w + (j - j0)) as usize] =
              img.values[(k + c * (i * w + j)) as usize]
          }
        }
        fc.chunk_compressor(
          &buf[..((i1 - i0) * real_w) as usize],
          &config,
        )?
        .write(&mut dst)?;
      }
    }
  }
  fc.write_footer(dst)?;
  Ok(())
}

fn read_naegling_from(src: &[u8]) -> Result<DynamicImage> {
  let header_size = u32::from_le_bytes(src[..SIZE_SIZEPREFIX].try_into().unwrap()) as usize;
  let header = naegling::size_prefixed_root_as_header(src).unwrap();
  println!("{:?}", header);
  let h = header.h() as usize;
  let w = header.w() as usize;
  let c = header.c() as usize;
  let chunk_h = header.chunk_h() as usize;
  let chunk_w = header.chunk_w() as usize;
  let src = BetterBufReader::new(
    &[],
    &src[SIZE_SIZEPREFIX + header_size..],
    1 << 15,
  );
  let chunk_n = h * w;
  let mut data = vec![0; chunk_n * c];
  let (fd, mut src) = FileDecompressor::new(src)?;
  let mut buf = vec![0; chunk_n];
  for chunk_i in 0..h.div_ceil(chunk_h) {
    for chunk_j in 0..w.div_ceil(chunk_w) {
      let i0 = chunk_i * chunk_h;
      let j0 = chunk_j * chunk_w;
      let i1 = (i0 + chunk_h).min(h);
      let j1 = (j0 + chunk_w).min(w);
      for k in 0..c as usize {
        let DecompressorItem::Chunk(mut cd) = fd.chunk_decompressor(src)? else {
          return Err(anyhow!(
            "ran out of pco chunks before expected"
          ));
        };
        let expected_n = (i1 - i0) * (j1 - j0);
        assert!(cd.n() == expected_n);
        let progress = cd.read(&mut buf[..expected_n])?;
        assert!(progress.n_processed == expected_n);
        assert!(progress.finished);
        for i in i0..i1 {
          for j in j0..j1 {
            data[k + c * (i * w + j)] = buf[(i - i0) * (j1 - j0) + j - j0];
          }
        }
        src = cd.into_src();
      }
    }
  }
  let img = if c == 3 {
    DynamicImage::ImageRgb8(RgbImage::from_raw(w as u32, h as u32, data).unwrap())
  } else if c == 4 {
    DynamicImage::ImageRgba8(RgbaImage::from_raw(w as u32, h as u32, data).unwrap())
  } else {
    return Err(anyhow!("unknown number of channels"));
  };
  Ok(img)
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
