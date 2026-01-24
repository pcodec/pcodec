use anyhow::{Result, anyhow};
use std::{marker::PhantomData, ops::Range};

use image::{DynamicImage, ImageBuffer, Rgb, RgbImage, RgbaImage};

#[derive(Clone)]
pub struct Img {
  pub h: usize,
  pub w: usize,
  pub c: usize,
  pub values: Vec<u8>,
}

impl Img {
  pub fn empty(h: usize, w: usize, c: usize) -> Self {
    Self {
      h,
      w,
      c,
      values: vec![0; h * w * c],
    }
  }

  pub fn iter_chunks<'a>(
    &'a self,
    chunk_h: usize,
    chunk_w: usize,
  ) -> impl Iterator<Item = ImgChunk<'a>> {
    let &Img { h, w, c, .. } = self;
    (0..h).step_by(chunk_h).flat_map(move |i0| {
      let i1 = (i0 + chunk_h).min(h);
      (0..w).step_by(chunk_w).map(move |j0| {
        let j1 = (j0 + chunk_w).min(w);
        let offset = c * (i0 * w + j0);
        ImgChunk {
          img: self,
          offset,
          h: i1 - i0,
          w: j1 - j0,
        }
      })
    })
  }

  pub fn iter_chunks_mut<'a>(
    &'a mut self,
    chunk_h: usize,
    chunk_w: usize,
  ) -> impl Iterator<Item = ImgChunkMut<'a>> {
    let &mut Img { h, w, c, .. } = self;
    let img = self as *mut Img;
    (0..h).step_by(chunk_h).flat_map(move |i0| {
      let i1 = (i0 + chunk_h).min(h);
      (0..w).step_by(chunk_w).map(move |j0| {
        let j1 = (j0 + chunk_w).min(w);
        let offset = c * (i0 * w + j0);
        ImgChunkMut {
          img,
          offset,
          h: i1 - i0,
          w: j1 - j0,
          _phantom: PhantomData,
        }
      })
    })
  }
}

impl From<ImageBuffer<Rgb<u8>, Vec<u8>>> for Img {
  fn from(value: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Self {
    Img {
      h: value.height() as usize,
      w: value.width() as usize,
      c: 3,
      values: value.into_raw(),
    }
  }
}

impl Into<DynamicImage> for Img {
  fn into(self) -> DynamicImage {
    let &Img { h, w, c, .. } = &self;
    if self.c == 3 {
      DynamicImage::ImageRgb8(RgbImage::from_raw(w as u32, h as u32, self.values).unwrap())
    } else if c == 4 {
      DynamicImage::ImageRgba8(RgbaImage::from_raw(w as u32, h as u32, self.values).unwrap())
    } else {
      panic!("unknown number of channels");
    }
  }
}

// struct ImgChunkIter<'a> {
//   img: &'a mut Img,
//   chunk_h: usize,
//   chunk_w: usize,
//   i: usize,
//   j: usize,
// }

// impl<'a> Iterator for ImgChunkIter<'a> {
//   type Item = ImgChunk<'a>;

//   fn next(&mut self) -> Option<Self::Item> {
//     if self.i >= self.img.h.div_ceil(self.chunk_h) {
//       return None;
//     }

//     let w = self.img.w;
//     let i0 = self.i * self.chunk_h;
//     let i1 = (i0 + self.chunk_h).min(self.img.h);
//     let j0 = self.j * self.chunk_w;
//     let j1 = (j0 + self.chunk_w).min(w);

//     let chunk = self.img.slice(i0..i1, j0..j1);

//     if self.j >= w.div_ceil(self.chunk_w) {
//       self.j = 0;
//       self.i += 1;
//     } else {
//       self.j += 1;
//     }

//     Some(chunk)
//   }
// }

pub struct ImgChunk<'a> {
  img: &'a Img,
  offset: usize,
  h: usize,
  w: usize,
}

impl<'a> ImgChunk<'a> {
  pub fn n(&self) -> usize {
    self.h * self.w
  }

  pub fn write_flat(&self, k: usize, dst: &mut [u8]) {
    let &ImgChunk { offset, h, w, .. } = self;
    let img_w = self.img.w;
    let c = self.img.c;
    for i in 0..h {
      for j in 0..w {
        dst[i * w + j] = self.img.values[offset + k + c * (i * img_w + j)];
      }
    }
  }
}

pub struct ImgChunkMut<'a> {
  img: *mut Img,
  offset: usize,
  h: usize,
  w: usize,
  _phantom: PhantomData<&'a mut Img>,
}

impl<'a> ImgChunkMut<'a> {
  pub fn n(&self) -> usize {
    self.h * self.w
  }

  pub fn read_flat(&self, k: usize, src: &[u8]) {
    let &ImgChunkMut { offset, h, w, .. } = self;
    unsafe {
      let img = &mut *self.img;
      let img_w = img.w;
      let c = img.c;
      for i in 0..h {
        for j in 0..w {
          img.values[offset + k + c * (i * img_w + j)] = src[i * w + j];
        }
      }
    }
  }
}
