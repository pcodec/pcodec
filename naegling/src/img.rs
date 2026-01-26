use std::marker::PhantomData;

use image::{DynamicImage, RgbImage, RgbaImage};

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

  pub fn as_view_mut<'a>(&'a mut self, h: usize, w: usize) -> ImgViewMut<'a> {
    ImgViewMut {
      img: self as *mut Img,
      offset: 0,
      h,
      w,
      _phantom: PhantomData,
    }
  }

  pub fn iter_chunks<'a>(
    &'a self,
    chunk_h: usize,
    chunk_w: usize,
  ) -> impl Iterator<Item = ImgView<'a>> {
    let &Img { h, w, c, .. } = self;
    (0..h).step_by(chunk_h).flat_map(move |i0| {
      let i1 = (i0 + chunk_h).min(h);
      (0..w).step_by(chunk_w).map(move |j0| {
        let j1 = (j0 + chunk_w).min(w);
        let offset = c * (i0 * w + j0);
        ImgView {
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
  ) -> impl Iterator<Item = ImgViewMut<'a>> {
    let &mut Img { h, w, c, .. } = self;
    let img = self as *mut Img;
    (0..h).step_by(chunk_h).flat_map(move |i0| {
      let i1 = (i0 + chunk_h).min(h);
      (0..w).step_by(chunk_w).map(move |j0| {
        let j1 = (j0 + chunk_w).min(w);
        let offset = c * (i0 * w + j0);
        ImgViewMut {
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

impl From<DynamicImage> for Img {
  fn from(dyn_img: DynamicImage) -> Self {
    match dyn_img {
      DynamicImage::ImageLuma8(img) => Img {
        h: img.height() as usize,
        w: img.width() as usize,
        c: 1,
        values: img.into_raw(),
      },
      DynamicImage::ImageLumaA8(img) => Img {
        h: img.height() as usize,
        w: img.width() as usize,
        c: 2,
        values: img.into_raw(),
      },
      DynamicImage::ImageRgb8(img) => Img {
        h: img.height() as usize,
        w: img.width() as usize,
        c: 3,
        values: img.into_raw(),
      },
      DynamicImage::ImageRgba8(img) => Img {
        h: img.height() as usize,
        w: img.width() as usize,
        c: 4,
        values: img.into_raw(),
      },
      _ => panic!("unsupported input image format"),
    }
  }
}

impl From<Img> for DynamicImage {
  fn from(img: Img) -> DynamicImage {
    let &Img { h, w, c, .. } = &img;
    if img.c == 3 {
      DynamicImage::ImageRgb8(RgbImage::from_raw(w as u32, h as u32, img.values).unwrap())
    } else if c == 4 {
      DynamicImage::ImageRgba8(RgbaImage::from_raw(w as u32, h as u32, img.values).unwrap())
    } else {
      panic!("unknown number of channels");
    }
  }
}

pub struct ImgView<'a> {
  img: &'a Img,
  offset: usize,
  pub h: usize,
  pub w: usize,
}

impl<'a> ImgView<'a> {
  pub fn hwc(&self) -> (usize, usize, usize) {
    (self.h, self.w, self.img.c)
  }

  pub fn n(&self) -> usize {
    self.h * self.w
  }

  #[inline]
  pub fn get(&self, i: usize, j: usize, k: usize) -> u8 {
    self.img.values[self.offset + k + self.img.c * (i * self.img.w + j)]
  }
}

pub struct ImgViewMut<'a> {
  img: *mut Img,
  offset: usize,
  pub h: usize,
  pub w: usize,
  _phantom: PhantomData<&'a mut Img>,
}

impl<'a> ImgViewMut<'a> {
  pub fn hwc(&self) -> (usize, usize, usize) {
    (self.h, self.w, unsafe { (*self.img).c })
  }

  pub fn n(&self) -> usize {
    self.h * self.w
  }

  #[inline]
  pub fn get(&self, i: usize, j: usize, k: usize) -> u8 {
    unsafe {
      let img = &*self.img;
      img.values[self.offset + k + img.c * (i * img.w + j)]
    }
  }

  #[inline]
  pub fn set(&self, i: usize, j: usize, k: usize, val: u8) {
    unsafe {
      let img = &mut *self.img;
      img.values[self.offset + k + img.c * (i * img.w + j)] = val;
    }
  }

  pub fn binary_op_in_place(&self, other: &ImgView, op: impl Fn(u8, u8) -> u8) {
    let &ImgViewMut { offset, h, w, .. } = self;
    let img = unsafe { &mut *self.img };
    let other_img = other.img;
    let other_offset = other.offset;
    let img_w = img.w;
    let c = img.c;
    for i in 0..h {
      for j in 0..w {
        for k in 0..c {
          let self_idx = offset + k + c * (i * img_w + j);
          let self_value = img.values[self_idx];
          let other_value = other_img.values[other_offset + k + c * (i * other_img.w + j)];
          img.values[self_idx] = op(self_value, other_value);
        }
      }
    }
  }

  pub fn read_channel_flat(&self, k: usize, src: &[u8]) {
    for i in 0..self.h {
      for j in 0..self.w {
        self.set(i, j, k, src[i * self.w + j]);
      }
    }
  }

  pub fn write_channel_flat(&self, k: usize, dst: &mut [u8]) {
    let &ImgViewMut { offset, h, w, .. } = self;
    let img = unsafe { &*self.img };
    let img_w = img.w;
    let c = img.c;
    for i in 0..h {
      for j in 0..w {
        dst[i * w + j] = img.values[offset + k + c * (i * img_w + j)];
      }
    }
  }
}
