use core::f64;

type Real = f64;

// poor man's nalgebra so we don't need a whole new dep
#[derive(Clone, Debug)]
pub struct Matrix {
  pub data: Vec<Real>,
  pub h: usize,
  pub w: usize,
}

fn safe_sqrt(x: Real) -> Real {
  x.max(0.0).sqrt()
}

impl Matrix {
  pub fn constant(value: Real, h: usize, w: usize) -> Self {
    Self {
      data: vec![value; h * w],
      h,
      w,
    }
  }

  #[inline]
  fn physical_idx(&self, i: usize, j: usize) -> usize {
    // column-major is more efficient for our use cases
    i + j * self.h
  }

  #[inline]
  pub unsafe fn set(&mut self, i: usize, j: usize, value: Real) {
    let idx = self.physical_idx(i, j);
    *self.data.get_unchecked_mut(idx) = value;
  }

  #[inline]
  pub unsafe fn get(&self, i: usize, j: usize) -> Real {
    let idx = self.physical_idx(i, j);
    *self.data.get_unchecked(idx)
  }

  #[inline(never)]
  fn into_cholesky(mut self) -> Self {
    // returns L matrix from X = LL* assuming X is positive semi-definite
    // Cholesky-Crout algorithm
    assert_eq!(self.h, self.w);
    let h = self.h;
    for j in 0..h {
      unsafe {
        // top half of matrix is 0s
        for i in 0..j {
          self.set(i, j, 0.0);
        }

        // diagonal requires square root
        let mut s = 0.0;
        for k in 0..j {
          let value = self.get(j, k);
          s += value * value;
        }
        let diag_value = safe_sqrt(self.get(j, j) - s);
        self.set(j, j, diag_value);
        let scale = if diag_value == 0.0 {
          0.0
        } else {
          1.0 / diag_value
        };

        // bottom half
        for i in j + 1..h {
          let mut s = 0.0;
          for k in 0..j {
            s += self.get(i, k) * self.get(j, k);
          }
          self.set(i, j, scale * (self.get(i, j) - s));
        }
      }
    }

    self
  }

  #[inline(never)]
  fn transposed_backward_sub_into(&self, mut y: Matrix) -> Matrix {
    // assuming self is lower triangular, solves for x in
    //   Self^T * x = y
    let h = self.h;
    assert_eq!(h, self.w);
    assert_eq!(h, y.h);
    let w = y.w;
    for k in 0..w {
      for j in (0..h).rev() {
        unsafe {
          let diag_value = self.get(j, j);
          let xjk = if diag_value.abs() >= 1E-12 {
            y.get(j, k) / diag_value
          } else {
            0.0
          };
          y.set(j, k, xjk);
          for i in 0..j {
            y.set(i, k, y.get(i, k) - xjk * self.get(j, i));
          }
        }
      }
    }
    y
  }

  #[inline(never)]
  fn forward_sub_into(&self, mut y: Matrix) -> Matrix {
    // assuming self is lower triangular, solves for x in
    //   Self * x = y
    let h = self.h;
    assert_eq!(h, self.w);
    assert_eq!(h, y.h);
    let w = y.w;
    for k in 0..w {
      for j in 0..h {
        unsafe {
          let diag_value = self.get(j, j);
          let xjk = if diag_value.abs() >= 1E-12 {
            y.get(j, k) / diag_value
          } else {
            0.0
          };
          y.set(j, k, xjk);
          for i in j + 1..h {
            y.set(i, k, y.get(i, k) - xjk * self.get(i, j));
          }
        }
      }
    }
    y
  }
}

pub fn solve_least_squares(xtx: Matrix, xty: Matrix) -> Matrix {
  let cholesky = xtx.into_cholesky();
  let half_solved = cholesky.forward_sub_into(xty);
  cholesky.transposed_backward_sub_into(half_solved)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn matrix_from_rows(rows: Vec<Vec<Real>>) -> Matrix {
    let h = rows.len();
    let w = rows[0].len();
    let mut m = Matrix::constant(0.0, h, w);
    for i in 0..h {
      for j in 0..w {
        unsafe {
          m.set(i, j, rows[i][j]);
        }
      }
    }
    m
  }

  #[test]
  fn cholesky() {
    // here A = LL^T
    //  0.1  0   0
    //  -2   3   0
    //  -4   5   6
    let l = matrix_from_rows(vec![
      vec![0.1, 0.0, 0.0],
      vec![-2.0, 3.0, 0.0],
      vec![-4.0, 5.0, 6.0],
    ]);
    let a = matrix_from_rows(vec![
      vec![0.01, -0.2, -0.4],
      vec![-0.2, 13.0, 23.0],
      vec![-0.4, 23.0, 77.0],
    ]);
    let cholesky = a.into_cholesky();
    assert_eq!(l.data, cholesky.data);
  }

  #[test]
  fn forward_sub() {
    let a = matrix_from_rows(vec![
      vec![2.0, 0.0],  //
      vec![3.0, -4.0], //
    ]);
    let y = matrix_from_rows(vec![
      vec![1.0], //
      vec![2.0], //
    ]);
    let x = a.forward_sub_into(y);
    let expected = vec![0.5, -0.125];
    for i in 0..expected.len() {
      assert!((x.data[i] - expected[i]).abs() < 1E-6);
    }
  }

  #[test]
  fn transpose_backward_sub() {
    let a = matrix_from_rows(vec![
      vec![2.0, 0.0],  //
      vec![3.0, -4.0], //
    ]);
    let y = matrix_from_rows(vec![
      vec![1.0], //
      vec![2.0], //
    ]);
    let x = a.transposed_backward_sub_into(y);
    let expected = vec![1.25, -0.5];
    println!("{:?}", x.data);
    for i in 0..expected.len() {
      assert!((x.data[i] - expected[i]).abs() < 1E-6);
    }
  }
}
