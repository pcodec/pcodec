use std::ptr;

use crate::data_types::Latent;

pub fn choose_pivot<L: Latent>(latents: &[L]) -> L {
  // Minimum length to choose the median-of-medians method.
  // Shorter slices use the simple median-of-three method.
  const SHORTEST_MEDIAN_OF_MEDIANS: usize = 50;

  let len = latents.len();

  // Three indices near which we are going to choose a pivot.
  let mut a = len / 4;
  let mut b = len / 2;
  let mut c = (len * 3) / 4;

  if len >= 8 {
    // Swaps indices so that `v[a] <= v[b]`.
    // SAFETY: `len >= 8` so there are at least two elements in the neighborhoods of
    // `a`, `b` and `c`. This means the three calls to `sort_adjacent` result in
    // corresponding calls to `sort3` with valid 3-item neighborhoods around each
    // pointer, which in turn means the calls to `sort2` are done with valid
    // references. Thus the `v.get_unchecked` calls are safe, as is the `ptr::swap`
    // call.
    let sort2 = |a: &mut usize, b: &mut usize| unsafe {
      if *latents.get_unchecked(*b) < *latents.get_unchecked(*a) {
        ptr::swap(a, b);
      }
    };

    // Swaps indices so that `v[a] <= v[b] <= v[c]`.
    let sort3 = |a: &mut usize, b: &mut usize, c: &mut usize| {
      sort2(a, b);
      sort2(b, c);
      sort2(a, b);
    };

    if len >= SHORTEST_MEDIAN_OF_MEDIANS {
      // Finds the median of `v[a - 1], v[a], v[a + 1]` and stores the index into `a`.
      let sort_adjacent = |a: &mut usize| {
        let tmp = *a;
        sort3(&mut (tmp - 1), a, &mut (tmp + 1));
      };

      // Find medians in the neighborhoods of `a`, `b`, and `c`.
      sort_adjacent(&mut a);
      sort_adjacent(&mut b);
      sort_adjacent(&mut c);
    }

    // Find the median among `a`, `b`, and `c`.
    sort3(&mut a, &mut b, &mut c);
  }

  latents[b]
}

// Scatters some elements around in an attempt to break patterns that might cause imbalanced
// partitions in quicksort.
#[cold]
pub fn break_patterns<L>(v: &mut [L]) {
  let len = v.len();
  if len >= 8 {
    let mut seed = len;
    let mut gen_usize = || {
      // Pseudorandom number generator from the "Xorshift RNGs" paper by George Marsaglia.
      if usize::BITS <= 32 {
        let mut r = seed as u32;
        r ^= r << 13;
        r ^= r >> 17;
        r ^= r << 5;
        seed = r as usize;
        seed
      } else {
        let mut r = seed as u64;
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        seed = r as usize;
        seed
      }
    };

    // Take random numbers modulo this number.
    // The number fits into `usize` because `len` is not greater than `isize::MAX`.
    let modulus = len.next_power_of_two();

    // Some pivot candidates will be in the nearby of this index. Let's randomize them.
    let pos = len / 4 * 2;

    for i in 0..3 {
      // Generate a random number modulo `len`. However, in order to avoid costly operations
      // we first take it modulo a power of two, and then decrease by `len` until it fits
      // into the range `[0, len - 1]`.
      let mut other = gen_usize() & (modulus - 1);

      // `other` is guaranteed to be less than `2 * len`.
      if other >= len {
        other -= len;
      }

      v.swap(pos - 1 + i, other);
    }
  }
}

// returns (count on left side of pivot, was_bad_pivot)
// Uses lomuto partitioning
pub fn partition<L: Latent>(latents: &mut [L], pivot: L) -> (usize, bool) {
  // |-- < pivot--|-- >= pivot --|-- unprocessed --|
  let mut left_idx = 0;
  let mut pos = latents.as_mut_ptr();
  unsafe {
    let end = latents.as_mut_ptr().add(latents.len());
    while pos < end {
      let value = *pos;
      let is_lt_pivot = value < pivot;
      *pos = *latents.get_unchecked(left_idx);
      *latents.get_unchecked_mut(left_idx) = value;
      left_idx += is_lt_pivot as usize;
      pos = pos.add(1);
    }
  }
  let was_bad_pivot = 1 + left_idx.min(latents.len() - left_idx) < latents.len() / 8;
  (left_idx, was_bad_pivot)
}

// Sorts `v` using heapsort, which guarantees *O*(*n* \* log(*n*)) worst-case.
#[cold]
pub fn heapsort<L: Latent>(latents: &mut [L]) {
  // This binary heap respects the invariant `parent >= child`.
  let sift_down = |x: &mut [L], mut node| {
    loop {
      // Children of `node`.
      let mut child = 2 * node + 1;
      if child >= x.len() {
        break;
      }

      // Choose the greater child.
      if child + 1 < x.len() {
        // We need a branch to be sure not to out-of-bounds index,
        // but it's highly predictable.  The comparison, however,
        // is better done branchless, especially for primitives.
        child += (x[child] < x[child + 1]) as usize;
      }

      // Stop if the invariant holds at `node`.
      if x[node] >= x[child] {
        break;
      }

      // Swap `node` with the greater child, move one step down, and continue sifting.
      x.swap(node, child);
      node = child;
    }
  };

  // Build the heap in linear time.
  for i in (0..latents.len() / 2).rev() {
    sift_down(latents, i);
  }

  // Pop maximal elements from the heap.
  for i in (1..latents.len()).rev() {
    latents.swap(0, i);
    sift_down(&mut latents[..i], 0);
  }
}
