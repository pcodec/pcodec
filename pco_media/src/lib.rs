use nalgebra::{Const, DMatrix, Dyn, OMatrix, RowOVector, RowVector};
use pco::{ChunkConfig, ModeSpec, PagingSpec};

type Real = f32;

pub struct AudioInput {
  // data is laid out like interleaved (n_items x n_streams) array
  data: Vec<i32>,
  n_streams: usize,
}

impl AudioInput {
  fn len(&self) -> usize {
    self.data.len() / self.n_streams
  }
}

// Our approach here:
// For the nth stream, residualize out based on a weighted sum of the 0th
// through (n-1)st stream, then store just the residuals. The weights come from
// a linear regression and reduce the variance of each stream, while allowing us
// to exactly compute the nth stream given just the residuals of streams 0
// through n. We essentially compute each weight column via linear regression.
//
// You might think that PCA would be the natural tool for the job here, but
// variance reduction is vastly different from entropy reduction. Example: say
// we have some correlated random normal-ish integer streams with covar matrix
// 1000 900
// 900  1000
// Option 1: just store each stream.
//   Each one would take about log2(sqrt(1000)) for a total of 10 bits.
// Option 2: residualize the second stream using the first.
//   The first one would take the full log2(sqrt(1000)), leaving just 100
//   unexplained variance for the first for another log2(sqrt(100)) = 3.3 bits,
//   totalling 8.3 bits.
// Option 3: store a residual for each variable and one principal component.
//   The principal component would explain 950 of each variance, taking
//   about log2(sqrt(1900)) = 5.4 bits to store. Each adjustment would require
//   about log2(sqrt(50)) = 2.8 bits to store, so in total we'd use 11.0 bits.
//   We do badly because we store 3 sequences instead of 2, despite them being
//   "simpler" covariance-wise. And we couldn't get away with storing all
//   principal components and no residuals, since the purely float arithmetic
//   might not be lossless. If we could magically do that, this would take a
//   total of 8.7 bits.
struct Representation {
  weights: DMatrix<Real>,
  biases: RowOVector<Real, Dyn>,
}

impl AudioInput {
  fn matrix(&self) -> DMatrix<i32> {
    DMatrix::from_row_iterator(
      self.len(),
      self.n_streams,
      self.data.iter().copied(),
    )
  }

  fn calc_subsample(&self) -> Self {
    let mut subsampled_data = Vec::new();

    for item_idx in (0..self.len()).step_by(100) {
      let start_idx = item_idx * self.n_streams;
      subsampled_data.extend_from_slice(&self.data[start_idx..start_idx + self.n_streams]);
    }

    Self {
      data: subsampled_data,
      n_streams: self.n_streams,
    }
  }

  fn choose_representation(&self) -> Representation {
    let exact_matrix = self.matrix();
    let mut matrix = exact_matrix.map(|x| x as Real);
    let biases = matrix.row_mean();
    matrix.row_iter_mut().for_each(|mut row| row -= &biases);

    // Compute full covariance matrix (scaled by a factor of n-1)
    let full_cov = matrix.transpose() * &matrix;

    // Initialize weights matrix - each column i contains regression coefficients
    // to predict stream i using streams 0..i-1 as independent variables
    let mut weights = DMatrix::zeros(self.n_streams, self.n_streams);

    // For each stream i, find the best linear regression coefficients
    // using streams 0..i-1 as independent variables
    for i in 1..self.n_streams {
      // Slice the precomputed covariance matrix
      let xtx = full_cov.view((0, 0), (i, i));
      let xty = full_cov.view((0, i), (i, 1));

      // Solve for regression coefficients
      if let Some(beta) = xtx.try_inverse() {
        let coefficients = beta * xty;

        for j in 0..i {
          weights[(j, i)] = coefficients[j];
        }
      }
    }
    println!("weights={:?}", weights);

    Representation { weights, biases }
  }

  fn calc_arrays(&self, representation: Representation) -> Vec<Vec<i32>> {
    let original_matrix = self.matrix();
    let matrix = original_matrix.map(|x| x as Real);
    println!("orig var {:?}", matrix.variance());
    let mut prediction = matrix * representation.weights;
    prediction
      .row_iter_mut()
      .for_each(|mut row| row += &representation.biases);
    let residuals = original_matrix - prediction.map(|x| x.round() as i32);
    println!(
      "residual var {:?}",
      residuals.map(|x| x as Real).variance()
    );
    residuals
      .column_iter()
      .map(|c| c.iter().copied().collect())
      .collect()
  }
}

pub fn compress_audio(input: AudioInput) -> Vec<u8> {
  const MAX_CHUNK_SIZE: usize = 1 << 18; // 2^18 = 262,144 samples

  let total_len = input.len();
  let mut compressed_data = Vec::new();

  // Process data in chunks
  for chunk_start in (0..total_len).step_by(MAX_CHUNK_SIZE) {
    let chunk_end = (chunk_start + MAX_CHUNK_SIZE).min(total_len);
    let chunk_len = chunk_end - chunk_start;

    // Extract chunk data
    let chunk_data_start = chunk_start * input.n_streams;
    let chunk_data_end = chunk_end * input.n_streams;
    let chunk_data = input.data[chunk_data_start..chunk_data_end].to_vec();

    let chunk_input = AudioInput {
      data: chunk_data,
      n_streams: input.n_streams,
    };

    // Process the chunk through PCA and compression
    let subsample = chunk_input.calc_subsample();
    let representation = subsample.choose_representation();
    let arrays = chunk_input.calc_arrays(representation);

    let chunk_config = ChunkConfig::default()
      .with_mode_spec(ModeSpec::Classic)
      .with_paging_spec(PagingSpec::Exact(vec![chunk_len]));

    // Compress each array in the chunk and collect into the main buffer
    for array in arrays {
      let compressed_array = pco::standalone::simple_compress(&array, &chunk_config).unwrap();
      println!("compressed size {}", compressed_array.len());
      compressed_data.extend(compressed_array);
    }
  }

  compressed_data
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_compress_wav() {
    let mut reader = hound::WavReader::open(
      "../data/musdb18hq/A Classic Education - NightOwl.wav", // "../data/musdb18hq/Young Griffo - Pennies.wav"
    )
    .unwrap();
    let spec = reader.spec();

    let samples: Vec<i32> = reader
      .samples::<i32>()
      .collect::<Result<Vec<_>, _>>()
      .unwrap();

    println!(
      "sum {:?}",
      samples.iter().map(|x| *x as i64).sum::<i64>()
    );

    let input = AudioInput {
      data: samples,
      n_streams: spec.channels as usize,
    };

    let compressed = compress_audio(input);
    println!("{}", compressed.len());
  }
}
