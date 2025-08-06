use nalgebra::{Const, DMatrix, Dyn, OMatrix, RowOVector, RowVector};
use pco::{ChunkConfig, ModeSpec, PagingSpec};

type Real = f32;

struct Pca {
  svd: nalgebra::SVD<Real, Dyn, Dyn>,
  means: RowOVector<Real, Dyn>,
}

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

// Our approach here is to reduce rendancy in audio channels by running PCA on a
// subsample of the data.
// Then for the full data, each useful principal component will be stored by
// Pco, and each stream may get its own adjustment Pco array.
struct StreamRepresentation {
  has_adjustments: bool,
}

struct Representation {
  u_matrix: DMatrix<Real>,
  singular_values: nalgebra::DVector<Real>,
  means: RowOVector<Real, Dyn>,
  streams: Vec<StreamRepresentation>,
}

impl AudioInput {
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

  fn calc_pca(&self) -> Pca {
    let n_items = self.len();
    let mut matrix = DMatrix::from_row_iterator(
      n_items,
      self.n_streams,
      self.data.iter().map(|&x| x as Real),
    );
    println!("matrix shape {:?}", matrix.shape());
    let means = matrix.row_mean();
    matrix.row_iter_mut().for_each(|mut row| row -= &means);
    println!("means {:?}", means);
    let covariance = (matrix.transpose() * &matrix) / (n_items - 1) as Real;
    println!("covar {:?}", covariance);

    let svd = covariance.svd(true, true);
    println!("u {:?}", svd.u);
    println!("singular values {:?}", svd.singular_values);
    Pca { svd, means }
  }

  fn choose_representation(&self, pca: Pca) -> Representation {
    let threshold = 1.0 / self.n_streams as Real;
    let singular_values = &pca.svd.singular_values;
    let total_variance: Real = singular_values.iter().map(|s| s * s).sum();

    let n_components = singular_values
      .iter()
      .take_while(|&&sigma| (sigma * sigma) / total_variance >= threshold)
      .count();
    println!("taking {} pcs", n_components);

    let u = pca.svd.u.as_ref().unwrap();
    let u_matrix = u.columns(0, n_components).into_owned();
    let singular_values = singular_values.rows(0, n_components).into_owned();
    println!(
      "sub u, values: {:?} {:?}",
      u_matrix, singular_values
    );

    let streams = (0..self.n_streams)
      .map(|_| StreamRepresentation {
        has_adjustments: true,
      })
      .collect();

    Representation {
      u_matrix,
      singular_values,
      means: pca.means,
      streams,
    }
  }

  fn calc_arrays(&self, representation: Representation) -> Vec<Vec<i32>> {
    let n_items = self.len();
    let original_matrix = DMatrix::from_row_iterator(
      n_items,
      self.n_streams,
      self.data.iter().copied(),
    );

    let mut matrix = original_matrix.map(|x| x as Real);
    println!("orig var {:?}", matrix.variance());
    matrix
      .row_iter_mut()
      .for_each(|mut row| row -= &representation.means);
    println!("mean sub var {:?}", matrix.variance());
    let principal_components = &matrix * &representation.u_matrix;
    println!(
      "pc var {:?} (shape={:?})",
      principal_components.variance(),
      principal_components.shape(),
    );
    let pc_quantized = principal_components.map(|x| x as i32);
    println!(
      "pc quantized var {:?}",
      pc_quantized.map(|x| x as Real).variance()
    );

    let mut reconstructed = pc_quantized.map(|x| x as Real) * representation.u_matrix.transpose();
    println!("reconstructed var {:?}", matrix.variance());
    reconstructed
      .row_iter_mut()
      .for_each(|mut row| row += &representation.means);
    let reconstructed_quantized = reconstructed.map(|x| x as i32);
    println!(
      "reconstructed quantized var {:?}",
      matrix.variance()
    );

    let residuals = &original_matrix - &reconstructed_quantized;
    println!(
      "residual var {:?}",
      residuals.map(|x| x as Real).variance()
    );

    let mut arrays = Vec::new();

    // Add principal component arrays
    for col in pc_quantized.column_iter() {
      arrays.push(col.iter().copied().collect());
    }

    // Add adjustment arrays for streams that have adjustments
    for (stream_idx, stream) in representation.streams.iter().enumerate() {
      if stream.has_adjustments {
        arrays.push(residuals.column(stream_idx).iter().copied().collect());
      }
    }

    arrays
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
    let pca = subsample.calc_pca();
    let representation = subsample.choose_representation(pca);
    let arrays = chunk_input.calc_arrays(representation);

    let chunk_config = ChunkConfig::default()
      .with_mode_spec(ModeSpec::Classic)
      .with_paging_spec(PagingSpec::Exact(vec![chunk_len]));

    // Compress each array in the chunk and collect into the main buffer
    for array in arrays {
      let compressed_array = pco::standalone::simple_compress(&array, &chunk_config).unwrap();
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
    let mut reader =
      hound::WavReader::open("../data/musdb18hq/Young Griffo - Pennies.wav").unwrap();
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
