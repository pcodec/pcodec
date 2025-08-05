use nalgebra::{DMatrix, Dyn};
use pco::{ChunkConfig, ModeSpec, PagingSpec};

type Real = f32;
type Pca = nalgebra::SVD<Real, Dyn, Dyn>;

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
    let matrix = DMatrix::from_row_iterator(n_items, self.n_streams, 
      self.data.iter().map(|&x| x as Real));
    
    let mean = matrix.column_mean();
    let centered = matrix - DMatrix::from_rows(&vec![mean.transpose(); n_items]);
    let covariance = (centered.transpose() * &centered) / (n_items - 1) as Real;
    
    covariance.svd(true, true)
  }

  fn choose_representation(&self, pca: Pca) -> Representation {
    let threshold = 1.0 / self.n_streams as Real;
    let total_variance: Real = pca.singular_values.iter().map(|s| s * s).sum();
    
    let n_components = pca.singular_values.iter()
      .take_while(|&&sigma| (sigma * sigma) / total_variance >= threshold)
      .count();
    
    let u = pca.u.as_ref().unwrap();
    let u_matrix = u.columns(0, n_components).into_owned();
    let singular_values = pca.singular_values.rows(0, n_components).into_owned();
    
    let streams = (0..self.n_streams).map(|_| {
      StreamRepresentation {
        has_adjustments: true,
      }
    }).collect();
    
    Representation { u_matrix, singular_values, streams }
  }

  fn calc_arrays(&self, representation: Representation) -> Vec<Vec<i32>> {
    let n_items = self.len();
    let original_matrix = DMatrix::from_row_iterator(n_items, self.n_streams, 
      self.data.iter().copied());
    
    let matrix = original_matrix.cast::<Real>();
    let principal_components = &matrix * &representation.u_matrix;
    let pc_quantized = principal_components.cast::<i32>();
    
    let reconstructed = pc_quantized.cast::<Real>() * representation.u_matrix.transpose();
    let reconstructed_quantized = reconstructed.cast::<i32>();
    
    let residuals = &original_matrix - &reconstructed_quantized;
    
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
  let page_size = input.len();
  let subsample = input.calc_subsample();
  let pca = subsample.calc_pca();
  let representation = subsample.choose_representation(pca);
  let arrays = input.calc_arrays(representation);
  let chunk_config = ChunkConfig::default()
    .with_mode_spec(ModeSpec::Classic)
    .with_paging_spec(PagingSpec::Exact(vec![page_size]));
  arrays
    .into_iter()
    .map(|array| pco::standalone::simple_compress(&array, &chunk_config).unwrap())
    .flatten()
    .collect()
}
