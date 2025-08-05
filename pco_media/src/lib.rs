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
  component_scales: Vec<Real>,
  has_adjustments: bool,
}

struct Representation {
  components: DMatrix<Real>,
  streams: Vec<StreamRepresentation>,
}

impl AudioInput {
  fn calc_subsample(&self) -> Self {
    unimplemented!()
  }

  fn calc_pca(&self) -> Pca {
    unimplemented!()
  }

  fn choose_representation(&self, pca: Pca) -> Representation {
    unimplemented!()
  }

  fn calc_arrays(&self, representation: Representation) -> Vec<Vec<i32>> {
    unimplemented!()
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
