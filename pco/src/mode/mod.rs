pub mod classic;
pub mod dict;
pub mod float_mult;
pub mod float_quant;
pub mod int_mult;

fn single_category_entropy(p: f64) -> f64 {
  if p == 0.0 || p == 1.0 {
    0.0
  } else {
    -p * p.log2()
  }
}

fn worst_case_categorical_entropy(concentrated_p: f64, n_categories_m1: f64) -> f64 {
  single_category_entropy(concentrated_p)
    + n_categories_m1 * single_category_entropy((1.0 - concentrated_p) / n_categories_m1)
}
