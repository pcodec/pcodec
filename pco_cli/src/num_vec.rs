use crate::dtypes::PcoNumber;
use anyhow::{anyhow, Result};
use pco::data_types::{Number, NumberType};
use pco::{define_number_enum, match_number_enum};

fn check_equal<T: PcoNumber>(recovered: &[T], original: &[T]) -> Result<()> {
  if recovered.len() != original.len() {
    return Err(anyhow!(
      "recovered length {} != original length {}",
      recovered.len(),
      original.len()
    ));
  }

  for (i, (x, y)) in recovered.iter().zip(original.iter()).enumerate() {
    if x.to_latent_ordered() != y.to_latent_ordered() {
      return Err(anyhow!(
        "{} != {} at {}",
        recovered[i],
        original[i],
        i
      ));
    }
  }
  Ok(())
}

define_number_enum!(
  #[derive()]
  pub NumVec(Vec)
);

impl NumVec {
  pub fn n(&self) -> usize {
    match_number_enum!(
      self,
      NumVec<T>(nums) => { nums.len() }
    )
  }

  pub fn dtype(&self) -> NumberType {
    match_number_enum!(
      self,
      NumVec<T>(_inner) => { NumberType::new::<T>().unwrap() }
    )
  }

  pub fn truncated(&self, limit: usize) -> Self {
    match_number_enum!(
      self,
      NumVec<T>(nums) => { NumVec::new(nums[..limit].to_vec()).unwrap() }
    )
  }

  pub fn check_equal(&self, other: &NumVec) -> Result<()> {
    match_number_enum!(
      self,
      NumVec<T>(nums) => {
        let other_nums = other.downcast_ref::<T>().ok_or_else(|| anyhow!(
          "NumVecs had mismatched types"
        ))?;
        check_equal(nums, other_nums)?;
      }
    );
    Ok(())
  }
}
