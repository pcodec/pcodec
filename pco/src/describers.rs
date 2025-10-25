use crate::constants::{Bitlen, DeltaLookback};
use crate::data_types::{Float, Latent, Number};
use crate::metadata::delta_encoding::LatentVarDeltaEncoding;
use crate::metadata::per_latent_var::PerLatentVar;
use crate::metadata::{ChunkMeta, DeltaEncoding, DynLatent, LatentVarKey, Mode};
use std::marker::PhantomData;

/// Interprets the meaning of latent variables and values from [`ChunkMeta`].
///
/// Obtainable via [`crate::data_types::Number::get_latent_describers`].
pub trait DescribeLatent {
  /// Returns a description for this latent variable.
  fn latent_var(&self) -> String;
  /// Returns a description for this latent variable's units, when formatted
  /// using [`latent()`][Self::latent].
  ///
  /// Returns an empty string if the latents are already interpretable as
  /// numbers.
  fn latent_units(&self) -> String;
  /// Returns a more easily interpretable description for the latent.
  fn latent(&self, latent: DynLatent) -> String;
}

pub type LatentDescriber = Box<dyn DescribeLatent>;

// describer for the delta latent var (not a describer for a latent var that has
// delta encoding applied to it)
fn delta_latent_describer(delta_encoding: &DeltaEncoding) -> Option<LatentDescriber> {
  match delta_encoding {
    DeltaEncoding::NoOp | DeltaEncoding::Consecutive { .. } => None,
    DeltaEncoding::Lookback { .. } => {
      let describer = IntDescriber {
        description: "lookback".to_string(),
        units: "".to_string(),
        center: 0 as DeltaLookback,
        is_signed: false,
      };
      Some(Box::new(describer))
    }
  }
}

pub(crate) fn match_classic_mode<T: Number>(
  meta: &ChunkMeta,
  delta_units: &'static str,
) -> Option<PerLatentVar<LatentDescriber>> {
  let primary: LatentDescriber = match (&meta.mode, &meta.delta_encoding) {
    (Mode::Classic, DeltaEncoding::NoOp) => Box::new(ClassicDescriber::<T>::default()),
    (Mode::Classic, _) => {
      centered_delta_describer::<T::L>("delta".to_string(), delta_units.to_string())
    }
    _ => return None,
  };

  Some(PerLatentVar {
    delta: delta_latent_describer(&meta.delta_encoding),
    primary,
    secondary: None,
  })
}

pub(crate) fn match_int_modes<L: Latent>(
  meta: &ChunkMeta,
  is_signed: bool,
) -> Option<PerLatentVar<LatentDescriber>> {
  match meta.mode {
    Mode::IntMult(dyn_latent) => {
      let base = *dyn_latent.downcast_ref::<L>().unwrap();
      let dtype_center = if is_signed { L::MID } else { L::ZERO };
      let mult_center = dtype_center / base;
      let adj_center = dtype_center % base;
      let primary = if matches!(meta.delta_encoding, DeltaEncoding::NoOp) {
        Box::new(IntDescriber {
          description: format!("multiplier [x{}]", base),
          units: "x".to_string(),
          center: mult_center,
          is_signed,
        })
      } else {
        centered_delta_describer::<L>(
          format!("multiplier delta [x{}]", base),
          "x".to_string(),
        )
      };

      let secondary: LatentDescriber = if !matches!(
        meta.delta_encoding.for_latent_var(LatentVarKey::Secondary),
        LatentVarDeltaEncoding::NoOp
      ) {
        centered_delta_describer::<L>(
          "adjustment delta".to_string(),
          "".to_string(),
        )
      } else {
        Box::new(IntDescriber {
          description: "adjustment".to_string(),
          units: "".to_string(),
          center: adj_center,
          is_signed: false,
        })
      };

      Some(PerLatentVar {
        delta: delta_latent_describer(&meta.delta_encoding),
        primary,
        secondary: Some(secondary),
      })
    }
    Mode::Dict(_) => {
      let primary = if matches!(meta.delta_encoding, DeltaEncoding::None) {
        Box::new(IntDescriber {
          description: "index".to_string(),
          units: "".to_string(),
          center: L::ZERO,
          is_signed: false,
        })
      } else {
        centered_delta_describer::<L>(format!("index delta"), "".to_string())
      };

      Some(PerLatentVar {
        delta: delta_latent_describer(meta.delta_encoding),
        primary,
        secondary: None,
      })
    }
    _ => None,
  }
}

pub(crate) fn match_float_modes<F: Float>(
  meta: &ChunkMeta,
) -> Option<PerLatentVar<LatentDescriber>> {
  match meta.mode {
    Mode::FloatMult(dyn_latent) => {
      let base_latent = *dyn_latent.downcast_ref::<F::L>().unwrap();
      let base_string = F::from_latent_ordered(base_latent).to_string();
      let primary: LatentDescriber = if matches!(meta.delta_encoding, DeltaEncoding::NoOp) {
        Box::new(FloatMultDescriber {
          base_string,
          phantom: PhantomData::<F>,
        })
      } else {
        Box::new(IntDescriber {
          description: format!("multiplier delta [x{}]", base_string),
          units: "x".to_string(),
          center: F::L::MID,
          is_signed: true,
        })
      };

      let secondary: LatentDescriber = if !matches!(
        meta.delta_encoding.for_latent_var(LatentVarKey::Secondary),
        LatentVarDeltaEncoding::NoOp
      ) {
        centered_delta_describer::<F::L>(
          "adjustment delta".to_string(),
          "".to_string(),
        )
      } else {
        Box::new(IntDescriber {
          description: "adjustment".to_string(),
          units: " ULPs".to_string(),
          center: F::L::MID,
          is_signed: true,
        })
      };

      Some(PerLatentVar {
        delta: delta_latent_describer(&meta.delta_encoding),
        primary,
        secondary: Some(secondary),
      })
    }
    Mode::FloatQuant(k) => {
      let primary = if matches!(meta.delta_encoding, DeltaEncoding::NoOp) {
        Box::new(FloatQuantDescriber {
          k,
          phantom: PhantomData::<F>,
        })
      } else {
        centered_delta_describer::<F::L>(
          format!("quantums delta [<<{}]", k),
          "q".to_string(),
        )
      };

      let secondary: LatentDescriber = if !matches!(
        meta.delta_encoding.for_latent_var(LatentVarKey::Secondary),
        LatentVarDeltaEncoding::NoOp
      ) {
        centered_delta_describer::<F::L>(
          "magnitude adjustment delta".to_string(),
          "".to_string(),
        )
      } else {
        Box::new(IntDescriber {
          description: "magnitude adjustment".to_string(),
          units: " ULPs".to_string(),
          center: F::L::ZERO,
          is_signed: false,
        })
      };

      Some(PerLatentVar {
        delta: delta_latent_describer(&meta.delta_encoding),
        primary,
        secondary: Some(secondary),
      })
    }
    _ => None,
  }
}

#[derive(Default)]
struct ClassicDescriber<T: Number>(PhantomData<T>);

impl<T: Number> DescribeLatent for ClassicDescriber<T> {
  fn latent_var(&self) -> String {
    "primary".to_string()
  }

  fn latent_units(&self) -> String {
    "".to_string()
  }

  fn latent(&self, latent: DynLatent) -> String {
    T::from_latent_ordered(latent.downcast::<T::L>().unwrap()).to_string()
  }
}

struct IntDescriber<L: Latent> {
  description: String,
  units: String,
  center: L,
  is_signed: bool,
}

impl<L: Latent> DescribeLatent for IntDescriber<L> {
  fn latent_var(&self) -> String {
    self.description.to_string()
  }

  fn latent_units(&self) -> String {
    self.units.to_string()
  }

  fn latent(&self, latent: DynLatent) -> String {
    let centered = latent.downcast::<L>().unwrap().wrapping_sub(self.center);
    if centered < L::MID || !self.is_signed {
      centered.to_string()
    } else {
      format!("-{}", L::MAX - (centered - L::ONE),)
    }
  }
}

fn centered_delta_describer<L: Latent>(description: String, units: String) -> LatentDescriber {
  Box::new(IntDescriber {
    description,
    units,
    center: L::MID,
    is_signed: true,
  })
}

struct FloatMultDescriber<F: Float> {
  base_string: String,
  phantom: PhantomData<F>,
}

impl<F: Float> DescribeLatent for FloatMultDescriber<F> {
  fn latent_var(&self) -> String {
    format!("multiplier [x{}]", self.base_string)
  }

  fn latent_units(&self) -> String {
    "x".to_string()
  }

  fn latent(&self, latent: DynLatent) -> String {
    F::int_float_from_latent(latent.downcast::<F::L>().unwrap()).to_string()
  }
}

struct FloatQuantDescriber<F: Float> {
  k: Bitlen,
  phantom: PhantomData<F>,
}

impl<F: Float> DescribeLatent for FloatQuantDescriber<F> {
  fn latent_var(&self) -> String {
    "quantized".to_string()
  }

  fn latent_units(&self) -> String {
    "".to_string()
  }

  fn latent(&self, latent: DynLatent) -> String {
    let shifted = latent.downcast::<F::L>().unwrap() << self.k;
    if shifted >= F::L::MID {
      F::from_latent_ordered(shifted).to_string()
    } else {
      (-F::from_latent_ordered(F::L::MAX - shifted)).to_string()
    }
  }
}
