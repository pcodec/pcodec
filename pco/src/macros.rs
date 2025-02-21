#![allow(clippy::single_component_path_imports)]

dtype_dispatch::build_dtype_macros!(
  #[doc = "\
    **unstable API** Defines enums holding a container generic to `Number`.
  "]
  #[doc = "\
    You'll only want to use this if you're using pco's low level APIs.
    See the dtype_dispatch crate for more details.
  "]
  #[macro_export]
  define_number_enum,

  #[doc = "\
    Matches enums holding a container of `Number`s and puts the concrete type
    into scope.
  "]
  #[doc = "\
    You'll only want to use this if you're using pco's low level APIs.
    See the dtype_dispatch crate for more details.
  "]
  #[macro_export]
  match_number_enum,

  Number,
  {
    F16 => half::f16,
    F32 => f32,
    F64 => f64,
    I8 => i8,
    I16 => i16,
    I32 => i32,
    I64 => i64,
    U8 => u8,
    U16 => u16,
    U32 => u32,
    U64 => u64,
  },
);

dtype_dispatch::build_dtype_macros!(
  define_latent_enum,

  #[doc = "\
    Matches enums holding a container of `Latent`s and puts the concrete type
    into scope.
  "]
  #[doc = "\
    You'll only want to use this if you're using pco's low level APIs.
    See the dtype_dispatch crate for more details.
  "]
  #[macro_export]
  match_latent_enum,

  Latent,
  {
    U8 => u8,
    U16 => u16,
    U32 => u32,
    U64 => u64,
  },
);

pub(crate) use define_latent_enum;
pub(crate) use define_number_enum;
pub(crate) use match_latent_enum;
