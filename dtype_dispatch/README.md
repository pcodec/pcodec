`dtype_dispatch` solves the problem of interop between *generic* and
*dynamically typed (enum)* containers.

This is a common problem in numerical libraries (think numpy, torch, polars):
you have a variety of data types and data structures to hold them, but every
function involves matching an enum or converting from a generic to an enum.

Example with `i32` and `i64` data types for dynamically-typed vectors,
supporting `.length()` and `.add(other)` operations, plus generic
`new` and `downcast` functions:

```rust
pub trait Dtype: 'static {}
impl Dtype for i32 {}
impl Dtype for i64 {}

// register our two macros, `define_an_enum` and `match_an_enum`, constrained
// to the `Dtype` trait, with our variant => type mapping:
dtype_dispatch::build_dtype_macros!(
  define_an_enum,
  match_an_enum,
  Dtype,
  {
    I32 => i32,
    I64 => i64,
  },
);

// define any enum holding a Vec of any data type!
define_an_enum!(
  #[derive(Clone, Debug)]
  DynVec(Vec)
);

impl DynVec {
  pub fn length(&self) -> usize {
    match_an_enum!(self, DynVec<T>(inner) => { inner.len() })
  }

  pub fn add(&self, other: &DynVec) -> DynVec {
    match_an_enum!(self, DynVec<T>(inner) => {
      let other_inner = other.downcast_ref::<T>().unwrap();
      let added = inner.iter().zip(other_inner).map(|(a, b)| a + b).collect::<Vec<_>>();
      DynVec::new(added)
    })
  }
}

// we could also use `DynVec::I32()` here, but just to show we can convert generics:
let x_dynamic = DynVec::new(vec![1_i32, 2, 3]);
let x_doubled_generic = x_dynamic.add(&x_dynamic).downcast::<i32>().unwrap();
assert_eq!(x_doubled_generic, vec![2, 4, 6]);
```

Compare this with the same API written manually:

```rust
use std::{any, mem};

pub trait Dtype: 'static {}
impl Dtype for i32 {}
impl Dtype for i64 {}

#[derive(Clone, Debug)]
pub enum DynVec {
  I32(Vec<i32>),
  I64(Vec<i64>),
}

impl DynVec {
  pub fn length(&self) -> usize {
    match self {
      DynVec::I32(inner) => inner.len(),
      DynVec::I64(inner) => inner.len(),
    }
  }

  pub fn add(&self, other: &DynVec) -> DynVec {
    match (self, other) {
      (DynVec::I32(inner), DynVec::I32(other_inner)) => {
        let added = inner.iter().zip(other_inner).map(|(&a, &b)| a + b).collect::<Vec<_>>();
        DynVec::I32(added)
      }
      (DynVec::I64(inner), DynVec::I64(other_inner)) => {
        let added = inner.iter().zip(other_inner).map(|(&a, &b)| a + b).collect::<Vec<_>>();
        DynVec::I64(added)
      }
      _ => panic!("mismatched dtypes")
    }
  }

  pub fn new<T: Dtype>(inner: Vec<T>) -> DynVec {
    let type_id = any::TypeId::of::<T>();
    if type_id == any::TypeId::of::<i32>() {
      DynVec::I32(unsafe { mem::transmute(inner) })
    } else if type_id == any::TypeId::of::<i64>() {
      DynVec::I64(unsafe { mem::transmute(inner) })
    } else {
      panic!("unknown dtype")
    }
  }

  pub fn downcast<T: Dtype>(self) -> Vec<T> {
    let type_id = any::TypeId::of::<T>();
    match self {
      DynVec::I32(inner) => {
        if type_id == any::TypeId::of::<i32>() {
          unsafe { mem::transmute(inner) }
        } else {
          panic!("incorrect dtype")
        }
      }
      DynVec::I64(inner) => {
        if type_id == any::TypeId::of::<i64>() {
          unsafe { mem::transmute(inner) }
        } else {
          panic!("incorrect dtype")
        }
      }
    }
  }
}

let x_dynamic = DynVec::new(vec![1_i32, 2, 3]);
let x_doubled_generic = x_dynamic.add(&x_dynamic).downcast::<i32>();
assert_eq!(x_doubled_generic, vec![2, 4, 6]);
```

That's a lot of match/if clauses and repeated boilerplate!
It would become impossible to manage if we had 10 data types and multiple
containers (e.g. sparse arrays).
`dtype_dispatch` elegantly solves this with a single macro that generates two
powerful macros for you to use.
These building blocks can solve almost any dynamic<->generic data type dispatch
problem:


## Comparisons

|                             | `Box<dyn>` | `enum_dispatch` | `dtype_dispatch`        |
|-----------------------------|------------|-----------------|-------------------------|
| convert generic -> dynamic  | ✅          | ❌*              | ✅                       |
| convert dynamic -> generic  | ❌          | ❌*              | ✅                       |
| call trait fns directly     | ⚠️**       | ✅               | ❌                       |
| match with type information | ❌️         | ❌               | ✅                       |
| stack allocated             | ❌️         | ✅               | ✅                       |
| variant type requirements   | trait impl | trait impl      | container\<trait impl\> |

*Although `enum_dispatch` supports `From` and `TryInto`, it only works for
concrete types (not in generic contexts).

**Trait objects can only dispatch to functions that can be put in a vtable,
which is annoyingly restrictive.
For instance, traits with generic associated functions can't be put in a
`Box<dyn>`.

All enums are `#[non_exhaustive]` by default, but the matching macros generated
handle wildcard cases and can be used safely in downstream crates.

## Usage Details

Following the same `build_dtype_macros!` invocation from before,

```rust
pub trait Dtype: 'static {}
impl Dtype for i32 {}
impl Dtype for i64 {}
dtype_dispatch::build_dtype_macros!(
  define_an_enum,
  match_an_enum,
  Dtype,
  {
    I32 => i32,
    I64 => i64,
  },
);

// Enum with neither data nor descriminants
// Supports:
// * new
define_an_enum!(
  #[derive(Clone, Debug)]
  pub MyDtype
);

// Enum with descriminants, coming from T::BITS for each type
// Supports:
// * new
// * from_descriminant
define_an_enum!(
  #[derive(Clone, Debug)]
  #[repr(u32)]
  pub MyDescriminatingDtype = BITS
);

// Enum with owned data
// Supports:
// * new
// * downcast
// * downcast_ref
// * downcast_mut
define_an_enum!(
  #[derive(Clone, Debug)]
  pub MyVec(Vec)
);

// Enum with referenced data and lifetime parameter
// Supports:
// * new
// * downcast
type Arr<T> = [T];
define_an_enum!(
  #[derive(Debug)]
  pub MySlice(&Arr)
);

// Enum with mutably referenced data and lifetime parameter
// Supports:
// * new
// * downcast
define_an_enum!(
  #[derive(Debug)]
  pub MySliceMut(&mut Arr)
);

fn main() {
  let my_data = MyVec::new(vec![0_i32, 1, 2, 3]);
  let sliced = match_an_enum!(
    &my_data,
    MyVec<T>(inner) => {
      MySlice::new(&inner[1..3])
    }
  );
  // just to show that the slice worked:
  assert_eq!(format!("{:?}", sliced), "I32([1, 2])");
}
```

## Limitations

At present, enum and container type names must always be a single identifier.
For instance, `Vec` will work, but `std::vec::Vec` and `Vec<Foo>` will not.
You can satisfy this by `use`ing your type or making a type alias of it,
e.g. `type MyContainer<T: MyConstraint> = Vec<Foo<T>>`.

It is also mandatory that you place exactly one attribute when defining each
enum, e.g. with a `#[derive(Clone, Debug)]`.
If you don't want any attributes, you can just do `#[derive()]`.

As of v2.0, we assume the variant<=>trait impl mapping is bijective, so it
highly recommended that users keep their constraint trait either private or
sealed, or else panics will be possible.