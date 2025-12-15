`dtype_dispatch` solves the problem of interop between *generic* and
*dynamically typed (enum)* containers.

This is a common problem in numerical libraries (think numpy, torch, polars):
you have a variety of data types and data structures to hold them, but every
function involves matching an enum or converting from a generic to an enum.

Example with `i32` and `f32` data types for dynamically-typed vectors,
supporting `.length()` and `.add(other)` operations, plus generic
`new` and `downcast` functions:

```rust
pub trait Dtype: 'static {}
impl Dtype for i32 {}
impl Dtype for f32 {}

// register our two macros, `define_an_enum` and `match_an_enum`, constrained
// to the `Dtype` trait, with our variant => type mapping:
dtype_dispatch::build_dtype_macros!(
  define_an_enum,
  match_an_enum,
  Dtype,
  {
    I32 => i32,
    F32 => f32,
  },
);

// define any enum holding a Vec of any data type!
define_an_enum!(
  #[derive(Clone, Debug)]
  DynArray(Vec)
);

impl DynArray {
  pub fn length(&self) -> usize {
    match_an_enum!(self, DynArray<T>(inner) => { inner.len() })
  }

  pub fn add(&self, other: &DynArray) -> DynArray {
    match_an_enum!(self, DynArray<T>(inner) => {
      let other_inner = other.downcast_ref::<T>().unwrap();
      let added = inner.iter().zip(other_inner).map(|(a, b)| a + b).collect::<Vec<_>>();
      DynArray::new(added).unwrap()
    })
  }
}

// we could also use `DynArray::I32()` here, but just to show we can convert generics:
let x_dynamic = DynArray::new(vec![1_i32, 2, 3]).unwrap();
let x_doubled_generic = x_dynamic.add(&x_dynamic).downcast::<i32>().unwrap();
assert_eq!(x_doubled_generic, vec![2, 4, 6]);
```

Compare this with the same API written manually:

```rust
use std::{any, mem};

pub trait Dtype: 'static {}
impl Dtype for i32 {}
impl Dtype for f32 {}

#[derive(Clone, Debug)]
pub enum DynArray {
  I32(Vec<i32>),
  F32(Vec<f32>),
}

impl DynArray {
  pub fn length(&self) -> usize {
    match self {
      DynArray::I32(inner) => inner.len(),
      DynArray::F32(inner) => inner.len(),
    }
  }

  pub fn add(&self, other: &DynArray) -> DynArray {
    match (self, other) {
      (DynArray::I32(inner), DynArray::I32(other_inner)) => {
        let added = inner.iter().zip(other_inner).map(|(&a, &b)| a + b).collect::<Vec<_>>();
        DynArray::I32(added)
      }
      (DynArray::F32(inner), DynArray::F32(other_inner)) => {
        let added = inner.iter().zip(other_inner).map(|(&a, &b)| a + b).collect::<Vec<_>>();
        DynArray::F32(added)
      }
      _ => panic!("mismatched dtypes")
    }
  }

  pub fn new<T: Dtype>(inner: Vec<T>) -> DynArray {
    let type_id = any::TypeId::of::<T>();
    if type_id == any::TypeId::of::<i32>() {
      DynArray::I32(unsafe { mem::transmute(inner) })
    } else if type_id == any::TypeId::of::<f32>() {
      DynArray::F32(unsafe { mem::transmute(inner) })
    } else {
      panic!("unknown dtype")
    }
  }

  pub fn downcast<T: Dtype>(self) -> Vec<T> {
    let type_id = any::TypeId::of::<T>();
    match self {
      DynArray::I32(inner) => {
        if type_id == any::TypeId::of::<i32>() {
          unsafe { mem::transmute(inner) }
        } else {
          panic!("incorrect dtype")
        }
      }
      DynArray::F32(inner) => {
        if type_id == any::TypeId::of::<f32>() {
          unsafe { mem::transmute(inner) }
        } else {
          panic!("incorrect dtype")
        }
      }
    }
  }
}

let x_dynamic = DynArray::new(vec![1_i32, 2, 3]);
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

## Limitations

At present, enum and container type names must always be a single identifier.
For instance, `Vec` will work, but `std::vec::Vec` and `Vec<Foo>` will not.
You can satisfy this by `use`ing your type or making a type alias of it,
e.g. `type MyContainer<T: MyConstraint> = Vec<Foo<T>>`.

It is also mandatory that you place exactly one attribute when defining each
enum, e.g. with a `#[derive(Clone, Debug)]`.
If you don't want any attributes, you can just do `#[derive()]`.
