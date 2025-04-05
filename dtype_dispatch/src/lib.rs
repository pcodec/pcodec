#![doc = include_str!("../README.md")]
#![allow(unreachable_patterns)]

/// Produces two macros: an enum definer and an enum matcher.
///
/// See the crate-level documentation for more info.
#[macro_export]
macro_rules! build_dtype_macros {
  (
    $(#[$definer_attrs: meta])*
    $definer: ident,
    $(#[$matcher_attrs: meta])*
    $matcher: ident,
    $constraint: path,
    {$($variant: ident => $t: ty,)+}$(,)?
  ) => {
    $(#[$definer_attrs])*
    macro_rules! $definer {
      (#[$enum_attrs: meta] $vis: vis $name: ident) => {
        #[$enum_attrs]
        #[non_exhaustive]
        $vis enum $name {
          $($variant,)+
        }

        impl $name {
          #[inline]
          pub fn new<T: $constraint>() -> Option<Self> {
            let type_id = std::any::TypeId::of::<T>();
            $(
              if type_id == std::any::TypeId::of::<$t>() {
                return Some($name::$variant);
              }
            )+
            None
          }
        }
      };
      (#[$enum_attrs: meta] #[repr($desc_t: ty)] $vis: vis $name: ident = $desc_val: ident) => {
        #[$enum_attrs]
        #[repr($desc_t)]
        #[non_exhaustive]
        $vis enum $name {
          $($variant = <$t>::$desc_val,)+
        }

        impl $name {
          #[inline]
          pub fn new<T: $constraint>() -> Option<Self> {
            let type_id = std::any::TypeId::of::<T>();
            $(
              if type_id == std::any::TypeId::of::<$t>() {
                return Some($name::$variant);
              }
            )+
            None
          }

          pub fn from_descriminant(desc: $desc_t) -> Option<Self> {
            match desc {
              $(<$t>::$desc_val => Some(Self::$variant),)+
              _ => None
            }
          }
        }
      };
      (#[$enum_attrs: meta] $vis: vis $name: ident($container: ident)) => {
        #[$enum_attrs]
        #[non_exhaustive]
        $vis enum $name {
          $($variant($container<$t>),)+
        }

        impl $name {
          #[inline]
          pub fn new<S: $constraint>(inner: $container<S>) -> Option<Self> {
            let type_id = std::any::TypeId::of::<S>();
            $(
              if type_id == std::any::TypeId::of::<$t>() {
                // Transmute doesn't work for containers whose size depends on T,
                // so we use a hack from
                // https://users.rust-lang.org/t/transmuting-a-generic-array/45645/6
                let ptr = &inner as *const $container<S> as *const $container<$t>;
                let typed = unsafe { ptr.read() };
                std::mem::forget(inner);
                return Some($name::$variant(typed));
              }
            )+
            None
          }

          pub fn downcast<T: $constraint>(self) -> Option<$container<T>> {
            match self {
              $(
                Self::$variant(inner) => {
                  if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$t>() {
                    // same hack from `new`
                    let ptr = &inner as *const $container<$t> as *const $container<T>;
                    let typed = unsafe { ptr.read() };
                    std::mem::forget(inner);
                    Some(typed)
                  } else {
                    None
                  }
                }
              )+
            }
          }

          pub fn downcast_ref<T: $constraint>(&self) -> Option<&$container<T>> {
            match self {
              $(
                Self::$variant(inner) => {
                  if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$t>() {
                    unsafe {
                      Some(std::mem::transmute::<_, &$container<T>>(inner))
                    }
                  } else {
                    None
                  }
                }
              )+
            }
          }

          pub fn downcast_mut<T: $constraint>(&mut self) -> Option<&mut $container<T>> {
            match self {
              $(
                Self::$variant(inner) => {
                  if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$t>() {
                    unsafe {
                      Some(std::mem::transmute::<_, &mut $container<T>>(inner))
                    }
                  } else {
                    None
                  }
                }
              )+
            }
          }
        }
      };
    }

    $(#[$matcher_attrs])*
    macro_rules! $matcher {
      ($value: expr, $enum_: ident<$generic: ident> => $block: block) => {
        match $value {
          $($enum_::$variant => {
            type $generic = $t;
            $block
          })+
          _ => unreachable!()
        }
      };
      ($value: expr, $enum_: ident<$generic: ident>($inner: ident) => $block: block) => {
        match $value {
          $($enum_::$variant($inner) => {
            type $generic = $t;
            $block
          })+
          _ => unreachable!()
        }
      };
    }
  };
}

#[allow(dead_code)]
#[cfg(test)]
mod tests {
  use std::collections::HashMap;

  trait Constraint: 'static {}

  impl Constraint for u16 {}
  impl Constraint for u32 {}
  impl Constraint for u64 {}

  build_dtype_macros!(
    define_enum,
    match_enum,
    crate::tests::Constraint,
    {
      U16 => u16,
      U32 => u32,
      U64 => u64,
    }
  );

  define_enum!(
    #[derive(Clone, Debug)]
    MyEnum(Vec)
  );

  type Counter<T> = HashMap<T, usize>;

  define_enum!(
    #[derive(Clone, Debug)]
    AnotherContainerEnumInSameScope(Counter)
  );

  // we use this helper just to prove that we can handle generic types, not
  // just concrete types
  fn generic_new<T: Constraint>(inner: Vec<T>) -> MyEnum {
    MyEnum::new(inner).unwrap()
  }

  #[test]
  fn test_end_to_end() {
    let x = generic_new(vec![1_u16, 1, 2, 3, 5]);
    let bit_size = match_enum!(&x, MyEnum<L>(inner) => { inner.len() * L::BITS as usize });
    assert_eq!(bit_size, 80);
    let x = x.downcast::<u16>().unwrap();
    assert_eq!(x[0], 1);
  }

  #[test]
  fn test_multiple_enums_defined_in_same_scope() {
    // This was really tested during compilation, but I'm just using the new
    // enum here to ensure the code doesn't die.
    AnotherContainerEnumInSameScope::new(HashMap::<u16, usize>::new()).unwrap();
  }
}
