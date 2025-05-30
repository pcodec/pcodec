use std::env;

const DISMISS_WARNINGS_VAR: &str = "PCO_DISMISS_BUILD_WARNINGS";

fn main() {
  if cfg!(target_arch = "x86_64")
    && env::var("PROFILE").unwrap_or("".to_string()) == "release"
    && env::var(DISMISS_WARNINGS_VAR).unwrap_or("".to_string()) != "1"
  {
    let mut missing_instructions = Vec::new();
    if !cfg!(target_feature = "bmi1") {
      missing_instructions.push("bmi1");
    }
    if !cfg!(target_feature = "bmi2") {
      missing_instructions.push("bmi2");
    }
    if !cfg!(target_feature = "avx2") {
      missing_instructions.push("avx2");
    }

    if !missing_instructions.is_empty() {
      println!(
        "cargo:warning=[pco] Building on x64 in release mode without the \
        following instruction sets: {}. \
        This can substantially hinder performance. \
        To fix: follow the instructions at \
        https://github.com/pcodec/pcodec/tree/main/pco. \
        To ignore: set {}=1",
        missing_instructions.join(", "),
        DISMISS_WARNINGS_VAR,
      );
    }
  }
}
