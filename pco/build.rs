use std::env;

const DISMISS_WARNINGS_VAR: &str = "PCO_DISMISS_BUILD_WARNINGS";

fn main() {
  let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
  let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
  let target_features: Vec<&str> = target_features.split(',').collect();

  if target_arch == "x86_64"
    && env::var("PROFILE").unwrap_or_default() == "release"
    && env::var(DISMISS_WARNINGS_VAR).unwrap_or_default() != "1"
  {
    let missing_instructions: Vec<&str> = ["bmi1", "bmi2", "lzcnt", "avx2"]
      .into_iter()
      .filter(|feature| !target_features.contains(feature))
      .collect();

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
