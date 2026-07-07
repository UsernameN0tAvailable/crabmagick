#[cfg(all(target_arch = "x86_64", any(target_arch = "x86", target_arch = "x86_64")))]
pub use crate::zune_jpeg::unsafe_utils_avx2::*;
#[cfg(all(target_arch = "aarch64", target_arch = "aarch64"))]
pub use crate::zune_jpeg::unsafe_utils_neon::*;
