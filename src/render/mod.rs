//! Rendering backends

pub mod cpu;
pub mod gpu;

pub use cpu::CpuRenderer;
pub use gpu::{GpuRenderer, GpuConfig};
