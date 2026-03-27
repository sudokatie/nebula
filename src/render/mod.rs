//! Rendering backends

mod cpu;

pub use cpu::CpuRenderer;

#[cfg(feature = "gpu")]
mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::{
    GpuRenderer, GpuConfig, SceneValidation,
    GpuSphere, GpuTriangle, GpuMaterial,
    MAX_SPHERES, MAX_TRIANGLES, MAX_BVH_NODES, MAX_MATERIALS
};
