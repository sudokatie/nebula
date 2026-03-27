//! Mathematical primitives for ray tracing

mod vec3;
mod ray;
mod transform;
mod simd;

pub use vec3::Vec3;
pub use ray::{Ray, RayDifferential};
pub use transform::Transform;
pub use simd::Vec3x4;
