//! Math primitives for ray tracing

mod vec3;
mod vec3x4;
mod ray;
mod transform;

pub use vec3::Vec3;
pub use vec3x4::{Vec3x4, Ray4};
pub use ray::{Ray, RayDifferential};
pub use transform::Transform;
