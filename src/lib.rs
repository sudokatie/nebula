//! Nebula - A physically-based path tracer
//!
//! Renders 3D scenes using unbiased Monte Carlo path tracing.
//! Supports CPU rendering with SIMD acceleration and optional GPU compute.

pub mod math;
pub mod geometry;
pub mod accel;
pub mod material;
pub mod camera;
pub mod sampler;
pub mod integrator;
pub mod scene;
pub mod render;
pub mod output;

// Re-exports for convenience
pub use math::{Vec3, Ray};
pub use geometry::{HitRecord, Hittable, Sphere, Triangle};
pub use accel::{AABB, BVH};
pub use material::{Material, Lambertian, Metal, Dielectric, Emissive};
pub use camera::Camera;
pub use integrator::PathIntegrator;
pub use scene::Scene;
pub use render::CpuRenderer;
