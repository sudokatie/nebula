//! Nebula - A physically-based path tracer
//!
//! # Example
//! ```ignore
//! use nebula::prelude::*;
//!
//! let mut scene = Scene::new();
//! let mat = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5))));
//! scene.add_sphere(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, mat));
//! scene.build_bvh();
//!
//! let camera = Camera::new(
//!     Vec3::new(0.0, 0.0, 0.0),
//!     Vec3::new(0.0, 0.0, -1.0),
//!     Vec3::new(0.0, 1.0, 0.0),
//!     60.0, 1.0, 0.0, 1.0
//! );
//!
//! let renderer = CpuRenderer::new(800, 600, 100, 50);
//! let pixels = renderer.render(&scene, &camera);
//! ```

pub mod accel;
pub mod camera;
pub mod geometry;
pub mod integrator;
pub mod material;
pub mod math;
pub mod output;
pub mod render;
pub mod sampler;
pub mod scene;
pub mod volume;

/// Prelude - commonly used types
pub mod prelude {
    pub use crate::math::{Vec3, Vec3x4, Ray, RayDifferential, Transform};
    pub use crate::geometry::{Sphere, Triangle, Mesh, Instance, HitRecord, Hittable};
    pub use crate::material::{
        Material, MaterialExt, Lambertian, Metal, Dielectric, Emissive,
        Texture, SolidColor, Checker, ScatterRecord,
    };
    pub use crate::camera::Camera;
    pub use crate::scene::Scene;
    pub use crate::accel::{AABB, BVH, RayPacket, HitPacket};
    pub use crate::render::{CpuRenderer, GpuRenderer, GpuConfig};
    pub use crate::integrator::PathIntegrator;
    pub use crate::output::{
        ToneMap, save_png, save_ppm, save_hdr, save_exr,
        bilateral_filter, adaptive_bilateral, joint_bilateral, DenoiseConfig,
    };
    pub use crate::volume::{
        Volume, VolumeSample, Transmittance,
        HomogeneousVolume, HeterogeneousVolume,
        PhaseFunction, HenyeyGreenstein, Isotropic,
    };
}
