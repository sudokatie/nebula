//! Material trait and scatter record

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Record of scattered ray
pub struct ScatterRecord {
    pub attenuation: Vec3,
    pub scattered: Ray,
}

/// Trait for materials (uses concrete RNG for dyn compatibility)
pub trait Material: Send + Sync {
    /// Scatter incoming ray at hit point
    fn scatter(&self, ray: &Ray, hit: &HitRecord, rng: &mut Xoshiro256PlusPlus) -> Option<ScatterRecord>;
    
    /// Emitted light (default: none)
    fn emit(&self) -> Vec3 {
        Vec3::zero()
    }
}
