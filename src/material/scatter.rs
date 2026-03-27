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

    /// Evaluate BSDF for given incident and outgoing directions
    /// Returns the BSDF value (f(wi, wo))
    fn eval(&self, _ray_in: &Ray, hit: &HitRecord, _dir_out: &Vec3) -> Vec3 {
        // Default: Lambertian BSDF (albedo / pi)
        // Most materials should override this
        let _ = hit;
        Vec3::new(0.5, 0.5, 0.5) / std::f32::consts::PI
    }

    /// PDF for sampling the given outgoing direction
    fn pdf(&self, _ray_in: &Ray, hit: &HitRecord, dir_out: &Vec3) -> f32 {
        // Default: cosine-weighted hemisphere PDF
        let cos_theta = hit.normal.dot(dir_out).max(0.0);
        cos_theta / std::f32::consts::PI
    }

    /// Whether this material has specular (delta distribution) reflection
    fn is_specular(&self) -> bool {
        false
    }

    /// Whether this material is emissive
    fn is_emissive(&self) -> bool {
        self.emit().length_squared() > 0.0
    }
}
