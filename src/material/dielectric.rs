//! Dielectric (glass) material

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use super::{Material, ScatterRecord};
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Dielectric (glass/water) material
pub struct Dielectric {
    /// Index of refraction
    pub ior: f32,
}

impl Dielectric {
    pub fn new(ior: f32) -> Self {
        Self { ior }
    }

    /// Schlick approximation for reflectance
    fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
        let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit: &HitRecord, rng: &mut Xoshiro256PlusPlus) -> Option<ScatterRecord> {
        let refraction_ratio = if hit.front_face {
            1.0 / self.ior
        } else {
            self.ior
        };

        let unit_direction = ray.direction.normalize();
        let cos_theta = (-unit_direction).dot(&hit.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        // Check for total internal reflection
        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let should_reflect = Self::reflectance(cos_theta, refraction_ratio) > rng.gen::<f32>();

        let direction = if cannot_refract || should_reflect {
            unit_direction.reflect(&hit.normal)
        } else {
            unit_direction.refract(&hit.normal, refraction_ratio).unwrap_or_else(|| {
                unit_direction.reflect(&hit.normal)
            })
        };

        Some(ScatterRecord {
            attenuation: Vec3::one(),
            scattered: Ray::new(hit.point, direction),
        })
    }
}
