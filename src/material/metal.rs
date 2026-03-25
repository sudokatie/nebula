//! Metal (specular) material

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use super::{Material, ScatterRecord};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Metal reflective material
pub struct Metal {
    pub albedo: Vec3,
    pub roughness: f32,
}

impl Metal {
    pub fn new(albedo: Vec3, roughness: f32) -> Self {
        Self {
            albedo,
            roughness: roughness.clamp(0.0, 1.0),
        }
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitRecord, rng: &mut Xoshiro256PlusPlus) -> Option<ScatterRecord> {
        let reflected = ray.direction.normalize().reflect(&hit.normal);
        let scattered = Ray::new(
            hit.point,
            reflected + Vec3::random_in_unit_sphere(rng) * self.roughness,
        );
        
        // Only scatter if reflected ray is on same side as normal
        if scattered.direction.dot(&hit.normal) > 0.0 {
            Some(ScatterRecord {
                attenuation: self.albedo,
                scattered,
            })
        } else {
            None
        }
    }
}
