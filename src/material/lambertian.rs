//! Lambertian (diffuse) material

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use super::{Material, ScatterRecord};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Lambertian diffuse material
pub struct Lambertian {
    pub albedo: Vec3,
}

impl Lambertian {
    pub fn new(albedo: Vec3) -> Self {
        Self { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, hit: &HitRecord, rng: &mut Xoshiro256PlusPlus) -> Option<ScatterRecord> {
        let mut scatter_direction = hit.normal + Vec3::random_unit_vector(rng);
        
        // Catch degenerate scatter direction
        if scatter_direction.near_zero() {
            scatter_direction = hit.normal;
        }
        
        Some(ScatterRecord {
            attenuation: self.albedo,
            scattered: Ray::new(hit.point, scatter_direction),
        })
    }
}
