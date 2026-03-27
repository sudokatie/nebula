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

    fn eval(&self, _ray_in: &Ray, _hit: &HitRecord, _dir_out: &Vec3) -> Vec3 {
        // Delta distribution - eval is technically infinite at the perfect direction
        // Return 1.0 for the sampled direction (handled specially)
        Vec3::one()
    }

    fn pdf(&self, _ray_in: &Ray, _hit: &HitRecord, _dir_out: &Vec3) -> f32 {
        // Delta distribution has infinite PDF at the exact direction
        // Return 1.0 to cancel with eval in rendering equation
        1.0
    }

    fn is_specular(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_dielectric_scatter() {
        let glass = Dielectric::new(1.5);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let ray = Ray::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = HitRecord {
            t: 1.0,
            point: Vec3::zero(),
            normal: Vec3::new(0.0, 1.0, 0.0),
            uv: (0.0, 0.0),
            front_face: true,
            material_id: 0,
        };
        
        let scatter = glass.scatter(&ray, &hit, &mut rng);
        assert!(scatter.is_some());
    }

    #[test]
    fn test_dielectric_is_specular() {
        let glass = Dielectric::new(1.5);
        assert!(glass.is_specular());
    }

    #[test]
    fn test_reflectance_normal_incidence() {
        // At normal incidence, reflectance should be (n-1)^2 / (n+1)^2
        let r = Dielectric::reflectance(1.0, 1.5);
        let ratio: f32 = (1.5 - 1.0) / (1.5 + 1.0);
        let expected = ratio * ratio;
        assert!((r - expected).abs() < 1e-6);
    }

    #[test]
    fn test_reflectance_grazing_angle() {
        // At grazing angle, reflectance should approach 1
        let r = Dielectric::reflectance(0.0, 1.5);
        assert!(r > 0.99);
    }
}
