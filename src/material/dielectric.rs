//! Dielectric (glass) material

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use crate::sampler::sampling::fresnel_dielectric;
use super::{Material, ScatterRecord};
use rand::Rng;

/// Dielectric (glass/water) material
pub struct Dielectric {
    pub ior: f32,
}

impl Dielectric {
    pub fn new(ior: f32) -> Self {
        Self { ior }
    }

    pub fn glass() -> Self {
        Self::new(1.5)
    }

    pub fn water() -> Self {
        Self::new(1.33)
    }

    pub fn diamond() -> Self {
        Self::new(2.42)
    }

    fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Option<Vec3> {
        let cos_theta = (-uv).dot(&n).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        // Total internal reflection
        if etai_over_etat * sin_theta > 1.0 {
            return None;
        }

        let r_out_perp = (uv + n * cos_theta) * etai_over_etat;
        let r_out_parallel = n * -(1.0 - r_out_perp.length_squared()).abs().sqrt();
        Some(r_out_perp + r_out_parallel)
    }
}

impl Material for Dielectric {
    fn albedo(&self) -> Vec3 {
        Vec3::new(1.0, 1.0, 1.0)
    }

    fn ior(&self) -> f32 {
        self.ior
    }

    fn scatter_dyn(&self, ray: &Ray, hit: &HitRecord, rng: &mut dyn rand::RngCore) -> Option<ScatterRecord> {
        let attenuation = Vec3::new(1.0, 1.0, 1.0);
        let refraction_ratio = if hit.front_face {
            1.0 / self.ior
        } else {
            self.ior
        };

        let unit_direction = ray.direction.normalize();
        let cos_theta = (-unit_direction).dot(&hit.normal).min(1.0);

        // Fresnel reflectance
        let reflectance = fresnel_dielectric(cos_theta, refraction_ratio);

        // Probabilistically choose reflection or refraction
        let direction = if rng.gen::<f32>() < reflectance {
            unit_direction.reflect(&hit.normal)
        } else {
            match Self::refract(unit_direction, hit.normal, refraction_ratio) {
                Some(refracted) => refracted,
                None => unit_direction.reflect(&hit.normal),
            }
        };

        Some(ScatterRecord::new(
            attenuation,
            Ray::new(hit.point, direction),
        ))
    }

    fn is_delta(&self) -> bool {
        true
    }

    fn eval(&self, _wi: &Vec3, _wo: &Vec3, _normal: &Vec3) -> Vec3 {
        // Delta distribution - cannot evaluate BRDF directly
        Vec3::zero()
    }

    fn pdf(&self, _wi: &Vec3, _wo: &Vec3, _normal: &Vec3) -> f32 {
        // Delta distribution - PDF is infinite at exact refraction/reflection direction
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dielectric_glass() {
        let mat = Dielectric::glass();
        assert!((mat.ior - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_dielectric_is_delta() {
        let mat = Dielectric::new(1.5);
        assert!(mat.is_delta());
    }

    #[test]
    fn test_dielectric_scatter() {
        let mat = Dielectric::new(1.5);
        let ray = Ray::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = HitRecord {
            t: 1.0,
            point: Vec3::zero(),
            geometric_normal: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            uv: (0.0, 0.0),
            front_face: true,
            material_id: 0,
        };
        
        let mut rng = rand::thread_rng();
        let scatter = mat.scatter_dyn(&ray, &hit, &mut rng);
        assert!(scatter.is_some());
    }

    #[test]
    fn test_refract() {
        let dir = Vec3::new(0.0, -1.0, 0.0).normalize();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let refracted = Dielectric::refract(dir, n, 1.0 / 1.5);
        assert!(refracted.is_some());
    }
}
