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
        // Use cosine-weighted hemisphere sampling for importance sampling
        let (dir, _pdf) = crate::integrator::sample_cosine_hemisphere(&hit.normal, rng);
        
        // Catch degenerate scatter direction
        let scatter_direction = if dir.near_zero() {
            hit.normal
        } else {
            dir
        };
        
        Some(ScatterRecord {
            attenuation: self.albedo,
            scattered: Ray::new(hit.point, scatter_direction),
        })
    }

    fn eval(&self, _ray_in: &Ray, _hit: &HitRecord, _dir_out: &Vec3) -> Vec3 {
        // Lambertian BSDF: albedo / pi
        self.albedo / std::f32::consts::PI
    }

    fn pdf(&self, _ray_in: &Ray, hit: &HitRecord, dir_out: &Vec3) -> f32 {
        // Cosine-weighted hemisphere PDF
        let cos_theta = hit.normal.dot(dir_out).max(0.0);
        cos_theta / std::f32::consts::PI
    }

    fn is_specular(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_lambertian_scatter() {
        let mat = Lambertian::new(Vec3::new(0.5, 0.5, 0.5));
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
        
        let scatter = mat.scatter(&ray, &hit, &mut rng);
        assert!(scatter.is_some());
        
        // Scattered ray should be in upper hemisphere
        let s = scatter.unwrap();
        assert!(s.scattered.direction.dot(&hit.normal) >= 0.0);
    }

    #[test]
    fn test_lambertian_eval() {
        let albedo = Vec3::new(0.8, 0.2, 0.1);
        let mat = Lambertian::new(albedo);
        
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, -1.0, 0.0));
        let hit = HitRecord {
            t: 1.0,
            point: Vec3::zero(),
            normal: Vec3::new(0.0, 1.0, 0.0),
            uv: (0.0, 0.0),
            front_face: true,
            material_id: 0,
        };
        let dir_out = Vec3::new(0.0, 1.0, 0.0);
        
        let bsdf = mat.eval(&ray, &hit, &dir_out);
        let expected = albedo / std::f32::consts::PI;
        
        assert!((bsdf.x - expected.x).abs() < 1e-6);
        assert!((bsdf.y - expected.y).abs() < 1e-6);
        assert!((bsdf.z - expected.z).abs() < 1e-6);
    }

    #[test]
    fn test_lambertian_pdf() {
        let mat = Lambertian::new(Vec3::one());
        
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, -1.0, 0.0));
        let hit = HitRecord {
            t: 1.0,
            point: Vec3::zero(),
            normal: Vec3::new(0.0, 1.0, 0.0),
            uv: (0.0, 0.0),
            front_face: true,
            material_id: 0,
        };
        
        // At normal direction, cos_theta = 1
        let pdf = mat.pdf(&ray, &hit, &Vec3::new(0.0, 1.0, 0.0));
        assert!((pdf - 1.0 / std::f32::consts::PI).abs() < 1e-6);
        
        // At grazing angle, cos_theta = 0
        let pdf = mat.pdf(&ray, &hit, &Vec3::new(1.0, 0.0, 0.0));
        assert!(pdf < 1e-6);
    }

    #[test]
    fn test_lambertian_not_specular() {
        let mat = Lambertian::new(Vec3::one());
        assert!(!mat.is_specular());
    }
}
