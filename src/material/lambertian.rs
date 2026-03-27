//! Lambertian (diffuse) material

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use crate::sampler::sampling::{cosine_weighted_hemisphere, cosine_weighted_pdf, build_onb, local_to_world};
use super::{Material, ScatterRecord, Texture, SolidColor};
use rand::Rng;
use std::sync::Arc;
use std::f32::consts::PI;

/// Lambertian diffuse material
pub struct Lambertian {
    albedo: Arc<dyn Texture>,
}

impl Lambertian {
    pub fn new(color: Vec3) -> Self {
        Self {
            albedo: Arc::new(SolidColor::new(color)),
        }
    }

    pub fn textured(texture: Arc<dyn Texture>) -> Self {
        Self { albedo: texture }
    }
}

impl Material for Lambertian {
    fn albedo(&self) -> Vec3 {
        // Sample at UV (0,0) for solid colors
        self.albedo.sample(0.0, 0.0, &Vec3::zero())
    }
    
    fn albedo_at(&self, u: f32, v: f32, point: &Vec3, footprint: f32) -> Vec3 {
        self.albedo.sample_lod(u, v, point, footprint)
    }

    fn scatter_dyn(&self, _ray: &Ray, hit: &HitRecord, rng: &mut dyn rand::RngCore) -> Option<ScatterRecord> {
        // Cosine-weighted hemisphere sampling
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        let local_dir = cosine_weighted_hemisphere(u1, u2);
        
        // Transform to world space
        let (u, v, w) = build_onb(hit.normal);
        let direction = local_to_world(local_dir, u, v, w);
        
        let albedo = self.albedo.sample(hit.uv.0, hit.uv.1, &hit.point);
        let cos_theta = direction.dot(&hit.normal).max(0.0);
        let pdf = cosine_weighted_pdf(cos_theta);
        
        Some(ScatterRecord::with_pdf(
            albedo / PI,
            Ray::new(hit.point, direction),
            pdf,
        ))
    }
    
    fn scatter_with_lod(&self, _ray: &Ray, hit: &HitRecord, rng: &mut dyn rand::RngCore, footprint: f32) -> Option<ScatterRecord> {
        // Cosine-weighted hemisphere sampling
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        let local_dir = cosine_weighted_hemisphere(u1, u2);
        
        // Transform to world space
        let (u, v, w) = build_onb(hit.normal);
        let direction = local_to_world(local_dir, u, v, w);
        
        // Use LOD-aware texture sampling
        let albedo = self.albedo.sample_lod(hit.uv.0, hit.uv.1, &hit.point, footprint);
        let cos_theta = direction.dot(&hit.normal).max(0.0);
        let pdf = cosine_weighted_pdf(cos_theta);
        
        Some(ScatterRecord::with_pdf(
            albedo / PI,
            Ray::new(hit.point, direction),
            pdf,
        ))
    }

    fn eval(&self, wi: &Vec3, _wo: &Vec3, normal: &Vec3) -> Vec3 {
        // Lambertian BRDF = albedo / PI (constant, view-independent)
        // Returns BRDF value, NOT multiplied by cos_theta
        let cos_theta = wi.dot(normal).max(0.0);
        if cos_theta <= 0.0 {
            return Vec3::zero();
        }
        Vec3::new(1.0, 1.0, 1.0) / PI
    }

    fn pdf(&self, wi: &Vec3, _wo: &Vec3, normal: &Vec3) -> f32 {
        let cos_theta = wi.dot(normal).max(0.0);
        cosine_weighted_pdf(cos_theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambertian_scatter() {
        let mat = Lambertian::new(Vec3::new(0.5, 0.5, 0.5));
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            t: 1.0,
            point: Vec3::new(0.0, 0.0, -1.0),
            geometric_normal: Vec3::new(0.0, 0.0, 1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            uv: (0.0, 0.0),
            front_face: true,
            material_id: 0,
        };
        
        let mut rng = rand::thread_rng();
        let scatter = mat.scatter_dyn(&ray, &hit, &mut rng);
        assert!(scatter.is_some());
    }

    #[test]
    fn test_lambertian_pdf() {
        let mat = Lambertian::new(Vec3::new(0.5, 0.5, 0.5));
        let wi = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let pdf = mat.pdf(&wi, &wo, &normal);
        assert!((pdf - 1.0 / PI).abs() < 0.01);
    }
}
