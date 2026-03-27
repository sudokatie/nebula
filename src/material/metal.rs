//! Metal (specular) material with GGX microfacet BRDF

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use crate::sampler::sampling::{ggx_sample, ggx_d, build_onb, local_to_world, fresnel_schlick};
use super::{Material, ScatterRecord, Texture, SolidColor};
use rand::Rng;
use std::sync::Arc;

/// Metal reflective material with GGX microfacet BRDF
pub struct Metal {
    albedo_texture: Arc<dyn Texture>,
    pub roughness: f32,
}

impl Metal {
    pub fn new(albedo: Vec3, roughness: f32) -> Self {
        Self {
            albedo_texture: Arc::new(SolidColor::new(albedo)),
            roughness: roughness.clamp(0.001, 1.0),
        }
    }

    pub fn textured(texture: Arc<dyn Texture>, roughness: f32) -> Self {
        Self {
            albedo_texture: texture,
            roughness: roughness.clamp(0.001, 1.0),
        }
    }

    pub fn mirror(albedo: Vec3) -> Self {
        Self::new(albedo, 0.001)
    }

    /// Get albedo at given UV and point
    fn sample_albedo(&self, u: f32, v: f32, point: &Vec3) -> Vec3 {
        self.albedo_texture.sample(u, v, point)
    }
    
    /// Get albedo at given UV and point with LOD
    fn sample_albedo_lod(&self, u: f32, v: f32, point: &Vec3, footprint: f32) -> Vec3 {
        self.albedo_texture.sample_lod(u, v, point, footprint)
    }

    /// Smith's geometry function G1 for GGX
    fn ggx_g1(n_dot_v: f32, roughness: f32) -> f32 {
        let a = roughness * roughness;
        let a2 = a * a;
        let n_dot_v2 = n_dot_v * n_dot_v;
        let denom = n_dot_v + (a2 + (1.0 - a2) * n_dot_v2).sqrt();
        2.0 * n_dot_v / denom
    }

    /// Smith's geometry function G for GGX (separable form)
    fn ggx_g(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        Self::ggx_g1(n_dot_v, roughness) * Self::ggx_g1(n_dot_l, roughness)
    }
}

impl Material for Metal {
    fn albedo(&self) -> Vec3 {
        // Return default sample for materials that don't have UV context
        self.albedo_texture.sample(0.5, 0.5, &Vec3::zero())
    }

    fn roughness(&self) -> f32 {
        self.roughness
    }

    fn scatter_dyn(&self, ray: &Ray, hit: &HitRecord, rng: &mut dyn rand::RngCore) -> Option<ScatterRecord> {
        let v = (-ray.direction).normalize();
        let n = hit.normal;
        let albedo = self.sample_albedo(hit.uv.0, hit.uv.1, &hit.point);
        
        if self.roughness < 0.01 {
            // Perfect mirror - delta distribution
            let reflected = ray.direction.normalize().reflect(&n);
            return Some(ScatterRecord::new(
                albedo,
                Ray::new(hit.point, reflected),
            ));
        }
        
        // GGX importance sampling of the half-vector
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        let h_local = ggx_sample(u1, u2, self.roughness);
        let (u, vv, w) = build_onb(n);
        let h = local_to_world(h_local, u, vv, w);
        
        // Reflect view direction around half vector
        let wi = (h * 2.0 * v.dot(&h) - v).normalize();
        
        let n_dot_l = n.dot(&wi);
        if n_dot_l <= 0.0 {
            return None;
        }
        
        let n_dot_v = n.dot(&v).max(0.001);
        let n_dot_h = n.dot(&h).max(0.0);
        let v_dot_h = v.dot(&h).max(0.0);
        
        // Compute full Cook-Torrance BRDF
        let d = ggx_d(n_dot_h, self.roughness);
        let g = Self::ggx_g(n_dot_v, n_dot_l, self.roughness);
        let f = fresnel_schlick(v_dot_h, albedo);
        
        // BRDF = D * G * F / (4 * n_dot_v * n_dot_l)
        let brdf = f * (d * g / (4.0 * n_dot_v * n_dot_l + 0.0001));
        
        // PDF for sampling half-vector, converted to solid angle
        let pdf = d * n_dot_h / (4.0 * v_dot_h + 0.0001);
        
        Some(ScatterRecord::with_pdf(
            brdf * n_dot_l,
            Ray::new(hit.point, wi),
            pdf,
        ))
    }

    fn is_delta(&self) -> bool {
        self.roughness < 0.01
    }
    
    fn albedo_at(&self, u: f32, v: f32, point: &Vec3, footprint: f32) -> Vec3 {
        self.sample_albedo_lod(u, v, point, footprint)
    }
    
    fn scatter_with_lod(&self, ray: &Ray, hit: &HitRecord, rng: &mut dyn rand::RngCore, footprint: f32) -> Option<ScatterRecord> {
        let v = (-ray.direction).normalize();
        let n = hit.normal;
        let albedo = self.sample_albedo_lod(hit.uv.0, hit.uv.1, &hit.point, footprint);
        
        if self.roughness < 0.01 {
            // Perfect mirror - delta distribution
            let reflected = ray.direction.normalize().reflect(&n);
            return Some(ScatterRecord::new(
                albedo,
                Ray::new(hit.point, reflected),
            ));
        }
        
        // GGX importance sampling of the half-vector
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        let h_local = ggx_sample(u1, u2, self.roughness);
        let (u, vv, w) = build_onb(n);
        let h = local_to_world(h_local, u, vv, w);
        
        // Reflect view direction around half vector
        let wi = (h * 2.0 * v.dot(&h) - v).normalize();
        
        let n_dot_l = n.dot(&wi);
        if n_dot_l <= 0.0 {
            return None;
        }
        
        let n_dot_v = n.dot(&v).max(0.001);
        let n_dot_h = n.dot(&h).max(0.0);
        let v_dot_h = v.dot(&h).max(0.0);
        
        // Compute full Cook-Torrance BRDF
        let d = ggx_d(n_dot_h, self.roughness);
        let g = Self::ggx_g(n_dot_v, n_dot_l, self.roughness);
        let f = fresnel_schlick(v_dot_h, albedo);
        
        // BRDF = D * G * F / (4 * n_dot_v * n_dot_l)
        let brdf = f * (d * g / (4.0 * n_dot_v * n_dot_l + 0.0001));
        
        // PDF for sampling half-vector, converted to solid angle
        let pdf = d * n_dot_h / (4.0 * v_dot_h + 0.0001);
        
        Some(ScatterRecord::with_pdf(
            brdf * n_dot_l,
            Ray::new(hit.point, wi),
            pdf,
        ))
    }

    fn eval(&self, wi: &Vec3, wo: &Vec3, normal: &Vec3) -> Vec3 {
        if self.roughness < 0.01 {
            // Delta distribution - can't evaluate directly
            return Vec3::zero();
        }
        
        let n = *normal;
        let v = *wo; // View direction (outgoing)
        let l = *wi; // Light direction (incoming)
        
        let n_dot_l = n.dot(&l);
        let n_dot_v = n.dot(&v);
        
        if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
            return Vec3::zero();
        }
        
        // Compute half vector
        let h = (v + l).normalize();
        let n_dot_h = n.dot(&h).max(0.0);
        let v_dot_h = v.dot(&h).max(0.0);
        
        // Get default albedo (no UV context in eval)
        let albedo = self.albedo();
        
        // Evaluate GGX BRDF
        let d = ggx_d(n_dot_h, self.roughness);
        let g = Self::ggx_g(n_dot_v, n_dot_l, self.roughness);
        let f = fresnel_schlick(v_dot_h, albedo);
        
        f * (d * g / (4.0 * n_dot_v * n_dot_l + 0.0001))
    }

    fn pdf(&self, wi: &Vec3, wo: &Vec3, normal: &Vec3) -> f32 {
        if self.roughness < 0.01 {
            // Delta distribution - infinite PDF at exact reflection
            return 0.0;
        }
        
        let n = *normal;
        let v = *wo;
        let l = *wi;
        
        let n_dot_l = n.dot(&l);
        if n_dot_l <= 0.0 {
            return 0.0;
        }
        
        // Compute half vector
        let h = (v + l).normalize();
        let n_dot_h = n.dot(&h).max(0.0);
        let v_dot_h = v.dot(&h).max(0.0);
        
        // PDF = D(h) * n_dot_h / (4 * v_dot_h)
        let d = ggx_d(n_dot_h, self.roughness);
        d * n_dot_h / (4.0 * v_dot_h + 0.0001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_mirror() {
        let mat = Metal::mirror(Vec3::new(1.0, 1.0, 1.0));
        assert!(mat.is_delta());
    }

    #[test]
    fn test_metal_rough() {
        let mat = Metal::new(Vec3::new(0.8, 0.6, 0.2), 0.5);
        assert!(!mat.is_delta());
    }

    #[test]
    fn test_metal_textured() {
        let tex = Arc::new(SolidColor::new(Vec3::new(1.0, 0.8, 0.0)));
        let mat = Metal::textured(tex, 0.3);
        let albedo = mat.albedo();
        assert!((albedo.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_metal_scatter() {
        let mat = Metal::new(Vec3::new(0.8, 0.8, 0.8), 0.3);
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
    fn test_metal_eval_pdf_consistency() {
        let mat = Metal::new(Vec3::new(0.8, 0.8, 0.8), 0.3);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.3, 0.9, 0.1).normalize();
        let wi = Vec3::new(-0.2, 0.8, 0.3).normalize();
        
        let brdf = mat.eval(&wi, &wo, &normal);
        let pdf = mat.pdf(&wi, &wo, &normal);
        
        // Both should be positive for valid directions
        assert!(brdf.x >= 0.0);
        assert!(pdf >= 0.0);
    }
}
