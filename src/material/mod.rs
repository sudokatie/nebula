//! Materials and BSDFs

mod dielectric;
mod emissive;
mod lambertian;
mod metal;
mod scatter;
mod texture;

pub use dielectric::Dielectric;
pub use emissive::Emissive;
pub use lambertian::Lambertian;
pub use metal::Metal;
pub use scatter::ScatterRecord;
pub use texture::{Texture, SolidColor, Checker, ImageTexture, NoiseTexture};

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use rand::Rng;
use std::f32::consts::PI;

/// Material trait with BSDF evaluation
/// Using dynamic dispatch for RNG to make trait dyn-compatible
pub trait Material: Send + Sync {
    /// Scatter ray at hit point, return attenuation and new ray
    fn scatter_dyn(&self, ray: &Ray, hit: &HitRecord, rng: &mut dyn rand::RngCore) -> Option<ScatterRecord>;
    
    /// Scatter with texture LOD (footprint from ray differentials)
    /// Default implementation ignores footprint
    fn scatter_with_lod(&self, ray: &Ray, hit: &HitRecord, rng: &mut dyn rand::RngCore, _footprint: f32) -> Option<ScatterRecord> {
        self.scatter_dyn(ray, hit, rng)
    }
    
    /// Get the base albedo/color of the material
    fn albedo(&self) -> Vec3 {
        Vec3::new(0.5, 0.5, 0.5)
    }
    
    /// Get albedo at UV with LOD for texture filtering
    fn albedo_at(&self, _u: f32, _v: f32, _point: &Vec3, _footprint: f32) -> Vec3 {
        self.albedo()
    }
    
    /// Get roughness (0.0 = mirror, 1.0 = fully rough)
    fn roughness(&self) -> f32 {
        0.0
    }
    
    /// Get index of refraction (for dielectrics)
    fn ior(&self) -> f32 {
        1.5
    }
    
    /// Emission for emissive materials
    fn emit(&self) -> Vec3 {
        Vec3::zero()
    }
    
    /// Emission at a specific UV and point (for textured lights)
    fn emit_at(&self, _u: f32, _v: f32, _point: &Vec3) -> Vec3 {
        self.emit()
    }
    
    /// Is this a delta distribution (perfect mirror/glass)?
    fn is_delta(&self) -> bool {
        false
    }
    
    /// Evaluate BSDF for given incoming direction (light), outgoing direction (view), and normal
    /// wi: incoming light direction (toward the surface)
    /// wo: outgoing view direction (away from surface)
    /// normal: surface normal
    fn eval(&self, wi: &Vec3, wo: &Vec3, normal: &Vec3) -> Vec3 {
        // Default: Lambertian BSDF (view-independent)
        let _ = wo; // Lambertian doesn't depend on view direction
        let cos_theta = wi.dot(normal).max(0.0);
        Vec3::new(1.0, 1.0, 1.0) * cos_theta / PI
    }
    
    /// PDF for sampling direction wi given outgoing direction wo
    fn pdf(&self, wi: &Vec3, wo: &Vec3, normal: &Vec3) -> f32 {
        // Default: cosine-weighted hemisphere (view-independent)
        let _ = wo;
        let cos_theta = wi.dot(normal).max(0.0);
        cos_theta / PI
    }
}

/// Extension trait for generic RNG
pub trait MaterialExt: Material {
    fn scatter<R: Rng>(&self, ray: &Ray, hit: &HitRecord, rng: &mut R) -> Option<ScatterRecord> {
        self.scatter_dyn(ray, hit, rng)
    }
}

impl<T: Material + ?Sized> MaterialExt for T {}

/// GGX/Cook-Torrance material for PBR
pub struct CookTorrance {
    pub albedo: Vec3,
    pub roughness: f32,
    pub metallic: f32,
}

impl CookTorrance {
    pub fn new(albedo: Vec3, roughness: f32, metallic: f32) -> Self {
        Self {
            albedo,
            roughness: roughness.clamp(0.01, 1.0),
            metallic: metallic.clamp(0.0, 1.0),
        }
    }

    fn ggx_d(&self, n_dot_h: f32) -> f32 {
        let a = self.roughness * self.roughness;
        let a2 = a * a;
        let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
        a2 / (PI * denom * denom)
    }

    fn ggx_g1(&self, n_dot_v: f32) -> f32 {
        let a = self.roughness * self.roughness;
        let k = a / 2.0;
        n_dot_v / (n_dot_v * (1.0 - k) + k)
    }

    fn ggx_g(&self, n_dot_v: f32, n_dot_l: f32) -> f32 {
        self.ggx_g1(n_dot_v) * self.ggx_g1(n_dot_l)
    }

    fn fresnel(&self, cos_theta: f32) -> Vec3 {
        let f0 = Vec3::new(0.04, 0.04, 0.04) * (1.0 - self.metallic)
            + self.albedo * self.metallic;
        let t = (1.0 - cos_theta).max(0.0).powi(5);
        f0 + (Vec3::new(1.0, 1.0, 1.0) - f0) * t
    }
}

impl Material for CookTorrance {
    fn albedo(&self) -> Vec3 {
        self.albedo
    }

    fn roughness(&self) -> f32 {
        self.roughness
    }

    fn scatter_dyn(&self, ray: &Ray, hit: &HitRecord, rng: &mut dyn rand::RngCore) -> Option<ScatterRecord> {
        use crate::sampler::sampling::{ggx_sample, build_onb, local_to_world};
        
        let v = (-ray.direction).normalize();
        let n = hit.normal;
        
        // Sample GGX distribution
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        let h_local = ggx_sample(u1, u2, self.roughness);
        let (u, vv, w) = build_onb(n);
        let h = local_to_world(h_local, u, vv, w);
        
        // Reflect around half vector
        let wi = (h * 2.0 * v.dot(&h) - v).normalize();
        
        if wi.dot(&n) <= 0.0 {
            return None;
        }
        
        let n_dot_l = n.dot(&wi).max(0.0);
        let n_dot_v = n.dot(&v).max(0.0);
        let n_dot_h = n.dot(&h).max(0.0);
        let v_dot_h = v.dot(&h).max(0.0);
        
        // Evaluate BRDF
        let d = self.ggx_d(n_dot_h);
        let g = self.ggx_g(n_dot_v, n_dot_l);
        let f = self.fresnel(v_dot_h);
        
        let spec = f * (d * g / (4.0 * n_dot_v * n_dot_l + 0.001));
        let diff = self.albedo * (1.0 - self.metallic) / PI;
        
        let attenuation = diff + spec;
        let pdf = d * n_dot_h / (4.0 * v_dot_h + 0.001);
        
        Some(ScatterRecord {
            attenuation: attenuation * n_dot_l,
            scattered: Ray::new(hit.point, wi),
            pdf,
        })
    }
    
    fn eval(&self, wi: &Vec3, wo: &Vec3, normal: &Vec3) -> Vec3 {
        let n = *normal;
        let v = *wo;
        let l = *wi;
        
        let n_dot_l = n.dot(&l).max(0.0);
        let n_dot_v = n.dot(&v).max(0.001);
        
        if n_dot_l <= 0.0 {
            return Vec3::zero();
        }
        
        let h = (v + l).normalize();
        let n_dot_h = n.dot(&h).max(0.0);
        let v_dot_h = v.dot(&h).max(0.0);
        
        let d = self.ggx_d(n_dot_h);
        let g = self.ggx_g(n_dot_v, n_dot_l);
        let f = self.fresnel(v_dot_h);
        
        let spec = f * (d * g / (4.0 * n_dot_v * n_dot_l + 0.001));
        let diff = self.albedo * (1.0 - self.metallic) / PI;
        
        diff + spec
    }

    fn pdf(&self, wi: &Vec3, wo: &Vec3, normal: &Vec3) -> f32 {
        let n = *normal;
        let v = *wo;
        let l = *wi;
        
        let n_dot_l = n.dot(&l);
        if n_dot_l <= 0.0 {
            return 0.0;
        }
        
        let h = (v + l).normalize();
        let n_dot_h = n.dot(&h).max(0.0);
        let v_dot_h = v.dot(&h).max(0.0);
        
        let d = self.ggx_d(n_dot_h);
        d * n_dot_h / (4.0 * v_dot_h + 0.0001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cook_torrance() {
        let mat = CookTorrance::new(Vec3::new(1.0, 0.0, 0.0), 0.5, 0.0);
        assert_eq!(mat.metallic, 0.0);
    }

    #[test]
    fn test_cook_torrance_metallic() {
        let mat = CookTorrance::new(Vec3::new(1.0, 0.8, 0.0), 0.3, 1.0);
        assert_eq!(mat.metallic, 1.0);
    }
}
