//! Emissive (light) material

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use super::{Material, ScatterRecord, Texture, SolidColor};
use std::sync::Arc;

/// Emissive light source material
pub struct Emissive {
    emission: Arc<dyn Texture>,
    strength: f32,
}

impl Emissive {
    pub fn new(color: Vec3, strength: f32) -> Self {
        Self {
            emission: Arc::new(SolidColor::new(color)),
            strength,
        }
    }

    pub fn white(strength: f32) -> Self {
        Self::new(Vec3::new(1.0, 1.0, 1.0), strength)
    }

    pub fn textured(texture: Arc<dyn Texture>, strength: f32) -> Self {
        Self {
            emission: texture,
            strength,
        }
    }
}

impl Material for Emissive {
    fn albedo(&self) -> Vec3 {
        self.emission.sample(0.0, 0.0, &Vec3::zero())
    }

    fn scatter_dyn(&self, _ray: &Ray, _hit: &HitRecord, _rng: &mut dyn rand::RngCore) -> Option<ScatterRecord> {
        // Emissive materials don't scatter
        None
    }

    fn emit(&self) -> Vec3 {
        // Sample at origin for solid colors
        self.emission.sample(0.0, 0.0, &Vec3::zero()) * self.strength
    }

    fn emit_at(&self, u: f32, v: f32, point: &Vec3) -> Vec3 {
        self.emission.sample(u, v, point) * self.strength
    }

    fn is_delta(&self) -> bool {
        false
    }

    fn eval(&self, _wi: &Vec3, _wo: &Vec3, _normal: &Vec3) -> Vec3 {
        Vec3::zero()
    }

    fn pdf(&self, _wi: &Vec3, _wo: &Vec3, _normal: &Vec3) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emissive_emit() {
        let mat = Emissive::new(Vec3::new(1.0, 1.0, 1.0), 10.0);
        let emission = mat.emit();
        assert!((emission.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_emissive_no_scatter() {
        let mat = Emissive::white(5.0);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            t: 1.0,
            point: Vec3::zero(),
            geometric_normal: Vec3::new(0.0, 0.0, 1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            uv: (0.0, 0.0),
            front_face: true,
            material_id: 0,
        };
        
        let mut rng = rand::thread_rng();
        assert!(mat.scatter_dyn(&ray, &hit, &mut rng).is_none());
    }
}
