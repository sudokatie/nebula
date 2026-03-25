//! Emissive (light source) material

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use super::{Material, ScatterRecord};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Emissive light source material
pub struct Emissive {
    pub color: Vec3,
    pub strength: f32,
}

impl Emissive {
    pub fn new(color: Vec3, strength: f32) -> Self {
        Self { color, strength }
    }
}

impl Material for Emissive {
    fn scatter(&self, _ray: &Ray, _hit: &HitRecord, _rng: &mut Xoshiro256PlusPlus) -> Option<ScatterRecord> {
        None // Light sources don't scatter
    }

    fn emit(&self) -> Vec3 {
        self.color * self.strength
    }
}
