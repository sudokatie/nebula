//! Texture types for materials

mod checker;
mod noise;
mod solid;

pub use checker::CheckerTexture;
pub use noise::{NoiseTexture, Perlin};
pub use solid::SolidColor;

use crate::math::Vec3;

/// Trait for textures that can be sampled
pub trait Texture: Send + Sync {
    /// Sample texture at UV coordinates and 3D point
    fn value(&self, u: f32, v: f32, point: &Vec3) -> Vec3;
}
