//! Solid color texture

use crate::math::Vec3;
use super::Texture;

/// A solid color texture
pub struct SolidColor {
    color: Vec3,
}

impl SolidColor {
    pub fn new(color: Vec3) -> Self {
        Self { color }
    }

    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(Vec3::new(r, g, b))
    }
}

impl Texture for SolidColor {
    fn value(&self, _u: f32, _v: f32, _point: &Vec3) -> Vec3 {
        self.color
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solid_color() {
        let tex = SolidColor::rgb(1.0, 0.0, 0.0);
        let color = tex.value(0.0, 0.0, &Vec3::zero());
        assert_eq!(color, Vec3::new(1.0, 0.0, 0.0));
    }
}
