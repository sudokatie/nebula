//! Checker (alternating pattern) texture

use crate::math::Vec3;
use super::Texture;
use std::sync::Arc;

/// A 3D checker pattern texture
pub struct CheckerTexture {
    odd: Arc<dyn Texture>,
    even: Arc<dyn Texture>,
    scale: f32,
}

impl CheckerTexture {
    /// Create checker from two textures
    pub fn new(odd: Arc<dyn Texture>, even: Arc<dyn Texture>, scale: f32) -> Self {
        Self { odd, even, scale }
    }

    /// Create checker from two colors
    pub fn from_colors(odd: Vec3, even: Vec3, scale: f32) -> Self {
        use super::SolidColor;
        Self {
            odd: Arc::new(SolidColor::new(odd)),
            even: Arc::new(SolidColor::new(even)),
            scale,
        }
    }
}

impl Texture for CheckerTexture {
    fn value(&self, u: f32, v: f32, point: &Vec3) -> Vec3 {
        let inv_scale = 1.0 / self.scale;
        let x = (point.x * inv_scale).floor() as i32;
        let y = (point.y * inv_scale).floor() as i32;
        let z = (point.z * inv_scale).floor() as i32;

        let is_even = (x + y + z) % 2 == 0;

        if is_even {
            self.even.value(u, v, point)
        } else {
            self.odd.value(u, v, point)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checker_alternates() {
        let checker = CheckerTexture::from_colors(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            1.0,
        );

        // At origin, should be even (white)
        let c1 = checker.value(0.0, 0.0, &Vec3::new(0.5, 0.5, 0.5));
        assert_eq!(c1, Vec3::new(1.0, 1.0, 1.0));

        // One unit over, should be odd (black)
        let c2 = checker.value(0.0, 0.0, &Vec3::new(1.5, 0.5, 0.5));
        assert_eq!(c2, Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_checker_scale() {
        let checker = CheckerTexture::from_colors(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            0.5, // Smaller scale = more frequent pattern
        );

        // At 0.25, floor(0.25/0.5) = 0, even
        let c1 = checker.value(0.0, 0.0, &Vec3::new(0.25, 0.0, 0.0));
        assert_eq!(c1, Vec3::new(1.0, 1.0, 1.0));

        // At 0.75, floor(0.75/0.5) = 1, odd
        let c2 = checker.value(0.0, 0.0, &Vec3::new(0.75, 0.0, 0.0));
        assert_eq!(c2, Vec3::new(0.0, 0.0, 0.0));
    }
}
