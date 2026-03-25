//! Ray representation for path tracing

use super::Vec3;

/// A ray with origin and direction
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Create a new ray
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// Get point along ray at parameter t
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_at() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(ray.at(0.0), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(ray.at(1.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(ray.at(2.5), Vec3::new(2.5, 0.0, 0.0));
    }
}
