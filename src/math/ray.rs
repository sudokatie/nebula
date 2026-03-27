//! Ray representation for path tracing

use super::Vec3;

/// A ray with origin, direction, and parametric bounds
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub t_min: f32,
    pub t_max: f32,
}

impl Ray {
    /// Create a new ray with default bounds
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction,
            t_min: 0.001,
            t_max: f32::INFINITY,
        }
    }

    /// Create ray with explicit bounds
    pub fn with_bounds(origin: Vec3, direction: Vec3, t_min: f32, t_max: f32) -> Self {
        Self { origin, direction, t_min, t_max }
    }

    /// Get point along ray at parameter t
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Check if t is within ray bounds
    pub fn in_bounds(&self, t: f32) -> bool {
        t > self.t_min && t < self.t_max
    }
}

/// Ray with differentials for texture filtering (mipmap selection)
#[derive(Debug, Clone, Copy)]
pub struct RayDifferential {
    pub ray: Ray,
    /// Origin differential in x
    pub rx_origin: Vec3,
    /// Origin differential in y
    pub ry_origin: Vec3,
    /// Direction differential in x
    pub rx_direction: Vec3,
    /// Direction differential in y
    pub ry_direction: Vec3,
    /// Whether differentials are valid
    pub has_differentials: bool,
}

impl RayDifferential {
    /// Create ray differential without differentials
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            ray: Ray::new(origin, direction),
            rx_origin: Vec3::zero(),
            ry_origin: Vec3::zero(),
            rx_direction: Vec3::zero(),
            ry_direction: Vec3::zero(),
            has_differentials: false,
        }
    }

    /// Create ray differential with explicit differentials
    pub fn with_differentials(
        ray: Ray,
        rx_origin: Vec3,
        ry_origin: Vec3,
        rx_direction: Vec3,
        ry_direction: Vec3,
    ) -> Self {
        Self {
            ray,
            rx_origin,
            ry_origin,
            rx_direction,
            ry_direction,
            has_differentials: true,
        }
    }

    /// Scale differentials (used when ray bounces)
    pub fn scale_differentials(&mut self, s: f32) {
        self.rx_origin = self.ray.origin + (self.rx_origin - self.ray.origin) * s;
        self.ry_origin = self.ray.origin + (self.ry_origin - self.ray.origin) * s;
        self.rx_direction = self.ray.direction + (self.rx_direction - self.ray.direction) * s;
        self.ry_direction = self.ray.direction + (self.ry_direction - self.ray.direction) * s;
    }

    /// Compute texture filter footprint (returns dU, dV)
    pub fn compute_differentials_at(&self, point: Vec3, normal: Vec3) -> (f32, f32) {
        if !self.has_differentials {
            return (0.0, 0.0);
        }

        // Project ray differentials onto surface
        let d = normal.dot(&self.ray.direction);
        if d.abs() < 1e-8 {
            return (0.0, 0.0);
        }

        // Compute auxiliary intersection points
        let t = normal.dot(&(point - self.ray.origin)) / d;
        
        let px = self.rx_origin + self.rx_direction * t;
        let py = self.ry_origin + self.ry_direction * t;

        let dpdx = px - point;
        let dpdy = py - point;

        (dpdx.length(), dpdy.length())
    }
}

impl From<Ray> for RayDifferential {
    fn from(ray: Ray) -> Self {
        Self {
            ray,
            rx_origin: Vec3::zero(),
            ry_origin: Vec3::zero(),
            rx_direction: Vec3::zero(),
            ry_direction: Vec3::zero(),
            has_differentials: false,
        }
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

    #[test]
    fn test_ray_bounds() {
        let ray = Ray::with_bounds(Vec3::zero(), Vec3::new(1.0, 0.0, 0.0), 0.1, 10.0);
        assert!(!ray.in_bounds(0.05));
        assert!(ray.in_bounds(0.5));
        assert!(ray.in_bounds(9.0));
        assert!(!ray.in_bounds(11.0));
    }

    #[test]
    fn test_ray_differential() {
        let rd = RayDifferential::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        assert!(!rd.has_differentials);
    }

    #[test]
    fn test_ray_differential_from_ray() {
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let rd: RayDifferential = ray.into();
        assert_eq!(rd.ray.origin, ray.origin);
        assert!(!rd.has_differentials);
    }
}
