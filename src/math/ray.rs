//! Ray representation for path tracing

use super::Vec3;

/// A ray with origin, direction, and intersection bounds
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

    /// Create a ray with explicit bounds
    pub fn with_bounds(origin: Vec3, direction: Vec3, t_min: f32, t_max: f32) -> Self {
        Self {
            origin,
            direction,
            t_min,
            t_max,
        }
    }

    /// Get point along ray at parameter t
    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Precompute inverse direction for faster AABB tests
    #[inline]
    pub fn inv_direction(&self) -> Vec3 {
        Vec3::new(
            1.0 / self.direction.x,
            1.0 / self.direction.y,
            1.0 / self.direction.z,
        )
    }
}

/// Ray with differentials for texture filtering and antialiasing
#[derive(Debug, Clone, Copy)]
pub struct RayDifferential {
    /// The primary ray
    pub ray: Ray,
    /// Origin offset for x-adjacent pixel
    pub rx_origin: Vec3,
    /// Direction offset for x-adjacent pixel
    pub rx_direction: Vec3,
    /// Origin offset for y-adjacent pixel
    pub ry_origin: Vec3,
    /// Direction offset for y-adjacent pixel
    pub ry_direction: Vec3,
    /// Whether differentials have been set
    pub has_differentials: bool,
}

impl RayDifferential {
    /// Create ray differential without differentials
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            ray: Ray::new(origin, direction),
            rx_origin: Vec3::zero(),
            rx_direction: Vec3::zero(),
            ry_origin: Vec3::zero(),
            ry_direction: Vec3::zero(),
            has_differentials: false,
        }
    }

    /// Create ray differential with explicit differentials
    pub fn with_differentials(
        ray: Ray,
        rx_origin: Vec3,
        rx_direction: Vec3,
        ry_origin: Vec3,
        ry_direction: Vec3,
    ) -> Self {
        Self {
            ray,
            rx_origin,
            rx_direction,
            ry_origin,
            ry_direction,
            has_differentials: true,
        }
    }

    /// Scale differentials (e.g., for subpixel sampling)
    pub fn scale_differentials(&mut self, scale: f32) {
        self.rx_origin = self.ray.origin + (self.rx_origin - self.ray.origin) * scale;
        self.ry_origin = self.ray.origin + (self.ry_origin - self.ray.origin) * scale;
        self.rx_direction = self.ray.direction + (self.rx_direction - self.ray.direction) * scale;
        self.ry_direction = self.ray.direction + (self.ry_direction - self.ray.direction) * scale;
    }

    /// Get point along primary ray at parameter t
    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.ray.at(t)
    }

    /// Compute texture footprint (LOD) at intersection point
    /// Returns approximate screen-space derivatives
    pub fn compute_footprint(&self, t: f32, normal: &Vec3) -> (Vec3, Vec3) {
        if !self.has_differentials {
            return (Vec3::zero(), Vec3::zero());
        }

        // Compute intersection points for differential rays
        let p = self.ray.at(t);

        // Project differential ray intersections onto tangent plane
        let d = normal.dot(&p);

        // X differential
        let tx = -(normal.dot(&self.rx_origin) - d) / normal.dot(&self.rx_direction);
        let px = self.rx_origin + self.rx_direction * tx;
        let dpdx = px - p;

        // Y differential
        let ty = -(normal.dot(&self.ry_origin) - d) / normal.dot(&self.ry_direction);
        let py = self.ry_origin + self.ry_direction * ty;
        let dpdy = py - p;

        (dpdx, dpdy)
    }
}

impl From<Ray> for RayDifferential {
    fn from(ray: Ray) -> Self {
        Self {
            ray,
            rx_origin: Vec3::zero(),
            rx_direction: Vec3::zero(),
            ry_origin: Vec3::zero(),
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
        let ray = Ray::with_bounds(
            Vec3::zero(),
            Vec3::new(1.0, 0.0, 0.0),
            0.1,
            100.0,
        );
        assert_eq!(ray.t_min, 0.1);
        assert_eq!(ray.t_max, 100.0);
    }

    #[test]
    fn test_ray_inv_direction() {
        let ray = Ray::new(Vec3::zero(), Vec3::new(2.0, 4.0, 8.0));
        let inv = ray.inv_direction();
        assert_eq!(inv.x, 0.5);
        assert_eq!(inv.y, 0.25);
        assert_eq!(inv.z, 0.125);
    }

    #[test]
    fn test_ray_differential_new() {
        let rd = RayDifferential::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        assert!(!rd.has_differentials);
        assert_eq!(rd.ray.origin, Vec3::zero());
    }

    #[test]
    fn test_ray_differential_with_differentials() {
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let rd = RayDifferential::with_differentials(
            ray,
            Vec3::new(0.001, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 0.001, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        );
        assert!(rd.has_differentials);
    }

    #[test]
    fn test_ray_differential_from_ray() {
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let rd: RayDifferential = ray.into();
        assert!(!rd.has_differentials);
        assert_eq!(rd.ray.origin, ray.origin);
    }

    #[test]
    fn test_ray_differential_footprint() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let rd = RayDifferential::with_differentials(
            ray,
            Vec3::new(0.01, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 0.01, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        );
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let (dpdx, dpdy) = rd.compute_footprint(1.0, &normal);
        // dpdx should be approximately (0.01, 0, 0)
        assert!((dpdx.x - 0.01).abs() < 1e-6);
        assert!(dpdy.y.abs() > 0.0);
    }
}
