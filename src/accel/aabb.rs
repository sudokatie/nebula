//! Axis-aligned bounding box

use crate::math::{Vec3, Ray};

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Empty bounding box
    pub fn empty() -> Self {
        Self {
            min: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    /// Test ray intersection using optimized slab method
    /// Branch-free version with precomputed inverse direction
    #[inline]
    pub fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        self.hit_fast(&ray.origin, &ray.direction, t_min, t_max)
    }

    /// Fast hit test with precomputed values
    /// Useful when testing many AABBs with the same ray
    #[inline]
    pub fn hit_fast(&self, origin: &Vec3, direction: &Vec3, t_min: f32, t_max: f32) -> bool {
        // Compute inverse direction (can be precomputed)
        let inv_x = 1.0 / direction.x;
        let inv_y = 1.0 / direction.y;
        let inv_z = 1.0 / direction.z;

        // Compute t values for each axis
        let tx0 = (self.min.x - origin.x) * inv_x;
        let tx1 = (self.max.x - origin.x) * inv_x;
        let ty0 = (self.min.y - origin.y) * inv_y;
        let ty1 = (self.max.y - origin.y) * inv_y;
        let tz0 = (self.min.z - origin.z) * inv_z;
        let tz1 = (self.max.z - origin.z) * inv_z;

        // Min/max without branches
        let tx_min = tx0.min(tx1);
        let tx_max = tx0.max(tx1);
        let ty_min = ty0.min(ty1);
        let ty_max = ty0.max(ty1);
        let tz_min = tz0.min(tz1);
        let tz_max = tz0.max(tz1);

        // Compute intersection interval
        let t_enter = tx_min.max(ty_min).max(tz_min).max(t_min);
        let t_exit = tx_max.min(ty_max).min(tz_max).min(t_max);

        t_enter <= t_exit
    }

    /// Test ray intersection with precomputed inverse direction
    #[inline]
    pub fn hit_precomputed(&self, origin: &Vec3, inv_dir: &Vec3, t_min: f32, t_max: f32) -> bool {
        let tx0 = (self.min.x - origin.x) * inv_dir.x;
        let tx1 = (self.max.x - origin.x) * inv_dir.x;
        let ty0 = (self.min.y - origin.y) * inv_dir.y;
        let ty1 = (self.max.y - origin.y) * inv_dir.y;
        let tz0 = (self.min.z - origin.z) * inv_dir.z;
        let tz1 = (self.max.z - origin.z) * inv_dir.z;

        let tx_min = tx0.min(tx1);
        let tx_max = tx0.max(tx1);
        let ty_min = ty0.min(ty1);
        let ty_max = ty0.max(ty1);
        let tz_min = tz0.min(tz1);
        let tz_max = tz0.max(tz1);

        let t_enter = tx_min.max(ty_min).max(tz_min).max(t_min);
        let t_exit = tx_max.min(ty_max).min(tz_max).min(t_max);

        t_enter <= t_exit
    }

    /// Combine two bounding boxes
    pub fn surrounding(a: &AABB, b: &AABB) -> AABB {
        AABB::new(a.min.min(&b.min), a.max.max(&b.max))
    }

    /// Surface area (for SAH)
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Centroid of the box
    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Index of longest axis (0=x, 1=y, 2=z)
    pub fn longest_axis(&self) -> usize {
        let d = self.max - self.min;
        if d.x > d.y && d.x > d.z {
            0
        } else if d.y > d.z {
            1
        } else {
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_hit() {
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(aabb.hit(&ray, 0.0, f32::INFINITY));
    }

    #[test]
    fn test_aabb_miss() {
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(!aabb.hit(&ray, 0.0, f32::INFINITY));
    }

    #[test]
    fn test_aabb_hit_precomputed() {
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);
        let inv_dir = Vec3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);
        assert!(aabb.hit_precomputed(&origin, &inv_dir, 0.0, f32::INFINITY));
    }

    #[test]
    fn test_aabb_hit_diagonal() {
        let aabb = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0).normalize());
        assert!(aabb.hit(&ray, 0.0, f32::INFINITY));
    }

    #[test]
    fn test_aabb_hit_inside() {
        // Ray starts inside the box
        let aabb = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.0, 0.0, 1.0));
        assert!(aabb.hit(&ray, 0.0, f32::INFINITY));
    }

    #[test]
    fn test_surrounding() {
        let a = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let b = AABB::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        let c = AABB::surrounding(&a, &b);
        assert_eq!(c.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(c.max, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_surface_area() {
        let aabb = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 3.0, 4.0));
        // Surface area = 2*(2*3 + 3*4 + 4*2) = 2*(6 + 12 + 8) = 52
        assert_eq!(aabb.surface_area(), 52.0);
    }

    #[test]
    fn test_longest_axis() {
        let aabb = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 3.0, 2.0));
        assert_eq!(aabb.longest_axis(), 1); // y is longest
    }
}
