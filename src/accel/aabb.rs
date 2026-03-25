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

    /// Test ray intersection using slab method
    pub fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        let mut t_min = t_min;
        let mut t_max = t_max;

        for i in 0..3 {
            let inv_d = 1.0 / ray.direction[i];
            let mut t0 = (self.min[i] - ray.origin[i]) * inv_d;
            let mut t1 = (self.max[i] - ray.origin[i]) * inv_d;
            
            if inv_d < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }
            
            t_min = t0.max(t_min);
            t_max = t1.min(t_max);
            
            if t_max <= t_min {
                return false;
            }
        }
        true
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
    fn test_surrounding() {
        let a = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let b = AABB::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        let c = AABB::surrounding(&a, &b);
        assert_eq!(c.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(c.max, Vec3::new(3.0, 3.0, 3.0));
    }
}
