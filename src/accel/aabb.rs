//! Axis-Aligned Bounding Box

use crate::math::Vec3;

/// Axis-Aligned Bounding Box
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn empty() -> Self {
        Self {
            min: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    pub fn from_points(points: &[Vec3]) -> Self {
        let mut aabb = Self::empty();
        for p in points {
            aabb = aabb.extend_point(*p);
        }
        aabb
    }

    pub fn extend_point(self, p: Vec3) -> Self {
        Self {
            min: self.min.min(&p),
            max: self.max.max(&p),
        }
    }

    pub fn surrounding(a: &Self, b: &Self) -> Self {
        Self {
            min: a.min.min(&b.min),
            max: a.max.max(&b.max),
        }
    }

    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn min(&self) -> Vec3 {
        self.min
    }

    pub fn max(&self) -> Vec3 {
        self.max
    }

    pub fn diagonal(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.diagonal();
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    pub fn volume(&self) -> f32 {
        let d = self.diagonal();
        d.x * d.y * d.z
    }

    pub fn longest_axis(&self) -> usize {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z {
            0
        } else if d.y > d.z {
            1
        } else {
            2
        }
    }

    /// Ray-AABB intersection test
    pub fn hit(&self, ray_origin: &Vec3, ray_direction: &Vec3, t_min: f32, t_max: f32) -> bool {
        let inv_d = Vec3::new(
            1.0 / ray_direction.x,
            1.0 / ray_direction.y,
            1.0 / ray_direction.z,
        );
        self.hit_precomputed(ray_origin, &inv_d, t_min, t_max)
    }

    /// Ray-AABB intersection with precomputed inverse direction
    #[inline]
    pub fn hit_precomputed(&self, origin: &Vec3, inv_dir: &Vec3, t_min: f32, t_max: f32) -> bool {
        let mut tmin = t_min;
        let mut tmax = t_max;

        // X slab
        let t0x = (self.min.x - origin.x) * inv_dir.x;
        let t1x = (self.max.x - origin.x) * inv_dir.x;
        let (t0x, t1x) = if inv_dir.x < 0.0 { (t1x, t0x) } else { (t0x, t1x) };
        tmin = tmin.max(t0x);
        tmax = tmax.min(t1x);

        if tmax < tmin {
            return false;
        }

        // Y slab
        let t0y = (self.min.y - origin.y) * inv_dir.y;
        let t1y = (self.max.y - origin.y) * inv_dir.y;
        let (t0y, t1y) = if inv_dir.y < 0.0 { (t1y, t0y) } else { (t0y, t1y) };
        tmin = tmin.max(t0y);
        tmax = tmax.min(t1y);

        if tmax < tmin {
            return false;
        }

        // Z slab
        let t0z = (self.min.z - origin.z) * inv_dir.z;
        let t1z = (self.max.z - origin.z) * inv_dir.z;
        let (t0z, t1z) = if inv_dir.z < 0.0 { (t1z, t0z) } else { (t0z, t1z) };
        tmin = tmin.max(t0z);
        tmax = tmax.min(t1z);

        tmax >= tmin
    }

    /// SIMD ray-AABB intersection for 4 rays at once
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn hit_simd4(
        &self,
        origins: &[Vec3; 4],
        inv_dirs: &[Vec3; 4],
        t_mins: [f32; 4],
        t_maxs: [f32; 4],
    ) -> [bool; 4] {
        use std::arch::x86_64::*;

        unsafe {
            // Load AABB bounds
            let min_x = _mm_set1_ps(self.min.x);
            let min_y = _mm_set1_ps(self.min.y);
            let min_z = _mm_set1_ps(self.min.z);
            let max_x = _mm_set1_ps(self.max.x);
            let max_y = _mm_set1_ps(self.max.y);
            let max_z = _mm_set1_ps(self.max.z);

            // Load ray data
            let ox = _mm_set_ps(origins[3].x, origins[2].x, origins[1].x, origins[0].x);
            let oy = _mm_set_ps(origins[3].y, origins[2].y, origins[1].y, origins[0].y);
            let oz = _mm_set_ps(origins[3].z, origins[2].z, origins[1].z, origins[0].z);

            let idx = _mm_set_ps(inv_dirs[3].x, inv_dirs[2].x, inv_dirs[1].x, inv_dirs[0].x);
            let idy = _mm_set_ps(inv_dirs[3].y, inv_dirs[2].y, inv_dirs[1].y, inv_dirs[0].y);
            let idz = _mm_set_ps(inv_dirs[3].z, inv_dirs[2].z, inv_dirs[1].z, inv_dirs[0].z);

            let mut tmin = _mm_set_ps(t_mins[3], t_mins[2], t_mins[1], t_mins[0]);
            let mut tmax = _mm_set_ps(t_maxs[3], t_maxs[2], t_maxs[1], t_maxs[0]);

            // X slab
            let t0x = _mm_mul_ps(_mm_sub_ps(min_x, ox), idx);
            let t1x = _mm_mul_ps(_mm_sub_ps(max_x, ox), idx);
            let t0x_min = _mm_min_ps(t0x, t1x);
            let t1x_max = _mm_max_ps(t0x, t1x);
            tmin = _mm_max_ps(tmin, t0x_min);
            tmax = _mm_min_ps(tmax, t1x_max);

            // Y slab
            let t0y = _mm_mul_ps(_mm_sub_ps(min_y, oy), idy);
            let t1y = _mm_mul_ps(_mm_sub_ps(max_y, oy), idy);
            let t0y_min = _mm_min_ps(t0y, t1y);
            let t1y_max = _mm_max_ps(t0y, t1y);
            tmin = _mm_max_ps(tmin, t0y_min);
            tmax = _mm_min_ps(tmax, t1y_max);

            // Z slab
            let t0z = _mm_mul_ps(_mm_sub_ps(min_z, oz), idz);
            let t1z = _mm_mul_ps(_mm_sub_ps(max_z, oz), idz);
            let t0z_min = _mm_min_ps(t0z, t1z);
            let t1z_max = _mm_max_ps(t0z, t1z);
            tmin = _mm_max_ps(tmin, t0z_min);
            tmax = _mm_min_ps(tmax, t1z_max);

            // Compare
            let mask = _mm_cmpge_ps(tmax, tmin);
            let mask_bits = _mm_movemask_ps(mask);

            [
                (mask_bits & 1) != 0,
                (mask_bits & 2) != 0,
                (mask_bits & 4) != 0,
                (mask_bits & 8) != 0,
            ]
        }
    }

    /// SIMD ray-AABB intersection for ARM NEON (4 rays at once)
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub fn hit_simd4(
        &self,
        origins: &[Vec3; 4],
        inv_dirs: &[Vec3; 4],
        t_mins: [f32; 4],
        t_maxs: [f32; 4],
    ) -> [bool; 4] {
        use std::arch::aarch64::*;

        unsafe {
            // Load AABB bounds
            let min_x = vdupq_n_f32(self.min.x);
            let min_y = vdupq_n_f32(self.min.y);
            let min_z = vdupq_n_f32(self.min.z);
            let max_x = vdupq_n_f32(self.max.x);
            let max_y = vdupq_n_f32(self.max.y);
            let max_z = vdupq_n_f32(self.max.z);

            // Load ray origins
            let ox = vld1q_f32([origins[0].x, origins[1].x, origins[2].x, origins[3].x].as_ptr());
            let oy = vld1q_f32([origins[0].y, origins[1].y, origins[2].y, origins[3].y].as_ptr());
            let oz = vld1q_f32([origins[0].z, origins[1].z, origins[2].z, origins[3].z].as_ptr());

            // Load inverse directions
            let idx = vld1q_f32([inv_dirs[0].x, inv_dirs[1].x, inv_dirs[2].x, inv_dirs[3].x].as_ptr());
            let idy = vld1q_f32([inv_dirs[0].y, inv_dirs[1].y, inv_dirs[2].y, inv_dirs[3].y].as_ptr());
            let idz = vld1q_f32([inv_dirs[0].z, inv_dirs[1].z, inv_dirs[2].z, inv_dirs[3].z].as_ptr());

            let mut tmin = vld1q_f32(t_mins.as_ptr());
            let mut tmax = vld1q_f32(t_maxs.as_ptr());

            // X slab
            let t0x = vmulq_f32(vsubq_f32(min_x, ox), idx);
            let t1x = vmulq_f32(vsubq_f32(max_x, ox), idx);
            let t0x_min = vminq_f32(t0x, t1x);
            let t1x_max = vmaxq_f32(t0x, t1x);
            tmin = vmaxq_f32(tmin, t0x_min);
            tmax = vminq_f32(tmax, t1x_max);

            // Y slab
            let t0y = vmulq_f32(vsubq_f32(min_y, oy), idy);
            let t1y = vmulq_f32(vsubq_f32(max_y, oy), idy);
            let t0y_min = vminq_f32(t0y, t1y);
            let t1y_max = vmaxq_f32(t0y, t1y);
            tmin = vmaxq_f32(tmin, t0y_min);
            tmax = vminq_f32(tmax, t1y_max);

            // Z slab
            let t0z = vmulq_f32(vsubq_f32(min_z, oz), idz);
            let t1z = vmulq_f32(vsubq_f32(max_z, oz), idz);
            let t0z_min = vminq_f32(t0z, t1z);
            let t1z_max = vmaxq_f32(t0z, t1z);
            tmin = vmaxq_f32(tmin, t0z_min);
            tmax = vminq_f32(tmax, t1z_max);

            // Compare tmax >= tmin
            let mask = vcgeq_f32(tmax, tmin);
            let mask_bits: [u32; 4] = std::mem::transmute(mask);

            [
                mask_bits[0] != 0,
                mask_bits[1] != 0,
                mask_bits[2] != 0,
                mask_bits[3] != 0,
            ]
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn hit_simd4(
        &self,
        origins: &[Vec3; 4],
        inv_dirs: &[Vec3; 4],
        t_mins: [f32; 4],
        t_maxs: [f32; 4],
    ) -> [bool; 4] {
        [
            self.hit_precomputed(&origins[0], &inv_dirs[0], t_mins[0], t_maxs[0]),
            self.hit_precomputed(&origins[1], &inv_dirs[1], t_mins[1], t_maxs[1]),
            self.hit_precomputed(&origins[2], &inv_dirs[2], t_mins[2], t_maxs[2]),
            self.hit_precomputed(&origins[3], &inv_dirs[3], t_mins[3], t_maxs[3]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_empty() {
        let aabb = AABB::empty();
        assert!(aabb.min.x > aabb.max.x);
    }

    #[test]
    fn test_aabb_surrounding() {
        let a = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let b = AABB::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        let c = AABB::surrounding(&a, &b);
        assert_eq!(c.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(c.max, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_aabb_centroid() {
        let aabb = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let c = aabb.centroid();
        assert_eq!(c, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_aabb_surface_area() {
        let aabb = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert!((aabb.surface_area() - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_aabb_hit() {
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);
        assert!(aabb.hit(&origin, &direction, 0.0, f32::INFINITY));
    }

    #[test]
    fn test_aabb_miss() {
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let origin = Vec3::new(5.0, 5.0, 5.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);
        assert!(!aabb.hit(&origin, &direction, 0.0, f32::INFINITY));
    }

    #[test]
    fn test_aabb_longest_axis() {
        let aabb = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 2.0));
        assert_eq!(aabb.longest_axis(), 0);
    }

    #[test]
    fn test_aabb_simd4() {
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let origins = [
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(5.0, 5.0, 5.0),
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -5.0),
        ];
        // inv_dirs should be 1/direction, but we need to handle inf for zero components
        let inv_dirs = [
            Vec3::new(f32::INFINITY, f32::INFINITY, -1.0),  // dir = (0, 0, -1)
            Vec3::new(f32::INFINITY, f32::INFINITY, -1.0),  // dir = (0, 0, -1)
            Vec3::new(f32::INFINITY, f32::INFINITY, -1.0),  // dir = (0, 0, -1)
            Vec3::new(f32::INFINITY, f32::INFINITY, 1.0),   // dir = (0, 0, 1)
        ];
        let t_mins = [0.0, 0.0, 0.0, 0.0];
        let t_maxs = [f32::INFINITY; 4];
        
        let hits = aabb.hit_simd4(&origins, &inv_dirs, t_mins, t_maxs);
        assert!(hits[0]); // Should hit (straight into box)
        assert!(!hits[1]); // Should miss (origin off to side)
        assert!(hits[2]); // Should hit
        assert!(hits[3]); // Should hit (from other side)
    }
}
