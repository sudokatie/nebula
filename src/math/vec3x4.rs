//! SIMD-accelerated 4-wide Vec3 for ray packet tracing
//!
//! Processes 4 Vec3s simultaneously using platform SIMD.

use super::Vec3;

/// 4 Vec3s packed for SIMD operations
#[derive(Debug, Clone, Copy)]
pub struct Vec3x4 {
    pub x: [f32; 4],
    pub y: [f32; 4],
    pub z: [f32; 4],
}

impl Vec3x4 {
    /// Create from 4 individual Vec3s
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> Self {
        Self {
            x: [v0.x, v1.x, v2.x, v3.x],
            y: [v0.y, v1.y, v2.y, v3.y],
            z: [v0.z, v1.z, v2.z, v3.z],
        }
    }

    /// Create with all 4 slots set to the same Vec3
    pub fn splat(v: Vec3) -> Self {
        Self {
            x: [v.x; 4],
            y: [v.y; 4],
            z: [v.z; 4],
        }
    }

    /// Create zero vector
    pub fn zero() -> Self {
        Self {
            x: [0.0; 4],
            y: [0.0; 4],
            z: [0.0; 4],
        }
    }

    /// Extract individual Vec3 at index
    #[inline]
    pub fn get(&self, i: usize) -> Vec3 {
        Vec3::new(self.x[i], self.y[i], self.z[i])
    }

    /// Set individual Vec3 at index
    #[inline]
    pub fn set(&mut self, i: usize, v: Vec3) {
        self.x[i] = v.x;
        self.y[i] = v.y;
        self.z[i] = v.z;
    }

    /// Component-wise addition
    #[inline]
    pub fn add(&self, other: &Vec3x4) -> Vec3x4 {
        Vec3x4 {
            x: [
                self.x[0] + other.x[0],
                self.x[1] + other.x[1],
                self.x[2] + other.x[2],
                self.x[3] + other.x[3],
            ],
            y: [
                self.y[0] + other.y[0],
                self.y[1] + other.y[1],
                self.y[2] + other.y[2],
                self.y[3] + other.y[3],
            ],
            z: [
                self.z[0] + other.z[0],
                self.z[1] + other.z[1],
                self.z[2] + other.z[2],
                self.z[3] + other.z[3],
            ],
        }
    }

    /// Component-wise subtraction
    #[inline]
    pub fn sub(&self, other: &Vec3x4) -> Vec3x4 {
        Vec3x4 {
            x: [
                self.x[0] - other.x[0],
                self.x[1] - other.x[1],
                self.x[2] - other.x[2],
                self.x[3] - other.x[3],
            ],
            y: [
                self.y[0] - other.y[0],
                self.y[1] - other.y[1],
                self.y[2] - other.y[2],
                self.y[3] - other.y[3],
            ],
            z: [
                self.z[0] - other.z[0],
                self.z[1] - other.z[1],
                self.z[2] - other.z[2],
                self.z[3] - other.z[3],
            ],
        }
    }

    /// Scalar multiplication
    #[inline]
    pub fn mul_scalar(&self, s: f32) -> Vec3x4 {
        Vec3x4 {
            x: [self.x[0] * s, self.x[1] * s, self.x[2] * s, self.x[3] * s],
            y: [self.y[0] * s, self.y[1] * s, self.y[2] * s, self.y[3] * s],
            z: [self.z[0] * s, self.z[1] * s, self.z[2] * s, self.z[3] * s],
        }
    }

    /// Per-lane scalar multiplication
    #[inline]
    pub fn mul_scalars(&self, s: [f32; 4]) -> Vec3x4 {
        Vec3x4 {
            x: [
                self.x[0] * s[0],
                self.x[1] * s[1],
                self.x[2] * s[2],
                self.x[3] * s[3],
            ],
            y: [
                self.y[0] * s[0],
                self.y[1] * s[1],
                self.y[2] * s[2],
                self.y[3] * s[3],
            ],
            z: [
                self.z[0] * s[0],
                self.z[1] * s[1],
                self.z[2] * s[2],
                self.z[3] * s[3],
            ],
        }
    }

    /// Component-wise multiplication
    #[inline]
    pub fn mul(&self, other: &Vec3x4) -> Vec3x4 {
        Vec3x4 {
            x: [
                self.x[0] * other.x[0],
                self.x[1] * other.x[1],
                self.x[2] * other.x[2],
                self.x[3] * other.x[3],
            ],
            y: [
                self.y[0] * other.y[0],
                self.y[1] * other.y[1],
                self.y[2] * other.y[2],
                self.y[3] * other.y[3],
            ],
            z: [
                self.z[0] * other.z[0],
                self.z[1] * other.z[1],
                self.z[2] * other.z[2],
                self.z[3] * other.z[3],
            ],
        }
    }

    /// Dot product (returns 4 scalars)
    #[inline]
    pub fn dot(&self, other: &Vec3x4) -> [f32; 4] {
        [
            self.x[0] * other.x[0] + self.y[0] * other.y[0] + self.z[0] * other.z[0],
            self.x[1] * other.x[1] + self.y[1] * other.y[1] + self.z[1] * other.z[1],
            self.x[2] * other.x[2] + self.y[2] * other.y[2] + self.z[2] * other.z[2],
            self.x[3] * other.x[3] + self.y[3] * other.y[3] + self.z[3] * other.z[3],
        ]
    }

    /// Length squared (returns 4 scalars)
    #[inline]
    pub fn length_squared(&self) -> [f32; 4] {
        self.dot(self)
    }

    /// Length (returns 4 scalars)
    #[inline]
    pub fn length(&self) -> [f32; 4] {
        let sq = self.length_squared();
        [sq[0].sqrt(), sq[1].sqrt(), sq[2].sqrt(), sq[3].sqrt()]
    }

    /// Normalize each vector
    #[inline]
    pub fn normalize(&self) -> Vec3x4 {
        let len = self.length();
        Vec3x4 {
            x: [
                self.x[0] / len[0],
                self.x[1] / len[1],
                self.x[2] / len[2],
                self.x[3] / len[3],
            ],
            y: [
                self.y[0] / len[0],
                self.y[1] / len[1],
                self.y[2] / len[2],
                self.y[3] / len[3],
            ],
            z: [
                self.z[0] / len[0],
                self.z[1] / len[1],
                self.z[2] / len[2],
                self.z[3] / len[3],
            ],
        }
    }

    /// Cross product
    #[inline]
    pub fn cross(&self, other: &Vec3x4) -> Vec3x4 {
        Vec3x4 {
            x: [
                self.y[0] * other.z[0] - self.z[0] * other.y[0],
                self.y[1] * other.z[1] - self.z[1] * other.y[1],
                self.y[2] * other.z[2] - self.z[2] * other.y[2],
                self.y[3] * other.z[3] - self.z[3] * other.y[3],
            ],
            y: [
                self.z[0] * other.x[0] - self.x[0] * other.z[0],
                self.z[1] * other.x[1] - self.x[1] * other.z[1],
                self.z[2] * other.x[2] - self.x[2] * other.z[2],
                self.z[3] * other.x[3] - self.x[3] * other.z[3],
            ],
            z: [
                self.x[0] * other.y[0] - self.y[0] * other.x[0],
                self.x[1] * other.y[1] - self.y[1] * other.x[1],
                self.x[2] * other.y[2] - self.y[2] * other.x[2],
                self.x[3] * other.y[3] - self.y[3] * other.x[3],
            ],
        }
    }

    /// Component-wise minimum
    #[inline]
    pub fn min(&self, other: &Vec3x4) -> Vec3x4 {
        Vec3x4 {
            x: [
                self.x[0].min(other.x[0]),
                self.x[1].min(other.x[1]),
                self.x[2].min(other.x[2]),
                self.x[3].min(other.x[3]),
            ],
            y: [
                self.y[0].min(other.y[0]),
                self.y[1].min(other.y[1]),
                self.y[2].min(other.y[2]),
                self.y[3].min(other.y[3]),
            ],
            z: [
                self.z[0].min(other.z[0]),
                self.z[1].min(other.z[1]),
                self.z[2].min(other.z[2]),
                self.z[3].min(other.z[3]),
            ],
        }
    }

    /// Component-wise maximum
    #[inline]
    pub fn max(&self, other: &Vec3x4) -> Vec3x4 {
        Vec3x4 {
            x: [
                self.x[0].max(other.x[0]),
                self.x[1].max(other.x[1]),
                self.x[2].max(other.x[2]),
                self.x[3].max(other.x[3]),
            ],
            y: [
                self.y[0].max(other.y[0]),
                self.y[1].max(other.y[1]),
                self.y[2].max(other.y[2]),
                self.y[3].max(other.y[3]),
            ],
            z: [
                self.z[0].max(other.z[0]),
                self.z[1].max(other.z[1]),
                self.z[2].max(other.z[2]),
                self.z[3].max(other.z[3]),
            ],
        }
    }

    /// Negation
    #[inline]
    pub fn neg(&self) -> Vec3x4 {
        Vec3x4 {
            x: [-self.x[0], -self.x[1], -self.x[2], -self.x[3]],
            y: [-self.y[0], -self.y[1], -self.y[2], -self.y[3]],
            z: [-self.z[0], -self.z[1], -self.z[2], -self.z[3]],
        }
    }
}

/// 4 rays packed for SIMD ray packet tracing
#[derive(Debug, Clone, Copy)]
pub struct Ray4 {
    pub origins: Vec3x4,
    pub directions: Vec3x4,
    pub inv_directions: Vec3x4,
    pub t_min: [f32; 4],
    pub t_max: [f32; 4],
}

impl Ray4 {
    /// Create from 4 individual rays
    pub fn new(
        origins: Vec3x4,
        directions: Vec3x4,
        t_min: [f32; 4],
        t_max: [f32; 4],
    ) -> Self {
        let inv_directions = Vec3x4 {
            x: [
                1.0 / directions.x[0],
                1.0 / directions.x[1],
                1.0 / directions.x[2],
                1.0 / directions.x[3],
            ],
            y: [
                1.0 / directions.y[0],
                1.0 / directions.y[1],
                1.0 / directions.y[2],
                1.0 / directions.y[3],
            ],
            z: [
                1.0 / directions.z[0],
                1.0 / directions.z[1],
                1.0 / directions.z[2],
                1.0 / directions.z[3],
            ],
        };
        Self {
            origins,
            directions,
            inv_directions,
            t_min,
            t_max,
        }
    }

    /// Get point along ray i at parameter t
    #[inline]
    pub fn at(&self, i: usize, t: f32) -> Vec3 {
        Vec3::new(
            self.origins.x[i] + self.directions.x[i] * t,
            self.origins.y[i] + self.directions.y[i] * t,
            self.origins.z[i] + self.directions.z[i] * t,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3x4_new() {
        let v = Vec3x4::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
            Vec3::new(10.0, 11.0, 12.0),
        );
        assert_eq!(v.get(0), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(v.get(3), Vec3::new(10.0, 11.0, 12.0));
    }

    #[test]
    fn test_vec3x4_splat() {
        let v = Vec3x4::splat(Vec3::new(1.0, 2.0, 3.0));
        for i in 0..4 {
            assert_eq!(v.get(i), Vec3::new(1.0, 2.0, 3.0));
        }
    }

    #[test]
    fn test_vec3x4_add() {
        let a = Vec3x4::splat(Vec3::new(1.0, 2.0, 3.0));
        let b = Vec3x4::splat(Vec3::new(4.0, 5.0, 6.0));
        let c = a.add(&b);
        assert_eq!(c.get(0), Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vec3x4_dot() {
        let a = Vec3x4::splat(Vec3::new(1.0, 0.0, 0.0));
        let b = Vec3x4::splat(Vec3::new(1.0, 0.0, 0.0));
        let dots = a.dot(&b);
        assert_eq!(dots, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_vec3x4_normalize() {
        let v = Vec3x4::splat(Vec3::new(3.0, 0.0, 0.0));
        let n = v.normalize();
        let len = n.length();
        for l in len {
            assert!((l - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_vec3x4_cross() {
        let a = Vec3x4::splat(Vec3::new(1.0, 0.0, 0.0));
        let b = Vec3x4::splat(Vec3::new(0.0, 1.0, 0.0));
        let c = a.cross(&b);
        assert_eq!(c.get(0), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_ray4_at() {
        let origins = Vec3x4::splat(Vec3::zero());
        let directions = Vec3x4::splat(Vec3::new(1.0, 0.0, 0.0));
        let ray4 = Ray4::new(origins, directions, [0.0; 4], [f32::INFINITY; 4]);
        assert_eq!(ray4.at(0, 5.0), Vec3::new(5.0, 0.0, 0.0));
    }
}
