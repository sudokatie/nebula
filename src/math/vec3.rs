//! 3D vector type with operations for ray tracing

use std::ops::{Add, Sub, Mul, Div, Neg, Index, AddAssign, MulAssign, DivAssign};
use rand::Rng;

/// A 3D vector
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Create a new vector
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Zero vector
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Unit vector (1, 1, 1)
    pub fn one() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Dot product
    pub fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Vector length
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Squared length (avoids sqrt)
    pub fn length_squared(&self) -> f32 {
        self.dot(self)
    }

    /// Normalize to unit vector
    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        if len > 0.0 {
            *self / len
        } else {
            Vec3::zero()
        }
    }

    /// Reflect vector around normal
    pub fn reflect(&self, normal: &Vec3) -> Vec3 {
        *self - *normal * 2.0 * self.dot(normal)
    }

    /// Refract vector through surface
    /// Returns None for total internal reflection
    pub fn refract(&self, normal: &Vec3, eta_ratio: f32) -> Option<Vec3> {
        let cos_theta = (-*self).dot(normal).min(1.0);
        let r_out_perp = (*self + *normal * cos_theta) * eta_ratio;
        let r_out_perp_len_sq = r_out_perp.length_squared();
        
        if r_out_perp_len_sq > 1.0 {
            return None; // Total internal reflection
        }
        
        let r_out_parallel = *normal * -(1.0 - r_out_perp_len_sq).sqrt();
        Some(r_out_perp + r_out_parallel)
    }

    /// Check if vector is near zero (for avoiding division by zero)
    pub fn near_zero(&self) -> bool {
        const EPSILON: f32 = 1e-8;
        self.x.abs() < EPSILON && self.y.abs() < EPSILON && self.z.abs() < EPSILON
    }

    /// Random vector with components in [0, 1)
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        Vec3::new(rng.gen(), rng.gen(), rng.gen())
    }

    /// Random vector with components in [min, max)
    pub fn random_range<R: Rng>(rng: &mut R, min: f32, max: f32) -> Self {
        Vec3::new(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        )
    }

    /// Random point in unit sphere (rejection sampling)
    pub fn random_in_unit_sphere<R: Rng>(rng: &mut R) -> Self {
        loop {
            let p = Vec3::random_range(rng, -1.0, 1.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    /// Random unit vector (uniform on sphere surface)
    pub fn random_unit_vector<R: Rng>(rng: &mut R) -> Self {
        Vec3::random_in_unit_sphere(rng).normalize()
    }

    /// Random vector in hemisphere around normal
    pub fn random_in_hemisphere<R: Rng>(normal: &Vec3, rng: &mut R) -> Self {
        let on_unit_sphere = Vec3::random_unit_vector(rng);
        if on_unit_sphere.dot(normal) > 0.0 {
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
    }

    /// Random point in unit disk (for DOF)
    pub fn random_in_unit_disk<R: Rng>(rng: &mut R) -> Self {
        loop {
            let p = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    /// Component-wise minimum
    pub fn min(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    /// Component-wise maximum
    pub fn max(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    /// Create vector with all components equal
    pub fn splat(v: f32) -> Vec3 {
        Vec3::new(v, v, v)
    }

    /// Get maximum component
    pub fn max_component(&self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    /// Get minimum component
    pub fn min_component(&self) -> f32 {
        self.x.min(self.y).min(self.z)
    }

    /// Clamp components to range
    pub fn clamp(&self, min: Vec3, max: Vec3) -> Vec3 {
        Vec3::new(
            self.x.clamp(min.x, max.x),
            self.y.clamp(min.y, max.y),
            self.z.clamp(min.z, max.z),
        )
    }

    /// Sum of all components
    pub fn sum(&self) -> f32 {
        self.x + self.y + self.z
    }

    /// Average of components
    pub fn avg(&self) -> f32 {
        self.sum() / 3.0
    }

    /// Luminance (perceptual brightness)
    pub fn luminance(&self) -> f32 {
        0.2126 * self.x + 0.7152 * self.y + 0.0722 * self.z
    }
}

// Operator implementations

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, scalar: f32) -> Vec3 {
        Vec3::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, scalar: f32) -> Vec3 {
        Vec3::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of bounds"),
        }
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, scalar: f32) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, scalar: f32) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Vec3::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn test_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert_eq!(c, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(v.length(), 5.0);
    }

    #[test]
    fn test_normalize() {
        let v = Vec3::new(3.0, 0.0, 0.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
        assert_eq!(n, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0).normalize();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = v.reflect(&n);
        assert!((r.x - v.x).abs() < 1e-6);
        assert!((r.y + v.y).abs() < 1e-6);
    }

    #[test]
    fn test_refract() {
        let v = Vec3::new(0.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = v.refract(&n, 1.0).unwrap();
        assert!((r - v).length() < 1e-6);
    }

    #[test]
    fn test_near_zero() {
        assert!(Vec3::new(1e-9, 1e-9, 1e-9).near_zero());
        assert!(!Vec3::new(0.1, 0.0, 0.0).near_zero());
    }

    #[test]
    fn test_operators() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        
        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(a - b, Vec3::new(-3.0, -3.0, -3.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a / 2.0, Vec3::new(0.5, 1.0, 1.5));
        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn test_index() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }
}
