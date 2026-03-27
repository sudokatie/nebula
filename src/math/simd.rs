//! SIMD-accelerated vector types (4 Vec3s packed)

use super::Vec3;

/// 4 packed Vec3s for SIMD operations
#[derive(Clone, Copy)]
pub struct Vec3x4 {
    pub x: [f32; 4],
    pub y: [f32; 4],
    pub z: [f32; 4],
}

impl Vec3x4 {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> Self {
        Self {
            x: [v0.x, v1.x, v2.x, v3.x],
            y: [v0.y, v1.y, v2.y, v3.y],
            z: [v0.z, v1.z, v2.z, v3.z],
        }
    }

    pub fn splat(v: Vec3) -> Self {
        Self::new(v, v, v, v)
    }

    pub fn zero() -> Self {
        Self {
            x: [0.0; 4],
            y: [0.0; 4],
            z: [0.0; 4],
        }
    }

    pub fn get(&self, i: usize) -> Vec3 {
        Vec3::new(self.x[i], self.y[i], self.z[i])
    }

    pub fn set(&mut self, i: usize, v: Vec3) {
        self.x[i] = v.x;
        self.y[i] = v.y;
        self.z[i] = v.z;
    }

    /// SIMD dot product (returns 4 scalars)
    #[cfg(target_arch = "x86_64")]
    pub fn dot(&self, other: &Vec3x4) -> [f32; 4] {
        use std::arch::x86_64::*;
        unsafe {
            let x1 = _mm_loadu_ps(self.x.as_ptr());
            let y1 = _mm_loadu_ps(self.y.as_ptr());
            let z1 = _mm_loadu_ps(self.z.as_ptr());
            let x2 = _mm_loadu_ps(other.x.as_ptr());
            let y2 = _mm_loadu_ps(other.y.as_ptr());
            let z2 = _mm_loadu_ps(other.z.as_ptr());

            let xx = _mm_mul_ps(x1, x2);
            let yy = _mm_mul_ps(y1, y2);
            let zz = _mm_mul_ps(z1, z2);
            let sum = _mm_add_ps(_mm_add_ps(xx, yy), zz);

            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sum);
            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn dot(&self, other: &Vec3x4) -> [f32; 4] {
        use std::arch::aarch64::*;
        unsafe {
            let x1 = vld1q_f32(self.x.as_ptr());
            let y1 = vld1q_f32(self.y.as_ptr());
            let z1 = vld1q_f32(self.z.as_ptr());
            let x2 = vld1q_f32(other.x.as_ptr());
            let y2 = vld1q_f32(other.y.as_ptr());
            let z2 = vld1q_f32(other.z.as_ptr());

            let xx = vmulq_f32(x1, x2);
            let yy = vmulq_f32(y1, y2);
            let zz = vmulq_f32(z1, z2);
            let sum = vaddq_f32(vaddq_f32(xx, yy), zz);

            let mut result = [0.0f32; 4];
            vst1q_f32(result.as_mut_ptr(), sum);
            result
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn dot(&self, other: &Vec3x4) -> [f32; 4] {
        [
            self.x[0] * other.x[0] + self.y[0] * other.y[0] + self.z[0] * other.z[0],
            self.x[1] * other.x[1] + self.y[1] * other.y[1] + self.z[1] * other.z[1],
            self.x[2] * other.x[2] + self.y[2] * other.y[2] + self.z[2] * other.z[2],
            self.x[3] * other.x[3] + self.y[3] * other.y[3] + self.z[3] * other.z[3],
        ]
    }

    /// SIMD length squared
    pub fn length_squared(&self) -> [f32; 4] {
        self.dot(self)
    }

    /// SIMD length
    #[cfg(target_arch = "x86_64")]
    pub fn length(&self) -> [f32; 4] {
        use std::arch::x86_64::*;
        unsafe {
            let len_sq = self.length_squared();
            let v = _mm_loadu_ps(len_sq.as_ptr());
            let sqrt = _mm_sqrt_ps(v);
            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sqrt);
            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn length(&self) -> [f32; 4] {
        use std::arch::aarch64::*;
        unsafe {
            let len_sq = self.length_squared();
            let v = vld1q_f32(len_sq.as_ptr());
            let sqrt = vsqrtq_f32(v);
            let mut result = [0.0f32; 4];
            vst1q_f32(result.as_mut_ptr(), sqrt);
            result
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn length(&self) -> [f32; 4] {
        let sq = self.length_squared();
        [sq[0].sqrt(), sq[1].sqrt(), sq[2].sqrt(), sq[3].sqrt()]
    }

    /// SIMD add
    #[cfg(target_arch = "x86_64")]
    pub fn add(&self, other: &Vec3x4) -> Vec3x4 {
        use std::arch::x86_64::*;
        unsafe {
            let mut result = Vec3x4::zero();
            let x = _mm_add_ps(_mm_loadu_ps(self.x.as_ptr()), _mm_loadu_ps(other.x.as_ptr()));
            let y = _mm_add_ps(_mm_loadu_ps(self.y.as_ptr()), _mm_loadu_ps(other.y.as_ptr()));
            let z = _mm_add_ps(_mm_loadu_ps(self.z.as_ptr()), _mm_loadu_ps(other.z.as_ptr()));
            _mm_storeu_ps(result.x.as_mut_ptr(), x);
            _mm_storeu_ps(result.y.as_mut_ptr(), y);
            _mm_storeu_ps(result.z.as_mut_ptr(), z);
            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn add(&self, other: &Vec3x4) -> Vec3x4 {
        use std::arch::aarch64::*;
        unsafe {
            let mut result = Vec3x4::zero();
            let x = vaddq_f32(vld1q_f32(self.x.as_ptr()), vld1q_f32(other.x.as_ptr()));
            let y = vaddq_f32(vld1q_f32(self.y.as_ptr()), vld1q_f32(other.y.as_ptr()));
            let z = vaddq_f32(vld1q_f32(self.z.as_ptr()), vld1q_f32(other.z.as_ptr()));
            vst1q_f32(result.x.as_mut_ptr(), x);
            vst1q_f32(result.y.as_mut_ptr(), y);
            vst1q_f32(result.z.as_mut_ptr(), z);
            result
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn add(&self, other: &Vec3x4) -> Vec3x4 {
        Vec3x4 {
            x: [self.x[0] + other.x[0], self.x[1] + other.x[1], self.x[2] + other.x[2], self.x[3] + other.x[3]],
            y: [self.y[0] + other.y[0], self.y[1] + other.y[1], self.y[2] + other.y[2], self.y[3] + other.y[3]],
            z: [self.z[0] + other.z[0], self.z[1] + other.z[1], self.z[2] + other.z[2], self.z[3] + other.z[3]],
        }
    }

    /// SIMD sub
    #[cfg(target_arch = "x86_64")]
    pub fn sub(&self, other: &Vec3x4) -> Vec3x4 {
        use std::arch::x86_64::*;
        unsafe {
            let mut result = Vec3x4::zero();
            let x = _mm_sub_ps(_mm_loadu_ps(self.x.as_ptr()), _mm_loadu_ps(other.x.as_ptr()));
            let y = _mm_sub_ps(_mm_loadu_ps(self.y.as_ptr()), _mm_loadu_ps(other.y.as_ptr()));
            let z = _mm_sub_ps(_mm_loadu_ps(self.z.as_ptr()), _mm_loadu_ps(other.z.as_ptr()));
            _mm_storeu_ps(result.x.as_mut_ptr(), x);
            _mm_storeu_ps(result.y.as_mut_ptr(), y);
            _mm_storeu_ps(result.z.as_mut_ptr(), z);
            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn sub(&self, other: &Vec3x4) -> Vec3x4 {
        use std::arch::aarch64::*;
        unsafe {
            let mut result = Vec3x4::zero();
            let x = vsubq_f32(vld1q_f32(self.x.as_ptr()), vld1q_f32(other.x.as_ptr()));
            let y = vsubq_f32(vld1q_f32(self.y.as_ptr()), vld1q_f32(other.y.as_ptr()));
            let z = vsubq_f32(vld1q_f32(self.z.as_ptr()), vld1q_f32(other.z.as_ptr()));
            vst1q_f32(result.x.as_mut_ptr(), x);
            vst1q_f32(result.y.as_mut_ptr(), y);
            vst1q_f32(result.z.as_mut_ptr(), z);
            result
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn sub(&self, other: &Vec3x4) -> Vec3x4 {
        Vec3x4 {
            x: [self.x[0] - other.x[0], self.x[1] - other.x[1], self.x[2] - other.x[2], self.x[3] - other.x[3]],
            y: [self.y[0] - other.y[0], self.y[1] - other.y[1], self.y[2] - other.y[2], self.y[3] - other.y[3]],
            z: [self.z[0] - other.z[0], self.z[1] - other.z[1], self.z[2] - other.z[2], self.z[3] - other.z[3]],
        }
    }

    /// SIMD multiply by scalar (4 different scalars)
    #[cfg(target_arch = "x86_64")]
    pub fn mul_scalar(&self, s: [f32; 4]) -> Vec3x4 {
        use std::arch::x86_64::*;
        unsafe {
            let sv = _mm_loadu_ps(s.as_ptr());
            let mut result = Vec3x4::zero();
            _mm_storeu_ps(result.x.as_mut_ptr(), _mm_mul_ps(_mm_loadu_ps(self.x.as_ptr()), sv));
            _mm_storeu_ps(result.y.as_mut_ptr(), _mm_mul_ps(_mm_loadu_ps(self.y.as_ptr()), sv));
            _mm_storeu_ps(result.z.as_mut_ptr(), _mm_mul_ps(_mm_loadu_ps(self.z.as_ptr()), sv));
            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn mul_scalar(&self, s: [f32; 4]) -> Vec3x4 {
        use std::arch::aarch64::*;
        unsafe {
            let sv = vld1q_f32(s.as_ptr());
            let mut result = Vec3x4::zero();
            vst1q_f32(result.x.as_mut_ptr(), vmulq_f32(vld1q_f32(self.x.as_ptr()), sv));
            vst1q_f32(result.y.as_mut_ptr(), vmulq_f32(vld1q_f32(self.y.as_ptr()), sv));
            vst1q_f32(result.z.as_mut_ptr(), vmulq_f32(vld1q_f32(self.z.as_ptr()), sv));
            result
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn mul_scalar(&self, s: [f32; 4]) -> Vec3x4 {
        Vec3x4 {
            x: [self.x[0] * s[0], self.x[1] * s[1], self.x[2] * s[2], self.x[3] * s[3]],
            y: [self.y[0] * s[0], self.y[1] * s[1], self.y[2] * s[2], self.y[3] * s[3]],
            z: [self.z[0] * s[0], self.z[1] * s[1], self.z[2] * s[2], self.z[3] * s[3]],
        }
    }

    /// SIMD multiply (component-wise)
    #[cfg(target_arch = "x86_64")]
    pub fn mul(&self, other: &Vec3x4) -> Vec3x4 {
        use std::arch::x86_64::*;
        unsafe {
            let mut result = Vec3x4::zero();
            _mm_storeu_ps(result.x.as_mut_ptr(), _mm_mul_ps(_mm_loadu_ps(self.x.as_ptr()), _mm_loadu_ps(other.x.as_ptr())));
            _mm_storeu_ps(result.y.as_mut_ptr(), _mm_mul_ps(_mm_loadu_ps(self.y.as_ptr()), _mm_loadu_ps(other.y.as_ptr())));
            _mm_storeu_ps(result.z.as_mut_ptr(), _mm_mul_ps(_mm_loadu_ps(self.z.as_ptr()), _mm_loadu_ps(other.z.as_ptr())));
            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn mul(&self, other: &Vec3x4) -> Vec3x4 {
        use std::arch::aarch64::*;
        unsafe {
            let mut result = Vec3x4::zero();
            vst1q_f32(result.x.as_mut_ptr(), vmulq_f32(vld1q_f32(self.x.as_ptr()), vld1q_f32(other.x.as_ptr())));
            vst1q_f32(result.y.as_mut_ptr(), vmulq_f32(vld1q_f32(self.y.as_ptr()), vld1q_f32(other.y.as_ptr())));
            vst1q_f32(result.z.as_mut_ptr(), vmulq_f32(vld1q_f32(self.z.as_ptr()), vld1q_f32(other.z.as_ptr())));
            result
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn mul(&self, other: &Vec3x4) -> Vec3x4 {
        Vec3x4 {
            x: [self.x[0] * other.x[0], self.x[1] * other.x[1], self.x[2] * other.x[2], self.x[3] * other.x[3]],
            y: [self.y[0] * other.y[0], self.y[1] * other.y[1], self.y[2] * other.y[2], self.y[3] * other.y[3]],
            z: [self.z[0] * other.z[0], self.z[1] * other.z[1], self.z[2] * other.z[2], self.z[3] * other.z[3]],
        }
    }

    /// SIMD cross product
    pub fn cross(&self, other: &Vec3x4) -> Vec3x4 {
        // Cross product: (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let ax = _mm_loadu_ps(self.x.as_ptr());
            let ay = _mm_loadu_ps(self.y.as_ptr());
            let az = _mm_loadu_ps(self.z.as_ptr());
            let bx = _mm_loadu_ps(other.x.as_ptr());
            let by = _mm_loadu_ps(other.y.as_ptr());
            let bz = _mm_loadu_ps(other.z.as_ptr());

            let mut result = Vec3x4::zero();
            _mm_storeu_ps(result.x.as_mut_ptr(), _mm_sub_ps(_mm_mul_ps(ay, bz), _mm_mul_ps(az, by)));
            _mm_storeu_ps(result.y.as_mut_ptr(), _mm_sub_ps(_mm_mul_ps(az, bx), _mm_mul_ps(ax, bz)));
            _mm_storeu_ps(result.z.as_mut_ptr(), _mm_sub_ps(_mm_mul_ps(ax, by), _mm_mul_ps(ay, bx)));
            return result;
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let ax = vld1q_f32(self.x.as_ptr());
            let ay = vld1q_f32(self.y.as_ptr());
            let az = vld1q_f32(self.z.as_ptr());
            let bx = vld1q_f32(other.x.as_ptr());
            let by = vld1q_f32(other.y.as_ptr());
            let bz = vld1q_f32(other.z.as_ptr());

            let mut result = Vec3x4::zero();
            vst1q_f32(result.x.as_mut_ptr(), vsubq_f32(vmulq_f32(ay, bz), vmulq_f32(az, by)));
            vst1q_f32(result.y.as_mut_ptr(), vsubq_f32(vmulq_f32(az, bx), vmulq_f32(ax, bz)));
            vst1q_f32(result.z.as_mut_ptr(), vsubq_f32(vmulq_f32(ax, by), vmulq_f32(ay, bx)));
            return result;
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
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
    }

    /// Normalize all 4 vectors
    pub fn normalize(&self) -> Vec3x4 {
        let len = self.length();
        let inv_len = [1.0 / len[0], 1.0 / len[1], 1.0 / len[2], 1.0 / len[3]];
        self.mul_scalar(inv_len)
    }
}

impl Default for Vec3x4 {
    fn default() -> Self {
        Self::zero()
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
    fn test_vec3x4_dot() {
        let a = Vec3x4::splat(Vec3::new(1.0, 0.0, 0.0));
        let b = Vec3x4::splat(Vec3::new(1.0, 0.0, 0.0));
        let dots = a.dot(&b);
        assert!((dots[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_vec3x4_length() {
        let v = Vec3x4::splat(Vec3::new(3.0, 4.0, 0.0));
        let len = v.length();
        assert!((len[0] - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_vec3x4_add() {
        let a = Vec3x4::splat(Vec3::new(1.0, 2.0, 3.0));
        let b = Vec3x4::splat(Vec3::new(4.0, 5.0, 6.0));
        let c = a.add(&b);
        assert_eq!(c.get(0), Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vec3x4_cross() {
        let a = Vec3x4::splat(Vec3::new(1.0, 0.0, 0.0));
        let b = Vec3x4::splat(Vec3::new(0.0, 1.0, 0.0));
        let c = a.cross(&b);
        let result = c.get(0);
        assert!((result.x).abs() < 0.001);
        assert!((result.y).abs() < 0.001);
        assert!((result.z - 1.0).abs() < 0.001);
    }
}
