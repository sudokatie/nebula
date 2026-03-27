//! 4x4 transformation matrices

use super::{Vec3, Ray};

/// A 4x4 transformation matrix with cached inverse
#[derive(Debug, Clone)]
pub struct Transform {
    matrix: [[f32; 4]; 4],
    inverse: [[f32; 4]; 4],
}

impl Transform {
    /// Identity transform
    pub fn identity() -> Self {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self {
            matrix: identity,
            inverse: identity,
        }
    }

    /// Translation transform
    pub fn translate(v: Vec3) -> Self {
        let matrix = [
            [1.0, 0.0, 0.0, v.x],
            [0.0, 1.0, 0.0, v.y],
            [0.0, 0.0, 1.0, v.z],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inverse = [
            [1.0, 0.0, 0.0, -v.x],
            [0.0, 1.0, 0.0, -v.y],
            [0.0, 0.0, 1.0, -v.z],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self { matrix, inverse }
    }

    /// Scale transform
    pub fn scale(v: Vec3) -> Self {
        let matrix = [
            [v.x, 0.0, 0.0, 0.0],
            [0.0, v.y, 0.0, 0.0],
            [0.0, 0.0, v.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inverse = [
            [1.0 / v.x, 0.0, 0.0, 0.0],
            [0.0, 1.0 / v.y, 0.0, 0.0],
            [0.0, 0.0, 1.0 / v.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self { matrix, inverse }
    }

    /// Uniform scale transform
    pub fn uniform_scale(s: f32) -> Self {
        Self::scale(Vec3::new(s, s, s))
    }

    /// Rotation around X axis
    pub fn rotate_x(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inverse = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, s, 0.0],
            [0.0, -s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self { matrix, inverse }
    }

    /// Rotation around Y axis
    pub fn rotate_y(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let matrix = [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inverse = [
            [c, 0.0, -s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self { matrix, inverse }
    }

    /// Rotation around Z axis
    pub fn rotate_z(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let matrix = [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inverse = [
            [c, s, 0.0, 0.0],
            [-s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self { matrix, inverse }
    }

    /// Rotation around arbitrary axis
    pub fn rotate(axis: Vec3, angle: f32) -> Self {
        let a = axis.normalize();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;
        
        let matrix = [
            [t * a.x * a.x + c, t * a.x * a.y - s * a.z, t * a.x * a.z + s * a.y, 0.0],
            [t * a.x * a.y + s * a.z, t * a.y * a.y + c, t * a.y * a.z - s * a.x, 0.0],
            [t * a.x * a.z - s * a.y, t * a.y * a.z + s * a.x, t * a.z * a.z + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        // Inverse of rotation is rotation by -angle
        let c = (-angle).cos();
        let s = (-angle).sin();
        let t = 1.0 - c;
        
        let inverse = [
            [t * a.x * a.x + c, t * a.x * a.y - s * a.z, t * a.x * a.z + s * a.y, 0.0],
            [t * a.x * a.y + s * a.z, t * a.y * a.y + c, t * a.y * a.z - s * a.x, 0.0],
            [t * a.x * a.z - s * a.y, t * a.y * a.z + s * a.x, t * a.z * a.z + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        Self { matrix, inverse }
    }

    /// Look-at transform (useful for cameras)
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let forward = (target - eye).normalize();
        let right = forward.cross(&up).normalize();
        let new_up = right.cross(&forward);
        
        let matrix = [
            [right.x, new_up.x, -forward.x, eye.x],
            [right.y, new_up.y, -forward.y, eye.y],
            [right.z, new_up.z, -forward.z, eye.z],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        // Compute inverse (transpose of rotation, then negate translation)
        let inverse = [
            [right.x, right.y, right.z, -right.dot(&eye)],
            [new_up.x, new_up.y, new_up.z, -new_up.dot(&eye)],
            [-forward.x, -forward.y, -forward.z, forward.dot(&eye)],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        Self { matrix, inverse }
    }

    /// Compose this transform with another: self * other
    /// Result applies `other` first, then `self`
    pub fn then(&self, other: &Transform) -> Transform {
        Transform {
            matrix: Self::multiply(&self.matrix, &other.matrix),
            inverse: Self::multiply(&other.inverse, &self.inverse),
        }
    }

    /// Compose transforms: other * self
    /// Result applies `self` first, then `other`
    pub fn compose(&self, other: &Transform) -> Transform {
        other.then(self)
    }

    fn multiply(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut result = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = a[i][0] * b[0][j] 
                    + a[i][1] * b[1][j] 
                    + a[i][2] * b[2][j] 
                    + a[i][3] * b[3][j];
            }
        }
        result
    }

    /// Transform a point (applies translation)
    pub fn transform_point(&self, p: &Vec3) -> Vec3 {
        let m = &self.matrix;
        Vec3::new(
            m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
            m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
            m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3],
        )
    }

    /// Transform a vector (ignores translation)
    pub fn transform_vector(&self, v: &Vec3) -> Vec3 {
        let m = &self.matrix;
        Vec3::new(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        )
    }

    /// Transform a normal (uses inverse transpose)
    pub fn transform_normal(&self, n: &Vec3) -> Vec3 {
        let m = &self.inverse;
        // Transpose of inverse for normals
        Vec3::new(
            m[0][0] * n.x + m[1][0] * n.y + m[2][0] * n.z,
            m[0][1] * n.x + m[1][1] * n.y + m[2][1] * n.z,
            m[0][2] * n.x + m[1][2] * n.y + m[2][2] * n.z,
        ).normalize()
    }

    /// Transform a ray
    pub fn transform_ray(&self, r: &Ray) -> Ray {
        Ray::new(
            self.transform_point(&r.origin),
            self.transform_vector(&r.direction),
        )
    }

    /// Get the inverse transform
    pub fn inverse(&self) -> Transform {
        Transform {
            matrix: self.inverse,
            inverse: self.matrix,
        }
    }

    /// Check if this is approximately the identity transform
    pub fn is_identity(&self) -> bool {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        for i in 0..4 {
            for j in 0..4 {
                if (self.matrix[i][j] - identity[i][j]).abs() > 1e-6 {
                    return false;
                }
            }
        }
        true
    }

    /// Get the raw matrix
    pub fn matrix(&self) -> &[[f32; 4]; 4] {
        &self.matrix
    }

    /// Get the inverse matrix
    pub fn inverse_matrix(&self) -> &[[f32; 4]; 4] {
        &self.inverse
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let t = Transform::identity();
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(t.transform_point(&p), p);
        assert!(t.is_identity());
    }

    #[test]
    fn test_translate() {
        let t = Transform::translate(Vec3::new(1.0, 2.0, 3.0));
        let p = Vec3::new(0.0, 0.0, 0.0);
        assert_eq!(t.transform_point(&p), Vec3::new(1.0, 2.0, 3.0));
        assert!(!t.is_identity());
    }

    #[test]
    fn test_scale() {
        let t = Transform::scale(Vec3::new(2.0, 3.0, 4.0));
        let p = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(t.transform_point(&p), Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_uniform_scale() {
        let t = Transform::uniform_scale(2.0);
        let p = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(t.transform_point(&p), Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_inverse() {
        let t = Transform::translate(Vec3::new(1.0, 2.0, 3.0));
        let p = Vec3::new(5.0, 5.0, 5.0);
        let transformed = t.transform_point(&p);
        let back = t.inverse().transform_point(&transformed);
        assert!((back - p).length() < 1e-6);
    }

    #[test]
    fn test_compose() {
        let translate = Transform::translate(Vec3::new(1.0, 0.0, 0.0));
        let scale = Transform::scale(Vec3::new(2.0, 2.0, 2.0));
        
        // Scale then translate
        let composed = translate.then(&scale);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let result = composed.transform_point(&p);
        // Scale 1 -> 2, then translate +1 -> 3
        assert!((result.x - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotate_y() {
        use std::f32::consts::PI;
        let t = Transform::rotate_y(PI / 2.0);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let result = t.transform_point(&p);
        // Rotation by 90 degrees around Y: (1,0,0) -> (0,0,-1)
        assert!(result.x.abs() < 1e-6);
        assert!((result.z - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_normal_transform() {
        let scale = Transform::scale(Vec3::new(2.0, 1.0, 1.0));
        let n = Vec3::new(1.0, 0.0, 0.0);
        let transformed = scale.transform_normal(&n);
        // Normal should be normalized after transform
        assert!((transformed.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ray_transform() {
        let t = Transform::translate(Vec3::new(1.0, 0.0, 0.0));
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let transformed = t.transform_ray(&ray);
        assert!((transformed.origin.x - 1.0).abs() < 1e-6);
        assert_eq!(transformed.direction, ray.direction);
    }
}
