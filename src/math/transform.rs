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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let t = Transform::identity();
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(t.transform_point(&p), p);
    }

    #[test]
    fn test_translate() {
        let t = Transform::translate(Vec3::new(1.0, 2.0, 3.0));
        let p = Vec3::new(0.0, 0.0, 0.0);
        assert_eq!(t.transform_point(&p), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_scale() {
        let t = Transform::scale(Vec3::new(2.0, 3.0, 4.0));
        let p = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(t.transform_point(&p), Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_inverse() {
        let t = Transform::translate(Vec3::new(1.0, 2.0, 3.0));
        let p = Vec3::new(5.0, 5.0, 5.0);
        let transformed = t.transform_point(&p);
        let back = t.inverse().transform_point(&transformed);
        assert!((back - p).length() < 1e-6);
    }
}
