//! Instance - transformed geometry for instancing

use crate::math::{Vec3, Ray, Transform};
use crate::accel::AABB;
use super::{HitRecord, Hittable};

/// An instance wraps geometry with a transformation for instancing
pub struct Instance {
    /// The underlying geometry
    geometry: Box<dyn Hittable>,
    /// Object-to-world transform
    object_to_world: Transform,
    /// World-to-object transform (cached inverse)
    world_to_object: Transform,
    /// Transformed bounding box (cached)
    world_bounds: Option<AABB>,
    /// Override material (None = use geometry's material)
    material_override: Option<usize>,
}

impl Instance {
    /// Create a new instance with the given transform
    pub fn new(geometry: Box<dyn Hittable>, transform: Transform) -> Self {
        let world_to_object = transform.inverse();
        let world_bounds = Self::compute_world_bounds(&geometry, &transform);
        
        Self {
            geometry,
            object_to_world: transform,
            world_to_object,
            world_bounds,
            material_override: None,
        }
    }

    /// Create instance with material override
    pub fn with_material(mut self, material_id: usize) -> Self {
        self.material_override = Some(material_id);
        self
    }

    /// Create translated instance
    pub fn translated(geometry: Box<dyn Hittable>, offset: Vec3) -> Self {
        Self::new(geometry, Transform::translate(offset))
    }

    /// Create scaled instance
    pub fn scaled(geometry: Box<dyn Hittable>, scale: Vec3) -> Self {
        Self::new(geometry, Transform::scale(scale))
    }

    /// Create rotated instance (angle in radians)
    pub fn rotated_y(geometry: Box<dyn Hittable>, angle: f32) -> Self {
        Self::new(geometry, Transform::rotate_y(angle))
    }

    fn compute_world_bounds(geometry: &Box<dyn Hittable>, transform: &Transform) -> Option<AABB> {
        let local_bounds = geometry.bounding_box()?;
        
        // Transform all 8 corners of the bounding box
        let corners = [
            Vec3::new(local_bounds.min.x, local_bounds.min.y, local_bounds.min.z),
            Vec3::new(local_bounds.max.x, local_bounds.min.y, local_bounds.min.z),
            Vec3::new(local_bounds.min.x, local_bounds.max.y, local_bounds.min.z),
            Vec3::new(local_bounds.max.x, local_bounds.max.y, local_bounds.min.z),
            Vec3::new(local_bounds.min.x, local_bounds.min.y, local_bounds.max.z),
            Vec3::new(local_bounds.max.x, local_bounds.min.y, local_bounds.max.z),
            Vec3::new(local_bounds.min.x, local_bounds.max.y, local_bounds.max.z),
            Vec3::new(local_bounds.max.x, local_bounds.max.y, local_bounds.max.z),
        ];

        let transformed: Vec<Vec3> = corners
            .iter()
            .map(|c| transform.transform_point(c))
            .collect();

        Some(AABB::from_points(&transformed))
    }
}

impl Hittable for Instance {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        // Transform ray from world space to object space
        let local_ray = self.world_to_object.transform_ray(ray);
        
        // Test intersection in object space
        let mut hit = self.geometry.hit(&local_ray, t_min, t_max)?;
        
        // Transform hit back to world space
        hit.point = self.object_to_world.transform_point(&hit.point);
        hit.geometric_normal = self.object_to_world.transform_normal(&hit.geometric_normal);
        hit.normal = self.object_to_world.transform_normal(&hit.normal);
        
        // Recalculate t in world space
        hit.t = (hit.point - ray.origin).length() / ray.direction.length();
        
        // Apply material override if set
        if let Some(mat_id) = self.material_override {
            hit.material_id = mat_id;
        }
        
        Some(hit)
    }

    fn bounding_box(&self) -> Option<AABB> {
        self.world_bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;

    #[test]
    fn test_instance_translated() {
        let sphere = Box::new(Sphere::new(Vec3::zero(), 1.0, 0));
        let instance = Instance::translated(sphere, Vec3::new(5.0, 0.0, 0.0));
        
        // Ray should miss original position
        let ray1 = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(instance.hit(&ray1, 0.001, f32::INFINITY).is_none());
        
        // Ray should hit translated position
        let ray2 = Ray::new(Vec3::new(5.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(instance.hit(&ray2, 0.001, f32::INFINITY).is_some());
    }

    #[test]
    fn test_instance_scaled() {
        let sphere = Box::new(Sphere::new(Vec3::zero(), 1.0, 0));
        let instance = Instance::scaled(sphere, Vec3::new(2.0, 2.0, 2.0));
        
        // Ray at distance 1.5 should now hit (scaled radius = 2)
        let ray = Ray::new(Vec3::new(1.5, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(instance.hit(&ray, 0.001, f32::INFINITY).is_some());
    }

    #[test]
    fn test_instance_bounds() {
        let sphere = Box::new(Sphere::new(Vec3::zero(), 1.0, 0));
        let instance = Instance::translated(sphere, Vec3::new(10.0, 0.0, 0.0));
        
        let bounds = instance.bounding_box().unwrap();
        assert!(bounds.min.x > 8.0);
        assert!(bounds.max.x < 12.0);
    }

    #[test]
    fn test_instance_material_override() {
        let sphere = Box::new(Sphere::new(Vec3::zero(), 1.0, 0));
        let instance = Instance::new(sphere, Transform::identity()).with_material(42);
        
        let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = instance.hit(&ray, 0.001, f32::INFINITY).unwrap();
        assert_eq!(hit.material_id, 42);
    }
}
