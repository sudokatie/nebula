//! Bounding Volume Hierarchy

use crate::math::Ray;
use crate::geometry::{HitRecord, Hittable};
use super::AABB;

/// BVH node
pub struct BVHNode {
    bounds: AABB,
    left: Option<Box<BVHNode>>,
    right: Option<Box<BVHNode>>,
    primitive_start: usize,
    primitive_count: usize,
}

/// Bounding Volume Hierarchy
pub struct BVH {
    root: Option<BVHNode>,
    primitives: Vec<Box<dyn Hittable>>,
}

impl BVH {
    /// Build BVH from primitives
    pub fn new(primitives: Vec<Box<dyn Hittable>>) -> Self {
        if primitives.is_empty() {
            return Self { root: None, primitives };
        }

        let mut indices: Vec<usize> = (0..primitives.len()).collect();
        let bounds: Vec<AABB> = primitives
            .iter()
            .map(|p| p.bounding_box().unwrap_or_else(AABB::empty))
            .collect();

        let root = Self::build(&mut indices, &bounds, 0, primitives.len());
        
        // Reorder primitives according to indices
        let mut new_primitives: Vec<Option<Box<dyn Hittable>>> = primitives.into_iter().map(Some).collect();
        let primitives: Vec<Box<dyn Hittable>> = indices
            .iter()
            .map(|&i| new_primitives[i].take().unwrap())
            .collect();

        Self { root: Some(root), primitives }
    }

    fn build(indices: &mut [usize], bounds: &[AABB], start: usize, end: usize) -> BVHNode {
        let count = end - start;
        
        // Compute bounds for this node
        let mut node_bounds = AABB::empty();
        for &idx in &indices[start..end] {
            node_bounds = AABB::surrounding(&node_bounds, &bounds[idx]);
        }

        // Leaf node
        if count <= 2 {
            return BVHNode {
                bounds: node_bounds,
                left: None,
                right: None,
                primitive_start: start,
                primitive_count: count,
            };
        }

        // Find split axis and position
        let axis = node_bounds.longest_axis();
        let mid = start + count / 2;

        // Sort by centroid along axis
        indices[start..end].sort_by(|&a, &b| {
            let ca = bounds[a].centroid()[axis];
            let cb = bounds[b].centroid()[axis];
            ca.partial_cmp(&cb).unwrap()
        });

        let left = Box::new(Self::build(indices, bounds, start, mid));
        let right = Box::new(Self::build(indices, bounds, mid, end));

        BVHNode {
            bounds: node_bounds,
            left: Some(left),
            right: Some(right),
            primitive_start: 0,
            primitive_count: 0,
        }
    }

    /// Test ray against BVH
    pub fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        match &self.root {
            Some(node) => {
                // Precompute inverse direction for faster AABB tests
                let inv_dir = crate::math::Vec3::new(
                    1.0 / ray.direction.x,
                    1.0 / ray.direction.y,
                    1.0 / ray.direction.z,
                );
                self.hit_node(node, ray, &inv_dir, t_min, t_max)
            }
            None => None,
        }
    }

    fn hit_node(&self, node: &BVHNode, ray: &Ray, inv_dir: &crate::math::Vec3, t_min: f32, t_max: f32) -> Option<HitRecord> {
        // Use precomputed inverse direction for faster AABB test
        if !node.bounds.hit_precomputed(&ray.origin, inv_dir, t_min, t_max) {
            return None;
        }

        // Leaf node
        if node.left.is_none() && node.right.is_none() {
            let mut closest: Option<HitRecord> = None;
            let mut closest_t = t_max;

            for i in 0..node.primitive_count {
                let idx = node.primitive_start + i;
                if let Some(rec) = self.primitives[idx].hit(ray, t_min, closest_t) {
                    closest_t = rec.t;
                    closest = Some(rec);
                }
            }
            return closest;
        }

        // Interior node
        let mut closest: Option<HitRecord> = None;
        let mut closest_t = t_max;

        if let Some(left) = &node.left {
            if let Some(rec) = self.hit_node(left, ray, inv_dir, t_min, closest_t) {
                closest_t = rec.t;
                closest = Some(rec);
            }
        }

        if let Some(right) = &node.right {
            if let Some(rec) = self.hit_node(right, ray, inv_dir, t_min, closest_t) {
                closest = Some(rec);
            }
        }

        closest
    }
}

impl Hittable for BVH {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        BVH::hit(self, ray, t_min, t_max)
    }

    fn bounding_box(&self) -> Option<AABB> {
        self.root.as_ref().map(|n| n.bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;
    use crate::math::Vec3;

    #[test]
    fn test_bvh_empty() {
        let bvh = BVH::new(vec![]);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray, 0.001, f32::INFINITY).is_none());
    }

    #[test]
    fn test_bvh_single() {
        let sphere = Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, 0);
        let bvh = BVH::new(vec![Box::new(sphere)]);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray, 0.001, f32::INFINITY).is_some());
    }

    #[test]
    fn test_bvh_multiple() {
        let spheres: Vec<Box<dyn Hittable>> = vec![
            Box::new(Sphere::new(Vec3::new(-2.0, 0.0, -1.0), 0.5, 0)),
            Box::new(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, 1)),
            Box::new(Sphere::new(Vec3::new(2.0, 0.0, -1.0), 0.5, 2)),
        ];
        let bvh = BVH::new(spheres);
        
        // Hit middle sphere
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let hit = bvh.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().material_id, 1);
    }
}
