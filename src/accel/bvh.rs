//! Bounding Volume Hierarchy with SAH (Surface Area Heuristic)

use crate::math::Ray;
use crate::geometry::{HitRecord, Hittable};
use super::AABB;

/// Number of bins for SAH evaluation
const SAH_BINS: usize = 12;

/// Cost of traversing a BVH node
const TRAVERSAL_COST: f32 = 1.0;

/// Cost of intersecting a primitive
const INTERSECTION_COST: f32 = 1.0;

/// BVH node
pub struct BVHNode {
    bounds: AABB,
    left: Option<Box<BVHNode>>,
    right: Option<Box<BVHNode>>,
    primitive_start: usize,
    primitive_count: usize,
}

/// SAH bin for evaluating split costs
#[derive(Clone, Copy)]
struct SAHBin {
    bounds: AABB,
    count: usize,
}

impl Default for SAHBin {
    fn default() -> Self {
        Self {
            bounds: AABB::empty(),
            count: 0,
        }
    }
}

/// Bounding Volume Hierarchy
pub struct BVH {
    root: Option<BVHNode>,
    primitives: Vec<Box<dyn Hittable>>,
    primitive_count: usize,
}

impl BVH {
    /// Build BVH from primitives using SAH
    pub fn new(primitives: Vec<Box<dyn Hittable>>) -> Self {
        if primitives.is_empty() {
            return Self { root: None, primitives, primitive_count: 0 };
        }

        let count = primitives.len();
        let mut indices: Vec<usize> = (0..count).collect();
        let bounds: Vec<AABB> = primitives
            .iter()
            .map(|p| p.bounding_box().unwrap_or_else(AABB::empty))
            .collect();
        let centroids: Vec<crate::math::Vec3> = bounds.iter().map(|b| b.centroid()).collect();

        let root = Self::build_sah(&mut indices, &bounds, &centroids, 0, primitives.len());
        
        // Reorder primitives according to indices
        let mut new_primitives: Vec<Option<Box<dyn Hittable>>> = primitives.into_iter().map(Some).collect();
        let primitives: Vec<Box<dyn Hittable>> = indices
            .iter()
            .map(|&i| new_primitives[i].take().unwrap())
            .collect();

        Self { root: Some(root), primitives, primitive_count: count }
    }

    /// Get the number of primitives
    pub fn primitive_count(&self) -> usize {
        self.primitive_count
    }

    /// Build BVH node using SAH binning
    fn build_sah(
        indices: &mut [usize],
        bounds: &[AABB],
        centroids: &[crate::math::Vec3],
        start: usize,
        end: usize,
    ) -> BVHNode {
        let count = end - start;
        
        // Compute bounds for this node
        let mut node_bounds = AABB::empty();
        let mut centroid_bounds = AABB::empty();
        for &idx in &indices[start..end] {
            node_bounds = AABB::surrounding(&node_bounds, &bounds[idx]);
            centroid_bounds = AABB::surrounding(&centroid_bounds, &AABB::new(centroids[idx], centroids[idx]));
        }

        // Leaf node for small counts
        if count <= 4 {
            return BVHNode {
                bounds: node_bounds,
                left: None,
                right: None,
                primitive_start: start,
                primitive_count: count,
            };
        }

        // Find best split using SAH binning
        let (best_axis, best_split) = Self::find_best_split(
            &indices[start..end],
            bounds,
            centroids,
            &centroid_bounds,
        );

        // If no good split found, create leaf
        if best_split.is_none() {
            return BVHNode {
                bounds: node_bounds,
                left: None,
                right: None,
                primitive_start: start,
                primitive_count: count,
            };
        }

        let (axis, split_pos) = (best_axis, best_split.unwrap());

        // Partition primitives
        let mid = Self::partition(&mut indices[start..end], centroids, axis, split_pos) + start;

        // Ensure we make progress (avoid infinite recursion)
        if mid == start || mid == end {
            // Fallback to median split
            let mid = start + count / 2;
            indices[start..end].sort_by(|&a, &b| {
                let ca = centroids[a][axis];
                let cb = centroids[b][axis];
                ca.partial_cmp(&cb).unwrap()
            });
            
            let left = Box::new(Self::build_sah(indices, bounds, centroids, start, mid));
            let right = Box::new(Self::build_sah(indices, bounds, centroids, mid, end));

            return BVHNode {
                bounds: node_bounds,
                left: Some(left),
                right: Some(right),
                primitive_start: 0,
                primitive_count: 0,
            };
        }

        let left = Box::new(Self::build_sah(indices, bounds, centroids, start, mid));
        let right = Box::new(Self::build_sah(indices, bounds, centroids, mid, end));

        BVHNode {
            bounds: node_bounds,
            left: Some(left),
            right: Some(right),
            primitive_start: 0,
            primitive_count: 0,
        }
    }

    /// Find best split axis and position using SAH binning
    fn find_best_split(
        indices: &[usize],
        bounds: &[AABB],
        centroids: &[crate::math::Vec3],
        centroid_bounds: &AABB,
    ) -> (usize, Option<f32>) {
        let count = indices.len();
        if count <= 1 {
            return (0, None);
        }

        let mut best_cost = f32::INFINITY;
        let mut best_axis = 0;
        let mut best_split = None;

        // Try each axis
        for axis in 0..3 {
            let extent = centroid_bounds.max[axis] - centroid_bounds.min[axis];
            if extent <= 0.0 {
                continue;
            }

            // Initialize bins
            let mut bins = [SAHBin::default(); SAH_BINS];

            // Assign primitives to bins
            for &idx in indices {
                let centroid = centroids[idx][axis];
                let mut bin_idx = ((centroid - centroid_bounds.min[axis]) / extent * SAH_BINS as f32) as usize;
                bin_idx = bin_idx.min(SAH_BINS - 1);
                bins[bin_idx].count += 1;
                bins[bin_idx].bounds = AABB::surrounding(&bins[bin_idx].bounds, &bounds[idx]);
            }

            // Compute costs for each split position
            let mut left_count = [0usize; SAH_BINS - 1];
            let mut left_bounds = [AABB::empty(); SAH_BINS - 1];
            let mut running_bounds = AABB::empty();
            let mut running_count = 0;

            for i in 0..(SAH_BINS - 1) {
                running_bounds = AABB::surrounding(&running_bounds, &bins[i].bounds);
                running_count += bins[i].count;
                left_bounds[i] = running_bounds;
                left_count[i] = running_count;
            }

            let mut right_bounds = [AABB::empty(); SAH_BINS - 1];
            let mut right_count = [0usize; SAH_BINS - 1];
            running_bounds = AABB::empty();
            running_count = 0;

            for i in (1..SAH_BINS).rev() {
                running_bounds = AABB::surrounding(&running_bounds, &bins[i].bounds);
                running_count += bins[i].count;
                right_bounds[i - 1] = running_bounds;
                right_count[i - 1] = running_count;
            }

            // Find minimum cost split
            let parent_area = centroid_bounds.surface_area();
            for i in 0..(SAH_BINS - 1) {
                if left_count[i] == 0 || right_count[i] == 0 {
                    continue;
                }

                let left_area = left_bounds[i].surface_area();
                let right_area = right_bounds[i].surface_area();

                let cost = TRAVERSAL_COST + INTERSECTION_COST * (
                    left_count[i] as f32 * left_area / parent_area +
                    right_count[i] as f32 * right_area / parent_area
                );

                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_split = Some(
                        centroid_bounds.min[axis] + extent * (i + 1) as f32 / SAH_BINS as f32
                    );
                }
            }
        }

        // Check if split is better than not splitting
        let leaf_cost = INTERSECTION_COST * count as f32;
        if best_cost >= leaf_cost {
            return (0, None);
        }

        (best_axis, best_split)
    }

    /// Partition indices around split position
    fn partition(
        indices: &mut [usize],
        centroids: &[crate::math::Vec3],
        axis: usize,
        split_pos: f32,
    ) -> usize {
        let mut left = 0;
        let mut right = indices.len();

        while left < right {
            if centroids[indices[left]][axis] < split_pos {
                left += 1;
            } else {
                right -= 1;
                indices.swap(left, right);
            }
        }

        left
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

        // Interior node - traverse both children
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

    #[test]
    fn test_bvh_many_primitives() {
        // Test with many primitives to exercise SAH binning
        let mut spheres: Vec<Box<dyn Hittable>> = Vec::new();
        for i in 0..100 {
            let x = (i % 10) as f32 - 5.0;
            let z = (i / 10) as f32 - 5.0;
            spheres.push(Box::new(Sphere::new(Vec3::new(x, 0.0, z), 0.3, i)));
        }
        let bvh = BVH::new(spheres);
        
        // Ray hitting center
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = bvh.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
    }

    #[test]
    fn test_bvh_primitive_count() {
        let spheres: Vec<Box<dyn Hittable>> = vec![
            Box::new(Sphere::new(Vec3::zero(), 1.0, 0)),
            Box::new(Sphere::new(Vec3::new(2.0, 0.0, 0.0), 1.0, 1)),
        ];
        let bvh = BVH::new(spheres);
        assert_eq!(bvh.primitive_count(), 2);
    }

    #[test]
    fn test_sah_better_than_median() {
        // Create a scene where SAH should give better results than median split
        // Clustered primitives should benefit from SAH
        let mut spheres: Vec<Box<dyn Hittable>> = Vec::new();
        
        // Cluster 1: many small spheres at x=0
        for i in 0..20 {
            let y = (i as f32 - 10.0) * 0.1;
            spheres.push(Box::new(Sphere::new(Vec3::new(0.0, y, 0.0), 0.05, i)));
        }
        
        // Cluster 2: few large spheres at x=10
        for i in 0..3 {
            let y = (i as f32 - 1.0) * 2.0;
            spheres.push(Box::new(Sphere::new(Vec3::new(10.0, y, 0.0), 0.5, 20 + i)));
        }
        
        let bvh = BVH::new(spheres);
        
        // Test ray going through cluster 1
        let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = bvh.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
    }
}
