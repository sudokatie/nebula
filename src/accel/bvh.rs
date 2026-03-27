//! Bounding Volume Hierarchy with SAH

use crate::math::Ray;
use crate::geometry::{HitRecord, Hittable};
use super::AABB;

/// BVH node (flattened for cache efficiency)
#[derive(Clone)]
pub struct BVHNode {
    /// Bounding box of this node
    pub bounds: AABB,
    /// For interior: index of right child (left is always index + 1)
    /// For leaf: offset into primitives array
    pub offset: u32,
    /// Number of primitives (0 for interior nodes)
    pub count: u16,
    /// Split axis for interior nodes (0=X, 1=Y, 2=Z)
    pub axis: u8,
    _pad: u8,
}

/// Bounding Volume Hierarchy with SAH construction
pub struct BVH {
    pub(crate) nodes: Vec<BVHNode>,
    pub(crate) primitives: Vec<Box<dyn Hittable>>,
}

/// SAH cost constants
const TRAVERSAL_COST: f32 = 1.0;
const INTERSECTION_COST: f32 = 1.0;
const MAX_PRIMS_IN_LEAF: usize = 4;
const SAH_BUCKETS: usize = 12;

/// Primitive info for building
struct PrimInfo {
    index: usize,
    bounds: AABB,
    centroid: [f32; 3],
}

impl BVH {
    /// Build BVH from primitives using SAH
    pub fn new(primitives: Vec<Box<dyn Hittable>>) -> Self {
        if primitives.is_empty() {
            return Self {
                nodes: vec![],
                primitives: vec![],
            };
        }

        // Collect primitive info
        let mut prim_info: Vec<PrimInfo> = primitives
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let bounds = p.bounding_box().unwrap_or_else(AABB::empty);
                let c = bounds.centroid();
                PrimInfo {
                    index: i,
                    bounds,
                    centroid: [c.x, c.y, c.z],
                }
            })
            .collect();

        let mut nodes = Vec::with_capacity(primitives.len() * 2);
        let mut ordered: Vec<usize> = Vec::with_capacity(primitives.len());

        Self::build_recursive(&mut prim_info, 0, primitives.len(), &mut nodes, &mut ordered);

        // Reorder primitives
        let mut new_primitives: Vec<Option<Box<dyn Hittable>>> = 
            primitives.into_iter().map(Some).collect();
        let primitives: Vec<Box<dyn Hittable>> = ordered
            .iter()
            .map(|&i| new_primitives[i].take().unwrap())
            .collect();

        Self { nodes, primitives }
    }

    fn build_recursive(
        prim_info: &mut [PrimInfo],
        start: usize,
        end: usize,
        nodes: &mut Vec<BVHNode>,
        ordered: &mut Vec<usize>,
    ) -> usize {
        let node_idx = nodes.len();
        nodes.push(BVHNode {
            bounds: AABB::empty(),
            offset: 0,
            count: 0,
            axis: 0,
            _pad: 0,
        });

        // Compute bounds of all primitives
        let mut bounds = AABB::empty();
        for info in &prim_info[start..end] {
            bounds = AABB::surrounding(&bounds, &info.bounds);
        }

        let n_prims = end - start;

        if n_prims <= MAX_PRIMS_IN_LEAF {
            // Create leaf
            let first_prim_offset = ordered.len() as u32;
            for info in &prim_info[start..end] {
                ordered.push(info.index);
            }
            nodes[node_idx] = BVHNode {
                bounds,
                offset: first_prim_offset,
                count: n_prims as u16,
                axis: 0,
                _pad: 0,
            };
            return node_idx;
        }

        // Compute centroid bounds
        let mut centroid_bounds = AABB::empty();
        for info in &prim_info[start..end] {
            centroid_bounds = centroid_bounds.extend_point(crate::math::Vec3::new(
                info.centroid[0],
                info.centroid[1],
                info.centroid[2],
            ));
        }

        let dim = centroid_bounds.longest_axis();

        // Degenerate case
        if centroid_bounds.max()[dim] == centroid_bounds.min()[dim] {
            let first_prim_offset = ordered.len() as u32;
            for info in &prim_info[start..end] {
                ordered.push(info.index);
            }
            nodes[node_idx] = BVHNode {
                bounds,
                offset: first_prim_offset,
                count: n_prims as u16,
                axis: dim as u8,
                _pad: 0,
            };
            return node_idx;
        }

        // SAH bucketing
        let mid = Self::sah_partition(prim_info, start, end, dim, &centroid_bounds, &bounds);

        // Build children
        let _left_idx = Self::build_recursive(prim_info, start, mid, nodes, ordered);
        let right_idx = Self::build_recursive(prim_info, mid, end, nodes, ordered);

        nodes[node_idx] = BVHNode {
            bounds,
            offset: right_idx as u32,
            count: 0,
            axis: dim as u8,
            _pad: 0,
        };

        node_idx
    }

    fn sah_partition(
        prim_info: &mut [PrimInfo],
        start: usize,
        end: usize,
        dim: usize,
        centroid_bounds: &AABB,
        node_bounds: &AABB,
    ) -> usize {
        let n_prims = end - start;

        if n_prims <= 2 {
            // Simple median split
            let mid = start + n_prims / 2;
            prim_info[start..end].select_nth_unstable_by(mid - start, |a, b| {
                a.centroid[dim].partial_cmp(&b.centroid[dim]).unwrap()
            });
            return mid;
        }

        // Initialize buckets
        #[derive(Clone)]
        struct Bucket {
            count: usize,
            bounds: AABB,
        }
        let mut buckets = vec![
            Bucket {
                count: 0,
                bounds: AABB::empty(),
            };
            SAH_BUCKETS
        ];

        let extent = centroid_bounds.max()[dim] - centroid_bounds.min()[dim];

        // Fill buckets
        for info in &prim_info[start..end] {
            let offset = info.centroid[dim] - centroid_bounds.min()[dim];
            let mut b = (SAH_BUCKETS as f32 * offset / extent) as usize;
            if b >= SAH_BUCKETS {
                b = SAH_BUCKETS - 1;
            }
            buckets[b].count += 1;
            buckets[b].bounds = AABB::surrounding(&buckets[b].bounds, &info.bounds);
        }

        // Compute costs for splitting after each bucket
        let mut costs = [0.0f32; SAH_BUCKETS - 1];
        let inv_total_area = 1.0 / node_bounds.surface_area();

        for i in 0..SAH_BUCKETS - 1 {
            let mut left_bounds = AABB::empty();
            let mut left_count = 0;
            for j in 0..=i {
                left_bounds = AABB::surrounding(&left_bounds, &buckets[j].bounds);
                left_count += buckets[j].count;
            }

            let mut right_bounds = AABB::empty();
            let mut right_count = 0;
            for j in (i + 1)..SAH_BUCKETS {
                right_bounds = AABB::surrounding(&right_bounds, &buckets[j].bounds);
                right_count += buckets[j].count;
            }

            costs[i] = TRAVERSAL_COST
                + INTERSECTION_COST
                    * (left_count as f32 * left_bounds.surface_area()
                        + right_count as f32 * right_bounds.surface_area())
                    * inv_total_area;
        }

        // Find best split
        let mut min_cost = costs[0];
        let mut min_bucket = 0;
        for (i, &cost) in costs.iter().enumerate().skip(1) {
            if cost < min_cost {
                min_cost = cost;
                min_bucket = i;
            }
        }

        // Compare to leaf cost
        let leaf_cost = INTERSECTION_COST * n_prims as f32;
        if leaf_cost <= min_cost && n_prims <= MAX_PRIMS_IN_LEAF {
            // Better to make leaf
            return start;
        }

        // Partition using sort and binary search
        let pivot = centroid_bounds.min()[dim]
            + (min_bucket + 1) as f32 * extent / SAH_BUCKETS as f32;

        // Sort by centroid position
        prim_info[start..end].sort_by(|a, b| {
            a.centroid[dim].partial_cmp(&b.centroid[dim]).unwrap()
        });

        // Find partition point
        let mut mid = 0;
        for (i, info) in prim_info[start..end].iter().enumerate() {
            if info.centroid[dim] >= pivot {
                mid = i;
                break;
            }
            mid = i + 1;
        }

        if mid == 0 || mid == n_prims {
            // Fallback to median
            mid = n_prims / 2;
        }

        start + mid
    }

    /// Test ray against BVH
    pub fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if self.nodes.is_empty() {
            return None;
        }

        // Precompute inverse direction
        let inv_dir = crate::math::Vec3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );
        let dir_is_neg = [
            ray.direction.x < 0.0,
            ray.direction.y < 0.0,
            ray.direction.z < 0.0,
        ];

        let mut closest: Option<HitRecord> = None;
        let mut closest_t = t_max;

        // Stack-based traversal
        let mut stack = [0u32; 64];
        let mut stack_ptr = 0;
        let mut current = 0;

        loop {
            let node = &self.nodes[current];

            if node.bounds.hit_precomputed(&ray.origin, &inv_dir, t_min, closest_t) {
                if node.count > 0 {
                    // Leaf node
                    for i in 0..node.count as usize {
                        let prim_idx = node.offset as usize + i;
                        if let Some(rec) = self.primitives[prim_idx].hit(ray, t_min, closest_t) {
                            closest_t = rec.t;
                            closest = Some(rec);
                        }
                    }

                    if stack_ptr == 0 {
                        break;
                    }
                    stack_ptr -= 1;
                    current = stack[stack_ptr] as usize;
                } else {
                    // Interior node - visit children
                    let left = current + 1;
                    let right = node.offset as usize;

                    // Visit closer child first
                    if dir_is_neg[node.axis as usize] {
                        stack[stack_ptr] = left as u32;
                        stack_ptr += 1;
                        current = right;
                    } else {
                        stack[stack_ptr] = right as u32;
                        stack_ptr += 1;
                        current = left;
                    }
                }
            } else {
                if stack_ptr == 0 {
                    break;
                }
                stack_ptr -= 1;
                current = stack[stack_ptr] as usize;
            }
        }

        closest
    }

    pub fn primitive_count(&self) -> usize {
        self.primitives.len()
    }

    /// Get reference to nodes for GPU upload
    pub fn nodes(&self) -> &[BVHNode] {
        &self.nodes
    }

    /// Get reference to primitives for GPU upload
    pub fn primitives(&self) -> &[Box<dyn Hittable>] {
        &self.primitives
    }

    /// Stackless BVH traversal using rope-based skip pointers
    /// This is an alternative to stack-based traversal that avoids stack memory
    pub fn hit_stackless(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if self.nodes.is_empty() {
            return None;
        }

        // Precompute inverse direction
        let inv_dir = crate::math::Vec3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );

        let mut closest: Option<HitRecord> = None;
        let mut closest_t = t_max;
        let mut current = 0usize;
        let num_nodes = self.nodes.len();

        // Stackless traversal using implicit skip pointers
        // In a flattened BVH, left child is always at index+1
        // We can compute the "exit" node by walking up the implicit tree
        while current < num_nodes {
            let node = &self.nodes[current];

            if node.bounds.hit_precomputed(&ray.origin, &inv_dir, t_min, closest_t) {
                if node.count > 0 {
                    // Leaf node - test primitives
                    for i in 0..node.count as usize {
                        let prim_idx = node.offset as usize + i;
                        if let Some(rec) = self.primitives[prim_idx].hit(ray, t_min, closest_t) {
                            closest_t = rec.t;
                            closest = Some(rec);
                        }
                    }
                    // Move to next node in linear order
                    current += 1;
                } else {
                    // Interior node - descend to left child (index + 1)
                    current += 1;
                }
            } else {
                // Missed this subtree - skip to right sibling or parent's sibling
                if node.count > 0 {
                    // Leaf - just move to next
                    current += 1;
                } else {
                    // Interior - skip to right child
                    current = node.offset as usize;
                }
            }
        }

        closest
    }

    /// Test ray for any hit (shadow rays) - early exit on first hit
    pub fn hit_any(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let inv_dir = crate::math::Vec3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );
        let dir_is_neg = [
            ray.direction.x < 0.0,
            ray.direction.y < 0.0,
            ray.direction.z < 0.0,
        ];

        let mut stack = [0u32; 64];
        let mut stack_ptr = 0;
        let mut current = 0;

        loop {
            let node = &self.nodes[current];

            if node.bounds.hit_precomputed(&ray.origin, &inv_dir, t_min, t_max) {
                if node.count > 0 {
                    // Leaf node - test primitives
                    for i in 0..node.count as usize {
                        let prim_idx = node.offset as usize + i;
                        if self.primitives[prim_idx].hit(ray, t_min, t_max).is_some() {
                            return true; // Early exit
                        }
                    }

                    if stack_ptr == 0 {
                        break;
                    }
                    stack_ptr -= 1;
                    current = stack[stack_ptr] as usize;
                } else {
                    let left = current + 1;
                    let right = node.offset as usize;

                    if dir_is_neg[node.axis as usize] {
                        stack[stack_ptr] = left as u32;
                        stack_ptr += 1;
                        current = right;
                    } else {
                        stack[stack_ptr] = right as u32;
                        stack_ptr += 1;
                        current = left;
                    }
                }
            } else {
                if stack_ptr == 0 {
                    break;
                }
                stack_ptr -= 1;
                current = stack[stack_ptr] as usize;
            }
        }

        false
    }
}

impl Hittable for BVH {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        BVH::hit(self, ray, t_min, t_max)
    }

    fn bounding_box(&self) -> Option<AABB> {
        if self.nodes.is_empty() {
            None
        } else {
            Some(self.nodes[0].bounds)
        }
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

        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let hit = bvh.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().material_id, 1);
    }

    #[test]
    fn test_bvh_many_prims() {
        let spheres: Vec<Box<dyn Hittable>> = (0..100)
            .map(|i| {
                let x = (i % 10) as f32 - 5.0;
                let z = (i / 10) as f32 - 5.0;
                Box::new(Sphere::new(Vec3::new(x, 0.0, z), 0.3, i)) as Box<dyn Hittable>
            })
            .collect();
        let bvh = BVH::new(spheres);

        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = bvh.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
    }

    #[test]
    fn test_bvh_miss() {
        let sphere = Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, 0);
        let bvh = BVH::new(vec![Box::new(sphere)]);
        
        let ray = Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray, 0.001, f32::INFINITY).is_none());
    }
}
