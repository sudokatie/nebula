//! Ray packet tracing (4 rays at once) with SIMD acceleration

use crate::math::{Vec3, Vec3x4, Ray};
use crate::geometry::HitRecord;
use super::BVH;

/// A packet of 4 rays for SIMD tracing
#[derive(Clone)]
pub struct RayPacket {
    pub origins: Vec3x4,
    pub directions: Vec3x4,
    pub inv_directions: Vec3x4,
    pub active: [bool; 4],
    pub t_min: [f32; 4],
    pub t_max: [f32; 4],
}

impl RayPacket {
    pub fn new(rays: [Ray; 4]) -> Self {
        let origins = Vec3x4::new(
            rays[0].origin,
            rays[1].origin,
            rays[2].origin,
            rays[3].origin,
        );
        let directions = Vec3x4::new(
            rays[0].direction,
            rays[1].direction,
            rays[2].direction,
            rays[3].direction,
        );
        let inv_directions = Vec3x4::new(
            Vec3::new(1.0 / rays[0].direction.x, 1.0 / rays[0].direction.y, 1.0 / rays[0].direction.z),
            Vec3::new(1.0 / rays[1].direction.x, 1.0 / rays[1].direction.y, 1.0 / rays[1].direction.z),
            Vec3::new(1.0 / rays[2].direction.x, 1.0 / rays[2].direction.y, 1.0 / rays[2].direction.z),
            Vec3::new(1.0 / rays[3].direction.x, 1.0 / rays[3].direction.y, 1.0 / rays[3].direction.z),
        );
        Self {
            origins,
            directions,
            inv_directions,
            active: [true; 4],
            t_min: [rays[0].t_min, rays[1].t_min, rays[2].t_min, rays[3].t_min],
            t_max: [rays[0].t_max, rays[1].t_max, rays[2].t_max, rays[3].t_max],
        }
    }

    pub fn from_single(ray: Ray) -> Self {
        Self::new([ray, ray, ray, ray])
    }

    pub fn ray(&self, i: usize) -> Ray {
        Ray::with_bounds(
            self.origins.get(i),
            self.directions.get(i),
            self.t_min[i],
            self.t_max[i],
        )
    }

    /// Get origins as array for AABB testing
    pub fn origins_array(&self) -> [Vec3; 4] {
        [
            self.origins.get(0),
            self.origins.get(1),
            self.origins.get(2),
            self.origins.get(3),
        ]
    }

    /// Get inverse directions as array for AABB testing
    pub fn inv_dirs_array(&self) -> [Vec3; 4] {
        [
            self.inv_directions.get(0),
            self.inv_directions.get(1),
            self.inv_directions.get(2),
            self.inv_directions.get(3),
        ]
    }
}

/// Hit results for a ray packet
#[derive(Clone)]
pub struct HitPacket {
    pub hits: [Option<HitRecord>; 4],
    pub t_max: [f32; 4],
}

impl HitPacket {
    pub fn new() -> Self {
        Self {
            hits: [None, None, None, None],
            t_max: [f32::INFINITY; 4],
        }
    }

    pub fn with_t_max(t_max: [f32; 4]) -> Self {
        Self {
            hits: [None, None, None, None],
            t_max,
        }
    }

    /// Count number of valid hits
    pub fn hit_count(&self) -> usize {
        self.hits.iter().filter(|h| h.is_some()).count()
    }

    /// Check if all rays hit something
    pub fn all_hit(&self) -> bool {
        self.hits.iter().all(|h| h.is_some())
    }

    /// Check if any ray hit something
    pub fn any_hit(&self) -> bool {
        self.hits.iter().any(|h| h.is_some())
    }
}

impl Default for HitPacket {
    fn default() -> Self {
        Self::new()
    }
}

impl BVH {
    /// Trace a packet of 4 rays through the BVH
    pub fn hit_packet(&self, packet: &RayPacket, t_min: f32) -> HitPacket {
        let mut result = HitPacket::with_t_max(packet.t_max);
        
        if self.nodes.is_empty() {
            return result;
        }

        let origins = packet.origins_array();
        let inv_dirs = packet.inv_dirs_array();

        // Stack-based traversal
        let mut stack = [0u32; 64];
        let mut stack_ptr = 0;
        let mut current = 0;

        loop {
            let node = &self.nodes[current];

            // Test all 4 rays against this node's AABB
            let hits = node.bounds.hit_simd4(
                &origins,
                &inv_dirs,
                [t_min; 4],
                result.t_max,
            );

            // Check if any active ray hits this node
            let any_hit = (hits[0] && packet.active[0])
                || (hits[1] && packet.active[1])
                || (hits[2] && packet.active[2])
                || (hits[3] && packet.active[3]);

            if any_hit {
                if node.count > 0 {
                    // Leaf node - test primitives
                    for i in 0..node.count as usize {
                        let prim_idx = node.offset as usize + i;
                        
                        // Test each active ray
                        for r in 0..4 {
                            if !packet.active[r] || !hits[r] {
                                continue;
                            }
                            
                            let ray = packet.ray(r);
                            if let Some(rec) = self.primitives[prim_idx].hit(&ray, t_min, result.t_max[r]) {
                                result.t_max[r] = rec.t;
                                result.hits[r] = Some(rec);
                            }
                        }
                    }

                    if stack_ptr == 0 {
                        break;
                    }
                    stack_ptr -= 1;
                    current = stack[stack_ptr] as usize;
                } else {
                    // Interior node
                    let left = current + 1;
                    let right = node.offset as usize;

                    // Push both children (could optimize order based on ray direction)
                    stack[stack_ptr] = right as u32;
                    stack_ptr += 1;
                    current = left;
                }
            } else {
                if stack_ptr == 0 {
                    break;
                }
                stack_ptr -= 1;
                current = stack[stack_ptr] as usize;
            }
        }

        result
    }

    /// Trace multiple ray packets and collect results
    pub fn hit_packets(&self, packets: &[RayPacket], t_min: f32) -> Vec<HitPacket> {
        packets.iter().map(|p| self.hit_packet(p, t_min)).collect()
    }

    /// Trace packets in parallel
    pub fn hit_packets_parallel(&self, packets: &[RayPacket], t_min: f32) -> Vec<HitPacket> {
        use rayon::prelude::*;
        packets.par_iter().map(|p| self.hit_packet(p, t_min)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;

    #[test]
    fn test_ray_packet_new() {
        let rays = [
            Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0)),
            Ray::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
            Ray::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
            Ray::new(Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
        ];
        let packet = RayPacket::new(rays);
        assert!(packet.active[0]);
        assert!(packet.active[3]);
    }

    #[test]
    fn test_hit_packet() {
        let spheres: Vec<Box<dyn crate::geometry::Hittable>> = vec![
            Box::new(Sphere::new(Vec3::new(0.0, 0.0, -2.0), 0.5, 0)),
            Box::new(Sphere::new(Vec3::new(2.0, 0.0, -2.0), 0.5, 1)),
        ];
        let bvh = BVH::new(spheres);

        let rays = [
            Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0)),           // hits sphere 0
            Ray::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)), // hits sphere 1
            Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)), // miss
            Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0)),           // hits sphere 0
        ];
        let packet = RayPacket::new(rays);
        let result = bvh.hit_packet(&packet, 0.001);

        assert!(result.hits[0].is_some());
        assert!(result.hits[1].is_some());
        assert!(result.hits[2].is_none());
        assert!(result.hits[3].is_some());
        assert_eq!(result.hit_count(), 3);
    }

    #[test]
    fn test_hit_packet_counts() {
        let result = HitPacket::new();
        assert_eq!(result.hit_count(), 0);
        assert!(!result.any_hit());
        assert!(!result.all_hit());
    }
}
