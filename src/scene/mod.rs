//! Scene representation

mod loader;

pub use loader::{load_scene, RenderSettings};

use crate::math::{Vec3, Ray};
use crate::geometry::{Sphere, Triangle, Mesh, Instance, HitRecord, Hittable};
use crate::material::Material;
use crate::accel::BVH;

/// Light geometry type
#[derive(Clone)]
pub enum LightGeometry {
    Sphere { center: Vec3, radius: f32 },
    Triangle { v0: Vec3, v1: Vec3, v2: Vec3, normal: Vec3 },
}

/// Light info for NEE
#[derive(Clone)]
pub struct LightInfo {
    pub material_id: usize,
    pub geometry: LightGeometry,
}

impl LightInfo {
    /// Create a sphere light
    pub fn sphere(center: Vec3, radius: f32, material_id: usize) -> Self {
        Self {
            material_id,
            geometry: LightGeometry::Sphere { center, radius },
        }
    }

    /// Create a triangle light
    pub fn triangle(v0: Vec3, v1: Vec3, v2: Vec3, material_id: usize) -> Self {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(&edge2).normalize();
        Self {
            material_id,
            geometry: LightGeometry::Triangle { v0, v1, v2, normal },
        }
    }

    /// Get the surface area of this light
    pub fn area(&self) -> f32 {
        match &self.geometry {
            LightGeometry::Sphere { radius, .. } => {
                4.0 * std::f32::consts::PI * radius * radius
            }
            LightGeometry::Triangle { v0, v1, v2, .. } => {
                let edge1 = *v1 - *v0;
                let edge2 = *v2 - *v0;
                edge1.cross(&edge2).length() * 0.5
            }
        }
    }

    /// Sample a point on this light
    pub fn sample(&self, u1: f32, u2: f32) -> (Vec3, Vec3) {
        match &self.geometry {
            LightGeometry::Sphere { center, radius } => {
                // Sample point on sphere surface
                let theta = 2.0 * std::f32::consts::PI * u1;
                let phi = (1.0 - 2.0 * u2).acos();
                let x = phi.sin() * theta.cos();
                let y = phi.sin() * theta.sin();
                let z = phi.cos();
                let normal = Vec3::new(x, y, z);
                let point = *center + normal * *radius;
                (point, normal)
            }
            LightGeometry::Triangle { v0, v1, v2, normal } => {
                // Sample point on triangle using barycentric coordinates
                let su0 = u1.sqrt();
                let b0 = 1.0 - su0;
                let b1 = u2 * su0;
                let point = *v0 * b0 + *v1 * b1 + *v2 * (1.0 - b0 - b1);
                (point, *normal)
            }
        }
    }

    /// Get the center/centroid of this light (for distance calculations)
    pub fn centroid(&self) -> Vec3 {
        match &self.geometry {
            LightGeometry::Sphere { center, .. } => *center,
            LightGeometry::Triangle { v0, v1, v2, .. } => (*v0 + *v1 + *v2) / 3.0,
        }
    }
}

/// Primitive type for GPU upload
#[derive(Clone, Copy, Debug)]
pub enum PrimitiveType {
    Sphere,
    Triangle,
    Mesh,
    Instance,
}

/// Primitive reference for GPU upload
#[derive(Clone, Debug)]
pub struct PrimitiveRef {
    pub prim_type: PrimitiveType,
    pub index: usize,
    pub material_id: usize,
}

/// Scene containing all objects and materials
pub struct Scene {
    spheres: Vec<Sphere>,
    triangles: Vec<Triangle>,
    meshes: Vec<Mesh>,
    instances: Vec<Instance>,
    materials: Vec<Box<dyn Material>>,
    bvh: Option<BVH>,
    emissive_lights: Vec<LightInfo>,
    /// Track original primitives for GPU upload
    primitive_refs: Vec<PrimitiveRef>,
    /// Copies of spheres for GPU upload (preserved after BVH build)
    gpu_spheres: Vec<Sphere>,
    /// Copies of triangles for GPU upload (preserved after BVH build)
    gpu_triangles: Vec<Triangle>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            spheres: Vec::new(),
            triangles: Vec::new(),
            meshes: Vec::new(),
            instances: Vec::new(),
            materials: Vec::new(),
            bvh: None,
            emissive_lights: Vec::new(),
            primitive_refs: Vec::new(),
            gpu_spheres: Vec::new(),
            gpu_triangles: Vec::new(),
        }
    }

    /// Get reference to BVH (for packet tracing)
    pub fn bvh(&self) -> Option<&BVH> {
        self.bvh.as_ref()
    }

    pub fn add_material(&mut self, mat: Box<dyn Material>) -> usize {
        let id = self.materials.len();
        self.materials.push(mat);
        id
    }

    pub fn add_sphere(&mut self, sphere: Sphere) {
        // Check if material is emissive
        if let Some(mat) = self.materials.get(sphere.material_id) {
            if mat.emit().length_squared() > 0.0 {
                self.emissive_lights.push(LightInfo::sphere(
                    sphere.center,
                    sphere.radius,
                    sphere.material_id,
                ));
            }
        }
        self.primitive_refs.push(PrimitiveRef {
            prim_type: PrimitiveType::Sphere,
            index: self.spheres.len(),
            material_id: sphere.material_id,
        });
        self.spheres.push(sphere);
    }

    pub fn add_triangle(&mut self, triangle: Triangle) {
        // Check if material is emissive
        if let Some(mat) = self.materials.get(triangle.material_id) {
            if mat.emit().length_squared() > 0.0 {
                self.emissive_lights.push(LightInfo::triangle(
                    triangle.v0,
                    triangle.v1,
                    triangle.v2,
                    triangle.material_id,
                ));
            }
        }
        self.primitive_refs.push(PrimitiveRef {
            prim_type: PrimitiveType::Triangle,
            index: self.triangles.len(),
            material_id: triangle.material_id,
        });
        self.triangles.push(triangle);
    }

    pub fn add_mesh(&mut self, mesh: Mesh) {
        let mat_id = mesh.material_id();
        
        // Check if material is emissive and register triangle lights
        if let Some(mat) = self.materials.get(mat_id) {
            if mat.emit().length_squared() > 0.0 {
                // Register each triangle as a light source
                for tri in mesh.triangles() {
                    self.emissive_lights.push(LightInfo::triangle(
                        tri.v0, tri.v1, tri.v2, mat_id,
                    ));
                }
            }
        }
        
        self.primitive_refs.push(PrimitiveRef {
            prim_type: PrimitiveType::Mesh,
            index: self.meshes.len(),
            material_id: mat_id,
        });
        self.meshes.push(mesh);
    }

    pub fn add_instance(&mut self, instance: Instance) {
        // Note: Instances are added directly to BVH, we don't track their lights
        // because the underlying geometry handles that
        self.primitive_refs.push(PrimitiveRef {
            prim_type: PrimitiveType::Instance,
            index: self.instances.len(),
            material_id: 0, // Instance may override material
        });
        self.instances.push(instance);
    }

    pub fn material(&self, id: usize) -> Option<&dyn Material> {
        self.materials.get(id).map(|m| m.as_ref())
    }

    /// Get all spheres for GPU upload
    pub fn spheres(&self) -> &[Sphere] {
        &self.gpu_spheres
    }

    /// Get all triangles for GPU upload (flattened from all sources)
    pub fn triangles(&self) -> &[Triangle] {
        &self.gpu_triangles
    }

    /// Get primitive references
    pub fn primitive_refs(&self) -> &[PrimitiveRef] {
        &self.primitive_refs
    }

    pub fn build_bvh(&mut self) {
        // Preserve copies for GPU upload before draining
        self.gpu_spheres = self.spheres.clone();
        
        // Flatten all triangles including from meshes for GPU upload
        self.gpu_triangles = self.triangles.clone();
        for mesh in &self.meshes {
            self.gpu_triangles.extend(mesh.triangles().iter().cloned());
        }

        let mut primitives: Vec<Box<dyn Hittable>> = Vec::new();
        
        for sphere in self.spheres.drain(..) {
            primitives.push(Box::new(sphere));
        }
        
        for triangle in self.triangles.drain(..) {
            primitives.push(Box::new(triangle));
        }
        
        for mut mesh in self.meshes.drain(..) {
            mesh.build_bvh();
            primitives.push(Box::new(mesh));
        }

        for instance in self.instances.drain(..) {
            primitives.push(Box::new(instance));
        }
        
        self.bvh = Some(BVH::new(primitives));
    }

    pub fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if let Some(bvh) = &self.bvh {
            return bvh.hit(ray, t_min, t_max);
        }

        // Fallback: linear search
        let mut closest: Option<HitRecord> = None;
        let mut closest_t = t_max;

        for sphere in &self.spheres {
            if let Some(rec) = sphere.hit(ray, t_min, closest_t) {
                closest_t = rec.t;
                closest = Some(rec);
            }
        }

        for triangle in &self.triangles {
            if let Some(rec) = triangle.hit(ray, t_min, closest_t) {
                closest_t = rec.t;
                closest = Some(rec);
            }
        }

        for mesh in &self.meshes {
            if let Some(rec) = mesh.hit(ray, t_min, closest_t) {
                closest_t = rec.t;
                closest = Some(rec);
            }
        }

        for instance in &self.instances {
            if let Some(rec) = instance.hit(ray, t_min, closest_t) {
                closest_t = rec.t;
                closest = Some(rec);
            }
        }

        closest
    }

    /// Test for any hit (shadow rays) - early exit
    pub fn hit_any(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        if let Some(bvh) = &self.bvh {
            return bvh.hit_any(ray, t_min, t_max);
        }

        // Fallback: linear search with early exit
        for sphere in &self.spheres {
            if sphere.hit(ray, t_min, t_max).is_some() {
                return true;
            }
        }

        for triangle in &self.triangles {
            if triangle.hit(ray, t_min, t_max).is_some() {
                return true;
            }
        }

        for mesh in &self.meshes {
            if mesh.hit(ray, t_min, t_max).is_some() {
                return true;
            }
        }

        for instance in &self.instances {
            if instance.hit(ray, t_min, t_max).is_some() {
                return true;
            }
        }

        false
    }

    pub fn emissive_objects(&self) -> &[LightInfo] {
        &self.emissive_lights
    }

    /// Sample a point on a light source
    pub fn sample_light(&self, light: &LightInfo, u1: f32, u2: f32) -> (Vec3, Vec3, Vec3) {
        let (point, normal) = light.sample(u1, u2);
        
        let emission = self.materials.get(light.material_id)
            .map(|m| m.emit())
            .unwrap_or(Vec3::zero());
        
        (point, normal, emission)
    }

    /// Get material count
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Lambertian;
    use crate::math::Transform;

    #[test]
    fn test_scene_new() {
        let scene = Scene::new();
        assert_eq!(scene.emissive_objects().len(), 0);
    }

    #[test]
    fn test_scene_add_material() {
        let mut scene = Scene::new();
        let id = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5))));
        assert_eq!(id, 0);
    }

    #[test]
    fn test_scene_hit() {
        let mut scene = Scene::new();
        let mat_id = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5))));
        scene.add_sphere(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, mat_id));
        scene.build_bvh();
        
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let hit = scene.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
    }

    #[test]
    fn test_scene_hit_any() {
        let mut scene = Scene::new();
        let mat_id = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5))));
        scene.add_sphere(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, mat_id));
        scene.build_bvh();
        
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        assert!(scene.hit_any(&ray, 0.001, f32::INFINITY));
        
        let miss_ray = Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(!scene.hit_any(&miss_ray, 0.001, f32::INFINITY));
    }

    #[test]
    fn test_scene_spheres_preserved() {
        let mut scene = Scene::new();
        let mat_id = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5))));
        scene.add_sphere(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, mat_id));
        scene.add_sphere(Sphere::new(Vec3::new(2.0, 0.0, -1.0), 0.5, mat_id));
        scene.build_bvh();
        
        // Spheres should be preserved for GPU upload
        assert_eq!(scene.spheres().len(), 2);
    }

    #[test]
    fn test_scene_with_instance() {
        let mut scene = Scene::new();
        let mat_id = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5))));
        
        let sphere = Sphere::new(Vec3::zero(), 0.5, mat_id);
        let transform = Transform::translate(Vec3::new(0.0, 0.0, -2.0));
        let instance = Instance::new(Box::new(sphere), transform);
        scene.add_instance(instance);
        scene.build_bvh();
        
        // Ray should hit the transformed instance
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let hit = scene.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
    }
}
