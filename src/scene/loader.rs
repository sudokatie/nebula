//! Scene loading from JSON

use std::path::Path;
use std::io;

use crate::math::{Vec3, Transform};
use crate::geometry::{Sphere, Triangle, Instance, load_obj};
use crate::material::{
    Lambertian, Metal, Dielectric, Emissive, 
    Checker, NoiseTexture, ImageTexture,
};
use crate::camera::Camera;
use super::Scene;

use serde::Deserialize;
use std::sync::Arc;

/// Render settings from scene file
#[derive(Debug, Clone)]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u32,
    pub max_depth: u32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            samples_per_pixel: 100,
            max_depth: 50,
        }
    }
}

#[derive(Deserialize)]
struct SceneFile {
    camera: CameraConfig,
    #[serde(default)]
    settings: SettingsConfig,
    materials: Vec<MaterialConfig>,
    objects: Vec<ObjectConfig>,
}

#[derive(Deserialize)]
struct CameraConfig {
    position: [f32; 3],
    look_at: [f32; 3],
    #[serde(default = "default_up")]
    up: [f32; 3],
    #[serde(default = "default_fov")]
    fov: f32,
    #[serde(default)]
    aperture: f32,
    #[serde(default = "default_focus")]
    focus_distance: f32,
}

fn default_up() -> [f32; 3] { [0.0, 1.0, 0.0] }
fn default_fov() -> f32 { 60.0 }
fn default_focus() -> f32 { 1.0 }

#[derive(Deserialize, Default)]
struct SettingsConfig {
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default = "default_samples")]
    samples: u32,
    #[serde(default = "default_depth")]
    depth: u32,
}

fn default_width() -> u32 { 800 }
fn default_height() -> u32 { 600 }
fn default_samples() -> u32 { 100 }
fn default_depth() -> u32 { 50 }

#[derive(Deserialize, Default)]
struct TransformConfig {
    #[serde(default)]
    translate: Option<[f32; 3]>,
    #[serde(default)]
    scale: Option<[f32; 3]>,
    #[serde(default)]
    rotate_x: Option<f32>,
    #[serde(default)]
    rotate_y: Option<f32>,
    #[serde(default)]
    rotate_z: Option<f32>,
}

impl TransformConfig {
    fn to_transform(&self) -> Transform {
        let mut t = Transform::identity();
        
        // Apply transforms in order: scale -> rotate -> translate
        if let Some([sx, sy, sz]) = self.scale {
            t = t.then(&Transform::scale(Vec3::new(sx, sy, sz)));
        }
        if let Some(angle) = self.rotate_x {
            t = t.then(&Transform::rotate_x(angle.to_radians()));
        }
        if let Some(angle) = self.rotate_y {
            t = t.then(&Transform::rotate_y(angle.to_radians()));
        }
        if let Some(angle) = self.rotate_z {
            t = t.then(&Transform::rotate_z(angle.to_radians()));
        }
        if let Some([tx, ty, tz]) = self.translate {
            t = t.then(&Transform::translate(Vec3::new(tx, ty, tz)));
        }
        
        t
    }
    
    fn is_identity(&self) -> bool {
        self.translate.is_none() 
            && self.scale.is_none() 
            && self.rotate_x.is_none() 
            && self.rotate_y.is_none() 
            && self.rotate_z.is_none()
    }
}

#[derive(Deserialize)]
struct MaterialConfig {
    name: String,
    #[serde(rename = "type")]
    material_type: String,
    #[serde(default)]
    albedo: Option<[f32; 3]>,
    #[serde(default)]
    color: Option<[f32; 3]>,
    #[serde(default)]
    roughness: Option<f32>,
    #[serde(default)]
    ior: Option<f32>,
    #[serde(default)]
    strength: Option<f32>,
    // Checker texture
    #[serde(default)]
    checker: Option<[[f32; 3]; 2]>,
    #[serde(default)]
    scale: Option<f32>,
    // Image texture
    #[serde(default)]
    texture: Option<String>,
    // Noise texture
    #[serde(default)]
    noise: Option<f32>,
}

#[derive(Deserialize)]
struct ObjectConfig {
    #[serde(rename = "type")]
    object_type: String,
    #[serde(default)]
    center: Option<[f32; 3]>,
    #[serde(default)]
    radius: Option<f32>,
    #[serde(default)]
    vertices: Option<[[f32; 3]; 3]>,
    #[serde(default)]
    normals: Option<[[f32; 3]; 3]>,
    #[serde(default)]
    uvs: Option<[[f32; 2]; 3]>,
    #[serde(default)]
    quad_vertices: Option<[[f32; 3]; 4]>,
    #[serde(default)]
    mesh: Option<String>,
    material: String,
    #[serde(default)]
    transform: Option<TransformConfig>,
    // Instance specific
    #[serde(default)]
    geometry: Option<String>,
}

/// Load scene from JSON file
pub fn load_scene(path: &Path) -> io::Result<(Scene, Camera, RenderSettings)> {
    let content = std::fs::read_to_string(path)?;
    let file: SceneFile = serde_json::from_str(&content)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut scene = Scene::new();
    let scene_dir = path.parent().unwrap_or(Path::new("."));

    // Create material name -> id mapping
    let mut material_ids = std::collections::HashMap::new();
    for mat_config in &file.materials {
        let mat: Box<dyn crate::material::Material> = match mat_config.material_type.as_str() {
            "lambertian" => {
                // Check for different texture types
                if let Some([c1, c2]) = mat_config.checker {
                    let scale = mat_config.scale.unwrap_or(10.0);
                    Box::new(Lambertian::textured(Arc::new(Checker::colors(
                        Vec3::new(c1[0], c1[1], c1[2]),
                        Vec3::new(c2[0], c2[1], c2[2]),
                        scale,
                    ))))
                } else if let Some(noise_scale) = mat_config.noise {
                    Box::new(Lambertian::textured(Arc::new(NoiseTexture::new(noise_scale))))
                } else if let Some(tex_path) = &mat_config.texture {
                    let full_path = scene_dir.join(tex_path);
                    match ImageTexture::from_file(&full_path) {
                        Ok(tex) => Box::new(Lambertian::textured(Arc::new(tex))),
                        Err(e) => {
                            eprintln!("Warning: Failed to load texture {}: {}", tex_path, e);
                            let albedo = mat_config.albedo.unwrap_or([0.5, 0.5, 0.5]);
                            Box::new(Lambertian::new(Vec3::new(albedo[0], albedo[1], albedo[2])))
                        }
                    }
                } else {
                    let albedo = mat_config.albedo.unwrap_or([0.5, 0.5, 0.5]);
                    Box::new(Lambertian::new(Vec3::new(albedo[0], albedo[1], albedo[2])))
                }
            }
            "metal" => {
                let albedo = mat_config.albedo.unwrap_or([0.8, 0.8, 0.8]);
                let roughness = mat_config.roughness.unwrap_or(0.0);
                
                // Metal can also have textures
                if let Some(tex_path) = &mat_config.texture {
                    let full_path = scene_dir.join(tex_path);
                    match ImageTexture::from_file(&full_path) {
                        Ok(tex) => Box::new(Metal::textured(Arc::new(tex), roughness)),
                        Err(_) => Box::new(Metal::new(Vec3::new(albedo[0], albedo[1], albedo[2]), roughness)),
                    }
                } else {
                    Box::new(Metal::new(Vec3::new(albedo[0], albedo[1], albedo[2]), roughness))
                }
            }
            "dielectric" | "glass" => {
                let ior = mat_config.ior.unwrap_or(1.5);
                Box::new(Dielectric::new(ior))
            }
            "emissive" | "light" => {
                let color = mat_config.color.or(mat_config.albedo).unwrap_or([1.0, 1.0, 1.0]);
                let strength = mat_config.strength.unwrap_or(1.0);
                Box::new(Emissive::new(Vec3::new(color[0], color[1], color[2]), strength))
            }
            _ => {
                // Default to lambertian gray
                Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5)))
            }
        };
        let id = scene.add_material(mat);
        material_ids.insert(mat_config.name.clone(), id);
    }

    // Create objects
    for obj_config in &file.objects {
        let material_id = *material_ids.get(&obj_config.material).unwrap_or(&0);
        let transform = obj_config.transform.as_ref().map(|t| t.to_transform());
        let has_transform = obj_config.transform.as_ref().map(|t| !t.is_identity()).unwrap_or(false);
        
        match obj_config.object_type.as_str() {
            "sphere" => {
                let center = obj_config.center.unwrap_or([0.0, 0.0, 0.0]);
                let radius = obj_config.radius.unwrap_or(1.0);
                let sphere = Sphere::new(
                    Vec3::new(center[0], center[1], center[2]),
                    radius,
                    material_id,
                );
                
                if has_transform {
                    // Wrap in instance with transform
                    let instance = Instance::new(Box::new(sphere), transform.unwrap());
                    scene.add_instance(instance);
                } else {
                    scene.add_sphere(sphere);
                }
            }
            "triangle" => {
                if let Some(verts) = obj_config.vertices {
                    let v0 = Vec3::new(verts[0][0], verts[0][1], verts[0][2]);
                    let v1 = Vec3::new(verts[1][0], verts[1][1], verts[1][2]);
                    let v2 = Vec3::new(verts[2][0], verts[2][1], verts[2][2]);
                    
                    // Check for explicit normals
                    let triangle = if let Some(norms) = obj_config.normals {
                        let n0 = Vec3::new(norms[0][0], norms[0][1], norms[0][2]);
                        let n1 = Vec3::new(norms[1][0], norms[1][1], norms[1][2]);
                        let n2 = Vec3::new(norms[2][0], norms[2][1], norms[2][2]);
                        
                        // Check for explicit UVs
                        if let Some(uvs) = obj_config.uvs {
                            Triangle::with_normals_and_uvs(
                                v0, v1, v2,
                                n0, n1, n2,
                                (uvs[0][0], uvs[0][1]),
                                (uvs[1][0], uvs[1][1]),
                                (uvs[2][0], uvs[2][1]),
                                material_id,
                            )
                        } else {
                            Triangle::with_normals(v0, v1, v2, n0, n1, n2, material_id)
                        }
                    } else if let Some(uvs) = obj_config.uvs {
                        // UVs without explicit normals
                        let edge1 = v1 - v0;
                        let edge2 = v2 - v0;
                        let n = edge1.cross(&edge2).normalize();
                        Triangle::with_normals_and_uvs(
                            v0, v1, v2,
                            n, n, n,
                            (uvs[0][0], uvs[0][1]),
                            (uvs[1][0], uvs[1][1]),
                            (uvs[2][0], uvs[2][1]),
                            material_id,
                        )
                    } else {
                        Triangle::new(v0, v1, v2, material_id)
                    };
                    
                    if has_transform {
                        let instance = Instance::new(Box::new(triangle), transform.unwrap());
                        scene.add_instance(instance);
                    } else {
                        scene.add_triangle(triangle);
                    }
                }
            }
            "mesh" => {
                if let Some(mesh_path) = &obj_config.mesh {
                    let full_path = scene_dir.join(mesh_path);
                    match load_obj(&full_path, material_id) {
                        Ok(mesh) => {
                            if has_transform {
                                let instance = Instance::new(Box::new(mesh), transform.unwrap());
                                scene.add_instance(instance);
                            } else {
                                scene.add_mesh(mesh);
                            }
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to load mesh {}: {}", mesh_path, e);
                        }
                    }
                }
            }
            "quad" => {
                // Quad defined by 4 vertices (v0, v1, v2, v3) -> two triangles
                if let Some(verts) = obj_config.quad_vertices {
                    let v0 = Vec3::new(verts[0][0], verts[0][1], verts[0][2]);
                    let v1 = Vec3::new(verts[1][0], verts[1][1], verts[1][2]);
                    let v2 = Vec3::new(verts[2][0], verts[2][1], verts[2][2]);
                    let v3 = Vec3::new(verts[3][0], verts[3][1], verts[3][2]);
                    
                    // Compute quad normal
                    let edge1 = v1 - v0;
                    let edge2 = v3 - v0;
                    let n = edge1.cross(&edge2).normalize();
                    
                    // Create triangles with proper UVs for a quad
                    let tri1 = Triangle::with_normals_and_uvs(
                        v0, v1, v2,
                        n, n, n,
                        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
                        material_id,
                    );
                    let tri2 = Triangle::with_normals_and_uvs(
                        v0, v2, v3,
                        n, n, n,
                        (0.0, 0.0), (1.0, 1.0), (0.0, 1.0),
                        material_id,
                    );
                    
                    if has_transform {
                        let t = transform.unwrap();
                        scene.add_instance(Instance::new(Box::new(tri1), t.clone()));
                        scene.add_instance(Instance::new(Box::new(tri2), t));
                    } else {
                        scene.add_triangle(tri1);
                        scene.add_triangle(tri2);
                    }
                }
            }
            "instance" => {
                // Instance of existing geometry with transform
                if let (Some(geom_type), Some(transform_config)) = (&obj_config.geometry, &obj_config.transform) {
                    let t = transform_config.to_transform();
                    
                    match geom_type.as_str() {
                        "sphere" => {
                            let center = obj_config.center.unwrap_or([0.0, 0.0, 0.0]);
                            let radius = obj_config.radius.unwrap_or(1.0);
                            let sphere = Sphere::new(
                                Vec3::new(center[0], center[1], center[2]),
                                radius,
                                material_id,
                            );
                            let instance = Instance::new(Box::new(sphere), t)
                                .with_material(material_id);
                            scene.add_instance(instance);
                        }
                        "mesh" => {
                            if let Some(mesh_path) = &obj_config.mesh {
                                let full_path = scene_dir.join(mesh_path);
                                if let Ok(mesh) = load_obj(&full_path, material_id) {
                                    let instance = Instance::new(Box::new(mesh), t)
                                        .with_material(material_id);
                                    scene.add_instance(instance);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {} // Skip unknown types
        }
    }

    scene.build_bvh();

    // Create camera
    let cam = &file.camera;
    let aspect_ratio = file.settings.width as f32 / file.settings.height as f32;
    let camera = Camera::new(
        Vec3::new(cam.position[0], cam.position[1], cam.position[2]),
        Vec3::new(cam.look_at[0], cam.look_at[1], cam.look_at[2]),
        Vec3::new(cam.up[0], cam.up[1], cam.up[2]),
        cam.fov,
        aspect_ratio,
        cam.aperture,
        cam.focus_distance,
    );

    let settings = RenderSettings {
        width: file.settings.width,
        height: file.settings.height,
        samples_per_pixel: file.settings.samples,
        max_depth: file.settings.depth,
    };

    Ok((scene, camera, settings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_settings_default() {
        let settings = RenderSettings::default();
        assert_eq!(settings.width, 800);
        assert_eq!(settings.height, 600);
    }

    #[test]
    fn test_transform_config_identity() {
        let config = TransformConfig::default();
        assert!(config.is_identity());
    }

    #[test]
    fn test_transform_config_translate() {
        let config = TransformConfig {
            translate: Some([1.0, 2.0, 3.0]),
            ..Default::default()
        };
        assert!(!config.is_identity());
        let t = config.to_transform();
        let p = t.transform_point(&Vec3::zero());
        assert!((p.x - 1.0).abs() < 0.001);
        assert!((p.y - 2.0).abs() < 0.001);
        assert!((p.z - 3.0).abs() < 0.001);
    }
}
