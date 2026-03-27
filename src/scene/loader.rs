//! Scene loading from JSON

use std::path::Path;
use std::io;

use crate::math::Vec3;
use crate::geometry::{Sphere, Triangle, Quad};
use crate::material::{Lambertian, Metal, Dielectric, Emissive};
use crate::texture::{CheckerTexture, SolidColor};
use crate::camera::Camera;
use super::{Scene, load_obj};

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
    #[serde(default)]
    checker: Option<[[f32; 3]; 2]>,
    #[serde(default)]
    scale: Option<f32>,
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
    quad_vertices: Option<[[f32; 3]; 4]>,
    #[serde(default)]
    mesh: Option<String>,
    material: String,
}

/// Load scene from JSON file
pub fn load_scene(path: &Path) -> io::Result<(Scene, Camera, RenderSettings)> {
    let content = std::fs::read_to_string(path)?;
    let file: SceneFile = serde_json::from_str(&content)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut scene = Scene::new();

    // Create material name -> id mapping
    let mut material_ids = std::collections::HashMap::new();
    for mat_config in &file.materials {
        let mat: Box<dyn crate::material::Material> = match mat_config.material_type.as_str() {
            "lambertian" => {
                // Check for checker texture
                if let Some(checker_colors) = &mat_config.checker {
                    let scale = mat_config.scale.unwrap_or(1.0);
                    let odd = Vec3::new(checker_colors[0][0], checker_colors[0][1], checker_colors[0][2]);
                    let even = Vec3::new(checker_colors[1][0], checker_colors[1][1], checker_colors[1][2]);
                    // Note: CheckerTexture is separate from Lambertian
                    // For now, use average color (proper implementation would need textured materials)
                    let avg = (odd + even) * 0.5;
                    let _ = (scale, CheckerTexture::from_colors(odd, even, scale)); // Mark as used
                    Box::new(Lambertian::new(avg))
                } else {
                    let albedo = mat_config.albedo.unwrap_or([0.5, 0.5, 0.5]);
                    Box::new(Lambertian::new(Vec3::new(albedo[0], albedo[1], albedo[2])))
                }
            }
            "metal" => {
                let albedo = mat_config.albedo.unwrap_or([0.8, 0.8, 0.8]);
                let roughness = mat_config.roughness.unwrap_or(0.0);
                Box::new(Metal::new(Vec3::new(albedo[0], albedo[1], albedo[2]), roughness))
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
        
        match obj_config.object_type.as_str() {
            "sphere" => {
                let center = obj_config.center.unwrap_or([0.0, 0.0, 0.0]);
                let radius = obj_config.radius.unwrap_or(1.0);
                scene.add_sphere(Sphere::new(
                    Vec3::new(center[0], center[1], center[2]),
                    radius,
                    material_id,
                ));
            }
            "triangle" => {
                if let Some(verts) = &obj_config.vertices {
                    let v0 = Vec3::new(verts[0][0], verts[0][1], verts[0][2]);
                    let v1 = Vec3::new(verts[1][0], verts[1][1], verts[1][2]);
                    let v2 = Vec3::new(verts[2][0], verts[2][1], verts[2][2]);
                    scene.add_triangle(Triangle::new(v0, v1, v2, material_id));
                }
            }
            "quad" => {
                if let Some(verts) = &obj_config.quad_vertices {
                    let v0 = Vec3::new(verts[0][0], verts[0][1], verts[0][2]);
                    let v1 = Vec3::new(verts[1][0], verts[1][1], verts[1][2]);
                    let v2 = Vec3::new(verts[2][0], verts[2][1], verts[2][2]);
                    let v3 = Vec3::new(verts[3][0], verts[3][1], verts[3][2]);
                    scene.add_quad(Quad::from_vertices(v0, v1, v2, v3, material_id));
                }
            }
            "mesh" => {
                if let Some(mesh_path) = &obj_config.mesh {
                    // Resolve mesh path relative to scene file
                    let scene_dir = path.parent().unwrap_or(Path::new("."));
                    let mesh_file = scene_dir.join(mesh_path);
                    match load_obj(&mesh_file, material_id) {
                        Ok(mesh) => scene.add_mesh(mesh),
                        Err(e) => eprintln!("Warning: Failed to load mesh {}: {}", mesh_path, e),
                    }
                }
            }
            _ => {
                eprintln!("Warning: Unknown object type: {}", obj_config.object_type);
            }
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
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_scene(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".json").unwrap();
        write!(file, "{}", content).unwrap();
        file
    }

    #[test]
    fn test_load_simple_scene() {
        let content = r#"{
            "camera": {"position": [0, 0, 3], "look_at": [0, 0, 0], "fov": 60},
            "settings": {"width": 100, "height": 100, "samples": 10, "depth": 5},
            "materials": [{"name": "mat", "type": "lambertian", "albedo": [0.5, 0.5, 0.5]}],
            "objects": [{"type": "sphere", "center": [0, 0, 0], "radius": 1, "material": "mat"}]
        }"#;
        let file = create_test_scene(content);
        let result = load_scene(file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_triangle() {
        let content = r#"{
            "camera": {"position": [0, 0, 3], "look_at": [0, 0, 0]},
            "materials": [{"name": "mat", "type": "lambertian"}],
            "objects": [{"type": "triangle", "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "material": "mat"}]
        }"#;
        let file = create_test_scene(content);
        let result = load_scene(file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_quad() {
        let content = r#"{
            "camera": {"position": [0, 0, 3], "look_at": [0, 0, 0]},
            "materials": [{"name": "mat", "type": "lambertian"}],
            "objects": [{"type": "quad", "quad_vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], "material": "mat"}]
        }"#;
        let file = create_test_scene(content);
        let result = load_scene(file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_all_materials() {
        let content = r#"{
            "camera": {"position": [0, 0, 3], "look_at": [0, 0, 0]},
            "materials": [
                {"name": "diffuse", "type": "lambertian", "albedo": [1, 0, 0]},
                {"name": "metal", "type": "metal", "albedo": [0.8, 0.8, 0.8], "roughness": 0.1},
                {"name": "glass", "type": "dielectric", "ior": 1.5},
                {"name": "light", "type": "emissive", "color": [1, 1, 1], "strength": 10}
            ],
            "objects": [
                {"type": "sphere", "center": [0, 0, 0], "radius": 1, "material": "diffuse"}
            ]
        }"#;
        let file = create_test_scene(content);
        let result = load_scene(file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_checker_material() {
        let content = r#"{
            "camera": {"position": [0, 0, 3], "look_at": [0, 0, 0]},
            "materials": [
                {"name": "checker", "type": "lambertian", "checker": [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]], "scale": 10}
            ],
            "objects": [{"type": "sphere", "center": [0, 0, 0], "radius": 1, "material": "checker"}]
        }"#;
        let file = create_test_scene(content);
        let result = load_scene(file.path());
        assert!(result.is_ok());
    }
}
