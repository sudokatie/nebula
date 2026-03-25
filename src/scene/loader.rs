//! Scene loading from JSON

use std::path::Path;
use std::io;

use crate::math::Vec3;
use crate::geometry::Sphere;
use crate::material::{Lambertian, Metal, Dielectric, Emissive};
use crate::camera::Camera;
use super::Scene;

use serde::Deserialize;

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
}

#[derive(Deserialize)]
struct ObjectConfig {
    #[serde(rename = "type")]
    object_type: String,
    #[serde(default)]
    center: Option<[f32; 3]>,
    #[serde(default)]
    radius: Option<f32>,
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
                let albedo = mat_config.albedo.unwrap_or([0.5, 0.5, 0.5]);
                Box::new(Lambertian::new(Vec3::new(albedo[0], albedo[1], albedo[2])))
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
