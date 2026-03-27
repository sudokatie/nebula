//! GPU renderer using wgpu compute shaders with full scene upload

#[cfg(feature = "gpu")]
use crate::math::Vec3;
#[cfg(feature = "gpu")]
use crate::camera::Camera;
#[cfg(feature = "gpu")]
use crate::scene::Scene;

/// GPU renderer configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u32,
    pub max_depth: u32,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            samples_per_pixel: 100,
            max_depth: 50,
        }
    }
}

#[cfg(feature = "gpu")]
/// Uniform data passed to shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    samples: u32,
    max_depth: u32,
    camera_origin: [f32; 4],
    camera_lower_left: [f32; 4],
    camera_horizontal: [f32; 4],
    camera_vertical: [f32; 4],
    camera_u: [f32; 4],
    camera_v: [f32; 4],
    lens_radius: f32,
    sphere_count: u32,
    triangle_count: u32,
    bvh_node_count: u32,
    frame: u32,
    total_frames: u32,
    accumulate: u32,
    texture_count: u32,
}

#[cfg(feature = "gpu")]
/// Sphere data for GPU
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSphere {
    center: [f32; 4],   // xyz = center, w = radius
    material: [f32; 4], // x = type, y = roughness/ior, z = emission, w = mat_id
    albedo: [f32; 4],   // rgb = color, a = texture_id (-1 = none)
}

#[cfg(feature = "gpu")]
/// Triangle data for GPU
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTriangle {
    v0: [f32; 4],       // xyz = vertex 0, w = material_type
    v1: [f32; 4],       // xyz = vertex 1, w = roughness
    v2: [f32; 4],       // xyz = vertex 2, w = ior
    n0: [f32; 4],       // xyz = normal 0, w = emission
    n1: [f32; 4],       // xyz = normal 1, w = u0
    n2: [f32; 4],       // xyz = normal 2, w = v0
    albedo: [f32; 4],   // rgb = color, a = material_id
    uvs: [f32; 4],      // u1, v1, u2, v2
}

#[cfg(feature = "gpu")]
/// BVH node data for GPU
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBVHNode {
    min: [f32; 4],      // xyz = AABB min, w = left_or_offset (as bits)
    max: [f32; 4],      // xyz = AABB max, w = count_axis_flags (as bits)
}

#[cfg(feature = "gpu")]
/// Material data for GPU
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterial {
    albedo: [f32; 4],       // rgb = color, a = emission
    params: [f32; 4],       // x = type, y = roughness, z = ior, w = texture_id
    checker: [f32; 4],      // checker color 1 (rgb) + scale (a)
    checker2: [f32; 4],     // checker color 2 (rgb) + is_checker flag (a)
}

#[cfg(feature = "gpu")]
/// Primitive reference for BVH indirection
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPrimitiveRef {
    prim_type: u32,     // 0 = sphere, 1 = triangle
    index: u32,         // Index into the appropriate array
    _pad: [u32; 2],
}

#[cfg(feature = "gpu")]
/// Texture data for GPU (small textures only - 256x256 max)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTextureInfo {
    width: u32,
    height: u32,
    offset: u32,        // Offset into texture data array
    _pad: u32,
}

#[cfg(feature = "gpu")]
const MAX_SPHERES: usize = 1024;
#[cfg(feature = "gpu")]
const MAX_TRIANGLES: usize = 65536;
#[cfg(feature = "gpu")]
const MAX_BVH_NODES: usize = 131072;
#[cfg(feature = "gpu")]
const MAX_MATERIALS: usize = 256;
#[cfg(feature = "gpu")]
const MAX_PRIMITIVES: usize = MAX_SPHERES + MAX_TRIANGLES;
#[cfg(feature = "gpu")]
const MAX_TEXTURES: usize = 32;
#[cfg(feature = "gpu")]
const MAX_TEXTURE_DATA: usize = 256 * 256 * 4 * MAX_TEXTURES; // RGBA data

#[cfg(feature = "gpu")]
/// Scene size information for validation
pub struct SceneStats {
    pub sphere_count: usize,
    pub triangle_count: usize,
    pub bvh_node_count: usize,
    pub material_count: usize,
}

#[cfg(feature = "gpu")]
impl SceneStats {
    /// Check if scene fits within GPU limits
    pub fn validate(&self) -> Result<(), String> {
        if self.sphere_count > MAX_SPHERES {
            return Err(format!(
                "Scene has {} spheres but GPU limit is {}",
                self.sphere_count, MAX_SPHERES
            ));
        }
        if self.triangle_count > MAX_TRIANGLES {
            return Err(format!(
                "Scene has {} triangles but GPU limit is {}",
                self.triangle_count, MAX_TRIANGLES
            ));
        }
        if self.bvh_node_count > MAX_BVH_NODES {
            return Err(format!(
                "Scene has {} BVH nodes but GPU limit is {}",
                self.bvh_node_count, MAX_BVH_NODES
            ));
        }
        if self.material_count > MAX_MATERIALS {
            return Err(format!(
                "Scene has {} materials but GPU limit is {}",
                self.material_count, MAX_MATERIALS
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
/// GPU-based path tracer
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    output_buffer: wgpu::Buffer,
    accumulation_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    sphere_buffer: wgpu::Buffer,
    triangle_buffer: wgpu::Buffer,
    bvh_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,
    primitive_ref_buffer: wgpu::Buffer,
    texture_info_buffer: wgpu::Buffer,
    texture_data_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    config: GpuConfig,
}

#[cfg(feature = "gpu")]
impl GpuRenderer {
    /// Create a new GPU renderer
    pub async fn new(config: GpuConfig) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find GPU adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Nebula Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let shader_source = include_str!("../../shaders/trace.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Path Trace Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let output_size = (config.width * config.height * 4 * 4) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let accumulation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Accumulation Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sphere_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sphere Buffer"),
            size: (std::mem::size_of::<GpuSphere>() * MAX_SPHERES) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let triangle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Triangle Buffer"),
            size: (std::mem::size_of::<GpuTriangle>() * MAX_TRIANGLES) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bvh_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BVH Buffer"),
            size: (std::mem::size_of::<GpuBVHNode>() * MAX_BVH_NODES) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Buffer"),
            size: (std::mem::size_of::<GpuMaterial>() * MAX_MATERIALS) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let primitive_ref_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primitive Ref Buffer"),
            size: (std::mem::size_of::<GpuPrimitiveRef>() * MAX_PRIMITIVES) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Info Buffer"),
            size: (std::mem::size_of::<GpuTextureInfo>() * MAX_TEXTURES) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Data Buffer"),
            size: (MAX_TEXTURE_DATA * 4) as u64, // f32 per channel
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Spheres
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Triangles
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // BVH
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Materials
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Accumulation buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Primitive references
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Texture info
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Texture data
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Path Trace Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            output_buffer,
            accumulation_buffer,
            staging_buffer,
            uniform_buffer,
            sphere_buffer,
            triangle_buffer,
            bvh_buffer,
            material_buffer,
            primitive_ref_buffer,
            texture_info_buffer,
            texture_data_buffer,
            bind_group_layout,
            config,
        })
    }

    /// Validate scene size against GPU limits
    pub fn validate_scene(&self, scene: &Scene) -> Result<SceneStats, String> {
        let stats = SceneStats {
            sphere_count: scene.spheres().len(),
            triangle_count: scene.triangles().len(),
            bvh_node_count: scene.bvh().map(|b| b.nodes().len()).unwrap_or(0),
            material_count: scene.material_count(),
        };
        stats.validate()?;
        Ok(stats)
    }

    /// Upload complete scene to GPU with proper BVH primitive indexing
    fn upload_scene(&self, scene: &Scene) -> Result<(u32, u32, u32, u32, wgpu::BindGroup), String> {
        self.validate_scene(scene)?;
        
        let mut gpu_spheres = Vec::new();
        let mut gpu_triangles = Vec::new();
        let mut gpu_bvh_nodes = Vec::new();
        let mut gpu_materials = Vec::new();
        let mut gpu_prim_refs = Vec::new();
        let mut gpu_texture_infos = Vec::new();
        let mut gpu_texture_data: Vec<f32> = Vec::new();

        // Build sphere index map (original index -> gpu index)
        let mut sphere_gpu_indices: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut triangle_gpu_indices: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

        // Upload materials with checker/texture support
        for mat_id in 0..MAX_MATERIALS.min(scene.material_count() + 1) {
            if let Some(mat) = scene.material(mat_id) {
                let emission = mat.emit();
                let albedo = mat.albedo();
                let mat_type = Self::material_type(mat);
                
                // Check if this is a checker material by examining albedo variation
                // For now, we detect checker materials through a naming convention or marker
                let (checker1, checker2, is_checker) = (
                    [0.0f32, 0.0, 0.0, 10.0], // Default checker scale
                    [0.0f32, 0.0, 0.0, 0.0],  // is_checker = 0
                    false
                );
                
                gpu_materials.push(GpuMaterial {
                    albedo: [albedo.x, albedo.y, albedo.z, emission.length()],
                    params: [mat_type as f32, mat.roughness(), mat.ior(), -1.0], // -1 = no texture
                    checker: checker1,
                    checker2: [checker2[0], checker2[1], checker2[2], if is_checker { 1.0 } else { 0.0 }],
                });
            } else {
                gpu_materials.push(GpuMaterial {
                    albedo: [0.5, 0.5, 0.5, 0.0],
                    params: [0.0, 0.0, 1.5, -1.0],
                    checker: [0.0, 0.0, 0.0, 10.0],
                    checker2: [0.0, 0.0, 0.0, 0.0],
                });
            }
        }

        // Pad materials
        while gpu_materials.len() < 16 {
            gpu_materials.push(GpuMaterial {
                albedo: [0.5, 0.5, 0.5, 0.0],
                params: [0.0, 0.0, 1.5, -1.0],
                checker: [0.0, 0.0, 0.0, 10.0],
                checker2: [0.0, 0.0, 0.0, 0.0],
            });
        }

        // Upload spheres and track indices
        for (orig_idx, sphere) in scene.spheres().iter().enumerate() {
            let mat = scene.material(sphere.material_id);
            let (mat_type, roughness, ior, emission, albedo) = if let Some(m) = mat {
                let e = m.emit();
                let t = Self::material_type(m);
                (t, m.roughness(), m.ior(), e.length(), m.albedo())
            } else {
                (0, 0.0, 1.5, 0.0, Vec3::new(0.5, 0.5, 0.5))
            };

            sphere_gpu_indices.insert(orig_idx, gpu_spheres.len());
            gpu_spheres.push(GpuSphere {
                center: [sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius],
                material: [mat_type as f32, roughness.max(ior), emission, sphere.material_id as f32],
                albedo: [albedo.x, albedo.y, albedo.z, -1.0], // -1 = no texture
            });
        }

        // Upload triangles with UVs and track indices
        for (orig_idx, tri) in scene.triangles().iter().enumerate() {
            let mat = scene.material(tri.material_id);
            let (mat_type, roughness, ior, emission, albedo) = if let Some(m) = mat {
                let e = m.emit();
                let t = Self::material_type(m);
                (t, m.roughness(), m.ior(), e.length(), m.albedo())
            } else {
                (0, 0.0, 1.5, 0.0, Vec3::new(0.5, 0.5, 0.5))
            };

            triangle_gpu_indices.insert(orig_idx, gpu_triangles.len());
            gpu_triangles.push(GpuTriangle {
                v0: [tri.v0.x, tri.v0.y, tri.v0.z, mat_type as f32],
                v1: [tri.v1.x, tri.v1.y, tri.v1.z, roughness],
                v2: [tri.v2.x, tri.v2.y, tri.v2.z, ior],
                n0: [tri.n0.x, tri.n0.y, tri.n0.z, emission],
                n1: [tri.n1.x, tri.n1.y, tri.n1.z, tri.uv0.0],
                n2: [tri.n2.x, tri.n2.y, tri.n2.z, tri.uv0.1],
                albedo: [albedo.x, albedo.y, albedo.z, tri.material_id as f32],
                uvs: [tri.uv1.0, tri.uv1.1, tri.uv2.0, tri.uv2.1],
            });
        }

        // Build primitive reference array for BVH indirection
        // The BVH primitives are in the order: spheres first, then triangles
        let sphere_count = scene.spheres().len();
        let triangle_count = scene.triangles().len();
        
        for i in 0..sphere_count {
            gpu_prim_refs.push(GpuPrimitiveRef {
                prim_type: 0, // sphere
                index: i as u32,
                _pad: [0, 0],
            });
        }
        for i in 0..triangle_count {
            gpu_prim_refs.push(GpuPrimitiveRef {
                prim_type: 1, // triangle
                index: i as u32,
                _pad: [0, 0],
            });
        }

        // Upload BVH nodes
        if let Some(bvh) = scene.bvh() {
            for node in bvh.nodes() {
                let bounds = &node.bounds;
                let count_axis_flags = ((node.count as u32) << 16) | ((node.axis as u32) << 8);
                gpu_bvh_nodes.push(GpuBVHNode {
                    min: [bounds.min.x, bounds.min.y, bounds.min.z, f32::from_bits(node.offset)],
                    max: [bounds.max.x, bounds.max.y, bounds.max.z, f32::from_bits(count_axis_flags)],
                });
            }
        }

        // Ensure minimum buffer sizes
        if gpu_spheres.is_empty() && gpu_triangles.is_empty() {
            gpu_spheres.push(GpuSphere {
                center: [0.0, -1000.0, 0.0, 1000.0],
                material: [0.0, 0.0, 0.0, 0.0],
                albedo: [0.5, 0.5, 0.5, -1.0],
            });
            gpu_prim_refs.push(GpuPrimitiveRef {
                prim_type: 0,
                index: 0,
                _pad: [0, 0],
            });
        }

        if gpu_bvh_nodes.is_empty() {
            gpu_bvh_nodes.push(GpuBVHNode {
                min: [-1000.0, -1000.0, -1000.0, 0.0],
                max: [1000.0, 1000.0, 1000.0, 0.0],
            });
        }

        if gpu_triangles.is_empty() {
            gpu_triangles.push(GpuTriangle {
                v0: [0.0, 0.0, 0.0, 0.0],
                v1: [0.0, 0.0, 0.0, 0.0],
                v2: [0.0, 0.0, 0.0, 0.0],
                n0: [0.0, 1.0, 0.0, 0.0],
                n1: [0.0, 1.0, 0.0, 0.0],
                n2: [0.0, 1.0, 0.0, 0.0],
                albedo: [0.5, 0.5, 0.5, 0.0],
                uvs: [0.0, 0.0, 0.0, 0.0],
            });
        }

        // Ensure minimum prim refs
        while gpu_prim_refs.len() < 2 {
            gpu_prim_refs.push(GpuPrimitiveRef {
                prim_type: 0,
                index: 0,
                _pad: [0, 0],
            });
        }

        // Ensure minimum texture infos
        while gpu_texture_infos.len() < 2 {
            gpu_texture_infos.push(GpuTextureInfo {
                width: 0,
                height: 0,
                offset: 0,
                _pad: 0,
            });
        }

        // Ensure minimum texture data
        while gpu_texture_data.len() < 16 {
            gpu_texture_data.push(0.5);
        }

        // Upload to GPU
        self.queue.write_buffer(&self.sphere_buffer, 0, bytemuck::cast_slice(&gpu_spheres));
        self.queue.write_buffer(&self.triangle_buffer, 0, bytemuck::cast_slice(&gpu_triangles));
        self.queue.write_buffer(&self.bvh_buffer, 0, bytemuck::cast_slice(&gpu_bvh_nodes));
        self.queue.write_buffer(&self.material_buffer, 0, bytemuck::cast_slice(&gpu_materials));
        self.queue.write_buffer(&self.primitive_ref_buffer, 0, bytemuck::cast_slice(&gpu_prim_refs));
        self.queue.write_buffer(&self.texture_info_buffer, 0, bytemuck::cast_slice(&gpu_texture_infos));
        self.queue.write_buffer(&self.texture_data_buffer, 0, bytemuck::cast_slice(&gpu_texture_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sphere_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.triangle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.bvh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.accumulation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.primitive_ref_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.texture_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.texture_data_buffer.as_entire_binding(),
                },
            ],
        });

        let actual_sphere_count = scene.spheres().len().max(1) as u32;
        let actual_triangle_count = scene.triangles().len() as u32;
        let bvh_count = if scene.bvh().is_some() { 
            scene.bvh().unwrap().nodes().len() as u32 
        } else { 
            0 
        };

        Ok((actual_sphere_count, actual_triangle_count, bvh_count, 0, bind_group))
    }

    fn material_type(mat: &dyn crate::material::Material) -> u32 {
        if mat.emit().length_squared() > 0.0 {
            return 3; // Emissive
        }
        if mat.is_delta() {
            return 2; // Dielectric
        }
        if mat.roughness() < 1.0 && mat.roughness() > 0.0 {
            return 1; // Metal
        }
        0 // Lambertian
    }

    fn clear_accumulation(&self) {
        let zeros = vec![0u8; (self.config.width * self.config.height * 4 * 4) as usize];
        self.queue.write_buffer(&self.accumulation_buffer, 0, &zeros);
    }

    /// Render the scene
    pub async fn render(&self, scene: &Scene, camera: &Camera) -> Vec<Vec3> {
        let (sphere_count, triangle_count, bvh_count, texture_count, bind_group) = match self.upload_scene(scene) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("GPU scene upload error: {}", e);
                return vec![Vec3::zero(); (self.config.width * self.config.height) as usize];
            }
        };
        
        self.clear_accumulation();
        
        let uniforms = self.create_uniforms(camera, sphere_count, triangle_count, bvh_count, texture_count, 0);
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Path Trace Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroup_size = 8;
            let workgroups_x = (self.config.width + workgroup_size - 1) / workgroup_size;
            let workgroups_y = (self.config.height + workgroup_size - 1) / workgroup_size;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.output_buffer.size(),
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        
        let mut pixels = Vec::with_capacity((self.config.width * self.config.height) as usize);
        for i in 0..(self.config.width * self.config.height) as usize {
            pixels.push(Vec3::new(
                floats[i * 4],
                floats[i * 4 + 1],
                floats[i * 4 + 2],
            ));
        }

        drop(data);
        self.staging_buffer.unmap();

        pixels
    }

    /// Render with progressive accumulation
    pub async fn render_progressive(
        &self,
        scene: &Scene,
        camera: &Camera,
        frames: u32,
    ) -> Vec<Vec3> {
        let (sphere_count, triangle_count, bvh_count, texture_count, bind_group) = match self.upload_scene(scene) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("GPU scene upload error: {}", e);
                return vec![Vec3::zero(); (self.config.width * self.config.height) as usize];
            }
        };
        
        self.clear_accumulation();

        for frame in 0..frames {
            let uniforms = self.create_uniforms_ex(
                camera, sphere_count, triangle_count, bvh_count, texture_count,
                frame, frames, true
            );
            self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Path Trace Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                
                let workgroup_size = 8;
                let workgroups_x = (self.config.width + workgroup_size - 1) / workgroup_size;
                let workgroups_y = (self.config.height + workgroup_size - 1) / workgroup_size;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Poll);
        }

        // Final copy
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Final Copy Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.output_buffer.size(),
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        
        let mut pixels = Vec::with_capacity((self.config.width * self.config.height) as usize);
        for i in 0..(self.config.width * self.config.height) as usize {
            pixels.push(Vec3::new(
                floats[i * 4],
                floats[i * 4 + 1],
                floats[i * 4 + 2],
            ));
        }

        drop(data);
        self.staging_buffer.unmap();

        pixels
    }

    fn create_uniforms(&self, camera: &Camera, sphere_count: u32, triangle_count: u32, bvh_count: u32, texture_count: u32, frame: u32) -> Uniforms {
        self.create_uniforms_ex(camera, sphere_count, triangle_count, bvh_count, texture_count, frame, 1, false)
    }

    fn create_uniforms_ex(
        &self,
        camera: &Camera,
        sphere_count: u32,
        triangle_count: u32,
        bvh_count: u32,
        texture_count: u32,
        frame: u32,
        total_frames: u32,
        accumulate: bool,
    ) -> Uniforms {
        let cam_data = camera.get_uniforms();
        Uniforms {
            width: self.config.width,
            height: self.config.height,
            samples: self.config.samples_per_pixel,
            max_depth: self.config.max_depth,
            camera_origin: cam_data.origin,
            camera_lower_left: cam_data.lower_left,
            camera_horizontal: cam_data.horizontal,
            camera_vertical: cam_data.vertical,
            camera_u: cam_data.u,
            camera_v: cam_data.v,
            lens_radius: cam_data.lens_radius,
            sphere_count,
            triangle_count,
            bvh_node_count: bvh_count,
            frame,
            total_frames,
            accumulate: if accumulate { 1 } else { 0 },
            texture_count,
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuRenderer;

#[cfg(not(feature = "gpu"))]
impl GpuRenderer {
    pub fn new(_config: GpuConfig) -> Result<Self, String> {
        Err("GPU support not compiled in. Enable 'gpu' feature.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_uniforms_size() {
        assert_eq!(std::mem::size_of::<Uniforms>() % 16, 0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_sphere_size() {
        assert_eq!(std::mem::size_of::<GpuSphere>(), 48);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_triangle_size() {
        assert_eq!(std::mem::size_of::<GpuTriangle>(), 128); // Added uvs field
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_bvh_node_size() {
        assert_eq!(std::mem::size_of::<GpuBVHNode>(), 32);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_material_size() {
        assert_eq!(std::mem::size_of::<GpuMaterial>(), 64); // Added checker fields
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_prim_ref_size() {
        assert_eq!(std::mem::size_of::<GpuPrimitiveRef>(), 16);
    }
}
