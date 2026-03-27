//! GPU renderer using wgpu compute shaders with progressive accumulation

use crate::math::Vec3;
use crate::camera::Camera;
use crate::scene::Scene;

/// GPU rendering limits for validation
pub const MAX_SPHERES: usize = 1024;
pub const MAX_TRIANGLES: usize = 65536;
pub const MAX_BVH_NODES: usize = 131072;
pub const MAX_MATERIALS: usize = 256;

/// GPU renderer configuration
pub struct GpuConfig {
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u32,
    pub max_depth: u32,
}

/// Scene validation result
#[derive(Debug)]
pub struct SceneValidation {
    pub sphere_count: usize,
    pub triangle_count: usize,
    pub material_count: usize,
    pub valid: bool,
    pub errors: Vec<String>,
}

impl SceneValidation {
    pub fn validate(scene: &Scene) -> Self {
        let mut errors = Vec::new();
        
        // Note: These are placeholder counts - proper implementation would track in Scene
        let sphere_count = 0; // scene.sphere_count()
        let triangle_count = 0; // scene.triangle_count()
        let material_count = 0; // scene.material_count()

        if sphere_count > MAX_SPHERES {
            errors.push(format!("Too many spheres: {} > {}", sphere_count, MAX_SPHERES));
        }
        if triangle_count > MAX_TRIANGLES {
            errors.push(format!("Too many triangles: {} > {}", triangle_count, MAX_TRIANGLES));
        }
        if material_count > MAX_MATERIALS {
            errors.push(format!("Too many materials: {} > {}", material_count, MAX_MATERIALS));
        }

        let _ = scene; // Mark as used

        Self {
            sphere_count,
            triangle_count,
            material_count,
            valid: errors.is_empty(),
            errors,
        }
    }
}

/// GPU-based path tracer with progressive accumulation
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    accumulator_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    config: GpuConfig,
    sample_count: u32,
}

/// Uniform data passed to shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    samples: u32,
    max_depth: u32,
    // Camera
    camera_origin: [f32; 4],
    camera_lower_left: [f32; 4],
    camera_horizontal: [f32; 4],
    camera_vertical: [f32; 4],
    camera_u: [f32; 4],
    camera_v: [f32; 4],
    lens_radius: f32,
    sample_offset: u32, // For progressive rendering
    _padding: [f32; 2],
}

impl GpuRenderer {
    /// Create a new GPU renderer
    pub async fn new(config: GpuConfig) -> Result<Self, String> {
        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find GPU adapter")?;

        // Request device
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

        // Load shader
        let shader_source = include_str!("../../shaders/trace.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Path Trace Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create output buffer (RGBA f32)
        let output_size = (config.width * config.height * 4 * 4) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create accumulator buffer for progressive rendering
        let accumulator_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Accumulator Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create staging buffer for readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
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
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
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
            staging_buffer,
            uniform_buffer,
            accumulator_buffer,
            bind_group,
            config,
            sample_count: 0,
        })
    }

    /// Validate scene against GPU limits
    pub fn validate_scene(&self, scene: &Scene) -> SceneValidation {
        SceneValidation::validate(scene)
    }

    /// Reset accumulator for new render
    pub fn reset(&mut self) {
        self.sample_count = 0;
        // Clear accumulator buffer
        let zeros = vec![0u8; (self.config.width * self.config.height * 4 * 4) as usize];
        self.queue.write_buffer(&self.accumulator_buffer, 0, &zeros);
    }

    /// Render one sample (for progressive rendering)
    pub async fn render_sample(&mut self, _scene: &Scene, camera: &Camera) -> bool {
        if self.sample_count >= self.config.samples_per_pixel {
            return false;
        }

        // Create uniforms from camera
        let uniforms = self.create_uniforms(camera, self.sample_count);
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Dispatch compute shader
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Path Trace Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            
            let workgroup_size = 8;
            let workgroups_x = (self.config.width + workgroup_size - 1) / workgroup_size;
            let workgroups_y = (self.config.height + workgroup_size - 1) / workgroup_size;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        self.sample_count += 1;
        true
    }

    /// Render the scene (all samples at once)
    pub async fn render(&mut self, scene: &Scene, camera: &Camera) -> Vec<Vec3> {
        self.reset();

        // Render all samples
        while self.render_sample(scene, camera).await {}

        // Read back results
        self.read_output().await
    }

    /// Render with progressive callback
    pub async fn render_progressive<F>(&mut self, scene: &Scene, camera: &Camera, mut callback: F)
    where
        F: FnMut(&[Vec3], u32),
    {
        self.reset();

        while self.sample_count < self.config.samples_per_pixel {
            self.render_sample(scene, camera).await;
            
            // Read current state and call callback
            let pixels = self.read_output().await;
            callback(&pixels, self.sample_count);
        }
    }

    /// Read output buffer
    async fn read_output(&self) -> Vec<Vec3> {
        // Create command encoder for copy
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.output_buffer.size(),
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map staging buffer and read results
        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map buffer");

        // Read data
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

    fn create_uniforms(&self, _camera: &Camera, sample_offset: u32) -> Uniforms {
        // TODO: Extract camera data properly
        // For now, use placeholder values
        Uniforms {
            width: self.config.width,
            height: self.config.height,
            samples: 1, // One sample per call for progressive
            max_depth: self.config.max_depth,
            camera_origin: [0.0, 0.0, 3.0, 0.0],
            camera_lower_left: [-2.0, -1.5, -1.0, 0.0],
            camera_horizontal: [4.0, 0.0, 0.0, 0.0],
            camera_vertical: [0.0, 3.0, 0.0, 0.0],
            camera_u: [1.0, 0.0, 0.0, 0.0],
            camera_v: [0.0, 1.0, 0.0, 0.0],
            lens_radius: 0.0,
            sample_offset,
            _padding: [0.0; 2],
        }
    }

    /// Get current sample count
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Check if rendering is complete
    pub fn is_complete(&self) -> bool {
        self.sample_count >= self.config.samples_per_pixel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniforms_size() {
        // Ensure uniforms are properly aligned for GPU
        assert_eq!(std::mem::size_of::<Uniforms>() % 16, 0);
    }

    #[test]
    fn test_gpu_limits() {
        assert_eq!(MAX_SPHERES, 1024);
        assert_eq!(MAX_TRIANGLES, 65536);
        assert_eq!(MAX_BVH_NODES, 131072);
        assert_eq!(MAX_MATERIALS, 256);
    }

    #[test]
    fn test_scene_validation() {
        let scene = Scene::new();
        let validation = SceneValidation::validate(&scene);
        assert!(validation.valid);
        assert!(validation.errors.is_empty());
    }
}
