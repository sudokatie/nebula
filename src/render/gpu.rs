//! GPU renderer using wgpu compute shaders

use crate::math::Vec3;
use crate::camera::Camera;
use crate::scene::Scene;

/// GPU renderer configuration
pub struct GpuConfig {
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u32,
    pub max_depth: u32,
}

/// GPU-based path tracer
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    config: GpuConfig,
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
    _padding: [f32; 3],
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
            bind_group,
            config,
        })
    }

    /// Render the scene
    pub async fn render(&self, _scene: &Scene, camera: &Camera) -> Vec<Vec3> {
        // Create uniforms from camera
        let uniforms = self.create_uniforms(camera);
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
            
            // Dispatch one thread per pixel
            let workgroup_size = 8;
            let workgroups_x = (self.config.width + workgroup_size - 1) / workgroup_size;
            let workgroups_y = (self.config.height + workgroup_size - 1) / workgroup_size;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.output_buffer.size(),
        );

        // Submit commands
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

    fn create_uniforms(&self, _camera: &Camera) -> Uniforms {
        // TODO: Extract camera data properly
        // For now, use placeholder values
        Uniforms {
            width: self.config.width,
            height: self.config.height,
            samples: self.config.samples_per_pixel,
            max_depth: self.config.max_depth,
            camera_origin: [0.0, 0.0, 3.0, 0.0],
            camera_lower_left: [-2.0, -1.5, -1.0, 0.0],
            camera_horizontal: [4.0, 0.0, 0.0, 0.0],
            camera_vertical: [0.0, 3.0, 0.0, 0.0],
            camera_u: [1.0, 0.0, 0.0, 0.0],
            camera_v: [0.0, 1.0, 0.0, 0.0],
            lens_radius: 0.0,
            _padding: [0.0; 3],
        }
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
}
