use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

use nebula::scene::load_scene;
use nebula::render::CpuRenderer;
use nebula::output::{save_png, save_ppm};

#[cfg(feature = "gpu")]
use nebula::render::{GpuRenderer, GpuConfig};

#[derive(Parser)]
#[command(name = "nebula")]
#[command(about = "A physically-based path tracer", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Render a scene to an image
    Render {
        /// Scene file (JSON)
        scene: PathBuf,

        /// Output image path
        #[arg(short, long, default_value = "output.png")]
        output: PathBuf,

        /// Image width (overrides scene)
        #[arg(short = 'W', long)]
        width: Option<u32>,

        /// Image height (overrides scene)
        #[arg(short = 'H', long)]
        height: Option<u32>,

        /// Samples per pixel (overrides scene)
        #[arg(short, long)]
        samples: Option<u32>,

        /// Maximum ray depth (overrides scene)
        #[arg(short, long)]
        depth: Option<u32>,

        /// Number of threads (0 = auto)
        #[arg(short, long, default_value_t = 0)]
        threads: usize,

        /// Use GPU rendering
        #[arg(long)]
        gpu: bool,
    },

    /// Preview a scene with progressive rendering
    Preview {
        /// Scene file (JSON)
        scene: PathBuf,

        /// Image width
        #[arg(short = 'W', long, default_value_t = 400)]
        width: u32,

        /// Image height
        #[arg(short = 'H', long, default_value_t = 300)]
        height: u32,

        /// Maximum samples
        #[arg(short, long, default_value_t = 100)]
        samples: u32,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Render {
            scene: scene_path,
            output,
            width,
            height,
            samples,
            depth,
            threads,
            gpu,
        } => {
            // Load scene
            println!("Loading scene: {}", scene_path.display());
            let (scene, camera, settings) = match load_scene(&scene_path) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("Error loading scene: {}", e);
                    std::process::exit(1);
                }
            };

            // Apply overrides
            let width = width.unwrap_or(settings.width);
            let height = height.unwrap_or(settings.height);
            let samples = samples.unwrap_or(settings.samples_per_pixel);
            let depth = depth.unwrap_or(settings.max_depth);

            println!("Resolution: {}x{}", width, height);
            println!("Samples: {}, Depth: {}", samples, depth);

            let pixels = if gpu {
                #[cfg(feature = "gpu")]
                {
                    println!("GPU rendering...");
                    render_gpu(&scene, &camera, width, height, samples, depth)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    eprintln!("GPU support not compiled. Rebuild with: cargo build --features gpu");
                    std::process::exit(1);
                }
            } else {
                // CPU rendering
                let renderer = CpuRenderer::new(width, height, samples, depth);
                let start = Instant::now();
                
                let pixels = if threads == 1 {
                    println!("Rendering (single-threaded)...");
                    renderer.render(&scene, &camera)
                } else {
                    let t = if threads == 0 { rayon::current_num_threads() } else { threads };
                    println!("Rendering ({} threads)...", t);
                    renderer.render_parallel(&scene, &camera, threads)
                };

                let elapsed = start.elapsed();
                println!("Rendered in {:.2}s", elapsed.as_secs_f64());
                pixels
            };

            // Save output
            let result = if output.extension().map(|e| e == "ppm").unwrap_or(false) {
                save_ppm(&output, &pixels, width, height)
            } else {
                save_png(&output, &pixels, width, height)
            };

            match result {
                Ok(()) => println!("Saved to {}", output.display()),
                Err(e) => {
                    eprintln!("Error saving image: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Preview { scene: scene_path, width, height, samples } => {
            println!("Loading scene: {}", scene_path.display());
            let (scene, camera, _settings) = match load_scene(&scene_path) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("Error loading scene: {}", e);
                    std::process::exit(1);
                }
            };

            println!("Preview mode: {}x{}, {} samples", width, height, samples);
            println!("Progressive rendering...");

            let renderer = CpuRenderer::new(width, height, samples, 10);
            let start = Instant::now();

            renderer.render_progressive(&scene, &camera, |_pixels, sample| {
                // In a real preview, we'd update a window here
                print!("\rSample {}/{}...", sample, samples);
                use std::io::Write;
                std::io::stdout().flush().ok();
            });

            println!();
            let elapsed = start.elapsed();
            println!("Preview completed in {:.2}s", elapsed.as_secs_f64());

            // Final render and save
            let pixels = renderer.render_parallel(&scene, &camera, 0);
            let output = PathBuf::from("preview.png");
            if let Err(e) = save_png(&output, &pixels, width, height) {
                eprintln!("Error saving preview: {}", e);
            } else {
                println!("Saved preview to {}", output.display());
            }
        }
    }
}

#[cfg(feature = "gpu")]
fn render_gpu(
    scene: &nebula::Scene,
    camera: &nebula::Camera,
    width: u32,
    height: u32,
    samples: u32,
    depth: u32,
) -> Vec<nebula::Vec3> {
    use pollster::FutureExt;

    let config = GpuConfig {
        width,
        height,
        samples_per_pixel: samples,
        max_depth: depth,
    };

    let mut renderer = match GpuRenderer::new(config).block_on() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to initialize GPU: {}", e);
            std::process::exit(1);
        }
    };

    // Validate scene
    let validation = renderer.validate_scene(scene);
    if !validation.valid {
        for error in &validation.errors {
            eprintln!("Scene validation error: {}", error);
        }
        std::process::exit(1);
    }

    let start = Instant::now();
    let pixels = renderer.render(scene, camera).block_on();
    let elapsed = start.elapsed();
    println!("GPU rendered in {:.2}s", elapsed.as_secs_f64());

    pixels
}
