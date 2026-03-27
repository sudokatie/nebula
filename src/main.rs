use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

use nebula::scene::load_scene;
use nebula::render::CpuRenderer;
use nebula::output::{save_png, save_ppm, save_hdr, ToneMap};

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

        /// Tone mapping (linear, reinhard, aces, uncharted2)
        #[arg(long, default_value = "linear")]
        tonemap: String,

        /// Exposure adjustment (stops)
        #[arg(long, default_value_t = 0.0)]
        exposure: f32,
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

        /// Initial samples per iteration
        #[arg(short, long, default_value_t = 1)]
        samples: u32,

        /// Maximum iterations
        #[arg(short, long, default_value_t = 100)]
        iterations: u32,
    },
}

fn parse_tonemap(s: &str) -> ToneMap {
    match s.to_lowercase().as_str() {
        "reinhard" => ToneMap::Reinhard,
        "aces" => ToneMap::Aces,
        "uncharted2" => ToneMap::Uncharted2,
        _ => ToneMap::Linear,
    }
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
            tonemap,
            exposure,
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
            let tone_map = parse_tonemap(&tonemap);

            println!("Resolution: {}x{}", width, height);
            println!("Samples: {}, Depth: {}", samples, depth);

            let pixels = if gpu {
                #[cfg(feature = "gpu")]
                {
                    use nebula::render::{GpuRenderer, GpuConfig};
                    
                    println!("Rendering (GPU)...");
                    let config = GpuConfig {
                        width,
                        height,
                        samples_per_pixel: samples,
                        max_depth: depth,
                    };
                    
                    let start = Instant::now();
                    let result = pollster::block_on(async {
                        let renderer = GpuRenderer::new(config).await?;
                        Ok::<_, String>(renderer.render(&scene, &camera).await)
                    });
                    
                    match result {
                        Ok(pixels) => {
                            let elapsed = start.elapsed();
                            println!("Rendered in {:.2}s", elapsed.as_secs_f64());
                            pixels
                        }
                        Err(e) => {
                            eprintln!("GPU error: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                
                #[cfg(not(feature = "gpu"))]
                {
                    eprintln!("GPU support not compiled in. Recompile with --features gpu");
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
            let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("png");
            let result = match ext {
                "ppm" => save_ppm(&output, &pixels, width, height),
                "hdr" => save_hdr(&output, &pixels, width, height),
                "exr" => {
                    use nebula::output::save_exr;
                    save_exr(&output, &pixels, width, height)
                }
                _ => {
                    use nebula::output::save_png_with_options;
                    save_png_with_options(&output, &pixels, width, height, exposure, tone_map)
                }
            };

            match result {
                Ok(()) => println!("Saved to {}", output.display()),
                Err(e) => {
                    eprintln!("Error saving image: {}", e);
                    std::process::exit(1);
                }
            }
        }
        
        Commands::Preview {
            scene: scene_path,
            width,
            height,
            samples,
            iterations,
        } => {
            // Load scene
            println!("Loading scene: {}", scene_path.display());
            let (scene, camera, _settings) = match load_scene(&scene_path) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("Error loading scene: {}", e);
                    std::process::exit(1);
                }
            };

            println!("Preview: {}x{}, {} iterations", width, height, iterations);
            println!("Press Ctrl+C to stop and save current result\n");

            let renderer = CpuRenderer::new(width, height, samples, 10); // Lower depth for preview
            let mut accumulated = vec![nebula::math::Vec3::zero(); (width * height) as usize];
            let mut total_samples = 0u32;

            // Handle Ctrl+C
            let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
            let r = running.clone();
            ctrlc::set_handler(move || {
                r.store(false, std::sync::atomic::Ordering::SeqCst);
            }).ok();

            for iter in 1..=iterations {
                if !running.load(std::sync::atomic::Ordering::SeqCst) {
                    println!("\nStopped at iteration {}", iter - 1);
                    break;
                }

                let start = Instant::now();
                let pixels = renderer.render_parallel(&scene, &camera, 0);
                let elapsed = start.elapsed();

                // Accumulate
                for (i, p) in pixels.iter().enumerate() {
                    accumulated[i] = accumulated[i] + *p;
                }
                total_samples += samples;

                // Show progress
                let avg: Vec<nebula::math::Vec3> = accumulated.iter()
                    .map(|p| *p / total_samples as f32)
                    .collect();
                
                // Save intermediate result
                let preview_path = format!("preview_{:04}.png", iter);
                if let Err(e) = save_png(std::path::Path::new(&preview_path), &avg, width, height) {
                    eprintln!("Warning: Could not save preview: {}", e);
                }

                println!(
                    "Iteration {}/{}: {} spp total, {:.2}s",
                    iter, iterations, total_samples, elapsed.as_secs_f64()
                );
            }

            // Final save
            let final_pixels: Vec<nebula::math::Vec3> = accumulated.iter()
                .map(|p| *p / total_samples as f32)
                .collect();
            
            let output_path = std::path::Path::new("preview_final.png");
            match save_png(output_path, &final_pixels, width, height) {
                Ok(()) => println!("\nFinal result saved to {}", output_path.display()),
                Err(e) => eprintln!("Error saving final result: {}", e),
            }
        }
    }
}
