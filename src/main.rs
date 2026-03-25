use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

use nebula::scene::load_scene;
use nebula::render::CpuRenderer;
use nebula::output::{save_png, save_ppm};

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

            if gpu {
                eprintln!("GPU rendering not yet implemented");
                std::process::exit(1);
            }

            // Render
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
        Commands::Preview { scene, width, height } => {
            println!("Preview mode: {} ({}x{})", scene.display(), width, height);
            println!("Preview not yet implemented");
        }
    }
}
