use clap::{Parser, Subcommand};
use std::path::PathBuf;

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

        /// Image width
        #[arg(short = 'W', long, default_value_t = 800)]
        width: u32,

        /// Image height
        #[arg(short = 'H', long, default_value_t = 600)]
        height: u32,

        /// Samples per pixel
        #[arg(short, long, default_value_t = 100)]
        samples: u32,

        /// Maximum ray depth
        #[arg(short, long, default_value_t = 50)]
        depth: u32,

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
            scene,
            output,
            width,
            height,
            samples,
            depth,
            threads,
            gpu,
        } => {
            println!("Rendering scene: {}", scene.display());
            println!("Output: {}", output.display());
            println!("Resolution: {}x{}", width, height);
            println!("Samples: {}, Depth: {}", samples, depth);
            println!("Threads: {}", if threads == 0 { "auto".to_string() } else { threads.to_string() });
            if gpu {
                println!("GPU rendering enabled");
            }

            // TODO: Load scene and render
            println!("Rendering not yet implemented");
        }
        Commands::Preview { scene, width, height } => {
            println!("Preview mode: {} ({}x{})", scene.display(), width, height);
            println!("Preview not yet implemented");
        }
    }
}
