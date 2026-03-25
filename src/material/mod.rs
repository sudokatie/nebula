//! Material types

mod scatter;
mod lambertian;
mod metal;
mod dielectric;
mod emissive;

pub use scatter::{Material, ScatterRecord};
pub use lambertian::Lambertian;
pub use metal::Metal;
pub use dielectric::Dielectric;
pub use emissive::Emissive;
