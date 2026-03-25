//! Geometric primitives

mod hit;
mod sphere;
mod triangle;

pub use hit::{HitRecord, Hittable};
pub use sphere::Sphere;
pub use triangle::Triangle;
