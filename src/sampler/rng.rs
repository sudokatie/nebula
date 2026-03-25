//! Random number generation

use rand::{SeedableRng, Rng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Per-thread sampler
pub struct Sampler {
    rng: Xoshiro256PlusPlus,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    pub fn random(&mut self) -> f32 {
        self.rng.gen()
    }

    pub fn random_range(&mut self, min: f32, max: f32) -> f32 {
        self.rng.gen_range(min..max)
    }

    pub fn inner_mut(&mut self) -> &mut Xoshiro256PlusPlus {
        &mut self.rng
    }
}
