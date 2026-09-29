// rng - a fast, deterministic source of uniforms and normals
//
// generateRandom and randPoint are bound by how quickly numbers can be produced, so this is a
// xoshiro256++ stream with Box-Muller rather than <random>. rng.cpp is the one translation unit
// compiled with -ffast-math (the logarithm and the sine vectorize); the rest keeps strict IEEE.
// The output is split into fixed-size chunks, each from its own stream, so the numbers do not
// depend on how many threads filled them.
//
// Syntax:   Rng rng(seed);   rng.uniform(out, n, lo, hi);   rng.normal(out, n, scale);
// See also: contSet/zonotope/generateRandom.cpp

#pragma once

#include <cstddef>
#include <cstdint>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The seed every instance starts from, so a warm daemon behaves like a fresh process.
inline constexpr unsigned long long kSeed = 0;

class Rng {
  public:
    explicit Rng(std::uint64_t seed) : seed_(seed) {}

    /// Restarts the stream, so a warm daemon behaves like a fresh process.
    void reset() { draws_ = 0; }

    /// Fills out[0..n) with U[lo, hi).
    void uniform(double *out, std::size_t n, double lo, double hi);

    /// Fills out[0..n) with scale * N(0, 1).
    void normal(double *out, std::size_t n, double scale);

  private:
    std::uint64_t seed_;
    std::uint64_t draws_ = 0;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
