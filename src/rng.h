// A fast, deterministic source of uniforms and normals.
//
// `generateRandom` and `randPoint` are bound by how quickly numbers can be produced, not
// by any product, so this is a xoshiro256++ stream and Box–Muller rather than anything
// from <random>. rng.cpp is the one translation unit compiled with -ffast-math, which
// lets the compiler vectorize the logarithm and the sine through libmvec; the rest of the
// tool keeps strict IEEE arithmetic.
//
// The output is split into fixed-size chunks, each drawn from a stream of its own, so the
// numbers do not depend on how many threads happened to fill them.

#pragma once

#include <cstddef>
#include <cstdint>

namespace cora {

class Rng {
  public:
    explicit Rng(std::uint64_t seed) : seed_(seed) {}

    // Restarts the stream, so a warm daemon behaves like a fresh process.
    void reset() { draws_ = 0; }

    // Fills `out` with U[lo, hi).
    void uniform(double *out, std::size_t n, double lo, double hi);
    // Fills `out` with `scale · N(0, 1)`.
    void normal(double *out, std::size_t n, double scale);

  private:
    std::uint64_t seed_;
    std::uint64_t draws_ = 0;
};

} // namespace cora
