#include "rng.h"

#include <cmath>

namespace cora {
namespace {

// Numbers are drawn in chunks of this size, each from its own stream.
constexpr std::size_t kStream = 1 << 16;

// Below this many numbers a thread fan-out costs more than the work.
constexpr std::size_t kGrain = 1 << 14;

// SplitMix64, to turn a counter into an uncorrelated seed.
std::uint64_t mix(std::uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// xoshiro256++: four multiply-free operations per number, and long enough for any
// instance in the catalog.
struct Xoshiro {
    std::uint64_t s[4];

    explicit Xoshiro(std::uint64_t seed) {
        for (auto &v : s) {
            seed = mix(seed);
            v = seed;
        }
    }

    std::uint64_t next() {
        const std::uint64_t r = rotl(s[0] + s[3], 23) + s[0];
        const std::uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }

    // U(0, 1]: the open end keeps the logarithm of Box–Muller finite.
    double unit() { return static_cast<double>((next() >> 11) + 1) * 0x1p-53; }

    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
};

// Runs `fill(stream, out, count)` over the output, chunk by chunk.
template <typename F>
void chunked(double *out, std::size_t n, std::uint64_t base, F fill) {
    const std::size_t chunks = (n + kStream - 1) / kStream;
#pragma omp parallel for schedule(static) if (n >= kGrain)
    for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(chunks); ++c) {
        const std::size_t start = static_cast<std::size_t>(c) * kStream;
        Xoshiro stream(mix(base ^ mix(static_cast<std::uint64_t>(c))));
        fill(stream, out + start, std::min(kStream, n - start));
    }
}

} // namespace

void Rng::uniform(double *out, std::size_t n, double lo, double hi) {
    const double span = hi - lo;
    chunked(out, n, mix(seed_ ^ mix(++draws_)), [=](Xoshiro &s, double *o, std::size_t k) {
        for (std::size_t i = 0; i < k; ++i) o[i] = s.unit() * span + lo;
    });
}

void Rng::normal(double *out, std::size_t n, double scale) {
    chunked(out, n, mix(seed_ ^ mix(++draws_)), [=](Xoshiro &s, double *o, std::size_t k) {
        // Box–Muller over the whole chunk: the uniforms first, so the transcendentals
        // that follow sit in flat loops the vectorizer can take.
        const std::size_t pairs = k / 2;
        for (std::size_t i = 0; i < pairs; ++i) {
            o[2 * i] = s.unit();
            o[2 * i + 1] = s.unit();
        }
        for (std::size_t i = 0; i < pairs; ++i) {
            const double radius = scale * std::sqrt(-2.0 * std::log(o[2 * i]));
            const double angle = 6.283185307179586 * o[2 * i + 1];
            o[2 * i] = radius * std::cos(angle);
            o[2 * i + 1] = radius * std::sin(angle);
        }
        if (k & 1) {
            const double radius = scale * std::sqrt(-2.0 * std::log(s.unit()));
            o[k - 1] = radius * std::cos(6.283185307179586 * s.unit());
        }
    });
}

} // namespace cora
