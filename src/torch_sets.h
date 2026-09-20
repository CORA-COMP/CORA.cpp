// The set representations, as tensors.
//
// The libtorch counterpart of sets.h, and a different layout: every set carries leading
// batch dimensions that may be empty, an `Interval` holding `lo`/`hi` of shape `(..., n)`
// and a `Zonotope` a centre `(..., n)` and generators `(..., n, m)`. The operations touch
// only the trailing dimensions, so a batch of sets is one call into libtorch rather than
// a loop over sets, and nothing here names a device. Point clouds are `(..., N, n)`, one
// point per row, the layout batched matmul wants.
//
// Everything is an ordinary tensor expression, so autograd differentiates the operations
// with respect to a set whose `c`, `G`, `lo` or `hi` requires a gradient — on either
// device, and without a second implementation.

#pragma once

#include <torch/torch.h>

namespace cora::tb {

/// Samples of `U[low, high)` with the dtype and device of `like`.
inline torch::Tensor uniform(torch::IntArrayRef shape, double low, double high,
                             const torch::Tensor &like) {
    return torch::rand(shape, like.options()) * (high - low) + low;
}

/// An axis-aligned box, `{x | lo <= x <= hi}` elementwise.
struct Interval {
    torch::Tensor lo, hi;

    torch::Tensor center() const { return (hi + lo) / 2.0; }
    torch::Tensor radius() const { return (hi - lo) / 2.0; }

    /// CORA's `interval.generateRandom`: centre `U[-2, 2]`, radius `R/2 · u` with
    /// `R ~ U[0, 10]` and `u ~ U[0, 1]`.
    static Interval generate_random(torch::IntArrayRef batch, int64_t n,
                                    const torch::TensorOptions &opts) {
        std::vector<int64_t> shape{3};
        shape.insert(shape.end(), batch.begin(), batch.end());
        shape.push_back(n);
        const torch::Tensor u = torch::rand(shape, opts);
        const torch::Tensor c = u[0] * 4.0 - 2.0;
        const torch::Tensor r = u[1] * u[2] * 5.0;
        return {c - r, c + r};
    }

    /// `points` points per set, uniform in the box.
    torch::Tensor rand_point(int64_t points) const {
        std::vector<int64_t> shape = lo.sizes().vec();
        const int64_t n = shape.back();
        shape.back() = points;
        shape.push_back(n);
        return lo.unsqueeze(-2) + uniform(shape, 0.0, 1.0, lo) * (hi - lo).unsqueeze(-2);
    }

    /// `max_{x in self} d'x` for one direction `d` of shape `(..., n)` per set.
    torch::Tensor support_func(const torch::Tensor &d) const {
        return (center() * d).sum(-1) + (radius() * d.abs()).sum(-1);
    }

    /// `M · self` as CORA computes it, the interval hull: centres map by `M`, radii by
    /// `|M|`.
    Interval mat_mul(const torch::Tensor &m) const {
        const torch::Tensor c = center().matmul(m.transpose(-2, -1));
        const torch::Tensor r = radius().matmul(m.abs().transpose(-2, -1));
        return {c - r, c + r};
    }

    Interval mink_sum(const Interval &other) const { return {lo + other.lo, hi + other.hi}; }

    /// Whether each point of `p` `(..., N, n)` lies in its set: a `(..., N)` mask.
    torch::Tensor contains(const torch::Tensor &p) const {
        return torch::logical_and(p >= lo.unsqueeze(-2), p <= hi.unsqueeze(-2)).all(-1);
    }
};

/// `{c + G b | ||b||_inf <= 1}`, one generator per column of `G`.
struct Zonotope {
    torch::Tensor c, g;

    int64_t dim() const { return c.size(-1); }
    int64_t generators() const { return g.size(-1); }

    /// The unit zonotope at the origin: the `startup` instance.
    static Zonotope origin(int64_t n, const torch::TensorOptions &opts) {
        return {torch::zeros({n}, opts), torch::eye(n, opts)};
    }

    /// CORA's `zonotope.generateRandom`: centre `10 · randn`, `m` generators, each a
    /// uniformly random unit direction with a length `~ U[0, 1]`.
    static Zonotope generate_random(torch::IntArrayRef batch, int64_t n, int64_t m,
                                    const torch::TensorOptions &opts) {
        std::vector<int64_t> shape = batch.vec();
        shape.push_back(n);
        const torch::Tensor c = torch::randn(shape, opts) * 10.0;
        shape.push_back(m);
        const torch::Tensor g = torch::randn(shape, opts);
        std::vector<int64_t> lengths = batch.vec();
        lengths.push_back(1);
        lengths.push_back(m);
        const torch::Tensor length = torch::rand(lengths, opts);
        const torch::Tensor norm = g.norm(2, {-2}, true);
        return {c, g * (length / norm)};
    }

    /// `points` points per set, CORA's `standard`: `c + G b` with `b ~ U[-1, 1]^m`.
    torch::Tensor rand_point(int64_t points) const {
        std::vector<int64_t> shape = c.sizes().vec();
        shape.back() = points;
        shape.push_back(generators());
        const torch::Tensor beta = uniform(shape, -1.0, 1.0, c);
        return c.unsqueeze(-2) + beta.matmul(g.transpose(-2, -1));
    }

    /// `max_{x in self} d'x` for one direction `d` of shape `(..., n)` per set.
    torch::Tensor support_func(const torch::Tensor &d) const {
        return (c * d).sum(-1) + d.unsqueeze(-2).matmul(g).abs().sum({-2, -1});
    }

    Zonotope mat_mul(const torch::Tensor &m) const {
        return {c.matmul(m.transpose(-2, -1)), m.matmul(g)};
    }

    Zonotope mink_sum(const Zonotope &other) const {
        return {c + other.c, torch::cat({g, other.g}, -1)};
    }

    /// Whether each point of `p` `(..., N, n)` lies in its set: a `(..., N)` mask.
    ///
    /// Exact either way: through the facets while there are few of them, otherwise by
    /// certified alternating projections with an LP for the points they leave open.
    torch::Tensor contains(const torch::Tensor &p) const;
};

} // namespace cora::tb
