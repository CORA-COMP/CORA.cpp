#include "contSet/zonotope/zonotope.h"

#include "global/rng.h"

#include <cmath>

namespace cora::ct {

Zonotope Zonotope::generate_random(int64_t n, int64_t m, Rng &rng) {
    std::vector<double> c(n), G(n * m), length(m);
    rng.normal(c.data(), c.size(), 10.0);
    rng.normal(G.data(), G.size(), 1.0);
    rng.uniform(length.data(), length.size(), 0.0, 1.0);
    for (int64_t j = 0; j < m; ++j) {
        double norm = 0.0;
        for (int64_t i = 0; i < n; ++i) norm += G[i * m + j] * G[i * m + j];
        norm = std::sqrt(norm);
        for (int64_t i = 0; i < n; ++i) G[i * m + j] *= length[j] / norm;
    }
    return {Tensor::from_data(c, {n, 1}), Tensor::from_data(G, {n, m})};
}

Tensor Zonotope::support_func(const Tensor &d) const {
    const Tensor dt = d.transpose();
    return dt.matmul(c) + dt.matmul(G).abs().sum_last();
}

Tensor Zonotope::rand_point(int64_t N, Rng &rng, bool extreme) const {
    // One factor per generator and point, with the batch dimensions of G.
    std::vector<int64_t> shape = G.shape();
    shape[shape.size() - 2] = shape.back();
    shape.back() = N;
    int64_t count = 1;
    for (const int64_t d : shape) count *= d;
    std::vector<double> b(count);
    rng.uniform(b.data(), b.size(), -1.0, 1.0);
    if (extreme)
        for (double &v : b) v = v < 0.0 ? -1.0 : 1.0;
    return c + G.matmul(Tensor::from_data(b, shape, c.device()));
}

Interval Zonotope::interval() const {
    const Tensor r = G.abs().sum_last();
    return {c - r, c + r};
}

Zonotope Zonotope::mtimes(const Tensor &M) const { return {M.matmul(c), M.matmul(G)}; }

Zonotope Zonotope::mtimes(const Interval &M) const {
    const Tensor reach = c.abs() + G.abs().sum_last();
    const Tensor extra = M.rad().matmul(reach);
    const Tensor centre = M.center();
    return {centre.matmul(c), Tensor::cat_last({centre.matmul(G), extra.diag()})};
}

Zonotope Zonotope::plus(const Zonotope &Z2) const {
    return {c + Z2.c, Tensor::cat_last({G, Z2.G})};
}

Zonotope Zonotope::lin_comb(const Zonotope &Z2) const {
    return {(c + Z2.c) * 0.5,
            Tensor::cat_last({(G + Z2.G) * 0.5, (G - Z2.G) * 0.5, (c - Z2.c) * 0.5})};
}

} // namespace cora::ct
