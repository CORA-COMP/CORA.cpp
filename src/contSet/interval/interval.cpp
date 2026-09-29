#include "contSet/interval/interval.h"

#include "global/rng.h"

#include <algorithm>

namespace cora::ct {

Interval Interval::generate_random(int64_t n, Rng &rng) {
    std::vector<double> c(n), r(n), u(n);
    rng.uniform(c.data(), c.size(), -2.0, 2.0);
    rng.uniform(r.data(), r.size(), 0.0, 10.0);
    rng.uniform(u.data(), u.size(), 0.0, 1.0);
    for (int64_t i = 0; i < n; ++i) r[i] *= 0.5 * u[i];
    std::vector<double> lo(n), hi(n);
    for (int64_t i = 0; i < n; ++i) {
        lo[i] = c[i] - r[i];
        hi[i] = c[i] + r[i];
    }
    return {Tensor::from_data(lo, {n, 1}), Tensor::from_data(hi, {n, 1})};
}

Tensor Interval::support_func(const Tensor &d) const {
    return d.transpose().matmul(center()) + d.abs().transpose().matmul(rad());
}

Interval Interval::mtimes(const Tensor &M) const {
    const Tensor c = M.matmul(center()), r = M.abs().matmul(rad());
    return {c - r, c + r};
}

bool Interval::contains(const Tensor &p) const {
    // How far the point is outside the box, summed over dimensions.
    const Tensor outside = ((inf - p).pos() + (p - sup).pos()).transpose().sum_last();
    for (const double v : outside.data())
        if (v > 1e-12) return false;
    return true;
}

} // namespace cora::ct
