#include "sets.h"

#include "threads.h"

namespace cora {

Interval<double> random_interval(Rng &rng, Eigen::Index n, Eigen::Index batch) {
    Interval<double> s{Mat<double>(n, batch), Mat<double>(n, batch)};
    Mat<double> u(n * batch, 3);
    rng.uniform(u.data(), static_cast<std::size_t>(n) * batch * 3, 0.0, 1.0);
    const Eigen::Index len = n * batch;
#pragma omp parallel for schedule(static) if (len >= (1 << 14))
    for (Eigen::Index i = 0; i < len; ++i) {
        const double c = 4.0 * u.data()[i] - 2.0;
        const double r = 5.0 * u.data()[len + i] * u.data()[2 * len + i];
        s.lo.data()[i] = c - r;
        s.hi.data()[i] = c + r;
    }
    return s;
}

Zonotope<double> random_zonotope(Rng &rng, Eigen::Index n, Eigen::Index m,
                                 Eigen::Index batch) {
    Zonotope<double> s{Mat<double>(n, batch), Mat<double>(n, m * batch), m};
    rng.normal(s.c.data(), static_cast<std::size_t>(n) * batch, 10.0);
    rng.normal(s.g.data(), static_cast<std::size_t>(n) * m * batch, 1.0);
    Eigen::VectorXd length(m * batch);
    rng.uniform(length.data(), static_cast<std::size_t>(m) * batch, 0.0, 1.0);
    const Eigen::Index cols = m * batch;
#pragma omp parallel for schedule(static) if (n * cols >= (1 << 14))
    for (Eigen::Index j = 0; j < cols; ++j) {
        auto column = s.g.col(j);
        column *= length(j) / column.norm();
    }
    return s;
}

Mat<double> rand_point(const Interval<double> &s, Eigen::Index points, Rng &rng) {
    const Eigen::Index n = s.dim();
    Mat<double> out(n, points * s.batch());
    rng.uniform(out.data(), static_cast<std::size_t>(out.size()), 0.0, 1.0);
#pragma omp parallel for schedule(static) if (out.size() >= (1 << 14))
    for (Eigen::Index k = 0; k < out.cols(); ++k) {
        const Eigen::Index b = k / points;
        out.col(k) = s.lo.col(b).array() + out.col(k).array() * (s.hi.col(b) - s.lo.col(b)).array();
    }
    return out;
}

Mat<double> rand_point(const Zonotope<double> &s, Eigen::Index points, Rng &rng) {
    const Eigen::Index n = s.dim(), m = s.m, batch = s.batch();
    Mat<double> beta(m, points * batch);
    rng.uniform(beta.data(), static_cast<std::size_t>(beta.size()), -1.0, 1.0);
    Mat<double> out(n, points * batch);
    // One set per thread while there are enough of them to be worth it; a single product
    // gets all of them, since Eigen leaves its own threading off inside a parallel region.
    const int nt = threads_for(batch, batch * points * n * m);
#pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (Eigen::Index b = 0; b < batch; ++b) {
        out.middleCols(b * points, points).noalias() =
            s.block(b) * beta.middleCols(b * points, points);
        out.middleCols(b * points, points).colwise() += s.c.col(b);
    }
    return out;
}

Mask contains(const Interval<double> &s, const Mat<double> &p, Eigen::Index points) {
    Mask out(static_cast<std::size_t>(p.cols()));
    const Eigen::Index cols = p.cols();
#pragma omp parallel for schedule(static) if (p.size() >= (1 << 14))
    for (Eigen::Index k = 0; k < cols; ++k) {
        const Eigen::Index b = k / points;
        out[static_cast<std::size_t>(k)] =
            (p.col(k).array() >= s.lo.col(b).array()).all()
                    && (p.col(k).array() <= s.hi.col(b).array()).all()
                ? 1
                : 0;
    }
    return out;
}

} // namespace cora
