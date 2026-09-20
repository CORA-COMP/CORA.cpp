// Points placed where an approximate containment answer breaks.
//
// Shared by the Eigen and the libtorch tests so that both are judged on the same points:
// on a facet, just past one, and scattered across the boundary where only the LP knows
// the answer.

#pragma once

#include "rng.h"
#include "sets.h"

namespace cora {

/// `points` points per set, each on a facet and pulled just inside it: `c + G(0.999 b)`
/// with `b`'s largest coordinate saturated.
inline Mat<double> just_inside(const Zonotope<double> &z, Eigen::Index points, Rng &rng) {
    const Eigen::Index n = z.dim(), m = z.m, batch = z.batch();
    Mat<double> betas(m, points * batch), out(n, points * batch);
    rng.uniform(betas.data(), static_cast<std::size_t>(betas.size()), -1.0, 1.0);
    for (Eigen::Index b = 0; b < batch; ++b) {
        for (Eigen::Index k = 0; k < points; ++k) {
            const Eigen::Index col = b * points + k;
            Eigen::VectorXd beta = betas.col(col);
            Eigen::Index top = 0;
            beta.cwiseAbs().maxCoeff(&top);
            beta(top) = beta(top) >= 0 ? 1.0 : -1.0;
            out.col(col) = z.c.col(b) + z.block(b) * (0.999 * beta);
        }
    }
    return out;
}

/// `points` points per set, each just past the support in a random direction.
inline Mat<double> just_outside(const Zonotope<double> &z, Eigen::Index points, Rng &rng) {
    const Eigen::Index n = z.dim(), batch = z.batch();
    Mat<double> dirs(n, points * batch), out(n, points * batch);
    rng.normal(dirs.data(), static_cast<std::size_t>(dirs.size()), 1.0);
    for (Eigen::Index b = 0; b < batch; ++b) {
        for (Eigen::Index k = 0; k < points; ++k) {
            const Eigen::Index col = b * points + k;
            const Eigen::VectorXd d = dirs.col(col);
            const double reach = (z.block(b).transpose() * d).array().abs().sum();
            out.col(col) = z.c.col(b) + 1.001 * reach / d.squaredNorm() * d;
        }
    }
    return out;
}

/// `points` points per set scattered across the boundary, so roughly half land inside:
/// a corner of the generator box, jittered. Two implementations that disagree at all
/// disagree here.
inline Mat<double> across_the_boundary(const Zonotope<double> &z, Eigen::Index points,
                                       Rng &rng) {
    const Eigen::Index n = z.dim(), batch = z.batch();
    Mat<double> jitter(n, points * batch), offset(n, batch), out(n, points * batch);
    rng.normal(jitter.data(), static_cast<std::size_t>(jitter.size()), 0.3);
    rng.uniform(offset.data(), static_cast<std::size_t>(offset.size()), -1.0, 1.0);
    for (Eigen::Index b = 0; b < batch; ++b) {
        const Eigen::VectorXd reach = z.block(b).cwiseAbs().rowwise().sum();
        const Eigen::VectorXd corner = z.c.col(b) + reach.cwiseProduct(offset.col(b));
        for (Eigen::Index k = 0; k < points; ++k)
            out.col(b * points + k) = corner + jitter.col(b * points + k).cwiseProduct(reach);
    }
    return out;
}

} // namespace cora
