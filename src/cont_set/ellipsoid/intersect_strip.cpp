// intersect_strip - an ellipsoid that contains the intersection of an ellipsoid and a strip
//
// Syntax:   auto [E2, sigma_sq] = E.intersect_strip(C, phi, y, sigma_sq_prev);
//           auto [E2, sigma_sq] = E.intersect_strip(C, sys, y, sigma_sq_prev);
// Inputs:   C - the strip's matrix (m, n)
//           y - its center (m, 1)
//           phi - the half-width of the
//           strip ||C x - y|| <= phi (method of Gollamudi 1996), or sys - the measurement model
//           with V and bar_V (m, m) (method of Liu 2016);  sigma_sq_prev - the level of the
//           previous update (default 0). A single full-dimensional ellipsoid
// Outputs:  E2 - an ellipsoid with the center and the information matrix of the combination
//           (1 - lambda) E + lambda strip, scaled to its level sigma_sq
//           sigma_sq - that level.
//           E and sigma_sq_prev are returned if the strip does not cut E or the weight lambda
//           leaves [0, 1) (Gollamudi) or (0, 1) (Liu); the empty ellipsoid if sigma_sq < 0. CORA's
//           shapes are too small (Gollamudi: divided by sigma_sq instead of multiplied, and the
//           center moved with the old Q; Liu: by the factor (1 - lambda)^2), so the result can miss
//           points of the intersection; the corrected shapes are used (docs/cora-differences.md)
// Gradients: not differentiable (the weight and the level are host numbers)
// See also: Zonotope::intersect_strip, ConZonotope::intersect_strip
//
// CORA: contSet/@ellipsoid/intersectStrip.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Updates -----------------------------------------------------------------------------------------

/// The data of an update: the innovation delta = y - C q and its squared norm.
struct Innovation {
    Tensor delta;
    double squared;
};

Innovation aux_innovation(const Ellipsoid &E, const Tensor &C, const Tensor &y) {
    const Tensor delta = y - C.matmul(E.q());
    return {delta, delta.matrix_transpose().matmul(delta).item()};
}

/// The largest singular value of the positive semidefinite matrix G.
double aux_largest(const Tensor &G) { return G.svd()[1].max().item(); }

/// delta' M delta for a matrix M.
double aux_quadratic(const Tensor &delta, const Tensor &M) {
    return delta.matrix_transpose().matmul(M).matmul(delta).item();
}

/// The ellipsoid with the shape Q (symmetrized) and the center c, or the empty one for a negative
/// level, where the combination has no point.
Ellipsoid aux_result(const Ellipsoid &E, const Tensor &Q, const Tensor &c, double sigma_sq) {
    if (sigma_sq < 0.0) return Ellipsoid::empty(E.dim());
    return {(Q + Q.matrix_transpose()) * 0.5, c};
}

/// The update of Gollamudi (1996), Theorems 1 and 2, with the corrected shape and center.
Ellipsoid::StripResult aux_gollamudi(const Ellipsoid &E, const Tensor &C, double phi,
                                     const Tensor &y, double sigma_sq_prev) {
    const auto [delta, squared] = aux_innovation(E, C, y);
    if (!(sigma_sq_prev + squared > phi)) return {E, sigma_sq_prev};
    const double eps = machine_eps(E.Q().dtype());
    const Tensor G = C.matmul(E.Q()).matmul(C.matrix_transpose());
    const double g = aux_largest(G);
    // nu of Theorem 2; the order matters: delta = 0 first
    double nu = numerics::kStripWeight;
    if (!within_tol(delta, delta.zeros_like(), eps).min().item()) {
        const double beta = (phi - sigma_sq_prev) / squared;
        const double omega = 1.0 + beta * (g - 1.0);
        if (within_tol(g, 1.0, eps))
            nu = (1.0 - beta) / 2.0;
        else if (omega > 0.0)
            nu = 1.0 / (1.0 - g) * (1.0 - std::sqrt(g / omega));
    }
    const double lambda = std::min(numerics::kStripWeight, nu);
    // the combination needs a weight in [0, 1); NaN fails the test
    if (!(lambda >= 0.0 && lambda < 1.0)) return {E, sigma_sq_prev};
    const Tensor weighted = G.eye_like(G.shape().back()) * (1.0 - lambda) + G * lambda;
    const double sigma_sq = 1.0 - lambda + lambda * phi * phi -
                            lambda * (1.0 - lambda) * aux_quadratic(delta, weighted.inv());
    const Tensor information =
        E.Q().inv() * (1.0 - lambda) + C.matrix_transpose().matmul(C) * lambda;
    const Tensor shape = information.inv();
    const Tensor center = E.q() + shape.matmul(C.matrix_transpose()).matmul(delta) * lambda;
    return {aux_result(E, shape * sigma_sq, center, sigma_sq), sigma_sq};
}

/// The update of Liu et al. (2016), Theorem 3, with the corrected shape.
Ellipsoid::StripResult aux_liu(const Ellipsoid &E, const Tensor &C,
                               const Ellipsoid::StripMeasurement &sys, const Tensor &y,
                               double sigma_sq_prev) {
    const auto [delta, squared] = aux_innovation(E, C, y);
    if (!(sigma_sq_prev + squared > 1.0)) return {E, sigma_sq_prev};
    const double eps = machine_eps(E.Q().dtype());
    const Tensor CQC = C.matmul(E.Q()).matmul(C.matrix_transpose());
    const double g = aux_largest(sys.bar_V.matmul(CQC).matmul(sys.bar_V.matrix_transpose()));
    const double beta = (1.0 - sigma_sq_prev) / squared;
    const double omega = 1.0 + beta * (g - 1.0);
    const double lambda = within_tol(g, 1.0, eps) ? (1.0 - beta) / 2.0
                                                  : 1.0 / (1.0 - g) * (1.0 - std::sqrt(g / omega));
    // the combination needs a weight in (0, 1); NaN fails the test
    if (!(lambda > 0.0 && lambda < 1.0)) return {E, sigma_sq_prev};
    const Tensor weighted_inv = (sys.V / lambda + CQC / (1.0 - lambda)).inv();
    const Tensor K = E.Q().matmul(C.matrix_transpose()).matmul(weighted_inv) / (1.0 - lambda);
    const double sigma_sq =
        (1.0 - lambda) * sigma_sq_prev + lambda - aux_quadratic(delta, weighted_inv);
    const Tensor identity = E.Q().eye_like(E.dim());
    const Tensor shape = (identity - K.matmul(C)).matmul(E.Q()) * (sigma_sq / (1.0 - lambda));
    return {aux_result(E, shape, E.q() + K.matmul(delta), sigma_sq), sigma_sq};
}

/// Why the ellipsoid, the strip or the number is refused, if it is.
detail::Refusal aux_refusal(const Ellipsoid &E, const Tensor &C, const Tensor &y,
                            double sigma_sq_prev) {
    const int64_t n = E.dim();
    if (priv_is_empty(E))
        return std::pair{Error::empty_set, std::string("intersect_strip: E is empty")};
    if (E.Q().ndim() != 2)
        return std::pair{Error::not_supported,
                         std::string("intersect_strip: it updates one ellipsoid, not a batch")};
    if (!E.is_full_dim())
        return std::pair{Error::degenerate_set,
                         std::string("intersect_strip: the ellipsoid must be full-dimensional")};
    if (!C.defined() || C.ndim() != 2 || C.shape()[1] != n)
        return std::pair{Error::dimension_mismatch,
                         "intersect_strip: C must be a matrix (m, " + std::to_string(n) + ")"};
    if (!y.defined() || y.ndim() != 2 || !is_column(y, C.shape()[0]))
        return std::pair{Error::dimension_mismatch, "intersect_strip: y must be a column (" +
                                                        std::to_string(C.shape()[0]) +
                                                        ", 1), one center per row of C"};
    if (!std::isfinite(sigma_sq_prev))
        return std::pair{Error::wrong_value,
                         std::string("intersect_strip: sigma_sq_prev must be finite")};
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Entries -----------------------------------------------------------------------------------------

Ellipsoid::StripResult Ellipsoid::intersect_strip(const Tensor &C, double phi, const Tensor &y,
                                                  double sigma_sq_prev) const {
    detail::Entry entry;
    detail::Refusal refusal = aux_refusal(*this, C, y, sigma_sq_prev);
    if (!refusal && !std::isfinite(phi))
        refusal = std::pair{Error::wrong_value, std::string("intersect_strip: phi must be finite")};
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_gollamudi(*this, C, phi, y, sigma_sq_prev);
}

Ellipsoid::StripResult Ellipsoid::intersect_strip(const Tensor &C, const StripMeasurement &sys,
                                                  const Tensor &y, double sigma_sq_prev) const {
    detail::Entry entry;
    // the measurement model is checked with the strip
    detail::Refusal refusal = aux_refusal(*this, C, y, sigma_sq_prev);
    const int64_t m = C.defined() && C.ndim() == 2 ? C.shape()[0] : 0;
    const auto is_square = [&](const Tensor &M) {
        return M.defined() && M.ndim() == 2 && M.shape()[0] == m && M.shape()[1] == m;
    };
    if (!refusal && !(is_square(sys.V) && is_square(sys.bar_V)))
        refusal =
            std::pair{Error::wrong_value, "intersect_strip: V and bar_V must be matrices (" +
                                              std::to_string(m) + ", " + std::to_string(m) + ")"};
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_liu(*this, C, sys, y, sigma_sq_prev);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
