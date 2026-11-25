// region_polys - the merge buckets of an activation and its polynomials region by region
//
// getMergeBuckets: the finite values of the activation at -inf and inf, where its output settles
// (a ReLU's 0; a sigmoid's 0 and 1); computeReducedNetwork merges the neurons whose output range
// lies near one. findRegionPolys divides [-inf, inf] into regions, each with a polynomial of the
// regression fit whose error radius is below a tolerance: it walks from l_max up to u_max by
// halving the region's upper end until the order-k fit is accurate enough (trying lower orders on
// the region found, down to a constant), then from u_max down by halving the lower end, and
// closes the two outer regions with the given polynomials. No progress is printed (CORA prints
// percentages).
//
// Syntax:   region_polys() - the buckets and regions, e.g.
//           buckets = NeuralNetwork::merge_buckets(Activation::sigmoid);
//           regions = NeuralNetwork::find_region_polys(a, tol, order, l_max, u_max, ...);
// Inputs:   activation - the layer's function
//           tol - the error below which a region's polynomial is taken
//           order - the order of the polynomials, at least 1
//           l_max, u_max - the domain between the outer regions
//           pS, dS, pE, dE - the polynomials (highest power first) and radii of the outer regions
// Outputs:  buckets - the finite limits
//           regions - sorted by their lower end, each with p and d
// See also: cora/nn/neural_network/neural_network.h, priv_activation_poly.h
// References: none (CORA's methods, no paper's algorithm)
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/getMergeBuckets.m
// CORA: nn/layers/nonlinear/nnReLULayer.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/findRegionPolys.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       23-November-2026
// Last update:   25-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "nn/neural_network/private/priv_activation_poly.h"
#include "nn/neural_network/private/priv_layers.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

/// The polynomial of the regression fit of the order over [l, u], with CORA's error.
PrivApprox aux_fit(Activation a, double l, double u, int order) {
    return priv_approx_poly(a, l, u, order, PolyMethod::regression);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The merge buckets -------------------------------------------------------------------------------

CORA_FROM_NUMBERS std::vector<double> NeuralNetwork::merge_buckets(Activation activation) {
    detail::Entry entry;
    // the activation at the two infinities (NaN, as sqrt(-inf) is for MATLAB's complex inf, too)
    const Tensor limits = Tensor::from_data({-kInfinity, kInfinity}, {2});
    std::vector<double> buckets;
    for (const double v : priv_apply(activation, limits).data())
        if (std::isfinite(v)) buckets.push_back(v);
    return buckets;
}

// The polynomials of the regions ------------------------------------------------------------------

std::vector<RegionPoly>
NeuralNetwork::find_region_polys(Activation activation, double tol, int order, double l_max,
                                 double u_max, const std::vector<double> &p_start, double d_start,
                                 const std::vector<double> &p_end, double d_end) {
    detail::Entry entry;
    CORA_CHECK(tol > 0 && order >= 1 && l_max < u_max && d_start >= 0 && d_end >= 0 &&
                   !p_start.empty() && !p_end.empty(),
               Error::wrong_value, [&] {
                   return "find_region_polys: tol > 0, order >= 1, l_max < u_max, radii >= 0 and "
                          "outer polynomials, got tol = " +
                          std::to_string(tol) + ", order = " + std::to_string(order);
               });
    if (d_start > tol) warn("CORA:nn", "findRegionPolys: dStart > tol.");
    if (d_end > tol) warn("CORA:nn", "findRegionPolys: dEnd > tol.");
    std::vector<RegionPoly> regions{{-kInfinity, l_max, p_start, d_start}};
    double l = l_max, u = u_max;
    while (l < u) {
        // from below: the upper end is halved until the fit of the order is within tol
        double u_i = u, d = kInfinity;
        std::vector<double> p{0.0, 0.0};
        while (d > tol) {
            if (std::isfinite(d)) {
                u_i = l + (u_i - l) * 0.5;
                if (u_i - l < tol)
                    CORA_FAIL(Error::not_converged,
                              "find_region_polys: no polynomial at [" + std::to_string(l) + ", " +
                                  std::to_string(u) + "], reached " + std::to_string(u_i));
            }
            const PrivApprox fit = aux_fit(activation, l, u_i, order);
            p = fit.coeffs;
            d = fit.d;
        }
        // a lower order on the region, down to a constant, while it stays within tol (CORA's)
        for (int o = order - 1; o >= 0; --o) {
            const PrivApprox lower = aux_fit(activation, l, u_i, o);
            if (lower.d >= tol) break;
            p = lower.coeffs;
            d = lower.d;
        }
        regions.push_back({l, u_i, p, d});
        l = u_i;
        if (l == u) break;
        // from above: the lower end is halved the same way
        double l_i = l;
        d = kInfinity;
        p = {0.0, 0.0};
        while (d > tol) {
            if (std::isfinite(d)) {
                l_i = u - (u - l_i) / 2;
                if (u - l_i < tol)
                    CORA_FAIL(Error::not_converged,
                              "find_region_polys: no polynomial at [" + std::to_string(l) + ", " +
                                  std::to_string(u) + "], reached " + std::to_string(l_i));
            }
            // the polynomial, then its error again (CORA calls computeApproxError on it)
            const PrivApprox fit =
                priv_approx_error(activation, aux_fit(activation, l_i, u, order).coeffs, l_i, u);
            p = fit.coeffs;
            d = fit.d;
        }
        regions.push_back({l_i, u, p, d});
        u = l_i;
    }
    regions.push_back({u_max, kInfinity, p_end, d_end});
    std::sort(regions.begin(), regions.end(),
              [](const RegionPoly &a, const RegionPoly &b) { return a.l < b.l; });
    return regions;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
