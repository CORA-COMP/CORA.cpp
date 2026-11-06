// hausdorff_distance_to_box - a bound on the Hausdorff distance between a zonotope and its box
//
// "exact": the largest gap of the support functions of the box and of Z over about 10000
// directions on a sphere grid; "naive" (the default above): the norm of the radius of the box;
// "wgreedy" and "wopt": bounds from the generators; "ell" is not supported.
//
// Syntax:   dH = Z.hausdorff_distance_to_box();   dH = Z.hausdorff_distance_to_box("wopt");
// Inputs:   method - "exact" (CORA's default for dim <= 3), "naive", "wgreedy" or "wopt"
// Outputs:  dH - the number, (..., 1, 1); 2-norm for "naive", "wgreedy" and "wopt"
// See also: interval, box
//
// CORA: contSet/@zonotope/dH2box.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr int kDirections = 10000;
constexpr double kPi = 3.14159265358979323846;

// Directions --------------------------------------------------------------------------------------

/// The directions of CORA's randEqdistDirections(n, sections) as columns (n, sections^(n-1)):
/// the first axis rotated in the planes (i, i+1) by multiples of pi / (sections + 1).
std::vector<double> aux_directions(int64_t n, int64_t sections) {
    std::vector<std::vector<double>> dirs{std::vector<double>(n, 0.0)};
    dirs[0][0] = 1.0;
    const double increment = kPi / double(sections + 1);
    for (int64_t axis = 0; axis + 1 < n; ++axis) {
        std::vector<std::vector<double>> rotated;
        for (int64_t k = 1; k <= sections; ++k) {
            const double c = std::cos(double(k) * increment), s = std::sin(double(k) * increment);
            for (const std::vector<double> &d : dirs) {
                std::vector<double> r = d;
                r[axis] = c * d[axis] - s * d[axis + 1];
                r[axis + 1] = s * d[axis] + c * d[axis + 1];
                rotated.push_back(r);
            }
        }
        dirs = rotated;
    }
    std::vector<double> flat;
    for (const std::vector<double> &d : dirs) flat.insert(flat.end(), d.begin(), d.end());
    // the directions as one flat list
    return flat;
}

/// The largest | |l' Gbox|_1 - |l' G|_1 | over the grid of directions l (CORA's aux_dH2box_exact).
Tensor aux_exact(const Tensor &G) {
    const int64_t n = G.shape()[G.ndim() - 2];
    if (n == 1) return G.abs_sum().zeros_like();
    // nthroot(10000, n - 1), cut to an integer; at least one section
    const double root = std::pow(double(kDirections), 1.0 / double(n - 1)) + numerics::kRootFloor;
    const int64_t sections = std::max<int64_t>(int64_t(std::floor(root)), 1);
    const std::vector<double> flat = aux_directions(n, sections);
    const int64_t count = int64_t(flat.size()) / n;
    const Tensor L = Tensor::like(G, flat, {count, n});
    const Tensor box_norm = L.abs().matmul(G.abs_sum()); // |l' diag(r)|_1 for r the row sums of |G|
    const Tensor norm = L.matmul(G).abs_sum();           // |l' G|_1
    return (box_norm - norm).abs().max(-2, true);
}

// Bounds from the generators ----------------------------------------------------------------------

/// 2 |sum_k |G_.k| without the largest entry of each generator|_2 (CORA's aux_dH2box_wgreedy).
Tensor aux_wgreedy(const Tensor &G) {
    const int64_t n = G.shape()[G.ndim() - 2];
    const Tensor magnitude = G.abs();
    const Tensor largest = magnitude.detach().argmax(-2, true);
    const Tensor rows = G.arange_like(n).reshape({n, 1});
    const Tensor truncated = Tensor::where(rows.equal(largest), magnitude.zeros_like(), magnitude);
    return truncated.sum(-1, true).vector_norm(Tensor::Norm::two, -2, true) * 2.0;
}

/// 2 sqrt(sum_i (sum_k |G_ik| (1 - G_ik^2 / |g_k|^2))^2) (CORA's aux_dH2box_wopt).
Tensor aux_wopt(const Tensor &G) {
    const Tensor lsquared = G.vector_norm(Tensor::Norm::two, -2, true).square();
    const Tensor zero = lsquared.detach().equal(lsquared.zeros_like());
    const Tensor safe = Tensor::where(zero, lsquared.ones_like(), lsquared);
    const Tensor terms = G.abs().multiply(G.square().divide(safe).negative() + 1.0);
    const Tensor summand = Tensor::where(zero, terms.zeros_like(), terms).sum(-1, true);
    return summand.square().sum(-2, true).sqrt() * 2.0;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::hausdorff_distance_to_box(const std::optional<std::string> &chosen) const {
    detail::Entry entry;
    const std::string method = chosen.value_or(dim() <= 3 ? "exact" : "naive");
    CORA_CHECK(method == "exact" || method == "naive" || method == "ell" || method == "wgreedy" ||
                   method == "wopt",
               Error::wrong_value, [&] {
                   return "hausdorff_distance_to_box: the method must be 'exact', 'naive', "
                          "'ell', 'wgreedy' or 'wopt', got '" +
                          method + "'";
               });
    CORA_CHECK(method != "ell", Error::not_supported, [] {
        return std::string("hausdorff_distance_to_box: method 'ell' (a row reduced echelon form "
                           "for the parallelepipeds) is not supported");
    });
    if (method == "exact") return aux_exact(G_);
    if (method == "naive") return interval().rad().vector_norm(Tensor::Norm::two, -2, true);
    if (method == "wgreedy") return aux_wgreedy(G_);
    if (method == "wopt") return aux_wopt(G_);
    CORA_FAIL(Error::report_to_dev, "hausdorff_distance_to_box: a method without a branch");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
