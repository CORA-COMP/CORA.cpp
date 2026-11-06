// zonotope - a zonotope that contains a constrained zonotope, as CORA's conZonotope.zonotope
//
// "nullSpace" solves the constraints for a particular solution p and a basis T of the null space of
// A: the factors are p + T y, and y ranges over a box found by a linear program per bound; the
// zonotope is the image of that box. "reduce" reduces the set to about n + 1 generators per
// dimension (Girard) and drops the constraints. A set without constraints is its zonotope; an empty
// set is Error::empty_set (CORA.cpp has no empty zonotope). Not a batch.
//
// Syntax:   Zonotope Z = cz.zonotope();   Zonotope Z = cz.zonotope("reduce");
// Inputs:   method - "nullSpace" (default) or "reduce"
// Outputs:  Z - a zonotope that contains the set
// See also: zono_bundle, polytope
//
// CORA: contSet/@conZonotope/zonotope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/linprog.h"
#include "cora/global/tensor/tolerance.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The range [lb, ub] of y_i for every i over the null space coordinates y of the factors p + T y
/// with |p + T y|_inf <= 1; the rows are the bounds -T y <= 1 + p and T y <= 1 - p.
std::pair<std::vector<double>, std::vector<double>> aux_ranges(const Tensor &T, const Tensor &p) {
    // the rows [T; -T] y <= [1 - p; 1 + p] keep every factor in [-1, 1]
    const int64_t m = T.shape()[0], k = T.shape()[1];
    const std::vector<double> t = T.data(), q = p.data();
    LinProg lp;
    lp.n = int(k);
    lp.Aineq.resize(std::size_t(2 * m * k));
    lp.bineq.resize(std::size_t(2 * m));
    for (int64_t i = 0; i < m; ++i) {
        lp.bineq[std::size_t(i)] = 1.0 - q[std::size_t(i)];
        lp.bineq[std::size_t(m + i)] = 1.0 + q[std::size_t(i)];
        for (int64_t j = 0; j < k; ++j) {
            lp.Aineq[std::size_t(i * k + j)] = t[std::size_t(i * k + j)];
            lp.Aineq[std::size_t((m + i) * k + j)] = -t[std::size_t(i * k + j)];
        }
    }
    // each bound is a linear program: the least and the largest value of y_j
    std::vector<double> lb(static_cast<std::size_t>(k)), ub(static_cast<std::size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
        lp.f.assign(std::size_t(k), 0.0);
        lp.f[std::size_t(j)] = 1.0;
        lb[std::size_t(j)] = linprog(lp).fval;
        lp.f[std::size_t(j)] = -1.0;
        ub[std::size_t(j)] = -linprog(lp).fval;
    }
    return {lb, ub};
}

/// The zonotope of the null space method: the factors p + T y with y in a box.
Zonotope aux_null_space(const ConZonotope &cz) {
    const Tensor &c = cz.c(), &G = cz.generators();
    const Tensor p = cz.A().pinv().matmul(cz.b()), T = priv_null_space(cz.A());
    const int64_t k = T.shape()[1];
    const auto [lb, ub] = aux_ranges(T, p);
    // a box of one point (up to rounding) is that point, as CORA's relative spread test
    double spread = 0.0, size = 0.0;
    std::vector<double> mid(static_cast<std::size_t>(k)), rad(static_cast<std::size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
        spread +=
            (ub[std::size_t(j)] - lb[std::size_t(j)]) * (ub[std::size_t(j)] - lb[std::size_t(j)]);
        size += ub[std::size_t(j)] * ub[std::size_t(j)];
    }
    const bool point =
        std::sqrt(spread) / (size == 0.0 ? 1.0 : std::sqrt(size)) < numerics::kNullSpaceSpread;
    for (int64_t j = 0; j < k; ++j) {
        mid[std::size_t(j)] = 0.5 * (ub[std::size_t(j)] + lb[std::size_t(j)]);
        rad[std::size_t(j)] = point ? 0.0 : 0.5 * (ub[std::size_t(j)] - lb[std::size_t(j)]);
    }
    const Tensor center = Tensor::like(c, mid, {k, 1}), radius = Tensor::like(c, rad, {1, k});
    const Tensor offset = p + T.matmul(center);
    return {c + G.matmul(offset), G.matmul(T.multiply(radius))};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope ConZonotope::zonotope(const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(method == "nullSpace" || method == "reduce", Error::wrong_value, [&] {
        return "zonotope: the method must be 'nullSpace' or 'reduce', got '" + method + "'";
    });
    CORA_CHECK(c_.ndim() == 2, Error::not_supported,
               [] { return std::string("zonotope: it supports a single set, not a batch"); });
    if (A_.shape()[A_.ndim() - 2] == 0) return {c_, G_};
    CORA_CHECK(!priv_is_empty(priv_host_con_zonotope(*this)), Error::empty_set, [] {
        return std::string("zonotope: the set is empty and CORA.cpp has no empty zonotope");
    });
    if (method == "nullSpace") return aux_null_space(*this);
    const ConZonotope reduced = reduce(int(G_.shape().back() / dim()) + 1);
    return {reduced.c(), reduced.generators()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
