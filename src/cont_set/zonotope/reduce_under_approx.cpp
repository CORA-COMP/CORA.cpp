// reduce_under_approx - a zonotope inside a zonotope with fewer generators
//
// Syntax:   Zred = Z.reduce_under_approx("yang", order);
// Inputs:   method - "yang": the closest pair of generators becomes their sum or difference,
//           whichever is longer in the coordinates of the others, until at most order * n are
//           left; "raghuraman": each of the shortest generators is added with a sign to the big
//           one it is most aligned with; "kochdumper" (old name "sum"): the shortest generators
//           with their signs aligned to the first coordinate become one generator.
//           "sadraddini", "scale", "boxLP", "boxCone", "boxWetzlinger", "nlp", "cluster" (and the
//           old names "linProg", "wetzlinger") are Error::not_supported;
//           order - generators per dimension, at least 1/n
// Outputs:  Zred - the same center, at most order * n generators, and contained in Z; Z without
//           its zero generators if it has few enough. CORA divides by zero in "raghuraman" for a
//           small generator orthogonal to all big ones (NaN); CORA.cpp drops that generator.
//           Not a batch
// See also: reduce, compact
//
// CORA: contSet/@zonotope/reduceUnderApprox.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Choosing generators -----------------------------------------------------------------------------

/// The generators by decreasing squared length (equal ones keep their order): the first N, and
/// the rest (CORA's aux_selectSmallestGenerators with type 'length').
std::pair<Tensor, Tensor> aux_by_length(const Tensor &G, int64_t N) {
    const Tensor sorted = G.take(G.detach().square().sum(-2).argsort(0, true), -1);
    const int64_t m = G.shape().back();
    return {sorted.slice(-1, 0, N), sorted.slice(-1, N, m)};
}

/// The columns of G at the positions that are not i and j.
Tensor aux_without(const Tensor &G, int64_t i, int64_t j) {
    std::vector<int64_t> keep;
    for (int64_t k = 0; k < G.shape().back(); ++k)
        if (k != i && k != j) keep.push_back(k);
    return priv_select_generators(G, keep);
}

// The reductions ----------------------------------------------------------------------------------

/// closeness(i, j) = |g_i| |g_j - (g_i'g_j / |g_i|^2) g_i| for i < j, else infinity (Yang and
/// Ozay's distance of g_j from the line of g_i).
Tensor aux_closeness(const Tensor &G) {
    const int64_t m = G.shape().back();
    const Tensor norm = G.vector_norm(Tensor::Norm::two, -2, true); // (1, m)
    const Tensor along = G.matrix_transpose().matmul(G).divide(norm.square().matrix_transpose());
    const Tensor residual = G.expand_dims(1) - G.expand_dims(2).multiply(along.expand_dims(0));
    const Tensor distance =
        norm.matrix_transpose().multiply(residual.vector_norm(Tensor::Norm::two, 0));
    std::vector<double> upper(m * m, 0.0);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = i + 1; j < m; ++j) upper[i * m + j] = 1.0;
    return Tensor::where(Tensor::like(G, upper, {m, m}), distance,
                         distance.full_like({m, m}, std::numeric_limits<double>::infinity()));
}

/// Yang and Ozay's reduction (CORA's aux_reduceUnderApproxYang): the generators of the closest
/// pair are replaced by their sum or their difference, whichever is longer in the dual basis of
/// the others, until at most order * n are left.
Tensor aux_yang(Tensor G, double order) {
    const int64_t n = G.shape()[0];
    while (double(G.shape().back()) > order * double(n)) {
        const int64_t m = G.shape().back();
        // the first smallest entry in the column-major order of MATLAB's min(..., [], 'all')
        const Tensor closeness = aux_closeness(G.detach()).matrix_transpose().reshape({m * m});
        const int64_t at = int64_t(closeness.argmin(0).item());
        const int64_t i = at % m, j = at / m;
        const Tensor gi = G.slice(-1, i, i + 1), gj = G.slice(-1, j, j + 1);
        const Tensor rest = aux_without(G, i, j);
        const Tensor sum = gi + gj, difference = gi - gj;
        if (rest.shape().back() == 0) { // nothing else to compare with: the longer one
            const auto size = [](const Tensor &g) {
                return g.vector_norm(Tensor::Norm::two, 0).item();
            };
            G = size(sum) >= size(difference) ? sum : difference;
            continue;
        }
        // |(g_i + g_j)' G_r'| / |(g_i - g_j)' G_r'| with G_r = G' (G G')^-1 = pinv(G)
        const Tensor dual = rest.pinv().matrix_transpose();
        const auto length = [&](const Tensor &g) {
            return g.matrix_transpose().matmul(dual).vector_norm(Tensor::Norm::two, -1).item();
        };
        const double up = length(sum), down = length(difference);
        G = Tensor::concat({rest, up / down >= 1 ? sum : difference}, -1);
    }
    return G;
}

/// The sign pattern of Raghuraman and Koeln (CORA's aux_reduceUnderApproxRaghuraman): row r
/// has the sign of the product of the r-th small generator with the big one it is most aligned
/// with, below the identity of the big ones; zero where the small one is orthogonal to all.
Tensor aux_raghuraman_pattern(const Tensor &big, const Tensor &small) {
    const int64_t kept = big.shape().back();
    const Tensor products = small.matrix_transpose().matmul(big); // (rest, kept)
    const Tensor aligned = products.abs().argmax(-1, true); // the big generator of each small one
    const Tensor chosen = products.arange_like(kept).reshape({1, kept}).equal(aligned);
    return Tensor::concat({big.eye_like(kept), chosen.multiply(products.sign())}, 0);
}

// The methods -------------------------------------------------------------------------------------

/// The names of CORA's methods that this port does not have.
bool aux_unsupported(const std::string &method) {
    for (const char *name : {"sadraddini", "scale", "boxLP", "boxCone", "boxWetzlinger", "nlp",
                             "cluster", "linProg", "wetzlinger"})
        if (method == name) return true;
    return false;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::reduce_under_approx(const std::string &method, double order) const {
    detail::Entry entry;
    // a method of CORA, then one this port has; an order; a single zonotope
    CORA_CHECK(method == "yang" || method == "raghuraman" || method == "kochdumper" ||
                   method == "sum" || aux_unsupported(method),
               Error::wrong_value, [&] {
                   return "reduce_under_approx: '" + method + "' is not a method of CORA";
               });
    CORA_CHECK(!aux_unsupported(method), Error::not_supported, [&] {
        return "reduce_under_approx: the method '" + method + "' is not supported";
    });
    CORA_CHECK(!std::isnan(order) && order > 0, Error::wrong_value, [&] {
        return "reduce_under_approx: the order must be positive, got " + format_number(order);
    });
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("reduce_under_approx: it supports a single zonotope, not a batch");
    });
    // CORA reduces only when order * n generators would be fewer than there are
    const Zonotope Z = compact("zeros", machine_eps(c_.dtype()));
    const int64_t n = Z.dim(), m = Z.G_.shape().back();
    if (!(double(n) * order < double(m))) return Z;

    if (method == "yang") return {Z.c_, aux_yang(Z.G_, order)};
    if (method == "raghuraman") {
        const int64_t kept = int64_t(std::floor(order * double(n)));
        CORA_CHECK(kept >= 1, Error::wrong_value, [&] {
            return "reduce_under_approx: the order " + format_number(order) +
                   " leaves no generator";
        });
        const auto [big, small] = aux_by_length(Z.G_, kept);
        return {Z.c_, Z.G_.matmul(aux_raghuraman_pattern(big, small))};
    }
    // kochdumper: the shortest generators, signed by their first coordinate, become one
    const int64_t kept = int64_t(std::floor(order * double(n) - 1));
    CORA_CHECK(kept >= 0, Error::wrong_value, [&] {
        return "reduce_under_approx: the order " + format_number(order) +
               " leaves no generator for the sum";
    });
    const auto [big, small] = aux_by_length(Z.G_, kept);
    const Tensor merged = small.multiply(small.slice(-2, 0, 1).sign()).sum(-1, true);
    return {Z.c_, kept == 0 ? merged : Tensor::concat({big, merged}, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
