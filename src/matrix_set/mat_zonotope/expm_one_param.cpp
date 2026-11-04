// expm_one_param - the Taylor series of the matrix exponential for one parameter, as CORA's
//
// Syntax:   auto R = A.expm_one_param(r, max_order);   R = A.expm_one_param(r, max_order, u);
// Inputs:   r - the step;  max_order - the order of the series, at least 1;  A - a square set
//           with exactly one generator: the matrices M(b) = C + b G, |b| <= 1;  u - an input
//           zonotope c + b g with at most one generator, in the same b (a column u is (u, 0))
// Outputs:  R - enclosure: the matrix zonotope K of the sum of r^k M(b)^k / k! for k = 0 to
//           max_order, whose l-th generator is the coefficient of b^l (halved for even l, which
//           moves the other half to the center, as b^l lies in [0, 1] there); remainder: the
//           remainder box of the interval matrix around A r, to be added; powers: the powers
//           (r M)^k as matrix zonotopes with a generator per power of b; constant_input: with u,
//           the zonotope of the sum of r^k M^(k-1) u / k! for k = 1 to max_order, in the same
//           form (CORA returns a zero zonotope without u)
// See also: expm_independent, IntervalMatrix::exponential_remainder
//
// CORA: matrixSet/@matZonotope/expmOneParam.m
// CORA starts the l-th generator's sum at the coefficient of b^1 of M^l, not of b^l, and drops
// the generators after the first (docs/cora-differences.md); CORALean's definition is followed.

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The coefficients of the powers ------------------------------------------------------------------

/// T[k][i] for k = 0 to N and i = 0 to k: the coefficient of b^i in (C + b G)^k.
std::vector<std::vector<Tensor>> aux_coefficients(const Tensor &C, const Tensor &G, int64_t N) {
    std::vector<std::vector<Tensor>> T(N + 1);
    T[0] = {C.eye_like(C.size(-1)) + C.zeros_like()};
    for (int64_t k = 1; k <= N; ++k)
        for (int64_t i = 0; i <= k; ++i) {
            // the power one lower, times G (one more b) or times C
            Tensor term;
            if (i >= 1) term = T[k - 1][i - 1].matmul(G);
            if (i <= k - 1) {
                const Tensor with_center = T[k - 1][i].matmul(C);
                term = term.defined() ? term + with_center : with_center;
            }
            T[k].push_back(term);
        }
    return T;
}

/// The coefficients of b^i in M(b)^(k-1) u(b) for u(b) = c + b g, as aux_coefficients: [k][i].
std::vector<std::vector<Tensor>> aux_input_coefficients(const std::vector<std::vector<Tensor>> &T,
                                                        const Tensor &c, const Tensor &g) {
    std::vector<std::vector<Tensor>> U(T.size());
    for (std::size_t k = 1; k < T.size(); ++k)
        for (std::size_t i = 0; i <= k; ++i) {
            Tensor term;
            if (i >= 1) term = T[k - 1][i - 1].matmul(g);
            if (i <= k - 1) {
                const Tensor with_center = T[k - 1][i].matmul(c);
                term = term.defined() ? term + with_center : with_center;
            }
            U[k].push_back(term);
        }
    return U;
}

// The sums of the series --------------------------------------------------------------------------

/// The sum of r^k / k! T[k] over the orders k = 1 to N with the leading term of each b^l: the
/// center (plus `start`, the identity for the state) with the halves of the even ones added,
/// and the generators for l = 1 to N.
std::pair<Tensor, Tensor> aux_taylor_terms(const std::vector<std::vector<Tensor>> &T, double r,
                                           int64_t N, const Tensor &start) {
    const auto factor = [&](int64_t k) { return std::pow(r, double(k)) / priv_factorial(k); };
    Tensor center = T[1][0] * r;
    if (start.defined()) center = start + center;
    for (int64_t k = 2; k <= N; ++k) center = center + T[k][0] * factor(k);
    std::vector<Tensor> generators;
    for (int64_t l = 1; l <= N; ++l) {
        Tensor sum = T[l][l] * factor(l);
        for (int64_t k = l + 1; k <= N; ++k) sum = sum + T[k][l] * factor(k);
        if (l % 2 == 0) {
            sum = sum * 0.5;
            center = center + sum;
        }
        generators.push_back(sum);
    }
    return {center, Tensor::stack(generators, -3)};
}

/// The result for the set (C, G) and the input u, if there is one.
MatZonotope::ExpmOneParamResult aux_expm(const MatZonotope &A, double r, int64_t N,
                                         const Zonotope *u) {
    const Tensor C = A.center();
    const Tensor G = A.generators().slice(-3, 0, 1).squeeze(-3);
    const auto T = aux_coefficients(C, G, N);
    const auto [center, generators] = aux_taylor_terms(T, r, N, C.eye_like(C.size(-1)));
    std::vector<MatZonotope> powers{A * r};
    for (int64_t k = 2; k <= N; ++k) {
        const std::vector<Tensor> terms(T[k].begin() + 1, T[k].end());
        powers.push_back(MatZonotope(T[k][0], Tensor::stack(terms, -3)) * std::pow(r, double(k)));
    }
    std::optional<Zonotope> constant_input;
    if (u) {
        const Tensor &c = u->center();
        const Tensor g = u->generators().size(-1) == 0 ? c.zeros_like() : u->generators();
        const auto [input_center, input_generators] =
            aux_taylor_terms(aux_input_coefficients(T, c, g), r, N, Tensor());
        // the n x 1 matrices of the input terms are the columns of the zonotope
        constant_input = Zonotope(input_center, input_generators.squeeze(-1).matrix_transpose());
    }
    return {MatZonotope(center, generators), (A * r).interval_matrix().exponential_remainder(N),
            std::move(powers), std::move(constant_input)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

MatZonotope::ExpmOneParamResult
MatZonotope::expm_one_param(double r, int64_t max_order,
                            const std::optional<Zonotope> &u_trans) const {
    detail::Entry entry;
    // a square set with one generator, an order, and an input that fits
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm_one_param: the matrix zonotope must be square, got " + std::to_string(rows()) +
               " x " + std::to_string(cols());
    });
    CORA_CHECK(num_generators() == 1, Error::wrong_value, [&] {
        return "expm_one_param: the matrix zonotope needs exactly one generator, got " +
               std::to_string(num_generators());
    });
    CORA_CHECK(max_order >= 1, Error::wrong_value, [&] {
        return "expm_one_param: the order must be at least 1, got " + std::to_string(max_order);
    });
    CORA_CHECK(!u_trans || u_trans->dim() == rows(), Error::dimension_mismatch, [&] {
        return "expm_one_param: the input needs " + std::to_string(rows()) + " dimensions, got " +
               std::to_string(u_trans->dim());
    });
    CORA_CHECK(!u_trans || u_trans->generators().size(-1) <= 1, Error::not_supported, [] {
        return std::string("expm_one_param: an input with more than one generator is not "
                           "supported, as in CORA the set it gives is meant for one");
    });
    return aux_expm(*this, r, max_order, u_trans ? &*u_trans : nullptr);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
