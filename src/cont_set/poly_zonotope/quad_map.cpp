// quad_map - the image of a polynomial zonotope under z_i = x' Q_i x, as CORA's quadMap
//
// With [c, G] the generators of the dependent part and their exponents, the products of every
// two of them are generators of the result with the sums of the exponents, exact; the constant
// one goes to the center. The independent part IR is enclosed by the quadratic maps of the
// zonotopes R = (0, GI) and D = zonotope of the dependent part: map(R) + map(D, R) + map(R, D).
//
// Syntax:   S = pZ.quad_map(Q);
// Inputs:   Q - the n_out matrices Q_i (n, n)
// Outputs:  S - the image, of the dimension n_out
// See also: zonotope
//
// CORA: contSet/@polyZonotope/quadMap.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/rng.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The generators of the products (j, k), j <= k, of the columns of Gext: M_i(j,j) and
/// M_i(j,k) + M_i(k,j) for the matrices M_i = Gext' Q_i Gext, one row per output.
Tensor aux_products(const Tensor &Gext, const std::vector<Tensor> &Q) {
    const int64_t N = Gext.shape()[1];
    std::vector<int64_t> upper, lower;
    std::vector<double> weight;
    for (int64_t j = 0; j < N; ++j)
        for (int64_t k = j; k < N; ++k) {
            upper.push_back(j * N + k);
            lower.push_back(k * N + j);
            weight.push_back(k == j ? 0.0 : 1.0);
        }
    const Tensor w = Tensor::like(Gext, weight, {1, int64_t(weight.size())});
    std::vector<Tensor> rows;
    for (const Tensor &Qi : Q) {
        const Tensor M = Gext.matrix_transpose().matmul(Qi).matmul(Gext).reshape({1, N * N});
        rows.push_back(M.take(upper, -1) + M.take(lower, -1).multiply(w));
    }
    return Tensor::concat(rows, -2);
}

/// The exponents of the products (j, k), j <= k, of the columns of Eext.
Exponents aux_exponents(const Exponents &Eext) {
    Exponents out;
    for (std::size_t j = 0; j < Eext.size(); ++j)
        for (std::size_t k = j; k < Eext.size(); ++k) {
            std::vector<int> col = Eext[j];
            for (std::size_t i = 0; i < col.size(); ++i) col[i] += Eext[k][i];
            out.push_back(std::move(col));
        }
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::quad_map(const std::vector<Tensor> &Q) const {
    detail::Entry entry;
    CORA_CHECK(!Q.empty(), Error::wrong_value,
               [] { return std::string("quad_map: Q needs at least one matrix"); });
    for (const Tensor &Qi : Q)
        CORA_CHECK(Qi.ndim() == 2 && Qi.shape()[0] == dim() && Qi.shape()[1] == dim(),
                   Error::dimension_mismatch, [&] {
                       return "quad_map: every matrix must be (" + std::to_string(dim()) + ", " +
                              std::to_string(dim()) + "), got " + shape_text(Qi.shape());
                   });
    const auto m = int64_t(Q.size());
    const bool independent = GI_.shape()[1] > 0;
    // no dependent part: the quadratic map of the zonotope (c, GI)
    if (num_dependent() == 0) {
        const Zonotope Zr = Zonotope(c_, GI_).quad_map(Q);
        return {Zr.center(), G_.zeros_like({m, 0}), Zr.generators(), {}, id_};
    }
    // the dependent part: all products of the columns of [c, G]
    Exponents Eext = E_;
    Eext.insert(Eext.begin(), std::vector<int>(id_.size(), 0));
    const Tensor Gquad = aux_products(Tensor::concat({c_, G_}, -1), Q);
    Tensor c = c_.zeros_like({m, 1}), GI = G_.zeros_like({m, 0});
    if (independent) {
        const Zonotope R(c_.zeros_like(), GI_);
        const Zonotope Dnoc = PolyZonotope(c_, G_, G_.zeros_like({dim(), 0}), E_, id_).zonotope();
        const Zonotope A = R.quad_map(Q), B = Dnoc.quad_map(R, Q), C = R.quad_map(Dnoc, Q);
        c = A.center() + B.center() + C.center();
        GI = Tensor::concat({A.generators(), B.generators(), C.generators()}, -1);
    }
    return priv_fold_constants({c, Gquad, GI, aux_exponents(Eext), id_});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
