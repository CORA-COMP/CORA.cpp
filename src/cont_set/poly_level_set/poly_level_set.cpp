// poly_level_set - the constructor and the operations of the polynomial constraints
//
// The rows are stored as CORALean's PolyLevelSet: A (m, q), b (m) and one exponent column per
// monomial. The block operations place the coefficient blocks with +0 off the diagonal; the
// operations on monomials (merge, substitution) run on lhs, the rows as a polynomial zonotope,
// and the point test is LevelSet::contains of the rows.
//
// Syntax:   PolyLevelSet P(A, b, R, p);   Q = P.cart_prod(P2);   Q = P.get_subset(k, cen, rad);
// Inputs:   A - (m, q) coefficients
//           b - (m) right-hand side
//           R - exponents
//           p - the factors
// Outputs:  the constraint system
// See also: poly_level_set.h, con_poly_zono.h
//
// CORA.cpp only: CORALean's PolyLevelSet and its operations (Float/Operations of PolyLevelSet)
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/poly_level_set/poly_level_set.h"
#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/error.h"
#include "global/format.h"

#include <cstdint>
#include <string>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The coefficient blocks A1 (m1, q1) and A2 (m2, q2) on the diagonal, +0 elsewhere.
Tensor aux_diag(const Tensor &A1, const Tensor &A2) {
    const int64_t m1 = A1.shape()[0], q1 = A1.shape()[1], m2 = A2.shape()[0], q2 = A2.shape()[1];
    const Tensor top = Tensor::concat({A1, A1.zeros_like({m1, q2})}, 1);
    return Tensor::concat({top, Tensor::concat({A1.zeros_like({m2, q1}), A2}, 1)}, 0);
}

/// The rows of lhs (a polynomial zonotope of center -b) as the system, -(-b) read back as b.
PolyLevelSet aux_of_lhs(const PolyZonotope &L, const Tensor &b, int64_t p) {
    return {L.generators().matrix_transpose(), b, L.exponents(), p};
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyLevelSet::PolyLevelSet(Tensor A, Tensor b, Exponents R, int64_t p)
    : A_(std::move(A)), b_(std::move(b)), R_(std::move(R)), p_(p) {
    detail::Entry entry;
    CORA_CHECK(A_.ndim() == 2 && b_.ndim() == 1 && b_.shape()[0] == A_.shape()[0] &&
                   A_.shape()[1] == int64_t(R_.size()),
               Error::wrong_input_in_constructor, [&] {
                   return "PolyLevelSet: A (m, q), b (m) and one exponent column per column of A, "
                          "got A " +
                          shape_text(A_.shape()) + " and b " + shape_text(b_.shape());
               });
    for (const std::vector<int> &col : R_)
        CORA_CHECK(int64_t(col.size()) == p_, Error::wrong_input_in_constructor, [&] {
            return "PolyLevelSet: every exponent column needs " + std::to_string(p_) + " entries";
        });
}

PolyLevelSet PolyLevelSet::none(const Tensor &like, int64_t p) {
    return {like.zeros_like({0, 0}), like.zeros_like({0}), {}, p};
}

// Conversions ---------------------------------------------------------------------------------

PolyZonotope PolyLevelSet::lhs(const std::vector<int64_t> &ids) const {
    // the center -b as 0 - b, so that a zero row stays +0
    return {b_.zeros_like() - b_, A_.matrix_transpose(), A_.zeros_like({0, num_rows()}), R_, ids,
            detail::Verbatim{}};
}

LevelSet PolyLevelSet::level_set() const {
    // row i: sum_k A(i, k) prod_j x_j^R[k][j] - b_i
    const std::vector<double> a = A_.data(), b = b_.data();
    const int64_t m = num_rows(), q = num_monomials();
    std::vector<Expr> rows;
    for (int64_t i = 0; i < m; ++i) {
        Expr f(-b[std::size_t(i)]);
        for (int64_t k = 0; k < q; ++k) {
            Expr mono(a[std::size_t(i * q + k)]);
            for (int64_t j = 0; j < p_; ++j)
                if (const int e = R_[std::size_t(k)][std::size_t(j)]; e > 0)
                    mono = mono * pow(Expr::var(int(j)), e);
            f = f + mono;
        }
        rows.push_back(f);
    }
    return {rows, p_, std::vector<CompOp>(std::size_t(m), CompOp::eq)};
}

bool PolyLevelSet::contains(const Tensor &x, Tol tol) const {
    detail::Entry entry;
    // the box [-1, 1]^p, then the rows
    const double t = resolve_tol(entry, tol, x.dtype(), "contains");
    if (x.numel() > 0 && x.abs().max().item() > 1.0 + t) return false;
    return num_rows() == 0 || level_set().contains(x, tol);
}

// Operations ----------------------------------------------------------------------------------

PolyLevelSet PolyLevelSet::cart_prod(const PolyLevelSet &P) const {
    return {aux_diag(A_, P.A_), Tensor::concat({b_, P.b_}, 0),
            priv_block_exponents(R_, p_, P.R_, P.p_), p_ + P.p_};
}

PolyLevelSet PolyLevelSet::and_rows(const PolyLevelSet &P) const {
    // the monomials of both, the rows of P below
    Exponents R = R_;
    R.insert(R.end(), P.R_.begin(), P.R_.end());
    return {aux_diag(A_, P.A_), Tensor::concat({b_, P.b_}, 0), std::move(R), p_};
}

PolyLevelSet PolyLevelSet::pad_vars(int64_t r) const {
    // the new factors read by no monomial
    Exponents R = R_;
    for (std::vector<int> &col : R) col.resize(std::size_t(p_ + r), 0);
    return {A_, b_, std::move(R), p_ + r};
}

PolyLevelSet PolyLevelSet::reindex_vars(const std::vector<int64_t> &f) const {
    // factor k of the new system reads the old factor f[k]
    Exponents R;
    for (const std::vector<int> &col : R_) {
        std::vector<int> e;
        e.reserve(f.size());
        for (const int64_t k : f) e.push_back(col[std::size_t(k)]);
        R.push_back(std::move(e));
    }
    return {A_, b_, std::move(R), int64_t(f.size())};
}

PolyLevelSet PolyLevelSet::copies(int64_t a) const {
    // copy j: its rows, its monomials and its block of factors
    PolyLevelSet out = none(b_, 0);
    for (int64_t j = 0; j < a; ++j) out = out.cart_prod(*this);
    return out;
}

PolyLevelSet PolyLevelSet::compact() const {
    // the merge of the monomials as the merge of the generators of lhs
    std::vector<int64_t> factors(static_cast<std::size_t>(p_));
    for (int64_t i = 0; i < p_; ++i) factors[std::size_t(i)] = i + 1;
    return aux_of_lhs(lhs(factors).compact_gens(), b_, p_);
}

PolyLevelSet PolyLevelSet::get_subset(int64_t k, double cen, double rad) const {
    ExpandedColumns x = priv_get_subset_expand(A_.matrix_transpose(), R_, std::size_t(k), cen, rad);
    return {x.G.matrix_transpose(), b_, std::move(x.E), p_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
