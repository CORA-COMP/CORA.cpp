// poly_level_set - polynomial equality constraints on factors, as CORALean's PolyLevelSet
//
// The rows A mu_R(x) = b on the box [-1, 1]^p: A (m, q) holds the coefficient of every monomial
// mu_k(x) = prod_i x_i^R[k][i] (one exponent column of p entries per monomial), b (m) the
// right-hand side. It is the constraint of a constrained polynomial zonotope (ConPolyZono), as
// CORALean's ConPolyZono holds it; CORA stores the same data in the fields A, b and EC of
// conPolyZono. The polynomial part A mu_R(x) - b is a polynomial zonotope of the rows over the
// factors (lhs), which carries the operations on monomials (merge, substitution); the test of a
// point is the LevelSet of the rows. Not a ContSet: a set of factors, with no center, interval or
// plot of its own.
//
// Syntax:     PolyLevelSet P(A, b, R, p);   PolyLevelSet::none(like, p);
// Operations: lhs, level_set, contains, cart_prod, and_rows, pad_vars, reindex_vars, copies,
//             compact, get_subset (poly_level_set.cpp)
// See also:   cora/cont_set/con_poly_zono/con_poly_zono.h, cora/cont_set/level_set/level_set.h
//
// CORA.cpp only: CORALean's PolyLevelSet, the constraint object of conPolyZono (CORA's A, b, EC)
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/level_set/level_set.h"
#include "cora/cont_set/poly_zonotope/poly_zonotope.h"

#include <cstdint>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class PolyLevelSet {
  public:
    /// The rows A mu_R(x) = b of p factors: A (m, q), b (m), R one exponent column per monomial.
    PolyLevelSet(Tensor A, Tensor b, Exponents R, int64_t p);

    /// No rows on p factors; the tensors are of the backend and dtype of `like`.
    static PolyLevelSet none(const Tensor &like, int64_t p);

    const Tensor &A() const noexcept { return A_; }
    const Tensor &b() const noexcept { return b_; }
    const Exponents &exponents() const noexcept { return R_; }

    /// The rows m, the monomials q and the factors p.
    int64_t num_rows() const noexcept { return A_.shape()[0]; }
    int64_t num_monomials() const noexcept { return int64_t(R_.size()); }
    int64_t num_vars() const noexcept { return p_; }

    // Conversions ---------------------------------------------------------------------------------

    /// The polynomial A mu_R(x) - b as a polynomial zonotope of the m rows over the factors ids:
    /// center -b, one dependent generator per monomial (a column of A), nothing merged.
    PolyZonotope lhs(const std::vector<int64_t> &ids) const;

    /// The rows as a level set of the p factors, each row an equation.
    LevelSet level_set() const;

    /// Whether the points x (N, p) are in the box and meet every row within tol (LevelSet's test).
    bool contains(const Tensor &x, Tol tol = std::nullopt) const;

    // Operations ----------------------------------------------------------------------------------

    /// Both systems, each on its own block of factors (CORALean's cartProd).
    PolyLevelSet cart_prod(const PolyLevelSet &P) const;

    /// The rows of P below these, on the same factors, over both monomial lists (andRows).
    PolyLevelSet and_rows(const PolyLevelSet &P) const;

    /// The same rows over p + r factors, the last r read by none (padVars).
    PolyLevelSet pad_vars(int64_t r) const;

    /// The rows over the factors f: factor k is this system's factor f[k] (reindexVars).
    PolyLevelSet reindex_vars(const std::vector<int64_t> &f) const;

    /// `a` copies of the rows, copy j on the j-th block of factors (copies).
    PolyLevelSet copies(int64_t a) const;

    /// The monomials of equal exponents merged into the one of the least index, each sum from the
    /// last to the first (CORALean's compact; PolyZonotope::compact_gens of lhs).
    PolyLevelSet compact() const;

    /// The rows at x_k = cen + rad x_k: every monomial expanded binomially in x_k, one column per
    /// power of each monomial, nothing merged (CORALean's getSubset).
    PolyLevelSet get_subset(int64_t k, double cen, double rad) const;

  private:
    Tensor A_, b_;
    Exponents R_;
    int64_t p_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
