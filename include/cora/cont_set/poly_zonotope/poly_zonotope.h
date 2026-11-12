// poly_zonotope - the class of polynomial zonotopes, as CORA's polyZonotope
//
// The set is { c + sum_j (prod_i a_i^E(i,j)) G(:,j) + GI b | a_i, b_k in [-1, 1] }: the dependent
// generators G carry monomials of the factors a_i, the independent generators GI the factors b_k.
// The factors have identifiers id, so that two sets with the same identifier share the factor and
// keep their dependency. The set is a single one (no batch) of the tensors' backend; the exponents
// and the identifiers are integers on the host. As in CORA the constructor merges the dependent
// generators that have the same exponents and drops those that are zero.
//
// Syntax:     PolyZonotope pZ(c, G, GI, E, id);   PolyZonotope pZ(Z);   PolyZonotope pZ(I);
// Operations: one file each in src/cont_set/poly_zonotope/, named as CORA's @polyZonotope; the
//             constructor checks its input (poly_zonotope.cpp)
// See also:   cora/cont_set/zonotope/zonotope.h, cora/cont_set/cont_set.h
//
// CORA: contSet/@polyZonotope/polyZonotope.m
// CORA: contSet/@polyZonotope/dim.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <cstdint>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The exponent matrix E (p factors, h generators), one entry per dependent generator holding
/// its p exponents.
using Exponents = std::vector<std::vector<int>>;

class Polytope;

namespace detail {
/// Tag of the constructor for generators whose exponents are known to be distinct already (a set
/// made from another by an operation that keeps the exponents): the merge is skipped.
struct Normalized {};
} // namespace detail

class PolyZonotope : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::poly_zonotope;

    /// The set of the center c (n, 1), the dependent generators G (n, h) with the exponents E of
    /// the factors id (p identifiers), and the independent generators GI (n, q).
    PolyZonotope(Tensor c, Tensor G, Tensor GI, Exponents E, std::vector<int64_t> id);

    /// As the constructor above for a set whose dependent generators have distinct exponents, which
    /// stay as they are (no merge of equal exponents, zero generators stay).
    PolyZonotope(Tensor c, Tensor G, const Tensor &GI, Exponents E, std::vector<int64_t> id,
                 detail::Normalized);

    /// The zonotope as a polynomial zonotope: its generators are dependent, E = I (CORA's
    /// polyZonotope(Z)); the identifiers are 1, 2, ...
    explicit PolyZonotope(const Zonotope &Z);

    /// The box as a polynomial zonotope with one dependent generator per dimension.
    explicit PolyZonotope(const Interval &I);

    const Tensor &center() const noexcept { return c_; }
    const Tensor &generators() const noexcept { return G_; }
    const Tensor &independent_generators() const noexcept { return GI_; }
    const Exponents &exponents() const noexcept { return E_; }
    const std::vector<int64_t> &ids() const noexcept { return id_; }

    /// The number of dependent generators and of factors.
    int64_t num_dependent() const noexcept { return int64_t(E_.size()); }
    int64_t num_factors() const noexcept { return int64_t(id_.size()); }

    // Conversions ---------------------------------------------------------------------------------

    /// A zonotope that contains the set (CORA's zonotope): a generator with only even exponents
    /// is a half-length generator and moves half of itself to the center.
    Zonotope zonotope() const;

    // Operations ----------------------------------------------------------------------------------

    /// The Minkowski sum with a zonotope, a box or a point (n, 1): they join the independent part.
    PolyZonotope plus(const Zonotope &Z) const;
    PolyZonotope plus(const Interval &I) const;
    PolyZonotope plus(const Tensor &v) const;

    /// The Minkowski sum with a polynomial zonotope of its own factors (CORA's plus): the factors
    /// are independent of those of this set.
    PolyZonotope plus(const PolyZonotope &S) const;

    /// The sum in which equal identifiers are the same factor (CORA's exactPlus): the dependency
    /// between the summands stays, a subset of the plus.
    PolyZonotope exact_plus(const PolyZonotope &S) const;

    /// M * pZ for a matrix M (k, n): exact, the factors stay.
    PolyZonotope linear_map(const Tensor &M) const;

    /// [M] * pZ for an interval matrix (with (n, n) bounds): an enclosure, the width of the matrix
    /// times the box of the set is added as independent generators (CORA's intervalMatrix mtimes).
    PolyZonotope linear_map(const Interval &M) const;

    /// A polynomial zonotope that contains the segments between the points of this set and of S
    /// (CORA's enclose): the convex hull, with one new factor.
    PolyZonotope enclose(const PolyZonotope &S) const;

    /// The Cartesian product with a zonotope (CORA's cartProd): its generators are independent.
    PolyZonotope cart_prod(const Zonotope &Z) const;

    /// A polynomial zonotope of at most order * n generators that contains this one (CORA's reduce
    /// with "girard"): the longest generators stay, the others become a zonotope that is reduced
    /// to order 1 and joins the independent part. Not "adaptive" (docs/cora-differences.md).
    PolyZonotope reduce(int order = 1) const;

    /// The image of the quadratic map z_i = x' Q[i] x (CORA's quadMap(pZ, Q)): the dependent part
    /// is exact (all products of two generators), the independent part is enclosed.
    PolyZonotope quad_map(const std::vector<Tensor> &Q) const;

    /// Whether the set may meet a polytope (CORA's isIntersecting_ with type "approx"): false is
    /// certain, true says nothing. tol (default: the repository tolerance) widens the equalities.
    bool is_intersecting(const Polytope &P, Tol tol = std::nullopt) const;

    /// The halves along the factor with the identifier id (CORA's splitDepFactor): alpha becomes
    /// (alpha +- 1) / 2, so the halves cover the set and have the identifiers of this one.
    std::pair<PolyZonotope, PolyZonotope> split_dep_factor(int64_t id) const;

    /// The halves along the factor of the exponent largest in the longest dependent generator
    /// (CORA's splitLongestGen).
    std::pair<PolyZonotope, PolyZonotope> split_longest_gen() const;

    /// The set without the generators that are zero and the factors that no generator uses (CORA's
    /// compact_ "all", whose exponent merging the constructor does).
    PolyZonotope compact() const;


  protected:
    int64_t dim_impl() const override { return c_.shape()[0]; }
    Tensor center_impl() const override;
    Set reduce_impl(int order) const override;
    Set translate_impl(const Tensor &v) const override;
    Set linear_map_impl(const Tensor &M) const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;
    bool is_bounded_impl() const override { return true; }

  private:
    Tensor c_, G_, GI_;
    Exponents E_;
    std::vector<int64_t> id_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
