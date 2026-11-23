// poly_zonotope - the class of polynomial zonotopes, as CORA's polyZonotope
//
// The set is { c + sum_j (prod_i a_i^E(i,j)) g_j + sum_k b_k gi_k | a_i, b_k in [-1, 1] }: the
// dependent generators g_j carry monomials of the factors a_i, the independent ones gi_k the
// factors b_k. Stored batch first as Zonotope (#163): the center c (n), one generator per row.
// The factors have identifiers id, so that two sets with the same identifier share the factor and
// keep their dependency. The set is a single one (no batch) of the tensors' backend; the exponents
// and the identifiers are integers on the host. As in CORA the constructor merges the dependent
// generators that have the same exponents and drops those that are zero.
//
// Syntax:     PolyZonotope pZ(c, G, GI, E, id);   PolyZonotope pZ(Z);   PolyZonotope pZ(I);
// Operations: one file each in src/cont_set/poly_zonotope/, named as CORA's @polyZonotope; the
//             constructor checks its input (poly_zonotope.cpp)
// See also:   cora/cont_set/zonotope/zonotope.h, cora/cont_set/cont_set.h
// References:
//    [1] N. Kochdumper, M. Althoff. "Sparse polynomial zonotopes: a novel set representation for
//        reachability analysis", IEEE TAC, 2021.
//
// CORA: contSet/@polyZonotope/polyZonotope.m
// CORA: contSet/@polyZonotope/dim.m
// CORA: contSet/@polyZonotope/getSubset.m
// CORA: contSet/@polyZonotope/replaceId.m
// CORA: contSet/@polyZonotope/restoreId.m
// CORA: contSet/@polyZonotope/rmId.m
// CORA: contSet/@polyZonotope/onlyId.m
// CORA: contSet/@polyZonotope/noIndep.m
// CORA: contSet/@polyZonotope/isBounded.m
// CORA: contSet/@polyZonotope/isFullDim.m
// CORA: contSet/@polyZonotope/isequal.m
// CORA: contSet/@polyZonotope/empty.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <cstdint>
#include <optional>
#include <string>
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

/// Tag of the constructor that stores its generators exactly as given: no merge of equal exponents
/// and no zero generator dropped (the poly-adaptive algorithm, whose float definition keeps both).
struct Verbatim {};
} // namespace detail

class ConZonotope;
class Capsule;
class Ellipsoid;
class ZonoBundle;
class PolyZonotope : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::poly_zonotope;

    // Constructors --------------------------------------------------------------------------------

    /// The polynomial zonotope of another set class: CORA defines them by vertices or Taylor
    /// models, Error::not_supported.
    explicit PolyZonotope(const ConZonotope &);
    explicit PolyZonotope(const Capsule &);
    explicit PolyZonotope(const Ellipsoid &);
    explicit PolyZonotope(const Polytope &);
    explicit PolyZonotope(const ZonoBundle &);

    /// The set of the center c (n), the dependent generator rows G (h, n) with the exponents E of
    /// the factors id (p identifiers), and the independent generator rows GI (q, n).
    PolyZonotope(Tensor c, Tensor G, Tensor GI, Exponents E, std::vector<int64_t> id);

    /// As the constructor above for a set whose dependent generators have distinct exponents, which
    /// stay as they are (no merge of equal exponents, zero generators stay).
    PolyZonotope(Tensor c, Tensor G, const Tensor &GI, Exponents E, std::vector<int64_t> id,
                 detail::Normalized);

    /// As the constructors above for a set whose generators are kept exactly as they are: equal
    /// exponents stay apart, and zero generators stay, dependent and independent.
    PolyZonotope(Tensor c, Tensor G, Tensor GI, Exponents E, std::vector<int64_t> id,
                 detail::Verbatim);

    /// The zonotope as a polynomial zonotope: its generators are dependent, E = I (CORA's
    /// polyZonotope(Z)); the identifiers are 1, 2, ...
    explicit PolyZonotope(const Zonotope &Z);

    /// The box as a polynomial zonotope with one dependent generator per dimension.
    explicit PolyZonotope(const Interval &I);

    /// The set {0} of n dimensions (CORA's polyZonotope.origin).
    static PolyZonotope origin(int64_t n);

    /// The center c (n) and the generator rows G (h, n) and GI (q, n).
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
    bool is_intersecting(const Polytope &P, const std::string &method,
                         Tol tol = std::nullopt) const {
        detail::check_predicate_method(method, "is_intersecting");
        return is_intersecting(P, tol);
    }

    /// The halves along the factor with the identifier id (CORA's splitDepFactor): alpha becomes
    /// (alpha +- 1) / 2, so the halves cover the set and have the identifiers of this one.
    std::pair<PolyZonotope, PolyZonotope> split_dep_factor(int64_t id) const;

    /// The halves along the factor of the exponent largest in the longest dependent generator
    /// (CORA's splitLongestGen).
    std::pair<PolyZonotope, PolyZonotope> split_longest_gen() const;

    /// The set without the generators that are zero and the factors that no generator uses (CORA's
    /// compact_ "all", whose exponent merging the constructor does). The only method is "all" (the
    /// default); tol is checked and unused.
    PolyZonotope compact(const std::optional<std::string> &method = std::nullopt,
                         Tol tol = std::nullopt) const;

    /// The dependent generators of equal exponents added into the one of the least index, the sum
    /// from the last column to the first (CORALean's compactGens); no zero generator is dropped.
    PolyZonotope compact_gens() const;

    // The identifiers and the factors -------------------------------------------------------------

    /// The subset in which the factors ids take the values val (CORA's getSubset): the factors
    /// leave the set; a set without generators is the point, as a PolyZonotope. Unknown
    /// identifiers are Error::wrong_value.
    PolyZonotope get_subset(const std::vector<int64_t> &ids, const Tensor &val) const;

    /// The same for a factor restricted to the box val[i] of [-1, 1]: the generators are expanded
    /// binomially in the factor (a factor with the box [-1, 1] stays). An empty box is
    /// Error::wrong_value.
    PolyZonotope get_subset(const std::vector<int64_t> &ids, const Interval &val) const;

    /// The set with the identifiers old_ids renamed to new_ids (CORA's replaceId): factors that get
    /// the same identifier merge, their exponents add. Without old_ids all identifiers are renamed.
    PolyZonotope replace_id(const std::vector<int64_t> &old_ids,
                            const std::vector<int64_t> &new_ids) const;
    PolyZonotope replace_id(const std::vector<int64_t> &new_ids) const;

    /// The set with the factors ids that it lacks added, unused (CORA's restoreId): their exponents
    /// are 0. Repeated identifiers in ids are Error::wrong_value.
    PolyZonotope restore_id(const std::vector<int64_t> &ids) const;

    /// The set without the factors ids (CORA's rmId): the factors are fixed to 1 and equal
    /// generators merge. An identifier the set lacks is Error::wrong_value (CORA ignores it).
    PolyZonotope rm_id(const std::vector<int64_t> &ids) const;

    /// The part that depends on the factors ids alone (with the center) and the remainder (with the
    /// independent generators), whose sum is this set (CORA's onlyId).
    std::pair<PolyZonotope, PolyZonotope> only_id(const std::vector<int64_t> &ids) const;

    /// The set without its independent generators (CORA's noIndep): a subset.
    PolyZonotope no_indep() const;

    /// CORA's empty: a polynomial zonotope has no empty member here (CORALean's wrapper holds it),
    /// Error::not_supported.
    static PolyZonotope empty(int64_t n);

    /// Whether the set has a nonempty interior (CORA's isFullDim): true only if shown, see
    /// docs/cora-differences.md. tol (default: the repository tolerance) is the rank tolerance.
    bool is_full_dim(Tol tol = std::nullopt) const;

    /// Whether the set equals S up to tol (CORA's isequal): both compacted, the identifiers
    /// aligned, the generators compared up to order (the independent ones also up to sign). A
    /// zonotope or a box is compared as a polynomial zonotope; another dimension is false.
    bool isequal(const PolyZonotope &S, Tol tol = std::nullopt) const;
    bool isequal(const Zonotope &Z, Tol tol = std::nullopt) const;
    bool isequal(const Interval &I, Tol tol = std::nullopt) const;

    /// The sum with polynomial zonotopes whose factors are the same where the identifiers are
    /// (CORA's sum): the centers add, the generators join over the union of the identifiers.
    PolyZonotope sum(const std::vector<PolyZonotope> &list) const;

    /// The sets one below the other, in pairs, shared identifiers staying one factor (CORA's
    /// stack).
    static PolyZonotope stack(const std::vector<PolyZonotope> &sets);

    /// The factors ids (default: all) set to the values x (CORA's resolve): constants move to the
    /// center, equal exponents merge. A set without generators is a point, as a PolyZonotope here.
    PolyZonotope resolve(const Tensor &x,
                         const std::optional<std::vector<int64_t>> &ids = std::nullopt) const;

    /// The factors ids (default: all) replaced by the coordinates of inner, polynomials in its own
    /// factors, which are the same where the identifiers are (CORA's subs); inner has no
    /// independent generators.
    PolyZonotope subs(const PolyZonotope &inner,
                      const std::optional<std::vector<int64_t>> &ids = std::nullopt) const;

    /// A polynomial zonotope that contains the set and has no independent generators (CORALean's
    /// restructure): the independent part is boxed to one generator per coordinate, made dependent.
    PolyZonotope restructure() const;

    /// Whether the set is also of the CORA type (CORA's representsa_), judged on the compact set;
    /// the converted set is not returned. The vertex test of "polytope", "conZonotope" and
    /// "zonoBundle" for a set that is no zonotope is Error::not_supported.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;


  protected:
    // The ContSet interface, as CORA's methods of the same names
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
    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;

  private:
    Tensor c_, G_, GI_;
    Exponents E_;
    std::vector<int64_t> id_;
};

/// M * PZ maps, PZ + S sums, a point v translates; PZ - v translates by -v.
inline PolyZonotope operator*(const Tensor &M, const PolyZonotope &PZ) { return PZ.linear_map(M); }
inline PolyZonotope operator*(const Interval &M, const PolyZonotope &PZ) {
    return PZ.linear_map(M);
}
inline PolyZonotope operator+(const PolyZonotope &PZ, const PolyZonotope &S) { return PZ.plus(S); }
inline PolyZonotope operator+(const PolyZonotope &PZ, const Zonotope &Z) { return PZ.plus(Z); }
inline PolyZonotope operator+(const PolyZonotope &PZ, const Interval &I) { return PZ.plus(I); }
inline PolyZonotope operator+(const PolyZonotope &PZ, const Tensor &v) { return PZ.plus(v); }
inline PolyZonotope operator+(const Tensor &v, const PolyZonotope &PZ) { return PZ.plus(v); }
inline PolyZonotope operator-(const PolyZonotope &PZ, const Tensor &v) { return PZ.plus(v * -1.0); }

/// The set scaled by the number s from either side (ContSet::times), in its own class.
inline PolyZonotope operator*(double s, const PolyZonotope &S) {
    return static_cast<const ContSet &>(S).times(s).as<PolyZonotope>();
}
inline PolyZonotope operator*(const PolyZonotope &S, double s) { return s * S; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
