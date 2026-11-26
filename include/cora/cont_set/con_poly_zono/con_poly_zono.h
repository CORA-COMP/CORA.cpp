// con_poly_zono - the class of constrained polynomial zonotopes, as CORA's conPolyZono
//
// The set is { poly(a, b) | a in [-1, 1]^p, b in [-1, 1]^q, A mu(a) = b }: a polynomial zonotope
// poly (center c, dependent generators G with exponents E over the factors id, independent
// generators GI) and polynomial constraints on its dependent factors, a PolyLevelSet (CORA's A, b,
// EC), as CORALean's ConPolyZono composes them. Stored as one set (no batch); the operations follow
// CORALean's float definitions in its order of operations, so the lean backend matches the oracle
// bit for bit; no merge happens unless an operation says so (compact, reduce).
//
// Syntax:     ConPolyZono cPZ(c, G, E, A, b, EC, GI, id);   ConPolyZono cPZ(poly, cons);
//             ConPolyZono cPZ(Z);   ConPolyZono cPZ(poly);   ConPolyZono cPZ(cz);   ConPolyZono
//             cPZ(I);
// Operations: one file each in src/cont_set/con_poly_zono/, named as CORA's @conPolyZono
// See also:   poly_zonotope/poly_zonotope.h, poly_level_set/poly_level_set.h
// References:
//    [1] N. Kochdumper, M. Althoff. "Constrained polynomial zonotopes", Acta Informatica, 2023.
//
// CORA: contSet/@conPolyZono/conPolyZono.m
// CORA: contSet/@conPolyZono/dim.m
// CORA: contSet/@conPolyZono/center.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/poly_level_set/poly_level_set.h"
#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class ConPolyZono : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::con_poly_zono;

    // Constructors --------------------------------------------------------------------------------

    /// The set of the center c (n), the dependent generator rows G (h, n) with the exponents E,
    /// the constraint rows A (m, q), b (m) with the monomial exponents EC, the independent
    /// generator rows GI (r, n) and the identifiers id (p) of the factors, kept as given.
    ConPolyZono(Tensor c, Tensor G, Exponents E, Tensor A, Tensor b, Exponents EC, Tensor GI,
                const std::vector<int64_t> &id);

    /// The polynomial zonotope poly constrained by cons (on the dependent factors of poly).
    ConPolyZono(PolyZonotope poly, PolyLevelSet cons);

    /// A zonotope (its generators dependent, identity exponents), a polynomial zonotope (no
    /// constraint), a constrained zonotope (its rows A x = b on the identity monomials) and a box
    /// (through its zonotope, a generator per side), as CORA's conPolyZono(S).
    explicit ConPolyZono(const Zonotope &Z);
    explicit ConPolyZono(const PolyZonotope &poly);
    explicit ConPolyZono(const ConZonotope &cz);
    explicit ConPolyZono(const Interval &I);

    /// The set {0} and the empty set (the row 0 = 1) of n dimensions.
    static ConPolyZono origin(int64_t n);
    static ConPolyZono empty(int64_t n);

    /// The parts: the polynomial zonotope and the constraint.
    const PolyZonotope &poly_part() const noexcept { return poly_; }
    const PolyLevelSet &constraints() const noexcept { return cons_; }

    /// CORA's fields c, G, E, GI, id, A, b and EC.
    const Tensor &c() const noexcept { return poly_.center(); }
    const Tensor &generators() const noexcept { return poly_.generators(); }
    const Exponents &exponents() const noexcept { return poly_.exponents(); }
    const Tensor &independent_generators() const noexcept { return poly_.independent_generators(); }
    const std::vector<int64_t> &ids() const noexcept { return poly_.ids(); }
    const Tensor &A() const noexcept { return cons_.A(); }
    const Tensor &b() const noexcept { return cons_.b(); }
    const Exponents &constraint_exponents() const noexcept { return cons_.exponents(); }

    // Conversions ---------------------------------------------------------------------------------

    /// The set without its constraint (CORALean's polyZonotope, a superset).
    PolyZonotope poly_zonotope() const { return poly_; }

    /// A zonotope that holds the set: the dependent and the independent generators side by side,
    /// the constraint dropped (CORALean's zonotope).
    Zonotope zonotope() const;

    // Operations ----------------------------------------------------------------------------------

    /// M * cPZ for a matrix M (k, n): exact, the constraint stays.
    ConPolyZono linear_map(const Tensor &M) const;

    /// The Minkowski sum: with a constrained polynomial zonotope its factors and constraint join
    /// on blocks; a zonotope or a point (n) joins the independent part (CORA's plus).
    ConPolyZono plus(const ConPolyZono &S) const;
    ConPolyZono plus(const Zonotope &Z) const;
    ConPolyZono plus(const Tensor &v) const;

    /// The sum over the same factors (CORA's exactPlus): both sets have the identifiers of this one
    /// and the same constraint; the generators join.
    ConPolyZono exact_plus(const ConPolyZono &S) const;

    /// The Cartesian product, each set on its own factors (CORA's cartProd).
    ConPolyZono cart_prod(const ConPolyZono &S) const;

    /// The intersection (CORA's and_): the independent generators become dependent, and the rows
    /// G1 a1 - G2 a2 = c2 - c1 pin the two points together (CORALean's intersection).
    ConPolyZono intersect(const ConPolyZono &S) const;

    /// The points (1 - t)/2 x + (1 + t)/2 y, x in this set, y in S (CORA's linComb).
    ConPolyZono linear_combination(const ConPolyZono &S) const;

    /// The convex hull (CORA's convHull_): n + 1 weighted copies of the linear combination, the
    /// weights summing to one; the independent generators are made dependent first.
    ConPolyZono convex_hull(const ConPolyZono &S) const;

    /// The independent generators as dependent ones on new factors (CORALean's dependent).
    ConPolyZono dependent() const;

    /// The generators of equal exponents merged, then the monomials, then the factors no block
    /// reads dropped (CORA's compact_ "all", CORALean's order); tol is checked and unused.
    ConPolyZono compact(const std::optional<std::string> &method = std::nullopt,
                        Tol tol = std::nullopt) const;

    /// At most order * n generators (CORA's reduce "girard"): compact, then the generators of the
    /// two blocks ranked by ||g||_1 - ||g||_inf; the (order - 1) n first stay, the rest is a box.
    ConPolyZono reduce(int order = 1) const;

    /// The halves of the box I (default: interval()) along coordinate dim, each intersected with
    /// the set (CORA's split at one dimension).
    std::pair<ConPolyZono, ConPolyZono>
    split(int64_t dim, const std::optional<Interval> &I = std::nullopt) const;

    /// The set at the factor id restricted to val, a sub-box of [-1, 1]: the factor becomes
    /// center + radius * a in the generators and in the rows (CORA's getSubset).
    ConPolyZono get_subset(int64_t id, const Interval &val) const;

    /// The halves a in [-1, 0] and [0, 1] of the factor id (CORA's splitDepFactor).
    std::pair<ConPolyZono, ConPolyZono> split_dep_factor(int64_t id) const;

    /// The halves along the factor of the largest summed squared length (CORA's splitLongestGen);
    /// a set without factors twice.
    std::pair<ConPolyZono, ConPolyZono> split_longest_gen() const;

    /// The set in N dimensions with state i at position dims[i] (CORA's lift_, a permutation).
    ConPolyZono lift(int64_t N, const std::vector<int64_t> &dims) const;

    /// Whether the stored data agree within tol, field by field (CORA's isequal at tol 0).
    bool isequal(const ConPolyZono &S, Tol tol = std::nullopt) const;

    /// Whether the set is a point (type "point": no generator in either block) or empty (type
    /// "emptySet": only the row 0 = 1 of no monomial is recognized); other types are not supported.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

  protected:
    // The ContSet interface, as CORA's methods of the same names
    int64_t dim_impl() const override { return c().shape()[0]; }
    Tensor center_impl() const override { return c(); }
    Set reduce_impl(int order) const override { return reduce(order); }
    Set translate_impl(const Tensor &v) const override { return plus(v); }
    Set linear_map_impl(const Tensor &M) const override { return linear_map(M); }
    Tensor support_function_impl(const Tensor &d) const override;
    // the enclosures, the random points and the plot (one file each)
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;
    // a constrained polynomial zonotope is compact
    bool is_bounded_impl() const override { return true; }
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Set compact_impl(const std::optional<std::string> &method, Tol tol) const override {
        return compact(method, tol);
    }
    std::optional<std::vector<std::string>> compact_methods_impl() const override {
        return std::vector<std::string>{"all"};
    }

  private:
    PolyZonotope poly_;
    PolyLevelSet cons_;
};

/// M * cPZ maps, cPZ + S sums, a point v translates.
inline ConPolyZono operator*(const Tensor &M, const ConPolyZono &S) { return S.linear_map(M); }
inline ConPolyZono operator+(const ConPolyZono &S, const ConPolyZono &T) { return S.plus(T); }
inline ConPolyZono operator+(const ConPolyZono &S, const Zonotope &Z) { return S.plus(Z); }
inline ConPolyZono operator+(const ConPolyZono &S, const Tensor &v) { return S.plus(v); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
