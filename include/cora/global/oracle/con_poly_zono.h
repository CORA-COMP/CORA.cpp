// con_poly_zono - lean::ConPolyZono, a constrained polynomial zonotope of CORALean
//
// Syntax:     lean::ConPolyZono S(cora_cpz);   lean::ConPolyZono r = S.intersect(S2); r.gather();
// Operations: one method per `conPolyZono.*` op of the oracle, named as CORA.cpp names them
//             (`and` is intersect, `mtimes` linear_map, `linComb` linear_combination)
// Parts:      the oracle's set {c, G, GI, Exp, E, A, b, EC, b_error}: the polynomial part on the
//             wire of lean::PolyZonotope and the rows A mu_EC(a) = b with their slack; at dtype
//             nearest the error box and the slack are zero. A result has no identifiers: gather()
//             gives it 1..p
// Crossing:   gather() returns the set as a cora::ConPolyZono, exactly, or throws
// See also:   lean/poly_zonotope.h, lean/request.h, cora/cont_set/con_poly_zono/con_poly_zono.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/con_poly_zono/con_poly_zono.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/poly_zonotope.h"
#include "cora/global/oracle/tensor.h"
#include "cora/global/oracle/zonotope.h"

#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Reply;

class ConPolyZono {
  public:
    /// An exact set (empty error box, no slack) in the current dtype, with its identifiers.
    ConPolyZono(const cora::ConPolyZono &S);

    /// The oracle's conversions of a zonotope, a polynomial zonotope and a constrained zonotope.
    static ConPolyZono from_zonotope(const cora::Zonotope &Z);
    static ConPolyZono from_poly_zonotope(const cora::PolyZonotope &P);
    static ConPolyZono from_con_zonotope(const cora::ConZonotope &cz);

    /// The set {0} and the empty set of n dimensions.
    static ConPolyZono origin(int64_t n);
    static ConPolyZono empty(int64_t n);

    /// The set as CORA.cpp's, the identifiers 1..p; throws if the error box or a slack is not 0.
    cora::ConPolyZono gather() const;

    // Operations ----------------------------------------------------------------------------------

    ConPolyZono linear_map(const Tensor &M) const;
    ConPolyZono plus(const ConPolyZono &S) const;
    ConPolyZono exact_plus(const ConPolyZono &S) const;
    ConPolyZono cart_prod(const ConPolyZono &S) const;
    ConPolyZono intersect(const ConPolyZono &S) const;
    ConPolyZono linear_combination(const ConPolyZono &S) const;
    ConPolyZono convex_hull(const ConPolyZono &S) const;

    /// The coordinates dims, and the permutation with state i at position dims[i].
    ConPolyZono project(const std::vector<int64_t> &dims) const;
    ConPolyZono lift(int64_t N, const std::vector<int64_t> &dims) const;

    /// CORALean's compact and reduce (girard) at the order.
    ConPolyZono compact() const;
    ConPolyZono reduce(int order) const;

    /// The halves of the box I along coordinate dim, intersected with the set (splitWith).
    std::pair<ConPolyZono, ConPolyZono> split(int64_t dim, const Interval &I) const;

    /// The factor id as cen + rad a; its halves [-1, 0] and [0, 1]; the longest factor's halves.
    ConPolyZono get_subset(int64_t id, double cen, double rad) const;
    std::pair<ConPolyZono, ConPolyZono> split_dep_factor(int64_t id) const;
    std::pair<ConPolyZono, ConPolyZono> split_longest_gen() const;

    /// The dimension, the center, the box, the zonotope and the polynomial part.
    int64_t dim() const;
    Tensor center() const;
    Interval interval() const;
    Zonotope zonotope() const;
    PolyZonotope poly_zonotope() const;

    /// Structural equality at tol 0, and whether the set has no generator.
    bool isequal(const ConPolyZono &S) const;
    bool representsa_point() const;

    Json to_json() const { return json_; }
    const std::string &dtype() const noexcept { return dtype_; }

  private:
    /// The set of the reply member `key` (default `cPZ`), in the current dtype.
    static ConPolyZono of_reply(const Reply &r, const char *key = "cPZ");

    ConPolyZono(std::string dtype, Json json) : dtype_(std::move(dtype)), json_(std::move(json)) {}

    std::string dtype_;
    Json json_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
