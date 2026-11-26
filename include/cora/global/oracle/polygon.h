// polygon - lean::Polygon, a polygon of exact values, computed by CORALean
//
// Syntax:     lean::Polygon pgon(cora_polygon);   lean::Polygon r = pgon.plus(other); r.nominal();
// Operations: one method per `polygon.*` op of the oracle, named as CORA.cpp names CORA's (`or` is
//             unite, `linComb` linear_combination, `convHull` convex_hull, `mtimes` linear_map)
// Parts:      the oracle's polygon is a list of lean vertex polytopes, each with its own error box
//             (at dtype nearest they are empty)
// Crossing:   nominal() returns the pieces as a cora::Polygon, exactly or throws
// See also:   lean/polytope.h, lean/request.h, cora/cont_set/polygon/polygon.h
//
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/polygon/polygon.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/polytope.h"
#include "cora/global/oracle/tensor.h"

#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Polygon {
  public:
    /// An exact polygon (empty error boxes) in the current dtype, piece by piece.
    Polygon(const cora::Polygon &pgon);
    Polygon(std::string dtype, std::vector<Polytope> pieces);

    const std::string &dtype() const noexcept { return dtype_; }
    const std::vector<Polytope> &pieces() const noexcept { return pieces_; }

    /// The pieces as a cora::Polygon; throws if an error box is not empty or a value no double.
    cora::Polygon nominal() const;

    // Operations ---------------------------------------------------------------------------------

    /// The maps and sums of CORALean's Polygon: M pgon, diag(factor) pgon, the pairwise sums, the
    /// translate by -p, the union, the pairwise hulls and the swap or copy of the coordinates.
    Polygon linear_map(const Tensor &M) const;
    Polygon times(const Tensor &factor) const;
    Polygon plus(const Polygon &other) const;
    Polygon minus(const Tensor &p) const;
    Polygon unite(const Polygon &other) const;
    Polygon linear_combination(const Polygon &other) const;
    Polygon project(const std::vector<int64_t> &dims) const;
    static Polygon empty();

    /// The hull of the pieces, their box, the stored pieces and vertices (2, k), the support bound.
    Polytope convex_hull() const;
    Interval interval() const;
    std::vector<Polytope> split_into_convex_sets() const;
    cora::Tensor vertices() const;
    double support_function(const Tensor &dir) const;

    /// The predicates: a point held, disjointness proven, pieces covering each other, one piece
    /// holding all, an interior, no vertex.
    bool contains(const Tensor &point) const;
    bool is_disjoint(const Polygon &other) const;
    bool isequal(const Polygon &other) const;
    bool is_convex() const;
    bool is_full_dim() const;
    bool is_bounded() const;
    bool represents_empty_set() const;

    // Wire ---------------------------------------------------------------------------------------

    Json to_json() const;
    static Polygon from_json(const std::string &dtype, const Json &j);

  private:
    std::string dtype_;
    std::vector<Polytope> pieces_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
