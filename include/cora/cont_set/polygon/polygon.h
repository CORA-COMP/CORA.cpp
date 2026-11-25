// polygon - the class of polygons, possibly non-convex regions of the plane, as CORA's polygon
//
// A list of convex pieces, each a Polytope in vertex form (p, 2), denoting their union, as
// CORALean's Polygon; a boundary (CORA's polyshape) is split into convex pieces. The maps, sums and
// predicates are CORALean's, bit for bit at nearest; the Boolean operations, the area and the
// centroid clip the pieces exactly. Immutable, not a batch; holes are not supported yet.
//
// Syntax:     Polygon pgon(V);   Polygon pgon({P1, P2});   Polygon pgon = Polygon::from(Z);
// Operations: constructor (polygon.cpp), empty, from, pieces, center, interval, vertices,
//             support_function, linear_map, times, plus, minus, unite, intersect,
//             set_difference, linear_combination, convex_hull, polytope, project, contains,
//             is_intersecting, isequal, is_bounded, is_convex, is_full_dim, representsa, volume,
//             triangulation, split_into_convex_sets, rand_point, display (one file each)
// Not ported: holes (not supported yet: a region inside another is refused), compact,
//             removeCollinearPoints, minkDiff, quadMap, hausdorffDist, enclosePoints,
//             expandBoundaries, generateRandom
// See also:   cora/cont_set/polytope/polytope.h, cora/cont_set/cont_set.h
//
// CORA: contSet/@polygon/polygon.m
// CORA: contSet/@polygon/dim.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/set.h"

#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Polygon : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::polygon;

    /// The union of convex pieces of the plane (CORALean's Polygon.ofPieces): polytopes of two
    /// dimensions, each bounded; a halfspace polytope is kept by its vertices. None: the empty set.
    explicit Polygon(const std::vector<Polytope> &pieces);

    /// The region inside the boundary V (p, 2), one vertex per row, in either orientation (CORA's
    /// polygon(V)); NaN rows separate regions. A convex boundary is one piece, a simple
    /// non-convex one is cut into convex pieces (ear clipping, merged greedily).
    explicit Polygon(const Tensor &V);

    /// The empty polygon (CORA's polygon.empty(2)): no piece; n must be 2.
    static Polygon empty(int64_t n = 2);

    /// A two-dimensional set as a polygon (CORA's polygon(S)): its outline, cut into convex
    /// pieces if it is not convex.
    static Polygon from(const ContSet &S);

    /// The convex pieces, in their order, and their number.
    const std::vector<Polytope> &pieces() const noexcept { return pieces_; }
    int64_t num_pieces() const noexcept { return int64_t(pieces_.size()); }

    /// A tensor (0, 2) of the pieces' backend and dtype, which new tensors are made like.
    const Tensor &like() const noexcept { return like_; }

    // Conversions ---------------------------------------------------------------------------------

    /// The convex hull of the pieces (CORA's convHull_ and polytope(pgon)): the vertex polytope of
    /// the vertices of every piece, in their order.
    Polytope convex_hull() const;
    Polytope polytope() const;

    /// The pieces themselves (CORA's splitIntoConvexSets), each convex.
    std::vector<Polytope> split_into_convex_sets() const;

    /// Triangles that cover the polygon without overlap (CORA's triangulation): the fan of the
    /// hull of every piece, after the pieces are made disjoint.
    std::vector<Polytope> triangulation() const;

    // Queries -------------------------------------------------------------------------------------

    /// Whether the polygon holds all the points (..., 2): some vertex triangle of a piece holds
    /// each (CORALean's containsPoint). tol: as CORA's, only 0 is modelled (default: unset).
    bool contains(const Tensor &p, Tol tol = std::nullopt) const;

    /// Whether the polygons meet: false only if every pair of pieces is separated strictly along
    /// the normal of a vertex difference (CORALean's isDisjoint).
    bool is_intersecting(const Polygon &S, Tol tol = std::nullopt) const;

    /// Whether every piece of each polygon lies in a piece of the other (CORALean's
    /// isequalPieces): true means the same region; false may still be the same region.
    bool isequal(const Polygon &S, Tol tol = std::nullopt) const;

    /// Whether the polygon is convex: no piece, or one piece holds all (CORALean's isConvex);
    /// true means convex, false may still be convex.
    bool is_convex() const;

    /// Whether the polygon is also a set of the CORA type: "emptySet" (no piece has a vertex).
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

    /// The area of the region (CORA's volume_): the pieces made disjoint by clipping, summed.
    double volume() const;

    // Combining with other sets -------------------------------------------------------------------

    /// The Minkowski sum with a polygon (every pair of pieces, all vertex sums) or a point (2).
    Polygon plus(const Polygon &S) const;
    Polygon plus(const Tensor &p) const;

    /// The union (CORA's or): the pieces of both.
    Polygon unite(const Polygon &S) const;

    /// The intersection (CORA's and_): every pair of pieces clipped, exactly up to rounding.
    Polygon intersect(const Polygon &S) const;

    /// The difference (CORA's subtract and mldivide): every piece cut by the pieces of S.
    Polygon set_difference(const Polygon &S) const;

    /// The convex hulls of every pair of pieces, one of each (CORA's linComb).
    Polygon linear_combination(const Polygon &S) const;

  protected:
    // The unchecked implementations of the ContSet interface and its generic operations.
    int64_t dim_impl() const override { return 2; }
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Set linear_map_impl(const Tensor &M) const override;
    Set translate_impl(const Tensor &v) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Outline> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;
    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;

  private:
    /// The pieces as they are, checked by the public constructor.
    struct Unchecked {};
    Polygon(Unchecked, std::vector<Polytope> pieces, Tensor like);
    friend Polygon priv_polygon(std::vector<Polytope> pieces, const Tensor &like);

    std::vector<Polytope> pieces_;
    Tensor like_; // (0, 2), of the pieces' backend and dtype: what new tensors are made like
};

/// M * pgon, the image under the matrix M (2, 2).
inline Set operator*(const Tensor &M, const Polygon &P) { return P.linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
