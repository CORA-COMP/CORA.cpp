// priv - the private functions of polygon: the pieces on the host, CORALean's tests, clipping
//
// Not for callers of the class. A piece on the host is its vertex list in stored order. The
// CORALean part rounds as CORALean's float definitions at nearest (sums nested to the right and
// ended by + 0, maximum and minimum as CORALean's, no fused multiply-add); the clipping part is
// CORA.cpp's exact geometry of convex pieces, counter-clockwise.
//
// Syntax:   Piece v = priv_host(P);   bool b = priv_piece_holds(v, x);   Piece c = priv_clip(v, a,
// b); See also: polygon.h, priv_lean.cpp, priv_clip.cpp, priv_split.cpp
//
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/polygon/polygon.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A piece on the host: its vertices in stored order.
using Piece = std::vector<Point>;

// Crossing ----------------------------------------------------------------------------------------

/// The vertices of a piece, and a piece of the vertices made like `like` ((0, 2)).
Piece priv_host(const Polytope &P);
std::vector<Piece> priv_hosts(const Polygon &pgon);
Polytope priv_piece(const Piece &v, const Tensor &like);

/// The polygon of the pieces, unchecked; like: the (0, 2) tensor new tensors are made like.
Polygon priv_polygon(std::vector<Polytope> pieces, const Tensor &like);
Polygon priv_polygon_of(const std::vector<Piece> &pieces, const Tensor &like);

// CORALean's float definitions at nearest -----------------------------------------------------

/// CORALean's maximum and minimum (a NaN stays).
double priv_maximum(double a, double b);
double priv_minimum(double a, double b);

/// (q - p) x (r - p) as CORALean's orientBounds: six products summed to the right, then + 0.
double priv_orient(Point p, Point q, Point r);

/// Whether no two of the points agree (CORALean's distinctPoints): a triangle with a repeated
/// corner rounds its zero orientation at nearest and is skipped.
bool priv_distinct(Point a, Point b, Point c);

/// Whether a vertex triangle of the piece holds x (CORALean's pieceHolds), and whether every
/// vertex of p lies in q that way (pieceSubset).
bool priv_piece_holds(const Piece &p, Point x);
bool priv_piece_subset(const Piece &p, const Piece &q);

/// The upper bound of a . x over the piece (VPolytope.boundDot): the largest a0 x0 + (a1 x1 + 0)
/// from the last vertex, + 0 for the empty error box.
double priv_bound_dot(const Piece &p, Point a);

// Exact geometry of convex pieces (CORA.cpp only) ---------------------------------------------

/// The extreme vertices of the piece, counter-clockwise from the lowest-leftmost (monotone chain).
Piece priv_hull(const Piece &p);

/// The part of the counter-clockwise convex polygon c in {x | a . x <= b} (Sutherland-Hodgman).
Piece priv_clip(const Piece &c, Point a, double b);

/// r \ q for convex counter-clockwise r and q: disjoint convex parts, one per edge of q.
std::vector<Piece> priv_minus_piece(const Piece &r, const Piece &q);

/// Disjoint convex counter-clockwise parts whose union is the union of the pieces.
std::vector<Piece> priv_disjoint(const std::vector<Piece> &pieces);

/// The signed area of a closed vertex loop (shoelace): positive counter-clockwise.
double priv_area(const Piece &loop);

/// Convex pieces of the region inside the simple boundary loop (either orientation): the loop
/// itself if it is convex, else ear-clipping triangles merged greedily into convex pieces.
std::vector<Piece> priv_convex_split(Piece loop);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
