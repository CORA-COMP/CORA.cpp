// priv - the private functions of polytope, as CORA's @polytope/private
//
// Not for callers of the class: the operations of the class use them. The algorithms that branch
// on the data (compaction, vertex enumeration, linear programs) run on the host on one polytope,
// as lists of rows; the operations of the class turn the result back into tensors on the backend
// of their input.
//
// Syntax:   HostH H = priv_host_h(P.constraints());   bool empty = priv_is_empty(H);
//           auto s = priv_support(H, direction, true);   Rows V = priv_vertices(H);
// See also: polytope.h
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/cont_set/polytope/polytope.h"

#include <optional>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// Host form ---------------------------------------------------------------------------------------

/// A row of numbers (a halfspace normal or a point), and a list of rows.
using Row = std::vector<double>;
using Rows = std::vector<Row>;

/// A single polytope on the host in halfspace form: A x <= b, Ae x = be with n columns.
struct HostH {
    int64_t n = 0;
    Rows A;
    Row b;
    Rows Ae;
    Row be;
};

/// The halfspace form of a single (unbatched) polytope on the host; Error::not_supported for a
/// batch.
HostH priv_host_h(const Polytope::Constraints &C);

/// The tensors (rows, n) and (rows, 1) of a host polytope, on the backend of `like`.
Polytope::Constraints priv_tensors_h(const Tensor &like, const HostH &H);

/// The points of a vertex tensor (n, p) as p rows of n numbers; Error::not_supported for a batch.
Rows priv_host_points(const Tensor &V);

/// The matrix (n, p) of p points of n numbers, on the backend of `like`.
Tensor priv_points_tensor(const Tensor &like, int64_t n, const Rows &points);

/// The column (n, 1) of the numbers, on the backend of `like`.
Tensor priv_column_tensor(const Tensor &like, const Row &values);

/// The dot product and the Euclidean norm of rows (summed in order, as MATLAB's loops do).
double priv_dot(const Row &a, const Row &b);
double priv_norm(const Row &a);

/// The solution of the square system M x = r by elimination with partial pivoting; none if a
/// pivot is not larger than pivot_tol times the largest entry of M.
std::optional<Row> priv_solve(Rows M, Row r, double pivot_tol);

// Normalization and compaction --------------------------------------------------------------------

/// CORA's priv_normalizeConstraints: "b" scales every row so that its nonzero offset is +-1 (an
/// equality's is 1); "A" scales every row to unit length (a zero row stays).
void priv_normalize_offsets(HostH &H);
void priv_normalize_rows(HostH &H);

/// The equalities as pairs of inequalities (CORA's priv_equalityToInequality): [A; Ae; -Ae].
std::pair<Rows, Row> priv_equality_to_inequality(const HostH &H);

/// CORA's priv_compact_zeros: drops the rows 0 <= b and 0 = 0; true if one is infeasible
/// (everything is then cleared).
bool priv_compact_zeros(HostH &H, double tol);

/// CORA's priv_compact_toEquality: the pairs a x <= c, -a x <= -c of normalized rows become the
/// equality a x = c.
void priv_compact_to_equality(HostH &H, double tol);

/// CORA's priv_compact_alignedIneq: of rows with one normal only the smallest offset stays (in
/// the order of sorting the rows); the equalities' version returns true if two aligned ones differ.
void priv_compact_aligned_inequalities(HostH &H, double tol);
bool priv_compact_aligned_equalities(HostH &H);

/// The minimal rows of CORA's priv_compact_1D, priv_compact_2D and priv_compact_nD (an empty
/// result is returned as true) and of priv_compact_all (which also says if the result is minimal).
bool priv_compact_1d(HostH &H, double tol);
bool priv_compact_2d(HostH &H);
bool priv_compact_nd(HostH &H, double tol);
bool priv_compact_all(HostH &H, double tol, bool &minimal);

// Linear programs ---------------------------------------------------------------------------------

/// The optimum of a linear function over a polytope: its value (+-Inf if unbounded, the sign of
/// the direction; -Inf for the maximum and +Inf for the minimum of an empty set) and a point.
struct Support {
    enum class Status { optimal, infeasible, unbounded } status = Status::infeasible;
    double value = 0.0;
    Row x;
};

/// max (or min) of direction'x over {A x <= b, Ae x = be}; without any row unbounded unless the
/// direction is zero (CORA's priv_supportFunc).
Support priv_support(const HostH &H, const Row &direction, bool upper);

/// Whether the polytope has no point (CORA's priv_representsa_emptySet, by a primal program).
bool priv_is_empty(const HostH &H);

/// Whether the (nonempty) polytope is bounded: the support function along the axes and the
/// all-ones negative is finite (CORA's isBounded).
bool priv_is_bounded(const HostH &H);

/// The center of the largest ball (CORA's center 'chebyshev'): the point, or none if the set is
/// empty, or the point of NaNs if the radius is unbounded.
std::optional<Row> priv_chebyshev_center(const HostH &H);

/// Whether the point x is in conv(points) (points as rows), by a feasibility program.
bool priv_in_hull(const Rows &points, const Row &x);

/// The indices of the extreme points of the cloud, in order; of equal points the first is kept.
std::vector<int64_t> priv_extreme_points(const Rows &points);

// Vertices and facets -----------------------------------------------------------------------------

/// The vertices of a 1D polytope (CORA's priv_vertices_1D): none if empty, else one or two, -Inf
/// and Inf for the unbounded sides.
Row priv_vertices_1d(const HostH &H, bool &empty);

/// The vertices of a bounded polytope in several dimensions, by combinations of its active
/// rows; empty and unbounded sets are Error::not_supported / an empty list.
Rows priv_vertices(const HostH &H);

/// The halfspace form of the convex hull of the points (rows of n numbers): CORA's constraints
/// of a vertex polytope.
HostH priv_constraints_of_points(const Tensor &like, const Rows &points, int64_t n);

// Distance ----------------------------------------------------------------------------------------

/// The Euclidean distance of x to the nonempty polytope (CORA's aux_distPolyPoint: the quadratic
/// program min y'y - 2 x'y, solved as a least-distance problem by nonnegative least squares).
double priv_distance_to_polytope(const HostH &H, const Row &x);

// Shared pieces of the operations -----------------------------------------------------------------

/// Whether the polytope is R^n (CORA's representsa fullspace with tol 0): all rows trivially
/// satisfied; of a batch, every member.
bool priv_is_fullspace(const Polytope &P);

/// Whether the vertex form has no vertex.
bool priv_has_no_vertices(const Polytope &P);

/// Whether the polytope has no point (CORA's representsa emptySet): no vertices, or no solution
/// of the halfspaces; a single polytope.
bool priv_polytope_is_empty(const Polytope &P);

/// Whether the polytope is bounded (CORA's isBounded): a vertex form always, except in 1D with
/// an infinite vertex; a halfspace form by programs. The empty set is bounded.
bool priv_polytope_is_bounded(const Polytope &P);

/// The least and the largest value of every coordinate over the polytope (2n programs, CORA's
/// priv_box_H); none if it is empty. The bounds may be infinite.
std::optional<std::pair<Row, Row>> priv_box_bounds(const HostH &H);

/// The vertices of a single polytope as rows (CORA's vertices_): a vertex polytope's without the
/// vertices that are not extreme, a halfspace polytope's by enumeration; none for the empty set.
Rows priv_polytope_vertices(const Polytope &P);

/// The polytope of a box (n, 1) (CORA's interval/polytope.m for the interval's rows).
Polytope priv_polytope_of_interval(const Interval &I);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
