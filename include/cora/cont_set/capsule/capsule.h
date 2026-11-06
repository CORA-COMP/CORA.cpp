// capsule - the class of capsules, as CORA's capsule
//
// A capsule is the set of points within the radius r of the segment c - g to c + g: C = {x |
// min_{t in [-1, 1]} |x - (c + t g)|_2 <= r}. The center c and the generator g are columns
// (..., n, 1), the radius r is (..., 1, 1) and nonnegative; r = 0 is a segment, g = 0 a ball.
// Immutable: every operation returns a new set. A negative radius (-1) marks the empty set, as
// in CORALean; empty(n) is such a set, and its center has no columns (n, 0), as in CORA.
//
// Syntax:     Capsule C(c, g, r);   Capsule C(c);   Capsule::empty(n);   Capsule::origin(n);
// Operators:  M * C (linear_map)
// Operations: constructors (capsule.cpp), empty, origin, enclose_points, center, interval,
//             is_bounded, is_full_dim (the ContSet entries), isequal, contains, is_intersecting,
//             convex_hull, enclose, linear_map, lift, display (one file each). The other
//             operations of the ContSet interface (support_function, rand_point, project,
//             vertices) are Error::not_supported until they are ported.
// See also:   cora/cont_set/cont_set.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: contSet/@capsule/capsule.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Capsule : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::capsule;

    /// The capsule of the center c (..., n, 1), the generator g (..., n, 1) and the radius r
    /// (..., 1, 1) >= 0; the generator defaults to 0 and the radius to 0, so a center alone is a
    /// point. A number r is the radius of every member of the batch.
    explicit Capsule(const Tensor &c);
    Capsule(const Tensor &c, Tensor g);
    Capsule(const Tensor &c, Tensor g, Tensor r);
    Capsule(const Tensor &c, Tensor g, double r);

    /// The stored center c, generator g and radius r (-1 for the empty set).
    const Tensor &c() const noexcept { return c_; }
    const Tensor &generator() const noexcept { return g_; }
    const Tensor &r() const noexcept { return r_; }

    /// The empty capsule of n >= 0 dimensions (CORA's capsule.empty): the radius -1, as in
    /// CORALean (CORA's has a center without columns, which has no radius).
    static Capsule empty(int64_t n = 0);

    /// The capsule that is the origin of R^n, n >= 1 (CORA's capsule.origin).
    static Capsule origin(int64_t n);

    /// Whether the capsule is empty (a negative radius); for a batch, whether every member is.
    bool is_empty() const;

    /// A capsule that contains the points (..., n, N) (CORA's enclosePoints): the mean as center,
    /// the generator to the farthest point, the largest distance from the segment as radius.
    static Capsule enclose_points(const Tensor &points);

    // Queries -------------------------------------------------------------------------------------

    /// Whether the capsule equals S: the centers, generators and radii agree as within_tol, or
    /// both are empty; a different dimension is false. For a batch, whether every member does.
    /// tol default: the repository tolerance.
    bool isequal(const Capsule &S, Tol tol = std::nullopt) const;

    /// Whether the capsule contains the points p (..., n, N), a capsule, a zonotope or a box, up
    /// to tol in the distance to the axis; for a batch, every member.
    bool contains(const Tensor &p, Tol tol = std::nullopt) const;
    bool contains(const Capsule &S, Tol tol = std::nullopt) const;
    bool contains(const Zonotope &Z, Tol tol = std::nullopt) const;
    bool contains(const Interval &I, Tol tol = std::nullopt) const;

    /// Whether the capsule meets the points p (all of them in it) or a capsule: the least
    /// distance of the two axes is at most the sum of the radii, up to tol. Exact.
    bool is_intersecting(const Tensor &p, Tol tol = std::nullopt) const;
    bool is_intersecting(const Capsule &S, Tol tol = std::nullopt) const;

    // Combining with other sets -------------------------------------------------------------------

    /// A capsule that encloses this one and S (CORA's convHull_ of two capsules): the axis runs
    /// between the two centers, and the radius is the larger of |g| + r. The hull with an empty
    /// capsule is the other one (CORA returns nothing there).
    Capsule convex_hull(const Capsule &S) const;

    /// CORA's enclose: for one capsule the convex hull. CORA's three-argument form returns the
    /// hull of C and M * Cplus and never adds Cplus, so it does not enclose M * C + Cplus as
    /// documented; with (M, Cplus) it is Error::not_supported until plus is ported.
    Capsule enclose(const Capsule &S) const;
    Capsule enclose(const Tensor &M, const Capsule &Cplus) const;

    /// M * C for a matrix M (..., m, n) (CORA's mtimes): center and generator map by M, the
    /// radius scales by the Frobenius norm of M, an upper bound of its operator norm.
    Capsule linear_map(const Tensor &M) const;

    /// The capsule with its coordinates permuted: coordinate i of this one becomes coordinate
    /// dims[i] (CORA's lift_), N = dim() and dims a permutation (a capsule is bounded, so new
    /// dimensions cannot be added).
    Capsule lift(int64_t N, const std::vector<int64_t> &dims) const;

  protected:
    int64_t dim_impl() const override { return c_.shape()[c_.shape().size() - 2]; }
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;

    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;
    Set linear_map_impl(const Tensor &M) const override;
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Set enclose_impl(const ContSet &S2) const override;

  private:
    /// The parts as they are, without the checks of the constructor (the empty capsule has the
    /// radius -1).
    struct Unchecked {};
    Capsule(Unchecked, Tensor c, Tensor g, Tensor r);

    Tensor c_, g_, r_;
};

/// M * C, as CORA writes it.
inline Capsule operator*(const Tensor &M, const Capsule &C) { return C.linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
