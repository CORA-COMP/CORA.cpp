// zonotope - lean::Zonotope, a zonotope of exact values plus an error box, computed by CORALean
//
// Syntax:     lean::Zonotope Z(c, G);   Z = Z.linear_map(M).plus(Z2).reduce(1);   Z.gather();
// Operations: one method per `zonotope.*` op of the oracle, named as CORA.cpp names CORA's (`and`
//             is intersect, `supportFunc` support_function), with CORA's option names, 0-based
//             indices and `tol` accepted at 0 only; `p` and `k` tune the approximations
// Parts:      nominal() the exact zonotope c + G[-1,1]^h, error() the box E that the sound float
//             operations add; together they enclose the real result (enclosure())
// Crossing:   round_to(dtype) rounds c and G and grows E; gather() returns the nominal part as
//             a cora::Zonotope and the error as a cora::Interval, exactly or throws
// See also:   lean/tensor.h, lean/interval.h, lean/request.h, cora/cont_set/zonotope/zonotope.h

#pragma once

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/tensor.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Zonotope {
  public:
    /// An exact zonotope (empty error box) in the current dtype.
    Zonotope(const cora::Tensor &c, const cora::Tensor &G);
    Zonotope(const cora::Zonotope &Z) : Zonotope(Z.center(), Z.generators()) {}

    Zonotope(Tensor c, Tensor G, Interval E);

    /// The exact part: c and G with an empty error box.
    Zonotope nominal() const;

    /// The center (n, 1), the generators (n, h), exact, and the error box (n, 1) alone.
    const Tensor &center() const noexcept { return c_; }
    const Tensor &generators() const noexcept { return G_; }
    const Interval &error() const noexcept { return E_; }

    /// nominal() + error(): a lean zonotope with an empty error box enclosing this one.
    Zonotope enclosure() const;

    /// c and G rounded into `dtype`; the rounding errors are added to the error box.
    Zonotope round_to(const std::string &dtype, const std::string &mode = "nearest") const;

    // Linear and Minkowski operations ----------------------------------------------------------

    /// M * Z for a matrix or an interval matrix.
    Zonotope linear_map(const Tensor &M) const;
    Zonotope linear_map(const Interval &M) const;

    /// The Minkowski sum with a zonotope or with a box.
    Zonotope plus(const Zonotope &Z2) const;
    Zonotope plus(const Interval &I) const;

    /// CORA's linComb with a zonotope Z2 that shares the generator factors of this one.
    Zonotope linear_combination(const Zonotope &Z2) const;

    /// The sum with the first k generator pairs added (CORA's exactPlus).
    Zonotope exact_plus(const Zonotope &Z2, int k) const;

    /// Order reduction to `order` with method "girard" or "combastel".
    Zonotope reduce(int order, const std::string &method = "girard") const;

    // Set operations ---------------------------------------------------------------------------

    Zonotope abs() const;
    Zonotope intersect(const Zonotope &Z2) const;
    Zonotope convex_hull(const Zonotope &Z2) const;
    Zonotope enclose(const Zonotope &Z2) const;
    Zonotope cart_prod(const Zonotope &Z2) const;

    /// The zonotope without zero generators ("zeros").
    Zonotope compact(const std::string &method = "zeros", double tol = 0) const;

    /// Z intersected with the strips |C x - y| <= phi at the gain Lambda.
    Zonotope intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                             const Tensor &Lambda) const;

    /// The image of the quadratic map x' Q_i x and of the cubic map T_ij x x x.
    Zonotope quad_map(const std::vector<Tensor> &Q) const;
    Zonotope cub_map(const std::vector<std::vector<Tensor>> &T) const;

    /// The oriented box around the zonotopes along `direction` (nothing for the zero direction).
    static std::optional<Zonotope> enclose_many(const std::vector<Zonotope> &Zs,
                                                const Tensor &direction, int p = 20);

    /// A zonotope enclosing the columns of `points`, by the method "stursberg".
    static Zonotope enclose_points(const Tensor &points, const Tensor &U,
                                   const std::string &method = "stursberg");

    /// The indices of the zonotopes that no other one is certified to contain.
    static std::vector<int64_t> filter_out(const std::vector<Zonotope> &Zs);

    static Zonotope origin(int64_t n);

    // Selection and reshaping ------------------------------------------------------------------

    Zonotope project(const std::vector<int64_t> &dims) const;
    Zonotope lift(int64_t N, const std::vector<int64_t> &dims) const;
    Zonotope project_high_dim(int64_t N, const std::vector<int64_t> &dims) const;
    /// The rotation by `angle` in the plane of axes i and j, with p Newton steps.
    Zonotope rotate(int p, int i, int j, double angle) const;

    /// The two halves along generator `gen`, along the first or along the longest generator.
    std::pair<Zonotope, Zonotope> split(int64_t gen) const;
    std::pair<Zonotope, Zonotope> split_first_gen() const;
    std::pair<Zonotope, Zonotope> split_longest_gen() const;

    // Predicates (exact answers) ---------------------------------------------------------------

    bool constr_sat(const Tensor &C, const Tensor &d) const;
    bool is_bounded(double tol = 0) const;
    bool is_full_dim(double tol = 0) const;
    bool isequal(const Zonotope &Z2, double tol = 0) const;

    /// type "point" or "origin".
    bool representsa(const std::string &type, double tol = 0) const;

    // Certified predicates: contains proves the containment, is_intersecting the disjointness ---

    Certified contains(const Zonotope &Z2, double tol = 0) const;
    Certified is_intersecting(const Zonotope &Z2, double tol = 0) const;

    // Numbers ----------------------------------------------------------------------------------

    /// An upper bound (type 2, mode "ub") of the norm; the support function along `dir`
    /// (type "upper"); the radius of a ball about the center, the length of every generator (k
    /// Newton steps); an upper bound on the least facet offset (nothing if there is none).
    double norm(int type = 2, const std::string &mode = "ub") const;
    double support_function(const Tensor &dir, const std::string &type = "upper") const;
    double radius(int k = 20) const;
    std::optional<double> minnorm(int p = 20) const;
    Tensor generator_length(int k = 20) const;

    /// The center as the oracle returns it (`zonotope.center`); center() is the stored one.
    Tensor center_value() const;

    /// The principal directions (n x n), nothing if there are none; the filter lengths default to
    /// CORA's n + 5 and n + 3.
    std::optional<Tensor> dominant_directions() const;
    std::optional<Tensor> dominant_directions(int filter_length1, int filter_length2) const;

    /// The corners (up to 2^20) as columns of V, and a box E for the rounding errors.
    Vertices vertices() const;

    /// The interval hull of the enclosure.
    Interval interval() const;

    /// The nominal part as a cora::Zonotope and the error as a cora::Interval (double, exact).
    std::pair<cora::Zonotope, cora::Interval> gather() const;

    Json to_json() const;
    static Zonotope from_json(const std::string &dtype, const Json &j);

  private:
    Tensor c_, G_;
    Interval E_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
