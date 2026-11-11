// level_set - the class of level sets {x | f_i(x) ~_i 0}, as CORA's levelSet
//
// The constraints are expressions of the state (Expr) with one comparison each: f_i(x) <= 0,
// f_i(x) < 0 or f_i(x) == 0, all of which must hold. CORA keeps them as symbolic equations and
// their function handles; the partial derivatives are formed once, for the contractors and the
// guard methods. An operation that needs the range of f over a box evaluates the expression over
// it (Expr::enclose), so a range is an enclosure and not exact. The set has no interval,
// center, support function, vertices or random points (CORA raises noops for them as well).
// Not ported: CORA's solved forms (the symbolic solution for one variable, `solved`), plot,
// enclosePoints, generateRandom, isequal, compact_, and the methods split, taylor and linear of
// tightenDomain (docs/cora-differences.md).
//
// Syntax:     LevelSet ls(f, n, CompOp::eq);   LevelSet ls({f1, f2}, n, {CompOp::le, CompOp::eq});
// Operations: contains, is_intersecting, tighten_domain, complement, lift_impl, split_equality (one
//             file each in src/cont_set/level_set/, named as CORA's); the constructor checks its
//             input (level_set.cpp)
// See also:   cora/global/expr.h, cora/cont_set/cont_set.h, cora/cont_set/interval/interval.h
//
// CORA: contSet/@levelSet/levelSet.m
// CORA: contSet/@levelSet/dim.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/dimension_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/global/expr.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The comparison of a constraint with 0: CORA's '<=', '<' and '=='.
enum class CompOp { le, lt, eq };

class LevelSet : public detail::DimensionSet {
  public:
    static constexpr ClassId kClassId = ClassId::level_set;

    /// The constraints f_i(x) ops_i 0 of the n states (the variables of Expr, 0-based).
    LevelSet(std::vector<Expr> eq, int64_t n, std::vector<CompOp> ops);

    /// One constraint f(x) op 0.
    LevelSet(const Expr &eq, int64_t n, CompOp op);

    /// The empty set (the constraint 1 <= 0) and the whole space (0 x - 1 <= 0) of n dimensions,
    /// as CORA's levelSet.empty and levelSet.Inf.
    static LevelSet empty(int64_t n = 0);
    static LevelSet inf(int64_t n = 0);

    const std::vector<Expr> &equations() const noexcept { return eq_; }
    const std::vector<CompOp> &comp_ops() const noexcept { return ops_; }
    int64_t num_equations() const noexcept { return int64_t(eq_.size()); }

    /// The derivative of constraint i with respect to state j (CORA's der.grad).
    const Expr &gradient(int64_t i, int64_t j) const;

    /// The constraints at the points x (..., n, N): the values (..., m, N) (CORA's funHan).
    Tensor fun_han(const Tensor &x) const;

    /// A box (..., m, 1) that holds the values of the constraints over the box I (n, 1).
    Interval fun_han(const Interval &I) const;

    // Queries -------------------------------------------------------------------------------------

    /// Whether all points (n, N) satisfy every constraint: f_i < tol for '<' and '<=' (up to
    /// within_tol), |f_i| <= tol for '=='. tol default: the repository tolerance.
    bool contains(const Tensor &points, Tol tol = std::nullopt) const;

    /// Whether the box I is in the set by the interval evaluation of f over it: the upper bound
    /// of f_i below 0, for '==' both bounds within tol of 0. True is certain, false says nothing.
    bool contains(const Interval &I, Tol tol = std::nullopt) const;

    /// Whether the set may meet S (CORA's isIntersecting_ with type "approx"): the interval
    /// evaluation of f over the box of S. False is certain, true says nothing.
    bool is_intersecting(const ContSet &S, Tol tol = std::nullopt) const;

    // Operations ----------------------------------------------------------------------------------

    /// The box I contracted to the points that can satisfy every constraint f_i = 0 (CORA's
    /// tightenDomain, "forwardBackward"): all constraints must be '=='. None if the contraction
    /// shows that no point of I satisfies them.
    std::optional<Interval> tighten_domain(const Interval &I,
                                           const std::string &type = "forwardBackward") const;

    /// The complement of a single constraint with '<' or '<=': -f <= 0 or -f < 0 (CORA's not).
    LevelSet complement() const;

    /// The set split into its equality constraints and the others (CORA's splitEquality); None
    /// for a part without constraints.
    std::pair<std::optional<LevelSet>, std::optional<LevelSet>> split_equality() const;

  protected:
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;

  private:
    std::vector<Expr> eq_;
    std::vector<CompOp> ops_;
    std::vector<std::vector<Expr>> grad_; ///< grad_[i][j] = d f_i / d x_j
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
