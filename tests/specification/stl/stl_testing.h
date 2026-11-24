// stl_testing - what the tests of the STL formulas share
//
// CORA.cpp only: test helpers
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/rng.h"
#include "cora/specification/stl.h"

#include <cmath>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace stl_test {

using cora::Stl;
using cora::StlCmp;
using cora::StlExpr;
using cora::StlExprKind;
using cora::StlKind;

/// The value of an expression at the state s (the variable of index i reads s[i]).
inline double eval(const StlExpr &e, const std::vector<double> &s) {
    switch (e.kind()) {
    case StlExprKind::variable:
        return s[static_cast<std::size_t>(e.index())];
    case StlExprKind::number:
        return e.value();
    case StlExprKind::add:
        return eval(e.left(), s) + eval(e.right(), s);
    case StlExprKind::sub:
        return eval(e.left(), s) - eval(e.right(), s);
    case StlExprKind::mul:
        return eval(e.left(), s) * eval(e.right(), s);
    case StlExprKind::pow:
        return std::pow(eval(e.left(), s), static_cast<double>(e.exponent()));
    }
    return 0;
}

/// Whether a formula without temporal operators holds at the state s.
inline bool holds(const Stl &phi, const std::vector<double> &s) {
    // the logic by its truth tables, a comparison by the values of its sides
    switch (phi.kind()) {
    // the constants and the connectives
    case StlKind::true_:
        return true;
    case StlKind::false_:
        return false;
    // the connectives by their truth tables
    case StlKind::not_:
        return !holds(phi.lhs(), s);
    case StlKind::and_:
        return holds(phi.lhs(), s) && holds(phi.rhs(), s);
    case StlKind::or_:
        return holds(phi.lhs(), s) || holds(phi.rhs(), s);
    // a comparison by the values of its two sides
    case StlKind::comparison: {
        const double a = eval(phi.expr_lhs(), s), b = eval(phi.expr_rhs(), s);
        switch (phi.cmp()) {
        case StlCmp::lt:
            return a < b;
        case StlCmp::le:
            return a <= b;
        case StlCmp::gt:
            return a > b;
        case StlCmp::ge:
            return a >= b;
        }
        return false;
    }
    default:
        return false;
    }
}

/// A random formula of the given depth over x1, x2 with true, false, comparisons, &, | and ~; the
/// comparisons are against the integers -1, 0, 1, so that states on a grid hit the boundaries.
inline Stl random_formula(cora::Rng &rng, int depth) {
    const auto x = StlExpr::variables("x", 2);
    // three uniforms decide the shape, the comparison and the constant
    double u[3];
    rng.uniform(u, 3, 0.0, 1.0);
    if (depth == 0 || u[0] < 0.15) {
        const double c = std::floor(u[2] * 3) - 1;
        switch (static_cast<int>(u[1] * 6)) {
        case 0:
            return x[0] < c;
        case 1:
            return x[0] <= c;
        case 2:
            return x[1] > c;
        case 3:
            return x[1] >= c;
        case 4:
            return x[0] - x[1] < c;
        default:
            return u[2] < 0.5 ? Stl(true) : Stl(false);
        }
    }
    // an inner node: ~a, a & b or a | b
    const Stl a = random_formula(rng, depth - 1);
    if (u[1] < 0.3) return ~a;
    const Stl b = random_formula(rng, depth - 1);
    return u[1] < 0.65 ? (a & b) : (a | b);
}

/// The states with coordinates in {-1.5, -1, ..., 1.5}: they meet every comparison's boundary.
inline std::vector<std::vector<double>> states() {
    std::vector<std::vector<double>> out;
    for (int i = -3; i <= 3; ++i)
        for (int j = -3; j <= 3; ++j) out.push_back({i / 2.0, j / 2.0});
    return out;
}

/// Whether two formulas without temporal operators hold at the same states.
inline bool equivalent(const Stl &a, const Stl &b) {
    for (const auto &s : states())
        if (holds(a, s) != holds(b, s)) return false;
    return true;
}

} // namespace stl_test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
