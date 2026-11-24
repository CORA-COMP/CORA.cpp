// stl - signal temporal logic formulas and their arithmetic, as CORA's stl
//
// A formula of signal temporal logic over the state x of a trace: true, false, a comparison of two
// arithmetic expressions, ~, &, |, until, release, finally, globally and next, with closed time
// windows (CORA's `implies` builds ~a | b and has no node of its own). Both classes are immutable
// values that share their sub-formulas. The syntax and every operation follow CORALean's float
// definitions of STL.Expr and STL.Tree (Specification/STL/), with the numbers and times doubles at
// nearest rounding; CORA's single stl class is split into StlExpr (the arithmetic, never a
// formula) and Stl (the formula), so an operation on the wrong kind cannot be written.
//
// Syntax:     auto x = StlExpr::variables("x", 2);   (x1, x2: StlExpr)
//             Stl phi = (x[0] < 5.0) & (x[0] - x[1] >= 0.0).finally(StlInterval(0, 2));
//             Stl psi = phi.until(Stl(true), StlInterval(1, 3));
// Operations: the node builders are in stl.cpp (and_, or_, not_, implies, lt, le, gt, ge, plus,
//             minus, times, isequal); the others are one file each in src/specification/stl/
// See also: specification/stl_interval.h
//
// CORA: specification/@stl/stl.m
// CORA: specification/@stl/and.m
// CORA: specification/@stl/or.m
// CORA: specification/@stl/not.m
// CORA: specification/@stl/implies.m
// CORA: specification/@stl/lt.m
// CORA: specification/@stl/le.m
// CORA: specification/@stl/gt.m
// CORA: specification/@stl/ge.m
// CORA: specification/@stl/plus.m
// CORA: specification/@stl/minus.m
// CORA: specification/@stl/times.m
// CORA: specification/@stl/isequal.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/specification/stl_interval.h"

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Stl;
struct StlAccess;

/// A comparison of a predicate: <, <=, >, >= (CORALean's STL.Cmp).
enum class StlCmp { lt, le, gt, ge };

/// The comparison read from the other side: a < b is b > a.
StlCmp stl_cmp_dual(StlCmp op);

/// The comparison that holds exactly where op does not.
StlCmp stl_cmp_negate(StlCmp op);

/// The names ordered as CORA's getVariables does: shortest first, then alphabetically, each once.
void stl_sort_names(std::vector<std::string> &names);

/// Whether the formula is turned into normal form from an already negation-normal formula
/// (CORA's `nnf` flag of conjunctiveNormalForm and disjunctiveNormalForm).
enum class StlNnf { compute, given };

/// The kinds of an arithmetic expression node.
enum class StlExprKind { variable, number, add, sub, mul, pow };

/// An arithmetic expression over the state: a variable x(i), a number, and sums, differences,
/// products and integer powers of them (CORALean's STL.Expr). A number converts to an expression.
class StlExpr {
  public:
    /// The number c (CORA: a numeric operand).
    StlExpr(double c = 0.0);

    /// The variable of that name and 0-based index (CORA names the i-th of stl('x', n) x<i+1>).
    static StlExpr variable(const std::string &name, int64_t index);

    /// The n variables name1, ..., namen (CORA's stl(name, n)); Error::wrong_input_in_constructor
    /// for an empty name or n < 1.
    static std::vector<StlExpr> variables(const std::string &name, int64_t n);

    // Inspection ----------------------------------------------------------------------------------
    StlExprKind kind() const;

    /// The value of a number.
    double value() const;

    /// The name of a variable, e.g. "x2", and its 0-based index.
    const std::string &name() const;
    int64_t index() const;

    /// The operands of a sum, difference or product, the base of a power.
    StlExpr left() const;
    StlExpr right() const;

    /// The exponent of a power.
    int64_t exponent() const;

    /// The names of the variables in the expression, as getVariables orders them.
    std::vector<std::string> variable_names() const;

    /// Whether the expression is built from variables and numbers by sums, differences and
    /// products with a number (the predicate is then linear, CORA's isLinearPredicate).
    bool is_linear() const;

    // Arithmetic ----------------------------------------------------------------------------------
    /// The sum of two expressions; a numeric left operand moves to the right.
    StlExpr plus(const StlExpr &other) const;

    /// The difference, the operands in the given order.
    StlExpr minus(const StlExpr &other) const;

    /// The product, the numeric operand first (if both are expressions, the right one first).
    StlExpr times(const StlExpr &other) const;

    /// The power with the integer exponent k; Error::not_supported for a non-integer exponent.
    StlExpr power(double k) const;

    // Comparisons: the formula `this op other` ----------------------------------------------------
    /// A number on the right stays there; one on the left swaps sides (dual comparison); two
    /// expressions become `this - other op 0`.
    Stl lt(const StlExpr &other) const;
    Stl le(const StlExpr &other) const;
    Stl gt(const StlExpr &other) const;
    Stl ge(const StlExpr &other) const;

    /// Same structure; sums and products compare up to the order of their operands.
    bool isequal(const StlExpr &other) const;

    /// "x1 + 2*x2", CORA's display of the expression.
    std::string to_string() const;

  private:
    struct Node;
    friend struct StlAccess;
    std::shared_ptr<const Node> node_;
    explicit StlExpr(std::shared_ptr<const Node> node) : node_(std::move(node)) {}
    friend class Stl;
};

/// The kinds of a formula node.
enum class StlKind {
    true_,
    false_,
    comparison,
    not_,
    and_,
    or_,
    until,
    release,
    finally,
    globally,
    next
};

/// The times of the atomic formulas of an identified formula (CORA's getTimes): entry id - 1 holds
/// the times formula number id is looked at, as points or as time intervals; both lists have the
/// same length.
struct StlTimes {
    std::vector<std::vector<StlInterval>> time_int;
    std::vector<std::vector<double>> time_point;
};

/// A formula of signal temporal logic (CORALean's STL.Tree).
class Stl {
  public:
    /// The formula true or false (CORA's stl(true), stl(false)).
    explicit Stl(bool value);

    // Construction --------------------------------------------------------------------------------
    /// The comparison `e1 op e2` with the sides arranged as lt, le, gt, ge do (CORA's lt.m).
    static Stl compare(StlCmp op, const StlExpr &e1, const StlExpr &e2);

    /// ~this, this & other, this | other, ~this | other.
    Stl not_() const;
    Stl and_(const Stl &other) const;
    Stl or_(const Stl &other) const;
    Stl implies(const Stl &other) const;

    /// X[a] this: this holds a time units later (a >= 0 and not NaN, else
    /// Error::wrong_input_in_constructor).
    Stl next(double a) const;

    /// F[time] this, G[time] this, this U[time] other, this R[time] other; the window must not be
    /// empty (Error::wrong_value), its ends' openness is ignored as CORALean does.
    Stl finally(const StlInterval &time) const;
    Stl globally(const StlInterval &time) const;
    Stl until(const Stl &other, const StlInterval &time) const;
    Stl release(const Stl &other, const StlInterval &time) const;

    /// The same formula carrying the identifier id (CORA's id of an atomic sub-formula).
    Stl with_id(int64_t id) const;

    // Inspection ----------------------------------------------------------------------------------
    StlKind kind() const;

    /// The operand of ~, finally, globally, next; the left operand of and, or, until, release.
    Stl lhs() const;
    Stl rhs() const;

    /// The sides and the comparison of a comparison node.
    StlExpr expr_lhs() const;
    StlExpr expr_rhs() const;
    StlCmp cmp() const;

    /// The window of a temporal operator (a point for next); from() and to() are its bounds.
    const StlInterval &interval() const;
    double from() const;
    double to() const;

    /// Whether a temporal operator occurs in the formula.
    bool temporal() const;

    /// The identifier of an atomic sub-formula (0 for none).
    int64_t id() const;

    // Operations ----------------------------------------------------------------------------------
    /// Same structure; and, or and comparisons' operands compare as isequal of CORA, window bounds
    /// by value (identifiers are ignored).
    bool isequal(const Stl &other) const;

    /// The names of the variables in the formula, shortest name first, then alphabetically.
    std::vector<std::string> get_variables() const;

    /// The times at which the identified atomic sub-formulas are evaluated.
    StlTimes get_times() const;

    /// The latest and the earliest time the formula looks at.
    double maximum_time() const;
    double minimum_time() const;

    /// Only until and the logic: release, finally, globally and next rewritten.
    Stl desugar() const;

    /// Negations only at atomic formulas, which negate into the comparison that fails.
    Stl negation_normal_form() const;

    /// The formula without true and false under the logic and temporal operators.
    Stl eliminate_true_false() const;

    /// The conjunctive and the disjunctive normal form over the maximal non-logic parts.
    Stl conjunctive_normal_form(StlNnf nnf = StlNnf::compute) const;
    Stl disjunctive_normal_form(StlNnf nnf = StlNnf::compute) const;

    /// The formula with its non-temporal sub-formulas numbered 1, 2, ... left to right, and the
    /// numbered sub-formulas.
    std::pair<Stl, std::vector<Stl>> assign_identifiers() const;

    /// Nested next operators merged and pushed through temporal & and |.
    Stl combine_next() const;

    /// The conjuncts ("cnf") or the disjuncts ("dnf") of the formula; Error::wrong_value for
    /// another type.
    std::vector<Stl> get_clauses(const std::string &type = "cnf") const;

    /// Whether the formula is a conjunction of comparisons; and whether those are linear.
    bool is_predicate() const;
    bool is_linear_predicate() const;

    /// CORA's display: "(x1 < 5 & F[0, 2](x2 >= 0))".
    std::string to_string() const;

    bool operator==(const Stl &other) const { return isequal(other); }
    bool operator!=(const Stl &other) const { return !isequal(other); }

  private:
    struct Node;
    friend struct StlAccess;
    std::shared_ptr<const Node> node_;
    explicit Stl(std::shared_ptr<const Node> node) : node_(std::move(node)) {}
};

// Operators ---------------------------------------------------------------------------------------

inline StlExpr operator+(const StlExpr &a, const StlExpr &b) { return a.plus(b); }
inline StlExpr operator-(const StlExpr &a, const StlExpr &b) { return a.minus(b); }
inline StlExpr operator*(const StlExpr &a, const StlExpr &b) { return a.times(b); }
inline Stl operator<(const StlExpr &a, const StlExpr &b) { return a.lt(b); }
inline Stl operator<=(const StlExpr &a, const StlExpr &b) { return a.le(b); }
inline Stl operator>(const StlExpr &a, const StlExpr &b) { return a.gt(b); }
inline Stl operator>=(const StlExpr &a, const StlExpr &b) { return a.ge(b); }
inline Stl operator&(const Stl &a, const Stl &b) { return a.and_(b); }
inline Stl operator|(const Stl &a, const Stl &b) { return a.or_(b); }
inline Stl operator~(const Stl &a) { return a.not_(); }

std::ostream &operator<<(std::ostream &os, const StlExpr &e);
std::ostream &operator<<(std::ostream &os, const Stl &phi);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
