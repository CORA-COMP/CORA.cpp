// stl - the nodes of signal temporal logic: expressions, comparisons and the logic operators
//
// Syntax:   StlExpr e = x[0] + 2.0 * x[1];   Stl phi = (e < 5.0) & ~(x[0] >= 1.0);
// See also: stl.h
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

#include "cora/specification/stl.h"

#include "specification/stl/private/priv_node.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// MATLAB's num2str of a scalar: integers plainly, else at least 5 significant digits.
std::string aux_num2str(double x) {
    if (std::isnan(x)) return "NaN";
    if (std::isinf(x)) return x < 0 ? "-Inf" : "Inf";
    char buf[64];
    if (x == std::floor(x) && std::fabs(x) < 1e15) {
        std::snprintf(buf, sizeof buf, "%.0f", x == 0 ? 0.0 : x);
        return buf;
    }
    const int digits =
        std::min(16, std::max(static_cast<int>(std::floor(std::log10(std::fabs(x)))) + 5, 5));
    std::snprintf(buf, sizeof buf, "%.*g", digits, x);
    return buf;
}

} // namespace


// ===========================================  MAIN  =========================================== //

void stl_sort_names(std::vector<std::string> &names) {
    std::sort(names.begin(), names.end(), [](const std::string &a, const std::string &b) {
        return a.size() != b.size() ? a.size() < b.size() : a < b;
    });
    names.erase(std::unique(names.begin(), names.end()), names.end());
}

// The comparisons ---------------------------------------------------------------------------------

StlCmp stl_cmp_dual(StlCmp op) {
    // < and > swap, <= and >= swap
    switch (op) {
    case StlCmp::lt:
        return StlCmp::gt;
    case StlCmp::le:
        return StlCmp::ge;
    case StlCmp::gt:
        return StlCmp::lt;
    case StlCmp::ge:
        return StlCmp::le;
    }
    return op;
}

StlCmp stl_cmp_negate(StlCmp op) {
    // the comparison that holds exactly where op fails
    switch (op) {
    case StlCmp::lt:
        return StlCmp::ge;
    case StlCmp::le:
        return StlCmp::gt;
    case StlCmp::gt:
        return StlCmp::le;
    case StlCmp::ge:
        return StlCmp::lt;
    }
    return op;
}

// The expressions ---------------------------------------------------------------------------------

StlExpr::StlExpr(double c) {
    auto node = std::make_shared<Node>();
    node->value = c;
    node_ = node;
}

// A variable node: the name and the 0-based index of its state component.
StlExpr StlExpr::variable(const std::string &name, int64_t index) {
    auto node = std::make_shared<Node>();
    node->kind = StlExprKind::variable;
    node->name = name;
    node->index = index;
    return StlExpr(std::shared_ptr<const Node>(node));
}

// The variables name1, ..., namen, numbered from 1 as CORA names them.
std::vector<StlExpr> StlExpr::variables(const std::string &name, int64_t n) {
    detail::Entry entry;
    CORA_CHECK(!name.empty() && n >= 1, Error::wrong_input_in_constructor,
               [] { return std::string("stl: the name must be non-empty and n at least 1"); });
    std::vector<StlExpr> out;
    for (int64_t i = 0; i < n; ++i) out.push_back(variable(name + std::to_string(i + 1), i));
    return out;
}

// The accessors of an expression node.
StlExprKind StlExpr::kind() const { return node_->kind; }

double StlExpr::value() const { return node_->value; }

const std::string &StlExpr::name() const { return node_->name; }

int64_t StlExpr::index() const { return node_->index; }

int64_t StlExpr::exponent() const { return node_->exponent; }

StlExpr StlExpr::left() const { return StlExpr(node_->l); }

StlExpr StlExpr::right() const { return StlExpr(node_->r); }

// The names below the node, sorted as getVariables sorts them.
std::vector<std::string> StlExpr::variable_names() const {
    std::vector<std::string> names;
    // a variable names itself, an operation collects its operands'
    switch (node_->kind) {
    case StlExprKind::variable:
        names.push_back(node_->name);
        break;
    case StlExprKind::number:
        break;
    case StlExprKind::pow:
        names = left().variable_names();
        break;
    default:
        names = left().variable_names();
        for (auto &name : right().variable_names()) names.push_back(name);
    }
    stl_sort_names(names);
    return names;
}

// A variable or number is linear, a product with a number as one factor is, a power is not.
bool StlExpr::is_linear() const {
    switch (node_->kind) {
    case StlExprKind::variable:
    case StlExprKind::number:
        return true;
    case StlExprKind::add:
    case StlExprKind::sub:
        return left().is_linear() && right().is_linear();
    case StlExprKind::mul:
        // a product is linear with a number as one factor
        if (left().kind() == StlExprKind::number) return right().is_linear();
        return right().kind() == StlExprKind::number && left().is_linear();
    case StlExprKind::pow:
        return false;
    }
    return false;
}

// The arithmetic: CORALean's STL.Expr.plus, minus, times, power (the operands arranged as CORA's).
StlExpr StlExpr::plus(const StlExpr &other) const {
    auto node = std::make_shared<Node>();
    node->kind = StlExprKind::add;
    // a numeric left operand moves to the right
    const bool swap = kind() == StlExprKind::number;
    node->l = swap ? other.node_ : node_;
    node->r = swap ? node_ : other.node_;
    return StlExpr(std::shared_ptr<const Node>(node));
}

StlExpr StlExpr::minus(const StlExpr &other) const {
    auto node = std::make_shared<Node>();
    node->kind = StlExprKind::sub;
    node->l = node_;
    node->r = other.node_;
    return StlExpr(std::shared_ptr<const Node>(node));
}

StlExpr StlExpr::times(const StlExpr &other) const {
    auto node = std::make_shared<Node>();
    node->kind = StlExprKind::mul;
    // the number comes first; two expressions: the right one first
    const bool keep = kind() == StlExprKind::number;
    node->l = keep ? node_ : other.node_;
    node->r = keep ? other.node_ : node_;
    return StlExpr(std::shared_ptr<const Node>(node));
}

bool StlExpr::isequal(const StlExpr &other) const {
    if (kind() != other.kind()) return false;
    switch (kind()) {
    case StlExprKind::variable:
        return name() == other.name();
    case StlExprKind::number:
        return value() == other.value();
    case StlExprKind::pow:
        return exponent() == other.exponent() && left().isequal(other.left());
    case StlExprKind::sub:
        return left().isequal(other.left()) && right().isequal(other.right());
    default:
        // sums and products commute
        return (left().isequal(other.left()) && right().isequal(other.right())) ||
               (left().isequal(other.right()) && right().isequal(other.left()));
    }
}

std::string StlExpr::to_string() const {
    const auto operand = [](const StlExpr &e) { return e.to_string(); };
    switch (kind()) {
    case StlExprKind::variable:
        return name();
    case StlExprKind::number:
        return aux_num2str(value());
    case StlExprKind::add:
        return operand(left()) + " + " + operand(right());
    case StlExprKind::sub:
        return operand(left()) + " - " + operand(right());
    case StlExprKind::mul:
        return operand(left()) + "*" + operand(right());
    case StlExprKind::pow: {
        // a sum, difference or product as base is bracketed
        const StlExprKind b = left().kind();
        const bool bracket =
            b == StlExprKind::add || b == StlExprKind::sub || b == StlExprKind::mul;
        return (bracket ? "(" + operand(left()) + ")" : operand(left())) + "^" +
               std::to_string(exponent());
    }
    }
    return {};
}

std::ostream &operator<<(std::ostream &os, const StlExpr &e) { return os << e.to_string(); }

// The comparisons of expressions ------------------------------------------------------------------

Stl StlExpr::lt(const StlExpr &other) const { return Stl::compare(StlCmp::lt, *this, other); }

Stl StlExpr::le(const StlExpr &other) const { return Stl::compare(StlCmp::le, *this, other); }

Stl StlExpr::gt(const StlExpr &other) const { return Stl::compare(StlCmp::gt, *this, other); }

Stl StlExpr::ge(const StlExpr &other) const { return Stl::compare(StlCmp::ge, *this, other); }

// The formulas ------------------------------------------------------------------------------------

Stl::Stl(bool value) { node_ = StlAccess::make(value ? StlKind::true_ : StlKind::false_); }

// The comparison node; the sides are arranged as CORALean's STL.Tree.compare does.
Stl Stl::compare(StlCmp op, const StlExpr &e1, const StlExpr &e2) {
    auto node = StlAccess::make(StlKind::comparison);
    const bool n1 = e1.kind() == StlExprKind::number, n2 = e2.kind() == StlExprKind::number;
    if (n2) {
        node->op = op; // a number on the right stays there
        node->e1 = e1;
        node->e2 = e2;
    } else if (n1) {
        node->op = stl_cmp_dual(op); // a number on the left swaps the sides
        node->e1 = e2;
        node->e2 = e1;
    } else {
        node->op = op; // two expressions: e1 - e2 op 0
        node->e1 = e1.minus(e2);
        node->e2 = StlExpr(0.0);
    }
    return Stl(std::shared_ptr<const Node>(node));
}

// The logic nodes remember whether a temporal operator occurs below them.
Stl Stl::not_() const {
    auto node = StlAccess::make(StlKind::not_);
    node->l = node_;
    node->temporal = node_->temporal;
    return Stl(std::shared_ptr<const Node>(node));
}

// Conjunction and disjunction: the node of both operands.
Stl Stl::and_(const Stl &other) const {
    auto node = StlAccess::make(StlKind::and_);
    node->l = node_;
    node->r = other.node_;
    node->temporal = node_->temporal || other.node_->temporal;
    return Stl(std::shared_ptr<const Node>(node));
}

Stl Stl::or_(const Stl &other) const {
    auto node = StlAccess::make(StlKind::or_);
    node->l = node_;
    node->r = other.node_;
    node->temporal = node_->temporal || other.node_->temporal;
    return Stl(std::shared_ptr<const Node>(node));
}

// The implication is ~this | other, as CORA builds it.
Stl Stl::implies(const Stl &other) const { return not_().or_(other); }

Stl Stl::with_id(int64_t id) const {
    auto node = std::make_shared<Node>(*node_);
    node->id = id;
    return Stl(std::shared_ptr<const Node>(node));
}

StlKind Stl::kind() const { return node_->kind; }

Stl Stl::lhs() const { return Stl(node_->l); }

Stl Stl::rhs() const { return Stl(node_->r); }

StlExpr Stl::expr_lhs() const { return node_->e1; }

StlExpr Stl::expr_rhs() const { return node_->e2; }

StlCmp Stl::cmp() const { return node_->op; }

const StlInterval &Stl::interval() const { return node_->interval; }

double Stl::from() const { return node_->interval.infimum().value; }

double Stl::to() const { return node_->interval.supremum().value; }

bool Stl::temporal() const { return node_->temporal; }

int64_t Stl::id() const { return node_->id; }

// Same structure; and and or commute, the windows compare by their bounds (CORA's isequal).
bool Stl::isequal(const Stl &other) const {
    if (kind() != other.kind()) return false;
    switch (kind()) {
    case StlKind::true_:
    case StlKind::false_:
        return true;
    case StlKind::comparison:
        return cmp() == other.cmp() && expr_lhs().isequal(other.expr_lhs()) &&
               expr_rhs().isequal(other.expr_rhs());
    case StlKind::and_:
    case StlKind::or_:
        return (lhs().isequal(other.lhs()) && rhs().isequal(other.rhs())) ||
               (lhs().isequal(other.rhs()) && rhs().isequal(other.lhs()));
    // the windows compare by their bounds
    case StlKind::finally:
    case StlKind::globally:
    case StlKind::next:
        return from() == other.from() && to() == other.to() && lhs().isequal(other.lhs());
    case StlKind::until:
    case StlKind::release:
        return from() == other.from() && to() == other.to() && lhs().isequal(other.lhs()) &&
               rhs().isequal(other.rhs());
    case StlKind::not_:
        return lhs().isequal(other.lhs());
    }
    return false;
}

std::ostream &operator<<(std::ostream &os, const Stl &phi) { return os << phi.to_string(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
