// registry - the binary operations between set classes: one table of (class, class) -> function
//
// plus, intersect (CORA's and), unite (or), convex_hull, cart_prod, contains, is_intersecting and
// isequal between two sets of any classes go through one table indexed by the two
// class ids: an O(1) lookup, no string compare and no dynamic_cast. A commutative operation is
// stored once, the class of lower CORA precedence first, and its operands are swapped to match;
// a pair that is not in the table raises Error::noops naming both classes. binary_pairs() lists
// the table, for the documentation of which pairs work. Classes defined outside the library
// (Python subclasses) register their pairs at run time by class name; those are looked up only
// when the table has no entry.
//
// Syntax:     Set S = plus(Z, I);   Set S = Set(Z) + Set(I);   bool b = contains(S1, S2, tol);
//             bool same = (Set(Z) == Set(I));   for (const BinaryPair &p : binary_pairs()) ...
// Operations: plus, intersect, unite, convex_hull, cart_prod, contains, is_intersecting, isequal,
//             eq, ne (one file each in src/cont_set/cont_set/); the table is
//             src/cont_set/registry.cpp. The unchecked forms (CORA's and_, contains_,
//             isIntersecting_) are detail::intersect_impl, contains_impl, is_intersecting_impl
// See also:   cont_set/set.h, cont_set/cont_set.h
//
// CORA: contSet/@contSet/reorder.m

#pragma once

#include "cora/cont_set/set.h"
#include "cora/global/tensor/tolerance.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The Table ---------------------------------------------------------------------------------------

/// The binary operations of the table.
enum class BinaryOp : unsigned char {
    plus,
    intersect,
    unite,
    convex_hull,
    cart_prod,
    contains,
    is_intersecting,
    isequal,
    count,
};

/// The name of the operation, as the functions are called ("plus", "intersect", ...).
const char *binary_op_name(BinaryOp op) noexcept;

/// Whether op(A, B) = op(B, A), so that one order is stored and the other swapped.
constexpr bool is_commutative(BinaryOp op) noexcept {
    return op != BinaryOp::cart_prod && op != BinaryOp::contains;
}

/// One entry of the table: op is implemented for the classes (left, right); `custom` if it was
/// registered at run time (from Python).
struct BinaryPair {
    BinaryOp op;
    std::string left, right;
    bool custom = false;
};

/// The entries of the table, by operation, then class; then those registered at run time.
std::vector<BinaryPair> binary_pairs();

/// The operation of a name ("plus", "intersect", ...); an Error for an unknown name.
BinaryOp binary_op(const std::string &name);

/// Implementations for classes the table does not know (a Python subclass), by class name; the
/// table is asked first, so a registered pair never slows down a built-in one.
using SetFunction = std::function<Set(const ContSet &S1, const ContSet &S2)>;
using PredicateFunction = std::function<bool(const ContSet &S1, const ContSet &S2, Tol tol)>;
void register_binary(BinaryOp op, const std::string &left, const std::string &right, SetFunction f);
void register_binary(BinaryOp op, const std::string &left, const std::string &right,
                     PredicateFunction f);

// The Operations ----------------------------------------------------------------------------------

/// The Minkowski sum S1 + S2, in the class of the operand of lower precedence.
Set plus(const ContSet &S1, const ContSet &S2);

/// The intersection of S1 and S2 (CORA's and).
Set intersect(const ContSet &S1, const ContSet &S2);

/// A set that encloses the union of S1 and S2 (CORA's or).
Set unite(const ContSet &S1, const ContSet &S2);

/// A set that encloses the convex hull of S1 and S2.
Set convex_hull(const ContSet &S1, const ContSet &S2);

/// The Cartesian product {(x, y) | x in S1, y in S2}.
Set cart_prod(const ContSet &S1, const ContSet &S2);

/// Whether S2 is a subset of S1 enlarged by tol in the infinity norm. tol default: the
/// repository tolerance (docs/architecture.md).
bool contains(const ContSet &S1, const ContSet &S2, Tol tol = std::nullopt);

/// Whether S1 and S2 intersect, up to tol (same default).
bool is_intersecting(const ContSet &S1, const ContSet &S2, Tol tol = std::nullopt);

/// Whether S1 and S2 are the same set, up to tol (same default; what tol relaxes is the class's:
/// the bounds of an interval, the center and generators of a zonotope).
bool isequal(const ContSet &S1, const ContSet &S2, Tol tol = std::nullopt);

/// CORA's eq and ne: isequal and its negation.
bool eq(const ContSet &S1, const ContSet &S2, Tol tol = std::nullopt);
bool ne(const ContSet &S1, const ContSet &S2, Tol tol = std::nullopt);

// Operators of Set, as of the classes: + is plus, & intersect, | unite.
inline Set operator+(const Set &S1, const Set &S2) { return plus(S1, S2); }
inline Set operator&(const Set &S1, const Set &S2) { return intersect(S1, S2); }
inline Set operator|(const Set &S1, const Set &S2) { return unite(S1, S2); }
inline bool operator==(const ContSet &S1, const ContSet &S2) { return eq(S1, S2); }
inline bool operator!=(const ContSet &S1, const ContSet &S2) { return ne(S1, S2); }

namespace detail {

/// The implementations the table holds; the operands are of the classes of the entry.
using SetFn = Set (*)(const ContSet &S1, const ContSet &S2);
using PredicateFn = bool (*)(const ContSet &S1, const ContSet &S2, Tol tol);

/// op(S1, S2) by the table, operands swapped for a commutative op stored the other way round;
/// Error::noops if the table has neither. The unchecked core of every binary operation.
Set lookup(BinaryOp op, const ContSet &S1, const ContSet &S2);
bool lookup(BinaryOp op, const ContSet &S1, const ContSet &S2, Tol tol);

/// The refusal of the operands of a binary entry: they must share a dimension (cart_prod
/// excepted), and tol must be nonnegative. The entry raises it with CORA_CHECK.
Refusal refusal(BinaryOp op, const ContSet &S1, const ContSet &S2);
Refusal refusal(BinaryOp op, const ContSet &S1, const ContSet &S2, Tol tol);

/// lookup behind the checks: the checked entry of the operations that have no unchecked form of
/// their own.
Set dispatch(BinaryOp op, const ContSet &S1, const ContSet &S2);
bool dispatch(BinaryOp op, const ContSet &S1, const ContSet &S2, Tol tol);

/// CORA's and_, contains_ and isIntersecting_: the table's function for the classes of the
/// operands, with no check.
Set intersect_impl(const ContSet &S1, const ContSet &S2);
bool contains_impl(const ContSet &S1, const ContSet &S2, Tol tol);
bool is_intersecting_impl(const ContSet &S1, const ContSet &S2, Tol tol);

} // namespace detail
} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
