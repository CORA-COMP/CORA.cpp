// registry - the table of the binary operations between set classes, and its dispatch
//
// One entry per (operation, class, class): a captureless function that casts the operands to
// their classes and calls the class's own operation. The table is built at compile time; a
// lookup is two array indexes. A commutative entry is registered once, the class of lower CORA
// precedence on the left (checked at compile time), and found for either order of operands.
//
// Syntax:   Set S = detail::dispatch(BinaryOp::plus, S1, S2);   binary_pairs();
//           register_binary(BinaryOp::plus, "Ball", "Zonotope", f);
// Inputs:   op - the operation;  S1, S2 - the operands (already checked by the entry point)
// Outputs:  S - the result of the entry for (class of S1, class of S2)
// See also: cora/cont_set/registry.h, plus, contains
//
// CORA: contSet/@contSet/reorder.m

#include "cora/cont_set/registry.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <array>
#include <map>
#include <mutex>
#include <tuple>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The Table ---------------------------------------------------------------------------------------

constexpr int kClasses = static_cast<int>(ClassId::count);
constexpr int kOps = static_cast<int>(BinaryOp::count);

/// The table: a function per operation and pair of classes, or none.
struct Table {
    detail::SetFn set[kOps][kClasses][kClasses] = {};
    detail::PredicateFn predicate[kOps][kClasses][kClasses] = {};
};

/// Registers F(A, B) -> Set for the operation Op.
template <BinaryOp Op, class A, class B, class F>
constexpr void aux_add_set(Table &table, F) {
    static_assert(!is_commutative(Op) || precedence(A::kClassId) <= precedence(B::kClassId),
                  "a commutative pair is registered with the class of lower precedence first");
    table.set[static_cast<int>(Op)][static_cast<int>(A::kClassId)][static_cast<int>(B::kClassId)] =
        [](const ContSet &S1, const ContSet &S2) -> Set {
        return F{}(static_cast<const A &>(S1), static_cast<const B &>(S2));
    };
}

/// Registers F(A, B, tol) -> bool for the operation Op.
template <BinaryOp Op, class A, class B, class F>
constexpr void aux_add_predicate(Table &table, F) {
    static_assert(!is_commutative(Op) || precedence(A::kClassId) <= precedence(B::kClassId),
                  "a commutative pair is registered with the class of lower precedence first");
    table.predicate[static_cast<int>(Op)][static_cast<int>(A::kClassId)][static_cast<int>(
        B::kClassId)] = [](const ContSet &S1, const ContSet &S2, Tol tol) -> bool {
        return F{}(static_cast<const A &>(S1), static_cast<const B &>(S2), tol);
    };
}

// The pairs
// ----------------------------------------------------------------------------------------

/// Every pair that is implemented: one line each.
constexpr Table aux_build() {
    Table t;
    aux_add_set<BinaryOp::plus, Zonotope, Zonotope>(
        t, [](const Zonotope &Z1, const Zonotope &Z2) { return Z1.plus(Z2); });
    aux_add_set<BinaryOp::plus, Zonotope, Interval>(
        t, [](const Zonotope &Z, const Interval &I) { return Z.plus(I); });
    aux_add_set<BinaryOp::plus, Interval, Interval>(
        t, [](const Interval &I1, const Interval &I2) { return I1.plus(I2); });
    aux_add_set<BinaryOp::intersect, Interval, Interval>(
        t, [](const Interval &I1, const Interval &I2) { return I1.intersect(I2); });
    aux_add_set<BinaryOp::cart_prod, Interval, Interval>(
        t, [](const Interval &I1, const Interval &I2) { return I1.cart_prod(I2); });
    aux_add_set<BinaryOp::convex_hull, Interval, Interval>(
        t, [](const Interval &I1, const Interval &I2) { return I1.convex_hull(I2); });
    aux_add_set<BinaryOp::unite, Interval, Interval>(
        t, [](const Interval &I1, const Interval &I2) { return I1.unite(I2); });
    aux_add_predicate<BinaryOp::is_intersecting, Interval, Interval>(
        t, [](const Interval &I1, const Interval &I2, Tol tol) {
            return I1.is_intersecting(I2, tol);
        });
    // the zonotope pairs: intersection, convex hull, containment and Cartesian product
    aux_add_set<BinaryOp::intersect, Zonotope, Zonotope>(
        t, [](const Zonotope &Z1, const Zonotope &Z2) { return Z1.intersect(Z2); });
    aux_add_set<BinaryOp::intersect, Zonotope, Interval>(
        t, [](const Zonotope &Z, const Interval &I) { return Z.intersect(I); });
    aux_add_set<BinaryOp::convex_hull, Zonotope, Zonotope>(
        t, [](const Zonotope &Z1, const Zonotope &Z2) { return Z1.convex_hull(Z2); });
    aux_add_set<BinaryOp::convex_hull, Zonotope, Interval>(
        t, [](const Zonotope &Z, const Interval &I) { return Z.convex_hull(I); });
    // a zonotope contains sets of its own and of the next class
    aux_add_predicate<BinaryOp::contains, Zonotope, Zonotope>(
        t, [](const Zonotope &Z1, const Zonotope &Z2, Tol tol) {
            return Z1.contains(Z2, "exact", tol);
        });
    aux_add_predicate<BinaryOp::contains, Zonotope, Interval>(
        t, [](const Zonotope &Z, const Interval &I, Tol tol) {
            return Z.contains(I, "exact", tol);
        });
    // the product is not commutative: the zonotope comes first
    aux_add_set<BinaryOp::cart_prod, Zonotope, Zonotope>(
        t, [](const Zonotope &Z1, const Zonotope &Z2) { return Z1.cart_prod(Z2); });
    aux_add_set<BinaryOp::cart_prod, Zonotope, Interval>(
        t, [](const Zonotope &Z, const Interval &I) { return Z.cart_prod(I); });
    // intersection tests of a zonotope with a set of its own or the next class
    aux_add_predicate<BinaryOp::is_intersecting, Zonotope, Zonotope>(
        t, [](const Zonotope &Z1, const Zonotope &Z2, Tol tol) {
            return Z1.is_intersecting(Z2, "exact", tol);
        });
    aux_add_predicate<BinaryOp::is_intersecting, Zonotope, Interval>(
        t, [](const Zonotope &Z, const Interval &I, Tol tol) {
            return Z.is_intersecting(I, "exact", tol);
        });
    // the empty set has the lowest precedence, so it is on the left; the sum is empty
    aux_add_set<BinaryOp::plus, EmptySet, EmptySet>(
        t, [](const EmptySet &E, const EmptySet &S) { return E.plus(S); });
    aux_add_set<BinaryOp::plus, EmptySet, Interval>(
        t, [](const EmptySet &E, const Interval &I) { return E.plus(I); });
    aux_add_set<BinaryOp::plus, EmptySet, Zonotope>(
        t, [](const EmptySet &E, const Zonotope &Z) { return E.plus(Z); });
    // the intersection is empty, the hull is the other set
    aux_add_set<BinaryOp::intersect, EmptySet, EmptySet>(
        t, [](const EmptySet &E, const EmptySet &S) { return E.intersect(S); });
    aux_add_set<BinaryOp::intersect, EmptySet, Interval>(
        t, [](const EmptySet &E, const Interval &I) { return E.intersect(I); });
    aux_add_set<BinaryOp::intersect, EmptySet, Zonotope>(
        t, [](const EmptySet &E, const Zonotope &Z) { return E.intersect(Z); });
    aux_add_set<BinaryOp::convex_hull, EmptySet, EmptySet>(
        t, [](const EmptySet &E, const EmptySet &S) { return E.convex_hull(Set(S)); });
    aux_add_set<BinaryOp::convex_hull, EmptySet, Interval>(
        t, [](const EmptySet &E, const Interval &I) { return E.convex_hull(Set(I)); });
    aux_add_set<BinaryOp::convex_hull, EmptySet, Zonotope>(
        t, [](const EmptySet &E, const Zonotope &Z) { return E.convex_hull(Set(Z)); });
    // nothing but an empty set is contained in it, and nothing meets it
    aux_add_predicate<BinaryOp::contains, EmptySet, EmptySet>(
        t, [](const EmptySet &E, const EmptySet &S, Tol tol) { return E.contains(S, tol); });
    aux_add_predicate<BinaryOp::contains, EmptySet, Interval>(
        t, [](const EmptySet &E, const Interval &I, Tol tol) { return E.contains(I, tol); });
    aux_add_predicate<BinaryOp::contains, EmptySet, Zonotope>(
        t, [](const EmptySet &E, const Zonotope &Z, Tol tol) { return E.contains(Z, tol); });
    aux_add_predicate<BinaryOp::is_intersecting, EmptySet, EmptySet>(
        t, [](const EmptySet &E, const EmptySet &S, Tol tol) {
            return E.is_intersecting(S, tol);
        });
    aux_add_predicate<BinaryOp::is_intersecting, EmptySet, Interval>(
        t, [](const EmptySet &E, const Interval &I, Tol tol) {
            return E.is_intersecting(I, tol);
        });
    aux_add_predicate<BinaryOp::is_intersecting, EmptySet, Zonotope>(
        t, [](const EmptySet &E, const Zonotope &Z, Tol tol) {
            return E.is_intersecting(Z, tol);
        });
    // the whole space is of the lowest precedence of these classes: it comes first
    aux_add_set<BinaryOp::plus, Fullspace, Fullspace>(
        t, [](const Fullspace &F1, const Fullspace &F2) { return F1.plus(F2); });
    aux_add_set<BinaryOp::plus, Fullspace, Zonotope>(
        t, [](const Fullspace &F, const Zonotope &Z) { return F.plus(Z); });
    aux_add_set<BinaryOp::plus, Fullspace, Interval>(
        t, [](const Fullspace &F, const Interval &I) { return F.plus(I); });
    // the intersection is the other operand; the hull and the sum are the whole space
    aux_add_set<BinaryOp::intersect, Fullspace, Fullspace>(
        t, [](const Fullspace &F1, const Fullspace &F2) { return F1.intersect(F2); });
    aux_add_set<BinaryOp::intersect, Fullspace, Zonotope>(
        t, [](const Fullspace &F, const Zonotope &Z) { return F.intersect(Z); });
    aux_add_set<BinaryOp::intersect, Fullspace, Interval>(
        t, [](const Fullspace &F, const Interval &I) { return F.intersect(I); });
    aux_add_set<BinaryOp::convex_hull, Fullspace, Fullspace>(
        t, [](const Fullspace &F1, const Fullspace &F2) { return F1.convex_hull(F2); });
    aux_add_set<BinaryOp::convex_hull, Fullspace, Zonotope>(
        t, [](const Fullspace &F, const Zonotope &Z) { return F.convex_hull(Z); });
    aux_add_set<BinaryOp::convex_hull, Fullspace, Interval>(
        t, [](const Fullspace &F, const Interval &I) { return F.convex_hull(I); });
    // the whole space contains every set; it meets every set that is not empty
    aux_add_predicate<BinaryOp::contains, Fullspace, Fullspace>(
        t, [](const Fullspace &F1, const Fullspace &F2, Tol tol) { return F1.contains(F2, tol); });
    aux_add_predicate<BinaryOp::contains, Fullspace, Zonotope>(
        t, [](const Fullspace &F, const Zonotope &Z, Tol tol) { return F.contains(Z, tol); });
    aux_add_predicate<BinaryOp::contains, Fullspace, Interval>(
        t, [](const Fullspace &F, const Interval &I, Tol tol) { return F.contains(I, tol); });
    aux_add_predicate<BinaryOp::is_intersecting, Fullspace, Fullspace>(
        t, [](const Fullspace &F1, const Fullspace &F2, Tol tol) {
            return F1.is_intersecting(F2, tol);
        });
    aux_add_predicate<BinaryOp::is_intersecting, Fullspace, Zonotope>(
        t, [](const Fullspace &F, const Zonotope &Z, Tol tol) {
            return F.is_intersecting(Z, tol);
        });
    aux_add_predicate<BinaryOp::is_intersecting, Fullspace, Interval>(
        t, [](const Fullspace &F, const Interval &I, Tol tol) {
            return F.is_intersecting(I, tol);
        });
    return t;
}

constexpr Table kTable = aux_build();

// Registered Pairs --------------------------------------------------------------------------------

/// The pairs registered at run time, by operation and class names, and the lock that guards them.
struct Dynamic {
    std::mutex mutex;
    std::map<std::tuple<int, std::string, std::string>, SetFunction> set;
    std::map<std::tuple<int, std::string, std::string>, PredicateFunction> predicate;
};

Dynamic &aux_dynamic() {
    static Dynamic dynamic;
    return dynamic;
}

/// The function registered at run time for op(S1, S2), swapped for a commutative op; none if
/// there is none. `swapped` says whether the operands are to be swapped.
template <class F>
F aux_find(const std::map<std::tuple<int, std::string, std::string>, F> &table, BinaryOp op,
           const ContSet &S1, const ContSet &S2, bool &swapped) {
    const int o = static_cast<int>(op);
    auto it = table.find({o, S1.class_name(), S2.class_name()});
    swapped = false;
    if (it != table.end()) return it->second;
    if (is_commutative(op)) {
        it = table.find({o, S2.class_name(), S1.class_name()});
        if (it != table.end()) {
            swapped = true;
            return it->second;
        }
    }
    return {};
}

/// The text of a missing pair: both classes and what to do.
std::string aux_noops(BinaryOp op, const ContSet &S1, const ContSet &S2) {
    return std::string(binary_op_name(op)) + " is not implemented for " + S1.class_name() +
           " and " + S2.class_name() + "; binary_pairs() lists the pairs that are, and " +
           "register_binary adds one";
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Names and Pairs ---------------------------------------------------------------------------------

/// The name of an operation, as in Python and in the messages.
const char *binary_op_name(BinaryOp op) noexcept {
    constexpr const char *names[] = {"plus",      "intersect", "unite",          "convex_hull",
                                     "cart_prod", "contains",  "is_intersecting"};
    return names[static_cast<int>(op)];
}

/// The operation of a name; an error for an unknown one.
BinaryOp binary_op(const std::string &name) {
    for (int op = 0; op < kOps; ++op)
        if (name == binary_op_name(static_cast<BinaryOp>(op))) return static_cast<BinaryOp>(op);
    CORA_FAIL(Error::wrong_value, "unknown binary operation '" + name +
                                      "'; use plus, intersect, "
                                      "unite, convex_hull, cart_prod, contains or is_intersecting");
}

/// Every implemented pair: the built-in table, then the registered ones.
std::vector<BinaryPair> binary_pairs() {
    std::vector<BinaryPair> pairs;
    for (int op = 0; op < kOps; ++op)
        for (int a = 0; a < kClasses; ++a)
            for (int b = 0; b < kClasses; ++b)
                if (kTable.set[op][a][b] || kTable.predicate[op][a][b])
                    pairs.push_back({static_cast<BinaryOp>(op), class_name(static_cast<ClassId>(a)),
                                     class_name(static_cast<ClassId>(b))});
    Dynamic &dynamic = aux_dynamic();
    std::lock_guard<std::mutex> lock(dynamic.mutex);
    for (const auto &[key, f] : dynamic.set)
        pairs.push_back(
            {static_cast<BinaryOp>(std::get<0>(key)), std::get<1>(key), std::get<2>(key), true});
    for (const auto &[key, f] : dynamic.predicate)
        pairs.push_back(
            {static_cast<BinaryOp>(std::get<0>(key)), std::get<1>(key), std::get<2>(key), true});
    return pairs;
}

// Registration ------------------------------------------------------------------------------------

/// A pair of sets registered at run time (Python), by class names.
void register_binary(BinaryOp op, const std::string &left, const std::string &right,
                     SetFunction f) {
    Dynamic &dynamic = aux_dynamic();
    std::lock_guard<std::mutex> lock(dynamic.mutex);
    dynamic.set[{static_cast<int>(op), left, right}] = std::move(f);
}

/// A pair of a predicate registered at run time.
void register_binary(BinaryOp op, const std::string &left, const std::string &right,
                     PredicateFunction f) {
    Dynamic &dynamic = aux_dynamic();
    std::lock_guard<std::mutex> lock(dynamic.mutex);
    dynamic.predicate[{static_cast<int>(op), left, right}] = std::move(f);
}

// Dispatch ----------------------------------------------------------------------------------------

namespace detail {

Set dispatch(BinaryOp op, const ContSet &S1, const ContSet &S2) {
    detail::Entry entry;
    CORA_CHECK(op == BinaryOp::cart_prod || S1.dim() == S2.dim(), Error::dimension_mismatch, [&] {
        return std::string(binary_op_name(op)) + ": the sets differ in dimension, " +
               S1.class_name() + " of " + std::to_string(S1.dim()) + " and " + S2.class_name() +
               " of " + std::to_string(S2.dim());
    });
    const int o = static_cast<int>(op), a = static_cast<int>(S1.class_id()),
              b = static_cast<int>(S2.class_id());
    // the table, by class ids: O(1), the lower precedence first
    if (const SetFn f = kTable.set[o][a][b]) return f(S1, S2);
    if (is_commutative(op))
        if (const SetFn f = kTable.set[o][b][a]) return f(S2, S1);
    // not in the table: a pair registered at run time, or none
    SetFunction g;
    bool swapped = false;
    {
        Dynamic &dynamic = aux_dynamic();
        std::lock_guard<std::mutex> lock(dynamic.mutex);
        g = aux_find(dynamic.set, op, S1, S2, swapped);
    }
    if (g) return swapped ? g(S2, S1) : g(S1, S2);
    CORA_FAIL(Error::noops, aux_noops(op, S1, S2));
}

bool dispatch(BinaryOp op, const ContSet &S1, const ContSet &S2, Tol tol) {
    detail::Entry entry;
    CORA_CHECK(op == BinaryOp::cart_prod || S1.dim() == S2.dim(), Error::dimension_mismatch, [&] {
        return std::string(binary_op_name(op)) + ": the sets differ in dimension, " +
               S1.class_name() + " of " + std::to_string(S1.dim()) + " and " + S2.class_name() +
               " of " + std::to_string(S2.dim());
    });
    CORA_CHECK(!tol || *tol >= 0, Error::wrong_value, [&] {
        return std::string(binary_op_name(op)) + ": tol must be nonnegative, got " +
               std::to_string(*tol);
    });
    const int o = static_cast<int>(op), a = static_cast<int>(S1.class_id()),
              b = static_cast<int>(S2.class_id());
    // the table, by class ids: O(1), the lower precedence first
    if (const PredicateFn f = kTable.predicate[o][a][b]) return f(S1, S2, tol);
    if (is_commutative(op))
        if (const PredicateFn f = kTable.predicate[o][b][a]) return f(S2, S1, tol);
    // not in the table: a pair registered at run time, or none
    PredicateFunction g;
    bool swapped = false;
    {
        Dynamic &dynamic = aux_dynamic();
        std::lock_guard<std::mutex> lock(dynamic.mutex);
        g = aux_find(dynamic.predicate, op, S1, S2, swapped);
    }
    if (g) return swapped ? g(S2, S1, tol) : g(S1, S2, tol);
    CORA_FAIL(Error::noops, aux_noops(op, S1, S2));
}

} // namespace detail
} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
