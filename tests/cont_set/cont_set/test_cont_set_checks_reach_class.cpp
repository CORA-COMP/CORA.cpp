// test_cont_set_checks_reach_class - a call through the generic entry runs the checks of the class
//
// Every operation of ContSet and every pair of the registry validates what it owns, then hands over
// to the class: the class's own entry must be the outermost one, as for a direct call, so that its
// checks are not skipped (#48). A probe set records, in each hook, whether an entry would check.
//
// CORA: contSet/@contSet/contSet.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       16-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

#include <set>
#include <sstream>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The hooks that saw an entry that checks.
std::set<std::string> &seen() {
    static std::set<std::string> names;
    return names;
}

/// Whether an entry opened here would check; records the hook if so.
void aux_record(const std::string &hook) {
    // an entry opened inside a hook is outermost exactly if the caller released its own
    cora::detail::Entry entry;
    if (entry.check()) seen().insert(hook);
}

/// A set whose every hook records whether the class would check.
class ProbeSet : public test::UnregisteredSet {
  protected:
    // each hook of the unchecked core records itself, then returns a trivial result
    Tensor center_impl() const override {
        aux("center");
        return Tensor({0.0, 0.0});
    }
    // the interval and the support function
    Interval interval_impl() const override {
        aux("interval");
        return {Tensor({0.0, 0.0}), Tensor({0.0, 0.0})};
    }
    // the queries
    Tensor support_function_impl(const Tensor &d) const override {
        aux("support_function");
        return d.zeros_like({1, 1});
    }
    Tensor rand_point_impl(int64_t, Rng &) const override {
        aux("rand_point");
        return Tensor({0.0, 0.0});
    }
    Set project_impl(const std::vector<int64_t> &) const override { return other("project"); }
    // the vertices
    std::vector<Polygon> vertices_impl() const override {
        aux("vertices");
        return {{{0.0, 0.0}}};
    }
    std::vector<std::vector<Point3>> vertices_3d_impl() const override {
        aux("vertices_3d");
        return {};
    }
    std::string display_impl() const override {
        aux("display");
        return "ProbeSet";
    }
    // the predicates
    bool is_bounded_impl() const override {
        aux("is_bounded");
        return true;
    }
    bool is_full_dim_impl(Tol) const override {
        aux("is_full_dim");
        return true;
    }
    // the maps and the other operations that return a set
    Set reduce_impl(int) const override { return other("reduce"); }
    Set linear_map_impl(const Tensor &) const override { return other("linear_map"); }
    // the maps
    Set translate_impl(const Tensor &) const override { return other("translate"); }
    Set uminus_impl() const override { return other("uminus"); }
    Set uplus_impl() const override { return other("uplus"); }
    Set times_impl(const Tensor &) const override { return other("times"); }
    std::optional<std::vector<std::string>> compact_methods_impl() const override {
        return std::nullopt;
    }
    // compacting and the difference
    Set compact_impl(const std::optional<std::string> &, Tol) const override {
        return other("compact");
    }
    // the operations with an argument
    Set enlarge_impl(const Tensor &) const override { return other("enlarge"); }
    Set minkowski_difference_impl(const ContSet &,
                                  const std::optional<std::string> &) const override {
        return other("minkowski_difference");
    }
    // the dimension changes
    Set project_high_dim_impl(int64_t, const std::vector<int64_t> &) const override {
        return other("project_high_dim");
    }
    Set lift_impl(int64_t, const std::vector<int64_t> &) const override { return other("lift"); }
    Tensor norm_impl(Tensor::Norm, const std::string &) const override {
        aux("norm");
        return Tensor({0.0});
    }
    // the last operations
    Set enclose_impl(const ContSet &) const override { return other("enclose"); }

  private:
    static void aux(const char *hook) { aux_record(hook); }
    static Set other(const char *hook) {
        aux_record(hook);
        return Set(Interval(test::vec({0.0, 0.0}), test::vec({0.0, 0.0})));
    }
};

/// The hooks of the operations of ContSet.
void every_operation_hands_over_to_checking_class() {
    // every operation of ContSet that hands over to a hook of the class is called once
    const ProbeSet P;
    const Tensor I2 = Tensor({{1.0, 0.0}, {0.0, 1.0}}), v = Tensor({0.0, 0.0});
    Rng rng(1);
    static_cast<void>(P.center());
    static_cast<void>(P.interval());
    static_cast<void>(P.display());
    static_cast<void>(P.is_bounded());
    static_cast<void>(P.is_full_dim());
    // the queries
    static_cast<void>(P.support_function(v));
    static_cast<void>(P.rand_point(1, rng));
    static_cast<void>(P.project({0}));
    static_cast<void>(P.vertices());
    // the maps
    static_cast<void>(P.reduce(1));
    static_cast<void>(P.linear_map(I2));
    // the shifts and scalings
    static_cast<void>(P.enlarge(v));
    static_cast<void>(P.enlarge(2.0));
    static_cast<void>(P.minus(v));
    static_cast<void>(ContSet::minus(v, P));
    static_cast<void>(P.uminus());
    static_cast<void>(P.uplus());
    // the operations with a second set or a size
    static_cast<void>(P.times(v));
    static_cast<void>(P.times(2.0));
    // compact, difference and the dimension changes
    static_cast<void>(P.compact());
    static_cast<void>(P.minkowski_difference(P));
    static_cast<void>(P.project_high_dim(3, {0, 1}));
    static_cast<void>(P.lift(3));
    static_cast<void>(P.norm());
    static_cast<void>(P.enclose(P));
    std::istringstream hooks(
        "center interval display is_bounded is_full_dim support_function rand_point project "
        "vertices reduce linear_map enlarge translate uminus uplus times compact "
        "minkowski_difference project_high_dim lift norm enclose");
    std::string hook;
    while (hooks >> hook)
        check(seen().count(hook) == 1, hook + ": the class checks when it is reached");
}

/// The pairs of the registry: the function of a pair is the class's operation.
void every_pair_hands_over_to_checking_class() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    // a pair of a registered class and the unregistered one, per operation, at run time
    const test::UnregisteredSet U;
    // how many operations were tried
    int sets = 0, predicates = 0;
    for (const BinaryOp op : {BinaryOp::plus, BinaryOp::intersect, BinaryOp::unite,
                              BinaryOp::convex_hull, BinaryOp::cart_prod}) {
        register_binary(op, "Zonotope", "UnregisteredSet",
                        SetFunction([op](const ContSet &, const ContSet &) {
                            aux_record(binary_op_name(op));
                            return Set(Interval(test::vec({0.0, 0.0}), test::vec({0.0, 0.0})));
                        }));
    }
    for (const BinaryOp op : {BinaryOp::contains, BinaryOp::is_intersecting, BinaryOp::isequal}) {
        register_binary(op, "Zonotope", "UnregisteredSet",
                        PredicateFunction([op](const ContSet &, const ContSet &, Tol) {
                            aux_record(binary_op_name(op));
                            return true;
                        }));
    }
    // the generic entries of the registry
    static_cast<void>(plus(Z, U));
    static_cast<void>(intersect(Z, U));
    static_cast<void>(unite(Z, U));
    static_cast<void>(convex_hull(Z, U));
    // the products and the queries
    static_cast<void>(cart_prod(Z, U));
    // the predicates
    static_cast<void>(contains(Z, U));
    static_cast<void>(is_intersecting(Z, U));
    static_cast<void>(isequal(Z, U));
    static_cast<void>(eq(Z, U));
    static_cast<void>(ne(Z, U));
    for (const BinaryOp op : {BinaryOp::plus, BinaryOp::intersect, BinaryOp::unite,
                              BinaryOp::convex_hull, BinaryOp::cart_prod}) {
        ++sets;
        check(seen().count(binary_op_name(op)) == 1,
              std::string(binary_op_name(op)) + ": the class checks when it is reached");
    }
    // the predicates saw an entry that checks too
    for (const BinaryOp op : {BinaryOp::contains, BinaryOp::is_intersecting, BinaryOp::isequal}) {
        ++predicates;
        check(seen().count(binary_op_name(op)) == 1,
              std::string(binary_op_name(op)) + ": the class checks when it is reached");
    }
    check(sets == 5 && predicates == 3, "every operation of the registry was tried");
}

/// With the checks off nothing checks, and the entry is back at depth 0 after a release.
void checks_off_and_depth() {
    seen().clear();
    {
        ChecksScope off(false);
        static_cast<void>(ProbeSet().enclose(ProbeSet()));
    }
    check(seen().empty(), "the checks off: no hook checks");
    {
        cora::detail::Entry entry;
        entry.release();
        entry.release();
    }
    cora::detail::Entry again;
    check(again.check(), "after a released entry the next one is the outermost");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    every_operation_hands_over_to_checking_class();
    every_pair_hands_over_to_checking_class();
    checks_off_and_depth();
    return test::finish("cont_set checks reach the class");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
