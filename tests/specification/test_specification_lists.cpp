// test_specification_lists - add, is_empty, project, isequal of specifications
//
// CORA: specification/@specification/add.m
// CORA: specification/@specification/isempty.m
// CORA: specification/@specification/project.m
// CORA: specification/@specification/isequal.m
// CORA: specification/@specification/eq.m
// CORA: specification/@specification/ne.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/specification/specification.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box [lo1, hi1] x [lo2, hi2] as a polytope.
Polytope box(double lo1, double hi1, double lo2, double hi2) {
    return {Tensor({{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}}),
            vec({hi1, -lo1, hi2, -lo2})};
}


void lists(const std::string &b) {
    const Specification safe = Specification::safe_set(box(0, 1, 0, 1));
    const Specification unsafe = Specification::unsafe_set(box(2, 3, 2, 3));
    // an empty list, a list of one, and add
    check(is_empty({}), b + ": an empty list");
    check(!is_empty({safe}), b + ": a list of one");
    const std::vector<Specification> both = add({safe}, {unsafe});
    check(both.size() == 2 && both[1].type() == SpecType::unsafe_set, b + ": add keeps the order");
    check(add({}, {}).empty(), b + ": add of nothing");
}

void projection(const std::string &b) {
    const Specification spec = Specification::safe_set(box(0, 1, 2, 3)).with_time(1.0, 2.0);
    const Specification p = spec.project({1});
    // the set is projected, the window and the type stay
    check(p.set().as<Polytope>().dim() == 1, b + ": one dimension left");
    check(p.time().has_value() && p.type() == SpecType::safe_set, b + ": time and type stay");
    check(p.set().as<Polytope>().contains(vec({2.5})) &&
              !p.set().as<Polytope>().contains(vec({3.5})),
          b + ": the projected box is [2, 3]");
    // a list, a custom predicate, and a dimension that is not there
    check(project({spec, spec}, {0}).size() == 2, b + ": a list");
    check(test::raises(
              [&] { Specification::custom([](const ContSet &) { return true; }).project({0}); },
              Error::not_supported),
          b + ": custom");
    check(test::raises([&] { spec.project({5}); }, Error::wrong_value),
          b + ": a dimension outside");
}

void equality(const std::string &b) {
    const Specification a = Specification::safe_set(box(0, 1, 0, 1));
    check(a.isequal(Specification::safe_set(box(0, 1, 0, 1))), b + ": same");
    check(!a.isequal(Specification::unsafe_set(box(0, 1, 0, 1))), b + ": another type");
    check(!a.isequal(Specification::safe_set(box(0, 1, 0, 2))), b + ": another set");
    // the window, the locations and a custom predicate
    check(!a.isequal(a.with_time(0.0, 1.0)), b + ": a time against none");
    check(a.with_time(0.0, 1.0).isequal(a.with_time(0.0, 1.0 + 1e-14)), b + ": times within tol");
    check(!a.with_time(0.0, 1.0).isequal(a.with_time(0.0, 1.1)), b + ": another time");
    check(!a.isequal(a.with_locations({0})), b + ": a location against none");
    check(a.with_locations({0, 1}).isequal(a.with_locations({0, 1})), b + ": same locations");
    check(test::raises(
              [&] {
                  const Specification c =
                      Specification::custom([](const ContSet &) { return true; });
                  c.isequal(c);
              },
              Error::not_supported),
          b + ": custom");
    // lists match up to the order
    const Specification u = Specification::unsafe_set(box(2, 3, 2, 3));
    check(isequal({a, u}, {u, a}), b + ": a list up to the order");
    check(!isequal({a, u}, {u, u}), b + ": a list with a different member");
    check(!isequal({a}, {a, u}), b + ": lists of different length");
    check(test::raises([&] { a.isequal(a, -1.0); }, Error::wrong_value), b + ": negative tol");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        lists(b);
        projection(b);
        equality(b);
    });
    return test::finish("test_specification_lists");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
