// test_specification_time - time windows, invariants, points and reach sets in check
//
// CORA: specification/@specification/check.m
// CORA: specification/@specification/specification.m
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

/// A point (zonotope without generators) or a small box around (x, y).
Zonotope around(double x, double y, double r) {
    return {vec({x, y}), Tensor({{r, 0.0}, {0.0, r}})};
}


/// A window outside the time of the set is not active; inside it is; the time is required.
void timed(const std::string &b) {
    const Specification spec = Specification::unsafe_set(box(0, 1, 0, 1)).with_time(1.0, 2.0);
    check(spec.time().has_value() && spec.time()->first == 1.0, b + ": the window");
    const Zonotope in = around(0.5, 0.5, 0.1);
    check(spec.check(in, {0.0, 0.5}), b + ": before the window: not active");
    check(!spec.check(in, {1.5, 1.5}), b + ": inside the window: violated");
    check(!spec.check(in, {0.0, 1.0}), b + ": touching the window at 1");
    check(test::raises([&] { spec.check(in); }, Error::special_error), b + ": time required");
    check(test::raises([&] { spec.with_time(2.0, 1.0); }, Error::wrong_input_in_constructor),
          b + ": a reversed window");
    check(Specification::unsafe_set(box(0, 1, 0, 1)).with_locations({0, 2}).locations().size() == 2,
          b + ": locations");
    check(test::raises([&] { Specification::safe_set(box(0, 1, 0, 1)).with_locations({-1}); },
                       Error::wrong_input_in_constructor),
          b + ": a negative location");
}

/// An invariant must intersect the set.
void invariant(const std::string &b) {
    const Specification spec = Specification::invariant(box(0, 1, 0, 1));
    check(spec.type() == SpecType::invariant, b + ": the type");
    check(spec.check(around(0.5, 0.5, 0.1)), b + ": inside intersects");
    check(spec.check(around(1.05, 0.5, 0.1)), b + ": overlapping intersects");
    check(!spec.check(around(3.0, 3.0, 0.1)), b + ": disjoint violates");
}

/// Points: each row against the set, with times for a timed specification.
void points(const std::string &b) {
    // the points against each type
    const Tensor P({{0.5, 0.5}, {3.0, 3.0}});
    check(Specification::safe_set(box(0, 4, 0, 4)).check_points(P), b + ": both inside");
    check(!Specification::safe_set(box(0, 1, 0, 1)).check_points(P), b + ": one outside");
    check(!Specification::unsafe_set(box(0, 1, 0, 1)).check_points(P),
          b + ": one in the unsafe set");
    check(Specification::unsafe_set(box(0, 1, 0, 1)).check_points(Tensor({{3.0, 3.0}})),
          b + ": clear of the unsafe set");
    check(Specification::invariant(box(0, 1, 0, 1)).check_points(Tensor({{0.5, 0.5}})),
          b + ": an invariant holds for points inside");
    check(!Specification::invariant(box(0, 1, 0, 1)).check_points(P),
          b + ": an invariant fails for a point outside");
    // the window and the times of the points
    const Specification timed = Specification::unsafe_set(box(0, 1, 0, 1)).with_time(1.0, 2.0);
    check(timed.check_points(P, {0.0, 1.5}), b + ": the point in the set is not active");
    check(!timed.check_points(P, {1.5, 0.0}), b + ": the point in the set is active");
    // the times needed, and the custom predicate
    check(test::raises([&] { timed.check_points(P); }, Error::special_error), b + ": times needed");
    check(test::raises([&] { timed.check_points(P, {1.0, 2.0, 3.0}); }, Error::special_error),
          b + ": wrong number of times");
    check(test::raises(
              [&] { Specification::custom([](const ContSet &) { return true; }).check_points(P); },
              Error::not_supported),
          b + ": custom");
}

/// A reach set: the windows against the steps [k, k+1] and the time points.
void reach_set(const std::string &b) {
    ReachSet R;
    R.time_step = 1.0;
    for (const double x : {0.5, 1.5, 2.5}) R.time_interval.push_back(around(x, 0.5, 0.1));
    for (const double x : {0.0, 1.0, 2.0, 3.0}) R.time_point.push_back(around(x, 0.5, 0.0));
    // untimed, then windows on the steps and outside the horizon
    const Polytope band = box(1.4, 1.6, 0.0, 1.0);
    check(Specification::unsafe_set(band).check(R) == false, b + ": untimed: step 1 touches");
    check(Specification::unsafe_set(box(5, 6, 0, 1)).check(R), b + ": untimed: clear");
    check(!Specification::unsafe_set(band).with_time(1.2, 1.3).check(R), b + ": window on step 1");
    check(Specification::unsafe_set(band).with_time(2.5, 2.6).check(R), b + ": window on step 2");
    check(!Specification::unsafe_set(band).with_time(5.0, 6.0).check(R),
          b + ": a window outside the horizon");
    // an instant takes the time point on a border of a step, else the step
    const Polytope at_one = box(0.9, 1.1, 0.0, 1.0);
    check(!Specification::unsafe_set(at_one).with_time(1.0, 1.0).check(R),
          b + ": an instant checks the time point");
    check(Specification::unsafe_set(at_one).with_time(1.5, 1.5).check(R),
          b + ": an instant inside a step checks the step set");
    ReachSet unknown = R;
    unknown.time_step = 0.0;
    check(test::raises([&] { Specification::unsafe_set(band).with_time(1.0, 2.0).check(unknown); },
                       Error::special_error),
          b + ": unknown times");
    // an adaptive step size
    ReachSet adaptive = R;
    adaptive.time = {0.0, 0.5, 2.0, 3.0};
    check(Specification::unsafe_set(band).with_time(0.1, 0.4).check(adaptive),
          b + ": adaptive steps: the first step is clear");
    check(!Specification::unsafe_set(band).with_time(1.0, 1.1).check(adaptive),
          b + ": adaptive steps: the second step touches");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        timed(b);
        invariant(b);
        points(b);
        reach_set(b);
    });
    return test::finish("test_specification_time");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
