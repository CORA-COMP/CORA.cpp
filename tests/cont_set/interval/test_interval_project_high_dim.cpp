// test_interval_project_high_dim - interval project_high_dim: a box placed into a larger space
//
// CORA: contSet/@interval/projectHighDim_.m
// CORA: contSet/@contSet/projectHighDim.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void project_high_dim_places_the_coordinates(const std::string &b) {
    const Interval I(column({1.0, 3.0}), column({2.0, 4.0}));
    const Interval R = I.project_high_dim(4, {2, 0});
    check(R.inf().data() == std::vector<double>({3, 0, 1, 0}) &&
              R.sup().data() == std::vector<double>({4, 0, 2, 0}),
          b + ": coordinate proj[j] is coordinate j of the box, the rest [0, 0]");
    check(I.project_high_dim(2, {0, 1}).sup().data() == I.sup().data(), b + ": N = dim");
    check(I.project_high_dim(3, {1, 1}).inf().data() == std::vector<double>({0, 3, 0}),
          b + ": a repeated coordinate keeps the last, as CORA");
    const Interval S = Interval::stack({I, I}).project_high_dim(3, {1, 2});
    check(S.inf().shape() == std::vector<int64_t>({2, 3, 1}) &&
              S.sup().data() == std::vector<double>({0, 2, 4, 0, 2, 4}),
          b + ": a batch");
    const Interval E = Interval::empty(2).project_high_dim(5, {0, 3});
    check(E.is_empty() && E.dim() == 5, b + ": the empty interval stays empty");
}

/// A space that is too small, a wrong number or a range of coordinates are refused; with the
/// checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({1.0, 3.0}), column({2.0, 4.0}));
    check(test::raises([&] { I.project_high_dim(1, {0, 0}); }, Error::wrong_value),
          "project_high_dim: N below the dimension");
    check(test::raises([&] { I.project_high_dim(3, {0}); }, Error::wrong_value),
          "project_high_dim: one coordinate for two");
    check(test::raises([&] { I.project_high_dim(3, {0, 3}); }, Error::wrong_value),
          "project_high_dim: coordinate 3 of 3");
    check(test::raises([&] { I.project_high_dim(3, {-1, 0}); }, Error::wrong_value),
          "project_high_dim: a negative coordinate");
    const std::vector<double> on = I.project_high_dim(3, {2, 0}).inf().data();
    ChecksScope off(false);
    check(I.project_high_dim(3, {2, 0}).inf().data() == on,
          "project_high_dim: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(project_high_dim_places_the_coordinates);
    checks_and_their_switch();
    return test::finish("interval project_high_dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
