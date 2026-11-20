// test_con_zonotope_lift - ConZonotope::lift: the states at given positions
//
// CORA: contSet/@conZonotope/lift_.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: state i goes to position dims[i] (CORA's lift_ would put (2,3,1) there).
void lift_known(const std::string &b) {
    const ConZonotope a(vec({1.0, 2.0, 3.0}), Tensor({{1.0, 2.0, 3.0}}), Tensor({{1.0}}),
                        vec({0.5}));
    const ConZonotope r = a.lift(3, {1, 2, 0});
    check(close(r.c().expand_dims(-1), {3, 1, 2}, 0) &&
              close(r.generators().matrix_transpose(), {3, 1, 2}, 0),
          b + ": state i at position dims[i]");
    check(close(r.A(), a.A(), 0), b + ": the constraints stay");
    check(Set(a).get().lift(3, std::vector<int64_t>{1, 2, 0}).as<ConZonotope>().c().data() ==
              r.c().data(),
          b + ": through the set interface");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(lift_known);
    const ConZonotope a(vec({1.0, 2.0}), Tensor({{1.0, 2.0}}));
    check(test::raises([&] { a.lift(3, {0, 1}); }, Error::not_defined), "lift: new dimensions");
    check(test::raises([&] { a.lift(2, {1, 1}); }, Error::wrong_value), "lift: a position twice");
    check(test::raises([&] { a.lift(2, {0, 5}); }, Error::wrong_value), "lift: a position outside");
    ChecksScope off(false);
    check(a.lift(2, {1, 0}).dim() == 2, "lift: the same with the checks off");
    return test::finish("con_zonotope lift");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
