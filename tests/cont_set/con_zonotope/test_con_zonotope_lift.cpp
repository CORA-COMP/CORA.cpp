// test_con_zonotope_lift - ConZonotope::lift: the states at given positions
//
// CORA: contSet/@conZonotope/lift_.m

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: state i goes to position dims[i] (CORA's lift_ would put (2,3,1) there).
void lift_known(const std::string &b) {
    const ConZonotope a(column({1.0, 2.0, 3.0}), Tensor({{1.0}, {2.0}, {3.0}}), Tensor({{1.0}}),
                        column({0.5}));
    const ConZonotope r = a.lift(3, {1, 2, 0});
    check(close(r.c(), {3, 1, 2}, 0) && close(r.generators(), {3, 1, 2}, 0),
          b + ": state i at position dims[i]");
    check(close(r.A(), a.A(), 0), b + ": the constraints stay");
    check(Set(a).get().lift(3, std::vector<int64_t>{1, 2, 0}).as<ConZonotope>().c().data() ==
              r.c().data(),
          b + ": through the set interface");
}

} // namespace

int main() {
    test::for_each_backend(lift_known);
    const ConZonotope a(column({1.0, 2.0}), Tensor({{1.0}, {2.0}}));
    check(test::raises([&] { a.lift(3, {0, 1}); }, Error::not_defined), "lift: new dimensions");
    check(test::raises([&] { a.lift(2, {1, 1}); }, Error::wrong_value), "lift: a position twice");
    check(test::raises([&] { a.lift(2, {0, 5}); }, Error::wrong_value), "lift: a position outside");
    ChecksScope off(false);
    check(a.lift(2, {1, 0}).dim() == 2, "lift: the same with the checks off");
    return test::finish("con_zonotope lift");
}
