// test_cont_set_rand_point - ContSet::rand_point: the number of points is checked
//
// CORA: contSet/@contSet/randPoint.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// A negative count is refused; the checks off draw the same points.
void the_count_is_checked() {
    const Interval I = Interval(test::column({0, -1}), test::column({1, 1}));
    const ContSet &S = I;
    cora::Rng rng(1), again(1);
    check(test::raises([&] { S.rand_point(-2, rng); }, Error::wrong_value), "N = -2");
    const Tensor on = S.rand_point(5, rng);
    ChecksScope off(false);
    check(test::close(S.rand_point(5, again), on, 0), "the same with the checks off");
}

} // namespace

int main() {
    the_count_is_checked();
    return test::finish("cont_set rand_point");
}
