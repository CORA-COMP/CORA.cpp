// test_cont_set_support_function - ContSet::support_function: the checked entry of every set class
//
// CORA: contSet/@contSet/supportFunc.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// A direction of the wrong dimension is refused for every class; the checks off skip it.
void the_direction_is_checked() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const Interval I = Interval(test::column({0, -1}), test::column({1, 1}));
    for (const ContSet *S : {static_cast<const ContSet *>(&Z), static_cast<const ContSet *>(&I)}) {
        check(test::raises([&] { S->support_function(test::column({1, 0, 0})); },
                           Error::dimension_mismatch),
              S->class_name() + ": a direction of the wrong dimension");
        const double on = test::support(*S, {0, 1});
        ChecksScope off(false);
        check(test::close(test::support(*S, {0, 1}), on, 0), S->class_name() + ": the checks off");
    }
}

} // namespace

int main() {
    the_direction_is_checked();
    return test::finish("cont_set support_function");
}
