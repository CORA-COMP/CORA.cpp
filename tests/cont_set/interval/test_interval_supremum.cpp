// test_interval_supremum - interval supremum: the upper bound of a box
//
// CORA: contSet/@interval/supremum.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void supremum_is_the_upper_bound(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({2.0, 0.5}));
    check(close(I.supremum(), std::vector<double>{2.0, 0.5}), b + ": the upper bound");
    const Interval M(Tensor::from_data({1, 2, 3, 4}, {2, 2}),
                     Tensor::from_data({5, 6, 7, 8}, {2, 2}));
    check(M.supremum().shape() == std::vector<int64_t>({2, 2}), b + ": a matrix keeps its shape");
    check(Interval::stack({I, I}).supremum().shape() == std::vector<int64_t>({2, 2, 1}),
          b + ": a batch keeps its batch shape");
}

/// supremum has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({1.0, 2.0}));
    const Tensor on = I.supremum();
    ChecksScope off(false);
    check(close(I.supremum(), on, 0), "supremum: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(supremum_is_the_upper_bound);
    checks_and_their_switch();
    return test::finish("interval supremum");
}
