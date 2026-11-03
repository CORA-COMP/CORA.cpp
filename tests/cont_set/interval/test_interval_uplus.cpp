// test_interval_uplus - interval uplus: the unary plus of a box
//
// CORA: contSet/@interval/uplus.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void uplus_is_the_identity(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({2.0, 0.5}));
    check(close(I.uplus().inf(), I.inf(), 0) && close(I.uplus().sup(), I.sup(), 0),
          b + ": the box itself");
    check(close((+I).inf(), I.inf(), 0) && close((+I).sup(), I.sup(), 0), b + ": the operator");
}

/// uplus has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({1.0, 2.0}));
    const Tensor on = I.uplus().sup();
    ChecksScope off(false);
    check(close(I.uplus().sup(), on, 0), "uplus: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(uplus_is_the_identity);
    checks_and_their_switch();
    return test::finish("interval uplus");
}
