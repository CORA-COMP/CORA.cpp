// test_empty_set_interval - EmptySet interval: the empty interval of the same dimension
//
// CORA: contSet/@emptySet/interval.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_hull_is_the_empty_interval(const std::string &b) {
    const Interval I = EmptySet(3).interval();
    check(I.is_empty() && I.dim() == 3, b + ": empty, 3 dimensions");
    check(Set(EmptySet(2))->interval().dim() == 2, b + ": through a Set");
}

/// interval takes no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(EmptySet(2).interval().is_empty(), "interval: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_hull_is_the_empty_interval);
    checks_and_their_switch();
    return test::finish("empty_set interval");
}
