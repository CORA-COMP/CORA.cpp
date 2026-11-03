// test_empty_set_radius - EmptySet radius: 0
//
// CORA: contSet/@emptySet/radius.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_radius_is_zero(const std::string &b) {
    for (const int64_t n : {0, 1, 7})
        check(EmptySet(n).radius() == 0.0, b + ": dimension " + std::to_string(n));
}

/// radius takes no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(EmptySet(2).radius() == 0.0, "radius: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_radius_is_zero);
    checks_and_their_switch();
    return test::finish("empty_set radius");
}
