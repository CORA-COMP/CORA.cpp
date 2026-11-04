// test_capsule_origin - Capsule::origin: the capsule that is the origin
//
// CORA: contSet/@capsule/origin.m

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// The origin has center 0, generator 0 and radius 0, and contains exactly the origin.
void origin_known(const std::string &b) {
    const Capsule C = Capsule::origin(3);
    check(C.dim() == 3 && close(C.c(), {0, 0, 0}, 0) && close(C.generator(), {0, 0, 0}, 0) &&
              close(C.r(), {0}, 0),
          b + ": center, generator and radius are 0");
    check(C.contains(column({0.0, 0.0, 0.0})) && !C.contains(column({0.0, 0.0, 1e-3})),
          b + ": it contains only the origin");
    check(!C.is_empty(), b + ": it is not empty");
}

/// A dimension below 1 is refused; the same call with the checks off still works.
void checks_and_their_switch() {
    check(test::raises([] { Capsule::origin(0); }, Error::wrong_value),
          "origin: a positive dimension");
    ChecksScope off(false);
    check(Capsule::origin(2).dim() == 2, "origin: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(origin_known);
    checks_and_their_switch();
    return test::finish("capsule origin");
}
