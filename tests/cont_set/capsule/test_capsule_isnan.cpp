// test_capsule_isnan - Capsule isnan and dim: no NaN, and the dimension from the center
//
// CORA: contSet/@capsule/isnan.m
// CORA: contSet/@capsule/dim.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// isnan is false; dim is the number of rows of the center, also for the empty capsule.
void isnan_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(!C.isnan() && !Capsule::empty(2).isnan(), b + ": isnan");
    check(C.dim() == 2 && Capsule::empty(3).dim() == 3 && Capsule(column({1.0})).dim() == 1,
          b + ": dim");
}

/// The call works with the checks off too.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    ChecksScope off(false);
    check(!C.isnan() && C.dim() == 2, "isnan: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { isnan_known(b); });
    checks_and_their_switch();
    return test::finish("capsule isnan");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
