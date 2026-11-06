// test_capsule_reduce - Capsule reduce: a capsule has one generator and stays as it is
//
// CORA: contSet/@capsule/reduce.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The reduced capsule is the same, for every order.
void reduce_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    for (const int order : {1, 3}) {
        const Capsule R = C.reduce(order).as<Capsule>();
        check(C.isequal(R, 0.0), b + ": the same capsule");
    }
}

/// An order below 1 is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(test::raises([&] { C.reduce(0); }, Error::wrong_value), "reduce: order >= 1");
    ChecksScope off(false);
    check(C.reduce(2)->dim() == 2, "reduce: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { reduce_known(b); });
    checks_and_their_switch();
    return test::finish("capsule reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
