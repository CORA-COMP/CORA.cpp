// test_capsule_isequal - Capsule isequal: the same set up to a tolerance
//
// CORA: contSet/@capsule/isequal.m
//
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Equal fields, a field off by more than tol, the tolerance that closes the gap, empty capsules.
void isequal_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(C.isequal(Capsule(column({1.0, 2.0}), column({3.0, 4.0}), 0.5)), b + ": equal");
    check(C.isequal(C, 0.0), b + ": equal with tol = 0 (CORA's strict < never holds there)");
    check(!C.isequal(Capsule(column({1.0, 2.1}), column({3.0, 4.0}), 0.5)), b + ": center");
    check(!C.isequal(Capsule(column({1.0, 2.0}), column({3.0, 4.1}), 0.5)), b + ": generator");
    check(!C.isequal(Capsule(column({1.0, 2.0}), column({3.0, 4.0}), 0.6)), b + ": radius");
    check(C.isequal(Capsule(column({1.0, 2.0 + 1e-3}), column({3.0, 4.0}), 0.5), 1e-2),
          b + ": a tolerance closes a small gap");
    check(!C.isequal(Capsule(column({1.0, 2.0, 3.0}))), b + ": another dimension is false");
    check(Capsule::empty(2).isequal(Capsule::empty(2)), b + ": two empty capsules");
    check(!Capsule::empty(2).isequal(C) && !C.isequal(Capsule::empty(2)),
          b + ": empty against nonempty");
    check(Capsule::empty(1).isequal(Capsule::empty(1)), b + ": in one dimension");
}

/// Through the registry: == is isequal with the default tolerance; a batch needs every member.
void isequal_registry_and_batch(const std::string &b) {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(Set(C) == Set(Capsule(column({1.0, 2.0}), column({3.0, 4.0}), 0.5)), b + ": operator==");
    check(Set(C) != Set(Capsule(column({1.0, 2.0}), column({3.0, 4.0}), 0.7)), b + ": operator!=");
    check(isequal(C, C), b + ": the free function");
    const Tensor c = Tensor::zeros({2, 2, 1});
    const Capsule batch(c, c, Tensor::from_data({0.1, 0.2}, {2, 1, 1}));
    check(batch.isequal(batch), b + ": a batch with itself");
    check(!batch.isequal(Capsule(c, c, Tensor::from_data({0.1, 0.3}, {2, 1, 1}))),
          b + ": one member differs");
}

/// A negative tolerance is refused; with the checks off the answer is the same.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(test::raises([&] { C.isequal(C, -1.0); }, Error::wrong_value),
          "isequal: tol is nonnegative");
    ChecksScope off(false);
    check(C.isequal(C), "isequal: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        isequal_known(b);
        isequal_registry_and_batch(b);
    });
    checks_and_their_switch();
    return test::finish("capsule isequal");
}


// ---------------------------------------  END OF CODE  ---------------------------------------- //
