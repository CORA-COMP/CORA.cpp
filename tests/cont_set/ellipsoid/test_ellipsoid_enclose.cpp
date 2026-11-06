// test_ellipsoid_enclose - Ellipsoid::enclose: the checks of the outer union, which needs a solver
//
// CORA: contSet/@ellipsoid/enclose.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The call is checked as CORA does and then refused: the outer union is an SDP, which CORA.cpp
/// does not have.
void enclose_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}})),
        E2(Tensor({{1.0, 0.0}, {0.0, 1.0}}), column({1.0, 0.0}));
    check(test::raises([&] { E.enclose(E2); }, Error::not_supported), b + ": two ellipsoids");
    const Tensor M({{2.0, 1.0}, {1.0, 1.0}});
    check(test::raises([&] { E.enclose(M, E2); }, Error::not_supported), b + ": E and M Eplus");
    const Tensor wide({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}});
    check(test::raises([&] { E.enclose(wide, Ellipsoid(Tensor::eye(3))); }, Error::not_supported),
          b + ": Eplus of another dimension, M (n, m)");
    // through the interface of every set: the class is checked
    const Set S = E;
    check(test::raises([&] { S->enclose(Set(E2)); }, Error::not_supported),
          b + ": ContSet::enclose");
    check(test::raises([&] { S->enclose(Set(Interval(column({0.0, 0.0}), column({1.0, 1.0})))); },
                       Error::noops),
          b + ": an ellipsoid and a box have no enclose");
}

/// The dimensions, the empty ellipsoid, a batch and the shape of M are refused; with the checks
/// off the same call is refused in the same way.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { E.enclose(Ellipsoid(Tensor({{1.0}}))); }, Error::dimension_mismatch),
          "enclose: the dimensions agree");
    check(test::raises([&] { E.enclose(Ellipsoid::empty(2)); }, Error::empty_set),
          "enclose: the empty ellipsoid");
    check(test::raises([&] { E.enclose(Tensor({{1.0, 0.0}}), E); }, Error::wrong_value),
          "enclose: M is (n, m)");
    check(test::raises([&] { E.enclose(Tensor::eye(2), Ellipsoid::empty(2)); }, Error::empty_set),
          "enclose: Eplus is empty");
    check(
        test::raises([&] { Ellipsoid(Tensor::zeros({2, 2, 2})).enclose(E); }, Error::not_supported),
        "enclose: a batch is not supported");
    ChecksScope off(false);
    check(test::raises([&] { E.enclose(E); }, Error::not_supported),
          "enclose: not supported with the checks off as well");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { enclose_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid enclose");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
