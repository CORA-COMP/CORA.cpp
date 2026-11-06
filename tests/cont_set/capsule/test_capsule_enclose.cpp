// test_capsule_enclose - Capsule enclose: a capsule that encloses two capsules
//
// CORA: contSet/@capsule/enclose.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
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

/// The one-capsule form is the convex hull, also through the set interface.
void enclose_known(const std::string &b) {
    const Capsule C1(column({0.0, 0.0}), column({1.0, 0.0}), 0.5);
    const Capsule C2(column({4.0, 0.0}), column({0.0, 1.0}), 0.25);
    check(C1.enclose(C2).isequal(C1.convex_hull(C2), 0.0), b + ": the hull of the two");
    const Set S = C1;
    check(S->enclose(C2).as<Capsule>().isequal(C1.convex_hull(C2), 0.0), b + ": ContSet::enclose");
    check(test::raises([&] { S->enclose(Zonotope(column({0.0, 0.0}), Tensor({{1.0}, {0.0}}))); },
                       Error::noops),
          b + ": another class has no pair");
}

/// The form with a matrix is CORA's wrong one, refused with a reason; its input is checked.
void enclose_matrix_form(const std::string &b) {
    const Capsule C1(column({0.0, 0.0}), column({1.0, 0.0}), 0.5);
    const Tensor M({{1.0, 0.0}, {0.0, 1.0}});
    check(test::raises([&] { C1.enclose(M, C1); }, Error::not_supported),
          b + ": enclose(M, Cplus) is not supported");
    check(
        test::raises([&] { C1.enclose(Tensor({{1.0, 0.0, 0.0}}), C1); }, Error::dimension_mismatch),
        b + ": M has the dimensions of C and Cplus");
}

/// Capsules of different dimensions are refused; with the checks off the same call works.
void checks_and_their_switch() {
    const Capsule C1(column({0.0, 0.0}), column({1.0, 0.0}), 0.5);
    check(test::raises([&] { C1.enclose(Capsule(column({1.0}))); }, Error::dimension_mismatch),
          "enclose: the dimensions agree");
    const Capsule on = C1.enclose(C1);
    ChecksScope off(false);
    check(close(C1.enclose(C1).r(), on.r(), 0), "enclose: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        enclose_known(b);
        enclose_matrix_form(b);
    });
    checks_and_their_switch();
    return test::finish("capsule enclose");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
