// test_zono_bundle_isequal - ZonoBundle::isequal: whether two bundles are the same set
//
// CORA: contSet/@zonoBundle/isequal.m
// CORA authors: Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Equal members decide at once; reordered members are the same set (found by containment); other
/// sets and other dimensions are not equal.
void known_bundles(const std::string &b) {
    const Tensor I2({{1.0, 0.0}, {0.0, 1.0}});
    const Zonotope A(column({0.0, 0.0}), I2), B(column({1.0, 1.0}), I2), C(column({0.0, 0.5}), I2);
    check(ZonoBundle({A, B}).isequal(ZonoBundle({A, B})), b + ": the same members");
    check(ZonoBundle({A, B}).isequal(ZonoBundle({B, A})), b + ": the members reordered");
    check(ZonoBundle({A, B}).isequal(ZonoBundle({A, B, A})), b + ": a redundant member");
    check(!ZonoBundle({A, B}).isequal(ZonoBundle({A, C})), b + ": another member");
    check(!ZonoBundle({A}).isequal(ZonoBundle({A, B})), b + ": a smaller intersection");
    check(!ZonoBundle({A}).isequal(ZonoBundle({Zonotope(column({0.0}), Tensor({{1.0}}))})),
          b + ": another dimension");
    check(isequal(Set(ZonoBundle({A, B})), Set(ZonoBundle({B, A}))), b + ": through the registry");
    check(Set(ZonoBundle({A, B})) == Set(ZonoBundle({B, A})), b + ": the operator ==");
}

/// The members are compared up to tol.
void tolerance(const std::string &b) {
    const Tensor I2({{1.0, 0.0}, {0.0, 1.0}});
    const Zonotope A(column({0.0, 0.0}), I2), near(column({1e-15, 0.0}), I2),
        apart(column({1e-3, 0.0}), I2);
    check(ZonoBundle({A}).isequal(ZonoBundle({near})), b + ": equal up to the default tolerance");
    check(!ZonoBundle({A}).isequal(ZonoBundle({apart})), b + ": not equal at 1e-3");
    check(ZonoBundle({A}).isequal(ZonoBundle({apart}), 1e-2), b + ": equal with tol 1e-2");
}

/// A negative tol and a batch are refused; the same call with the checks off.
void checks_and_their_switch() {
    const Zonotope A(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const ZonoBundle bundle({A});
    check(test::raises([&] { bundle.isequal(bundle, -1.0); }, Error::wrong_value),
          "isequal: tol < 0");
    check(test::raises([&] { bundle.isequal(ZonoBundle({Zonotope::stack({A, A})})); },
                       Error::not_supported),
          "isequal: a batch");
    ChecksScope off(false);
    check(bundle.isequal(bundle), "isequal: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        known_bundles(b);
        tolerance(b);
    });
    checks_and_their_switch();
    return test::finish("zono_bundle isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
