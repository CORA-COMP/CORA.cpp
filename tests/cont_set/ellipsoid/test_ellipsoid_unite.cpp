// test_ellipsoid_unite - Ellipsoid unite: the empty, point and refused cases
//
// CORA: contSet/@ellipsoid/or.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: empty operands, a point inside either way, inner, a point; proper ones are
/// refused.
void unite_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    const Ellipsoid inside(Tensor::zeros({2, 2}), column({1.0, 2.0}));
    check(E.unite(inside).isequal(E) && inside.unite(E).isequal(E),
          b + ": a point inside, either way");
    check(E.unite(Ellipsoid::empty(2)).isequal(E) && Ellipsoid::empty(2).unite(E).isequal(E),
          b + ": empty");
    const Ellipsoid far(Tensor::eye(2) * 0.01, column({9.0, 9.0}));
    check(E.unite(far, "inner").isequal(E), b + ": inner is the ellipsoid itself");
    check(E.unite(column({1.0, 2.0})).isequal(E), b + ": a point inside");
    check(unite(E, inside).is<Ellipsoid>(), b + ": registry");
    check(test::raises([&] { E.unite(far); }, Error::not_supported),
          b + ": two proper ones need an SDP");
    check(test::raises([&] { E.unite(Ellipsoid(Tensor::zeros({2, 2}), column({9.0, 9.0}))); },
                       Error::not_supported),
          b + ": a point outside needs an SDP");
}

/// A dimension mismatch and a bad mode are refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(test::raises([&] { E.unite(Ellipsoid(Tensor::eye(3))); }, Error::dimension_mismatch),
          "unite: dimension");
    check(test::raises([&] { E.unite(E, "middle"); }, Error::wrong_value), "unite: mode");
    check(test::raises([&] { E.unite(column({1.0})); }, Error::dimension_mismatch), "unite: point");
    ChecksScope off(false);
    check(E.unite(E, "inner").isequal(E), "unite: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { unite_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid unite");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
