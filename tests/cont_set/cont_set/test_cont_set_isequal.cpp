// test_cont_set_isequal - isequal between sets of any classes
//
// CORA: contSet/@contSet/isequal.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The same box as an interval and as a zonotope is equal in either order; a different one is not.
void equality_across_classes(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({1.0, 2.0})),
        J(column({-1.0, 0.0}), column({1.0, 2.001}));
    const Zonotope Z(column({0.0, 1.0}), Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2})),
        W(column({0.0, 1.0}), Tensor::from_data({1.0, 0.0, 0.0, 2.0}, {2, 2}));
    check(isequal(I, I) && isequal(Z, I) && isequal(I, Z) && isequal(Z, Z),
          b + ": equal sets of every class pair");
    check(!isequal(I, J) && !isequal(Z, W) && !isequal(W, I) && !isequal(I, W),
          b + ": different sets");
    check(!isequal(I, J) && isequal(I, J, 1e-2), b + ": tol relaxes the bounds");
}

/// Different dimensions and a negative tol are refused; a pair without an entry is noops; the
/// checks off skip them.
void errors_and_their_switch() {
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0})), I1(column({0.0}), column({1.0}));
    check(test::raises([&] { isequal(I, I1); }, Error::dimension_mismatch), "dimensions");
    check(test::raises([&] { isequal(I, I, -1.0); }, Error::wrong_value), "tol -1");
    check(test::raises([&] { isequal(I, test::UnregisteredSet()); }, Error::noops), "no pair");
    ChecksScope off(false);
    check(isequal(I, I, -1.0), "the checks off skip tol");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(equality_across_classes);
    errors_and_their_switch();
    return test::finish("cont_set isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
