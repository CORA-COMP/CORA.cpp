// test_polytope_empty - Polytope::empty: the empty set of n dimensions
//
// CORA: contSet/@polytope/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The row 0 <= -1, no vertices; nothing for n = 0.
void empty_set(const std::string &b) {
    const Polytope P = Polytope::empty(3);
    check(P.dim() == 3 && close(P.A(), {0, 0, 0}, 0) && close(P.b(), {-1}, 0),
          b + ": the row 0 <= -1");
    check(P.is_v_rep() && P.V().shape() == Shape{3, 0}, b + ": no vertices");
    check(P.Ae().shape() == Shape{0, 3}, b + ": no equalities");
    check(Polytope::empty(0).A().shape() == Shape{0, 0}, b + ": n = 0: no row");
    check(Polytope::empty().dim() == 0, b + ": the default dimension is 0");
    check(Polytope::empty(1).A().shape() == Shape{1, 1}, b + ": one row in 1D");
}

/// The empty set has no point, no center, and an empty box.
void properties(const std::string &b) {
    const Polytope P = Polytope::empty(2);
    check(!P.contains(Tensor({{0.0}, {0.0}})), b + ": it holds no point");
    check(P.center().shape() == Shape{2, 0}, b + ": it has no center");
    check(P.interval().is_empty(), b + ": its box is empty");
}

/// A negative dimension is refused; with the checks off the dimension is still valid.
void refusals() {
    check(test::raises([] { Polytope::empty(-1); }, Error::wrong_value),
          "empty: the dimension is nonnegative");
    ChecksScope off(false);
    check(Polytope::empty(2).dim() == 2, "empty: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        empty_set(b);
        properties(b);
    });
    refusals();
    return test::finish("polytope empty");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
