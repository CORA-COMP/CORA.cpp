// test_polytope_inf - Polytope::inf: R^n as a polytope
//
// CORA: contSet/@polytope/Inf.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// No constraints; it holds every point and is its own box.
void whole_space(const std::string &b) {
    const Polytope P = Polytope::inf(3);
    check(P.dim() == 3 && P.is_h_rep() && P.A().shape() == Shape{0, 3} && P.b().shape() == Shape{0},
          b + ": no constraints");
    check(P.contains(Tensor({{1e9, -1e9, 0.0}})), b + ": it holds every point");
    check(close(P.center(), {0, 0, 0}, 0), b + ": its center is the origin");
    const Interval box = P.interval();
    check(box.inf().min().item() == -std::numeric_limits<double>::infinity() &&
              box.sup().max().item() == std::numeric_limits<double>::infinity(),
          b + ": its box is unbounded");
}

/// A negative dimension is refused; with the checks off the dimension is still valid.
void refusals() {
    check(test::raises([] { Polytope::inf(-1); }, Error::wrong_value),
          "inf: the dimension is nonnegative");
    ChecksScope off(false);
    check(Polytope::inf(2).dim() == 2, "inf: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { whole_space(b); });
    refusals();
    return test::finish("polytope inf");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
