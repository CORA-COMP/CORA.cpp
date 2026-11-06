// test_polytope_complement - Polytope::complement: of a halfspace, the empty set or R^n
//
// CORA: contSet/@polytope/not.m
// PROPERTY: contSet/@polytope/not.m
//
// Property: no point is in a halfspace and in its complement
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A halfspace, the empty set and the whole space.
void complements(const std::string &b) {
    const Polytope halfspace(Tensor({{1.0, 2.0}}), column({3.0}));
    const Polytope other = halfspace.complement();
    const double eps = machine_eps(DType::float64);
    check(close(other.A(), {-1.0, -2.0}, 0) && close(other.b(), {-3.0 - eps}, 0),
          b + ": the opposite halfspace, moved by eps");
    check(Polytope::empty(2).complement().representsa("fullspace"), b + ": the empty set");
    check(Polytope::inf(2).complement().representsa("emptySet"), b + ": R^n");
}

/// A point of the halfspace is not in the complement.
void properties(const std::string &b) {
    Rng rng(4);
    bool disjoint = true;
    for (int k = 0; k < 30; ++k) {
        std::vector<double> a = test::random_direction(rng, 3);
        const Polytope halfspace(Tensor::from_data(a, {1, 3}), column({0.5}));
        const Polytope other = halfspace.complement();
        const Tensor x = column(test::random_direction(rng, 3));
        disjoint = disjoint && !(halfspace.contains(x) && other.contains(x));
    }
    check(disjoint, b + ": no point is in both");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        complements(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().complement(); }, Error::not_supported),
          "complement: several inequalities");
    ChecksScope off(false);
    check(Polytope::empty(2).complement().representsa("fullspace"),
          "complement: the same with the checks off");
    return test::finish("polytope complement");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
