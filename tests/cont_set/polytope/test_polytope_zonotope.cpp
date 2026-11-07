// test_polytope_zonotope - Polytope::zonotope: a zonotope around a bounded polytope
//
// CORA: contSet/@polytope/zonotope.m
// PROPERTY: contSet/@polytope/zonotope.m
//
// Property: the zonotope contains the vertices of the polytope
//
// CORA authors: Mark Wetzlinger, Niklas Kochdumper
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

/// Squares and triangles are enclosed; the zonotope of a box contains the box.
void enclosures(const std::string &b) {
    for (const Polytope &P : {test::square_h(), test::square_v(), test::triangle_h()}) {
        const Zonotope Z = P.zonotope();
        check(Z.dim() == 2 && Z.contains(test::vertices_of(P), "exact", 1e-8),
              b + ": it contains the vertices");
    }
    // a point is a zonotope without generators' volume
    check(close(Polytope(column({1.0, 2.0})).zonotope().center(), {1.0, 2.0}, 1e-9),
          b + ": a point");
}

/// The zonotope contains the polytope of random sets.
void properties(const std::string &b) {
    Rng rng(4);
    bool contained = true;
    for (int n = 2; n <= 4; ++n) {
        const Polytope P = test::random_polytope(rng, n, n + 2);
        contained = contained && P.zonotope().contains(test::vertices_of(P), "exact", 1e-7);
    }
    check(contained, b + ": it contains the vertices");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        enclosures(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().zonotope("big"); }, Error::wrong_value),
          "zonotope: the method");
    check(test::raises([] { test::square_h().zonotope("exact"); }, Error::not_supported),
          "zonotope: exact");
    check(test::raises([] { Polytope::inf(2).zonotope(); }, Error::special_error), "zonotope: R^n");
    check(test::raises([] { Polytope::empty(2).zonotope(); }, Error::special_error),
          "zonotope: empty");
    ChecksScope off(false);
    check(test::square_h().zonotope().dim() == 2, "zonotope: the same with the checks off");
    return test::finish("polytope zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
