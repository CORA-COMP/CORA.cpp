// test_poly_zonotope_interval - PolyZonotope interval: the box of the enclosing zonotope
//
// CORA: contSet/@polyZonotope/interval.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// pZ = {(1, 0) + a (1, 1) + a b (0, 2) + (0, 1) t}: generators on a and a b, one independent.
PolyZonotope known() {
    return {column({1.0, 0.0}),
            Tensor::from_data({1, 0, 1, 2}, {2, 2}),
            Tensor::from_data({0, 1}, {2, 1}),
            {{1, 0}, {1, 1}},
            {1, 2}};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const PolyZonotope P = known();
        const Interval I = P.interval();
        // c +- (|1| + 0) and (|1| + |2| + 1): the zonotope has the generators (1,1), (0,2), (0,1)
        check(close(I.inf(), {0.0, -4.0}, 0) && close(I.sup(), {2.0, 4.0}, 0),
              b + ": the box of the zonotope");
    });

    ChecksScope off(false);
    check(known().interval().inf().shape()[0] == 2, "interval: valid with the checks off");
    return test::finish("poly_zonotope interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
