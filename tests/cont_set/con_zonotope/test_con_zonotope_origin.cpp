// test_con_zonotope_origin - ConZonotope::origin: the set {0}
//
// CORA: contSet/@conZonotope/origin.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const ConZonotope O = ConZonotope::origin(3);
        check(O.dim() == 3 && close(O.c(), {0, 0, 0}, 0), b + ": the center is 0");
        check(O.generators().shape() == Shape{3, 0} && O.A().shape()[0] == 0,
              b + ": no generators, no constraints");
        check(O.interval().sup().abs().sum().item() == 0, b + ": the box is the origin");
        check(O.representsa("origin"), b + ": it represents the origin");
    });
    check(test::raises([] { ConZonotope::origin(0); }, Error::wrong_value), "origin: n < 1");
    ChecksScope off(false);
    check(ConZonotope::origin(2).dim() == 2, "origin: the same with the checks off");
    return test::finish("con_zonotope origin");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
