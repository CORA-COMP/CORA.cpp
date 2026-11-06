// test_zono_bundle_origin - ZonoBundle::origin: the bundle {0}
//
// CORA: contSet/@zonoBundle/origin.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const ZonoBundle O = ZonoBundle::origin(3);
        check(O.dim() == 3 && O.parallel_sets() == 1, b + ": one member of three dimensions");
        check(close(O.zonotopes()[0].center(), {0, 0, 0}, 0) &&
                  O.zonotopes()[0].generators().shape() == Shape{3, 0},
              b + ": the center is 0 and there are no generators");
        check(O.representsa("origin"), b + ": it represents the origin");
    });
    check(test::raises([] { ZonoBundle::origin(0); }, Error::wrong_value), "origin: n < 1");
    ChecksScope off(false);
    check(ZonoBundle::origin(2).dim() == 2, "origin: the same with the checks off");
    return test::finish("zono_bundle origin");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
