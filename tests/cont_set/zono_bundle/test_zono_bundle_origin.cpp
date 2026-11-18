// test_zono_bundle_origin - ZonoBundle::origin: the bundle {0}
//
// CORA: contSet/@zonoBundle/origin.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const ZonoBundle O = ZonoBundle::origin(3);
        check(O.dim() == 3 && O.parallel_sets() == 1, b + ": one member of three dimensions");
        // layout: #163 step 2
        check(close(cora::detail::cora_center(O.zonotopes()[0]), {0, 0, 0}, 0) &&
                  // layout: #163 step 2
                  cora::detail::cora_generators(O.zonotopes()[0]).shape() == Shape{3, 0},
              b + ": the center is 0 and there are no generators");
        check(O.representsa("origin"), b + ": it represents the origin");
    });
    check(test::raises([] { ZonoBundle::origin(0); }, Error::wrong_value), "origin: n < 1");
    ChecksScope off(false);
    check(ZonoBundle::origin(2).dim() == 2, "origin: the same with the checks off");
    return test::finish("zono_bundle origin");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
