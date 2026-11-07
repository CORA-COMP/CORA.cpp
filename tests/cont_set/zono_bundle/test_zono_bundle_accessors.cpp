// test_zono_bundle_accessors - ZonoBundle: the dimension
//
// CORA: contSet/@zonoBundle/dim.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const ZonoBundle zono_bundle = test::two_zonotopes();
        check(zono_bundle.dim() == 2 && Set(zono_bundle)->dim() == 2,
              b + ": the dimension of the members");
        check(ZonoBundle::origin(5).dim() == 5, b + ": origin");
    });
    return test::finish("zono_bundle accessors");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
