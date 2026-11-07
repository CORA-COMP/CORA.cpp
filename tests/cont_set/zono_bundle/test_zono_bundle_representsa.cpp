// test_zono_bundle_representsa - ZonoBundle::representsa
//
// CORA: contSet/@zonoBundle/representsa_.m
// CORA authors: Mark Wetzlinger, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const ZonoBundle zono_bundle = test::two_zonotopes();
        const Zonotope box(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
        check(ZonoBundle::origin(2).representsa("origin") && !zono_bundle.representsa("origin"),
              b + ": origin");
        check(ZonoBundle::empty(2).representsa("emptySet") && !zono_bundle.representsa("emptySet"),
              b + ": emptySet");
        // the types that every bundle is, and the ones that none is
        check(zono_bundle.representsa("conZonotope") && zono_bundle.representsa("polytope") &&
                  zono_bundle.representsa("zonoBundle") && zono_bundle.representsa("convexSet"),
              b + ": the types every bundle is");
        check(!zono_bundle.representsa("halfspace") && !zono_bundle.representsa("fullspace") &&
                  !zono_bundle.representsa("hyperplane") &&
                  !zono_bundle.representsa("probZonotope"),
              b + ": the types no bundle is");
        check(ZonoBundle({box, box}).representsa("interval") &&
                  !zono_bundle.representsa("interval"),
              b + ": interval: every member a box");
        check(ZonoBundle::origin(1).representsa("interval"), b + ": interval: one dimension");
    });
    const ZonoBundle zono_bundle = test::two_zonotopes();
    // wrong input is refused; the same call works with the checks off
    check(test::raises([&] { zono_bundle.representsa("point"); }, Error::not_supported),
          "representsa: point");
    check(test::raises([&] { zono_bundle.representsa("capsule"); }, Error::not_supported),
          "representsa: capsule");
    check(test::raises([&] { zono_bundle.representsa("nonsense"); }, Error::wrong_value),
          "representsa: a name");
    ChecksScope off(false);
    check(zono_bundle.representsa("polytope"), "representsa: the same with the checks off");
    return test::finish("zono_bundle representsa");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
