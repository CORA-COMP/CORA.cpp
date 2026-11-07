// test_zono_bundle_replace - ZonoBundle::replace: a member replaced
//
// CORA: contSet/@zonoBundle/replace.m
// CORA authors: Matthias Althoff
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
        const ZonoBundle zono_bundle = test::two_zonotopes();
        const Zonotope W(column({3.0, 3.0}), Tensor({{1.0}, {1.0}}));
        const ZonoBundle R = zono_bundle.replace(1, W);
        check(R.parallel_sets() == 2 && close(R.zonotopes()[1].center(), {3, 3}, 0) &&
                  close(R.zonotopes()[0].center(), {0, 0}, 0),
              b + ": member 1 replaced, member 0 kept");
        check(close(zono_bundle.zonotopes()[1].center(), {1, 0}, 0),
              b + ": the original is unchanged");
    });
    const ZonoBundle zono_bundle = test::two_zonotopes();
    const Zonotope W(column({3.0, 3.0}), Tensor({{1.0}, {1.0}}));
    // wrong input is refused; the same call works with the checks off
    check(test::raises([&] { zono_bundle.replace(2, W); }, Error::wrong_value),
          "replace: an index too big");
    check(test::raises([&] { zono_bundle.replace(0, Zonotope(column({0.0}), Tensor({{1.0}}))); },
                       Error::dimension_mismatch),
          "replace: a member of another dimension");
    ChecksScope off(false);
    check(zono_bundle.replace(0, W).parallel_sets() == 2, "replace: the same with the checks off");
    return test::finish("zono_bundle replace");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
