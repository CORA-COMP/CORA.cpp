// test_zono_bundle_zonotope - ZonoBundle::zonotope: the member of least volume
//
// CORA: contSet/@zonoBundle/zonotope.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
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
        const Zonotope big(column({0.0, 0.0}), Tensor({{2.0, 0.0}, {0.0, 1.0}}));
        const Zonotope small(column({0.5, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.25}}));
        check(close(ZonoBundle({big, small}).zonotope().generators(), small.generators(), 0),
              b + ": the member of volume 0.125 (before 2)");
        check(close(ZonoBundle({small, big}).zonotope().center(), small.center(), 0),
              b + ": the order does not matter");
        check(close(ZonoBundle({big, big}).zonotope().generators(), big.generators(), 0),
              b + ": a tie gives the first");
        check(ZonoBundle::origin(2).zonotope().generators().shape() == Shape{2, 0},
              b + ": a point is its own member");
    });
    const Zonotope Z(column({0.0}), Tensor({{1.0}}));
    // wrong input is refused; the same call works with the checks off
    check(test::raises([&] { ZonoBundle({Zonotope::stack({Z, Z})}).zonotope(); },
                       Error::not_supported),
          "zonotope: a batch");
    ChecksScope off(false);
    check(test::two_zonotopes().zonotope().dim() == 2, "zonotope: the same with the checks off");
    return test::finish("zono_bundle zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
