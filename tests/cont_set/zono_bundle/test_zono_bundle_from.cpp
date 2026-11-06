// test_zono_bundle_from - ZonoBundle::from: a zonotope or a box as a bundle of one member
//
// CORA: contSet/@zonotope/zonoBundle.m
// CORA: contSet/@interval/zonoBundle.m
// CORA authors: Niklas Kochdumper
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
        const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.5, 1.0}}));
        const ZonoBundle from_zonotope = ZonoBundle::from(Z);
        check(from_zonotope.parallel_sets() == 1 &&
                  close(from_zonotope.zonotopes()[0].center(), Z.center(), 0) &&
                  close(from_zonotope.zonotopes()[0].generators(), Z.generators(), 0),
              b + ": a zonotope is its own member");
        const ZonoBundle from_box =
            ZonoBundle::from(Interval(column({0.0, -1.0}), column({2.0, 1.0})));
        check(from_box.parallel_sets() == 1 && close(from_box.zonotopes()[0].center(), {1, 0}, 0) &&
                  close(from_box.zonotopes()[0].generators(), {1, 0, 0, 1}, 0),
              b + ": a box is the zonotope of its center and radius");
        check(close(from_box.interval().sup(), {2, 1}, 0), b + ": the box is the same");
    });
    // wrong input is refused; the same call works with the checks off
    check(test::raises([] { ZonoBundle::from(Interval::empty(2)); }, Error::not_supported),
          "from: the empty box has no zonotope");
    ChecksScope off(false);
    check(ZonoBundle::from(Interval(column({0.0}), column({1.0}))).dim() == 1,
          "from: the same with the checks off");
    return test::finish("zono_bundle from");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
