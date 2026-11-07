// test_lean_zonotope_boundary_point_enclosure - Zonotope::boundary_point against the CORALean
//
// oracle, as enclosure
//
// The oracle gives the box swept from `start` to an upper bound of the exit step (`boundaryPoint`),
// nothing if no facet row bounds the ray, and states that CORA's single point lies in it. So the
// check is that CORA.cpp's point lies in the box; the box is not a point at nearest,
// so no bit-exact test exists and there is no ORACLE tag.
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("zonotope.boundaryPoint", [](test::oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {3, 6}, {4, 8}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const Tensor dir = rnd.vector(n);
            // from the center, and from a point of the zonotope off the center
            const Tensor inside = Z.center() + Z.generators().slice(-1, 0, 1) * 0.5;
            for (const Tensor &start : {Z.center(), inside}) {
                const Tensor x = Z.boundary_point(dir, start);
                const std::optional<lean::Interval> box = lean::Zonotope(Z).boundary_point(
                    lean::Tensor::from(dir), lean::Tensor::from(start));
                test::check(box.has_value(), "zonotope.boundaryPoint: the oracle bounds the ray");
                if (box) test::oracle::check_inside(Interval(x, x), *box);
            }
        }
    });
    return test::finish("lean zonotope boundary_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
