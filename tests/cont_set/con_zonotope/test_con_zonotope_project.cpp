// test_con_zonotope_project - ConZonotope::project: the projection onto some dimensions
//
// CORA: contSet/@conZonotope/project.m
// PROPERTY: contSet/@conZonotope/project.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the selected rows in the given order, repeats allowed, constraints unchanged.
void project_known(const std::string &b) {
    const ConZonotope a(column({1.0, 2.0, 3.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}, {2.0, 2.0}}),
                        Tensor({{1.0, 1.0}}), column({0.5}));
    const Set s = a.project({2, 0});
    const ConZonotope &r = s.as<ConZonotope>();
    check(close(r.c(), {3, 1}, 0) && close(r.generators(), {2, 2, 1, 0}, 0),
          b + ": the rows in order");
    check(close(r.A(), a.A(), 0) && close(r.b(), a.b(), 0), b + ": the constraints stay");
    check(s.as<ConZonotope>().dim() == 2 && a.project({1, 1}).as<ConZonotope>().dim() == 2,
          b + ": repeats");
}

/// Stated property: the projection of every sampled point is in the projection.
void project_property(const std::string &b) {
    cora::Rng rng(43);
    for (int k = 0; k < 4; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 4, 5, 2);
        const ConZonotope r = cz.project({3, 1}).as<ConZonotope>();
        const Tensor points = test::sample_points(cz, rng, 20);
        check(r.contains(points.take({3, 1}, 0)), b + ": projections are inside");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        project_known(b);
        project_property(b);
    });
    const ConZonotope a(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    check(test::raises([&] { a.project({2}); }, Error::wrong_value),
          "project: a dimension too big");
    ChecksScope off(false);
    check(a.project({1}).as<ConZonotope>().dim() == 1, "project: the same with the checks off");
    return test::finish("con_zonotope project");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
