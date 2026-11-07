// test_polytope_volume - Polytope::volume: the volume of a bounded polytope
//
// CORA: contSet/@polytope/volume_.m
// PROPERTY: contSet/@polytope/volume_.m
//
// Property: the volume of M [0, 1]^n is |det M|
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed volumes in both forms, and the special sets.
void volumes(const std::string &b) {
    check(close(test::square_h().volume(), 1.0) && close(test::square_v().volume(), 1.0),
          b + ": the unit square");
    check(close(test::triangle_h().volume(), 2.0), b + ": the triangle");
    // the unit simplex in 3D: 1/6
    const Polytope simplex(
        Tensor({{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}}));
    check(close(simplex.volume(), 1.0 / 6.0), b + ": the 3D simplex");
    const Polytope box(Interval(column({0.0, 1.0, 2.0}), column({2.0, 4.0, 3.0})));
    check(close(box.volume(), 6.0), b + ": a 3D box");
    constexpr double inf = std::numeric_limits<double>::infinity();
    check(Polytope::inf(2).volume() == inf &&
              Polytope(Tensor({{1.0, 0.0}}), column({1.0})).volume() == inf,
          b + ": unbounded: Inf");
    check(Polytope::empty(2).volume() == 0.0 && Polytope(column({1.0, 2.0})).volume() == 0.0,
          b + ": empty and a point: 0");
    check(Polytope(Tensor({{0.0, 1.0}, {0.0, 1.0}})).volume() == 0.0, b + ": a segment in 2D: 0");
    check(close(Polytope(Tensor({{-1.0}, {1.0}}), column({0.0, 3.0})).volume(), 3.0), b + ": 1D");
}

/// The image of the unit cube under M has the volume |det M|.
void determinant(const std::string &b) {
    Rng rng(4);
    bool same = true;
    const Polytope cube(Tensor({{0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}}));
    for (int k = 0; k < 5; ++k) {
        std::vector<double> m(4);
        rng.normal(m.data(), m.size(), 1.0);
        const double det = m[0] * m[3] - m[1] * m[2];
        const Polytope image = cube.linear_map(Tensor::from_data(m, {2, 2}));
        same = same && close(image.volume(), std::abs(det), 1e-8);
    }
    check(same, b + ": 2D: the volume of M [0, 1]^2 is |det M|");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        volumes(b);
        determinant(b);
    });
    check(test::raises([] { Polytope(Tensor::ones({3, 2, 2}), Tensor::ones({3, 2, 1})).volume(); },
                       Error::not_supported),
          "volume: a batch");
    ChecksScope off(false);
    check(close(test::square_h().volume(), 1.0), "volume: the same with the checks off");
    return test::finish("polytope volume");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
