// test_polygon_class - Polygon::Polygon: boundaries split into convex pieces, pieces, regions
//
// CORA: contSet/@polygon/polygon.m
// PROPERTY: contSet/@polygon/polygon.m
// CORA: contSet/@polygon/dim.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A boundary: convex is one piece, the L two or more, NaN rows separate regions.
void boundaries(const std::string &b) {
    check(test::unit_square().num_pieces() == 1 && test::unit_square().dim() == 2,
          b + ": a convex boundary is one piece of two dimensions");
    check(test::l_shape().num_pieces() >= 2 && close(test::l_shape().volume(), 3.0),
          b + ": the L is cut into convex pieces of area 3");
    const double nan = std::nan("");
    const Polygon two =
        test::boundary({{0, 0}, {1, 0}, {0, 1}, {nan, nan}, {3, 3}, {4, 3}, {3, 4}});
    check(two.num_pieces() == 2 && close(two.volume(), 1.0), b + ": two regions");
    // clockwise is the same region
    check(close(test::boundary({{0, 1}, {1, 1}, {1, 0}, {0, 0}}).volume(), 1.0), b + ": clockwise");
}

/// Stated property: the pieces of a random star-shaped boundary are convex, disjoint and cover it.
void star_split() {
    Rng rng(5);
    for (int trial = 0; trial < 20; ++trial) {
        const int k = 5 + trial % 9;
        std::vector<double> r(k);
        rng.uniform(r.data(), r.size(), 0.3, 2.0);
        std::vector<std::vector<double>> rows;
        double shoelace = 0;
        for (int i = 0; i < k; ++i) {
            const double t = 2 * M_PI * i / k, s = 2 * M_PI * ((i + 1) % k) / k;
            rows.push_back({r[i] * std::cos(t), r[i] * std::sin(t)});
            shoelace += r[i] * r[(i + 1) % k] * std::sin(s - t) / 2;
        }
        const Polygon pgon = test::boundary(rows);
        // the pieces' own areas against the shoelace area of the boundary
        double sum = 0;
        for (const Polytope &P : pgon.pieces()) sum += P.volume();
        check(close(sum, shoelace, 1e-9) && close(pgon.volume(), shoelace, 1e-9),
              "class: the pieces of a star boundary add up to its area, without overlap");
        check(pgon.is_full_dim(), "class: a star boundary has an interior");
    }
}

/// Pieces: kept in their order, a halfspace piece by its vertices; the refusals.
void pieces_and_refusals() {
    const Polygon pgon({Polytope(Interval(vec({0.0, 0.0}), vec({1.0, 2.0}))),
                        Polytope(Tensor({{3.0, 0.0}, {4.0, 0.0}, {3.0, 1.0}}))});
    check(pgon.num_pieces() == 2 && pgon.pieces()[0].is_v_rep() && close(pgon.volume(), 2.5),
          "class: a box piece is held by its vertices");
    check(
        test::raises([] { Polygon(Tensor({{0.0, 0.0, 1.0}})); }, Error::wrong_input_in_constructor),
        "class: a boundary of three columns is refused");
    check(test::raises([] { Polygon({Polytope(Tensor({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}))}); },
                       Error::dimension_mismatch),
          "class: a piece of three dimensions is refused");
    const double nan = std::nan("");
    check(test::raises(
              [&] {
                  test::boundary(
                      {{0, 0}, {4, 0}, {4, 4}, {0, 4}, {nan, nan}, {1, 1}, {2, 1}, {1, 2}});
              },
              Error::not_supported),
          "class: a hole is not supported yet");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { boundaries(b); });
    star_split();
    pieces_and_refusals();
    {
        ChecksScope off(false);
        check(test::l_shape().num_pieces() >= 2, "class: the same with the checks off");
    }
    return test::finish("polygon class");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
