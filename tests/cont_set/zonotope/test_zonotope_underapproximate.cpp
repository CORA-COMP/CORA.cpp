// test_zonotope_underapproximate - zonotope underapproximate: vertices in given directions
//
// CORA: contSet/@zonotope/underapproximate.m
// PROPERTY: contSet/@zonotope/underapproximate.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// CORA's unit test: Z = zonotope([-4 -3 -2 -1; 1 2 3 4]) with the default directions (more
/// generators than dimensions: dominant_directions) and with S; then the generators as directions.
void cora_unit_test(const std::string &b) {
    const Zonotope Z(column({-4.0, 1.0}), Tensor({{-3.0, -2.0, -1.0}, {2.0, 3.0, 4.0}}));
    check(close(Z.underapproximate(), std::vector<double>{-10, 2, -10, 2, 10, -8, 10, -8}, 1e-12),
          b + ": the default directions");
    const Tensor S({{1.0, 1.0}, {0.0, 1.0}});
    check(close(Z.underapproximate(S), std::vector<double>{2, -10, -4, -4, -8, 10, 6, -4}, 1e-12),
          b + ": the directions S");
    // a zonotope with at least as many generators as dimensions uses them as directions
    const Zonotope Y(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    check(close(Y.underapproximate(), std::vector<double>{1, -1, 0, 0, 0, 0, 2, -2}, 0),
          b + ": the generators as directions");
}

/// Stated property: every vertex lies in the zonotope, and the vertex for a direction attains the
/// support function in that direction (the vertex of a direction maximizes s' x).
void the_vertices_are_in_the_zonotope(const std::string &b) {
    cora::Rng rng(51);
    const int n = 3, m = 6;
    const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, m);
    std::vector<double> data(n * 4);
    rng.normal(data.data(), data.size(), 1.0);
    const Tensor S = Tensor::from_data(data, {n, 4});
    const Tensor V = Z.underapproximate(S);
    check(V.shape() == Shape{n, 8}, b + ": 2k vertices");
    for (int j = 0; j < 8; ++j)
        check(Z.contains(V.slice(-1, j, j + 1), "exact", 1e-9), b + ": a vertex lies in Z");
    for (int i = 0; i < 4; ++i) {
        std::vector<double> d(n);
        for (int k = 0; k < n; ++k) d[k] = data[k * 4 + i];
        const Tensor direction = S.slice(-1, i, i + 1), vertex = V.slice(-1, 2 * i, 2 * i + 1);
        const double reach = direction.matrix_transpose().matmul(vertex).item();
        check(close(reach, test::support(Z, d), 1e-9), b + ": the vertex attains the support");
    }
}

/// A batch is the result of each member.
void a_batch(const std::string &b) {
    cora::Rng rng(52);
    const Zonotope A = Zonotope::generate_random(rng, 2, std::nullopt, 3),
                   B = Zonotope::generate_random(rng, 2, std::nullopt, 3);
    const Tensor S({{1.0, 0.5}, {0.0, 1.0}});
    const Tensor V = Zonotope::stack({A, B}).underapproximate(S);
    check(V.shape() == Shape{2, 2, 4}, b + ": batch shape");
    check(close(V.slice(0, 1, 2).reshape({2, 4}), B.underapproximate(S), 1e-12), b + ": member 2");
}

/// Directions of the wrong dimension are refused; a batch cannot choose default directions; with
/// the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}));
    check(test::raises([&] { Z.underapproximate(Tensor({{1.0}, {2.0}, {3.0}})); },
                       Error::dimension_mismatch),
          "underapproximate: directions of dimension 3");
    check(test::raises([&] { Zonotope::stack({Z, Z}).underapproximate(); }, Error::not_supported),
          "underapproximate: default directions of a batch");
    const Tensor on = Z.underapproximate(Tensor({{1.0}, {0.0}}));
    ChecksScope off(false);
    check(close(Z.underapproximate(Tensor({{1.0}, {0.0}})), on, 0),
          "underapproximate: same with checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        cora_unit_test(b);
        the_vertices_are_in_the_zonotope(b);
        a_batch(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope underapproximate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
