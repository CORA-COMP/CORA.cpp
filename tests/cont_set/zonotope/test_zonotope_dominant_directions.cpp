// test_zonotope_dominant_directions - zonotope dominant_directions: a tight parallelotope
//
// CORA: contSet/@zonotope/dominantDirections.m
// PROPERTY: contSet/@zonotope/dominantDirections.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Three generators in the plane: the two long ones span the smallest box around the rest.
void a_known_choice(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0, 0.1}, {0.0, 2.0, 0.1}}));
    const Tensor S = Z.dominant_directions();
    check(close(S, std::vector<double>{0, 1, 2, 0}, 0), b + ": columns (0, 2), (1, 0)");
    check(close(Z.dominant_directions(5, 2), S, 0), b + ": explicit filter lengths");
    check(close(Z.dominant_directions(2, 1), S, 0), b + ": the two longest generators");
}

/// Stated property: the chosen generators are generators of Z, and the parallelotope they span
/// around the box of P^-1 Z contains the zonotope.
void the_parallelotope_encloses_the_zonotope(const std::string &b) {
    cora::Rng rng(25);
    const int n = 3, m = 9;
    const Zonotope Z = Zonotope::generate_random(n, m, rng);
    const Tensor P = Z.dominant_directions();
    const Tensor R = P.inv().matmul(Z.generators()).abs_sum(); // radius of the box of P^-1 Z
    for (int k = 0; k < n; ++k) {
        bool found = false;
        for (int j = 0; j < m; ++j)
            found = found || close(P.slice(-1, k, k + 1), Z.generators().slice(-1, j, j + 1), 0);
        check(found, b + ": every column is a generator of Z");
    }
    for (int k = 0; k < 40; ++k) {
        const std::vector<double> d = test::random_direction(rng, n);
        const Tensor dt = column(d), Pd = P.matrix_transpose().matmul(dt).abs();
        const double box = Pd.matrix_transpose().matmul(R).item() +
                           dt.matrix_transpose().matmul(Z.center()).item();
        check(box >= test::support(Z, d) - 1e-9, b + ": P box(P^-1 Z) contains Z along a direction");
    }
}

/// A batch, too few generators, and a filter length of 0 are refused; with the checks off the
/// result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0, 0.1}, {0.0, 2.0, 0.1}}));
    check(test::raises([&] { Zonotope::stack({Z, Z}).dominant_directions(); }, Error::not_supported),
          "dominant_directions: a batch");
    check(test::raises([&] { Z.dominant_directions(0, 1); }, Error::wrong_value),
          "dominant_directions: filter length 0");
    const Zonotope thin(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {1.0, 0.0}}));
    check(test::raises([&] { thin.dominant_directions(); }, Error::degenerate_set),
          "dominant_directions: one nonzero generator in the plane");
    const Tensor on = Z.dominant_directions();
    ChecksScope off(false);
    check(test::close(Z.dominant_directions(), on, 0), "dominant_directions: same with checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        a_known_choice(b);
        the_parallelotope_encloses_the_zonotope(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope dominant_directions");
}
