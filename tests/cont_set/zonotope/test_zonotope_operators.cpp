// test_zonotope_operators - the operators of a zonotope, as CORA writes them
//
// CORA: contSet/@zonotope/mtimes.m
// CORA: contSet/@zonotope/plus.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

Zonotope box() { return Zonotope(Tensor({1.0, 2.0}), Tensor({{0.5, 0.0}, {0.0, 0.25}})); }

void a_matrix_maps_the_zonotope(const std::string &b) {
    const Tensor M({{0.0, 1.0}, {-1.0, 0.0}});
    const Zonotope Z = M * box();
    check(close(Z.center(), box().linear_map(M).center()) &&
              close(Z.generators(), box().linear_map(M).generators()),
          b + ": M * Z is linear_map");
    const Interval I =
        Interval::matrix(Tensor({{-1.0, -1.0}, {-1.0, -1.0}}), Tensor({{1.0, 1.0}, {1.0, 1.0}}));
    check((I * box()).nr_generators() == 4,
          b + ": an interval matrix widens by one generator each");
}

void numbers_scale(const std::string &b) {
    const Zonotope Z = -2.0 * box();
    check(close(Z.center(), test::vec({-2.0, -4.0})) &&
              close(Z.generators(), Tensor({{-1.0, 0.0}, {0.0, -0.5}})),
          b + ": s * Z");
    check(close((box() * 2.0).center(), test::vec({2.0, 4.0})), b + ": Z * s");
    // unary minus negates center and generators
    check(close((-box()).center(), test::vec({-1.0, -2.0})) &&
              close((-box()).generators(), Tensor({{-0.5, 0.0}, {0.0, -0.25}})),
          b + ": -Z negates centre and generators (CORA's uminus)");
}

void sums_and_translations(const std::string &b) {
    const Zonotope S = box() + box();
    check(close(S.center(), test::vec({2.0, 4.0})) && S.nr_generators() == 4,
          b + ": Z + Z2 is the Minkowski sum");
    const Tensor v({1.0, -1.0});
    check(close((box() + v).center(), test::vec({2.0, 1.0})) &&
              close((v + box()).center(), test::vec({2.0, 1.0})),
          b + ": Z + v translates");
    check(close((box() - v).center(), test::vec({0.0, 3.0})), b + ": Z - v");
    check(close((box() + v).generators(), box().generators()),
          b + ": translation keeps the generators");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_matrix_maps_the_zonotope(b);
        numbers_scale(b);
        sums_and_translations(b);
    });
    return test::finish("zonotope operators");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
