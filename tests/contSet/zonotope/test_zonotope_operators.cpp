// test_zonotope_operators - the operators of a zonotope, as CORA writes them

#include "contSet/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

Zonotope box() { return Zonotope(Tensor({1.0, 2.0}), Tensor({{0.5, 0.0}, {0.0, 0.25}})); }

void a_matrix_maps_the_zonotope(const std::string &b) {
    const Tensor M({{0.0, 1.0}, {-1.0, 0.0}});
    const Zonotope Z = M * box();
    check(close(Z.c, box().mtimes(M).c) && close(Z.G, box().mtimes(M).G), b + ": M * Z is mtimes");
    const Interval I(Tensor({{-1.0, -1.0}, {-1.0, -1.0}}), Tensor({{1.0, 1.0}, {1.0, 1.0}}));
    check((I * box()).G.shape()[1] == 4, b + ": an interval matrix widens by one generator each");
}

void numbers_scale(const std::string &b) {
    const Zonotope Z = -2.0 * box();
    check(close(Z.c, Tensor({-2.0, -4.0})) && close(Z.G, Tensor({{-1.0, 0.0}, {0.0, -0.5}})),
          b + ": s * Z");
    check(close((box() * 2.0).c, Tensor({2.0, 4.0})), b + ": Z * s");
    check(close((-box()).c, Tensor({-1.0, -2.0})), b + ": -Z");
}

void sums_and_translations(const std::string &b) {
    const Zonotope S = box() + box();
    check(close(S.c, Tensor({2.0, 4.0})) && S.G.shape()[1] == 4, b + ": Z + Z2 is the Minkowski sum");
    const Tensor v({1.0, -1.0});
    check(close((box() + v).c, Tensor({2.0, 1.0})) && close((v + box()).c, Tensor({2.0, 1.0})),
          b + ": Z + v translates");
    check(close((box() - v).c, Tensor({0.0, 3.0})), b + ": Z - v");
    check(close((box() + v).G, box().G), b + ": translation keeps the generators");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        a_matrix_maps_the_zonotope(b);
        numbers_scale(b);
        sums_and_translations(b);
    });
    return test::finish("zonotope operators");
}
