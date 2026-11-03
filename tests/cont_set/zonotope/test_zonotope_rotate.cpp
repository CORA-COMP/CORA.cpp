// test_zonotope_rotate - zonotope rotate: the zonotope turned in the plane of two states
//
// CORA: contSet/@zonotope/rotate.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

constexpr double kPi = 3.14159265358979323846;

/// A quarter turn in the plane of states 0 and 1 takes (1, 0) to (0, 1) and leaves state 2; angle
/// 0 is the identity; the rotation is the matrix [cos -sin; sin cos].
void rotation_known(const std::string &b) {
    const Zonotope Z(column({1.0, 0.0, 5.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}, {2.0, 0.0}}));
    const Zonotope R = Z.rotate({0, 1}, kPi / 2);
    check(close(R.center(), std::vector<double>{0, 1, 5}, 1e-15), b + ": the center");
    check(close(R.generators(), std::vector<double>{0, -2, 1, 0, 2, 0}, 1e-15),
          b + ": the generators");
    check(close(Z.rotate({0, 1}, 0.0).generators(), Z.generators(), 0), b + ": angle 0");
    const double a = 0.3;
    const Zonotope S = Z.rotate({2, 0}, a); // the plane (state 2, state 0), in that order
    const std::vector<double> expected{std::sin(a) * 5 + std::cos(a), 0,
                                       std::cos(a) * 5 - std::sin(a)};
    check(close(S.center(), expected, 1e-14), b + ": the order of dims is the order of the axes");
    const Zonotope batch =
        Zonotope::stack({Z, Z + column({1.0, 1.0, 1.0})}).rotate({0, 1}, kPi / 2);
    check(close(batch.center().slice(0, 0, 1).squeeze(0), R.center(), 1e-15), b + ": a batch");
}

/// Stated property: the support function along d is that of Z along R' d, and a rotation by -a
/// undoes a rotation by a.
void rotation_is_orthogonal(const std::string &b) {
    cora::Rng rng(81);
    const Zonotope Z = Zonotope::generate_random(rng, 4, std::nullopt, 5);
    const double a = 1.1;
    const Zonotope R = Z.rotate({1, 3}, a);
    for (int k = 0; k < 20; ++k) {
        std::vector<double> d = test::random_direction(rng, 4), back = d;
        back[1] = std::cos(a) * d[1] + std::sin(a) * d[3];
        back[3] = -std::sin(a) * d[1] + std::cos(a) * d[3];
        check(close(test::support(R, d), test::support(Z, back), 1e-12),
              b + ": the support function");
    }
    const Zonotope undone = R.rotate({1, 3}, -a);
    check(close(undone.center(), Z.center(), 1e-12) &&
              close(undone.generators(), Z.generators(), 1e-12),
          b + ": the inverse rotation");
}

/// Wrong dims and a NaN angle are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({1.0, 0.0}), Tensor({{1.0}, {0.0}}));
    check(test::raises([&] { Z.rotate({0}, 1.0); }, Error::wrong_value), "rotate: one state");
    check(test::raises([&] { Z.rotate({0, 0}, 1.0); }, Error::wrong_value),
          "rotate: the same state");
    check(test::raises([&] { Z.rotate({0, 2}, 1.0); }, Error::wrong_value), "rotate: out of range");
    check(test::raises([&] { Z.rotate({-1, 1}, 1.0); }, Error::wrong_value), "rotate: negative");
    check(test::raises([&] { Z.rotate({0, 1}, std::nan("")); }, Error::wrong_value), "rotate: NaN");
    ChecksScope off(false);
    check(close(Z.rotate({0, 1}, kPi).center(), std::vector<double>{-1, 0}, 1e-15),
          "rotate: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        rotation_known(b);
        rotation_is_orthogonal(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope rotate");
}
