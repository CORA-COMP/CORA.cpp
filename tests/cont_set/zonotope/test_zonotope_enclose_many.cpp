// test_zonotope_enclose_many - zonotope enclose_many: an oriented box around many zonotopes
//
// CORA: contSet/@zonotope/encloseMany.m
// PROPERTY: contSet/@zonotope/encloseMany.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
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

/// Hand-computed: two unit squares along the x-axis in the standard frame, and the frame of the
/// diagonal direction (1, 1).
void enclose_many_known(const std::string &b) {
    const Zonotope A(column({0.5, 0.5}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const Zonotope B(column({2.5, 0.5}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const auto [E, inverse] = Zonotope::enclose_many({A, B}, column({1.0, 0.0}));
    check(close(E.center(), std::vector<double>{1.5, 0.5}, 1e-14),
          b + ": the box [0, 3] x [0, 1]: center");
    check(close(E.generators(), std::vector<double>{1.5, 0, 0, 0.5}, 1e-14),
          b + ": the box: generators");
    check(close(inverse, std::vector<double>{1, 0, 0, 1}, 0), b + ": the frame is the identity");

    const auto [D, rot_inv] = Zonotope::enclose_many({A, B}, column({3.0, 3.0}));
    const double a = std::sqrt(0.5);
    // the frame [d, e_0] with d = (1, 1) / sqrt(2): its inverse is [[0, 1 / a], [1, -1]]
    check(close(rot_inv, std::vector<double>{0, 1 / a, 1, -1}, 1e-12),
          b + ": the inverse of the frame of (1, 1)");
    for (const Zonotope &S : {A, B}) {
        const Interval box = S.interval();
        check(D.contains(box, "exact", 1e-9), b + ": the oriented box contains the square");
    }
}

/// Stated property: the box contains every zonotope, is a box in the frame (its generators are
/// the frame's columns), and the returned inverse inverts the frame.
void enclose_many_encloses(const std::string &b) {
    cora::Rng rng(44);
    std::vector<Zonotope> Zs;
    Zs.reserve(4);
    for (int k = 0; k < 4; ++k)
        Zs.push_back(Zonotope::generate_random(rng, 3, std::nullopt, 5) +
                     column({1.0 * k, -2.0 * k, 0.5}));
    const Tensor direction = column({1.0, -2.0, 0.5});
    const auto [E, inverse] = Zonotope::enclose_many(Zs, direction);
    for (const Zonotope &Z : Zs) {
        for (int j = 0; j < 40; ++j) {
            const std::vector<double> d = test::random_direction(rng, 3);
            check(test::support(E, d) >= test::support(Z, d) - 1e-9,
                  b + ": the box leaves a zonotope");
        }
    }
    // in the frame the box is axis-aligned: inverse * G is diagonal up to the zero columns
    const Tensor G = inverse.matmul(E.generators());
    const std::vector<double> g = G.data();
    const int64_t m = G.shape()[1];
    for (int64_t i = 0; i < 3; ++i)
        for (int64_t j = 0; j < m; ++j)
            if (i != j)
                check(std::abs(g[i * m + j]) < 1e-12, b + ": the box is axis-aligned in the frame");
    check(
        close(inverse.matmul(inverse.inv()), std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1}, 1e-12),
        b + ": the inverse is invertible");
}

/// Wrong inputs are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    check(test::raises([&] { Zonotope::enclose_many({}, column({1.0, 0.0})); }, Error::wrong_value),
          "enclose_many: none");
    check(
        test::raises([&] { Zonotope::enclose_many({Z}, column({0.0, 0.0})); }, Error::wrong_value),
        "enclose_many: zero direction");
    check(test::raises([&] { Zonotope::enclose_many({Z}, column({1.0, 0.0, 0.0})); },
                       Error::dimension_mismatch),
          "enclose_many: direction");
    check(test::raises(
              [&] {
                  Zonotope::enclose_many({Z, Zonotope(column({0.0}), Tensor({{1.0}}))},
                                         column({1.0, 0.0}));
              },
              Error::dimension_mismatch),
          "enclose_many: dimensions");
    check(
        test::raises([&] { Zonotope::enclose_many({Zonotope::stack({Z, Z})}, column({1.0, 0.0})); },
                     Error::dimension_mismatch),
        "enclose_many: a batch");
    const Tensor on = Zonotope::enclose_many({Z}, column({1.0, 0.0})).first.generators();
    ChecksScope off(false);
    check(close(Zonotope::enclose_many({Z}, column({1.0, 0.0})).first.generators(), on, 0),
          "enclose_many: checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        enclose_many_known(b);
        enclose_many_encloses(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope enclose_many");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
