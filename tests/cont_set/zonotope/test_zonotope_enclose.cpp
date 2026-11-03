// test_zonotope_enclose - zonotope enclose: a zonotope that contains two zonotopes
//
// CORA: contSet/@zonotope/enclose.m
// PROPERTY: contSet/@zonotope/enclose.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: the shorter zonotope's generators are combined with the leading ones of the other.
void enclose_known_zonotopes(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Zonotope Z2(column({2.0, 0.0}), Tensor({{1.0}, {0.0}}));
    const Zonotope E = Z.enclose(Z2);
    check(close(E.center(), std::vector<double>{1, 0}, 0), b + ": center (c+c2)/2");
    check(close(E.generators(), std::vector<double>{1, -1, 0, 0, 0, 0, 0, 1}, 0),
          b + ": [(G+G2)/2, (c-c2)/2, (G-G2)/2, rest]");
    const Zonotope F = Z2.enclose(Z);
    check(close(F.center(), std::vector<double>{1, 0}, 0), b + ": the other way: center");
    check(close(F.generators(), std::vector<double>{1, -1, 0, 0, 0, 0, 0, 1}, 0),
          b + ": the other way: (c2-c)/2 in the middle");
}

/// Stated property: the result contains both zonotopes, also for a batch.
void enclose_contains_both(const std::string &b) {
    cora::Rng rng(18);
    const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 5), Z2 = Zonotope::generate_random(rng, 3, std::nullopt, 3);
    for (const Zonotope &E : {Z.enclose(Z2), Z2.enclose(Z)}) {
        check(E.generators().shape().back() == 2 * 3 + 1 + 2, b + ": 2m + 1 + extras generators");
        for (int k = 0; k < 40; ++k) {
            const std::vector<double> d = test::random_direction(rng, 3);
            check(test::support(E, d) >= std::max(test::support(Z, d), test::support(Z2, d)) - 1e-9,
                  b + ": enclose leaves one of its sets");
        }
    }
    const Zonotope batch =
        Zonotope::stack({Z, Z + column({1.0, 1.0, 1.0})}).enclose(Zonotope::stack({Z2, Z2}));
    check(close(batch.center().slice(0, 0, 1).squeeze(0), Z.enclose(Z2).center(), 0),
          b + ": a batch, member 1");
}

/// A set of another dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {0.0}})), Z1(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { Z.enclose(Z1); }, Error::dimension_mismatch), "enclose: dimensions 2 and 1");
    const Tensor on = Z.enclose(Z).generators();
    ChecksScope off(false);
    check(test::close(Z.enclose(Z).generators(), on, 0), "enclose: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        enclose_known_zonotopes(b);
        enclose_contains_both(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope enclose");
}
