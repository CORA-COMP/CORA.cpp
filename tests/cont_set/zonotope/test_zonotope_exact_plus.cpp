// test_zonotope_exact_plus - zonotope exact_plus: the sum that shares coefficients on the first
// generators
//
// CORA: contSet/@zonotope/exactPlus.m
// PROPERTY: contSet/@zonotope/exactPlus.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: the first min(m, m2, k) generators add, the rest of both follow.
void exact_plus_known(const std::string &b) {
    const Zonotope Z(column({0.0, 1.0}), Tensor({{1.0, 2.0}, {0.0, 1.0}}));
    const Zonotope Z2(column({1.0, 1.0}), Tensor({{1.0}, {1.0}}));
    const Zonotope S = Z.exact_plus(Z2);
    check(close(S.center(), std::vector<double>{1, 2}, 0), b + ": the centers add");
    check(close(S.generators(), std::vector<double>{2, 2, 1, 1}, 0),
          b + ": one pair added, the second generator of Z follows");
    check(close(Z.exact_plus(Z2, 0).generators(), std::vector<double>{1, 2, 1, 0, 1, 1}, 0),
          b + ": k = 0 adds no pair: all generators side by side");
    check(close(Z.exact_plus(Z2, 5).generators(), S.generators(), 0),
          b + ": k above m2 adds m2 pairs");
    check(close(Z2.exact_plus(Z).generators(), std::vector<double>{2, 2, 1, 1}, 0),
          b + ": the other way round, the same pair is added");
}

/// Stated property: the result lies in the plus, and equals it for k = 0; a batch is member-wise.
void exact_plus_is_inside_the_plus(const std::string &b) {
    cora::Rng rng(21);
    const Zonotope Z = Zonotope::generate_random(3, 5, rng),
                   Z2 = Zonotope::generate_random(3, 4, rng);
    for (const int64_t k : {0, 2, 4}) {
        const Zonotope S = Z.exact_plus(Z2, k), P = Z.plus(Z2);
        check(S.generators().shape().back() == 5 + 4 - std::min<int64_t>(4, k),
              b + ": m + m2 - min(m, m2, k) generators");
        for (int j = 0; j < 40; ++j) {
            const std::vector<double> d = test::random_direction(rng, 3);
            check(test::support(S, d) <= test::support(P, d) + 1e-9,
                  b + ": exact_plus leaves the plus");
        }
        if (k == 0) check(close(S.generators(), P.generators(), 0), b + ": k = 0 is the plus");
    }
    const Zonotope batch =
        Zonotope::stack({Z, Z + column({1.0, 1.0, 1.0})}).exact_plus(Zonotope::stack({Z2, Z2}));
    check(close(batch.generators().slice(0, 1, 2).squeeze(0), Z.exact_plus(Z2).generators(), 0),
          b + ": a batch, member 2");
}

/// A set of another dimension and a negative k are refused; with the checks off the result is the
/// same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {0.0}})),
        Z1(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { Z.exact_plus(Z1); }, Error::dimension_mismatch),
          "exact_plus: dimensions 2 and 1");
    check(test::raises([&] { Z.exact_plus(Z, -1); }, Error::wrong_value), "exact_plus: k = -1");
    const Tensor on = Z.exact_plus(Z).generators();
    ChecksScope off(false);
    check(close(Z.exact_plus(Z).generators(), on, 0), "exact_plus: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        exact_plus_known(b);
        exact_plus_is_inside_the_plus(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope exact_plus");
}
