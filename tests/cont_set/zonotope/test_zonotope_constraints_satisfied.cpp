// test_zonotope_constraints_satisfied - zonotope constraints_satisfied: C x < d for all of Z
//
// CORA: contSet/@zonotope/constrSat.m
// PROPERTY: contSet/@zonotope/constrSat.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// The box [-1,1]^2 against x1 + x2 < d: true above 2, false at and below it.
void strict_inequality_on_a_box(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Tensor C({{1.0, 1.0}});
    check(Z.constraints_satisfied(C, column({3.0})), b + ": x1 + x2 < 3");
    check(!Z.constraints_satisfied(C, column({2.0})),
          b + ": x1 + x2 < 2 is not certain (equality)");
    check(!Z.constraints_satisfied(C, column({1.9})), b + ": x1 + x2 < 1.9 fails");
    const Tensor C2({{1.0, 1.0}, {1.0, -1.0}});
    check(Z.constraints_satisfied(C2, column({2.5, 2.5})), b + ": two constraints");
    check(!Z.constraints_satisfied(C2, column({2.5, 1.0})), b + ": the second fails");
    const Zonotope batch = Zonotope::stack({Z, Z + column({2.0, 0.0})});
    check(!batch.constraints_satisfied(C, column({3.0})), b + ": a batch needs every member");
    check(batch.constraints_satisfied(C, column({5.0})), b + ": a batch, every member");
}

/// Stated property: "true" implies C x < d for every sampled point of the zonotope.
void true_is_sound(const std::string &b) {
    cora::Rng rng(17);
    int accepted = 0;
    for (int trial = 0; trial < 30; ++trial) {
        const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 4);
        std::vector<double> data(6), rhs(2);
        rng.normal(data.data(), data.size(), 1.0);
        rng.normal(rhs.data(), rhs.size(), 8.0);
        const Tensor C = Tensor::from_data(data, {2, 3}), d = column(rhs);
        if (!Z.constraints_satisfied(C, d)) continue;
        ++accepted;
        const Tensor values = C.matmul(Z.rand_point(100, rng)) - d;
        check(values.max().item() < 0, b + ": a certified constraint holds for all samples");
    }
    check(accepted > 3, b + ": the property was exercised");
}

/// Wrong shapes are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { Z.constraints_satisfied(Tensor({{1.0, 1.0, 1.0}}), column({1.0})); },
                       Error::dimension_mismatch),
          "constraints_satisfied: C with 3 columns");
    check(test::raises([&] { Z.constraints_satisfied(Tensor({{1.0, 1.0}}), column({1.0, 2.0})); },
                       Error::dimension_mismatch),
          "constraints_satisfied: d with 2 rows");
    const bool on = Z.constraints_satisfied(Tensor({{1.0, 1.0}}), column({3.0}));
    ChecksScope off(false);
    check(Z.constraints_satisfied(Tensor({{1.0, 1.0}}), column({3.0})) == on,
          "constraints_satisfied: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        strict_inequality_on_a_box(b);
        true_is_sound(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope constraints_satisfied");
}
