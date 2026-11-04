// test_zonotope_is_bounded - zonotope is_bounded: always true
//
// CORA: contSet/@zonotope/isBounded.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

int main() {
    test::for_each_backend([](const std::string &b) {
        cora::Rng rng(1);
        check(Zonotope::generate_random(rng, 3, std::nullopt, 5).is_bounded(),
              b + ": a random zonotope");
        check(Zonotope(column({1.0, 2.0}), Tensor::zeros({2, 0})).is_bounded(), b + ": a point");
        check(Zonotope::stack({Zonotope::generate_random(rng, 2, std::nullopt, 3)}).is_bounded(),
              b + ": a batch");
    });
    const Zonotope Z(column({0.0}), Tensor({{1.0}}));
    ChecksScope off(false);
    check(Z.is_bounded(), "is_bounded: checks off");
    return test::finish("zonotope is_bounded");
}
