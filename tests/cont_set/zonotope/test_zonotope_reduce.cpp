// test_zonotope_reduce - zonotope reduce: fewer generators, and a superset of the original
//
// CORA: contSet/@zonotope/reduce.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::random_direction;
using test::support;

namespace {

void reduce_limits_the_generators(const std::string &b) {
    cora::Rng rng(5);
    const int n = 3;
    const Zonotope Z = Zonotope::generate_random(n, 20, rng);
    const Zonotope R = Z.reduce(2);
    check(R.generators().shape() == std::vector<int64_t>({n, 2 * n}),
          b + ": order 2 leaves 2 n generators");
    check(test::close(R.center(), Z.center()), b + ": the center stays");
}

void reduce_encloses_the_zonotope(const std::string &b) {
    cora::Rng rng(6);
    const int n = 3;
    const Zonotope Z = Zonotope::generate_random(n, 25, rng);
    const Zonotope R = Z.reduce(3);
    for (int k = 0; k < 40; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        check(support(R, d) >= support(Z, d) - 1e-9, b + ": the support along a direction only grows");
    }
}

void a_small_zonotope_stays_as_it_is(const std::string &b) {
    cora::Rng rng(7);
    const Zonotope Z = Zonotope::generate_random(2, 4, rng);
    check(Z.reduce(2).generators().shape() == Z.generators().shape(),
          b + ": 4 generators fit into order 2 in 2-D");
    check(test::close(Z.reduce(2).generators(), Z.generators()), b + ": and are not changed");
}

void wrong_arguments_are_described(const std::string &b) {
    cora::Rng rng(8);
    const Zonotope Z = Zonotope::generate_random(2, 4, rng);
    check(test::throws([&] { Z.reduce(0); }), b + ": order 0 is refused");
}

void a_batch_reduces_member_by_member(const std::string &b) {
    cora::Rng rng(9);
    const Zonotope Z1 = Zonotope::generate_random(3, 9, rng);
    const Zonotope Z2 = Zonotope::generate_random(3, 9, rng);
    const Zonotope batch = Zonotope::stack({Z1, Z2}).reduce(2);
    const Tensor G = batch.generators();
    check(test::close(G.slice(0, 0, 1).squeeze(0), Z1.reduce(2).generators(), 0), b + ": member 1");
    check(test::close(G.slice(0, 1, 2).squeeze(0), Z2.reduce(2).generators(), 0), b + ": member 2");
}

/// An order below one is refused; with the checks off the reduction is the same.
void checks_and_their_switch() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    check(test::raises([&] { Z.reduce(0); }, Error::wrong_value), "reduce: order 0");
    const Tensor on = Z.reduce(1).generators();
    ChecksScope off(false);
    check(test::close(Z.reduce(1).generators(), on, 0), "reduce: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        reduce_limits_the_generators(b);
        reduce_encloses_the_zonotope(b);
        a_small_zonotope_stays_as_it_is(b);
        wrong_arguments_are_described(b);
        a_batch_reduces_member_by_member(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope reduce");
}
