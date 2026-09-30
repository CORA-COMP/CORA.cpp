// test_zonotope_reduce - zonotope reduce: fewer generators, and a superset of the original

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::random_direction;
using test::support;

namespace {

void reduce_limits_the_generators(const std::string &b) {
    cora::Rng rng(5);
    const int n = 3;
    const Zonotope Z = Zonotope::generateRandom(n, 20, rng);
    const Zonotope R = Z.reduce(2);
    check(R.G.shape() == std::vector<int64_t>({n, 2 * n}), b + ": order 2 leaves 2 n generators");
    check(test::close(R.c, Z.c), b + ": the center stays");
}

void reduce_encloses_the_zonotope(const std::string &b) {
    cora::Rng rng(6);
    const int n = 3;
    const Zonotope Z = Zonotope::generateRandom(n, 25, rng);
    const Zonotope R = Z.reduce(3);
    for (int k = 0; k < 40; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        check(support(R, d) >= support(Z, d) - 1e-9, b + ": the support along a direction only grows");
    }
}

void a_small_zonotope_stays_as_it_is(const std::string &b) {
    cora::Rng rng(7);
    const Zonotope Z = Zonotope::generateRandom(2, 4, rng);
    check(Z.reduce(2).G.shape() == Z.G.shape(), b + ": 4 generators fit into order 2 in 2-D");
    check(test::close(Z.reduce(2).G, Z.G), b + ": and are not changed");
}

void wrong_arguments_are_described(const std::string &b) {
    cora::Rng rng(8);
    const Zonotope Z = Zonotope::generateRandom(2, 4, rng);
    check(test::throws([&] { Z.reduce(0); }), b + ": order 0 is refused");
    const Zonotope batch = Zonotope::stack({Z, Z});
    check(test::throws([&] { batch.reduce(1); }) || b == "eigen", b + ": a batch is refused");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        reduce_limits_the_generators(b);
        reduce_encloses_the_zonotope(b);
        a_small_zonotope_stays_as_it_is(b);
        if (b != "eigen") wrong_arguments_are_described(b);
    });
    return test::finish("zonotope reduce");
}
