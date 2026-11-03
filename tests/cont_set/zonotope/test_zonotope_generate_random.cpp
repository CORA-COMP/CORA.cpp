// test_zonotope_generate_random - zonotope generate_random
//
// CORA: contSet/@zonotope/generateRandom.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

namespace {

void generate_random_follows_cora(const std::string &b) {
    cora::Rng rng(7);
    const Zonotope Z = Zonotope::generate_random(rng, 5, std::nullopt, 8);
    check(Z.center().shape() == std::vector<int64_t>({5, 1}) &&
              Z.generators().shape() == std::vector<int64_t>({5, 8}),
          b + ": shapes");
    const std::vector<double> G = Z.generators().data();
    bool short_enough = true;
    for (int j = 0; j < 8; ++j) {
        double norm = 0.0;
        for (int i = 0; i < 5; ++i) norm += G[i * 8 + j] * G[i * 8 + j];
        short_enough &= std::sqrt(norm) <= 1.0 + 1e-12;
    }
    check(short_enough, b + ": generators are at most unit length");
    check(!close(Zonotope::generate_random(rng, 5, std::nullopt, 8).center(), Z.center()),
          b + ": successive sets differ");
}

/// A dimension below one is refused; with the checks off the draw is the same.
void checks_and_their_switch() {
    cora::Rng rng(4), again(4);
    check(test::raises([&] { Zonotope::generate_random(rng, 0, std::nullopt, 2); }, Error::wrong_value),
          "generate_random: dimension 0");
    const Tensor on = Zonotope::generate_random(rng, 3, std::nullopt, 2).center();
    ChecksScope off(false);
    check(test::close(Zonotope::generate_random(again, 3, std::nullopt, 2).center(), on, 0),
          "generate_random: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { generate_random_follows_cora(b); });
    checks_and_their_switch();
    return test::finish("zonotope generate_random");
}
