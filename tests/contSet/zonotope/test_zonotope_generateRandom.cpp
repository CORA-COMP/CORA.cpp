// test_zonotope_generateRandom - zonotope generateRandom

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

namespace {

void generateRandom_follows_cora(const std::string &b) {
    cora::Rng rng(7);
    const Zonotope Z = Zonotope::generateRandom(5, 8, rng);
    check(Z.c.shape() == std::vector<int64_t>({5, 1}) &&
              Z.G.shape() == std::vector<int64_t>({5, 8}),
          b + ": shapes");
    const std::vector<double> G = Z.G.data();
    bool short_enough = true;
    for (int j = 0; j < 8; ++j) {
        double norm = 0.0;
        for (int i = 0; i < 5; ++i) norm += G[i * 8 + j] * G[i * 8 + j];
        short_enough &= std::sqrt(norm) <= 1.0 + 1e-12;
    }
    check(short_enough, b + ": generators are at most unit length");
    check(!close(Zonotope::generateRandom(5, 8, rng).c, Z.c), b + ": successive sets differ");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { generateRandom_follows_cora(b); });
    return test::finish("zonotope generateRandom");
}
