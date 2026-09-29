// test_zonotope_interval - zonotope interval: the smallest box around a zonotope

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;
using test::column;
using test::random_direction;
using test::support;

namespace {

void hull_of_a_known_zonotope(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0, 0.5}, {0.0, 2.0, 0.5}}));
    check(close(Z.interval().inf, std::vector<double>{-0.5, -0.5}),
          b + ": the hull's lower corner");
    check(close(Z.interval().sup, std::vector<double>{2.5, 4.5}), b + ": the hull's upper corner");
}

void interval_is_the_hull(const std::string &b) {
    cora::Rng rng(6);
    const Zonotope Z = Zonotope::generateRandom(4, 6, rng);
    const Interval I = Z.interval();
    for (int i = 0; i < 4; ++i) {
        std::vector<double> e(4, 0.0), f(4, 0.0);
        e[i] = 1.0;
        f[i] = -1.0;
        check(close(I.supportFunc(column(e)).data()[0], support(Z, e), 1e-12), b + ": hull, upper");
        check(close(I.supportFunc(column(f)).data()[0], support(Z, f), 1e-12), b + ": hull, lower");
    }
    for (int k = 0; k < 10; ++k) {
        const std::vector<double> d = random_direction(rng, 4);
        check(I.supportFunc(column(d)).data()[0] >= support(Z, d) - 1e-10,
              b + ": the hull contains Z");
    }
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        hull_of_a_known_zonotope(b);
        interval_is_the_hull(b);
    });
    return test::finish("zonotope interval");
}
