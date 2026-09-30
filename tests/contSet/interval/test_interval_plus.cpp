// test_interval_plus - interval plus: the Minkowski sum

#include "contSet/interval/interval.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void plus_is_the_minkowski_sum(const std::string &b) {
    const Interval A(column({0.0, 1.0}), column({1.0, 3.0})),
        B(column({-2.0, 0.5}), column({0.0, 0.5}));
    const Interval S = A.plus(B);
    check(close(S.inf, std::vector<double>{-2.0, 1.5}) &&
              close(S.sup, std::vector<double>{1.0, 3.5}),
          b + ": bounds add");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { plus_is_the_minkowski_sum(b); });
    return test::finish("interval plus");
}
