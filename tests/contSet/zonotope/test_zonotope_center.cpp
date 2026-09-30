// test_zonotope_center - zonotope center and dim

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void dim_and_center(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0, 0.5}, {0.0, 2.0, 0.5}}));
    check(Z.dim() == 2, b + ": dim");
    check(close(Z.center(), std::vector<double>{1, 2}), b + ": center");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { dim_and_center(b); });
    return test::finish("zonotope center");
}
