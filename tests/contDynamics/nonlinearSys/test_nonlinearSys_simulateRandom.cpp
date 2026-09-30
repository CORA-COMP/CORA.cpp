// test_nonlinearSys_simulateRandom - nonlinearSys simulateRandom: points of the set, repeatable

#include "contDynamics/nonlinearSys/nonlinearSys.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

NonlinearSys decay() {
    return NonlinearSys([](const std::vector<Expr> &x) { return std::vector<Expr>{-x[0], -2 * x[1]}; }, 2);
}

void the_start_points_lie_in_the_set_and_repeat(const std::string &b) {
    const Zonotope X0(Tensor({1.0, 2.0}), Tensor({{0.1, 0.0}, {0.0, 0.2}}));
    cora::Rng first(3), second(3);
    const std::vector<Tensor> a = decay().simulateRandom(X0, 6, 0.1, 0.5, first);
    const std::vector<Tensor> c = decay().simulateRandom(X0, 6, 0.1, 0.5, second);
    check(a.size() == 6 && a[0].shape() == std::vector<int64_t>({2, 6}), b + ": six columns");
    check(test::close(a.back(), c.back(), 0.0), b + ": the same seed gives the same points");
    const std::vector<double> start = a[0].data();
    bool inside = true;
    for (int j = 0; j < 6; ++j)
        inside = inside && std::abs(start[j] - 1.0) <= 0.1 && std::abs(start[6 + j] - 2.0) <= 0.2;
    check(inside, b + ": the start points are points of the set");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { the_start_points_lie_in_the_set_and_repeat(b); });
    return test::finish("nonlinearSys simulateRandom");
}
