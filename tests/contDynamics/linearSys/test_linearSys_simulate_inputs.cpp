// test_linearSys_simulate_inputs - linearSys simulate and simulateRandom with constant inputs:
// exact against the closed form, one input per trajectory, and the errors

#include "contDynamics/linearSys/linearSysTesting.h"
#include "global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::throws;
using namespace test::lin;

namespace {

/// x(t) = e^{At} x0 + A^{-1} (e^{At} - I) B u for an invertible A.
void follows_the_closed_form(const std::string &b) {
    const Eigen::MatrixXd A = oscillator3();
    Eigen::MatrixXd B(3, 2), x0(3, 4), u(2, 4);
    B << 1, 0, 0.5, -1, 0, 0.3;
    x0 << 1, 0, -1, 0.5, 0, 1, 0.2, -0.5, 0.3, -0.2, 1, 0;
    u << 0.4, -1, 0, 2, -0.2, 0.5, 1, 0;
    const double dt = 0.15;
    const std::vector<Tensor> x =
        LinearSys(tensor_of(A), tensor_of(B)).simulate(tensor_of(x0), tensor_of(u), dt, 1.0);

    check(x.size() == 8, b + ": ceil(1.0 / 0.15) + 1 time points");
    check(close(x[0], tensor_of(x0)), b + ": the start is the start");
    double worst = 0.0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const Eigen::MatrixXd E = (A * dt * double(k)).exp();
        const Eigen::MatrixXd exact = E * x0 + A.inverse() * (E - Eigen::MatrixXd::Identity(3, 3)) * B * u;
        worst = std::max(worst, (host_of(x[k]) - exact).cwiseAbs().maxCoeff());
    }
    check(worst < 1e-12, b + ": a trajectory differs from the closed form by " + std::to_string(worst));
}

/// x' = -x + u from 0 with u = 1 is 1 - e^{-t}; a single input column serves every trajectory.
void one_input_serves_all_trajectories(const std::string &b) {
    const LinearSys sys(Tensor({{-1.0}}), Tensor({{1.0}}));
    const std::vector<Tensor> x = sys.simulate(Tensor({{0.0, 0.0, 0.0}}), Tensor({{1.0}}), 0.5, 2.0);
    for (std::size_t k = 0; k < x.size(); ++k)
        check(close(x[k], std::vector<double>(3, 1 - std::exp(-0.5 * double(k))), 1e-12),
              b + ": 1 - e^{-t} at step " + std::to_string(k));
}

/// simulateRandom draws the start points and then the inputs from the same generator.
void random_inputs_come_from_the_set(const std::string &b) {
    const Tensor A = tensor_of(oscillator3());
    const Tensor B = Tensor::fromData({1, 0, 0.5, -1, 0, 0.3}, {3, 2});
    const LinearSys sys(A, B);
    const Zonotope X0(test::column({1, 0, 0}), 0.1 * Tensor::eye(3));
    const Zonotope U(test::column({0.5, -0.5}), 0.2 * Tensor::eye(2));

    cora::Rng rng(3), same(3);
    const std::vector<Tensor> x = sys.simulateRandom(X0, U, 7, 0.1, 0.5, rng);
    const Tensor x0 = X0.randPoint(7, same);
    const std::vector<Tensor> y = sys.simulate(x0, U.randPoint(7, same), 0.1, 0.5);
    check(x.size() == y.size() && close(x.back(), y.back(), 1e-14), b + ": simulateRandom with inputs");
    check(x[0].shape() == std::vector<int64_t>({3, 7}), b + ": one trajectory per point");
}

void reports_errors(const std::string &b) {
    const Tensor A = tensor_of(oscillator3());
    const Tensor B = Tensor::fromData({1, 0, 0.5, -1, 0, 0.3}, {3, 2});
    const Tensor x0 = Tensor::zeros({3, 4});
    check(throws([&] { LinearSys(A).simulate(x0, Tensor::zeros({2, 4}), 0.1, 0.5); }),
          b + ": inputs without B are refused");
    check(throws([&] { LinearSys(A, B).simulate(x0, Tensor::zeros({2, 3}), 0.1, 0.5); }),
          b + ": one input per trajectory, or one for all");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        follows_the_closed_form(b);
        one_input_serves_all_trajectories(b);
        random_inputs_come_from_the_set(b);
        reports_errors(b);
    });
    return test::finish("linearSys simulate (inputs)");
}
