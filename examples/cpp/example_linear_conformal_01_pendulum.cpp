// example_linear_conformal_01_pendulum - a linear model of a nonlinear system with a conformal guarantee
//
// Noisy trajectories of a damped pendulum start in a known set. The linearization at the origin is
// the model, and its reachable set misses trajectories where the pendulum is nonlinear. Split
// conformal prediction repairs that from held-out calibration trajectories: the score of a
// trajectory is how far it leaves the reachable boxes at its worst time, and enlarging the boxes by
// the (1 - alpha) quantile of the scores makes a new trajectory stay inside with probability at
// least 1 - alpha, whatever the error of the model is, as long as trajectories are exchangeable.
//
// Syntax:   build/examples/cpp/example_linear_conformal_01_pendulum
// Outputs:  the conformal radius, the coverage on test trajectories, and the figure as an SVG file
// See also: example_linear_learn_02_conformal of the Python examples (which learns the model)

#include "contDynamics/linearSys/linearSys.h"
#include "contDynamics/nonlinearSys/nonlinearSys.h"
#include "global/plot/plot.h"
#include "global/rng.h"

#include <algorithm>
#include <cmath>
#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora::ct;

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double timeStep = 0.1, tFinal = 3.0;
    const Zonotope X0(Tensor({1.2, 0.0}), Tensor({{0.2, 0.0}, {0.0, 0.2}}));
    const double alpha = 0.1;   // the miss rate: coverage of at least 1 - alpha is guaranteed
    const double noise = 0.01;  // standard deviation of the measurement noise

    // Ground Truth and Model ------------------------------------------------------------------

    // the damped pendulum: the angle and the angular velocity
    const NonlinearSys pendulum(
        [](const std::vector<Expr> &x) { return std::vector<Expr>{x[1], -sin(x[0]) - 0.2 * x[1]}; }, 2);

    // the model: the pendulum linearized at the origin
    const LinearSys model(Tensor({{0.0, 1.0}, {-1.0, -0.2}}));
    const Reach R = model.reach(X0, timeStep, tFinal, 6);

    // Conformal Prediction --------------------------------------------------------------------

    // noisy trajectories from random points of X0: time k holds the 2 x count points
    const auto measure = [&](int64_t count, uint64_t seed) {
        cora::Rng rng(seed);
        std::vector<Tensor> trajectories = pendulum.simulate(X0.randPoint(count, rng), timeStep, tFinal);
        std::vector<double> jitter(2 * count);
        for (Tensor &x : trajectories) {
            rng.normal(jitter.data(), jitter.size(), noise);
            x = x + Tensor::fromData(jitter, {2, count});
        }
        return trajectories;
    };

    // how far every trajectory leaves the reachable boxes at its worst time
    const auto scores = [&](const std::vector<Tensor> &trajectories) {
        const int64_t count = trajectories[0].shape()[1];
        std::vector<double> worst(count, 0.0);
        for (std::size_t k = 0; k < trajectories.size(); ++k) {
            const Interval box = R.timePoint[k].interval();
            const std::vector<double> lo = box.inf.data(), hi = box.sup.data(), x = trajectories[k].data();
            for (int64_t i = 0; i < 2; ++i)
                for (int64_t j = 0; j < count; ++j)
                    worst[j] = std::max({worst[j], lo[i] - x[i * count + j], x[i * count + j] - hi[i]});
        }
        return worst;
    };

    // the ceil((n + 1) (1 - alpha))-th smallest calibration score
    std::vector<double> calibration = scores(measure(200, 2));
    std::sort(calibration.begin(), calibration.end());
    const std::size_t rank = std::min<std::size_t>(
        calibration.size(), static_cast<std::size_t>(std::ceil((calibration.size() + 1) * (1 - alpha))));
    const double radius = calibration[rank - 1];

    // Evaluation ------------------------------------------------------------------------------

    const std::vector<double> test = scores(measure(1000, 3));
    const auto coverage = [&](double r) {
        return double(std::count_if(test.begin(), test.end(), [r](double s) { return s <= r; })) / test.size();
    };
    std::cout << "conformal radius: " << radius << "\n";
    std::cout << "coverage of the reachable set alone: " << coverage(0.0) << "\n";
    // the guarantee is on average over calibration sets: one set of 200 varies by about +-0.02
    std::cout << "coverage with the conformal radius: " << coverage(radius) << " (target "
              << 1 - alpha << ")\n";

    // Visualization ---------------------------------------------------------------------------

    useCORAcolors("CORA:contDynamics");
    cora::Rng rng(4);
    plot(R, {0, 1}, {.label = "reachable set of the linear model"});
    for (std::size_t k = 0; k < R.timePoint.size(); ++k) {
        const Interval box = R.timePoint[k].interval();
        const Tensor margin = Tensor({radius, radius});
        plot(Interval(box.inf - margin, box.sup + margin), {0, 1},
             {.label = k == 0 ? "enlarged by the conformal radius" : "", .color = "CORA:red", .filled = false});
    }
    plot(X0, {0, 1}, {.label = "initial set"});
    plot(pendulum.simulateRandom(X0, 30, timeStep, tFinal, rng), {0, 1}, {.label = "pendulum trajectories"});
    figure().title = "conformal prediction";
    figure().save("example_linear_conformal_01_pendulum.svg");
    std::cout << "figure written to example_linear_conformal_01_pendulum.svg\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
