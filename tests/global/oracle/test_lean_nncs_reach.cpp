// test_lean_nncs_reach - NeurNetContrSys::reach on a nonlinear plant against the CORALean oracle
//
// ORACLE: contDynamics/@neurNetContrSys/reach.m
// ORACLE: contDynamics/@nonlinearSys/initReach.m
// ORACLE: contDynamics/@nonlinearSys/post.m
//
// The oracle's neurNetContrSys.reach, bit for bit: per control period the network's image appended
// to the state set (aux_appendU2X), the steps of alg lin with its error loop on the plant extended
// by the network's output (initReach, then post, the error handed on inside the period), each end
// set reduced, and the projection to the plant's states. Hand-made loops with the networks of
// sigmoid, tanh and relu at poly_method singh and regression (CORALean's bracket error), with an
// external input, with two steps per period and on NAV's plant, two periods each (a prefix of the
// horizon: the error hand-over between periods is what the second one checks). The instances'
// networks are not run: the oracle needs minutes for one. The oracle takes the horizon and the
// period as exact ratios of the doubles. The oracle's joint set drops the zero generators of the
// state set as CORA.cpp's evaluate_with_input does.
//
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/neur_net_contr_sys/neur_net_contr_sys.h"
#include "cora/global/oracle/neur_net_contr_sys.h"
#include "oracle_testing.h"

// the folder of the instance data, set before the benchmark header is read
#define CORACPP_BENCHMARK_DATA "."
#include "../../../examples/cpp/benchmarks/benchmark_neur_net.h"

#include <iostream>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A network of the sizes: a linear layer and then `act` after each but the last.
struct Net {
    std::vector<NeuralNetwork::Layer> layers;
    std::vector<Activation> activations;
    std::vector<lean::NnLayer> lean_layers;

    Net &add(const Tensor &W, const Tensor &b, Activation a) {
        layers.push_back({W, b});
        activations.push_back(a);
        lean_layers.push_back(lean::NnLayer::linear(W, b));
        if (a == Activation::sigmoid) lean_layers.push_back(lean::NnLayer::sigmoid());
        if (a == Activation::tanh) lean_layers.push_back(lean::NnLayer::tanh());
        if (a == Activation::relu) lean_layers.push_back(lean::NnLayer::relu());
        return *this;
    }
    NeuralNetwork network() const { return {layers, activations}; }

    /// The layers of a network as the oracle's: a linear layer, then its activation unless
    /// identity.
    static Net of(const NeuralNetwork &nn) {
        Net net;
        for (std::size_t k = 0; k < nn.layers().size(); ++k)
            net.add(nn.layers()[k].W, nn.layers()[k].b, nn.activations()[k]);
        return net;
    }
};

/// The 2-3-1 network of the pendulum loops with `act` hidden.
Net pendulum_net(Activation act) {
    Net net;
    net.add(Tensor({{1.0, -0.5}, {0.5, 1.0}, {-1.0, 0.25}}), Tensor({0.0, 0.25, -0.25}), act);
    net.add(Tensor({{-1.0, 0.5, 0.5}}), Tensor({0.0}), Activation::identity);
    return net;
}

/// The pendulum x1' = -sin(x0) - x1/8 + u0 (+ u1 for an external input).
NonlinearSys pendulum(bool external) {
    return NonlinearSys(
        [external](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            Expr f = -sin(x[0]) - 0.125 * x[1] + u[0];
            if (external) f = -x[0] + u[0] + u[1];
            return std::vector<Expr>{x[1], f};
        },
        2, external ? 2 : 1);
}

Zonotope pendulum_X0() {
    return Zonotope(Tensor({1.0, 0.5}), Tensor({{0.125, 0.0625, 0.0}, {0.0, 0.0625, 0.125}}));
}

/// The state sets at the ends of the control periods, CORA.cpp's and the oracle's.
void aux_compare(const std::string &name, const NonlinearSys &plant, const Net &net,
                 int64_t controls, const Zonotope &X0, const std::optional<Zonotope> &U, double dt,
                 double time_step, double t_final, bool regression = false, int taylor_terms = 4,
                 int order = 10) {
    ClosedLoopOptions options;
    options.plant = {.time_step = time_step,
                     .taylor_terms = taylor_terms,
                     .zonotope_order = order,
                     .tensor_order = 2,
                     .linear_step = alg::LinearStep::coralean,
                     .error_bound = alg::ErrorBound::loop};
    options.nn.poly_method = regression ? PolyMethod::regression : PolyMethod::singh;
    options.nn.regression_error = RegressionError::bracket;
    const NeurNetContrSys sys(plant, net.network(), dt);
    const ClosedLoopReach<Zonotope> mine = sys.reach(X0, U, t_final, options);
    const std::optional<std::vector<lean::Zonotope>> theirs =
        lean::NeurNetContrSys(plant, net.lean_layers, controls, dt)
            .reach(X0, U, t_final, options.plant, regression ? "regression" : "singh");
    test::check(theirs.has_value() && mine.completed, name + ": the run");
    if (!theirs || !mine.completed) return;
    // the sets at the ends of the periods are every (dt / time_step)-th of the steps
    const std::size_t steps = std::size_t(std::llround(dt / time_step));
    test::check(theirs->size() * steps + 1 == mine.time_point.size(), name + ": the periods");
    for (std::size_t p = 0; p < theirs->size(); ++p)
        oracle::check(name + " period " + std::to_string(p + 1), mine.time_point[(p + 1) * steps],
                      (*theirs)[p]);
}


} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // hand-made plants and networks, one control and two
    oracle::run("neurNetContrSys.reach", [&](oracle::Random &) {
        const NonlinearSys decay(
            [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
                return std::vector<Expr>{-x[0] + u[0]};
            },
            1, 1);
        Net net1;
        net1.add(Tensor({{1.0}, {-0.5}}), Tensor({0.0, 0.25}), Activation::tanh);
        net1.add(Tensor({{0.5, 0.25}}), Tensor({0.0}), Activation::identity);
        const Zonotope X1(Tensor({0.5}), Tensor({{0.125}}));
        aux_compare("decay, tanh", decay, net1, 1, X1, std::nullopt, 0.125, 0.125, 0.25);
        aux_compare("decay, two steps per period", decay, net1, 1, X1, std::nullopt, 0.125, 0.0625,
                    0.25);
        // the pendulum with each activation and both relaxations
        const Zonotope X2 = pendulum_X0();
        for (const bool regression : {false, true})
            aux_compare(std::string("pendulum, sigmoid") + (regression ? ", regression" : ""),
                        pendulum(false), pendulum_net(Activation::sigmoid), 1, X2, std::nullopt,
                        0.125, 0.125, 0.25, regression);
        aux_compare("pendulum, tanh", pendulum(false), pendulum_net(Activation::tanh), 1, X2,
                    std::nullopt, 0.125, 0.125, 0.25);
        aux_compare("pendulum, relu", pendulum(false), pendulum_net(Activation::relu), 1, X2,
                    std::nullopt, 0.125, 0.125, 0.25);
        aux_compare("pendulum, two steps per period", pendulum(false),
                    pendulum_net(Activation::sigmoid), 1, X2, std::nullopt, 0.25, 0.125, 0.5);
        // the plant of NAV (cos and sin of a state) with a hand-made network of two controls
        const NonlinearSys nav(
            [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
                return std::vector<Expr>{x[2] * cos(x[3]), x[2] * sin(x[3]), u[0] + 0.0 * x[0],
                                         u[1] + 0.0 * x[0]};
            },
            4, 2);
        Net net_nav;
        net_nav.add(
            Tensor({{0.5, -0.25, 0.125, 0.0}, {-0.5, 0.25, 0.0, 0.25}, {0.25, 0.5, -0.125, 0.5}}),
            Tensor({0.0, 0.125, -0.125}), Activation::tanh);
        net_nav.add(Tensor({{-0.5, 0.25, 0.5}, {0.25, -0.5, 0.125}}), Tensor({0.0, 0.0}),
                    Activation::identity);
        aux_compare("NAV plant, hand-made network, regression", nav, net_nav, 2,
                    cora::bench::box_zonotope({2.9, 2.9, 0, 0}, {3.1, 3.1, 0, 0}), std::nullopt,
                    0.25, 0.125, 0.5, true, 1, 10);
        // an external input, and the period of two steps
        aux_compare("pendulum, external input", pendulum(true), pendulum_net(Activation::sigmoid),
                    1, X2, Zonotope(Tensor({0.0}), Tensor({{0.0625}})), 0.125, 0.125, 0.25);
    });
    return test::finish("lean neurNetContrSys reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
