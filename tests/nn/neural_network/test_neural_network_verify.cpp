// test_neural_network_verify - the verdicts of verify, its options, strategies and sets
//
// Small ReLU, sigmoid and tanh networks whose properties are known: an unsafe set far below the
// outputs is "unsat" for both boundings, one around a sampled output is "sat" with a counterexample
// whose output lies in it. The batched zonotope with every slot is the output zonotope of
// NeuralNetwork::evaluate, and with few slots (an interval center) it still holds sampled outputs;
// Fourier-Motzkin in both orders holds the feasible points of random systems; the three repairs and
// every switch of CORA's verify run. exact_backprop's F-radius gradient is the difference quotient.
//
// CORA: nn/@neuralNetwork/verify.m
// CORA: nn/@neuralNetwork/evaluateZonotopeBatch_.m
// CORA: nn/@neuralNetwork/prepareForZonoBatchEval.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateZonotopeBatch.m
// CORA: nn/@neuralNetwork/computePGDAttack.m
// CORA: nn/@neuralNetwork/computeFGSMAttack.m
// PROPERTY: nn/@neuralNetwork/evaluateZonotopeBatch_.m
// CORA authors: Lukas Koller, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/rng.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/nn/neural_network/neural_network.h"
#include "cora/nn/neural_network/verify.h"
#include "nn/neural_network/private/priv_zono_batch.h"
#include "testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A random network of the sizes with the activation after each hidden layer.
NeuralNetwork aux_network(Rng &rng, const std::vector<int64_t> &sizes, Activation hidden) {
    std::vector<NeuralNetwork::Layer> layers;
    std::vector<Activation> acts;
    for (std::size_t k = 0; k + 1 < sizes.size(); ++k) {
        std::vector<double> W(static_cast<std::size_t>(sizes[k + 1] * sizes[k])),
            b(static_cast<std::size_t>(sizes[k + 1]));
        rng.normal(W.data(), W.size(), 1.0 / std::sqrt(double(sizes[k])));
        rng.normal(b.data(), b.size(), 0.3);
        layers.push_back(
            {Tensor::from_data(W, {sizes[k + 1], sizes[k]}), Tensor::from_data(b, {sizes[k + 1]})});
        acts.push_back(k + 2 < sizes.size() ? hidden : Activation::identity);
    }
    return {layers, acts};
}

/// The box [lo, hi]^n with the unsafe set y_0 <= t.
NnProperty aux_below(int64_t n, int64_t m, double lo, double hi, double t) {
    NnProperty p;
    p.lo = Tensor::from_data(std::vector<double>(std::size_t(n), lo), {1, n});
    p.hi = Tensor::from_data(std::vector<double>(std::size_t(n), hi), {1, n});
    std::vector<double> A(std::size_t(m), 0.0);
    A[0] = 1.0;
    p.A = Tensor::from_data(A, {1, 1, m});
    p.b = Tensor::from_data(std::vector<double>{t}, {1, 1});
    p.poly_case = {0};
    return p;
}

/// The least sampled y_0 of the network over the box.
double aux_least_output(const NeuralNetwork &nn, Rng &rng, int64_t n, double lo, double hi) {
    std::vector<double> x(std::size_t(4000 * n));
    rng.uniform(x.data(), x.size(), lo, hi);
    const std::vector<double> y = nn.evaluate(Tensor::from_data(x, {4000, n})).data();
    double least = std::numeric_limits<double>::infinity();
    const std::size_t m = y.size() / 4000;
    for (std::size_t i = 0; i < 4000; ++i) least = std::min(least, y[i * m]);
    return least;
}

/// The options of a small run: the bounding, dtype float64 on lean (which has no float32).
NnVerifyOptions aux_options(const std::string &bounding, const std::string &b) {
    NnVerifyOptions o = bounding == "conzonotope" ? vnncomp_options("test") : NnVerifyOptions{};
    if (b == "lean") o.dtype = DType::float64;
    o.pgd_restarts = 8;
    o.pgd_steps = 20;
    o.mini_batch_size = 16;
    return o;
}

/// unsat far below the outputs, sat around the least sampled output, for both boundings.
void verdicts(const std::string &b) {
    Rng rng(3);
    for (const Activation a : {Activation::relu, Activation::sigmoid, Activation::tanh}) {
        const NeuralNetwork nn = aux_network(rng, {3, 8, 8, 2}, a);
        const double least = aux_least_output(nn, rng, 3, -0.2, 0.2);
        for (const std::string bounding : {"conzonotope", "zonotope"}) {
            NnVerifyOptions o = aux_options(bounding, b);
            o.falsification_order = "bound-first";
            const NnVerifyResult safe = nn.verify(aux_below(3, 2, -0.2, 0.2, least - 1.0), 20, o);
            check(safe.status == "unsat", b + " " + bounding + ": far below the outputs is unsat");
            const NnVerifyResult unsafe =
                nn.verify(aux_below(3, 2, -0.2, 0.2, least + 0.01), 20, o);
            check(unsafe.status == "sat" && unsafe.y.size() == 2 && unsafe.y[0] <= least + 0.01,
                  b + " " + bounding + ": around the least output is sat, with its output");
        }
    }
    // every attack finds an unsafe set that holds most of the box
    const NeuralNetwork nn = aux_network(rng, {2, 6, 2}, Activation::relu);
    for (const std::string attack : {"pgd", "fgsm", "center", "zonotack"}) {
        NnVerifyOptions o = aux_options("conzonotope", b);
        o.falsification_method = attack;
        const double t = aux_least_output(nn, rng, 2, -0.5, 0.5) + 10.0;
        check(nn.verify(aux_below(2, 2, -0.5, 0.5, t), 20, o).status == "sat",
              b + ": the attack " + attack + " finds a counterexample");
    }
}

/// The batched zonotope: every slot is NeuralNetwork::evaluate's set; few slots hold the outputs.
void batched_zonotopes(const std::string &b) {
    Rng rng(5);
    const NeuralNetwork nn = aux_network(rng, {3, 6, 5, 2}, Activation::relu);
    const ZonoNet full(nn, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity(), "singh", "length");
    const Tensor x = Tensor({{0.1, -0.2, 0.3}, {0.0, 0.5, -0.1}}),
                 r = Tensor({{0.2, 0.1, 0.3}, {0.05, 0.2, 0.1}});
    const ZonoInput in = full.input_zonotope(x, r);
    const ZonoPropagation out = full.forward(in.c, in.dr, in.G, Tensor(), nullptr, true);
    const Zonotope Y = nn.evaluate(Zonotope(in.c, in.G));
    check(test::close(out.c, Y.center(), 1e-12) && test::close(out.G, Y.generators(), 1e-12),
          b + ": every slot gives evaluate's zonotope");
    check(out.layers.size() == 2 && !out.dr.defined(), b + ": two records, a point center");
    // two error slots per layer and two input generators: the rest goes into dr
    for (const std::string order : {"length", "sensitivity*length", "sequential"}) {
        const ZonoNet few(nn, 2, 2, "bounds", order);
        const ZonoInput fin = few.input_zonotope(x, r);
        const ZonoSensitivity s = few.sensitivity(x);
        const ZonoPropagation fo = few.forward(fin.c, fin.dr, fin.G, Tensor(), &s.layer_sum, false);
        check(few.interval_center && fo.dr.defined() && fo.G.shape()[1] == 6,
              b + " " + order + ": 2 + 2 + 2 generators and an interval center");
        // sampled outputs of the boxes lie in the zonotopes with their interval centers
        const Tensor lo = fo.c - fo.G.abs().sum(1) - fo.dr, hi = fo.c + fo.G.abs().sum(1) + fo.dr;
        std::vector<double> u(400 * 3);
        rng.uniform(u.data(), u.size(), -1.0, 1.0);
        bool inside = true;
        for (int64_t i = 0; i < 2; ++i) {
            const Tensor pts =
                x.slice(0, i, i + 1) + r.slice(0, i, i + 1).multiply(Tensor::like(x, u, {400, 3}));
            const Tensor y = nn.evaluate(pts);
            inside = inside && y.greater_equal(lo.slice(0, i, i + 1) - 1e-12).min().item() > 0 &&
                     y.less_equal(hi.slice(0, i, i + 1) + 1e-12).min().item() > 0;
        }
        check(inside, b + " " + order + ": the sampled outputs lie in the sets");
    }
}

/// The F-radius of the output of the batch's set i, the input zonotope with G.
double aux_frad(const ZonoNet &net, const ZonoInput &in, const Tensor &G, const Tensor &splits,
                int64_t i) {
    const ZonoPropagation out = net.forward(in.c, in.dr, G, splits, nullptr, false);
    const std::vector<double> rows = out.G.square().sum(-1).sum(-1).data();
    return std::sqrt(rows[std::size_t(i)] + numerics::kFradSmoothing);
}

/// exact_backprop's reverse pass is the central difference of the F-radius in every input
/// generator's value, for each activation and line, with splits and an interval center.
void frad_gradients(const std::string &b) {
    Rng rng(11);
    const Tensor x = Tensor({{0.1, -0.2, 0.3}, {0.0, 0.5, -0.1}}),
                 r = Tensor({{0.4, 0.3, 0.5}, {0.25, 0.4, 0.3}});
    const Tensor splits = Tensor({{-3.0, 5.0}, {0.0, 2.0}});
    for (const Activation a : {Activation::relu, Activation::sigmoid, Activation::tanh}) {
        const NeuralNetwork nn = aux_network(rng, {3, 6, 5, 2}, a);
        for (const std::string method : {"bounds", "singh"}) {
            const ZonoNet net(nn, 2, 3, method, "length");
            const ZonoInput in = net.input_zonotope(x, r);
            ZonoPropagation out = net.forward(in.c, in.dr, in.G, splits, nullptr, true);
            const std::vector<double> g = priv_frad_gradients(out, in, net, splits, true).data();
            // the difference quotient at each generator's value
            const std::vector<double> dims = in.dims.data();
            bool close = true;
            for (int64_t i = 0; i < 2; ++i) {
                for (int64_t s = 0; s < net.q0; ++s) {
                    const double h = 1e-6;
                    std::vector<double> e(std::size_t(2 * net.q0 * 3), 0.0);
                    e[std::size_t((i * net.q0 + s) * 3) +
                      std::size_t(dims[std::size_t(i * net.q0 + s)])] = h;
                    const Tensor E = Tensor::like(in.G, e, {2, net.q0, 3});
                    const double fd = (aux_frad(net, in, in.G + E, splits, i) -
                                       aux_frad(net, in, in.G - E, splits, i)) /
                                      (2 * h);
                    const double exact = g[std::size_t(i * net.q0 + s)];
                    close = close && std::abs(fd - exact) <= 1e-5 * std::max(1.0, std::abs(fd));
                }
            }
            check(close, b + " " + method + ": the exact gradient is the difference quotient");
        }
    }
}

/// Fourier-Motzkin in both orders holds the feasible points, and the orders agree closely.
void fourier_motzkin(const std::string &b) {
    Rng rng(9);
    const int64_t M = 4, p = 5, q = 6;
    std::vector<double> A(std::size_t(M * p * q)), x0(std::size_t(M * q)), d(std::size_t(M * p));
    rng.normal(A.data(), A.size(), 1.0);
    rng.uniform(x0.data(), x0.size(), -0.5, 0.5);
    // the offsets: a feasible point x0 and a slack
    for (int64_t m = 0; m < M; ++m)
        for (int64_t i = 0; i < p; ++i) {
            double s = 0.2;
            for (int64_t j = 0; j < q; ++j)
                s += A[std::size_t((m * p + i) * q + j)] * x0[std::size_t(m * q + j)];
            d[std::size_t(m * p + i)] = s;
        }
    const Tensor At = Tensor::from_data(A, {M, p, q}), bt = Tensor::from_data(d, {M, p});
    const Tensor one = Tensor::ones({M, q});
    const auto [l1, u1] = priv_fourier_motzkin(At, bt, -one, one, 6, FmOrder::cora, 0.0);
    const auto [l2, u2] = priv_fourier_motzkin(At, bt, -one, one, 6, FmOrder::coralean, 0.0);
    const Tensor x = Tensor::from_data(x0, {M, q});
    check(l1.less_equal(x).min().item() > 0 && x.less_equal(u1).min().item() > 0 &&
              l2.less_equal(x).min().item() > 0 && x.less_equal(u2).min().item() > 0,
          b + ": both orders hold the feasible point");
    check(test::close(l1, l2, 1e-10) && test::close(u1, u2, 1e-10),
          b + ": the orders agree up to rounding");
    check(l1.less_equal(u1).min().item() > 0 && u1.less_equal(one).min().item() > 0,
          b + ": the bounds tighten the unit box");
}

/// The switches of CORA's verify run: repairs off, naive, fgsm, neuron splits, the dual-iter name.
void switches(const std::string &b) {
    Rng rng(11);
    const NeuralNetwork nn = aux_network(rng, {3, 6, 6, 2}, Activation::relu);
    const double least = aux_least_output(nn, rng, 3, -0.3, 0.3);
    for (int variant = 0; variant < 8; ++variant) {
        NnVerifyOptions o = aux_options("conzonotope", b);
        o.falsification_order = "bound-first";
        o.falsification_method = "center";
        o.num_neuron_splits = 1;
        o.num_relu_constraints = 4;
        o.num_approx_err = 3;
        if (variant == 1)
            o.project_stale_factors = o.drop_dummy_duplicates = o.remember_point_splits_only =
                false;
        // one switch per variant, the repairs off in the first
        if (variant == 2) o.refinement_method = "naive";
        if (variant == 3) o.bab_falsification_method = "fgsm";
        if (variant == 4) o.neuron_split_position = "middle";
        if (variant == 5) o.verify_enqueue_type = "append";
        if (variant == 6) o.fourier_motzkin_order = "coralean";
        if (variant == 7) o.num_init_gens = 2;
        const NnVerifyResult r = nn.verify(aux_below(3, 2, -0.3, 0.3, least - 0.5), 20, o);
        check(r.status == "unsat", b + ": the variant " + std::to_string(variant) + " verifies");
    }
}

/// Wrong inputs are refused; the checks off give the same verdict.
void checks_and_their_switch() {
    Rng rng(2);
    const NeuralNetwork nn = aux_network(rng, {2, 3, 2}, Activation::relu);
    check(test::raises([&] { nn.verify(aux_below(3, 2, 0, 1, 0), 1); }, Error::dimension_mismatch),
          "verify: a box of another dimension");
    check(test::raises([&] { nn.verify(aux_below(2, 3, 0, 1, 0), 1); }, Error::dimension_mismatch),
          "verify: a polytope of another output dimension");
    NnProperty p = aux_below(2, 2, 0, 1, 0);
    p.poly_case = {1};
    check(test::raises([&] { nn.verify(p, 1); }, Error::wrong_value), "verify: a missing case");
    check(test::raises([&] { nn.verify(aux_below(2, 2, 0, 1, 0), -1); }, Error::wrong_value),
          "verify: a negative timeout");
    const NnProperty far = aux_below(2, 2, 0, 1, -100);
    const std::string on = nn.verify(far, 5, vnncomp_options("x")).status;
    ChecksScope off(false);
    check(nn.verify(far, 5, vnncomp_options("x")).status == on,
          "verify: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        verdicts(b);
        batched_zonotopes(b);
        frad_gradients(b);
        fourier_motzkin(b);
        switches(b);
    });
    checks_and_their_switch();
    return test::finish("neuralNetwork verify");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
