// test_neural_network_verify - the verdicts of verify, its options, strategies and sets
//
// Small ReLU, sigmoid and tanh networks whose properties are known: an unsafe set far below the
// outputs is "unsat" for both boundings, one around a sampled output is "sat" with a counterexample
// whose output lies in it. The batched zonotope with every slot is the output zonotope of
// NeuralNetwork::evaluate, and with few slots (an interval center) it still holds sampled outputs;
// Fourier-Motzkin in both orders holds the feasible points of random systems; the three repairs and
// every switch of CORA's verify run. exact_backprop's F-radius gradient is the difference quotient.
// Many cases of few polytopes (lindex) decide chunk by chunk.
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
// Last update:   24-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/rng.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/nn/neural_network/neural_network.h"
#include "cora/nn/neural_network/verify.h"
#include "nn/neural_network/private/priv_verify.h"
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
    NnVerifyOptions o = bounding == "zonotope" ? NnVerifyOptions{} : cora_vnncomp_options("test");
    o.bounding_method = bounding;
    o.alpha_iterations = 5;
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
        for (const std::string bounding : {"conzonotope", "zonotope", "zonotope-alpha"}) {
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
    // the look-ahead neuron splits (fsb), the backward bound and the chord rows, on a ReLU network
    {
        const NeuralNetwork nn = aux_network(rng, {3, 8, 8, 2}, Activation::relu);
        const double least = aux_least_output(nn, rng, 3, -0.2, 0.2);
        NnVerifyOptions o = aux_options("conzonotope", b);
        o.falsification_order = "bound-first";
        o.neuron_split_heuristic = "fsb";
        o.num_neuron_splits = 1;
        o.backward_bound = true;
        o.num_relu_constraints = std::numeric_limits<double>::infinity();
        o.relu_constraint_faces = "triangle";
        NnVerifyResult r = nn.verify(aux_below(3, 2, -0.2, 0.2, least - 0.05), 20, o);
        check(r.status == "unsat", b + ": fsb splits and chord rows verify below the outputs");
        r = nn.verify(aux_below(3, 2, -0.2, 0.2, least + 0.01), 20, o);
        check(r.status == "sat", b + ": fsb splits falsify around the least output");
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

/// The lower bound of a y over the output of set i, the slopes alpha, and the sampled least.
double aux_alpha_lower(const ZonoNet &net, const ZonoInput &in, const Tensor &splits,
                       const std::vector<Tensor> &alpha, const Tensor &a, int64_t i) {
    const ZonoPropagation out = net.forward(in.c, in.dr, in.G, splits, nullptr, false, &alpha);
    Tensor lower = out.c.matmul(a) - out.G.matmul(a).abs().sum(1);
    if (out.dr.defined()) lower = lower - out.dr.matmul(a.abs());
    return lower.data()[std::size_t(i)];
}

/// zonotope-alpha's reverse pass is the central difference of a lower bound in every slope, with
/// splits and an interval center; every slope in [0, 1] bounds the sampled outputs.
void alpha_gradients(const std::string &b) {
    Rng rng(13);
    const Tensor x = Tensor({{0.1, -0.2, 0.3}, {0.0, 0.5, -0.1}}),
                 r = Tensor({{0.4, 0.3, 0.5}, {0.25, 0.4, 0.3}});
    const Tensor splits = Tensor({{-3.0, 5.0}, {0.0, 2.0}}), a = Tensor({{1.0}, {-0.7}});
    const NeuralNetwork nn = aux_network(rng, {3, 6, 5, 2}, Activation::relu);
    for (const double slots : {2.0, std::numeric_limits<double>::infinity()}) {
        const ZonoNet net(nn, 3, slots, "bounds", "length");
        // the start of the slopes, then random ones in (0.1, 0.9)
        const ZonoInput in = net.input_zonotope(x, r);
        const ZonoPropagation first = net.forward(in.c, in.dr, in.G, splits, nullptr, true);
        std::vector<Tensor> alpha = priv_alpha_start(first, net, splits);
        for (Tensor &t : alpha) {
            if (!t.defined()) continue;
            std::vector<double> v(t.data().size());
            rng.uniform(v.data(), v.size(), 0.1, 0.9);
            t = Tensor::like(t, v, t.shape());
        }
        // the seeds of the lower bound of a y, then the gradient in every slope
        const ZonoPropagation out = net.forward(in.c, in.dr, in.G, splits, nullptr, true, &alpha);
        const Tensor at = a.matrix_transpose();
        const Tensor S = out.G.matmul(a).sign();
        const std::vector<Tensor> g = priv_alpha_gradients(
            out, net, splits, alpha, at.broadcast_to({2, 2}), S.matmul(at).negative(),
            out.dr.defined() ? at.abs().broadcast_to({2, 2}).negative() : Tensor());
        // the difference quotient in each slope of each set
        bool close = true;
        int64_t compared = 0;
        for (std::size_t k = 0; k < alpha.size(); ++k) {
            if (!alpha[k].defined()) continue;
            const std::vector<double> gk = g[k].data(), ak = alpha[k].data();
            for (std::size_t j = 0; j < ak.size(); ++j) {
                const double h = 1e-6;
                std::vector<Tensor> plus = alpha, minus = alpha;
                std::vector<double> e(ak.size(), 0.0);
                e[j] = h;
                plus[k] = alpha[k] + Tensor::like(alpha[k], e, alpha[k].shape());
                minus[k] = alpha[k] - Tensor::like(alpha[k], e, alpha[k].shape());
                // the set of the slope, its lower bound up and down
                const auto i = int64_t(j) / alpha[k].shape()[1];
                const double fd = (aux_alpha_lower(net, in, splits, plus, a, i) -
                                   aux_alpha_lower(net, in, splits, minus, a, i)) /
                                  (2 * h);
                close = close && std::abs(fd - gk[j]) <= 1e-5 * std::max(1.0, std::abs(fd));
                compared += gk[j] != 0.0 ? 1 : 0;
            }
        }
        check(close && compared > 0, b + ": the slopes' gradient is the difference quotient");
        // the sampled outputs of set 0 lie above the lower bound, at every slope
        std::vector<double> pts(std::size_t(2000 * 3));
        rng.uniform(pts.data(), pts.size(), -1.0, 1.0);
        const Tensor xs =
            Tensor::from_data(pts, {2000, 3}).multiply(r.slice(0, 0, 1)) + x.slice(0, 0, 1);
        // the points that respect the splits of set 0 (neuron 3 at least 0, neuron 5 at most 0)
        const Tensor h1 = xs.matmul(net.W[0]) + net.b[0];
        const std::vector<double> hv = h1.data(), y = nn.evaluate(xs).matmul(a).data();
        double least = std::numeric_limits<double>::infinity();
        for (std::size_t p = 0; p < 2000; ++p)
            if (hv[p * 6 + 2] >= 0 && hv[p * 6 + 4] <= 0) least = std::min(least, y[p]);
        check(aux_alpha_lower(net, in, splits, alpha, a, 0) <= least,
              b + ": the slopes' zonotope holds the sampled outputs");
    }
}

/// CROWN's backward bound holds the sampled outputs of the boxes, with splits, for every
/// activation and with given ReLU slopes; on a box around a point it is close to the output.
void backward_bounds(const std::string &b) {
    Rng rng(17);
    const Tensor x = Tensor({{0.1, -0.2, 0.3}, {0.0, 0.5, -0.1}}),
                 r = Tensor({{0.4, 0.3, 0.5}, {0.25, 0.4, 0.3}});
    const Tensor splits = Tensor({{-3.0, 5.0}, {0.0, 0.0}}),
                 rows = Tensor({{1.0, -0.7}, {-0.3, 1.0}});
    std::vector<double> pts(std::size_t(3000 * 3));
    rng.uniform(pts.data(), pts.size(), -1.0, 1.0);
    for (const Activation a : {Activation::relu, Activation::sigmoid, Activation::tanh}) {
        const NeuralNetwork nn = aux_network(rng, {3, 6, 5, 2}, a);
        const ZonoNet net(nn, 3, 2, "bounds", "length");
        const ZonoInput in = net.input_zonotope(x, r);
        const ZonoPropagation out = net.forward(in.c, in.dr, in.G, splits, nullptr, true);
        std::vector<Tensor> half;
        if (a == Activation::relu) half = priv_alpha_start(out, net, splits);
        for (Tensor &t : half)
            if (t.defined()) t = t.full_like(t.shape(), 0.5);
        // CROWN's slopes, then the given ones (ReLU only)
        for (const bool given : {false, true}) {
            if (given && a != Activation::relu) continue;
            const std::vector<double> lower =
                priv_backward_lower(out, net, splits, x, r, rows, given ? &half : nullptr).data();
            bool holds = true;
            for (int64_t i = 0; i < 2; ++i) {
                const Tensor xs = Tensor::from_data(pts, {3000, 3}).multiply(r.slice(0, i, i + 1)) +
                                  x.slice(0, i, i + 1);
                const std::vector<double> hv = (xs.matmul(net.W[0]) + net.b[0]).data();
                const std::vector<double> y =
                    nn.evaluate(xs).matmul(rows.matrix_transpose()).data();
                for (std::size_t p = 0; p < 3000; ++p) {
                    // set 0 holds neuron 3 at least 0 and neuron 5 at most 0
                    if (i == 0 && (hv[p * 6 + 2] < 0 || hv[p * 6 + 4] > 0)) continue;
                    for (std::size_t j = 0; j < 2; ++j)
                        holds = holds && lower[std::size_t(i) * 2 + j] <= y[p * 2 + j];
                }
            }
            check(holds, b + ": CROWN's backward bound holds the sampled outputs" +
                             (given ? " (slopes given)" : ""));
        }
        // alpha-CROWN: the optimized slopes hold the sampled outputs, at least CROWN's bound
        const Tensor crown = priv_backward_lower(out, net, splits, x, r, rows);
        const Tensor opt = priv_backward_alpha(out, net, splits, x, r, rows, 20, 0.1);
        bool holds = opt.greater_equal(crown - 1e-9).min().item() > 0;
        const std::vector<double> lower = opt.data();
        for (int64_t i = 0; i < 2; ++i) {
            const Tensor xs = Tensor::from_data(pts, {3000, 3}).multiply(r.slice(0, i, i + 1)) +
                              x.slice(0, i, i + 1);
            const std::vector<double> hv = (xs.matmul(net.W[0]) + net.b[0]).data();
            const std::vector<double> y = nn.evaluate(xs).matmul(rows.matrix_transpose()).data();
            for (std::size_t p = 0; p < 3000; ++p) {
                if (i == 0 && (hv[p * 6 + 2] < 0 || hv[p * 6 + 4] > 0)) continue;
                for (std::size_t j = 0; j < 2; ++j)
                    holds = holds && lower[std::size_t(i) * 2 + j] <= y[p * 2 + j];
            }
        }
        check(holds, b + ": alpha-CROWN holds the sampled outputs, at least CROWN's bound");
    }
}

/// The chord above a crossing ReLU holds at both ends at the stored slope, in long double, at the
/// witness l = -0.006077605996286801, u = 0.008781622029526388, where -chord l alone is 1.9e-19
/// short of relu(u) (CORALean !971, fixture 9529).
void relu_chord_both_ends(const std::string &b) {
    const double lv = -0.006077605996286801, uv = 0.008781622029526388;
    const Tensor l({{lv}}), u({{uv}});
    const Tensor chord = u.divide((u - l).maximum(numerics::kTiny));
    const long double s = chord.item(), o = priv_relu_chord_offset(l, u, chord).item();
    const long double old = chord.multiply(l).negative().item();
    check(s * uv + o >= (long double)uv && s * lv + o >= 0.0L,
          b + ": the ReLU's chord above holds at both ends at the witness");
    // Only an extended long double sees the 1.9e-19 shortfall (MSVC's long double is double).
    if (sizeof(long double) > sizeof(double))
        check(s * uv + old < (long double)uv, b + ": -chord l alone is short at the witness");
}

/// The dual rows' chord face y <= s x + o holds at both ends at the stored slope in long double at
/// relu_chord_both_ends' witness: one neuron's record with zero centers and output generators, so
/// the face's offset is priv_relu_chord_offset itself (-s l alone is 1.9e-19 short).
void dual_chord_face(const std::string &b) {
    const double lv = -0.006077605996286801, uv = 0.008781622029526388;
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor::zeros({1})}}, {Activation::relu});
    const ZonoNet net(nn, 1, 1, "bounds", "length");
    ZonoPropagation out;
    ZonoLayerRecord rec;
    rec.l = Tensor({{lv}});
    rec.u = Tensor({{uv}});
    rec.er = Tensor({{1.0}});
    rec.c_in = rec.c_out = Tensor({{0.0}});
    rec.G_in = Tensor::ones({1, 1, 1});
    rec.G_out = Tensor::zeros({1, net.q, 1});
    out.layers.push_back(rec);
    const auto [C, d] = priv_dual_rows(out, net, Tensor(), 1);
    check(d.defined() && d.shape()[1] == 3, b + ": the three faces of the crossing neuron");
    if (!d.defined() || d.shape()[1] != 3) return;
    // the third face y - s x <= o at (u, u) and (l, 0); its row G_out - s G_in is -s exactly
    const long double o = d.slice(1, 2, 3).item(), s = -C.slice(1, 2, 3).slice(2, 0, 1).item();
    check(s * uv + o >= (long double)uv && s * lv + o >= 0.0L,
          b + ": the chord face of the dual rows holds at both ends at the witness");
}

/// The certificate on rows of very different norms is at most the program's optimum: min w beta
/// over beta in [-1, 1] with c_i beta <= d_i, exact in long double (weak duality on the rows as
/// given, not on rows rescaled to unit norm and rounded).
void dual_unscaled_rows(const std::string &b) {
    Rng rng(29);
    std::vector<double> v(std::size_t(200 * 7));
    rng.uniform(v.data(), v.size(), -1.0, 1.0);
    int violations = 0;
    for (std::size_t t = 0; t < 200; ++t) {
        const double *x = v.data() + 7 * t;
        // three rows c beta <= d of norms 1e-3 to 1e3, all holding at beta = x[6] / 2
        const double beta0 = x[6] / 2, w = x[0];
        std::vector<double> c, d;
        long double lo = -1, hi = 1;
        for (int i = 0; i < 3; ++i) {
            const double ci = x[1 + i] * std::pow(10.0, 3.0 * (i - 1)),
                         di = ci * beta0 + 0.1 * x[4];
            c.push_back(ci);
            d.push_back(std::max(di, ci * beta0));
            const long double r = (long double)(d.back()) / ci;
            if (ci > 0) hi = std::min(hi, r);
            if (ci < 0) lo = std::max(lo, r);
        }
        // the optimum at an end of the feasible interval, and the certificate
        const long double opt = w > 0 ? w * lo : w * hi;
        const double h =
            priv_dual_lower(Tensor::from_data({w}, {1, 1, 1}), Tensor::from_data(c, {1, 3, 1}),
                            Tensor::from_data(d, {1, 3}), 40, 0.1, DType::float64)
                .item();
        violations += (long double)h > opt + 1e-15L;
    }
    check(violations == 0, b + ": the dual certificate is at most the optimum on unequal rows");
}

/// The dual certificate holds the sampled outputs under the splits, is at least the zonotope's
/// bound, and on a small program reaches its optimum.
void dual_bounds(const std::string &b) {
    Rng rng(23);
    // min beta1 + beta2 over [-1, 1]^2 with beta1 + beta2 >= 0 is 0; lambda = 1 attains it
    const Tensor w = Tensor::from_data({1.0, 1.0}, {1, 1, 2});
    const Tensor C = Tensor::from_data({-1.0, -1.0}, {1, 1, 2}), d = Tensor::zeros({1, 1});
    const double opt = priv_dual_lower(w, C, d, 60, 0.1, DType::float32).item();
    check(opt <= 0 && opt > -0.05, b + ": the dual bound reaches the program's optimum");
    const Tensor x = Tensor({{0.1, -0.2, 0.3}, {0.0, 0.5, -0.1}}),
                 r = Tensor({{0.4, 0.3, 0.5}, {0.25, 0.4, 0.3}});
    const Tensor splits = Tensor({{-3.0, 5.0}, {0.0, 0.0}}),
                 rows = Tensor({{1.0, -0.7}, {-0.3, 1.0}});
    std::vector<double> pts(std::size_t(3000 * 3));
    rng.uniform(pts.data(), pts.size(), -1.0, 1.0);
    const NeuralNetwork nn = aux_network(rng, {3, 6, 5, 2}, Activation::relu);
    for (const double slots : {2.0, 6.0}) {
        const ZonoNet net(nn, 3, slots, "bounds", "length");
        const ZonoInput in = net.input_zonotope(x, r);
        const ZonoPropagation out = net.forward(in.c, in.dr, in.G, splits, nullptr, true);
        const auto [Cr, dr] = priv_dual_rows(out, net, splits, 8);
        // the zonotope's bound of the rows and the dual one
        const Tensor wr = out.G.matmul(rows.matrix_transpose()).permute_dims({0, 2, 1});
        Tensor base = out.c.matmul(rows.matrix_transpose());
        if (out.dr.defined()) base = base - out.dr.matmul(rows.abs().matrix_transpose());
        const Tensor zono = base - wr.abs().sum(-1);
        const Tensor dual = base + priv_dual_lower(wr, Cr, dr, 30, 0.1, DType::float32);
        const std::vector<double> lower = dual.data();
        bool holds = dual.greater_equal(zono).min().item() > 0;
        for (int64_t i = 0; i < 2; ++i) {
            const Tensor xs = Tensor::from_data(pts, {3000, 3}).multiply(r.slice(0, i, i + 1)) +
                              x.slice(0, i, i + 1);
            const std::vector<double> hv = (xs.matmul(net.W[0]) + net.b[0]).data();
            const std::vector<double> y = nn.evaluate(xs).matmul(rows.matrix_transpose()).data();
            for (std::size_t p = 0; p < 3000; ++p) {
                // set 0 holds neuron 3 at least 0 and neuron 5 at most 0
                if (i == 0 && (hv[p * 6 + 2] < 0 || hv[p * 6 + 4] > 0)) continue;
                for (std::size_t j = 0; j < 2; ++j)
                    holds = holds && lower[std::size_t(i) * 2 + j] <= y[p * 2 + j];
            }
        }
        check(holds, b + ": the dual certificate holds the sampled outputs, above the zonotope's");
    }
    // the table of one set's crossing neurons: priv_dual_rows' rows of their splits
    const ZonoNet net(nn, 3, 6.0, "bounds", "length");
    const Tensor x0 = x.slice(0, 0, 1), r0 = r.slice(0, 0, 1);
    const ZonoInput in = net.input_zonotope(x0, r0);
    const ZonoPropagation root = net.forward(in.c, in.dr, in.G, Tensor(), nullptr, true);
    const PrivSplitTable table = priv_split_table(root, net);
    const std::vector<double> ids = table.ids.data();
    const int64_t U = int64_t(ids.size());
    check(U >= 2, b + ": the set has crossing neurons to split");
    if (U < 2) return;
    // the first and the last crossing neuron, below and above 0: priv_dual_rows' -id and +id
    const Tensor some = Tensor::like(x0, {-ids.front(), ids.back()}, {1, 2});
    const auto [Cd, dd] = priv_dual_rows(root, net, some, 0);
    const int64_t L = Cd.shape()[1] / 2; // one block of the two splits per layer
    const Tensor Cs = Cd.reshape({1, L, 2, net.q}).sum(1), ds = dd.reshape({1, L, 2}).sum(1);
    const Tensor R = table.rows.take({0, U - 1}, 0), c = table.c.take({0, U - 1}, 0),
                 dr = table.dr.take({0, U - 1}, 0);
    const Tensor sg = Tensor::like(x0, {-1.0, 1.0}, {1, 2});
    check(test::close(R.multiply(sg.reshape({2, 1})).expand_dims(0), Cs, 1e-12) &&
              test::close(sg.multiply(c.negative()) + dr, ds, 1e-12),
          b + ": the table's rows with the signs are priv_dual_rows'");
    // the bound of nodes of shared rows: at 0 steps priv_dual_lower's, better with steps, sound
    const Tensor wr = root.G.matmul(rows.matrix_transpose()).permute_dims({0, 2, 1}).squeeze(0);
    const Tensor nodes = Tensor::like(x0, {-1.0, 1.0, 1.0, -1.0, 1.0, 1.0}, {3, 2});
    const Tensor at0 = priv_dual_lower_signs(wr, R, c, dr, nodes, 0, 0.1, DType::float32);
    bool same = true;
    for (int64_t k = 0; k < 3; ++k) {
        const Tensor sk = nodes.slice(0, k, k + 1);
        const Tensor Ck = R.multiply(sk.reshape({2, 1})).expand_dims(0);
        const Tensor dk = sk.multiply(c.negative()) + dr;
        same =
            same &&
            test::close(at0.slice(0, k, k + 1),
                        priv_dual_lower(wr.expand_dims(0), Ck, dk, 0, 0.1, DType::float32), 1e-10);
    }
    check(same, b + ": the nodes' bound at 0 steps is priv_dual_lower's");
    Tensor best;
    const Tensor at30 =
        priv_dual_lower_signs(wr, R, c, dr, nodes, 30, 0.1, DType::float32, Tensor(), &best);
    const Tensor warm = priv_dual_lower_signs(wr, R, c, dr, nodes, 0, 0.1, DType::float32, best);
    check(at30.greater_equal(at0 - 1e-12).min().item() > 0 && test::close(warm, at30, 1e-6),
          b + ": the steps raise the nodes' bound, a warm start keeps it");
    // sound: below the sampled generator sums of the factors that satisfy each node's rows
    std::vector<double> bs(std::size_t(20000 * net.q));
    rng.uniform(bs.data(), bs.size(), -1.0, 1.0);
    const Tensor beta = Tensor::from_data(bs, {20000, net.q});
    const Tensor zs = beta.matmul(R.matrix_transpose()) + c; // the split neurons' inputs
    const Tensor vals = beta.matmul(wr.matrix_transpose());  // (20000, K)
    const std::vector<double> z = zs.data(), v = vals.data(), lo = at30.data(), ns = nodes.data(),
                              rad = dr.data();
    const int64_t K = wr.shape()[0];
    bool sound = true;
    for (int64_t k = 0; k < 3; ++k)
        for (int64_t p = 0; p < 20000; ++p) {
            bool in_node = true;
            for (int64_t j = 0; j < 2; ++j) {
                // s_j z_j <= dr_j: the input's side, its error at the worst
                const double sj = ns[std::size_t(k * 2 + j)];
                in_node = in_node && sj * z[std::size_t(p * 2 + j)] <= rad[std::size_t(j)];
            }
            if (!in_node) continue;
            for (int64_t i = 0; i < K; ++i)
                sound = sound && lo[std::size_t(k * K + i)] <= v[std::size_t(p * K + i)] + 1e-9;
        }
    check(sound, b + ": the nodes' bound holds the sampled factors in each node");
    // z = beta, y = 2 beta - 0.5: inactive is infeasible by its y rows only (y = 0 at beta = 0.25)
    PrivSplitTable one;
    one.rows = Tensor::like(x0, {1.0}, {1, 1});
    one.c = one.dr = one.dr_out = Tensor::like(x0, {0.0}, {1});
    one.rows_out = Tensor::like(x0, {2.0}, {1, 1});
    one.c_out = Tensor::like(x0, {-0.5}, {1});
    const Tensor none = Tensor::like(x0, {0.0}, {1, 1}), inactive = Tensor::like(x0, {1.0}, {1, 1});
    const double halfspace =
        priv_dual_lower_signs(none, one.rows, one.c, one.dr, inactive, 20, 0.1, DType::float32)
            .item();
    const double phase =
        priv_dual_lower_phases(none, one, inactive, 20, 0.1, DType::float32).item();
    const double active =
        priv_dual_lower_phases(none, one, inactive.negative(), 20, 0.1, DType::float32).item();
    check(halfspace <= 0 && phase > 1e-3 && active <= 0,
          b + ": an inactive phase infeasible by its output rows only gets Farkas' certificate");
    // z = beta_0 over [-1, 1], y = z / 2 + 1/4 + beta_1 / 4: min y is -1/2, 0 with the faces
    PrivSplitTable tri;
    tri.rows = Tensor::like(x0, {1.0, 0.0}, {1, 2});
    tri.rows_out = Tensor::like(x0, {0.5, 0.25}, {1, 2});
    tri.c = tri.dr = tri.dr_out = Tensor::like(x0, {0.0}, {1});
    tri.c_out = Tensor::like(x0, {0.25}, {1});
    tri.l = Tensor::like(x0, {-1.0}, {1});
    tri.u = Tensor::like(x0, {1.0}, {1});
    const Tensor wy = Tensor::like(x0, {0.5, 0.25}, {1, 2}),
                 unsplit = Tensor::like(x0, {0.0}, {1, 1});
    const double plain = 0.25 - wy.abs().sum().item();
    const double faced =
        0.25 + priv_dual_lower_phases(wy, tri, unsplit, 200, 0.1, DType::float32).item();
    check(plain < -0.49 && faced > -0.05 && faced <= 1e-12,
          b + ": an unsplit neuron's faces raise the bound to the triangle's, not above");
}

/// Rows of norms 1e-6 and 1e6: the nodes' bounds are the Lagrangian of the returned multipliers on
/// the unscaled rows (evaluated here on the host), below the sampled least value.
void dual_bounds_unscaled(const std::string &b) {
    Rng rng(29);
    // two split neurons, the first of rows near 1e-6, the second near 1e6
    PrivSplitTable t;
    t.rows = Tensor::from_data({1e-6, -0.7e-6, 0.4e6, 1e6}, {2, 2});
    t.c = Tensor::from_data({0.1e-6, -0.2e6}, {2});
    t.dr = Tensor::from_data({0.3e-6, 0.5e6}, {2});
    t.rows_out = Tensor::from_data({0.6e-6, -0.2e-6, 0.3e6, 0.9e6}, {2, 2});
    t.c_out = Tensor::from_data({0.2e-6, 0.1e6}, {2});
    t.dr_out = Tensor::from_data({0.2e-6, 0.4e6}, {2});
    const Tensor w = Tensor::from_data({1.0, -0.5, 0.3, 1.0}, {2, 2});
    // the four nodes of their signs, bounded with 30 steps
    const std::vector<double> sv = {1, 1, 1, -1, -1, 1, -1, -1};
    const Tensor nodes = Tensor::from_data(sv, {4, 2});
    const std::vector<double> R = t.rows.data(), c = t.c.data(), dr = t.dr.data(),
                              Ro = t.rows_out.data(), co = t.c_out.data(), dro = t.dr_out.data(),
                              wv = w.data();
    Tensor nu, lam;
    const std::vector<double> phases =
        priv_dual_lower_phases(w, t, nodes, 30, 0.1, DType::float32, Tensor(), &nu).data();
    const std::vector<double> signs =
        priv_dual_lower_signs(w, t.rows, t.c, t.dr, nodes, 30, 0.1, DType::float32, Tensor(), &lam)
            .data();
    // the Lagrangian of lambda = nu / norm on the host, from the unscaled rows
    const std::vector<double> nv = nu.data(), lv = lam.data();
    const auto norm = [](double a0, double a1) { return std::abs(a0) + std::abs(a1); };
    bool same = true;
    for (std::size_t n = 0; n < 4; ++n)
        for (std::size_t k = 0; k < 2; ++k) {
            // the halfspaces s_j (R_j beta + c_j) <= dr_j; the phases' rows of either side
            double tp[2] = {wv[k * 2], wv[k * 2 + 1]}, ts[2] = {tp[0], tp[1]}, op = 0, os = 0;
            for (std::size_t j = 0; j < 2; ++j) {
                // the phase's row and offsets of split j at node n
                const double sj = sv[n * 2 + j], r0 = R[j * 2], r1 = R[j * 2 + 1];
                const bool on = sj < 0;
                const double p0 = on ? Ro[j * 2] - r0 : Ro[j * 2],
                             p1 = on ? Ro[j * 2 + 1] - r1 : Ro[j * 2 + 1];
                const double e = on ? c[j] - co[j] : -co[j], D = on ? dr[j] + dro[j] : dro[j];
                const auto at = [&](std::size_t blk) { return nv[(n * 2 + k) * 6 + blk * 2 + j]; };
                // its multipliers in the rows' own units, and their terms
                const double l1 = std::max(at(0), 0.0) / norm(r0, r1),
                             l2 = std::max(at(1), 0.0) / norm(p0, p1),
                             l3 = std::max(at(2), 0.0) / norm(p0, p1);
                tp[0] += l1 * sj * r0 + (l2 - l3) * p0;
                tp[1] += l1 * sj * r1 + (l2 - l3) * p1;
                op += l1 * (-sj * c[j] + dr[j]) + l2 * (e + D) + l3 * (D - e);
                const double ls = std::max(lv[(n * 2 + k) * 2 + j], 0.0) / norm(r0, r1);
                ts[0] += ls * sj * r0;
                ts[1] += ls * sj * r1;
                os += ls * (-sj * c[j] + dr[j]);
            }
            const double free = -norm(wv[k * 2], wv[k * 2 + 1]);
            const double ep = std::max(-norm(tp[0], tp[1]) - op, free),
                         es = std::max(-norm(ts[0], ts[1]) - os, free);
            same = same && std::abs(ep - phases[n * 2 + k]) <= 1e-12 * (1 + std::abs(ep)) &&
                   std::abs(es - signs[n * 2 + k]) <= 1e-12 * (1 + std::abs(es));
        }
    check(same, b + ": the nodes' bounds are the Lagrangian on the unscaled rows");
    // sound: below w beta at the sampled factors and input errors of the points of each node
    std::vector<double> bs(std::size_t(200000 * 2)), ds(std::size_t(200000 * 2));
    rng.uniform(bs.data(), bs.size(), -1.0, 1.0);
    rng.uniform(ds.data(), ds.size(), -1.0, 1.0);
    bool sound = true;
    int64_t hits = 0;
    for (std::size_t p = 0; p < 200000; ++p) {
        const double b0 = bs[p * 2], b1 = bs[p * 2 + 1];
        for (std::size_t n = 0; n < 4; ++n) {
            bool in_node = true, in_half = true;
            for (std::size_t j = 0; j < 2; ++j) {
                const double z = R[j * 2] * b0 + R[j * 2 + 1] * b1 + c[j] + dr[j] * ds[p * 2 + j];
                // the output y = z (active) or 0 (inactive) within its error of its affine form
                const double y = Ro[j * 2] * b0 + Ro[j * 2 + 1] * b1 + co[j];
                const bool on = sv[n * 2 + j] < 0;
                in_half = in_half && (on ? z >= 0 : z <= 0);
                in_node =
                    in_node && (on ? z >= 0 : z <= 0) && std::abs((on ? z : 0.0) - y) <= dro[j];
            }
            for (std::size_t k = 0; k < 2; ++k) {
                const double v = wv[k * 2] * b0 + wv[k * 2 + 1] * b1;
                if (in_node) sound = sound && phases[n * 2 + k] <= v + 1e-9;
                if (in_half) sound = sound && signs[n * 2 + k] <= v + 1e-9;
            }
            hits += in_node ? 1 : 0;
        }
    }
    check(sound && hits > 0, b + ": the bounds of rows of very different norms hold the samples");
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
    for (int variant = 0; variant < 11; ++variant) {
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
        if (variant == 8) o.exact_backprop = false;
        if (variant == 9) o.verify_dequeue_type = "front";
        if (variant == 10) o.refinement_min_move = 0.05;
        // close enough to the outputs that the first box needs the refinement
        const NnVerifyResult r = nn.verify(aux_below(3, 2, -0.3, 0.3, least - 0.1), 20, o);
        check(r.status == "unsat" && r.stats.count("artifacts_seconds") == 1,
              b + ": the variant " + std::to_string(variant) + " verifies after a refinement");
    }
}

/// An attack and a bounding that are always late (check_deadline of a passed time).
class LateAttack : public Attack {
  public:
    AttackResult run(NnProblem &, double, int64_t) override {
        check_deadline(now_seconds() - 1.0);
        return {};
    }
};
class LateBounding : public Bounding {
  public:
    bool prepare(NnProblem &) override { return true; }
    std::string branch_and_bound(NnProblem &, double, std::map<std::string, double> &,
                                 AttackResult &) override {
        check_deadline(now_seconds() - 1.0);
        return "unsat";
    }
};

/// A deadline that passed stops a pass at its next layer (check_deadline), and verify on a wide
/// network answers within a second of a short timeout.
void deadlines(const std::string &b) {
    Rng rng(9);
    const NeuralNetwork nn = aux_network(rng, {3, 6, 5, 2}, Activation::relu);
    ZonoNet net(nn, 2, 2, "bounds", "length");
    const Tensor x = Tensor({{0.1, -0.2, 0.3}}), r = Tensor({{0.2, 0.1, 0.3}});
    const ZonoInput in = net.input_zonotope(x, r);
    const auto stops = [](const auto &f) {
        try {
            f();
        } catch (const Error &error) {
            return is_deadline(error);
        }
        return false;
    };
    check(!stops([&] { net.forward(in.c, in.dr, in.G, Tensor(), nullptr, true); }),
          b + ": no deadline, no stop");
    net.deadline = now_seconds() - 1.0;
    check(stops([&] { net.forward(in.c, in.dr, in.G, Tensor(), nullptr, true); }) &&
              stops([&] { net.sensitivity(x); }) && stops([&] { net.evaluate(x); }),
          b + ": a passed deadline stops forward, sensitivity and evaluate");
    // a strategy that is late leaves the verdict open
    register_attack("late", [](const NnVerifyOptions &) { return std::make_unique<LateAttack>(); });
    register_bounding("late",
                      [](const NnVerifyOptions &) { return std::make_unique<LateBounding>(); });
    NnVerifyOptions late = aux_options("conzonotope", b);
    late.falsification_method = late.bounding_method = "late";
    const NnVerifyResult open = nn.verify(aux_below(3, 2, -0.2, 0.2, -100.0), 5, late);
    check(open.status == "unknown" && open.stats.at("attack_cut") == 2 &&
              open.stats.at("bab_cut") == 1,
          b + ": a late attack and bounding leave the verdict open");
    // the candidates in chunks of one (a memory budget of one byte)
    NnVerifyOptions tiny = aux_options("conzonotope", b);
    tiny.memory_budget = 1;
    const Tensor pts = Tensor({{0.1, -0.2, 0.3}, {0.0, 0.1, 0.0}, {-0.1, 0.2, 0.1}});
    const Tensor cases = pts.zeros_like({3});
    for (const double t : {-1000.0, 1000.0}) {
        NnProblem problem(nn, aux_below(3, 2, -0.2, 0.2, t), tiny, ConfirmFn{});
        AttackResult found;
        check(priv_check_points(problem, pts, cases, found) == (t > 0),
              b + ": the candidates in chunks, a counterexample only below +1000");
    }
    // and a passed deadline between them
    NnProblem past(nn, aux_below(3, 2, -0.2, 0.2, -1000.0), tiny, ConfirmFn{});
    past.deadline = now_seconds() - 1.0;
    AttackResult none;
    check(stops([&] { priv_check_points(past, pts, cases, none); }),
          b + ": a passed deadline stops the chunks");
    // PGD: a step past the deadline ends the search with its counts; too slow, fewer restarts
    NnVerifyOptions pgd = aux_options("conzonotope", b);
    NnProblem cut(nn, aux_below(3, 2, -0.2, 0.2, -1000.0), pgd, ConfirmFn{});
    cut.deadline = now_seconds() - 1.0;
    check(make_attack("pgd", pgd)->run(cut, now_seconds() + 10.0, 1).iterations == 0,
          b + ": a PGD step past the deadline ends the search");
    pgd.pgd_min_steps = int64_t{1} << 40;
    NnProblem slow(nn, aux_below(3, 2, -0.2, 0.2, -1000.0), pgd, ConfirmFn{});
    const AttackResult shrunk = make_attack("pgd", pgd)->run(slow, now_seconds() + 10.0, 1);
    check(shrunk.iterations == pgd.pgd_steps &&
              shrunk.evaluations == pgd.pgd_restarts + (pgd.pgd_steps - 1),
          b + ": PGD keeps one restart when its steps would not fit");
    if (b == "lean") return; // the wide network is slow on lean
    // a wide network whose box is not decided quickly: back within a second of the timeout
    const NeuralNetwork wide = aux_network(rng, {40, 400, 400, 400, 400, 2}, Activation::relu);
    const double least = aux_least_output(wide, rng, 40, -1.0, 1.0);
    NnVerifyOptions o = vnncomp_options("test");
    o.memory_budget = int64_t{1} << 28;
    const double start = now_seconds();
    const NnVerifyResult res = wide.verify(aux_below(40, 2, -1.0, 1.0, least - 3.0), 0.5, o);
    check(res.status != "sat" && now_seconds() - start < 1.5,
          b + ": verify answers within a second of its timeout");
}

/// PGD toward a target polytope: its gradient is that polytope's alone, and the targeted restarts
/// find a counterexample of one disjunct.
void targeted_attacks(const std::string &b) {
    Rng rng(5);
    const NeuralNetwork nn = aux_network(rng, {3, 8, 8, 2}, Activation::relu);
    // y_0 <= b0 or y_1 <= b1, both polytopes in case 0, and the second one alone
    NnProperty both = aux_below(3, 2, -0.2, 0.2, -1000.0);
    both.A = Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 1, 2});
    both.b = Tensor::from_data({-1000.0, 1000.0}, {2, 1});
    both.poly_case = {0, 0};
    NnProperty second = both;
    second.A = Tensor::from_data({0.0, 1.0}, {1, 1, 2});
    second.b = Tensor::from_data({1000.0}, {1, 1});
    second.poly_case = {0};
    NnVerifyOptions o = aux_options("conzonotope", b);
    o.pgd_targeted = true;
    // the gradient toward polytope 1 is the gradient of the property with polytope 1 only
    NnProblem pb(nn, both, o, ConfirmFn{}), ps(nn, second, o, ConfirmFn{});
    const Tensor x = (pb.lo + pb.hi) * 0.5;
    const Tensor targets = Tensor::like(x, {1.0}, {1, 1});
    check(test::close(pb.objective_gradient(x, pb.search_case, nullptr, &targets),
                      ps.objective_gradient(x, ps.search_case), 1e-6),
          b + ": the gradient toward a target polytope is that polytope's");
    // the least sampled y_1: just above it, only the second disjunct is reachable
    std::vector<double> xs(4000 * 3);
    rng.uniform(xs.data(), xs.size(), -0.2, 0.2);
    const std::vector<double> y = nn.evaluate(Tensor::from_data(xs, {4000, 3})).data();
    double low = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < 4000; ++i) low = std::min(low, y[i * 2 + 1]);
    both.b = Tensor::from_data({-1000.0, low + 0.01}, {2, 1});
    NnProblem reach(nn, both, o, ConfirmFn{});
    check(!make_attack("pgd", o)->run(reach, now_seconds() + 10.0, 2).x.empty(),
          b + ": targeted PGD finds the counterexample of one disjunct");
    // one group per polytope: twice the samples of an untargeted round, and nothing found
    NnVerifyOptions plain = o;
    plain.pgd_targeted = false;
    both.b = Tensor::from_data({-1000.0, -1000.0}, {2, 1});
    NnProblem none_t(nn, both, o, ConfirmFn{}), none_p(nn, both, plain, ConfirmFn{});
    const AttackResult t = make_attack("pgd", o)->run(none_t, now_seconds() + 10.0, 1);
    const AttackResult u = make_attack("pgd", plain)->run(none_p, now_seconds() + 10.0, 1);
    check(t.x.empty() && u.x.empty() && t.evaluations == 2 * u.evaluations,
          b + ": targeted PGD runs its restarts per polytope");
}

/// relu-bab on random ReLU networks over a box: never unsat where a sampled point violates the
/// property or zonotack confirms a counterexample; unsat where the margin is wide; nodes pruned as
/// infeasible (Farkas) on some networks.
void relu_bab_verdicts(const std::string &b) {
    if (b == "lean") return; // many small verifications: slow on lean
    Rng rng(31);
    int wrong = 0, against_zonotack = 0, proved = 0, wide = 0;
    bool slotted = true;
    double infeasible = 0;
    for (int net_i = 0; net_i < 6; ++net_i) {
        const NeuralNetwork nn = aux_network(rng, {2, 10, 10, 2}, Activation::relu);
        // the least sampled y_0 over [-1, 1]^2, and the samples to judge an unsat by
        std::vector<double> xs(std::size_t(20000 * 2));
        rng.uniform(xs.data(), xs.size(), -1.0, 1.0);
        const std::vector<double> y = nn.evaluate(Tensor::from_data(xs, {20000, 2})).data();
        double least = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < 20000; ++i) least = std::min(least, y[i * 2]);
        // relu-bab, zonotack, and relu-bab without faces from a net of one slot per layer
        NnVerifyOptions o = aux_options("zonotope-alpha", b);
        o.refinement_method = "relu-bab";
        o.num_approx_err = 1000; // a slot for every neuron's error: the phases pin the outputs
        NnVerifyOptions z = aux_options("zonotope-alpha", b), plain = o;
        plain.relu_bab_faces = 0; // the nodes' splits only: Farkas prunes some of them
        plain.num_approx_err = 1; // the root's own net: a slot per crossing neuron
        plain.relu_bab_order = "error";
        plain.relu_bab_alpha_iterations = 0; // the slopes may make more neurons cross
        plain.relu_bab_farkas_steps = 10;
        for (const double t : {least - 0.5, least - 0.02, least + 0.02}) {
            // the wide margin with time to spare: its verdict depends on the nodes, not the load
            const double timeout = t == least - 0.5 ? 120 : 3;
            const NnVerifyResult r = nn.verify(aux_below(2, 2, -1.0, 1.0, t), timeout, o);
            // a sample at most t makes an unsat wrong
            if (r.status == "unsat")
                for (std::size_t i = 0; i < 20000; ++i) wrong += y[i * 2] <= t ? 1 : 0;
            if (r.status == "unsat")
                against_zonotack +=
                    nn.verify(aux_below(2, 2, -1.0, 1.0, t), 3, z).status == "sat" ? 1 : 0;
            // the wide margin: proved; without faces from the root's own net, Farkas prunes
            if (t != least - 0.5) continue;
            ++wide;
            proved += r.status == "unsat" ? 1 : 0;
            const NnVerifyResult r0 = nn.verify(aux_below(2, 2, -1.0, 1.0, t), timeout, plain);
            if (r0.status == "unsat")
                for (std::size_t i = 0; i < 20000; ++i) wrong += y[i * 2] <= t ? 1 : 0;
            // every crossing neuron with a slot, and the nodes Farkas pruned
            slotted = slotted && r0.stats.count("relu_bab_slotless") &&
                      r0.stats.at("relu_bab_slotless") == 0;
            if (r0.stats.count("relu_bab_infeasible_nodes"))
                infeasible += r0.stats.at("relu_bab_infeasible_nodes");
        }
    }
    check(wrong == 0, b + ": relu-bab is never unsat where a sample violates the property");
    check(against_zonotack == 0, b + ": relu-bab is never unsat where zonotack finds a sat");
    check(2 * proved >= wide, b + ": relu-bab proves most properties of a wide margin");
    check(infeasible > 0, b + ": relu-bab prunes nodes infeasible by Farkas' certificate");
    check(slotted,
          b + ": relu-bab's root has a slot for every crossing neuron, whatever the options");
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
    const std::string on = nn.verify(far, 5, cora_vnncomp_options("x")).status;
    ChecksScope off(false);
    check(nn.verify(far, 5, cora_vnncomp_options("x")).status == on,
          "verify: the same with the checks off");
}

/// nn4sys' lindex shape: C cases, each a small box [x_k, x_k + w] with its own two polytopes
/// y <= f(x_k) - d and y >= f(x_k) + d; with sat_case >= 0 that case's first polytope holds f.
NnProperty aux_many_cases(const NeuralNetwork &nn, int64_t C, double w, double d,
                          int64_t sat_case) {
    std::vector<double> lo, hi, A, b;
    NnProperty p;
    for (int64_t k = 0; k < C; ++k) {
        const double x = -1.0 + 2.0 * double(k) / double(C);
        lo.push_back(x);
        hi.push_back(x + w);
        const double f = nn.evaluate(Tensor({{x + w / 2}})).reshape({1}).item();
        const double below = k == sat_case ? f + d : f - d;
        // y <= below and -y <= -(f + d)
        A.insert(A.end(), {1.0, -1.0});
        b.insert(b.end(), {below, -(f + d)});
        p.poly_case.insert(p.poly_case.end(), {k, k});
    }
    p.lo = Tensor::from_data(lo, {C, 1});
    p.hi = Tensor::from_data(hi, {C, 1});
    p.A = Tensor::from_data(A, {2 * C, 1, 1});
    p.b = Tensor::from_data(b, {2 * C, 1});
    return p;
}

/// Many cases of few polytopes (300, 600 rows): the attack's margins per case and the branch and
/// bound in chunks of cases decide as with all polytopes at once, within the memory budget.
void many_cases(const std::string &b) {
    Rng rng(11);
    const NeuralNetwork nn = aux_network(rng, {1, 16, 16, 1}, Activation::relu);
    for (const std::string bounding : {"conzonotope", "zonotope-alpha"}) {
        NnVerifyOptions o = aux_options(bounding, b);
        o.backward_bound = false;
        const NnVerifyResult unsat = nn.verify(aux_many_cases(nn, 300, 1e-4, 0.5, -1), 60, o);
        check(unsat.status == "unsat" && unsat.stats.count("case_chunks") > 0,
              b + " " + bounding + ": 300 cases of two polytopes are unsat, chunk by chunk");
        const NnVerifyResult sat = nn.verify(aux_many_cases(nn, 300, 1e-4, 0.5, 177), 60, o);
        check(sat.status == "sat" && std::abs(sat.x[0] - (-1.0 + 2.0 * 177 / 300.0)) < 2e-4,
              b + " " + bounding + ": a violated case is sat, the point in its box");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        verdicts(b);
        batched_zonotopes(b);
        frad_gradients(b);
        alpha_gradients(b);
        backward_bounds(b);
        // the dual certificates, on rows of very different norms too
        dual_bounds(b);
        dual_unscaled_rows(b);
        relu_chord_both_ends(b);
        dual_bounds_unscaled(b);
        dual_chord_face(b);
        fourier_motzkin(b);
        switches(b);
        deadlines(b);
        many_cases(b);
        targeted_attacks(b);
        relu_bab_verdicts(b);
    });
    checks_and_their_switch();
    return test::finish("neuralNetwork verify");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
