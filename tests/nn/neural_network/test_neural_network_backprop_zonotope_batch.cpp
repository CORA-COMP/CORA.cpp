// test_neural_network_backprop_zonotope_batch - the gradient of a loss on batched zonotopes
//
// Where every neuron is stable the lines are the layers' maps, so the held and the exact gradients
// are the chain rule of the network (backprop of the center, of each generator). Where neurons
// cross zero the exact gradients are those of the loss by finite differences, in the input's
// centers and generators, with and without an interval center; the held gradients are the chain
// rule with the slopes of the lines fixed. A network of convolutions has its dense twin's
// gradients, and a wrong size is refused. (CORALean's neuralNetwork.backwardBound is CROWN's
// bound, not a gradient: there is no oracle test.)
//
// PROPERTY: nn/@neuralNetwork/backpropZonotopeBatch.m
// PROPERTY: nn/@neuralNetwork/backpropZonotopeBatch_.m
// PROPERTY: nn/layers/nonlinear/@nnActivationLayer/backpropZonotopeBatch.m
// CORA: nn/@neuralNetwork/backpropZonotopeBatch.m
// CORA: nn/@neuralNetwork/backpropZonotopeBatch_.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/backpropZonotopeBatch.m
// CORA authors: Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

#include <cmath>
#include <functional>
#include <limits>

using namespace cora;
using test::check;
using Layer = NeuralNetwork::Layer;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Random values of a shape.
Tensor aux_random(Rng &rng, const std::vector<int64_t> &shape, double scale) {
    int64_t n = 1;
    for (const int64_t s : shape) n *= s;
    std::vector<double> v(static_cast<std::size_t>(n));
    rng.normal(v.data(), v.size(), scale);
    return Tensor::from_data(v, shape);
}

/// Two dense layers on three inputs with the activation a, then the identity.
NeuralNetwork aux_network(Rng &rng, Activation a) {
    return NeuralNetwork({{aux_random(rng, {4, 3}, 1.0), aux_random(rng, {4}, 0.3)},
                          {aux_random(rng, {2, 4}, 1.0), aux_random(rng, {2}, 0.3)}},
                         {a, Activation::identity});
}

/// The loss <w_c, c'> + <w_gen, G'> + <w_dr, dr'> of the batch through the network.
double aux_loss(const NeuralNetwork &nn, const Tensor &c, const Tensor &G, const Tensor &w_c,
                const Tensor &w_gen, const Tensor &w_dr, const ZonotopeBatchOptions &o) {
    const ZonotopeBatch Y = nn.evaluate_zonotope_batch(c, G, o);
    double loss = Y.c.multiply(w_c).sum().item() + Y.G.multiply(w_gen).sum().item();
    if (Y.dr.defined()) loss += Y.dr.multiply(w_dr).sum().item();
    return loss;
}

/// The derivative of the loss in each entry of x (c or G, the other one fixed) by central
/// differences.
std::vector<double> aux_numeric(const std::function<double(const Tensor &)> &loss,
                                const Tensor &x) {
    std::vector<double> v = x.data(), g(v.size());
    const double h = 1e-6;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double keep = v[i];
        v[i] = keep + h;
        const double up = loss(Tensor::from_data(v, x.shape()));
        v[i] = keep - h;
        const double down = loss(Tensor::from_data(v, x.shape()));
        v[i] = keep;
        g[i] = (up - down) / (2 * h);
    }
    return g;
}

/// The batch: B centers, q generators of a radius that keeps ReLU neurons stable or not.
struct Batch {
    Tensor c, G;
};

Batch aux_batch(Rng &rng, double radius) {
    return {aux_random(rng, {2, 3}, 0.5), aux_random(rng, {2, 3, 3}, radius)};
}

/// The chain rule where every neuron is stable: the network's backprop of each row.
void stable(const std::string &b) {
    Rng rng(3);
    const NeuralNetwork nn = aux_network(rng, Activation::relu);
    const Batch in = aux_batch(rng, 1e-4);
    const ZonotopeBatch Y = nn.evaluate_zonotope_batch(in.c, in.G);
    const Tensor grad_c = aux_random(rng, {2, 2}, 1.0),
                 grad_gen = aux_random(rng, Y.G.shape(), 1.0);
    const Tensor want_c = nn.backprop(in.c, grad_c);
    // the generators' gradients: each row of grad_gen[:, j] through the same Jacobian
    std::vector<double> want_g;
    for (int64_t j = 0; j < 3; ++j) {
        const std::vector<double> row =
            nn.backprop(in.c, grad_gen.slice(1, j, j + 1).reshape({2, 2})).data();
        want_g.insert(want_g.end(), row.begin(), row.end());
    }
    for (const bool exact : {false, true}) {
        const ZonotopeBatch g =
            nn.backprop_zonotope_batch(in.c, in.G, grad_c, grad_gen, Tensor(), {}, exact);
        // the layout of want_g is (j, b, n), that of g.G is (b, j, n)
        const Tensor swapped = g.G.permute_dims({1, 0, 2});
        check(test::close(g.c, want_c, 1e-9) && test::close(swapped, want_g, 1e-9),
              b + ": stable neurons, the gradients are the chain rule (exact " +
                  std::to_string(exact) + ")");
    }
}

/// The exact gradients against central differences, with every neuron a slot or the rest in dr.
void exact(const std::string &b) {
    Rng rng(4);
    for (const Activation a : {Activation::relu, Activation::tanh, Activation::sigmoid}) {
        for (const double slots : {std::numeric_limits<double>::infinity(), 2.0}) {
            const NeuralNetwork nn = aux_network(rng, a);
            const Batch in = aux_batch(rng, 0.4);
            ZonotopeBatchOptions o;
            o.num_approx_err = slots;
            const ZonotopeBatch Y = nn.evaluate_zonotope_batch(in.c, in.G, o);
            const Tensor w_c = aux_random(rng, Y.c.shape(), 1.0),
                         w_gen = aux_random(rng, Y.G.shape(), 1.0),
                         w_dr = Y.dr.defined() ? aux_random(rng, Y.dr.shape(), 1.0) : Tensor();
            // the exact gradients, and the differences of the loss in the centers and generators
            const ZonotopeBatch g =
                nn.backprop_zonotope_batch(in.c, in.G, w_c, w_gen, w_dr, o, true);
            const std::vector<double> num_c = aux_numeric(
                                          [&](const Tensor &c) {
                                              return aux_loss(nn, c, in.G, w_c, w_gen, w_dr, o);
                                          },
                                          in.c),
                                      num_g = aux_numeric(
                                          [&](const Tensor &G) {
                                              return aux_loss(nn, in.c, G, w_c, w_gen, w_dr, o);
                                          },
                                          in.G);
            check(test::close(g.c, num_c, 1e-5) && test::close(g.G, num_g, 1e-5),
                  b + ": exact gradients are the finite differences (activation " +
                      std::to_string(int(a)) + ", slots " + std::to_string(slots) + ")");
        }
    }
}

/// With the lines held a ReLU neuron's slope is that of its line, u / (u - l) where it crosses 0.
void held(const std::string &b) {
    Rng rng(6);
    const NeuralNetwork nn = aux_network(rng, Activation::relu);
    const Batch in = aux_batch(rng, 0.5);
    const Tensor grad_c = aux_random(rng, {2, 2}, 1.0),
                 grad_gen = aux_random(rng, {2, 3 + 4, 2}, 1.0);
    const ZonotopeBatch g = nn.backprop_zonotope_batch(in.c, in.G, grad_c, grad_gen);
    // the slopes of the hidden layer from its bounds: c W^T + b -+ the sum of |G W^T|
    const Layer &L1 = nn.layers()[0], &L2 = nn.layers()[1];
    const Tensor mid = L1.rows(in.c) + L1.b, rad = L1.rows(in.G).abs().sum(1);
    const Tensor lo = mid - rad, u = mid + rad, zero = lo.zeros_like();
    const Tensor slope = Tensor::where(lo.greater_equal(zero), lo.ones_like(),
                                       Tensor::where(u.less_equal(zero), zero, u.divide(u - lo)));
    // the center and the first three generators through W2^T, the slope and W1^T
    const Tensor want_c = L1.rows_t(slope.multiply(L2.rows_t(grad_c)));
    const Tensor want_G =
        L1.rows_t(slope.expand_dims(1).multiply(L2.rows_t(grad_gen.slice(1, 0, 3))));
    check(test::close(g.c, want_c, 1e-9) && test::close(g.G, want_G, 1e-9),
          b + ": held gradients are the chain rule with the lines' slopes");
    // the gradients of the error generators stop: only the first q rows matter
    check(g.G.shape() == in.G.shape() && g.c.shape() == in.c.shape(),
          b + ": the gradients have the input's shapes");
}

/// A network of convolutions has its dense twin's gradients; wrong sizes are refused.
void networks(const std::string &b) {
    Rng rng(8);
    NeuralNetwork::Conv2d spec;
    spec.pads = {1, 1, 0, 0};
    const Layer c1 = Layer::conv2d(aux_random(rng, {2, 1, 2, 2}, 0.7),
                                   aux_random(rng, {2 * 4 * 4}, 0.2), {1, 4, 4}, spec);
    const Layer d2{aux_random(rng, {2, 32}, 0.5), aux_random(rng, {2}, 0.1)};
    const NeuralNetwork conv({c1, d2}, {Activation::tanh, Activation::identity});
    const NeuralNetwork dense({Layer{c1.dense(), c1.b}, d2},
                              {Activation::tanh, Activation::identity});
    const Tensor c = aux_random(rng, {2, 16}, 0.5), G = aux_random(rng, {2, 3, 16}, 0.2);
    const ZonotopeBatch Y = dense.evaluate_zonotope_batch(c, G);
    const Tensor grad_c = aux_random(rng, Y.c.shape(), 1.0),
                 grad_gen = aux_random(rng, Y.G.shape(), 1.0);
    const ZonotopeBatch gd = dense.backprop_zonotope_batch(c, G, grad_c, grad_gen, Tensor(), {},
                                                           true),
                        gv = conv.backprop_zonotope_batch(c, G, grad_c, grad_gen, Tensor(), {},
                                                          true);
    check(test::close(gv.c, gd.c, 1e-10) && test::close(gv.G, gd.G, 1e-10),
          b + ": the convolutions' gradients are the dense network's");
    // a wrong size of the sets or of the gradients
    check(
        test::raises([&] { dense.backprop_zonotope_batch(c, G, grad_c.slice(1, 0, 1), grad_gen); },
                     Error::dimension_mismatch),
        b + ": a gradient of the wrong size");
    check(
        test::raises([&] { dense.backprop_zonotope_batch(c.slice(1, 0, 3), G, grad_c, grad_gen); },
                     Error::dimension_mismatch),
        b + ": a center of the wrong size");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        stable(b);
        exact(b);
        held(b);
        networks(b);
    });
    // a valid call with the checks off
    {
        ChecksScope off(false);
        Rng rng(9);
        const NeuralNetwork nn = aux_network(rng, Activation::relu);
        const Batch in = aux_batch(rng, 0.1);
        const ZonotopeBatch Y = nn.evaluate_zonotope_batch(in.c, in.G);
        const ZonotopeBatch g =
            nn.backprop_zonotope_batch(in.c, in.G, Y.c.ones_like(), Y.G.ones_like());
        check(g.c.shape() == in.c.shape(), "checks off: a valid call");
    }
    return test::finish("neuralNetwork backprop zonotope batch");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
