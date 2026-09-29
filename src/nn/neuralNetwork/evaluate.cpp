// evaluate - the outputs of a network at points, and an enclosure of them over a set
//
// A ReLU y = max(x, 0) over the range [l, u] of a neuron is exact where l >= 0 (y = x) or u <= 0
// (y = 0). Where the range crosses 0 it is enclosed by the parallelogram of slope u / (u - l):
//     y in lambda x + mu +- mu,   lambda = u / (u - l),   mu = -lambda l / 2,
// which is the smallest enclosure of one slope. The slope and the offset come from the ends of
// the ranges by tensor operations, so every neuron is handled at once and the result is
// differentiable.
//
// Syntax:   y = nn.evaluate(x);   Y = nn.evaluate(X);
// Inputs:   x - points, columns (in, N);  X - a zonotope of dimension in (or a batch)
// Outputs:  y - the outputs (out, N);  Y - a zonotope that contains the outputs of all points of X
// See also: neuralNetwork.h, Zonotope::interval

#include "nn/neuralNetwork/neuralNetwork.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A tensor of the shape and backend of `like` filled with `value`.
Tensor aux_filled(const Tensor &like, double value) {
    int64_t count = 1;
    for (const int64_t d : like.shape()) count *= d;
    return Tensor::like(like, std::vector<double>(count, value), like.shape());
}

/// A zonotope that contains max(x, 0) for every x in Z: the enclosure above for every neuron, with
/// one new generator per neuron (a zero one where the neuron is stable).
Zonotope aux_relu(const Zonotope &Z) {
    const Interval range = Z.interval();
    const Tensor upper = range.sup.pos(), lower = range.inf.neg();  // max(u, 0) and min(l, 0)
    // upper - lower is 0 only for l = u = 0; the tiny term keeps the slope 0 there, not NaN.
    const Tensor slope = upper.div(upper - lower + aux_filled(upper, 1e-300));
    const Tensor mu = slope.mul(lower) * -0.5;
    return {slope.mul(Z.c) + mu, Tensor::catLast({slope.mul(Z.G), mu.diag()})};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor NeuralNetwork::evaluate(const Tensor &x) const {
    if (x.shape().size() != 2 || x.shape()[0] != inputDim())
        throw std::invalid_argument("cora: the points of this network are columns (" +
                                    std::to_string(inputDim()) + ", N)");
    Tensor h = x;
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        h = layers_[k].W.matmul(h) + layers_[k].b;
        if (k + 1 < layers_.size()) h = h.pos();
    }
    return h;
}

Zonotope NeuralNetwork::evaluate(const Zonotope &X) const {
    if (X.dim() != inputDim())
        throw std::invalid_argument("cora: the set has dimension " + std::to_string(X.dim()) +
                                    " but the network takes " + std::to_string(inputDim()));
    Zonotope Z = X;
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        Z = Z.mtimes(layers_[k].W) + layers_[k].b;
        if (k + 1 < layers_.size()) Z = aux_relu(Z);
    }
    return Z;
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
