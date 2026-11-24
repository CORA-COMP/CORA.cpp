// nonlinear_sys_dt - a network as a discrete-time system, as CORA's neuralNetwork/nonlinearSysDT
//
// CORA builds a nonlinearSysDT whose state x stays (x+ = x, step size 1, Jacobians I and 0) and
// whose output is the network's: every linear layer adds an input offset u_k to its bias, so
// y = f(x, u) with u the offsets stacked from the first layer, and f(x, 0) is the network. The
// Jacobians are C = calcSensitivity(net, x) and D = [dy / du_1, ..., dy / du_L], the sensitivity of
// the output to the pre-activation of each layer. This port returns the same data in a
// NetworkSystemDT (CORA.cpp has no nonlinearSysDT class). Differences to CORA, in
// docs/cora-differences.md: only dense layers with the activations identity, ReLU, sigmoid and tanh
// (CORA leaves a softmax out of f and drops the scale of an elementwise affine layer), and D is
// exact for a last layer with an activation (CORA appends an identity).
//
// Syntax:   NetworkSystemDT sys = nn.nonlinear_sys_dt();   Tensor y = sys.output(x, u);
// Inputs:   the network; x (in) and u (inputs) for the output
// Outputs:  the system with output, output_jacobian_x (C) and output_jacobian_u (D)
// See also: neural_network.h, sensitivity.cpp
// References: none (CORA's conversion, no paper's algorithm)
//
// CORA: nn/@neuralNetwork/nonlinearSysDT.m
// CORA authors: Laura Luetzow
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_layers.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The output of the network with the offsets u added to the biases; derivative[k] is the
/// derivative of layer k's activation at its pre-activation (a column).
Tensor aux_forward(const NeuralNetwork &net, const Tensor &x, const Tensor &u,
                   std::vector<Tensor> &derivative) {
    Tensor h = x.expand_dims(-1);
    int64_t first = 0;
    derivative.clear();
    for (std::size_t k = 0; k < net.layers().size(); ++k) {
        // the layer with its slice of u added to the bias
        const NeuralNetwork::Layer &L = net.layers()[k];
        const int64_t count = L.b.shape()[0];
        const Tensor z =
            L.W.matmul(h) + L.b.expand_dims(-1) + u.slice(0, first, first + count).expand_dims(-1);
        const PrivDerivative d = priv_activation_derivative(net.activations()[k], z);
        h = d.value;
        derivative.push_back(d.derivative);
        first += count;
    }
    return h;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The system --------------------------------------------------------------------------------------

NetworkSystemDT NeuralNetwork::nonlinear_sys_dt() const { return NetworkSystemDT(*this); }

NetworkSystemDT::NetworkSystemDT(NeuralNetwork net) : net_(std::move(net)) {
    detail::Entry entry;
    CORA_CHECK(net_.is_dense() && net_.is_chain(), Error::not_supported, [] {
        return std::string("nonlinear_sys_dt: the layers must be a chain of matrices");
    });
    // the activations CORA converts
    for (const Activation a : net_.activations())
        CORA_CHECK(a == Activation::identity || a == Activation::relu || a == Activation::sigmoid ||
                       a == Activation::tanh,
                   Error::not_supported, [] {
                       return std::string("nonlinear_sys_dt: the activations are identity, ReLU, "
                                          "sigmoid and tanh (CORA's notSupported)");
                   });
    for (const NeuralNetwork::Layer &L : net_.layers()) num_inputs_ += L.b.shape()[0];
}

Tensor NetworkSystemDT::next_state(const Tensor &x, const Tensor &u) const {
    detail::Entry entry;
    CORA_CHECK(x.shape() == std::vector<int64_t>{num_states()} &&
                   u.shape() == std::vector<int64_t>{num_inputs()},
               Error::dimension_mismatch,
               [] { return std::string("next_state: x must be (states) and u (inputs)"); });
    return x;
}

// The output and its Jacobians --------------------------------------------------------------------

Tensor NetworkSystemDT::output(const Tensor &x, const Tensor &u) const {
    detail::Entry entry;
    CORA_CHECK(x.shape() == std::vector<int64_t>{num_states()} &&
                   u.shape() == std::vector<int64_t>{num_inputs()},
               Error::dimension_mismatch,
               [] { return std::string("output: x must be (states) and u (inputs)"); });
    std::vector<Tensor> derivative;
    return aux_forward(net_, x, u, derivative).squeeze(-1);
}

Tensor NetworkSystemDT::output_jacobian_x(const Tensor &x) const {
    detail::Entry entry;
    return net_.sensitivity(x);
}

Tensor NetworkSystemDT::output_jacobian_u(const Tensor &x, const Tensor &u) const {
    detail::Entry entry;
    CORA_CHECK(x.shape() == std::vector<int64_t>{num_states()} &&
                   u.shape() == std::vector<int64_t>{num_inputs()},
               Error::dimension_mismatch,
               [] { return std::string("output_jacobian_u: x must be (states) and u (inputs)"); });
    // the forward pass gives the derivative of each layer's activation
    std::vector<Tensor> derivative;
    aux_forward(net_, x, u, derivative);
    // backward: B = dy / dh_k from the identity, the block of layer k is B diag(a'_k)
    std::vector<Tensor> blocks(derivative.size());
    Tensor B = x.eye_like(num_outputs());
    for (std::size_t k = derivative.size(); k-- > 0;) {
        blocks[k] = B.multiply(derivative[k].matrix_transpose());
        B = blocks[k].matmul(net_.layers()[k].W);
    }
    return Tensor::concat(blocks, 1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
