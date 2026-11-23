// priv_activation_derivative - the value of an activation and its derivative, neuron by neuron
//
// See also: priv_layers.h, sensitivity.cpp, nonlinear_sys_dt.cpp
// References: none (the derivatives of the activations, CORA's getDf(1))
//
// CORA: nn/@neuralNetwork/calcSensitivity.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "nn/neural_network/private/priv_layers.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// The derivatives ---------------------------------------------------------------------------------

PrivDerivative priv_activation_derivative(Activation activation, const Tensor &z) {
    const Tensor one = z.ones_like(), zero = z.zeros_like();
    Tensor derivative = one, value = z;
    switch (activation) {
    case Activation::identity:
        value = z;
        break;
    case Activation::relu:
        value = z.maximum(0.0);
        derivative = Tensor::where(z.greater(zero), one, zero);
        break;
    case Activation::sigmoid:
        value = z.sigmoid();
        derivative = value.multiply(one - value);
        break;
    case Activation::tanh:
        value = z.tanh();
        derivative = one - value.multiply(value);
        break;
    case Activation::sign:
        // a step: the derivative 0
        value = z.sign();
        derivative = zero;
        break;
    case Activation::floor:
        // steps: the derivative 0
        value = z.floor();
        derivative = zero;
        break;
    case Activation::square:
        // x^2 and 2 x
        value = z.square();
        derivative = z + z;
        break;
    case Activation::cube:
        // x^3 and 3 x^2
        value = z.square().multiply(z);
        derivative = z.square() * 3.0;
        break;
    case Activation::inverse:
        // 1 / x and -1 / x^2
        value = one.divide(z);
        derivative = value.multiply(value).negative();
        break;
    case Activation::sin:
        value = z.sin();
        derivative = z.cos();
        break;
    case Activation::cos:
        value = z.cos();
        derivative = z.sin().negative();
        break;
    case Activation::exp:
        value = z.exp();
        derivative = value;
        break;
    case Activation::reciprocal:
        // 1 / max(x, 1) and -1 / x^2 above 1
        value = one.divide(z.maximum(1.0));
        derivative = Tensor::where(z.greater(one), value.multiply(value).negative(), zero);
        break;
    case Activation::softplus:
        // log(1 + e^x) and the sigmoid
        value = (one + z.exp()).log();
        derivative = z.sigmoid();
        break;
    case Activation::root:
        value = z.sqrt();
        derivative = one.divide(value) * 0.5;
        break;
    case Activation::inv_sqrt_root:
        value = z.pow(-0.5);
        derivative = value.divide(z).negative() * 0.5;
        break;
    case Activation::group_sort:
        CORA_FAIL(Error::not_supported, "sensitivity: a group sort permutes its neurons; only "
                                        "backprop at points takes it");
    case Activation::product:
        CORA_FAIL(Error::not_supported, "sensitivity: a product layer is not neuron by "
                                        "neuron; only NeuralNetwork::verify takes it");
    }
    return {value, derivative};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
