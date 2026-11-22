// priv_zono_batch - batched zonotopes through a dense network: slots, interval center, factors
//
// The set propagation of CORA's verify (evaluateZonotopeBatch_, prepareForZonoBatchEval), batch
// first: a batch of B zonotopes is the center c (B, n), the generators as rows G (B, q, n) and the
// radius dr (B, n) of the interval center (undefined when every error has a generator). The input
// gets q0 = min(num_init_gens, n0) generators, an activation layer min(num_approx_err, n_k) error
// generators in the order of approx_error_order, the rest goes to dr. The output's first q0
// factors are the input's and every error generator is a factor of its own (ICLR 2026, Sec. 3),
// so constraints on the output are constraints on the input. A ReLU takes CORALean's line, sigmoid
// and tanh CORA's "bounds" or CORALean's "singh". splits (B, s) hold signed neuron ids of earlier
// splits (-id: at least 0, +id: at most 0, 0: none), which clamp the bounds. On the lean backend
// every sum nests to the right, as CORALean's: the sets are the oracle's bit for bit. The exact
// F-radius gradient (exact_backprop) passes back through the slopes, offsets and errors too.
//
// Syntax:   ZonoNet net(nn, q0, dn, "bounds", "length");   auto in = net.input_zonotope(x, r);
//           ZonoPropagation out = net.forward(in.c, in.dr, in.G, splits, nullptr, true);
// See also: verify.cpp, priv_verify_zonotack.cpp
//
// CORA: nn/@neuralNetwork/evaluateZonotopeBatch_.m
// CORA: nn/@neuralNetwork/prepareForZonoBatchEval.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateZonotopeBatch.m
// CORA authors: Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "nn/neural_network/private/priv_layers.h"

#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// What an activation layer saw and made, for the constraints and heuristics of verify: its input
/// set, the unclamped bounds, the error radius, the output set and the neurons with a slot.
struct ZonoLayerRecord {
    int64_t index = 0;    ///< the layer
    int64_t first_id = 1; ///< the global id of its first neuron (ids count from 1)
    Tensor c_in, dr_in, G_in, l, u, er, c_out, dr_out, G_out;
    Tensor slot_neurons, slot_values; ///< (B, dn): the neurons with a slot, their generators
    Tensor grad; ///< (B, n): the F-radius gradient of each neuron's error (0 without a slot)
};

/// The output zonotope and, when recorded, every activation layer's record.
struct ZonoPropagation {
    Tensor c, dr, G;
    std::vector<ZonoLayerRecord> layers;
};

/// The input zonotope: the set and the input dimension of each generator (B, q0).
struct ZonoInput {
    Tensor c, dr, G, dims, values;
};

/// CORA's sensitivity at points: the input's max |dy/dx| (B, n0) floored, per activation layer
/// sum and max over the outputs of |dy/dh| (B, n_k), and the Jacobian (B, m, n0).
struct ZonoSensitivity {
    Tensor input, jacobian;
    std::vector<Tensor> layer_sum, layer_max;
};

class ZonoNet {
  public:
    ZonoNet(const NeuralNetwork &net, double num_init_gens, double num_approx_err,
            std::string poly_method, std::string approx_error_order);

    std::vector<Tensor> W, b, abs_w; ///< W and abs_w transposed (n_in, n_out), for rows
    std::vector<Tensor> WT, abs_wT;  ///< as stored (n_out, n_in), for the reverse pass
    std::vector<Activation> activations;
    std::vector<int64_t> slots, first_ids, widths;
    std::string poly_method, approx_error_order;
    int64_t n0 = 0, q0 = 0, q = 0, num_outputs = 0, num_neurons = 0;
    bool interval_center = false;

    /// The network at the points (B, n0).
    Tensor evaluate(const Tensor &x) const;

    /// The sensitivity at the points (B, n0).
    ZonoSensitivity sensitivity(const Tensor &x) const;

    /// The box (x, r) (B, n0) with a generator for each of the q0 dimensions dims (B, q0; all in
    /// order if undefined), the rest of the radius in dr.
    ZonoInput input_zonotope(const Tensor &x, const Tensor &r, const Tensor &dims = {}) const;

    /// The output of the batch: every linear layer exactly, every activation by its line, the
    /// errors in the slots and the rest in dr. layer_sens orders the slots ("sensitivity*length").
    ZonoPropagation forward(const Tensor &c, const Tensor &dr, const Tensor &G,
                            const Tensor &splits, const std::vector<Tensor> *layer_sens,
                            bool record) const;

    /// The bounds with the clamps of earlier splits (CORA's aux_setLayerBoundsFromPreviousSplits).
    std::pair<Tensor, Tensor> clamp(int64_t first_id, const Tensor &l, const Tensor &u,
                                    const Tensor &splits) const;
};

/// The F-radius sqrt(sum G^2)'s gradient in each generator's value (the input's, every slot's):
/// exact (CORA's exact_backprop) through the lines, else the slopes held, |G_g|^2 / (v_g F).
/// Sets the records' grad and returns the input's (B, q0).
Tensor priv_frad_gradients(ZonoPropagation &out, const ZonoInput &in, const ZonoNet &net,
                           const Tensor &splits, bool exact);

/// The line of CORA's poly_method "bounds" (aux_imgEncBatch) or CORALean's "singh" over [l, u].
PrivLine priv_verify_line(Activation activation, const Tensor &l, const Tensor &u,
                          const std::string &poly_method);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
