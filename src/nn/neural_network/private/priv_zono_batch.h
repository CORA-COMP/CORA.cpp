// priv_zono_batch - batched zonotopes through a dense network: slots, interval center, factors
//
// The set propagation of CORA's verify (evaluateZonotopeBatch_, prepareForZonoBatchEval), batch
// first: B zonotopes are the center c (B, n), the generator rows G (B, q, n) and the radius dr
// (B, n) of the interval center (undefined when every error has a generator). The input gets q0 =
// min(num_init_gens, n0) generators, an activation layer min(num_approx_err, n_k) error generators
// (approx_error_order), the rest goes to dr. The first q0 factors are the input's, every error
// generator is a factor of its own (ICLR 2026, Sec. 3), so output constraints constrain the input.
// A graph's layer reads an earlier output and adds residuals: their rows are the first factors of
// the layer's (generators add row by row). A ReLU takes CORALean's line, sigmoid and tanh CORA's
// "bounds" or CORALean's "singh", x^2 its chord-tangent parallelogram, a product its line on the
// set (priv_product_line), a softmax's sums also the exps' box; dr maps by |slope|. splits (B, s)
// hold signed neuron ids of earlier splits (-id: >= 0, +id: <= 0) that clamp the bounds; sums nest
// right on lean. One reverse pass gives the F-radius and margin gradients; CROWN reads the records.
//
// Syntax:   ZonoNet net(nn, q0, dn, "bounds", "length");   auto in = net.input_zonotope(x, r);
//           ZonoPropagation out = net.forward(in.c, in.dr, in.G, splits, nullptr, true);
// See also: verify.cpp, priv_verify_zonotack.cpp
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Set-based training for neural network verification",
//        Transactions on Machine Learning Research, 2025.
//    [2] G. Singh, T. Gehr, M. Mirman, M. Puschel, M. Vechev. "Fast and effective robustness
//        certification", NeurIPS 2018.
//
// CORA: nn/@neuralNetwork/evaluateZonotopeBatch_.m
// CORA: nn/@neuralNetwork/prepareForZonoBatchEval.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateZonotopeBatch.m
// CORA authors: Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   23-November-2026
// Reviewed:      ---

#pragma once

#include "nn/neural_network/private/priv_layers.h"

#include <limits>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The sets ----------------------------------------------------------------------------------------

/// What an activation layer saw and made, for the constraints and heuristics of verify: its input
/// set, the unclamped bounds, the error radius, the output set and the neurons with a slot.
struct ZonoLayerRecord {
    int64_t index = 0;    ///< the layer
    int64_t first_id = 1; ///< the global id of its first neuron (ids count from 1)
    Tensor c_in, dr_in, G_in, l, u, er, c_out, dr_out, G_out;
    Tensor slot_neurons, slot_values; ///< (B, dn): the neurons with a slot, their generators
    Tensor grad; ///< (B, n): the F-radius gradient of each neuron's error (0 without a slot)
    /// (B, n): where a given bound of forward replaced the zonotope's (its gradient is 0).
    Tensor held_l, held_u;
};

/// Sound bounds [l, u] (B, n) of the activation layers' inputs by layer (undefined: none), which
/// ZonoNet::forward intersects with the zonotope's.
struct ZonoBounds {
    std::vector<Tensor> l, u;
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

    std::vector<Tensor> W, b, abs_w;        ///< W and abs_w transposed (n_in, n_out), for rows
    std::vector<Tensor> WT, abs_wT;         ///< as stored (n_out, n_in), for the reverse pass
    std::vector<NeuralNetwork::Layer> maps; ///< the layers (a convolution: W undefined above)
    std::vector<Activation> activations;
    std::vector<int64_t> slots, first_ids, widths;
    /// The graph: the output each layer reads (-1: the input) and those it adds; kept[j + 1]
    /// whether output j (0: the input) is read by more than the next layer's own map.
    std::vector<int64_t> sources;
    std::vector<std::vector<NeuralNetwork::Residual>> residuals;
    std::vector<bool> kept;
    bool chain = true;
    std::string poly_method, approx_error_order;
    int64_t n0 = 0, q0 = 0, q = 0, num_outputs = 0, num_neurons = 0;
    bool interval_center = false;
    /// The deadline of the passes: evaluate, sensitivity, forward and the reverse and backward
    /// passes check it once per layer (check_deadline raises, is_deadline tells).
    double deadline = std::numeric_limits<double>::infinity();

    /// Layer k's linear map on rows x (..., in) without the bias, |W|, W^T on rows (..., out),
    /// |W|^T.
    Tensor map(std::size_t k, const Tensor &x) const;
    Tensor map_abs(std::size_t k, const Tensor &x) const;
    Tensor map_t(std::size_t k, const Tensor &y) const;
    Tensor map_abs_t(std::size_t k, const Tensor &y) const;
    /// Residual i of layer k on rows: its map (the identity without one), |map|, or a transpose.
    Tensor residual(std::size_t k, std::size_t i, const Tensor &x, bool absolute,
                    bool transposed) const;
    /// Layer k's linear part without the bias on outs (outs[j + 1]: output j, outs[0]: the input):
    /// its map of its source plus its residuals' (with absolute, |W| of each).
    Tensor linear(std::size_t k, const std::vector<Tensor> &outs, bool absolute = false) const;
    /// The transpose: g (rows of layer k's linear part) to each of its inputs, added into acc.
    void linear_t(std::size_t k, const Tensor &g, std::vector<Tensor> &acc,
                  bool absolute = false) const;

    /// The network at the points (B, n0).
    Tensor evaluate(const Tensor &x) const;

    /// The sensitivity at the points (B, n0).
    ZonoSensitivity sensitivity(const Tensor &x) const;

    /// The box (x, r) (B, n0) with a generator for each of the q0 dimensions dims (B, q0; all in
    /// order if undefined), the rest of the radius in dr.
    ZonoInput input_zonotope(const Tensor &x, const Tensor &r, const Tensor &dims = {}) const;

    /// The output of the batch: every linear layer exactly, every activation by its line, the
    /// errors in the slots and the rest in dr. layer_sens orders the slots ("sensitivity*length").
    /// alpha (per layer) holds a ReLU's slope where it crosses zero; bounds narrow the bounds.
    ZonoPropagation forward(const Tensor &c, const Tensor &dr, const Tensor &G,
                            const Tensor &splits, const std::vector<Tensor> *layer_sens,
                            bool record, const std::vector<Tensor> *alpha = nullptr,
                            const ZonoBounds *bounds = nullptr) const;

    /// The bounds with the clamps of earlier splits (CORA's aux_setLayerBoundsFromPreviousSplits).
    std::pair<Tensor, Tensor> clamp(int64_t first_id, const Tensor &l, const Tensor &u,
                                    const Tensor &splits) const;
};

/// The net of a batch of q input generators as CORA's evaluateZonotopeBatch sets it up: every input
/// generator a factor, the errors in slots and, past num_approx_err, in the radius dr.
ZonoNet priv_batch_net(const NeuralNetwork &nn, int64_t q, const ZonotopeBatchOptions &options);

// Gradients and bounds ----------------------------------------------------------------------------

/// The gradients of the input's center (B, n0) and generators (B, q0, n0) (the fields c and G) from
/// those of a recorded propagation's output: the lines fixed (held slopes, CORA's backprop without
/// exact_backprop), or exact through the lines' bounds (exact_backprop).
ZonotopeBatch priv_backprop_set(const ZonoPropagation &out, const ZonoNet &net, const Tensor &g_c,
                                const Tensor &g_gen, const Tensor &g_dr, bool exact);

/// The F-radius sqrt(sum G^2)'s gradient in each generator's value (the input's, every slot's):
/// exact (CORA's exact_backprop) through the lines, else the slopes held, |G_g|^2 / (v_g F).
/// Sets the records' grad and returns the input's (B, q0).
Tensor priv_frad_gradients(ZonoPropagation &out, const ZonoInput &in, const ZonoNet &net,
                           const Tensor &splits, bool exact);

/// The gradient in the ReLU slopes alpha (per layer, 0 off the crossing) of a margin whose partial
/// derivatives in the output's c, G and dr are the seeds (g_dr undefined without dr), exact through
/// the bounds of the later layers.
std::vector<Tensor> priv_alpha_gradients(const ZonoPropagation &out, const ZonoNet &net,
                                         const Tensor &splits, const std::vector<Tensor> &alpha,
                                         const Tensor &g_c, const Tensor &g_gen,
                                         const Tensor &g_dr);

/// The ReLU slopes of a recorded propagation's lines, u / (u - l) at the clamped bounds where a
/// neuron crosses zero (the start of the optimization), per layer; undefined for the others.
std::vector<Tensor> priv_alpha_start(const ZonoPropagation &out, const ZonoNet &net,
                                     const Tensor &splits);

/// CROWN's lower bounds (B, R) of the rows (R, m) times the output over the input boxes (x, rad),
/// with each activation's linear bounds at the record's clamped bounds: a ReLU's slope alpha (else
/// CROWN's 0 or 1), a sigmoid or tanh between the zonotope line's two parallels.
Tensor priv_backward_lower(const ZonoPropagation &out, const ZonoNet &net, const Tensor &splits,
                           const Tensor &x, const Tensor &rad, const Tensor &rows,
                           const std::vector<Tensor> *alpha = nullptr);

/// CROWN's lower bounds as priv_backward_lower with a ReLU's lower slope per row optimized
/// (alpha-CROWN): `iterations` projected Adam steps of size `step` on the slopes in [0, 1], the
/// record's bounds held, none after the deadline (now_seconds); each row's best bound.
Tensor priv_backward_alpha(const ZonoPropagation &out, const ZonoNet &net, const Tensor &splits,
                           const Tensor &x, const Tensor &rad, const Tensor &rows,
                           int64_t iterations, double step,
                           double deadline = std::numeric_limits<double>::infinity());

/// CROWN's bounds of the activation layers' inputs over the boxes (x, rad) of a record without
/// splits, intersected with its bounds (priv_zono_bounds.cpp): `iterations` alpha-CROWN steps of
/// size `step` per row, the rows in chunks of about max_bytes, the record's after the deadline.
ZonoBounds priv_crown_bounds(const ZonoPropagation &out, const ZonoNet &net, const Tensor &x,
                             const Tensor &rad, int64_t iterations, double step, int64_t max_bytes,
                             double deadline = std::numeric_limits<double>::infinity());

/// One projected Adam step t >= 1 of size `step` on the slopes alpha in [0, 1] (undefined ones and
/// those without a gradient g are held), the moments m1, m2 updated.
void priv_adam_slopes(std::vector<Tensor> &alpha, std::vector<Tensor> &m1, std::vector<Tensor> &m2,
                      const std::vector<Tensor> &g, int64_t t, double step);

/// The rows C beta <= d (B, p, q) on the factors of a recorded propagation (priv_verify_dual.cpp):
/// one per split of splits, the three faces of the num_relu crossing ReLU neurons of largest error.
/// Undefined if there is none.
std::pair<Tensor, Tensor> priv_dual_rows(const ZonoPropagation &out, const ZonoNet &net,
                                         const Tensor &splits, int64_t num_relu);

// The split rows of a frozen zonotope -------------------------------------------------------------

/// The split rows of one recorded propagation (B = 1), once for every neuron it can split: the
/// ReLU neurons whose input crosses 0. A branch and bound on this frozen zonotope splits them in a
/// fixed order, so the nodes of one depth share the rows and differ in their signs only.
struct PrivSplitTable {
    Tensor rows;  ///< (U, q): the neuron's input generators, G_in's column padded to q
    Tensor c, dr; ///< (U): its input's center and the radius of its error term
    Tensor ids;   ///< (U): its global id
    Tensor er;    ///< (U): the error of its line (the order of the splits: largest first)
    Tensor rows_out, c_out, dr_out; ///< (U, q), (U), (U): its output's, as rows, c and dr
    Tensor col;                     ///< (U): the factor of its error's slot (-1: none)
    Tensor l, u;                    ///< (U): its input's bounds (the faces of an unsplit one)
    int64_t slotless = 0;           ///< crossing neurons whose error has no slot (only in dr)
};

/// The table of the crossing ReLU neurons of out (B = 1), in global id order.
PrivSplitTable priv_split_table(const ZonoPropagation &out, const ZonoNet &net);

/// The Lagrangian lower bound (N, K) of w (K, q) on the nodes of signs s (N, d) over table rows
/// (R, c, dr) (priv_verify_dual.cpp): Adam from start (N, K, d; undefined: 0), best gets lambda.
Tensor priv_dual_lower_signs(const Tensor &w, const Tensor &R, const Tensor &c, const Tensor &dr,
                             const Tensor &s, int64_t iterations, double step, DType dtype,
                             const Tensor &start = Tensor(), Tensor *best = nullptr);

/// The certificate (B, R) of lambda (B, R, p): -|w + lambda C|_1 - lambda d at max(lambda, 0), at
/// least -|w|_1, on the rows as given (weak duality, CORALean's ConZonotope.dualBound); w (B, R,
/// q) or (R, q), C (B, p, q), d (B, p); sum gets w + lambda C (its minimizer: -sign).
Tensor priv_dual_certificate(const Tensor &w, const Tensor &C, const Tensor &d,
                             const Tensor &lambda, Tensor *sum = nullptr);

/// priv_dual_lower_signs with the phases: inactive (s = 1) y = 0, active (-1) y = z, unsplit (0)
/// the triangle's faces over [l, u]; t: the d neurons' rows, start (N, K, 3d); minimizer: the
/// factors (N, K, q) minimizing the Lagrangian of the best multipliers.
Tensor priv_dual_lower_phases(const Tensor &w, const PrivSplitTable &t, const Tensor &s,
                              int64_t iterations, double step, DType dtype,
                              const Tensor &start = Tensor(), Tensor *best = nullptr,
                              Tensor *minimizer = nullptr);

/// The Lagrangian lower bound (B, R) of w (B, R, q) times beta over [-1, 1]^q with C beta <= d,
/// lambda by `iterations` projected Adam steps in dtype; evaluated again in w's dtype.
Tensor priv_dual_lower(const Tensor &w, const Tensor &C, const Tensor &d, int64_t iterations,
                       double step, DType dtype);

/// The offset of a crossing ReLU's chord above at the stored slope `chord`: max(-chord l,
/// u - chord u), so that the line holds at both ends (CORALean !971).
Tensor priv_relu_chord_offset(const Tensor &l, const Tensor &u, const Tensor &chord);

/// A ReLU's line base with the slope a in [0, 1] where it crosses zero: relu(x) - a x lies in
/// [0, h] with h = max(-a l, u - a u), the offset and the radius h / 2 (CORALean's reluRelaxation).
PrivLine priv_alpha_line(const PrivLine &base, const Tensor &l, const Tensor &u, const Tensor &a);

/// The box [lo, hi] (B, n_in) through layer k's map: the center's image +- |W| times the radius,
/// widened by n_in + 4 ulps of the largest term (a softmax's sums, CORALean's softmaxSumBounds).
std::pair<Tensor, Tensor> priv_box_image(const ZonoNet &net, std::size_t k, const Tensor &lo,
                                         const Tensor &hi);

/// The line of CORA's poly_method "bounds" (aux_imgEncBatch) or CORALean's "singh" over [l, u].
PrivLine priv_verify_line(Activation activation, const Tensor &l, const Tensor &u,
                          const std::string &poly_method);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
