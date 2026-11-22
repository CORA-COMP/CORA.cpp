// priv_verify_naive - the bounding "zonotope": the network's output zonotope and input splits
//
// A box is bounded by the output zonotope of NeuralNetwork::evaluate (a generator per input
// dimension of positive radius, each activation by its parallelogram, DeepZ): the lower bound of
// a row is A c - sum |A G| - b. The branch and bound takes batches from the front (breadth first)
// or the back of the queue, checks the centers of the open boxes, and cuts each open box into
// num_splits equal parts along the dimension of its input_split_heuristic: "largest-radius", or
// "most-sensitive-input-radius" (the radius times the objective's gradient at the center).
//
// Syntax:   register_bounding("zonotope", ...)   (the built-in, by name)
// See also: priv_verify.h, priv_verify_zonotack.cpp (the bounding "conzonotope")
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Out of the shadows: exploring a latent space for
//        neural network verification", ICLR 2026 (arXiv:2505.17854).
//
// CORA: nn/@neuralNetwork/verify.m
// CORA authors: Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_verify.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Splits ------------------------------------------------------------------------------------------

/// The host values of an argmax per row.
std::vector<int64_t> aux_argmax_rows(const Tensor &score) {
    std::vector<int64_t> out;
    for (const double v : score.argmax(1).data()) out.push_back(int64_t(v));
    return out;
}

/// Every box cut into `pieces` equal parts along its dimension dims[i].
void aux_split(Tensor &lo, Tensor &hi, Tensor &cases, const std::vector<int64_t> &dims,
               int64_t pieces) {
    const int64_t B = lo.shape()[0], n = lo.shape()[1];
    std::vector<double> hot(std::size_t(B * n), 0.0);
    for (int64_t i = 0; i < B; ++i) hot[std::size_t(i * n + dims[std::size_t(i)])] = 1.0;
    const Tensor mask = Tensor::like(lo, hot, {B, n});
    const Tensor width = (hi - lo) / double(pieces);
    std::vector<Tensor> los, his, cs;
    for (int64_t k = 0; k < pieces; ++k) {
        los.push_back(Tensor::where(mask, lo + width * double(k), lo));
        his.push_back(k == pieces - 1 ? hi : Tensor::where(mask, lo + width * double(k + 1), hi));
        cs.push_back(cases);
    }
    lo = Tensor::concat(los, 0);
    hi = Tensor::concat(his, 0);
    cases = Tensor::concat(cs, 0);
}

// The bounding ------------------------------------------------------------------------------------

/// The output zonotope of NeuralNetwork::evaluate, with the naive branch and bound.
class ZonotopeBounding : public Bounding {
  public:
    explicit ZonotopeBounding(const NnVerifyOptions &o) : o_(o) {}

    bool prepare(NnProblem &problem) override {
        int64_t widest = problem.net().input_dim(), q = problem.net().input_dim();
        const auto &layers = problem.net().layers();
        for (std::size_t k = 0; k < layers.size(); ++k) {
            widest = std::max(widest, layers[k].W.shape()[0]);
            if (k + 1 < layers.size()) q += layers[k].W.shape()[0];
        }
        box_bytes_ = 8 * 3 * q * widest;
        return true;
    }

    /// The lower bounds (B, P, K) of A y - b over each box.
    Tensor lower(const NnProblem &problem, const Tensor &lo, const Tensor &hi) const {
        const int64_t B = lo.shape()[0], n = lo.shape()[1];
        const Tensor c = (lo + hi) * 0.5, r = (hi - lo) * 0.5;
        // a generator for each dimension that some box of the batch perturbs
        std::vector<int64_t> active;
        const std::vector<double> wide = r.max(0).data();
        for (int64_t i = 0; i < n; ++i)
            if (wide[std::size_t(i)] > 0) active.push_back(i);
        const int64_t g = int64_t(active.size());
        std::vector<double> sel(std::size_t(g * n), 0.0);
        for (int64_t j = 0; j < g; ++j) sel[std::size_t(j * n + active[std::size_t(j)])] = 1.0;
        const Tensor G = Tensor::like(lo, sel, {1, g, n}).multiply(r.expand_dims(1));
        // the output zonotope, and the lower bounds of its rows
        const Zonotope Y = problem.net().evaluate(Zonotope(c, G));
        const Tensor A = problem.A, b = problem.b;
        const int64_t P = A.shape()[0], K = A.shape()[1];
        const Tensor AT = A.reshape({P * K, A.shape()[2]}).matrix_transpose();
        const Tensor Ac = Y.center().matmul(AT);
        const Tensor AG = Y.generators().matmul(AT).abs().sum(-2).reshape({B, P * K});
        return (Ac - AG - b.reshape({P * K})).reshape({B, P, K});
    }

    std::string branch_and_bound(NnProblem &problem, double deadline,
                                 std::map<std::string, double> &stats,
                                 AttackResult &found) override {
        Tensor lo = problem.outer_lo, hi = problem.outer_hi;
        Tensor cases = problem.box_case;
        const int64_t batch = std::max<int64_t>(1, problem.memory_budget / box_bytes_);
        const InputSplitHeuristic &split = priv_input_split_heuristic(o_.input_split_heuristic);
        int64_t iteration = 0;
        double last = 0;
        while (lo.shape()[0] > 0 && now_seconds() + last < deadline &&
               iteration < o_.max_verif_iter) {
            const double started = now_seconds();
            // the next batch, from the front (breadth first) or the back (depth first)
            const int64_t N = lo.shape()[0], take = std::min(batch, N);
            if (o_.verify_dequeue_type != "front" && o_.verify_dequeue_type != "back")
                CORA_FAIL(Error::wrong_field_value,
                          "verify: the zonotope bounding takes the verify_dequeue_type 'front' or "
                          "'back', not '" +
                              o_.verify_dequeue_type + "'");
            const bool front = o_.verify_dequeue_type == "front";
            const int64_t a = front ? 0 : N - take, e = front ? take : N;
            Tensor blo = lo.slice(0, a, e), bhi = hi.slice(0, a, e), bcase = cases.slice(0, a, e);
            const int64_t ka = front ? take : 0, ke = front ? N : N - take;
            lo = lo.slice(0, ka, ke);
            hi = hi.slice(0, ka, ke);
            cases = cases.slice(0, ka, ke);
            // a box is verified when every polytope of its case is excluded by a row
            const Tensor decided =
                lower(problem, blo, bhi).greater(blo.full_like({1}, o_.verified_margin)).any(2);
            const Tensor excluded =
                decided.maximum(bcase.expand_dims(1).not_equal(problem.poly_case.expand_dims(0)));
            const std::vector<int64_t> open = (1.0 - excluded.min(1)).nonzero();
            stats["boxes"] += double(blo.shape()[0]);
            ++iteration;
            last = now_seconds() - started;
            if (open.empty()) continue;
            blo = blo.take(open, 0);
            bhi = bhi.take(open, 0);
            bcase = bcase.take(open, 0);
            // a cheap falsification of the open boxes: their centers
            if (priv_check_points(problem, (blo + bhi) * 0.5, bcase, found)) {
                stats["iterations"] = double(iteration);
                return "sat";
            }
            aux_split(blo, bhi, bcase, split(problem, blo, bhi, bcase), o_.num_splits);
            lo = Tensor::concat({lo, blo}, 0);
            hi = Tensor::concat({hi, bhi}, 0);
            cases = Tensor::concat({cases, bcase}, 0);
        }
        stats["iterations"] = double(iteration);
        return lo.shape()[0] == 0 ? "unsat" : "unknown";
    }

  private:
    NnVerifyOptions o_;
    int64_t box_bytes_ = 1;
};

} // namespace


// ===========================================  MAIN  =========================================== //

// The registry ------------------------------------------------------------------------------------

void priv_register_naive() {
    register_bounding(
        "zonotope", [](const NnVerifyOptions &o) { return std::make_unique<ZonotopeBounding>(o); });
    // the dimension of the largest radius
    register_input_split_heuristic("largest-radius",
                                   [](NnProblem &, const Tensor &lo, const Tensor &hi,
                                      const Tensor &) { return aux_argmax_rows(hi - lo); });
    // the radius times the objective's gradient at the center (CORA's sensitivity)
    register_input_split_heuristic(
        "most-sensitive-input-radius",
        [](NnProblem &problem, const Tensor &lo, const Tensor &hi, const Tensor &cases) {
            const DType dtype = problem.lo.dtype();
            const Tensor x = ((lo + hi) * 0.5).to(dtype);
            const Tensor g = problem.objective_gradient(x, cases.to(dtype));
            return aux_argmax_rows((hi - lo).to(dtype).multiply(g.abs()));
        });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
