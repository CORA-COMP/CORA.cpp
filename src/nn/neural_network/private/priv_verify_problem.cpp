// priv_verify_problem - an instance prepared once for verify's strategies, and its candidate checks
//
// The network goes to the search dtype and to float64 (on the device of the options). The search
// boxes are the float32 boxes inside the property's (the smallest float32 >= lo, the largest
// <= hi; nearest where that is empty), so that a point of them passes the input check in float32
// and in float64; the bounds take the float64 boxes that also hold the nearest float32 of each
// bound, since the scoring compares a float32 input with the bound rounded to float32. The loss of
// a point is negative inside the unsafe set; the attacks descend an objective whose gradient is
// the chain rule through the dense layers (CORA's backprop of computePGDAttack).
//
// Syntax:   NnProblem problem(nn, property, options, confirm);   Tensor l = problem.loss(y, cases);
// See also: verify.cpp, priv_verify.h
// References:
//    [1] A. Kurakin, I. J. Goodfellow, S. Bengio. "Adversarial machine learning at scale", ICLR
//        2017.
//    [2] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu. "Towards deep learning models
//        resistant to adversarial attacks", ICLR 2018.
//    [3] L. Koller, T. Ladner, M. Althoff. "Out of the shadows: exploring a latent space for
//        neural network verification", ICLR 2026 (arXiv:2505.17854).
//
// CORA: nn/@neuralNetwork/verify.m
// CORA: nn/@neuralNetwork/computePGDAttack.m
// CORA authors: Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   23-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_verify.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int64_t kBudget = int64_t{1} << 30; // 1 GiB per batch, unless the options say

// The network and the boxes -----------------------------------------------------------------------

/// The network with its weights in the dtype, on the device (empty: where they are).
std::shared_ptr<const NeuralNetwork> aux_network(const NeuralNetwork &net, DType dtype,
                                                 const std::string &device) {
    std::vector<NeuralNetwork::Layer> layers;
    for (const NeuralNetwork::Layer &layer : net.layers())
        layers.push_back(layer.to(dtype, device));
    return std::make_shared<const NeuralNetwork>(layers, net.activations());
}

/// The float32 box inside [lo, hi] (nearest where it is empty), and the float64 box around [lo, hi]
/// and the nearest float32 of its bounds, as host values.
void aux_boxes(const std::vector<double> &lo, const std::vector<double> &hi,
               std::vector<double> &in_lo, std::vector<double> &in_hi, std::vector<double> &out_lo,
               std::vector<double> &out_hi) {
    for (std::size_t i = 0; i < lo.size(); ++i) {
        float a = float(lo[i]), b = float(hi[i]);
        if (double(a) < lo[i]) a = std::nextafter(a, std::numeric_limits<float>::infinity());
        if (double(b) > hi[i]) b = std::nextafter(b, -std::numeric_limits<float>::infinity());
        const bool empty = a > b;
        in_lo.push_back(empty ? double(float(lo[i])) : double(a));
        in_hi.push_back(empty ? double(float(hi[i])) : double(b));
        out_lo.push_back(std::min(lo[i], double(float(lo[i]))));
        out_hi.push_back(std::max(hi[i], double(float(hi[i]))));
    }
}

// The default check -------------------------------------------------------------------------------

/// The confirm check without one: the point lies in a case's box and its float64 output in a
/// polytope of that case.
std::optional<std::vector<double>> aux_default_confirm(const NeuralNetwork &net,
                                                       const NnProperty &prop,
                                                       const std::vector<double> &x) {
    const int64_t n = int64_t(x.size()), C = prop.lo.shape()[0];
    const std::vector<double> lo = prop.lo.data(), hi = prop.hi.data();
    const std::vector<double> y = net.evaluate(Tensor::like(net.layers()[0].b, x, {1, n})).data();
    const int64_t P = prop.A.shape()[0], K = prop.A.shape()[1], m = prop.A.shape()[2];
    const std::vector<double> A = prop.A.data(), b = prop.b.data();
    for (int64_t k = 0; k < C; ++k) {
        bool inside = true;
        for (int64_t i = 0; i < n; ++i)
            inside = inside && x[std::size_t(i)] >= float(lo[std::size_t(k * n + i)]) &&
                     x[std::size_t(i)] <= float(hi[std::size_t(k * n + i)]);
        if (!inside) continue;
        // some polytope of the case holds the output
        for (int64_t p = 0; p < P; ++p) {
            if (prop.poly_case[std::size_t(p)] != k) continue;
            bool all = true;
            for (int64_t r = 0; r < K && all; ++r) {
                double s = 0;
                for (int64_t j = 0; j < m; ++j)
                    s += A[std::size_t((p * K + r) * m + j)] * y[std::size_t(j)];
                all = s <= b[std::size_t(p * K + r)];
            }
            if (all) return y;
        }
    }
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The problem -------------------------------------------------------------------------------------

NnProblem::NnProblem(const NeuralNetwork &net, const NnProperty &property,
                     const NnVerifyOptions &options, ConfirmFn confirm)
    : property_(property), options_(options), confirm_(std::move(confirm)) {
    net_ = aux_network(net, DType::float64, options.device); // NOLINT(backend): the bounds
    net32_ = aux_network(net, options.dtype, options.device);
    memory_budget = options.memory_budget > 0 ? options.memory_budget : kBudget;
    aux_set_property(property);
}

void NnProblem::aux_set_property(const NnProperty &property) {
    detail::Entry entry;
    property_ = property;
    const Tensor &like64 = net_->layers()[0].b, &like32 = net32_->layers()[0].b;
    // the search boxes inside the property's and the outer boxes of the bounds
    std::vector<double> in_lo, in_hi, out_lo, out_hi;
    aux_boxes(property.lo.data(), property.hi.data(), in_lo, in_hi, out_lo, out_hi);
    const Shape box = property.lo.shape();
    lo = Tensor::like(like32, in_lo, box);
    hi = Tensor::like(like32, in_hi, box);
    outer_lo = Tensor::like(like64, out_lo, box);
    outer_hi = Tensor::like(like64, out_hi, box);
    box_case = like64.arange_like(box[0]);
    search_case = like32.arange_like(box[0]);
    // the polytopes in float64 and in the search dtype
    A = Tensor::like(like64, property.A.data(), property.A.shape());
    b = Tensor::like(like64, property.b.data(), property.b.shape());
    std::vector<double> cases;
    cases.reserve(property.poly_case.size());
    for (const int64_t k : property.poly_case) cases.push_back(double(k));
    poly_case = Tensor::like(like64, cases, {int64_t(cases.size())});
    A32_ = A.to(options_.dtype);
    b32_ = b.to(options_.dtype);
    poly_case32_ = poly_case.to(options_.dtype);
    // the polytopes of each case; per case when that is cheaper than all of them per row
    case_polys_.assign(std::size_t(box[0]), {});
    max_polys_ = 0;
    for (std::size_t k = 0; k < property.poly_case.size(); ++k)
        case_polys_[std::size_t(property.poly_case[k])].push_back(int64_t(k));
    for (const std::vector<int64_t> &list : case_polys_)
        max_polys_ = std::max(max_polys_, int64_t(list.size()));
    const int64_t P = A.shape()[0], m = A.shape()[2];
    per_case_ = box[0] > 1 && max_polys_ * m < P;
}

// The cases in chunks -----------------------------------------------------------------------------

std::vector<std::vector<int64_t>> NnProblem::case_chunks(int64_t max_rows) const {
    detail::Entry entry;
    // none for few rows: the bounding takes them all at once
    const int64_t P = A.shape()[0], K = A.shape()[1];
    if (!per_case_ || P * K <= max_rows) return {};
    std::vector<std::vector<int64_t>> chunks(1);
    int64_t rows = 0;
    for (std::size_t c = 0; c < case_polys_.size(); ++c) {
        const int64_t mine = int64_t(case_polys_[c].size()) * K;
        if (rows > 0 && rows + mine > max_rows) {
            chunks.emplace_back();
            rows = 0;
        }
        chunks.back().push_back(int64_t(c));
        rows += mine;
    }
    return chunks;
}

NnProblem NnProblem::subproblem(const std::vector<int64_t> &cases) const {
    detail::Entry entry;
    // the cases' boxes, their polytopes, the cases renumbered in order
    const int64_t n = property_.lo.shape()[1], K = property_.A.shape()[1];
    const int64_t m = property_.A.shape()[2];
    // the property's values on the host, read once for all the chunks
    if (!host_) {
        host_ = std::make_shared<HostProperty>(HostProperty{
            property_.lo.data(), property_.hi.data(), property_.A.data(), property_.b.data()});
    }
    const std::vector<double> &lo_all = host_->lo, &hi_all = host_->hi;
    const std::vector<double> &A_all = host_->A, &b_all = host_->b;
    // each case's box and its polytopes in turn
    NnProperty sub;
    std::vector<double> lo_s, hi_s, A_s, b_s;
    for (std::size_t j = 0; j < cases.size(); ++j) {
        const auto c = std::size_t(cases[j]);
        lo_s.insert(lo_s.end(), lo_all.begin() + std::ptrdiff_t(c) * n,
                    lo_all.begin() + std::ptrdiff_t(c + 1) * n);
        hi_s.insert(hi_s.end(), hi_all.begin() + std::ptrdiff_t(c) * n,
                    hi_all.begin() + std::ptrdiff_t(c + 1) * n);
        for (const int64_t p : case_polys_[c]) {
            A_s.insert(A_s.end(), A_all.begin() + p * K * m, A_all.begin() + (p + 1) * K * m);
            b_s.insert(b_s.end(), b_all.begin() + p * K, b_all.begin() + (p + 1) * K);
            sub.poly_case.push_back(int64_t(j));
        }
    }
    const auto C = int64_t(cases.size()), P = int64_t(sub.poly_case.size());
    sub.lo = Tensor::like(property_.lo, lo_s, {C, n});
    sub.hi = Tensor::like(property_.hi, hi_s, {C, n});
    sub.A = Tensor::like(property_.A, A_s, {P, K, m});
    sub.b = Tensor::like(property_.b, b_s, {P, K});
    // the same networks, options and check
    return {*this, sub};
}

NnProblem::NnProblem(const NnProblem &parent, const NnProperty &property)
    : memory_budget(parent.memory_budget), deadline(parent.deadline), net_(parent.net_),
      net32_(parent.net32_), options_(parent.options_), confirm_(parent.confirm_) {
    aux_set_property(property);
}

NnProblem::CaseMargins NnProblem::aux_case_margins(const Tensor &y, const Tensor &cases) const {
    detail::Entry entry;
    const int64_t S = y.shape()[0], Q = max_polys_, K = A32_.shape()[1], m = A32_.shape()[2];
    // each row's slots: the polytopes of its case, padded to Q
    const std::vector<double> c = cases.data();
    CaseMargins out;
    out.poly.assign(std::size_t(S * Q), 0);
    std::vector<double> valid(std::size_t(S * Q), 0.0);
    for (int64_t s = 0; s < S; ++s) {
        const std::vector<int64_t> &list = case_polys_[std::size_t(c[std::size_t(s)])];
        for (std::size_t j = 0; j < list.size(); ++j) {
            out.poly[std::size_t(s * Q) + j] = list[j];
            valid[std::size_t(s * Q) + j] = 1.0;
        }
    }
    // A_k y - b_k of the slots, one batched product per row
    const Tensor rows = A32_.take(out.poly, 0).reshape({S, Q * K, m});
    out.margin = rows.matmul(y.expand_dims(-1)).squeeze(-1).reshape({S, Q, K}) -
                 b32_.take(out.poly, 0).reshape({S, Q, K});
    out.valid = Tensor::like(y, valid, {S, Q});
    return out;
}

Tensor NnProblem::loss(const Tensor &y, const Tensor &cases) const {
    if (per_case_) {
        // the worst row of each of the case's polytopes, the best of them
        const CaseMargins cm = aux_case_margins(y, cases);
        const Tensor worst = cm.margin.max(2);
        return Tensor::where(cm.valid, worst, worst.full_like(worst.shape(), kInf)).min(1);
    }
    const int64_t S = y.shape()[0], P = A32_.shape()[0], K = A32_.shape()[1];
    const Tensor margin = y.matmul(A32_.reshape({P * K, A32_.shape()[2]}).matrix_transpose())
                              .reshape({S, P, K})
                              .subtract(b32_);
    // the worst row of each polytope, the best polytope of the case
    const Tensor worst = margin.max(2);
    const Tensor mine = cases.expand_dims(1).equal(poly_case32_.expand_dims(0));
    return Tensor::where(mine, worst, worst.full_like(worst.shape(), kInf)).min(1);
}

Tensor NnProblem::objective_gradient(const Tensor &x, const Tensor &cases, Tensor *loss) const {
    const NeuralNetwork &net = *net32_;
    // the forward pass, keeping the inputs of the activations and the outputs (outs[k + 1])
    const std::vector<NeuralNetwork::Layer> &layers = net.layers();
    std::vector<Tensor> pre, outs{x};
    Tensor h = x;
    for (std::size_t k = 0; k < layers.size(); ++k) {
        check_deadline(deadline);
        // the map of the source's output, plus the residuals
        Tensor z = layers[k].rows(outs[std::size_t(net.source(k) + 1)]);
        for (const NeuralNetwork::Residual &r : layers[k].residuals) {
            const Tensor &hr = outs[std::size_t(r.source + 1)];
            z = z + (r.map ? r.map->rows(hr) : hr);
        }
        pre.push_back(z + layers[k].b);
        const Activation a = net.activations()[k];
        h = a == Activation::relu         ? pre.back().maximum(0.0)
            : a == Activation::sigmoid    ? pre.back().sigmoid()
            : a == Activation::tanh       ? pre.back().tanh()
            : a == Activation::square     ? pre.back().square()
            : a == Activation::sign       ? pre.back().sign()
            : a == Activation::floor      ? pre.back().floor()
            : a == Activation::exp        ? pre.back().exp()
            : a == Activation::reciprocal ? pre.back().ones_like().divide(pre.back().maximum(1.0))
                                          : pre.back();
        outs.push_back(h);
    }
    // the margins of the rows of every polytope (or of the case's own, per_case_)
    const int64_t S = x.shape()[0], P = A32_.shape()[0], K = A32_.shape()[1];
    CaseMargins cm;
    if (per_case_) cm = aux_case_margins(h, cases);
    const Tensor margin = per_case_
                              ? cm.margin
                              : h.matmul(A32_.reshape({P * K, A32_.shape()[2]}).matrix_transpose())
                                    .reshape({S, P, K})
                                    .subtract(b32_);
    // the objective of each polytope, and the best one of the case
    const Tensor value = margin.maximum(0.0).sum(2) + margin.max(2) * 0.1;
    const Tensor mine =
        per_case_ ? cm.valid : cases.expand_dims(1).equal(poly_case32_.expand_dims(0));
    if (loss)
        *loss = Tensor::where(mine, margin.max(2), value.full_like(value.shape(), kInf)).min(1);
    const Tensor best =
        Tensor::where(mine, value, value.full_like(value.shape(), kInf)).argmin(1, true);
    const Tensor rows = margin.take_along_axis(best.expand_dims(-1), 1).squeeze(1);
    // d/dy: the violated rows and a tenth of the worst row
    const Tensor worst = rows.argmax(1, true);
    const Tensor hot = worst.equal(rows.arange_like(K).reshape({1, K}));
    const Tensor weight = rows.greater(rows.zeros_like()) + hot * 0.1;
    Tensor chosen;
    if (per_case_) {
        // the slot's polytope
        const std::vector<double> slot = best.squeeze(1).data();
        std::vector<int64_t> ids(slot.size());
        for (std::size_t s = 0; s < slot.size(); ++s)
            ids[s] = cm.poly[s * std::size_t(max_polys_) + std::size_t(slot[s])];
        chosen = A32_.take(ids, 0);
    } else {
        chosen = A32_.take(best.squeeze(1), 0);
    }
    // back through the layers, each output's gradient summed over the layers that read it
    std::vector<Tensor> acc(layers.size() + 1);
    acc.back() = weight.expand_dims(1).matmul(chosen).squeeze(1);
    const auto add = [&](int64_t j, const Tensor &v) {
        Tensor &t = acc[std::size_t(j + 1)];
        t = t.defined() ? t + v : v;
    };
    for (std::size_t k = layers.size(); k-- > 0;) {
        check_deadline(deadline);
        Tensor g = std::move(acc[k + 1]);
        // through the activation, then the maps' transposes to the outputs read
        const Activation a = net.activations()[k];
        const Tensor &z = pre[k];
        if (a == Activation::relu) g = g.multiply(z.greater(z.zeros_like()));
        if (a == Activation::sigmoid) g = g.multiply(z.sigmoid().multiply(1.0 - z.sigmoid()));
        if (a == Activation::tanh) g = g.multiply(1.0 - z.tanh().square());
        if (a == Activation::square) g = g.multiply(z + z);
        // sign: the straight-through estimator (1 on [-1, 1]), so that the attack moves
        if (a == Activation::sign) g = g.multiply(z.abs().less_equal(z.ones_like()));
        // floor: the straight-through estimator (1), as x - frac(x)
        if (a == Activation::floor) g = g.multiply(z.ones_like());
        if (a == Activation::exp) g = g.multiply(z.exp());
        if (a == Activation::reciprocal) {
            // -1 / x^2 above 1, else 0
            const Tensor y = z.ones_like().divide(z.maximum(1.0));
            g = g.multiply(
                Tensor::where(z.greater(z.ones_like()), y.square().negative(), z.zeros_like()));
        }
        for (const NeuralNetwork::Residual &r : layers[k].residuals)
            add(r.source, r.map ? r.map->rows_t(g) : g);
        add(net.source(k), layers[k].rows_t(g));
    }
    return acc[0];
}

std::optional<std::vector<double>> NnProblem::confirm(const Tensor &x) {
    ++checks;
    // the float32 point as float64 values
    std::vector<double> point = x.data();
    for (double &v : point) v = double(float(v));
    if (confirm_) return confirm_(point);
    return aux_default_confirm(*net_, property_, point);
}

int64_t NnProblem::attack_batch() const {
    int64_t elements = net_->input_dim();
    int64_t work = 0;
    for (const NeuralNetwork::Layer &layer : net_->layers()) {
        // the values and gradients of every layer, and the largest patches of a product
        elements += 2 * layer.outputs();
        work = std::max(work, layer.work());
    }
    elements += 2 * work;
    const int64_t itemsize = options_.dtype == DType::float32 ? 4 : 8; // NOLINT(backend): bytes
    return std::max<int64_t>(1, memory_budget / (4 * elements * itemsize));
}

// Candidates --------------------------------------------------------------------------------------

bool priv_confirm_rows(NnProblem &problem, const Tensor &x, const Tensor &loss,
                       AttackResult &result) {
    const std::vector<double> values = loss.data();
    std::vector<int64_t> rows;
    for (std::size_t i = 0; i < values.size(); ++i)
        if (values[i] <= 0) rows.push_back(int64_t(i));
    // the least loss first, at most max_candidates
    std::stable_sort(rows.begin(), rows.end(), [&](int64_t a, int64_t b) {
        return values[std::size_t(a)] < values[std::size_t(b)];
    });
    if (int64_t(rows.size()) > problem.options().max_candidates)
        rows.resize(std::size_t(problem.options().max_candidates));
    // each candidate through the confirm check, the first that passes is the result
    for (const int64_t i : rows) {
        ++result.candidates;
        const Tensor point = x.slice(0, i, i + 1).squeeze(0);
        if (auto y = problem.confirm(point)) {
            result.x = point.data();
            for (double &v : result.x) v = double(float(v));
            result.y = *y;
            return true;
        }
    }
    return false;
}

bool priv_check_points(NnProblem &problem, const Tensor &x, const Tensor &cases,
                       AttackResult &result) {
    if (x.shape()[0] == 0) return false;
    // in the search dtype, clamped into the case's float32 box
    const Tensor index = cases.to(problem.lo.dtype());
    const Tensor x32 = x.to(problem.lo.dtype())
                           .maximum(problem.lo.take(index, 0))
                           .minimum(problem.hi.take(index, 0));
    // in chunks of a quarter of the attack's batch, the deadline checked between them
    const int64_t S = x32.shape()[0], chunk = std::max<int64_t>(1, problem.attack_batch() / 4);
    if (S <= chunk) {
        // one chunk: all at once
        const Tensor loss = problem.loss(problem.search_net().evaluate(x32), index);
        return priv_confirm_rows(problem, x32, loss, result);
    }
    for (int64_t start = 0; start < S; start += chunk) {
        // the chunk's losses and its candidates
        check_deadline(problem.deadline);
        const int64_t stop = std::min(S, start + chunk);
        const Tensor xs = x32.slice(0, start, stop);
        const Tensor loss =
            problem.loss(problem.search_net().evaluate(xs), index.slice(0, start, stop));
        if (priv_confirm_rows(problem, xs, loss, result)) return true;
    }
    return false;
}

bool priv_is_oom(const std::exception &error) {
    // std's, torch's and CORA's messages
    const std::string what = error.what();
    return dynamic_cast<const std::bad_alloc *>(&error) != nullptr ||
           what.find("out of memory") != std::string::npos ||
           what.find("can't allocate memory") != std::string::npos ||
           what.find("CORA:outOfMemory") != std::string::npos;
}

std::vector<int64_t> priv_rows_of(const std::vector<int64_t> &flat, int64_t cols) {
    std::vector<int64_t> rows;
    rows.reserve(flat.size());
    for (const int64_t f : flat) rows.push_back(f / cols);
    return rows;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
