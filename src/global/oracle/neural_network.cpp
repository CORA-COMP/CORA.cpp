// neural_network - the oracle's layers and networks on a zonotope (CORALean)
//
// See also: lean/request.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       12-November-2026
// Last update:   25-November-2026
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"

#include "cora/global/error.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/request.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Whether the kind is one of CORA.cpp's activations with a line of its own (nnLayer.relaxation's).
bool aux_is_line(const std::string &kind) {
    for (const char *line : {"exp", "reciprocal", "cube", "inverse", "softplus", "root", "invSqrt"})
        if (kind == line) return true;
    return false;
}

// The depths of the line of `kind` (nnLayer.relaxation's k and p, unused at nearest).
void aux_depths(const std::string &kind, std::vector<Operand> &args) {
    if (kind == "exp" || kind == "softplus" || kind == "invSqrt" || kind == "sin" || kind == "cos")
        args.push_back({"k", int64_t(30)});
    if (aux_is_line(kind)) args.push_back({"p", int64_t(60)});
}

// A vector as CORA's column (n, 1) on the wire.
Tensor aux_column(const cora::Tensor &t) { return Tensor(t.reshape({t.numel(), 1})); }

// A list of integers on the wire.
Json aux_ints(const std::vector<int64_t> &v) {
    Json a = Json::array();
    for (const int64_t x : v) a.push(x);
    return a;
}

// A linear layer of CORA.cpp's port: its kind, a tensor of its weights' dtype, its fields.
NnLayer aux_port_layer(const std::string &kind, const cora::Tensor &weights, Json fields) {
    NnLayer layer;
    layer.kind = kind;
    layer.W = weights;
    layer.geometry = std::move(fields);
    return layer;
}

// A layer in wire form, and the dtype of its weights ("" for a layer without).
Json aux_layer(const NnLayer &layer, std::string &dtype) {
    Json j = Json::object();
    j.set("kind", layer.kind);
    if (layer.kind == "linear") {
        const Tensor W(layer.W), b(layer.b);
        dtype = W.dtype();
        j.set("W", W.to_json()).set("b", b.to_json());
    } else if (layer.kind == "leakyRelu") {
        j.set("alpha", Tensor::encode(layer.alpha));
    } else if (layer.kind == "poly") {
        const Tensor C(layer.W), d(layer.b);
        dtype = C.dtype();
        j.set("coeffs", C.to_json()).set("d", d.to_json());
    } else if (layer.kind == "sigmoid" || layer.kind == "tanh") {
        j.set("k", int64_t(layer.k));
    } else if (aux_is_line(layer.kind)) {
        // the depths of exp's reduction and of the log and root iterations (unused at nearest)
        if (layer.kind == "exp" || layer.kind == "softplus" || layer.kind == "invSqrt")
            j.set("k", int64_t(30));
        j.set("p", int64_t(60));
    } else if (layer.kind == "conv") {
        // the geometry, then the kernel and the bias
        const Tensor K(layer.W), b(layer.b);
        dtype = K.dtype();
        j = layer.geometry;
        j.set("kind", layer.kind).set("kernel", K.to_json()).set("b", b.to_json());
    } else if (layer.kind == "elementwiseAffine" || layer.kind == "batchNorm" ||
               layer.kind == "avgPool" || layer.kind == "lipLinear") {
        // the fields as the factory wrote them, the weights' dtype from the first one
        if (layer.W.defined()) dtype = Tensor(layer.W).dtype();
        j = layer.geometry;
        j.set("kind", layer.kind);
    } else if (layer.kind != "relu" && layer.kind != "sign" && layer.kind != "square") {
        CORA_FAIL(Error::wrong_value, "lean: the layer kind '" + layer.kind + "' is unknown");
    }
    // the order of evaluate_taylm: one number for all neurons, else one per neuron
    if (layer.order.size() == 1) {
        j.set("order", int64_t(layer.order[0]));
    } else if (!layer.order.empty()) {
        Json orders = Json::array();
        for (const int k : layer.order) orders.push(int64_t(k));
        j.set("order", std::move(orders));
    }
    return j;
}

// A graph layer in wire form, and the dtype of its weights.
Json aux_graph_layer(const NnGraphLayer &layer, std::string &dtype) {
    const Tensor W(layer.W);
    dtype = W.dtype();
    Json j = Json::object();
    j.set("W", W.to_json()).set("b", aux_column(layer.b).to_json());
    if (layer.source != NnGraphLayer::kPrevious) j.set("source", layer.source);
    // each residual's source, and its map unless the identity
    if (!layer.residuals.empty()) {
        Json residuals = Json::array();
        for (const NnGraphResidual &r : layer.residuals) {
            Json e = Json::object();
            e.set("source", r.source);
            if (r.R.defined()) e.set("R", Tensor(r.R).to_json());
            residuals.push(e);
        }
        j.set("residuals", residuals);
    }
    j.set("activation", layer.activation).set("slots", layer.slots);
    // a product's line takes CORA.cpp's square root (the precision of nnLayer.productLine)
    if (layer.activation == "product") j.set("p", int64_t(60));
    return j;
}

// The name of an activation on the wire of the forms and passes.
std::string aux_activation_name(Activation a) {
    switch (a) {
    case Activation::identity:
        return "identity";
    case Activation::relu:
        return "relu";
    case Activation::sigmoid:
        return "sigmoid";
    case Activation::tanh:
        return "tanh";
    default:
        CORA_FAIL(Error::not_supported, "lean: the forms and passes take identity, relu, sigmoid "
                                        "and tanh activations");
    }
}

// A chain of dense layers as the wire's list {W, b, activation}, and the dtype of its weights.
Json aux_chain(const NeuralNetwork &nn, std::string &dtype) {
    if (!nn.is_dense() || !nn.is_chain())
        CORA_FAIL(Error::not_supported, "lean: the forms and passes take a chain of dense layers");
    Json list = Json::array();
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        const Tensor W(nn.layers()[k].W);
        dtype = W.dtype();
        Json j = Json::object();
        j.set("W", W.to_json())
            .set("b", aux_column(nn.layers()[k].b).to_json())
            .set("activation", aux_activation_name(nn.activations()[k]));
        list.push(std::move(j));
    }
    return list;
}

// A zonotope of center c and generators G (columns) on the wire, without an error box.
Json aux_zonotope(const cora::Tensor &c, const cora::Tensor &G) {
    return Json::object().set("c", aux_column(c).to_json()).set("G", Tensor(G).to_json());
}

// The gradients of the reply and the slopes of its layers (undefined for none).
NnZonoGrad aux_zono_grad(const Reply &r) {
    NnZonoGrad out{r.get<Interval>("gc"), r.get<Interval>("gG"), {}};
    for (const std::optional<Tensor> &m : r.get<std::vector<std::optional<Tensor>>>("slopes"))
        out.slopes.push_back(m ? m->gather() : cora::Tensor());
    return out;
}

// A count of the wire: the integer, or "inf".
Wire aux_count(double n) {
    if (std::isinf(n)) return Wire(std::string("inf"));
    return Wire(static_cast<int64_t>(n));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The layers --------------------------------------------------------------------------------------

NnLayer NnLayer::linear(const cora::Tensor &W, const cora::Tensor &b) {
    NnLayer layer;
    layer.kind = "linear";
    layer.W = W;
    layer.b = b.ndim() == 1 ? b.expand_dims(-1) : b; // the wire takes CORA's column (n, 1)
    return layer;
}

// the activations, which have no weights
NnLayer NnLayer::relu() {
    NnLayer layer;
    layer.kind = "relu";
    return layer;
}

NnLayer NnLayer::leaky_relu(double alpha) {
    NnLayer layer;
    layer.kind = "leakyRelu";
    layer.alpha = alpha;
    return layer;
}

// the two that take the depth of the exponential's reduction, left at its default
NnLayer NnLayer::sigmoid() {
    NnLayer layer;
    layer.kind = "sigmoid";
    return layer;
}

NnLayer NnLayer::tanh() {
    NnLayer layer;
    layer.kind = "tanh";
    return layer;
}

// the two lines of NeuralNetwork::verify without a CORA layer
NnLayer NnLayer::sign() {
    NnLayer layer;
    layer.kind = "sign";
    return layer;
}

NnLayer NnLayer::square() {
    NnLayer layer;
    layer.kind = "square";
    return layer;
}

// a convolution: the kernel as a matrix (out, C / groups * kh * kw) and its geometry
NnLayer NnLayer::conv(const cora::Tensor &kernel, const cora::Tensor &bias,
                      const std::vector<int64_t> &input, const NeuralNetwork::Conv2d &spec) {
    NnLayer layer;
    layer.kind = "conv";
    const std::vector<int64_t> &s = kernel.shape();
    layer.W = kernel.reshape({s[0], s[1] * s[2] * s[3]});
    layer.b = bias.reshape({bias.numel(), 1});
    // the geometry as lists of integers
    const auto ints = [](const auto &v) {
        Json a = Json::array();
        for (const int64_t x : v) a.push(x);
        return a;
    };
    layer.geometry = Json::object();
    layer.geometry.set("kshape", ints(std::vector<int64_t>{s[2], s[3]}))
        .set("input", ints(input))
        .set("stride", ints(spec.stride))
        .set("dilation", ints(spec.dilation))
        .set("pads", ints(spec.pads))
        .set("groups", spec.groups);
    return layer;
}

// an activation with a line of its own: no weights, the line's depths on the wire
NnLayer NnLayer::line(const std::string &kind) {
    if (!aux_is_line(kind))
        CORA_FAIL(Error::wrong_value, "lean: '" + kind + "' is not an activation with a line");
    NnLayer layer;
    layer.kind = kind;
    return layer;
}

// the fit of order k given: its coefficients and its errors
NnLayer NnLayer::poly(const cora::Tensor &coeffs, const cora::Tensor &d) {
    NnLayer layer;
    layer.kind = "poly";
    layer.W = coeffs;
    layer.b = d;
    return layer;
}

// The linear layers of the port -------------------------------------------------------------------

// each a kind, the tensor whose dtype the weights have, and its fields on the wire
NnLayer NnLayer::elementwise_affine(const cora::Tensor &scale, const cora::Tensor &offset) {
    return aux_port_layer("elementwiseAffine", scale,
                          Json::object()
                              .set("scale", aux_column(scale).to_json())
                              .set("offset", aux_column(offset).to_json()));
}

NnLayer NnLayer::batch_norm(const cora::Tensor &scale, const cora::Tensor &offset,
                            const cora::Tensor &mean, const cora::Tensor &var,
                            const std::vector<int64_t> &input, double epsilon) {
    // the depth of the root's iteration is unused at nearest
    return aux_port_layer("batchNorm", scale,
                          Json::object()
                              .set("scale", aux_column(scale).to_json())
                              .set("offset", aux_column(offset).to_json())
                              .set("mean", aux_column(mean).to_json())
                              .set("var", aux_column(var).to_json())
                              .set("epsilon", Tensor::encode(epsilon))
                              .set("input", aux_ints(input))
                              .set("p", int64_t(60)));
}

NnLayer NnLayer::avg_pool(const std::array<int64_t, 2> &pool, const std::vector<int64_t> &input,
                          const std::array<int64_t, 2> &stride,
                          const std::array<int64_t, 4> &pads) {
    // an empty stride is the pool's, as CORA.cpp's
    const std::array<int64_t, 2> step = stride[0] > 0 ? stride : pool;
    return aux_port_layer("avgPool", cora::Tensor(),
                          Json::object()
                              .set("pool", aux_ints({pool[0], pool[1]}))
                              .set("input", aux_ints(input))
                              .set("stride", aux_ints({step[0], step[1]}))
                              .set("pads", aux_ints({pads[0], pads[1], pads[2], pads[3]})));
}

NnLayer NnLayer::lip_linear(const cora::Tensor &W, const cora::Tensor &b, double lambda) {
    return aux_port_layer("lipLinear", W,
                          Json::object()
                              .set("W", Tensor(W).to_json())
                              .set("b", aux_column(b).to_json())
                              .set("lambda", Tensor::encode(lambda)));
}

// The lines ---------------------------------------------------------------------------------------

// one line per box, with the method and the depths its kind takes
NnLine relaxation(const std::string &kind, const cora::Tensor &l, const cora::Tensor &u,
                  const cora::Tensor &alpha, const std::string &method) {
    std::vector<Operand> args{{"kind", kind}, {"l", Tensor(l)}, {"u", Tensor(u)}};
    if (alpha.defined()) args.push_back({"alpha", Tensor(alpha)});
    if (!method.empty()) args.push_back({"method", method});
    aux_depths(kind, args);
    const Reply r = ask("nnLayer.relaxation", args);
    return {r.get<Tensor>("slope"), r.get<Tensor>("offset"), r.get<Tensor>("error")};
}

// the product line of the pairs (a_i, b_i), and the box of a softmax's sums
NnLine product_line(const cora::Tensor &c, const cora::Tensor &G, const cora::Tensor &dr) {
    // the zonotope of the two halves, the root's Newton steps and the interval center's radii
    std::vector<Operand> args{{"Z", Zonotope(c, G)}, {"p", int64_t(60)}};
    if (dr.defined()) args.push_back({"dr", Tensor(dr)});
    const Reply r = ask("nnLayer.productLine", args);
    return {r.get<Tensor>("slope"), r.get<Tensor>("offset"), r.get<Tensor>("error")};
}

std::pair<Tensor, Tensor> softmax_sum_bounds(const cora::Tensor &W, const cora::Tensor &b,
                                             const cora::Tensor &l, const cora::Tensor &u) {
    const Reply r = ask("nnLayer.softmaxSumBounds", {{"W", Tensor(W)},
                                                     {"b", Tensor(b)},
                                                     {"l", Tensor(l)},
                                                     {"u", Tensor(u)},
                                                     {"k", int64_t(30)}});
    return {r.get<Tensor>("inf"), r.get<Tensor>("sup")};
}

// CROWN's backward bound --------------------------------------------------------------------------

CrownLayer CrownLayer::linear(const cora::Tensor &W, const cora::Tensor &b) {
    CrownLayer layer;
    layer.kind = "linear";
    layer.W = W;
    layer.b = b.ndim() == 1 ? b.expand_dims(-1) : b;
    return layer;
}

// a ReLU at its box, with the slopes of its crossing neurons if given
CrownLayer CrownLayer::relu(const cora::Tensor &l, const cora::Tensor &u,
                            const cora::Tensor &alpha) {
    CrownLayer layer;
    layer.kind = "relu";
    // the boxes and the slopes as columns (n, 1)
    const auto column = [](const cora::Tensor &t) { return t.reshape({t.numel(), 1}); };
    layer.l = column(l);
    layer.u = column(u);
    if (alpha.defined()) layer.alpha = column(alpha);
    return layer;
}

Tensor backward_bound(const std::vector<CrownLayer> &layers, const cora::Tensor &x,
                      const cora::Tensor &r, const cora::Tensor &C) {
    // the layers in wire form, the dtype of the first weights
    std::string dtype;
    Json list = Json::array();
    for (const CrownLayer &layer : layers) {
        Json j = Json::object();
        j.set("kind", layer.kind);
        if (layer.kind == "linear") {
            const Tensor W(layer.W), b(layer.b);
            dtype = W.dtype();
            j.set("W", W.to_json()).set("b", b.to_json());
        } else {
            // a ReLU: its box and its slopes
            j.set("l", Tensor(layer.l).to_json()).set("u", Tensor(layer.u).to_json());
            if (layer.alpha.defined()) j.set("alpha", Tensor(layer.alpha).to_json());
        }
        list.push(j);
    }
    const Reply reply =
        ask("neuralNetwork.backwardBound",
            {{"layers", Wire(list, dtype)}, {"x", Tensor(x)}, {"r", Tensor(r)}, {"C", Tensor(C)}});
    return reply.get<Tensor>("lower");
}

// The evaluations ---------------------------------------------------------------------------------

Json layers_json(const std::vector<NnLayer> &layers, std::string &dtype) {
    Json list = Json::array();
    for (const NnLayer &layer : layers) list.push(aux_layer(layer, dtype));
    return list;
}

// one layer on the zonotope, the regression error only when one is asked for
Zonotope evaluate_layer(const NnLayer &layer, const Zonotope &Z, const std::string &poly_method,
                        const std::string &regression_error) {
    std::string dtype;
    const Json j = aux_layer(layer, dtype);
    std::vector<Operand> args{{"layer", Wire(j, dtype)}, {"Z", Z}, {"poly_method", poly_method}};
    if (!regression_error.empty()) args.push_back({"regression_error", regression_error});
    return ask("nnLayer.evaluateZonotope", args).get<Zonotope>("Z");
}

// the layer "poly" on the polynomial zonotope, its errors as independent generators
PolyZonotope evaluate_layer(const NnLayer &layer, const PolyZonotope &P) {
    std::string dtype;
    const Json j = aux_layer(layer, dtype);
    return ask("nnLayer.evaluatePolyZonotope",
               {{"layer", Wire(j, dtype)}, {"P", P}, {"errorForm", std::string("GI")}})
        .get<PolyZonotope>("P");
}

// the layers in order on the zonotope
Zonotope evaluate_network(const std::vector<NnLayer> &layers, const Zonotope &Z,
                          const std::string &poly_method, const std::string &regression_error) {
    std::string dtype;
    const Json list = layers_json(layers, dtype);
    std::vector<Operand> args{
        {"layers", Wire(list, dtype)}, {"Z", Z}, {"poly_method", poly_method}};
    if (!regression_error.empty()) args.push_back({"regression_error", regression_error});
    return ask("neuralNetwork.evaluate", args).get<Zonotope>("Z");
}

// the polynomial zonotope through the layers, the errors in the box or in independent generators
NnPolyResult evaluate_network_poly(const std::vector<NnLayer> &layers, const cora::Tensor &c,
                                   const cora::Tensor &G, const cora::Tensor &GI,
                                   const std::string &poly_method, const std::string &error_form,
                                   const std::string &regression_error) {
    // the layers and the set in wire form
    std::string dtype;
    const Json list = layers_json(layers, dtype);
    Json P = Json::object();
    P.set("c", Tensor(c).to_json()).set("G", Tensor(G).to_json()).set("GI", Tensor(GI).to_json());
    std::vector<Operand> args{{"layers", Wire(list, dtype)},
                              {"P", Wire(P, dtype)},
                              {"poly_method", poly_method},
                              {"errorForm", error_form}};
    if (!regression_error.empty()) args.push_back({"regression_error", regression_error});
    const Reply r = ask("neuralNetwork.evaluate", args);
    const Reply out = r.member("P");
    return {out.get<Tensor>("c").gather(), out.get<Tensor>("G").gather(),
            out.get<Tensor>("GI").gather()};
}

// the layers on the Taylor models, the regression error only when one is asked for
std::vector<Taylm> nn_evaluate_taylm(const std::vector<NnLayer> &layers,
                                     const std::vector<Taylm> &x, const std::string &poly_method,
                                     const std::string &regression_error) {
    std::string dtype;
    const Json list = layers_json(layers, dtype);
    Json models = Json::array();
    for (const Taylm &t : x) models.push(t.to_json());
    std::vector<Operand> args{{"layers", Wire(list, dtype)},
                              {"x", Wire(models, lean::dtype())},
                              {"poly_method", poly_method}};
    if (!regression_error.empty()) args.push_back({"regression_error", regression_error});
    std::vector<Taylm> out;
    for (const Reply &T : ask("neuralNetwork.evaluateTaylm", args).list("Y"))
        out.push_back(Taylm::from_model(T));
    return out;
}

// The graph networks ------------------------------------------------------------------------------

// the graph network on the zonotope, the last output with its error box
Zonotope evaluate_graph(const std::vector<NnGraphLayer> &layers, const Zonotope &Z) {
    std::string dtype;
    Json list = Json::array();
    for (const NnGraphLayer &layer : layers) list.push(aux_graph_layer(layer, dtype));
    return ask("neuralNetwork.evaluateGraph", {{"layers", Wire(list, dtype)}, {"Z", Z}})
        .get<Zonotope>("Z");
}

// The forms ---------------------------------------------------------------------------------------

Tensor merge_buckets(const std::string &kind) {
    return ask("nnLayer.mergeBuckets", {{"kind", kind}}).get<Tensor>("buckets");
}

Tensor group_sort(const cora::Tensor &x, int64_t group) {
    return ask("nnLayer.groupSort", {{"groupSize", group}, {"x", aux_column(x)}}).get<Tensor>("y");
}

Zonotope generator_reduction(const Zonotope &Z, int64_t max_gens) {
    return ask("nnLayer.generatorReduction", {{"maxGens", max_gens}, {"Z", Z}}).get<Zonotope>("Z");
}

// the layers as linear layers, each followed by its activation unless the identity
std::vector<NnDense> normal_form(const NeuralNetwork &nn) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    Json list = Json::array();
    for (const Json &layer : chain.items) {
        Json j = Json::object();
        j.set("kind", "linear").set("W", layer.at("W")).set("b", layer.at("b"));
        list.push(std::move(j));
        if (layer.at("activation").text != "identity")
            list.push(Json::object().set("kind", layer.at("activation")));
    }
    const Reply r =
        ask("neuralNetwork.normalForm", {{"layers", Wire(list, dtype)}, {"n", nn.input_dim()}});
    // an activation names the layer before it
    std::vector<NnDense> out;
    for (const Reply &layer : r.list("layers")) {
        const std::string kind = layer.get<std::string>("kind");
        if (kind == "linear")
            out.push_back({layer.get<Interval>("W"), layer.get<Interval>("b"), "identity"});
        else if (!out.empty())
            out.back().activation = kind;
    }
    return out;
}

// The passes of a ReLU chain ----------------------------------------------------------------------

// the bounds as the wire's {l, u} columns or null, and back as vectors
std::vector<NnBounds> propagate_bounds(const NeuralNetwork &nn, const std::vector<NnBounds> &bounds,
                                       int64_t i) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    // a layer without bounds is null on the wire
    Json list = Json::array();
    for (const NnBounds &b : bounds) {
        if (!b.l.defined()) {
            list.push(Json());
            continue;
        }
        list.push(
            Json::object().set("l", aux_column(b.l).to_json()).set("u", aux_column(b.u).to_json()));
    }
    const Reply r = ask("neuralNetwork.propagateBounds",
                        {{"layers", Wire(chain, dtype)}, {"bounds", Wire(list, dtype)}, {"i", i}});
    // a null reply is a layer without bounds (undefined tensors)
    std::vector<NnBounds> out;
    for (const Reply &b : r.list("bounds")) {
        if (b.is_null()) {
            out.push_back({});
            continue;
        }
        const cora::Tensor l = b.get<Tensor>("l").gather(), u = b.get<Tensor>("u").gather();
        out.push_back({l.reshape({l.numel()}), u.reshape({u.numel()})});
    }
    return out;
}

// the gradient's bounds over a box, each a box of the oracle
std::pair<Interval, Interval> backprop_interval(const NeuralNetwork &nn, const cora::Tensor &l,
                                                const cora::Tensor &u, const cora::Tensor &gl,
                                                const cora::Tensor &gu) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    const Reply r = ask("neuralNetwork.backpropIntervalBatch", {{"layers", Wire(chain, dtype)},
                                                                {"l", aux_column(l)},
                                                                {"u", aux_column(u)},
                                                                {"gl", aux_column(gl)},
                                                                {"gu", aux_column(gu)}});
    return {r.get<Interval>("gl"), r.get<Interval>("gu")};
}

// the gradient and the Jacobian at a point
Interval backprop(const NeuralNetwork &nn, const cora::Tensor &x, const cora::Tensor &g) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    return ask("neuralNetwork.backprop",
               {{"layers", Wire(chain, dtype)}, {"x", aux_column(x)}, {"g", aux_column(g)}})
        .get<Interval>("g");
}

Interval sensitivity(const NeuralNetwork &nn, const cora::Tensor &x) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    return ask("neuralNetwork.sensitivity", {{"layers", Wire(chain, dtype)}, {"x", aux_column(x)}})
        .get<Interval>("J");
}

// The zonotope batch's gradients ------------------------------------------------------------------


NnZonoGrad backprop_zonotope_batch(const NeuralNetwork &nn, const cora::Tensor &c,
                                   const cora::Tensor &G, const cora::Tensor &grad_c,
                                   const cora::Tensor &grad_gen) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    return aux_zono_grad(
        ask("neuralNetwork.backpropZonotopeBatch", {{"layers", Wire(chain, dtype)},
                                                    {"Z", Wire(aux_zonotope(c, G), dtype)},
                                                    {"gc", aux_column(grad_c)},
                                                    {"gG", Tensor(grad_gen)}}));
}

// a linear layer reads no set, a ReLU its input set's bounds
NnZonoGrad backprop_zonotope_batch_layer(const NnLayer &layer, const cora::Tensor &c,
                                         const cora::Tensor &G, const cora::Tensor &grad_c,
                                         const cora::Tensor &grad_gen) {
    std::string dtype = lean::dtype();
    const Json j = aux_layer(layer, dtype);
    std::vector<Operand> args{
        {"layer", Wire(j, dtype)}, {"gc", aux_column(grad_c)}, {"gG", Tensor(grad_gen)}};
    if (layer.kind != "linear") args.push_back({"Z", Wire(aux_zonotope(c, G), dtype)});
    const Reply r = ask("nnLayer.backpropZonotopeBatch", args);
    if (layer.kind == "linear") return {r.get<Interval>("gc"), r.get<Interval>("gG"), {}};
    return aux_zono_grad(r);
}

// The attack and the structure --------------------------------------------------------------------

Interval compute_fgsm_attack(const NeuralNetwork &nn, const cora::Tensor &x, const cora::Tensor &t,
                             double epsilon) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    return ask("neuralNetwork.computeFGSMAttack", {{"layers", Wire(chain, dtype)},
                                                   {"x", aux_column(x)},
                                                   {"t", aux_column(t)},
                                                   {"epsilon", epsilon}})
        .get<Interval>("z");
}

// the order of the method, the point only for the sensitivity
std::vector<int64_t> input_neuron_order(const NeuralNetwork &nn, const std::string &method,
                                        const cora::Tensor &x,
                                        const std::vector<int64_t> &input_size) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    std::vector<Operand> args{{"layers", Wire(chain, dtype)}, {"method", method}};
    if (method == "sensitivity") args.push_back({"x", aux_column(x)});
    if (!input_size.empty()) args.push_back({"inputSize", input_size});
    return ask("neuralNetwork.getInputNeuronOrder", args).get<std::vector<int64_t>>("order");
}

// the counts of the structure, plain integers
std::vector<int64_t> num_neurons(const NeuralNetwork &nn) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    return ask("neuralNetwork.getNumNeurons", {{"layers", Wire(chain, dtype)}})
        .get<std::vector<int64_t>>("numNeurons");
}

int64_t num_params(const NeuralNetwork &nn) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    return ask("neuralNetwork.getNumParams", {{"layers", Wire(chain, dtype)}})
        .get<int64_t>("numParams");
}

std::vector<int64_t> refinable_layers(const NeuralNetwork &nn) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    return ask("neuralNetwork.getRefinableLayers", {{"layers", Wire(chain, dtype)}})
        .get<std::vector<int64_t>>("layers");
}

// the orders given per layer, none when empty
std::vector<std::vector<int64_t>> order_pattern(const NeuralNetwork &nn,
                                                const std::vector<std::vector<int64_t>> &orders) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    std::vector<Operand> args{{"layers", Wire(chain, dtype)}};
    if (!orders.empty()) args.push_back({"orders", orders});
    return ask("neuralNetwork.getOrderPattern", args)
        .get<std::vector<std::vector<int64_t>>>("pattern");
}

// a layer without an activation is null on the wire, (0, 0) here
ZonoBatchCount prepare_zono_batch_eval(const NeuralNetwork &nn, double num_init_gens,
                                       double num_approx_err) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    const Reply r =
        ask("neuralNetwork.prepareForZonoBatchEval", {{"layers", Wire(chain, dtype)},
                                                      {"numInitGens", aux_count(num_init_gens)},
                                                      {"numApproxErr", aux_count(num_approx_err)}});
    ZonoBatchCount out{r.get<int64_t>("numGen"), {}};
    for (const std::vector<int64_t> &e : r.get<std::vector<std::vector<int64_t>>>("approxErrGens"))
        out.errors.emplace_back(e.empty() ? 0 : e[0], e.empty() ? 0 : e[1]);
    return out;
}

// The weights and the orders ----------------------------------------------------------------------

// the chain read at the current dtype, sent at the cast's: the reply's entries are of that dtype
std::vector<std::pair<cora::Tensor, cora::Tensor>> cast_weights(const NeuralNetwork &nn,
                                                                const std::string &dtype) {
    std::string chain_dtype;
    const Json chain = aux_chain(nn, chain_dtype);
    // one (W, b) per layer of the reply
    std::vector<std::pair<cora::Tensor, cora::Tensor>> out;
    for (const Reply &layer :
         ask("neuralNetwork.castWeights", {{"layers", Wire(chain, dtype)}}).list("layers"))
        out.emplace_back(layer.get<Tensor>("W").gather(), layer.get<Tensor>("b").gather());
    return out;
}

// the pattern of the refinable layers, every order 1
std::vector<std::vector<int64_t>> reset_approx_order(const NeuralNetwork &nn) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    return ask("neuralNetwork.resetApproxOrder", {{"layers", Wire(chain, dtype)}})
        .get<std::vector<std::vector<int64_t>>>("pattern");
}

// the heuristic per layer as lists of exact scalars, null for a layer without
std::vector<std::vector<int64_t>> refine(const NeuralNetwork &nn,
                                         const std::vector<std::vector<int64_t>> &orders,
                                         const std::vector<std::vector<double>> &heuristic,
                                         const std::string &type, const std::string &kind,
                                         int64_t max_order, double gamma) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    Json heu = Json::array();
    for (const std::vector<double> &h : heuristic) {
        if (h.empty()) {
            heu.push(Json());
            continue;
        }
        Json values = Json::array();
        for (const double v : h) values.push(Tensor::encode(v));
        heu.push(std::move(values));
    }
    // the options, the orders only if given
    std::vector<Operand> args{
        {"layers", Wire(chain, dtype)}, {"heuristic", Wire(heu, dtype)}, {"type", type},
        {"heuristicKind", kind},        {"maxOrder", max_order},         {"gamma", gamma}};
    if (!orders.empty()) args.push_back({"orders", orders});
    return ask("neuralNetwork.refine", args).get<std::vector<std::vector<int64_t>>>("orders");
}

// The weights on given units ----------------------------------------------------------------------

// the sizes given, then the units of the sizes drawn and of every layer
NnRandom generate_random(const std::array<int64_t, 4> &sizes, const std::vector<double> &size_draws,
                         const std::vector<std::pair<cora::Tensor, cora::Tensor>> &draws) {
    Json layers = Json::array();
    for (const auto &[W, b] : draws)
        layers.push(Json::object().set("W", Tensor(W).to_json()).set("b", aux_column(b).to_json()));
    const std::string dtype = lean::dtype();
    std::vector<Operand> args{{"inputs", sizes[0]},
                              {"outputs", sizes[1]},
                              {"layers", sizes[2]},
                              {"hidden", sizes[3]},
                              {"draws", Wire(layers, dtype)}};
    if (!size_draws.empty()) {
        // the units of the sizes as a column of exact scalars
        Json data = Json::array();
        for (const double u : size_draws) data.push(Tensor::encode(u));
        Json column = Json::object();
        column.set("rows", int64_t(size_draws.size())).set("cols", int64_t(1)).set("data", data);
        args.push_back({"sizeDraws", Wire(column, dtype)});
    }
    // the sizes of the reply, then each layer's boxes
    const Reply r = ask("neuralNetwork.generateRandom", args);
    NnRandom out;
    out.sizes = {r.get<int64_t>("inputs"), r.get<int64_t>("outputs"), r.get<int64_t>("layers"),
                 r.get<int64_t>("hidden")};
    for (const Reply &layer : r.list("network"))
        out.layers.emplace_back(layer.get<Interval>("W"), layer.get<Interval>("b"));
    return out;
}

// a layer drawn again has a box of weights, a layer kept its own matrix
std::vector<std::optional<Interval>>
init_weights(const NeuralNetwork &nn, const std::string &method, const std::vector<int64_t> &which,
             const std::vector<cora::Tensor> &draws, bool normals) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    Json units = Json::array();
    for (const cora::Tensor &u : draws) units.push(aux_column(u).to_json());
    std::vector<Operand> args{{"layers", Wire(chain, dtype)},
                              {"method", method},
                              {normals ? "normals" : "draws", Wire(units, dtype)}};
    if (!which.empty()) args.push_back({"which", which});
    // the layers drawn again by `which`, in the order of the reply
    std::vector<std::optional<Interval>> out;
    const std::vector<Reply> replies = ask("neuralNetwork.initWeights", args).list("layers");
    for (std::size_t k = 0; k < replies.size(); ++k) {
        const bool drawn =
            which.empty() || std::find(which.begin(), which.end(), int64_t(k)) != which.end();
        out.push_back(drawn ? std::optional<Interval>(replies[k].get<Interval>("W"))
                            : std::nullopt);
    }
    return out;
}

// The system, the explanation and the reduction ---------------------------------------------------

// the output, C, and one block of D per layer
NnSystem nonlinear_sys_dt(const NeuralNetwork &nn, const cora::Tensor &x, const cora::Tensor &u) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    const Reply r =
        ask("neuralNetwork.nonlinearSysDT",
            {{"layers", Wire(chain, dtype)}, {"x", aux_column(x)}, {"u", aux_column(u)}});
    NnSystem out{r.get<Interval>("y"), r.get<Interval>("C"), {}};
    for (const Reply &block : r.list("D"))
        out.D.emplace_back(block.get<Tensor>("inf"), block.get<Tensor>("sup"));
    return out;
}

// the network in the kinds of evaluate: each dense layer, then its activation
std::vector<int64_t> explain(const NeuralNetwork &nn, const cora::Tensor &x, int64_t label,
                             double epsilon, const std::vector<int64_t> &feat_order,
                             const std::vector<int64_t> &input_size, double output_threshold) {
    std::vector<NnLayer> layers;
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        layers.push_back(NnLayer::linear(nn.layers()[k].W, nn.layers()[k].b));
        const Activation a = nn.activations()[k];
        if (a == Activation::relu) layers.push_back(NnLayer::relu());
        if (a == Activation::sigmoid) layers.push_back(NnLayer::sigmoid());
        if (a == Activation::tanh) layers.push_back(NnLayer::tanh());
    }
    // the point, the label, the radius and the order; the image size only if given
    std::string dtype;
    const Json list = layers_json(layers, dtype);
    std::vector<Operand> args{{"layers", Wire(list, dtype)},
                              {"x", aux_column(x)},
                              {"label", label},
                              {"epsilon", epsilon},
                              {"featOrder", feat_order},
                              {"outputThreshold", output_threshold}};
    if (!input_size.empty()) args.push_back({"inputSize", input_size});
    return ask("neuralNetwork.explain", args).get<std::vector<int64_t>>("freed");
}

// the reduced layers, their intervals, the merged neurons, the rates and the set
NnReduced compute_reduced_network(const NeuralNetwork &nn, const cora::Zonotope &Z,
                                  const std::vector<double> &bucket_tol,
                                  const std::vector<double> &rate, int64_t min_searches) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    const auto values = [](const std::vector<double> &v) {
        Json a = Json::array();
        for (const double x : v) a.push(Tensor::encode(x));
        return a;
    };
    std::vector<Operand> args{
        {"layers", Wire(chain, dtype)}, {"Z", Zonotope(Z)}, {"minSearches", min_searches}};
    if (!bucket_tol.empty()) args.push_back({"bucketTol", Wire(values(bucket_tol), dtype)});
    if (!rate.empty()) args.push_back({"reductionRate", Wire(values(rate), dtype)});
    // the merged neurons, the rates and the set, then the layers and their intervals
    const Reply r = ask("neuralNetwork.computeReducedNetwork", args);
    NnReduced out{{},
                  {},
                  r.get<std::vector<std::vector<int64_t>>>("merged"),
                  r.get<Tensor>("reductionRate").gather(),
                  r.get<Zonotope>("Z")};
    for (const Reply &layer : r.list("layers"))
        out.layers.emplace_back(layer.get<Tensor>("W").gather(), layer.get<Tensor>("b").gather());
    for (const Reply &d : r.list("d"))
        out.d.push_back(d.is_null()
                            ? std::nullopt
                            : std::optional<std::pair<cora::Tensor, cora::Tensor>>(
                                  {d.get<Tensor>("c").gather(), d.get<Tensor>("r").gather()}));
    return out;
}

// the generators as columns (n, m), the constraints A (p, m) and b (p)
NnConZono evaluate_con_zonotope(const NeuralNetwork &nn, const cora::Tensor &c,
                                const cora::Tensor &G) {
    std::string dtype;
    const Json chain = aux_chain(nn, dtype);
    const Reply r = ask("neuralNetwork.evaluateConZonotope",
                        {{"layers", Wire(chain, dtype)}, {"c", aux_column(c)}, {"G", Tensor(G)}});
    if (r.get<bool>("empty")) return {true, {}, {}, {}, {}};
    return {false, r.get<Tensor>("c").gather(), r.get<Tensor>("G").gather(),
            r.get<Tensor>("A").gather(), r.get<Tensor>("b").gather()};
}

// The regions of a ReLU polynomial ----------------------------------------------------------------

// the regions between the outer two, each polynomial a column of the reply
std::vector<RegionPoly> find_region_polys(double tol, int64_t order, double l_max, double u_max) {
    const Reply r = ask(
        "nnLayer.findRegionPolys",
        {{"activation", "relu"}, {"tol", tol}, {"order", order}, {"lMax", l_max}, {"uMax", u_max}});
    std::vector<RegionPoly> out;
    for (const Reply &g : r.list("regions"))
        out.push_back({g.get<double>("l"), g.get<double>("u"), g.get<Tensor>("p").gather().data(),
                       g.get<double>("d")});
    return out;
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
