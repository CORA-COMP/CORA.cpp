// neural_network - the oracle's layers and networks on a zonotope (CORALean)
//
// See also: lean/request.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       12-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"

#include "cora/global/error.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/request.h"

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

// A column (n, 1) of a vector, the wire's form of a bias.
Json aux_column(const cora::Tensor &b) {
    return Tensor(b.ndim() == 1 ? b.expand_dims(-1) : b).to_json();
}

// A graph layer in wire form, and the dtype of its weights.
Json aux_graph_layer(const NnGraphLayer &layer, std::string &dtype) {
    const Tensor W(layer.W);
    dtype = W.dtype();
    Json j = Json::object();
    j.set("W", W.to_json()).set("b", aux_column(layer.b));
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

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
