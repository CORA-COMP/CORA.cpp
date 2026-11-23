// neural_network - the oracle's layers and networks on a zonotope (CORALean)
//
// See also: lean/request.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"

#include "cora/global/error.h"
#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

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
    } else if (layer.kind == "conv") {
        // the geometry, then the kernel and the bias
        const Tensor K(layer.W), b(layer.b);
        dtype = K.dtype();
        j = layer.geometry;
        j.set("kind", layer.kind).set("kernel", K.to_json()).set("b", b.to_json());
    } else if (layer.kind != "relu" && layer.kind != "sign" && layer.kind != "square") {
        CORA_FAIL(Error::wrong_value, "lean: the layer kind '" + layer.kind + "' is unknown");
    }
    return j;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The layers ---------------------------------------------------------------------------------

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

// the fit of order k given: its coefficients and its errors
NnLayer NnLayer::poly(const cora::Tensor &coeffs, const cora::Tensor &d) {
    NnLayer layer;
    layer.kind = "poly";
    layer.W = coeffs;
    layer.b = d;
    return layer;
}

// The lines ---------------------------------------------------------------------------------------

NnLine relaxation(const std::string &kind, const cora::Tensor &l, const cora::Tensor &u,
                  const cora::Tensor &alpha) {
    std::vector<Operand> args{{"kind", kind}, {"l", Tensor(l)}, {"u", Tensor(u)}};
    if (alpha.defined()) args.push_back({"alpha", Tensor(alpha)});
    // the depths of exp's reduction and of the log and root iterations (unused at nearest)
    if (kind == "exp") args.push_back({"k", int64_t(30)});
    if (kind == "exp" || kind == "reciprocal") args.push_back({"p", int64_t(60)});
    const Reply r = ask("nnLayer.relaxation", args);
    return {r.get<Tensor>("slope"), r.get<Tensor>("offset"), r.get<Tensor>("error")};
}

NnLine product_line(const cora::Tensor &c, const cora::Tensor &G, const cora::Tensor &dr) {
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

// The evaluations ----------------------------------------------------------------------------

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

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
