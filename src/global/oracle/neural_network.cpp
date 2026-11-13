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
    } else if (layer.kind == "sigmoid" || layer.kind == "tanh") {
        j.set("k", int64_t(layer.k));
    } else if (layer.kind != "relu") {
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
    layer.b = b;
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
