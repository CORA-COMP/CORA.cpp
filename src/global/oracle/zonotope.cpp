// zonotope - lean::Zonotope, a zonotope of exact values plus an error box, computed by CORALean
//
// See also: lean/tensor.h, lean/oracle.h

#include "cora/global/oracle/zonotope.h"
#include "cora/global/error.h"

#include "cora/global/oracle/oracle.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void aux_same_dtype(const std::string &a, const std::string &b) {
    if (a != b)
        CORA_FAIL(Error::wrong_value, "lean: an operation on the dtypes '" + a + "' and '" + b +
                                    "' is not allowed; round one with roundTo first");
}

Interval aux_zero_box(int64_t n, const std::string &dtype) {
    Json zero = Json::object(), data = Json::array();
    for (int64_t i = 0; i < n; ++i) data.push("0*2^0");
    zero.set("rows", n).set("cols", int64_t(1)).set("data", data);
    return Interval::from_json(dtype, Json::object().set("inf", zero).set("sup", zero));
}

using Args = std::vector<std::pair<std::string, Json>>;

// The response zonotope of `op` on Z in dtype `dt`, with the extra members `a`.
Zonotope aux_apply(const char *op, const Zonotope &Z, const std::string &dt, const Args &a) {
    Json request = Json::object();
    request.set("op", op).set("dtype", dt).set("Z", Z.to_json());
    for (const auto &arg : a) request.set(arg.first, arg.second);  // the op's own operands
    return Zonotope::from_json(dt, call(request).at("Z"));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction -------------------------------------------------------------------------------

Zonotope::Zonotope(const cora::Tensor &c, const cora::Tensor &G)
    : c_(Tensor::from(c)), G_(Tensor::from(G)), E_(aux_zero_box(c.shape()[0], dtype())) {}

Zonotope::Zonotope(Tensor c, Tensor G, Interval E)
    : c_(std::move(c)), G_(std::move(G)), E_(std::move(E)) {
    aux_same_dtype(c_.dtype(), G_.dtype());
    aux_same_dtype(c_.dtype(), E_.inf().dtype());
}

Zonotope Zonotope::nominal() const { return {c_, G_, aux_zero_box(c_.rows(), c_.dtype())}; }

Zonotope Zonotope::enclosure() const {
    return aux_apply("enclosure", *this, c_.dtype(), {});
}

Zonotope Zonotope::round_to(const std::string &dtype, const std::string &mode) const {
    return aux_apply("roundTo", *this, dtype, {{"mode", Json(mode)}});
}

// Operations ---------------------------------------------------------------------------------

Zonotope Zonotope::linear_map(const Tensor &M) const {
    aux_same_dtype(c_.dtype(), M.dtype());
    return aux_apply("mtimes", *this, c_.dtype(), {{"M", M.to_json()}});
}

// An interval matrix maps the error box soundly, so the op is separate.
Zonotope Zonotope::linear_map(const Interval &M) const {
    aux_same_dtype(c_.dtype(), M.inf().dtype());
    return aux_apply("mtimesInterval", *this, c_.dtype(), {{"M", M.to_json()}});
}

Zonotope Zonotope::plus(const Zonotope &Z2) const {
    aux_same_dtype(c_.dtype(), Z2.center().dtype());
    return aux_apply("plus", *this, c_.dtype(), {{"Z2", Z2.to_json()}});
}

Zonotope Zonotope::plus(const Interval &I) const {
    aux_same_dtype(c_.dtype(), I.inf().dtype());
    return aux_apply("plusBox", *this, c_.dtype(), {{"I", I.to_json()}});
}

// The operands must share one dtype, else the call throws before the oracle is asked.
Zonotope Zonotope::linear_combination(const Zonotope &Z2) const {
    aux_same_dtype(c_.dtype(), Z2.center().dtype());
    return aux_apply("linComb", *this, c_.dtype(), {{"Z2", Z2.to_json()}});
}

Zonotope Zonotope::reduce(int order, const std::string &method) const {
    if (method != "girard" && method != "combastel")
        CORA_FAIL(Error::wrong_value, "lean: the reduction method '" + method +
                                    "' is unknown; use 'girard' or 'combastel'");
    return aux_apply("reduce", *this, c_.dtype(),
                     {{"order", Json(static_cast<int64_t>(order))}, {"method", Json(method)}});
}

Interval Zonotope::interval() const {
    Json request = Json::object();
    request.set("op", "interval").set("dtype", c_.dtype()).set("Z", to_json());
    return Interval::from_json(c_.dtype(), call(request).at("interval"));
}

std::pair<cora::Zonotope, cora::Interval> Zonotope::gather() const {
    const auto [inf, sup] = E_.gather();
    return {cora::Zonotope(c_.gather(), G_.gather()), cora::Interval(inf, sup)};
}

// Crossing -----------------------------------------------------------------------------------

// the wire form is {c, G, E}; gather() above leaves the lean world

Json Zonotope::to_json() const {
    Json j = Json::object();
    j.set("c", c_.to_json()).set("G", G_.to_json()).set("E", E_.to_json());
    return j;
}

Zonotope Zonotope::from_json(const std::string &dtype, const Json &j) {
    return {Tensor::from_json(dtype, j.at("c")), Tensor::from_json(dtype, j.at("G")),
            Interval::from_json(dtype, j.at("E"))};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
