// zonotope - lean::Zonotope, a zonotope of exact values plus an error box, computed by CORALean
//
// See also: lean/tensor.h, lean/oracle.h

#include "global/oracle/zonotope.h"

#include "global/oracle/oracle.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void aux_sameDType(const std::string &a, const std::string &b) {
    if (a != b)
        throw std::invalid_argument("lean: an operation on the dtypes '" + a + "' and '" + b +
                                    "' is not allowed; round one with roundTo first");
}

Interval aux_zeroBox(int64_t n, const std::string &dtype) {
    Json zero = Json::object(), data = Json::array();
    for (int64_t i = 0; i < n; ++i) data.push("0*2^0");
    zero.set("rows", n).set("cols", int64_t(1)).set("data", data);
    return Interval::fromJson(dtype, Json::object().set("inf", zero).set("sup", zero));
}

using Args = std::vector<std::pair<std::string, Json>>;

// The response zonotope of `op` on Z in dtype `dt`, with the extra members `a`.
Zonotope aux_apply(const char *op, const Zonotope &Z, const std::string &dt, const Args &a) {
    Json request = Json::object();
    request.set("op", op).set("dtype", dt).set("Z", Z.toJson());
    for (const auto &arg : a) request.set(arg.first, arg.second);  // the op's own operands
    return Zonotope::fromJson(dt, call(request).at("Z"));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction -------------------------------------------------------------------------------

Zonotope::Zonotope(const cora::Tensor &c, const cora::Tensor &G)
    : c(Tensor::from(c)), G(Tensor::from(G)), E(aux_zeroBox(c.shape()[0], dtype())) {}

Zonotope::Zonotope(Tensor c, Tensor G, Interval E)
    : c(std::move(c)), G(std::move(G)), E(std::move(E)) {
    aux_sameDType(this->c.dtype(), this->G.dtype());
    aux_sameDType(this->c.dtype(), this->E.inf.dtype());
}

Zonotope Zonotope::nominal() const { return {c, G, aux_zeroBox(c.rows(), c.dtype())}; }

Zonotope Zonotope::enclosure() const {
    return aux_apply("enclosure", *this, c.dtype(), {});
}

Zonotope Zonotope::roundTo(const std::string &dtype, const std::string &mode) const {
    return aux_apply("roundTo", *this, dtype, {{"mode", Json(mode)}});
}

// Operations ---------------------------------------------------------------------------------

Zonotope Zonotope::mtimes(const Tensor &M) const {
    aux_sameDType(c.dtype(), M.dtype());
    return aux_apply("mtimes", *this, c.dtype(), {{"M", M.toJson()}});
}

// An interval matrix maps the error box soundly, so the op is separate.
Zonotope Zonotope::mtimes(const Interval &M) const {
    aux_sameDType(c.dtype(), M.inf.dtype());
    return aux_apply("mtimesInterval", *this, c.dtype(), {{"M", M.toJson()}});
}

Zonotope Zonotope::plus(const Zonotope &Z2) const {
    aux_sameDType(c.dtype(), Z2.c.dtype());
    return aux_apply("plus", *this, c.dtype(), {{"Z2", Z2.toJson()}});
}

Zonotope Zonotope::plus(const Interval &I) const {
    aux_sameDType(c.dtype(), I.inf.dtype());
    return aux_apply("plusBox", *this, c.dtype(), {{"I", I.toJson()}});
}

// The operands must share one dtype, else the call throws before the oracle is asked.
Zonotope Zonotope::linComb(const Zonotope &Z2) const {
    aux_sameDType(c.dtype(), Z2.c.dtype());
    return aux_apply("linComb", *this, c.dtype(), {{"Z2", Z2.toJson()}});
}

Zonotope Zonotope::reduce(int order, const std::string &method) const {
    if (method != "girard" && method != "combastel")
        throw std::invalid_argument("lean: the reduction method '" + method +
                                    "' is unknown; use 'girard' or 'combastel'");
    return aux_apply("reduce", *this, c.dtype(),
                     {{"order", Json(static_cast<int64_t>(order))}, {"method", Json(method)}});
}

Interval Zonotope::interval() const {
    Json request = Json::object();
    request.set("op", "interval").set("dtype", c.dtype()).set("Z", toJson());
    return Interval::fromJson(c.dtype(), call(request).at("interval"));
}

std::pair<cora::Zonotope, cora::Interval> Zonotope::gather() const {
    const auto [inf, sup] = E.gather();
    return {cora::Zonotope(c.gather(), G.gather()), cora::Interval(inf, sup)};
}

// Crossing -----------------------------------------------------------------------------------

// the wire form is {c, G, E}; gather() above leaves the lean world

Json Zonotope::toJson() const {
    Json j = Json::object();
    j.set("c", c.toJson()).set("G", G.toJson()).set("E", E.toJson());
    return j;
}

Zonotope Zonotope::fromJson(const std::string &dtype, const Json &j) {
    return {Tensor::fromJson(dtype, j.at("c")), Tensor::fromJson(dtype, j.at("G")),
            Interval::fromJson(dtype, j.at("E"))};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
