// mat_zonotope - lean::MatZonotope, a matrix zonotope of exact values plus an error box (CORALean)
//
// See also: lean/request.h, lean/zonotope.h

#include "cora/global/oracle/mat_zonotope.h"
#include "cora/global/error.h"

#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The error box of an exact set: a rows x cols matrix of zeros in the current dtype.
Interval aux_zero_box(int64_t rows, int64_t cols, const std::string &dtype) {
    Json zero = Json::object(), data = Json::array();
    for (int64_t i = 0; i < rows * cols; ++i) data.push("0*2^0");
    zero.set("rows", rows).set("cols", cols).set("data", data);
    return Interval::from_json(dtype, Json::object().set("inf", zero).set("sup", zero));
}

// The generator matrices of a tensor (h, rows, cols), in the current dtype.
std::vector<Tensor> aux_generators(const cora::Tensor &G) {
    std::vector<Tensor> list;
    list.reserve(static_cast<std::size_t>(G.size(-3)));
    for (int64_t i = 0; i < G.size(-3); ++i)
        list.push_back(Tensor::from(G.slice(-3, i, i + 1).reshape({G.size(-2), G.size(-1)})));
    return list;
}

// The matrix zonotope `mZ` of the response to `op` on the operands.
MatZonotope aux_set(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<MatZonotope>("mZ");
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction -------------------------------------------------------------------------------

MatZonotope::MatZonotope(const cora::MatZonotope &A)
    : c_(Tensor::from(A.center())), G_(aux_generators(A.generators())),
      E_(aux_zero_box(A.rows(), A.cols(), dtype())) {}

MatZonotope::MatZonotope(Tensor c, std::vector<Tensor> G, Interval E)
    : c_(std::move(c)), G_(std::move(G)), E_(std::move(E)) {}

// Operations ---------------------------------------------------------------------------------

Tensor MatZonotope::center_value() const {
    return ask("matZonotope.center", {{"mZ", *this}}).get<Tensor>("value");
}
Interval MatZonotope::interval_matrix() const {
    return ask("matZonotope.intervalMatrix", {{"mZ", *this}}).get<Interval>("IM");
}

MatZonotope MatZonotope::plus(const MatZonotope &B) const {
    return aux_set("matZonotope.plus", {{"mZ", *this}, {"mZ2", B}});
}
MatZonotope MatZonotope::matmul(const MatZonotope &B) const {
    return aux_set("matZonotope.mtimes", {{"mZ", *this}, {"mZ2", B}});
}
// The transpose and the powers of a square set.
MatZonotope MatZonotope::transpose() const {
    return aux_set("matZonotope.transpose", {{"mZ", *this}});
}
MatZonotope MatZonotope::matrix_power(int64_t exponent) const {
    return aux_set("matZonotope.mpower", {{"mZ", *this}, {"exponent", exponent}});
}
MatZonotope MatZonotope::powers(const MatZonotope &P, int64_t k) const {
    return aux_set("matZonotope.powers", {{"mZ", *this}, {"mZ2", P}, {"k", k}});
}

// The exponential has one op for CORA's expm and expmInd.
MatZonotope MatZonotope::expm(int64_t max_order) const {
    return aux_set("matZonotope.expm", {{"mZ", *this}, {"maxOrder", max_order}});
}
MatZonotope MatZonotope::expm_independent(int64_t max_order) const {
    return aux_set("matZonotope.expmInd", {{"mZ", *this}, {"maxOrder", max_order}});
}

// The norm of the box, and the exact test of a point or the origin.
double MatZonotope::norm(const std::string &type, int64_t steps) const {
    return ask("matZonotope.norm", {{"mZ", *this}, {"type", type}, {"steps", steps}})
        .get<double>("value");
}
bool MatZonotope::representsa(const std::string &type, double tol) const {
    return ask("matZonotope.representsa", {{"mZ", *this}, {"type", type}, {"tol", number(tol)}})
        .get<bool>("value");
}
Zonotope MatZonotope::zonotope() const {
    return ask("matZonotope.zonotope", {{"mZ", *this}}).get<Zonotope>("Z");
}

// Crossing -----------------------------------------------------------------------------------

// The wire form is {c, G, E}: the center, a list of generator matrices and the error box.
Json MatZonotope::to_json() const {
    Json generators = Json::array();
    for (const Tensor &G : G_) generators.push(G.to_json());
    Json j = Json::object();
    j.set("c", c_.to_json()).set("G", generators).set("E", E_.to_json());
    return j;
}

MatZonotope MatZonotope::from_json(const std::string &dtype, const Json &j) {
    std::vector<Tensor> generators;
    for (const Json &G : j.at("G").items) generators.push_back(Tensor::from_json(dtype, G));
    return {Tensor::from_json(dtype, j.at("c")), std::move(generators),
            Interval::from_json(dtype, j.at("E"))};
}

// The generators are stacked as (h, rows, cols); a set without generators has h = 0.
std::pair<cora::MatZonotope, cora::Interval> MatZonotope::gather() const {
    const cora::Tensor c = c_.gather();
    std::vector<cora::Tensor> parts;
    parts.reserve(G_.size());
    for (const Tensor &G : G_) parts.push_back(G.gather());
    const cora::Tensor G =
        parts.empty() ? c.zeros_like({0, c.size(-2), c.size(-1)}) : cora::Tensor::stack(parts, 0);
    const auto [inf, sup] = E_.gather();
    return {cora::MatZonotope(c, G), cora::Interval(inf, sup)};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
