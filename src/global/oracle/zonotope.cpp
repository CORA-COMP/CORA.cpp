// zonotope - lean::Zonotope, a zonotope of exact values plus an error box, computed by CORALean
//
// See also: lean/tensor.h, lean/oracle.h

#include "cora/global/oracle/zonotope.h"
#include "cora/global/error.h"

#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/request.h"

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

// The zonotope `Z` of the response to `op` on the operands, or both halves `Z1`, `Z2`.
Zonotope aux_set(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<Zonotope>("Z");
}

std::pair<Zonotope, Zonotope> aux_halves(const char *op, std::initializer_list<Operand> operands) {
    const Reply r = ask(op, operands);
    return {r.get<Zonotope>("Z1"), r.get<Zonotope>("Z2")};
}

bool aux_bool(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<bool>("value");
}

// What `contains` and `isIntersecting` answer, as Certified.
Certified aux_contains(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).proved_contains();
}

Certified aux_disjoint(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).proved_disjoint();
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

Zonotope Zonotope::enclosure() const { return aux_set("zonotope.enclosure", {{"Z", *this}}); }

// roundTo is the one op whose result is in another dtype than its operand.
Zonotope Zonotope::round_to(const std::string &dtype, const std::string &mode) const {
    Json request = Json::object();
    request.set("op", "roundTo").set("dtype", dtype).set("mode", mode).set("Z", to_json());
    return from_json(dtype, call(request).at("Z"));
}

// Linear and Minkowski operations ------------------------------------------------------------

Zonotope Zonotope::linear_map(const Tensor &M) const {
    return aux_set("zonotope.mtimes", {{"Z", *this}, {"M", M}});
}

// An interval matrix maps the error box soundly, so the op is separate.
Zonotope Zonotope::linear_map(const Interval &M) const {
    return aux_set("zonotope.mtimesInterval", {{"Z", *this}, {"M", M}});
}

Zonotope Zonotope::plus(const Zonotope &Z2) const {
    return aux_set("zonotope.plus", {{"Z", *this}, {"Z2", Z2}});
}

Zonotope Zonotope::plus(const Interval &I) const {
    return aux_set("zonotope.plusBox", {{"Z", *this}, {"I", I}});
}

// The sums that CORA's linComb and exactPlus name.
Zonotope Zonotope::linear_combination(const Zonotope &Z2) const {
    return aux_set("zonotope.linComb", {{"Z", *this}, {"Z2", Z2}});
}

Zonotope Zonotope::exact_plus(const Zonotope &Z2, int k) const {
    return aux_set("zonotope.exactPlus", {{"Z", *this}, {"Z2", Z2}, {"k", k}});
}

Zonotope Zonotope::reduce(int order, const std::string &method) const {
    if (method != "girard" && method != "combastel")
        CORA_FAIL(Error::wrong_value, "lean: the reduction method '" + method +
                                          "' is unknown; use 'girard' or 'combastel'");
    return aux_set("zonotope.reduce", {{"Z", *this}, {"order", order}, {"method", method}});
}

// Set operations -----------------------------------------------------------------------------

Zonotope Zonotope::abs() const { return aux_set("zonotope.abs", {{"Z", *this}}); }

Zonotope Zonotope::intersect(const Zonotope &Z2) const {
    return aux_set("zonotope.and", {{"Z", *this}, {"Z2", Z2}});
}
Zonotope Zonotope::convex_hull(const Zonotope &Z2) const {
    return aux_set("zonotope.convHull", {{"Z", *this}, {"Z2", Z2}});
}
Zonotope Zonotope::enclose(const Zonotope &Z2) const {
    return aux_set("zonotope.enclose", {{"Z", *this}, {"Z2", Z2}});
}
Zonotope Zonotope::cart_prod(const Zonotope &Z2) const {
    return aux_set("zonotope.cartProd", {{"Z", *this}, {"Z2", Z2}});
}

Zonotope Zonotope::compact(const std::string &method, double tol) const {
    return aux_set("zonotope.compact", {{"Z", *this}, {"method", method}, {"tol", number(tol)}});
}

// Intersection with strips, and the quadratic and cubic maps.
Zonotope Zonotope::intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                   const Tensor &Lambda) const {
    return aux_set("zonotope.intersectStrip",
                   {{"Z", *this}, {"C", C}, {"phi", phi}, {"y", y}, {"Lambda", Lambda}});
}

Zonotope Zonotope::quad_map(const std::vector<Tensor> &Q) const {
    return aux_set("zonotope.quadMap", {{"Z", *this}, {"Q", Q}});
}
Zonotope Zonotope::cub_map(const std::vector<std::vector<Tensor>> &T) const {
    return aux_set("zonotope.cubMap", {{"Z", *this}, {"T", T}});
}

// The operations without a zonotope operand: the dtype is the current one.
std::optional<Zonotope> Zonotope::enclose_many(const std::vector<Zonotope> &Zs,
                                               const Tensor &direction, int p) {
    return ask("zonotope.encloseMany", {{"Zs", Zs}, {"direction", direction}, {"p", p}})
        .get<std::optional<Zonotope>>("Z");
}

Zonotope Zonotope::enclose_points(const Tensor &points, const Tensor &U,
                                  const std::string &method) {
    return aux_set("zonotope.enclosePoints", {{"points", points}, {"U", U}, {"method", method}});
}

std::vector<int64_t> Zonotope::filter_out(const std::vector<Zonotope> &Zs) {
    return ask("zonotope.filterOut", {{"Zs", Zs}}).get<std::vector<int64_t>>("value");
}

Zonotope Zonotope::origin(int64_t n) { return aux_set("zonotope.origin", {{"n", n}}); }

// Selection and reshaping --------------------------------------------------------------------

Zonotope Zonotope::project(const std::vector<int64_t> &dims) const {
    return aux_set("zonotope.project", {{"Z", *this}, {"dims", dims}});
}
Zonotope Zonotope::lift(int64_t N, const std::vector<int64_t> &dims) const {
    return aux_set("zonotope.lift", {{"Z", *this}, {"N", N}, {"dims", dims}});
}
Zonotope Zonotope::project_high_dim(int64_t N, const std::vector<int64_t> &dims) const {
    return aux_set("zonotope.projectHighDim", {{"Z", *this}, {"N", N}, {"dims", dims}});
}
Zonotope Zonotope::rotate(int p, int i, int j, double angle) const {
    return aux_set("zonotope.rotate",
                   {{"Z", *this}, {"p", p}, {"i", i}, {"j", j}, {"angle", angle}});
}

std::pair<Zonotope, Zonotope> Zonotope::split(int64_t gen) const {
    return aux_halves("zonotope.split", {{"Z", *this}, {"gen", gen}});
}
std::pair<Zonotope, Zonotope> Zonotope::split_first_gen() const {
    return aux_halves("zonotope.splitFirstGen", {{"Z", *this}});
}
std::pair<Zonotope, Zonotope> Zonotope::split_longest_gen() const {
    return aux_halves("zonotope.splitLongestGen", {{"Z", *this}});
}

// Predicates ---------------------------------------------------------------------------------

bool Zonotope::constr_sat(const Tensor &C, const Tensor &d) const {
    return aux_bool("zonotope.constrSat", {{"Z", *this}, {"C", C}, {"d", d}});
}
bool Zonotope::is_bounded(double tol) const {
    return aux_bool("zonotope.isBounded", {{"Z", *this}, {"tol", number(tol)}});
}
bool Zonotope::is_full_dim(double tol) const {
    return aux_bool("zonotope.isFullDim", {{"Z", *this}, {"tol", number(tol)}});
}
bool Zonotope::isequal(const Zonotope &Z2, double tol) const {
    return aux_bool("zonotope.isequal", {{"Z", *this}, {"Z2", Z2}, {"tol", number(tol)}});
}
bool Zonotope::representsa(const std::string &type, double tol) const {
    return aux_bool("zonotope.representsa", {{"Z", *this}, {"type", type}, {"tol", number(tol)}});
}

// Certified predicates: a true proves what the operation names.
Certified Zonotope::contains(const Zonotope &Z2, double tol) const {
    return aux_contains("zonotope.contains", {{"Z", *this}, {"Z2", Z2}, {"tol", number(tol)}});
}
Certified Zonotope::is_intersecting(const Zonotope &Z2, double tol) const {
    return aux_disjoint("zonotope.isIntersecting",
                        {{"Z", *this}, {"Z2", Z2}, {"tol", number(tol)}});
}

// Numbers ------------------------------------------------------------------------------------

double Zonotope::norm(int type, const std::string &mode) const {
    return ask("zonotope.norm", {{"Z", *this}, {"type", type}, {"mode", mode}})
        .get<double>("normSq");
}
double Zonotope::support_function(const Tensor &dir, const std::string &type) const {
    return ask("zonotope.supportFunc", {{"Z", *this}, {"dir", dir}, {"type", type}})
        .get<double>("value");
}
double Zonotope::radius(int k) const {
    return ask("zonotope.radius", {{"Z", *this}, {"k", k}}).get<double>("value");
}
std::optional<double> Zonotope::minnorm(int p) const {
    return ask("zonotope.minnorm", {{"Z", *this}, {"p", p}}).get<std::optional<double>>("value");
}
Tensor Zonotope::generator_length(int k) const {
    return ask("zonotope.generatorLength", {{"Z", *this}, {"k", k}}).get<Tensor>("value");
}
// The filter lengths default to CORA's n + 5 and n + 3.
std::optional<Tensor> Zonotope::dominant_directions() const {
    const int n = static_cast<int>(c_.rows());
    return dominant_directions(n + 5, n + 3);
}
std::optional<Tensor> Zonotope::dominant_directions(int filter_length1, int filter_length2) const {
    return ask("zonotope.dominantDirections", {{"Z", *this}, {"filterLength1", filter_length1},
                                               {"filterLength2", filter_length2}})
        .get<std::optional<Tensor>>("value");
}

Vertices Zonotope::vertices() const {
    const Reply r = ask("zonotope.vertices", {{"Z", *this}});
    return {r.get<Tensor>("V"), r.get<Interval>("E")};
}

Interval Zonotope::interval() const {
    return ask("zonotope.box", {{"Z", *this}}).get<Interval>("I");
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
