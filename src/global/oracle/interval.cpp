// interval - lean::Interval, a box (or interval matrix) of exact values, and its oracle operations
//
// See also: lean/request.h, lean/zonotope.h
//
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/global/oracle/interval.h"
#include "cora/global/error.h"

#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The box `I` of the response to `op` on I and I2.
Interval aux_pair(const char *op, const Interval &I, const Interval &I2) {
    return ask(op, {{"I", I}, {"I2", I2}}).get<Interval>("I");
}

// The box `I` of the response to `op` on the operands.
Interval aux_box(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<Interval>("I");
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

Interval::Interval(Tensor inf, Tensor sup) : inf_(std::move(inf)), sup_(std::move(sup)) {
    if (inf_.dtype() != sup_.dtype())
        CORA_FAIL(Error::wrong_value, "lean: the bounds of an interval must share one dtype");
}

Interval::Interval(const cora::Interval &I) : Interval(from(I.inf(), I.sup())) {}

Interval Interval::from(const cora::Tensor &inf, const cora::Tensor &sup) {
    return {Tensor::from(inf), Tensor::from(sup)};
}

std::pair<cora::Tensor, cora::Tensor> Interval::gather() const {
    return {inf_.gather(), sup_.gather()};
}

Json Interval::to_json() const {
    Json j = Json::object();
    j.set("inf", inf_.to_json()).set("sup", sup_.to_json());
    return j;
}

// The wire form is {inf, sup}, two matrices.
Interval Interval::from_json(const std::string &dtype, const Json &j) {
    return {Tensor::from_json(dtype, j.at("inf")), Tensor::from_json(dtype, j.at("sup"))};
}

// The empty set, the origin and the unbounded box of n dimensions.
Interval Interval::empty(int64_t n) { return aux_box("interval.empty", {{"n", n}}); }
Interval Interval::origin(int64_t n) { return aux_box("interval.origin", {{"n", n}}); }
Interval Interval::unbounded(int64_t n) { return aux_box("interval.Inf", {{"n", n}}); }

Interval Interval::enclose_points(const Tensor &points) {
    return aux_box("interval.enclosePoints", {{"points", points}});
}

Interval Interval::interval_matrix(const Tensor &c, const Tensor &r) {
    return aux_box("interval.intervalMatrix", {{"c", c}, {"r", r}});
}

// Elementwise --------------------------------------------------------------------------------

Interval Interval::abs() const { return aux_box("interval.abs", {{"I", *this}}); }
Interval Interval::uminus() const { return aux_box("interval.uminus", {{"I", *this}}); }
Interval Interval::uplus() const { return aux_box("interval.uplus", {{"I", *this}}); }
Interval Interval::sign() const { return aux_box("interval.sign", {{"I", *this}}); }
Interval Interval::round() const { return aux_box("interval.round", {{"I", *this}}); }

// The functions with a tuning p (series terms or halvings) and, for the inverses, k.
Interval Interval::sqrt(int p) const { return aux_box("interval.sqrt", {{"I", *this}, {"p", p}}); }
Interval Interval::exp(int p) const { return aux_box("interval.exp", {{"I", *this}, {"p", p}}); }
Interval Interval::sin(int p) const { return aux_box("interval.sin", {{"I", *this}, {"p", p}}); }
Interval Interval::cos(int p) const { return aux_box("interval.cos", {{"I", *this}, {"p", p}}); }
Interval Interval::tan(int p) const { return aux_box("interval.tan", {{"I", *this}, {"p", p}}); }
Interval Interval::sinh(int p) const { return aux_box("interval.sinh", {{"I", *this}, {"p", p}}); }
Interval Interval::cosh(int p) const { return aux_box("interval.cosh", {{"I", *this}, {"p", p}}); }
Interval Interval::tanh(int p) const { return aux_box("interval.tanh", {{"I", *this}, {"p", p}}); }

// The inverse functions also take k.
Interval Interval::log(int p, int k) const {
    return aux_box("interval.log", {{"I", *this}, {"p", p}, {"k", k}});
}
Interval Interval::asin(int p, int k) const {
    return aux_box("interval.asin", {{"I", *this}, {"p", p}, {"k", k}});
}
Interval Interval::acos(int p, int k) const {
    return aux_box("interval.acos", {{"I", *this}, {"p", p}, {"k", k}});
}
Interval Interval::atan(int p, int k) const {
    return aux_box("interval.atan", {{"I", *this}, {"p", p}, {"k", k}});
}
Interval Interval::asinh(int p, int k) const {
    return aux_box("interval.asinh", {{"I", *this}, {"p", p}, {"k", k}});
}
// And the hyperbolic ones.
Interval Interval::acosh(int p, int k) const {
    return aux_box("interval.acosh", {{"I", *this}, {"p", p}, {"k", k}});
}
Interval Interval::atanh(int p, int k) const {
    return aux_box("interval.atanh", {{"I", *this}, {"p", p}, {"k", k}});
}
// The two-argument arctangent.
Interval Interval::atan2(const Interval &I2, int p, int k) const {
    return aux_box("interval.atan2", {{"I", *this}, {"I2", I2}, {"p", p}, {"k", k}});
}

// Scaling and powers.
Interval Interval::enlarge(double factor) const {
    return aux_box("interval.enlarge", {{"I", *this}, {"factor", factor}});
}
Interval Interval::power(int64_t exponent) const {
    return aux_box("interval.power", {{"I", *this}, {"exponent", exponent}});
}
Interval Interval::rdivide(const Interval &I2) const {
    return aux_pair("interval.rdivide", *this, I2);
}
Interval Interval::mrdivide(double d) const {
    return aux_box("interval.mrdivide", {{"I", *this}, {"d", d}});
}

// Arithmetic and set operations --------------------------------------------------------------

Interval Interval::plus(const Interval &I2) const { return aux_pair("interval.plus", *this, I2); }
Interval Interval::minus(const Interval &I2) const { return aux_pair("interval.minus", *this, I2); }
Interval Interval::times(const Interval &I2) const { return aux_pair("interval.times", *this, I2); }
Interval Interval::unite(const Interval &I2) const { return aux_pair("interval.or", *this, I2); }
Interval Interval::convex_hull(const Interval &I2) const {
    return aux_pair("interval.convHull", *this, I2);
}
Interval Interval::mink_diff(const Interval &I2) const {
    return aux_pair("interval.minkDiff", *this, I2);
}
Interval Interval::min(const Interval &I2) const { return aux_pair("interval.min", *this, I2); }
Interval Interval::max(const Interval &I2) const { return aux_pair("interval.max", *this, I2); }
Interval Interval::intersect(const Interval &I2) const {
    return aux_pair("interval.and", *this, I2);
}
Interval Interval::enclose(const Interval &I2) const {
    return aux_pair("interval.enclose", *this, I2);
}
Interval Interval::cart_prod(const Interval &I2) const {
    // the product of two boxes
    return aux_pair("interval.cartProd", *this, I2);
}

// The matrix M of mtimes comes first in the request, the box I second.
Interval Interval::linear_map(const Tensor &M) const {
    return aux_box("interval.mtimes", {{"M", M}, {"I", *this}});
}
Interval Interval::linear_map(const Interval &M) const {
    return aux_box("interval.mtimes", {{"M", M}, {"I", *this}});
}
Interval Interval::quad_map(const std::vector<Tensor> &Q) const {
    return aux_box("interval.quadMap", {{"I", *this}, {"Q", Q}});
}

// Predicates ---------------------------------------------------------------------------------

bool Interval::le(const Interval &I2, double tol) const {
    return aux_bool("interval.le", {{"I", *this}, {"I2", I2}, {"tol", number(tol)}});
}
bool Interval::lt(const Interval &I2, double tol) const {
    return aux_bool("interval.lt", {{"I", *this}, {"I2", I2}, {"tol", number(tol)}});
}
bool Interval::isequal(const Interval &I2, double tol) const {
    return aux_bool("interval.isequal", {{"I", *this}, {"I2", I2}, {"tol", number(tol)}});
}
bool Interval::is_full_dim(double tol) const {
    return aux_bool("interval.isFullDim", {{"I", *this}, {"tol", number(tol)}});
}
bool Interval::is_bounded(double tol) const {
    return aux_bool("interval.isBounded", {{"I", *this}, {"tol", number(tol)}});
}
bool Interval::representsa(const std::string &type, double tol) const {
    return aux_bool("interval.representsa", {{"I", *this}, {"type", type}, {"tol", number(tol)}});
}

// Certified predicates: a true proves what the operation names.
Certified Interval::contains(const Interval &I2, double tol) const {
    return aux_contains("interval.contains", {{"I", *this}, {"I2", I2}, {"tol", number(tol)}});
}
Certified Interval::contains(const Zonotope &Z, double tol) const {
    return aux_contains("interval.contains", {{"I", *this}, {"Z", Z}, {"tol", number(tol)}});
}
Certified Interval::is_intersecting(const Interval &I2, double tol) const {
    return aux_disjoint("interval.isIntersecting",
                        {{"I", *this}, {"I2", I2}, {"tol", number(tol)}});
}
Certified Interval::is_intersecting(const Zonotope &Z, double tol) const {
    return aux_disjoint("interval.isIntersecting", {{"I", *this}, {"Z", Z}, {"tol", number(tol)}});
}

// Numbers ------------------------------------------------------------------------------------

Tensor Interval::center() const {
    return ask("interval.center", {{"I", *this}}).get<Tensor>("value");
}
Tensor Interval::rad() const { return ask("interval.rad", {{"I", *this}}).get<Tensor>("value"); }
Tensor Interval::infimum() const {
    return ask("interval.infimum", {{"I", *this}}).get<Tensor>("value");
}
Tensor Interval::supremum() const {
    return ask("interval.supremum", {{"I", *this}}).get<Tensor>("value");
}
// The radius, the norm and the volume are upper bounds.
double Interval::radius(int type, const std::string &mode) const {
    return ask("interval.radius", {{"I", *this}, {"type", type}, {"mode", mode}})
        .get<double>("radiusSq");
}
double Interval::norm(int type, const std::string &mode) const {
    return ask("interval.norm", {{"I", *this}, {"type", type}, {"mode", mode}})
        .get<double>("normSq");
}
double Interval::volume() const {
    return ask("interval.volume", {{"I", *this}}).get<double>("value");
}
double Interval::support_function(const Tensor &dir, const std::string &type) const {
    return ask("interval.supportFunc", {{"I", *this}, {"dir", dir}, {"type", type}})
        .get<double>("value");
}

// Selection and reshaping --------------------------------------------------------------------

Interval Interval::project(const std::vector<int64_t> &dims) const {
    return aux_box("interval.project", {{"I", *this}, {"dims", dims}});
}
Interval Interval::lift(int64_t N, const std::vector<int64_t> &dims) const {
    return aux_box("interval.lift", {{"I", *this}, {"N", N}, {"dims", dims}});
}
Interval Interval::project_high_dim(int64_t N, const std::vector<int64_t> &dims) const {
    return aux_box("interval.projectHighDim", {{"I", *this}, {"N", N}, {"dims", dims}});
}
Interval Interval::partition(int64_t splits, const std::vector<int64_t> &ind) const {
    return aux_box("interval.partition", {{"I", *this}, {"splits", splits}, {"ind", ind}});
}
Interval Interval::dirac(int64_t n) const {
    return aux_box("interval.dirac", {{"I", *this}, {"n", n}});
}

// The two halves, and the point where a ray leaves the box.
std::pair<Interval, Interval> Interval::split(int64_t dim) const {
    const Reply r = ask("interval.split", {{"I", *this}, {"dim", dim}});
    return {r.get<Interval>("I1"), r.get<Interval>("I2")};
}

std::optional<Interval> Interval::boundary_point(const Tensor &dir, const Tensor &start) const {
    return ask("interval.boundaryPoint", {{"I", *this}, {"dir", dir}, {"start", start}})
        .get<std::optional<Interval>>("I");
}

Interval Interval::project_on_hyperplane(const Tensor &c, double d) const {
    return aux_box("interval.projectOnHyperplane", {{"I", *this}, {"c", c}, {"d", d}});
}

// Matrices -----------------------------------------------------------------------------------

// Matrix operations: transpose, product, diagonal and the block constructions.
Interval Interval::transpose() const { return aux_box("interval.transpose", {{"I", *this}}); }
Interval Interval::ctranspose() const { return aux_box("interval.ctranspose", {{"I", *this}}); }
Interval Interval::prod() const { return aux_box("interval.prod", {{"I", *this}}); }
Interval Interval::diag() const { return aux_box("interval.diag", {{"I", *this}}); }
Interval Interval::kron(const Interval &I2) const { return aux_pair("interval.kron", *this, I2); }
Interval Interval::horzcat(const Interval &I2) const {
    return aux_pair("interval.horzcat", *this, I2);
}
Interval Interval::vertcat(const Interval &I2) const {
    return aux_pair("interval.vertcat", *this, I2);
}
Interval Interval::sum(int dim) const {
    return aux_box("interval.sum", {{"I", *this}, {"dim", dim}});
}
Interval Interval::cat(const Interval &I2, int dim) const {
    return aux_box("interval.cat", {{"I", *this}, {"I2", I2}, {"dim", dim}});
}
// Powers and triangular parts of a matrix.
Interval Interval::mpower(int64_t exponent) const {
    return aux_box("interval.mpower", {{"I", *this}, {"exponent", exponent}});
}
Interval Interval::tril(int64_t K) const {
    return aux_box("interval.tril", {{"I", *this}, {"K", K}});
}
Interval Interval::triu(int64_t K) const {
    return aux_box("interval.triu", {{"I", *this}, {"K", K}});
}

// Other classes ------------------------------------------------------------------------------

Interval Interval::reduce(int order, const std::string &method) const {
    return aux_box("interval.reduce", {{"I", *this}, {"order", order}, {"method", method}});
}

Zonotope Interval::zonotope() const {
    return ask("interval.zonotope", {{"I", *this}}).get<Zonotope>("Z");
}

Vertices Interval::vertices() const {
    const Reply r = ask("interval.vertices", {{"I", *this}});
    return {r.get<Tensor>("V"), r.get<Interval>("E")};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
