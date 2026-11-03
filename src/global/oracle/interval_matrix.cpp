// interval_matrix - lean::IntervalMatrix and its oracle operations
//
// See also: lean/interval_matrix.h, lean/request.h

#include "cora/global/oracle/interval_matrix.h"

#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The matrix `value` of the response to `op` on the operands.
Tensor aux_value(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<Tensor>("value");
}

// The interval matrix `IM` of the response to `op` on the operands.
Interval aux_matrix(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<Interval>("IM");
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Matrices of the set.
Tensor IntervalMatrix::abs() const { return aux_value("intervalMatrix.abs", {{"IM", bounds_}}); }
Tensor IntervalMatrix::infimum() const {
    return aux_value("intervalMatrix.infimum", {{"IM", bounds_}});
}
Interval IntervalMatrix::center() const {
    return aux_matrix("intervalMatrix.center", {{"IM", bounds_}});
}
bool IntervalMatrix::contains(const Tensor &M, double tol) const {
    return ask("intervalMatrix.contains", {{"IM", bounds_}, {"M", M}, {"tol", tol}})
        .get<bool>("contains");
}

// Squares, powers and products.
Interval IntervalMatrix::exact_square() const {
    return aux_matrix("intervalMatrix.exactSquare", {{"IM", bounds_}});
}
std::pair<Interval, Interval> IntervalMatrix::dependent_terms(double r) const {
    const Reply reply = ask("intervalMatrix.dependentTerms", {{"IM", bounds_}, {"r", r}});
    return {reply.get<Interval>("intSq"), reply.get<Interval>("intH")};
}
Interval IntervalMatrix::matrix_power(int64_t exponent) const {
    return aux_matrix("intervalMatrix.mpower", {{"IM", bounds_}, {"exponent", exponent}});
}
Interval IntervalMatrix::matmul(const Interval &B) const {
    return aux_matrix("intervalMatrix.mtimes", {{"IM", bounds_}, {"IM2", B}});
}

// Norms.
double IntervalMatrix::infinity_norm() const {
    return ask("intervalMatrix.infNorm", {{"IM", bounds_}}).get<double>("value");
}
double IntervalMatrix::norm(const std::string &type, int64_t steps) const {
    return ask("intervalMatrix.norm", {{"IM", bounds_}, {"type", type}, {"steps", steps}})
        .get<double>("value");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
