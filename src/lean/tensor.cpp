// tensor - lean::Tensor and lean::Interval, matrices of exact values in a CORALean dtype
//
// See also: lean/oracle.h

#include "lean/tensor.h"

#include "lean/oracle.h"

#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The double x as "m*2^e" with an odd mantissa (zero is "0*2^0").
std::string aux_encode(double x) {
    if (!std::isfinite(x))
        throw std::invalid_argument("lean: NaN and infinity are not lean values");
    if (x == 0) return "0*2^0";
    int e;
    const double frac = std::frexp(x, &e);  // x = frac * 2^e with 0.5 <= |frac| < 1
    long long m = static_cast<long long>(std::ldexp(frac, 53));
    e -= 53;
    while (m % 2 == 0) { m /= 2; ++e; }
    return std::to_string(m) + "*2^" + std::to_string(e);
}

// The double of "m*2^e"; throws if it is not exactly one.
double aux_decode(const std::string &s) {
    const std::size_t star = s.find("*2^");
    if (star == std::string::npos)
        throw std::invalid_argument("lean: expected a value m*2^e, got '" + s + "'");
    const std::string digits = s.substr(0, star);
    const bool negative = !digits.empty() && digits[0] == '-';
    // the mantissa has at most 38 digits so that it fits 128 bits
    unsigned __int128 m = 0;
    for (std::size_t i = negative ? 1 : 0; i < digits.size(); ++i) {
        if (m > (~static_cast<unsigned __int128>(0)) / 20)
            throw std::range_error("lean: not a double (mantissa too large): " + s);
        m = m * 10 + static_cast<unsigned>(digits[i] - '0');
    }
    // trailing zero bits move into the exponent, so 53 bits decide the mantissa
    long long e = std::stoll(s.substr(star + 3));
    while (m != 0 && m % 2 == 0) { m /= 2; ++e; }
    if (m >> 53)
        throw std::range_error("lean: not a double (needs over 53 bits): " + s);
    const double x = std::ldexp(static_cast<double>(m), static_cast<int>(e));
    if (!std::isfinite(x) || (m != 0 && std::abs(x) < 0x1p-1022))
        throw std::range_error("lean: not a normal double (out of range): " + s);
    return negative ? -x : x;
}

int64_t aux_columns(const cora::Tensor &x) {
    const std::vector<int64_t> shape = x.shape();
    if (shape.size() != 2)
        throw std::invalid_argument("lean: a lean tensor is a single matrix, not a batch");
    return shape[1];
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Tensor ----------------------------------------------------------------------------------

Tensor Tensor::from(const cora::Tensor &x) {
    Tensor T;
    T.dtype_ = lean::dtype();
    T.rows_ = x.shape()[0];
    T.cols_ = aux_columns(x);
    for (const double v : x.data()) T.data_.push_back(aux_encode(v));
    if (T.dtype_ == "binary64") return T;
    Json request = Json::object();
    request.set("op", "from").set("dtype", T.dtype_).set("x", T.toJson());
    call(request);  // the oracle throws if a value is not representable
    return T;
}

// Crossing -----------------------------------------------------------------------------------

std::string Tensor::encode(double x) { return aux_encode(x); }

cora::Tensor Tensor::gather() const {
    std::vector<double> values;
    for (const std::string &s : data_) values.push_back(aux_decode(s));
    return cora::Tensor::fromData(values, {rows_, cols_});
}

Rounded Tensor::roundTo(const std::string &target, const std::string &mode) const {
    if (mode != "nearest" && mode != "down" && mode != "up")
        throw std::invalid_argument("lean: the rounding mode '" + mode +
                                    "' is unknown; use 'nearest', 'down' or 'up'");
    Json request = Json::object();
    request.set("op", "roundTo").set("dtype", target).set("mode", mode).set("x", toJson());
    // the oracle answers the value and an enclosure of the rounding error
    const Json response = call(request);
    const Json &error = response.at("error");
    return {fromJson(target, response.at("value")), Interval::fromJson(target, error)};
}

Json Tensor::toJson() const {
    Json j = Json::object(), values = Json::array();
    for (const std::string &s : data_) values.push(s);
    j.set("rows", rows_).set("cols", cols_).set("data", values);
    return j;
}

// The wire form is row-major "m*2^e" strings.
Tensor Tensor::fromJson(const std::string &dtype, const Json &j) {
    Tensor T;
    T.dtype_ = dtype;
    T.rows_ = std::stoll(j.at("rows").text);
    T.cols_ = std::stoll(j.at("cols").text);
    for (const Json &v : j.at("data").items) T.data_.push_back(v.text);
    if (static_cast<int64_t>(T.data_.size()) != T.rows_ * T.cols_)
        throw std::runtime_error("lean: a response matrix has the wrong number of values");
    return T;
}

// Interval --------------------------------------------------------------------------------

Interval::Interval(Tensor inf, Tensor sup) : inf(std::move(inf)), sup(std::move(sup)) {
    if (this->inf.dtype() != this->sup.dtype())
        throw std::invalid_argument("lean: the bounds of an interval must share one dtype");
}

Interval Interval::from(const cora::Tensor &inf, const cora::Tensor &sup) {
    return {Tensor::from(inf), Tensor::from(sup)};
}

std::pair<cora::Tensor, cora::Tensor> Interval::gather() const {
    return {inf.gather(), sup.gather()};
}

Json Interval::toJson() const {
    Json j = Json::object();
    j.set("inf", inf.toJson()).set("sup", sup.toJson());
    return j;
}

Interval Interval::fromJson(const std::string &dtype, const Json &j) {
    return {Tensor::fromJson(dtype, j.at("inf")), Tensor::fromJson(dtype, j.at("sup"))};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
