// tensor - lean::Tensor and lean::Interval, matrices of exact values in a CORALean dtype
//
// Syntax:     lean::Tensor T = lean::Tensor::from(x);   auto [v, err] = T.roundTo("dyadic:8");
//             cora::Tensor y = value.gather();
// Values:     exact rationals m*2^e, held as text; nothing converts to double unless it is exact
// Errors:     an operation on two dtypes throws; so does a conversion that is not exact
// See also:   lean/oracle.h, lean/zonotope.h

#pragma once

#include "global/oracle/json.h"
#include "global/tensor/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

struct Rounded;

/// A matrix (rows x cols) of exact values in one dtype.
class Tensor {
  public:
    /// From a cora::Tensor (a column or a matrix) of the current dtype; throws if a value is not
    /// representable in it.
    static Tensor from(const cora::Tensor &x);

    /// The exact text "m*2^e" of a double (the wire form of a scalar).
    static std::string encode(double x);

    /// Back to a cora::Tensor (double); throws if a value is not a double.
    cora::Tensor gather() const;

    /// The values rounded into `dtype` (mode "nearest", "down" or "up") and the error x - value
    /// enclosed outward.
    Rounded roundTo(const std::string &dtype, const std::string &mode = "nearest") const;

    const std::string &dtype() const { return dtype_; }
    int64_t rows() const { return rows_; }
    int64_t cols() const { return cols_; }

    /// The wire form {"rows", "cols", "data"} and back (`dtype` names the values' dtype).
    Json toJson() const;
    static Tensor fromJson(const std::string &dtype, const Json &j);

  private:
    std::string dtype_;
    int64_t rows_ = 0, cols_ = 0;
    std::vector<std::string> data_;  // row-major "m*2^e"
};

/// A box of lean values: inf <= x <= sup.
class Interval {
  public:
    Tensor inf, sup;

    Interval(Tensor inf, Tensor sup);

    /// The exact interval of double bounds (values in the current dtype).
    static Interval from(const cora::Tensor &inf, const cora::Tensor &sup);

    /// Back to double bounds; throws if a bound is not a double.
    std::pair<cora::Tensor, cora::Tensor> gather() const;

    Json toJson() const;
    static Interval fromJson(const std::string &dtype, const Json &j);
};

/// The result of roundTo: the rounded values and an interval enclosing x - value.
struct Rounded {
    Tensor value;
    Interval error;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
