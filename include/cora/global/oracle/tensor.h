// tensor - lean::Tensor, a matrix of exact values in a CORALean dtype
//
// Syntax:     lean::Tensor T = lean::Tensor::from(x);   auto [v, err] = T.round_to("dyadic:8");
//             cora::Tensor y = value.gather();
// Values:     exact rationals m*2^e, held as text; nothing converts to double unless it is exact
// Errors:     an operation on two dtypes throws; so does a conversion that is not exact
// See also:   lean/oracle.h, lean/interval.h, lean/zonotope.h

#pragma once

#include "cora/global/oracle/json.h"
#include "cora/global/tensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

struct Rounded;

/// A matrix (rows x cols) of exact values in one dtype.
class Tensor {
  public:
    Tensor() = default;

    /// From a cora::Tensor (a column or a matrix) of the current dtype; throws if a value is not
    /// representable in it. Implicit, so an operation takes a cora::Tensor where it wants one.
    Tensor(const cora::Tensor &x) : Tensor(from(x)) {}
    static Tensor from(const cora::Tensor &x);

    /// The exact text of a double in the current dtype ("m*2^e", "m*10^e" for fixedpoint): the
    /// wire form of a scalar. decode is its inverse; it throws if the text is not a double.
    static std::string encode(double x);
    static double decode(const std::string &text);

    /// Back to a cora::Tensor (double); throws if a value is not a double.
    cora::Tensor gather() const;

    /// The values rounded into `dtype` (mode "nearest", "down" or "up") and the error x - value
    /// enclosed outward.
    Rounded round_to(const std::string &dtype, const std::string &mode = "nearest") const;

    const std::string &dtype() const { return dtype_; }
    int64_t rows() const { return rows_; }
    int64_t cols() const { return cols_; }

    /// The wire form {"rows", "cols", "data"} and back (`dtype` names the values' dtype).
    Json to_json() const;
    static Tensor from_json(const std::string &dtype, const Json &j);

  private:
    std::string dtype_;
    int64_t rows_ = 0, cols_ = 0;
    std::vector<std::string> data_;  // row-major "m*2^e"
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
