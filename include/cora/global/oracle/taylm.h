// taylm - lean::Taylm, a Taylor model computed by CORALean
//
// Syntax:     lean::Taylm T(cora_T);   lean::Taylm R = T.times(T.plus(2.0), T.max_order());
// Operations: one method per `taylm.*` op of the oracle, named as CORA.cpp names CORA's; times with
//             a model takes the order the product is truncated at (CORA.cpp's is the first
//             model's)
// Parts:      the terms in the oracle's order (sorted by their exponents, as cora::Taylm's map),
//             the remainder and the error box that the sound float operations add; at dtype
//             nearest the box is zero
// See also:   lean/request.h, cora/cont_set/taylm/taylm.h
//
// Authors:       Claude Opus 5.5
// Written:       20-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/taylm/taylm.h"
#include "cora/global/oracle/request.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Taylm {
  public:
    /// One term: the exponents of the monomial and the coefficient.
    using Term = std::pair<cora::Taylm::Monomial, double>;

    /// The model of a cora::Taylm, its terms in the order of its map, no error box.
    explicit Taylm(const cora::Taylm &T);

    int64_t num_vars() const noexcept { return vars_; }
    int max_order() const noexcept { return max_order_; }
    const std::vector<Term> &terms() const noexcept { return terms_; }
    std::array<double, 2> remainder() const noexcept { return rem_; }
    /// The error box: what the roundings of the sound operations cost.
    std::array<double, 2> error() const noexcept { return err_; }

    // Operations -------------------------------------------------------------------------------

    /// The sum and the difference with a model of the same variables, and with a number.
    Taylm plus(const Taylm &Q) const;
    Taylm plus(double s) const;
    Taylm minus(const Taylm &Q) const;
    Taylm minus(double s) const;
    /// The negated model and the model itself.
    Taylm uminus() const;
    Taylm uplus() const;
    /// The product with a model, truncated at max_order, and with a number.
    Taylm times(const Taylm &Q, int max_order) const;
    Taylm times(double s) const;
    /// The k-th power, k >= 0.
    Taylm power(int k) const;
    /// The range over the variables' box (CORA's interval, method "int").
    std::array<double, 2> bounds() const;

    Json to_json() const;

  private:
    Taylm() = default;
    static Taylm from_reply(const Reply &reply);

    int64_t vars_ = 0;
    int max_order_ = 0;
    std::vector<Term> terms_;
    std::array<double, 2> rem_ = {0, 0}, err_ = {0, 0};
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
