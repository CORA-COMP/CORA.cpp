// power - the power of an expression with an integer exponent
//
// Syntax:   StlExpr y = x.power(2);
// Inputs:   k - the exponent, an integer
// Outputs:  y - the power
// See also: stl.h
//
// CORA: specification/@stl/power.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

#include "specification/stl/private/priv_node.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

StlExpr StlExpr::power(double k) const {
    detail::Entry entry;
    CORA_CHECK(k == std::floor(k) && std::fabs(k) < 1e15, Error::not_supported,
               [] { return std::string("power: only integer exponents are supported"); });
    auto node = std::make_shared<StlExpr::Node>();
    node->kind = StlExprKind::pow;
    node->l = node_;
    node->exponent = static_cast<int64_t>(k);
    return StlExpr(std::shared_ptr<const StlExpr::Node>(node));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
