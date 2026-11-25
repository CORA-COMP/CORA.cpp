// stl - the wire form of STL formulas and STL intervals for the oracle's stl ops (CORALean)
//
// Syntax:     Json f = lean::stl_json(phi);   Reply r = lean::ask_stl("stl.desugar", phi);
// Meaning:    a formula, an expression or an interval as CORALean's StlCodec writes it, with the
//             members in the oracle's (alphabetical) order, so that a reply of the oracle and
//             stl_json of CORA.cpp's result have the same dump when they are the same structure;
//             a variable is its 0-based index, the formula has stl_variables(phi) of them
// See also:   lean/request.h, cora/specification/stl.h, cora/specification/stl_interval.h
//
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/request.h"
#include "cora/specification/stl.h"
#include "cora/specification/stl_interval.h"

#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// The wire form of an expression, a formula and an interval.
Json stl_json(const StlExpr &e);
Json stl_json(const Stl &phi);
Json stl_json(const StlInterval &I);

/// The number of variables the oracle is told: one past the largest index, at least 1.
int64_t stl_variables(const Stl &phi);

/// `op` (stl.<name>) of the formula phi with `n` and further operands; the reply.
Reply ask_stl(const char *op, const Stl &phi, std::vector<Operand> more = {});

/// `op` (stlInterval.<name>) of the interval I with further operands; the reply.
Reply ask_stl_interval(const char *op, const StlInterval &I, std::vector<Operand> more = {});

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
