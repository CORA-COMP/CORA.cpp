// oracle - the dtype of new lean objects and the process that computes them (CORALean)
//
// Syntax:     lean::setDType("binary64");   Json r = lean::call(request);
// dtypes:     "binary64", "ieee:<format>" (e.g. ieee:binary32), "dyadic:<p>", "fixedpoint:<f>",
//             "float" (native binary64, sound), "nearest" (native round-to-nearest, unsound)
// Process:    the command in CORACPP_ORACLE (run by sh -c) speaks one JSON line per request and
//             response, e.g. "cd CORALean && lake exe oracle"; it starts on the first call
// See also:   lean/tensor.h, lean/zonotope.h

#pragma once

#include "lean/json.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// Sets the dtype of the lean objects created from now on; throws on an unknown spec.
void setDType(const std::string &spec);

/// The current dtype spec.
std::string dtype();

/// Sends one request (a "dtype" member is not needed for ops that carry their own objects) and
/// returns the response; throws with the oracle's message if it answers {"ok":false,...}.
Json call(const Json &request);

/// Stops the oracle process (the next call starts a new one).
void shutdown();

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
