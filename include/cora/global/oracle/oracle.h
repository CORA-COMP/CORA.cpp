// oracle - the dtype of new lean objects and the process that computes them (CORALean)
//
// Syntax:     lean::DTypeScope scope("ieee:binary32");   Json r = lean::call(request);
// dtypes:     "binary64", "ieee:<format>" (e.g. ieee:binary32), "dyadic:<p>", "fixedpoint:<f>",
//             "float" (native binary64, sound under hardware axioms), "nearest" (unsound)
// Process:    the command in CORACPP_ORACLE (run by sh -c) speaks one JSON line per request and
//             response, e.g. "cd CORALean && lake exe oracle"; it starts on the first call. The
//             the dtype nearest (no guarantees) goes to CORACPP_ORACLE_UNSOUND instead
// See also:   lean/tensor.h, lean/zonotope.h

#pragma once

#include "cora/global/oracle/json.h"
#include "cora/global/settings.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// The process default dtype of new lean objects, for threads without a DTypeScope
/// (initially "binary64"); an error on an unknown spec.
void set_default_dtype(const std::string &spec);

/// Makes new lean objects in the dtype `spec` on this thread until the scope ends.
class DTypeScope {
  public:
    explicit DTypeScope(const std::string &spec);
    ~DTypeScope();
    DTypeScope(const DTypeScope &) = delete;
    DTypeScope &operator=(const DTypeScope &) = delete;

  private:
    std::string dtype_;
    const std::string *previous_;
};

/// This thread's dtype: its innermost DTypeScope, else the process default.
std::string dtype();

/// Sends one request (a "dtype" member is not needed for ops that carry their own objects) and
/// returns the response; throws with the oracle's message if it answers {"ok":false,...}.
Json call(const Json &request);

/// Whether the running oracle serves the op (`ping` lists it), e.g. "zonotope.enlarge"; an older
/// oracle does not serve the newer ops. Asked once per oracle process.
bool serves(const std::string &op);

/// Stops the oracle process (the next call starts a new one).
void shutdown();

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
