// tolerance - the machine epsilon, the dtype's default and the scopes that override it
//
// Syntax:   resolve_tol(tol, dtype);   ToleranceScope loose(1e-10);   set_default_tolerance(t);
// See also: cora/global/tensor/tolerance.h

#include "cora/global/tensor/tolerance.h"

#include <atomic>
#include <cmath>
#include <limits>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The process default; NaN means unset (an atomic optional is not lock-free).
std::atomic<double> &aux_default() {
    static std::atomic<double> value(std::numeric_limits<double>::quiet_NaN());
    return value;
}

/// This thread's innermost scope.
thread_local Tol aux_scope;

} // namespace


// ===========================================  MAIN  =========================================== //

// The scope of this thread first, then the process default.
Tol current_tolerance() noexcept {
    if (aux_scope) return aux_scope;
    const double value = aux_default().load(std::memory_order_relaxed);
    if (std::isnan(value)) return std::nullopt;
    return value;
}

void set_default_tolerance(Tol tol) noexcept {
    aux_default().store(tol ? *tol : std::numeric_limits<double>::quiet_NaN(),
                        std::memory_order_relaxed);
}

// The argument, the settings, then the default of the dtype.
double resolve_tol(Tol tol, DType dtype) noexcept {
    if (tol) return *tol;
    if (const Tol current = current_tolerance()) return *current;
    return default_tol(dtype);
}

double resolve_tol(const detail::Entry &entry, Tol tol, DType dtype, const char *operation) {
    const double resolved = resolve_tol(tol, dtype);
    // NaN fails the comparison too
    if (entry.check() && !(resolved >= 0))
        CORA_FAIL(Error::wrong_value, std::string(operation) + ": tol must be nonnegative, got " +
                                          std::to_string(resolved));
    return resolved;
}

double resolve_tol(const detail::Entry &entry, Tol tol, const char *operation) {
    return resolve_tol(entry, tol, DType::float64, operation);
}

ToleranceScope::ToleranceScope(double tol) noexcept : previous_(aux_scope) { aux_scope = tol; }

ToleranceScope::~ToleranceScope() { aux_scope = previous_; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
