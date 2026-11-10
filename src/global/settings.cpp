// settings - the process default of the input checks and the scopes that override it per thread
//
// Syntax:   set_checks_enabled(false);   ChecksScope off(false);   checks_enabled();
// See also: cora/global/settings.h
//
// CORA.cpp only: CORA switches its checks by options; here by scopes.
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/settings.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The process default: false only if the environment says CORACPP_CHECKS=0 (or off, false).
bool aux_default_from_environment() {
    const char *value = std::getenv("CORACPP_CHECKS");
    if (!value) return true;
    return !(std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0 ||
             std::strcmp(value, "OFF") == 0 || std::strcmp(value, "false") == 0);
}

std::atomic<bool> &aux_default() {
    static std::atomic<bool> enabled(aux_default_from_environment());
    return enabled;
}

/// This thread's override: -1 none, 0 off, 1 on.
thread_local signed char aux_override = -1;

} // namespace


// ===========================================  MAIN  =========================================== //

bool checks_enabled() noexcept {
    if (aux_override >= 0) return aux_override == 1;
    return aux_default().load(std::memory_order_relaxed);
}

void set_checks_enabled(bool enabled) noexcept {
    aux_default().store(enabled, std::memory_order_relaxed);
}

ChecksScope::ChecksScope(bool enabled) noexcept : previous_(aux_override) {
    aux_override = enabled ? 1 : 0;
}

ChecksScope::~ChecksScope() { aux_override = previous_; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
