// settings - the input checks: on or off per thread, and the guard of a public entry point
//
// MATLAB CORA's CHECKS_ENABLED, in three forms: the CMake option CORACPP_CHECKS (OFF compiles
// every check out), a process default (set_checks_enabled, or CORACPP_CHECKS=0 in the
// environment), and a scope that overrides it on one thread (ChecksScope). A public operation
// starts with `detail::Entry entry;`: only the outermost one on a thread checks, so operations
// that the library calls internally never validate again (CORA's and/and_ split).
//
// Syntax:     cora::ChecksScope off(false);   cora::set_checks_enabled(false);
//             detail::Entry entry;   CORA_CHECK(cond, Error::wrong_value, [&] { return ...; });
// See also:   global/error.h, global/tensor/tensor.h (BackendScope), global/oracle/oracle.h
//
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   06-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Whether this thread checks inputs: its innermost ChecksScope, else the process default.
bool checks_enabled() noexcept;

/// The process default of the checks, for threads without a ChecksScope (initially true, or
/// false if the environment has CORACPP_CHECKS=0).
void set_checks_enabled(bool enabled) noexcept;

/// Turns the checks on or off on this thread until the scope ends.
class ChecksScope {
  public:
    explicit ChecksScope(bool enabled) noexcept;
    ~ChecksScope();
    ChecksScope(const ChecksScope &) = delete;
    ChecksScope &operator=(const ChecksScope &) = delete;

  private:
    signed char previous_;
};

namespace detail {

/// How many public entry points are active on this thread.
inline thread_local int entry_depth = 0;

/// The guard of a public entry point: counts the depth, and check() says whether this entry is
/// the outermost one with checks on. Empty with CORACPP_CHECKS=OFF.
class Entry {
  public:
    Entry() noexcept {
        if constexpr (checks_compiled) active_ = ++entry_depth == 1 && checks_enabled();
    }
    ~Entry() {
        if constexpr (checks_compiled) --entry_depth;
    }
    Entry(const Entry &) = delete;
    Entry &operator=(const Entry &) = delete;

    bool check() const noexcept { return checks_compiled && active_; }

  private:
    bool active_ = false;
};

/// The guard of a validate() of an option struct: checks whenever the checks are on, at any
/// depth, since the entry point that calls it does so only when it checks itself.
class Validation {
  public:
    bool check() const noexcept { return checks_compiled && checks_enabled(); }
};

} // namespace detail
} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
