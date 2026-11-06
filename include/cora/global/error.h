// error - cora::Error with CORA's identifiers, the check primitives and the warning hook
//
// Every error the library raises is a cora::Error: its identifier says what kind of mistake it
// is (CORA's CORAerror identifiers), its message what was wrong and how to fix it. Input checks
// go through CORA_CHECK, which a public entry point runs once (see settings.h); an algorithm
// that fails on valid input uses CORA_FAIL. Warnings go through one hook, which the Python
// package turns into warnings.warn.
//
// Syntax:     CORA_CHECK(Z.dim() == n, Error::dimension_mismatch, [&] { return "..."; });
//             CORA_FAIL(Error::not_converged, "...");   warn("CORA:solver", "...");
// See also:   global/settings.h, global/within_tol.h
//
// Authors:       Tobias Ladner, Claude Sonnet 5.5
// Written:       02-November-2026
// Last update:   06-November-2026
// Reviewed:      ---

#pragma once

#include "cora/config.h"

#include <functional>
#include <stdexcept>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// Errors ------------------------------------------------------------------------------------------

/// An error of the library: one of CORA's identifiers and a message.
class Error : public std::runtime_error {
  public:
    /// CORA's identifiers (errors.def), in CORA's order.
    enum Id {
#define CORA_ERROR(name, identifier, meaning) name,
#include "cora/global/errors.def"
#undef CORA_ERROR
    };

    Error(Id id, const std::string &message) : std::runtime_error(message), id_(id) {}

    Id id() const noexcept { return id_; }

    /// CORA's identifier, e.g. "CORA:dimensionMismatch".
    const char *identifier() const noexcept { return identifier(id_); }

    /// The identifier of `id` as CORA writes it, and its C++ name ("dimension_mismatch").
    static const char *identifier(Id id) noexcept;
    static const char *name(Id id) noexcept;

    /// The number of identifiers.
    static constexpr int count = report_to_dev + 1;

  private:
    Id id_;
};

// Warnings ----------------------------------------------------------------------------------------

/// What receives a warning: CORA's identifier ("CORA:deprecated", ...) and the message.
using WarningHandler =
    std::function<void(const std::string &identifier, const std::string &message)>;

/// Replaces the receiver of all warnings (default: one line on stderr); an empty one silences
/// them. Returns the previous one.
WarningHandler set_warning_handler(WarningHandler handler);

/// Hands a warning to the handler.
void warn(const std::string &identifier, const std::string &message);

// Checks ------------------------------------------------------------------------------------------

namespace detail {

/// Throws Error(id, message); the one place that raises a cora::Error.
[[noreturn]] void raise(Error::Id id, const std::string &message);

/// Whether the checks are compiled in (CMake option CORACPP_CHECKS).
inline constexpr bool checks_compiled = CORACPP_CHECKS != 0;

} // namespace detail
} // namespace cora

/// An input check of a public entry point: throws Error(id, message()) if `cond` is false and
/// the entry point checks (detail::Entry `entry` in scope); `message` is a callable building the
/// text, called only on failure. Compiles to nothing with CORACPP_CHECKS=OFF.
#define CORA_CHECK(cond, id, message)                                                              \
    do {                                                                                           \
        if constexpr (::cora::detail::checks_compiled) {                                           \
            if (entry.check() && !(cond)) ::cora::detail::raise((id), (message)());                \
        }                                                                                          \
    } while (0)

/// A failure that is not a wrong input (an algorithm that cannot go on, an option switch with no
/// branch left): always raised, checks or not.
#define CORA_FAIL(id, message) ::cora::detail::raise((id), (message))

// ---------------------------------------  END OF CODE  ---------------------------------------- //
