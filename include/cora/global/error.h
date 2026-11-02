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

#pragma once

#include "cora/config.h"

#include <functional>
#include <stdexcept>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// An error of the library: one of CORA's identifiers and a message.
class Error : public std::runtime_error {
  public:
    /// CORA's identifiers (CORAerror.m), in CORA's order.
    enum Id {
        wrong_input_in_constructor,  ///< a constructor got arguments it cannot build from
        no_input_in_set_constructor, ///< a set constructor got no arguments
        dimension_mismatch,          ///< operands of different dimensions or sizes
        empty_set,                   ///< the operation is not defined for an empty set
        converter_issue,             ///< a conversion between formats failed
        config_file,                 ///< a configuration file could not be written
        file_not_found,              ///< a file does not exist
        wrong_value,                 ///< an argument has a value outside its admissible range
        no_exact_alg,                ///< there is no exact algorithm for these operands
        no_specific_alg,             ///< the requested algorithm is not implemented for them
        solver_issue,                ///< a solver failed
        no_suitable_solver,          ///< no solver for the problem is available
        not_converged,               ///< an iteration did not converge
        empty_property,              ///< a property that is needed is empty
        wrong_field_value,           ///< an option has a value outside its admissible ones
        plot_properties,             ///< a plot got wrong dimensions or properties
        even_number_input_args,      ///< name-value pairs: an even number expected
        odd_number_input_args,       ///< name-value pairs: an odd number expected
        num_input_args_constructor,  ///< a constructor got the wrong number of arguments
        redundant_name_value_pair,   ///< an option name that is not one of the known ones
        degenerate_set,              ///< the operation needs a full-dimensional set
        yalmip,                      ///< (MATLAB CORA: YALMIP) kept for the numbering
        out_of_domain,               ///< a value lies outside the domain of a function
        special_error,               ///< a specific situation described by the message
        not_supported,               ///< the combination of arguments is not supported
        not_defined,                 ///< the operation is not defined for these arguments
        nn_layer_not_supported,      ///< a neural network layer is not supported
        reach_set_explosion,         ///< the reachable sets grew without bound
        out_of_memory,               ///< the computation needs more memory than there is
        noops,                       ///< no implementation for this pair of classes
        install,                     ///< a dependency is missing from the installation
        report_to_dev,               ///< an internal inconsistency: please report it
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
using WarningHandler = std::function<void(const std::string &identifier, const std::string &message)>;

/// Replaces the receiver of all warnings (default: one line on stderr); an empty one silences
/// them. Returns the previous one.
WarningHandler set_warning_handler(WarningHandler handler);

/// Hands a warning to the handler.
void warn(const std::string &identifier, const std::string &message);

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
#define CORA_CHECK(cond, id, message)                                                             \
    do {                                                                                           \
        if constexpr (::cora::detail::checks_compiled) {                                          \
            if (entry.check() && !(cond)) ::cora::detail::raise((id), (message)());               \
        }                                                                                          \
    } while (0)

/// A failure that is not a wrong input (an algorithm that cannot go on, an option switch with no
/// branch left): always raised, checks or not.
#define CORA_FAIL(id, message) ::cora::detail::raise((id), (message))

// ---------------------------------------  END OF CODE  ---------------------------------------- //
