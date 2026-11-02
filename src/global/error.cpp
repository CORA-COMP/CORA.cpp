// error - the identifiers of cora::Error, its one raise, and the warning hook
//
// Syntax:   detail::raise(Error::wrong_value, "...");   warn("CORA:deprecated", "...");
// See also: cora/global/error.h

#include "cora/global/error.h"

#include <iostream>
#include <mutex>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// CORA's identifier and the C++ name of every Error::Id, in the order of the enum.
struct Names {
    const char *identifier, *name;
};
constexpr Names kNames[Error::count] = {
    // the identifiers of CORA's error messages (global/functions/helper/.../CORAerror.m)
    {"CORA:wrongInputInConstructor", "wrong_input_in_constructor"},
    {"CORA:noInputInSetConstructor", "no_input_in_set_constructor"},
    {"CORA:dimensionMismatch", "dimension_mismatch"},
    {"CORA:emptySet", "empty_set"},
    {"CORA:converterIssue", "converter_issue"},
    {"CORA:configFile", "config_file"},
    {"CORA:fileNotFound", "file_not_found"},
    {"CORA:wrongValue", "wrong_value"},
    {"CORA:noExactAlg", "no_exact_alg"},
    {"CORA:noSpecificAlg", "no_specific_alg"},
    {"CORA:solverIssue", "solver_issue"},
    {"CORA:noSuitableSolver", "no_suitable_solver"},
    {"CORA:notConverged", "not_converged"},
    {"CORA:emptyProperty", "empty_property"},
    {"CORA:wrongFieldValue", "wrong_field_value"},
    {"CORA:plotProperties", "plot_properties"},
    {"CORA:evenNumberInputArgs", "even_number_input_args"},
    {"CORA:oddNumberInputArgs", "odd_number_input_args"},
    {"CORA:numInputArgsConstructor", "num_input_args_constructor"},
    {"CORA:redundantNameValuePair", "redundant_name_value_pair"},
    // further identifiers of CORAerror
    {"CORA:degenerateSet", "degenerate_set"},
    {"CORA:YALMIP", "yalmip"},
    {"CORA:outOfDomain", "out_of_domain"},
    {"CORA:specialError", "special_error"},
    {"CORA:notSupported", "not_supported"},
    {"CORA:notDefined", "not_defined"},
    {"CORA:nnLayerNotSupported", "nn_layer_not_supported"},
    {"CORA:reachSetExplosion", "reach_set_explosion"},
    {"CORA:outOfMemory", "out_of_memory"},
    {"CORA:noops", "noops"},
    {"CORA:install", "install"},
    {"CORA:reportToDev", "report_to_dev"},
};

/// The handler of warnings and the lock that guards replacing it.
std::mutex &aux_warning_mutex() {
    static std::mutex mutex;
    return mutex;
}

WarningHandler &aux_warning_handler() {
    static WarningHandler handler = [](const std::string &identifier, const std::string &message) {
        std::cerr << "Warning (" << identifier << "): " << message << "\n";
    };
    return handler;
}

} // namespace


// ===========================================  MAIN  =========================================== //

const char *Error::identifier(Id id) noexcept { return kNames[id].identifier; }

const char *Error::name(Id id) noexcept { return kNames[id].name; }

namespace detail {

void raise(Error::Id id, const std::string &message) { throw Error(id, message); }

} // namespace detail

WarningHandler set_warning_handler(WarningHandler handler) {
    std::lock_guard<std::mutex> lock(aux_warning_mutex());
    WarningHandler previous = std::move(aux_warning_handler());
    aux_warning_handler() = std::move(handler);
    return previous;
}

void warn(const std::string &identifier, const std::string &message) {
    WarningHandler handler;
    {
        std::lock_guard<std::mutex> lock(aux_warning_mutex());
        handler = aux_warning_handler();
    }
    if (handler) handler(identifier, message);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
