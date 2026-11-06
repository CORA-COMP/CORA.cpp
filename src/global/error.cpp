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
#define CORA_ERROR(name, identifier, meaning) {identifier, #name},
#include "cora/global/errors.def"
#undef CORA_ERROR
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
