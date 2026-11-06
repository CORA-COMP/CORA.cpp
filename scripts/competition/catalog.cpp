// catalog - the parameters of one catalog instance, read from its JSON
//
// Authors:       Claude Sonnet 5.5
// Written:       01-October-2026
// Last update:   ---
// Reviewed:      ---

#include "catalog.h"

#include "json.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::comp {


// ===========================================  MAIN  =========================================== //

Params::Params(const std::string &params) {
    const auto text = [&](const char *key, const char *fallback) {
        return json_string(params, key).value_or(fallback);
    };
    const auto num = [&](const char *key, double fallback) {
        return static_cast<long long>(json_number(params, key).value_or(fallback));
    };
    set = text("set", "");
    operation = text("operation", "");
    kind = text("type", "standard");
    device = text("device", "");
    n = num("dim", 0);
    m = num("generators", static_cast<double>(2 * n));
    points = num("points", 0);
    batch = num("batch_size", 1);
    repetition = num("repetition", 1);
}

bool known(const char *const *list, std::size_t n, const std::string &name) {
    for (std::size_t i = 0; i < n; ++i)
        if (name == list[i]) return true;
    // no entry of the list matches
    return false;
}

} // namespace cora::comp

// ---------------------------------------  END OF CODE  ---------------------------------------- //
