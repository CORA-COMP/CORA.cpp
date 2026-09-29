#include "json.h"

#include <cctype>
#include <cstdlib>

namespace cora {
namespace {

/// Where the value of `key` starts, or `npos`.
std::size_t value_at(const std::string &doc, const std::string &key) {
    const std::string quoted = "\"" + key + "\"";
    std::size_t at = 0;
    while ((at = doc.find(quoted, at)) != std::string::npos) {
        std::size_t i = at + quoted.size();
        while (i < doc.size() && std::isspace(static_cast<unsigned char>(doc[i]))) ++i;
        if (i < doc.size() && doc[i] == ':') {
            ++i;
            while (i < doc.size() && std::isspace(static_cast<unsigned char>(doc[i]))) ++i;
            return i;
        }
        at = i; // the same text as a value somewhere, not a key
    }
    return std::string::npos;
}

} // namespace

std::optional<std::string> json_string(const std::string &doc, const std::string &key) {
    const std::size_t at = value_at(doc, key);
    if (at == std::string::npos || doc[at] != '"') return std::nullopt;
    std::string out;
    for (std::size_t i = at + 1; i < doc.size(); ++i) {
        if (doc[i] == '\\' && i + 1 < doc.size()) {
            out += doc[++i];
        } else if (doc[i] == '"') {
            return out;
        } else {
            out += doc[i];
        }
    }
    return std::nullopt;
}

std::optional<double> json_number(const std::string &doc, const std::string &key) {
    const std::size_t at = value_at(doc, key);
    if (at == std::string::npos) return std::nullopt;
    char *end = nullptr;
    const double value = std::strtod(doc.c_str() + at, &end);
    if (end == doc.c_str() + at) return std::nullopt;
    return value;
}

} // namespace cora
