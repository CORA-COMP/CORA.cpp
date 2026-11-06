// json - a minimal JSON value with parse and dump, the wire format of the CORALean oracle
//
// Syntax:     Json j = Json::parse(text);   std::string text = j.dump();
// Supports:   null, booleans, strings (with \" \ \n escapes), numbers (kept as text), arrays and
//             objects (insertion order)
// See also:   lean/oracle.h
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Json {
  public:
    enum class Kind { null, boolean, number, string, array, object };

    Kind kind = Kind::null;
    bool boolean = false;
    std::string text; // the string, or the number as written
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> members;

    Json() = default;
    Json(const char *s) : kind(Kind::string), text(s) {}
    Json(std::string s) : kind(Kind::string), text(std::move(s)) {}
    Json(bool b) : kind(Kind::boolean), boolean(b) {}
    Json(int64_t n) : kind(Kind::number), text(std::to_string(n)) {}
    static Json array() {
        Json j;
        j.kind = Kind::array;
        return j;
    }
    static Json object() {
        Json j;
        j.kind = Kind::object;
        return j;
    }

    /// Appends to an array.
    Json &push(Json v) {
        items.insert(items.end(), std::move(v));
        return *this;
    }

    /// Sets a member of an object (appended; keys are not deduplicated).
    Json &set(const std::string &key, Json v) {
        members.insert(members.end(), {key, std::move(v)});
        return *this;
    }

    /// The member `key`, or nullptr.
    const Json *find(const std::string &key) const;

    /// The member `key`; throws if it is missing.
    const Json &at(const std::string &key) const;

    /// The compact text.
    std::string dump() const;

    /// Parses one value; throws on malformed input or trailing text.
    static Json parse(const std::string &s);
};

/// Whether a wire-form tensor or box has only zero entries ("0*2^0" is the text of zero).
bool zero_valued(const Json &j);

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
