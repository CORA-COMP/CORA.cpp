// json - a minimal JSON value with parse and dump, the wire format of the CORALean oracle
//
// See also: lean/oracle.h

#include "lean/json.h"

#include <cctype>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void aux_dumpString(const std::string &s, std::string &out) {
    out += '"';
    for (const char ch : s) {
        if (ch == '"' || ch == '\\') out += '\\';
        if (ch == '\n') out += "\\n";
        else out += ch;
    }
    out += '"';
}

void aux_dump(const Json &j, std::string &out) {
    switch (j.kind) {
    case Json::Kind::Null: out += "null"; break;
    case Json::Kind::Bool: out += j.boolean ? "true" : "false"; break;
    case Json::Kind::Number: out += j.text; break;
    case Json::Kind::String: aux_dumpString(j.text, out); break;
    case Json::Kind::Array:
        // the items separated by commas
        out += '[';
        for (std::size_t i = 0; i < j.items.size(); ++i) {
            if (i) out += ',';
            aux_dump(j.items[i], out);
        }
        out += ']';
        break;
    case Json::Kind::Object:
        // the members as "key":value
        out += '{';
        for (std::size_t i = 0; i < j.members.size(); ++i) {
            if (i) out += ',';
            aux_dumpString(j.members[i].first, out);
            out += ':';
            aux_dump(j.members[i].second, out);
        }
        out += '}';
        break;
    }
}

struct Parser {
    const std::string &s;
    std::size_t i = 0;

    [[noreturn]] void fail(const std::string &what) const {
        throw std::runtime_error("lean: malformed JSON at offset " + std::to_string(i) +
                                 ": " + what);
    }

    void skip() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }

    // Consumes `ch` after whitespace if it comes next.
    bool eat(char ch) {
        skip();
        if (i < s.size() && s[i] == ch) { ++i; return true; }
        return false;
    }

    std::string string() {
        std::string out;
        while (i < s.size() && s[i] != '"') {
            const bool escaped = s[i] == '\\' && i + 1 < s.size();
            if (escaped) ++i;
            out += (escaped && s[i] == 'n') ? '\n' : s[i];
            ++i;
        }
        if (i >= s.size()) fail("unterminated string");
        ++i;
        return out;
    }

    // A value of any kind, found by its first character.
    Json value() {
        skip();
        if (i >= s.size()) fail("unexpected end of input");
        if (eat('"')) return Json(string());
        if (eat('[')) {
            Json a = Json::array();
            if (eat(']')) return a;
            do a.push(value()); while (eat(','));
            if (!eat(']')) fail("expected ']'");
            return a;
        }
        // members are "key": value pairs
        if (eat('{')) {
            Json o = Json::object();
            if (eat('}')) return o;
            do {
                if (!eat('"')) fail("expected a member name");
                const std::string key = string();
                if (!eat(':')) fail("expected ':'");
                o.set(key, value());
            } while (eat(','));
            if (!eat('}')) fail("expected '}'");
            return o;
        }
        // literals, then numbers (kept as written)
        for (const char *lit : {"true", "false", "null"}) {
            const std::string w = lit;
            if (s.compare(i, w.size(), w) == 0) {
                i += w.size();
                Json j;
                if (w != "null") j = Json(w == "true");
                return j;
            }
        }
        const std::size_t start = i;
        while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '-' ||
                                s[i] == '+' || s[i] == '.' || s[i] == 'e' || s[i] == 'E'))
            ++i;
        if (i == start) fail("unexpected character");
        Json n;
        n.kind = Json::Kind::Number;
        n.text = s.substr(start, i - start);
        return n;
    }
};

} // namespace


// ===========================================  MAIN  =========================================== //

// Access -------------------------------------------------------------------------------------

const Json *Json::find(const std::string &key) const {
    for (const auto &m : members)
        if (m.first == key) return &m.second;
    return nullptr;
}

const Json &Json::at(const std::string &key) const {
    const Json *j = find(key);
    if (!j) throw std::runtime_error("lean: the response has no member '" + key + "'");
    return *j;
}

// Text ---------------------------------------------------------------------------------------

std::string Json::dump() const {
    std::string out;
    aux_dump(*this, out);
    return out;
}

Json Json::parse(const std::string &s) {
    Parser p{s};
    Json j = p.value();
    p.skip();
    if (p.i != s.size()) p.fail("trailing text after the value");
    return j;
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
