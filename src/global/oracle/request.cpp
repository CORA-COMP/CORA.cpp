// request - the one place that builds an oracle request and reads its response
//
// See also: lean/oracle.h, lean/interval.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/global/oracle/request.h"
#include "cora/global/error.h"

#include "cora/global/oracle/empty_set.h"
#include "cora/global/oracle/oracle.h"

#include <sstream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The members of a list as a JSON array, and the dtype of its first lean member.
template <class T>
Wire aux_list(const std::vector<T> &items) {
    Json list = Json::array();
    std::string dtype;
    for (const T &item : items) {
        const Wire w(item);
        if (dtype.empty()) dtype = w.dtype;
        list.push(w.json);
    }
    return Wire(std::move(list), dtype);
}

// Decoding ----------------------------------------------------------------------------------------

// Decoding by type: the dtype names what the response's values are in.
template <class T>
struct Decode;

template <>
struct Decode<double> {
    static double of(const std::string &, const Json &j) { return Tensor::decode(j.text); }
};
// An index is a JSON number, not a value of the dtype.
template <>
struct Decode<int64_t> {
    static int64_t of(const std::string &, const Json &j) { return std::stoll(j.text); }
};
template <>
struct Decode<bool> {
    static bool of(const std::string &, const Json &j) { return j.boolean; }
};
template <>
struct Decode<Tensor> {
    static Tensor of(const std::string &dt, const Json &j) { return Tensor::from_json(dt, j); }
};
template <>
struct Decode<Interval> {
    static Interval of(const std::string &dt, const Json &j) { return Interval::from_json(dt, j); }
};
// A whole space is its dimension, {n}.
template <>
struct Decode<Fullspace> {
    static Fullspace of(const std::string &dt, const Json &j) {
        return Fullspace::from_json(dt, j);
    }
};
// The sets with a from_json of their own: their wire form, read in the reply's dtype.
template <>
struct Decode<Zonotope> {
    static Zonotope of(const std::string &dt, const Json &j) { return Zonotope::from_json(dt, j); }
};
template <>
struct Decode<ConZonotope> {
    static ConZonotope of(const std::string &dt, const Json &j) {
        return ConZonotope::from_json(dt, j);
    }
};
template <>
struct Decode<MatZonotope> {
    static MatZonotope of(const std::string &dt, const Json &j) {
        return MatZonotope::from_json(dt, j);
    }
};
// A polytope is its wire form, in either representation.
template <>
struct Decode<ZonoBundle> {
    static ZonoBundle of(const std::string &dt, const Json &j) {
        return ZonoBundle::from_json(dt, j);
    }
};
template <>
struct Decode<Polytope> {
    static Polytope of(const std::string &dt, const Json &j) { return Polytope::from_json(dt, j); }
};
// An ellipsoid is {Q, q} and its error box.
template <>
struct Decode<Ellipsoid> {
    static Ellipsoid of(const std::string &dt, const Json &j) {
        return Ellipsoid::from_json(dt, j);
    }
};
// A capsule is its center, generator, radius and error box.
template <>
struct Decode<Capsule> {
    static Capsule of(const std::string &dt, const Json &j) { return Capsule::from_json(dt, j); }
};
// The empty set is its dimension, {"n": n}.
template <>
struct Decode<EmptySet> {
    static EmptySet of(const std::string &, const Json &j) {
        return EmptySet(std::stoll(j.at("n").text));
    }
};
// Indices and lists of matrices or zonotopes are JSON arrays.
template <>
struct Decode<std::vector<int64_t>> {
    static std::vector<int64_t> of(const std::string &, const Json &j) {
        std::vector<int64_t> indices;
        indices.reserve(j.items.size());
        for (const Json &i : j.items) indices.push_back(std::stoll(i.text));
        return indices;
    }
};
// Pairs of indices, such as the (transition, set) hits.
template <>
struct Decode<std::vector<std::vector<int64_t>>> {
    static std::vector<std::vector<int64_t>> of(const std::string &dt, const Json &j) {
        std::vector<std::vector<int64_t>> lists;
        lists.reserve(j.items.size());
        for (const Json &l : j.items) lists.push_back(Decode<std::vector<int64_t>>::of(dt, l));
        return lists;
    }
};
template <>
struct Decode<std::vector<Tensor>> {
    static std::vector<Tensor> of(const std::string &dt, const Json &j) {
        std::vector<Tensor> matrices;
        matrices.reserve(j.items.size());
        for (const Json &m : j.items) matrices.push_back(Tensor::from_json(dt, m));
        return matrices;
    }
};
// The zonotopes of a list.
template <>
struct Decode<std::vector<Zonotope>> {
    static std::vector<Zonotope> of(const std::string &dt, const Json &j) {
        std::vector<Zonotope> sets;
        sets.reserve(j.items.size());
        for (const Json &z : j.items) sets.push_back(Zonotope::from_json(dt, z));
        return sets;
    }
};
// JSON null is nothing.
template <class T>
struct Decode<std::optional<T>> {
    static std::optional<T> of(const std::string &dt, const Json &j) {
        if (j.kind == Json::Kind::null) return std::nullopt;
        return Decode<T>::of(dt, j);
    }
};

} // namespace


// ===========================================  MAIN  =========================================== //

// Wire ---------------------------------------------------------------------------------------

Wire::Wire(double x) : json(Tensor::encode(x)) {}
Wire::Wire(bool b) : json(b) {}
Wire::Wire(const char *s) : json(s) {}
Wire::Wire(const std::string &s) : json(s) {}
Wire::Wire(const Tensor &T) : json(T.to_json()), dtype(T.dtype()) {}
Wire::Wire(const Interval &I) : json(I.to_json()), dtype(I.inf().dtype()) {}
Wire::Wire(const Fullspace &F) : json(F.to_json()) {}
Wire::Wire(const Zonotope &Z) : json(Z.to_json()), dtype(Z.center().dtype()) {}
Wire::Wire(const ConZonotope &cz) : json(cz.to_json()), dtype(cz.dtype()) {}
Wire::Wire(const MatZonotope &A) : json(A.to_json()), dtype(A.center().dtype()) {}
Wire::Wire(const ZonoBundle &bundle)
    : json(bundle.to_json()), dtype(bundle.zonotopes().front().center().dtype()) {}
Wire::Wire(const Polytope &P) : json(P.to_json()), dtype(P.dtype()) {}
Wire::Wire(const Ellipsoid &E) : json(E.to_json()), dtype(E.dtype()) {}
Wire::Wire(const Capsule &C) : json(C.to_json()), dtype(C.dtype()) {}
Wire::Wire(const std::vector<Tensor> &Ts) : Wire(aux_list(Ts)) {}
Wire::Wire(const std::vector<Zonotope> &Zs) : Wire(aux_list(Zs)) {}
Wire::Wire(const std::vector<Polytope> &Ps) : Wire(aux_list(Ps)) {}

// Lists ---------------------------------------------------------------------------------------

Wire::Wire(const std::vector<std::vector<Tensor>> &Tss) {
    json = Json::array();
    for (const std::vector<Tensor> &Ts : Tss) {
        const Wire row(Ts);
        if (dtype.empty()) dtype = row.dtype;
        json.push(row.json);
    }
}

// Indices are a JSON array of numbers.
Wire::Wire(const std::vector<int64_t> &indices) : json(Json::array()) {
    for (const int64_t i : indices) json.push(Json(i));
}
Wire::Wire(const std::vector<std::vector<int64_t>> &index_lists) : json(Json::array()) {
    for (const std::vector<int64_t> &indices : index_lists) json.push(Wire(indices).json);
}

Wire number(double x) {
    Json j;
    j.kind = Json::Kind::number;
    std::ostringstream text;
    text.precision(17);
    text << x;
    j.text = text.str();
    return Wire(j, "");
}

// Requests -----------------------------------------------------------------------------------

template <class T>
T Reply::get(const char *key) const {
    return Decode<T>::of(dtype_, json_.at(key));
}

template double Reply::get<double>(const char *) const;
template bool Reply::get<bool>(const char *) const;
template int64_t Reply::get<int64_t>(const char *) const;
template Tensor Reply::get<Tensor>(const char *) const;
template Interval Reply::get<Interval>(const char *) const;
template Fullspace Reply::get<Fullspace>(const char *) const;
template Zonotope Reply::get<Zonotope>(const char *) const;
template ConZonotope Reply::get<ConZonotope>(const char *) const;
template MatZonotope Reply::get<MatZonotope>(const char *) const;
// the sets of the protocol, and the lists and optionals of them
template ZonoBundle Reply::get<ZonoBundle>(const char *) const;
template Polytope Reply::get<Polytope>(const char *) const;
template Ellipsoid Reply::get<Ellipsoid>(const char *) const;
template Capsule Reply::get<Capsule>(const char *) const;
template EmptySet Reply::get<EmptySet>(const char *) const;
// the lists
template std::vector<int64_t> Reply::get<std::vector<int64_t>>(const char *) const;
template std::vector<std::vector<int64_t>>
Reply::get<std::vector<std::vector<int64_t>>>(const char *) const;
template std::vector<Tensor> Reply::get<std::vector<Tensor>>(const char *) const;
template std::vector<Zonotope> Reply::get<std::vector<Zonotope>>(const char *) const;
template std::optional<double> Reply::get<std::optional<double>>(const char *) const;
template std::optional<Tensor> Reply::get<std::optional<Tensor>>(const char *) const;
template std::optional<Interval> Reply::get<std::optional<Interval>>(const char *) const;
template std::optional<Zonotope> Reply::get<std::optional<Zonotope>>(const char *) const;

// Replies ----------------------------------------------------------------------------------------

// The elements of an array member, as replies.
std::vector<Reply> Reply::list(const char *key) const {
    std::vector<Reply> replies;
    for (const Json &item : json_.at(key).items) replies.emplace_back(dtype_, item);
    return replies;
}

Reply Reply::member(const char *key) const { return {dtype_, json_.at(key)}; }

// A contains proves with true, an isIntersecting (disjoint) proves with false.
Certified Reply::proved_contains() const {
    const bool proved = get<bool>("contains");
    return {proved, proved};
}

Certified Reply::proved_disjoint() const {
    const bool proved = get<bool>("disjoint");
    return {!proved, proved};
}

// The dtype is that of the lean operands, which must agree; else the current one.
Reply ask(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, std::vector<Operand>(operands));
}

Reply ask(const char *op, const std::vector<Operand> &operands) {
    std::string dtype;
    for (const Operand &o : operands) {
        if (o.wire.dtype.empty()) continue;
        if (!dtype.empty() && dtype != o.wire.dtype)
            CORA_FAIL(Error::wrong_value, "lean: an operation on the dtypes '" + dtype + "' and '" +
                                              o.wire.dtype + "' is not allowed; round one with " +
                                              "round_to first");
        dtype = o.wire.dtype;
    }
    if (dtype.empty()) dtype = lean::dtype();
    Json request = Json::object();
    request.set("op", op).set("dtype", dtype);
    for (const Operand &o : operands) request.set(o.key, o.wire.json);
    return {dtype, call(request)};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
