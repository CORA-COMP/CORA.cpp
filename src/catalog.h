// What the catalog asks for: the names it uses and one instance's parameters.
//
// Kept apart from the set representations so that both backends — Eigen and libtorch —
// read an instance the same way without sharing anything else.

#pragma once

#include <string>

namespace cora {

/// The catalog's set names.
inline constexpr const char *kRepresentations[] = {"interval", "zonotope"};

inline constexpr const char *kOperations[] = {
    "startup", "generateRandom", "randPoint", "supportFunc", "matMul", "minkSum", "contains"};

/// Seed of every instance, so a warm daemon behaves like a fresh process.
inline constexpr unsigned long long kSeed = 0;

/// The fields of an instance's `params` an operation needs, resolved once.
struct Params {
    std::string set, operation, kind, device;
    long long n = 0, m = 0, points = 0, batch = 1, repetition = 1;

    explicit Params(const std::string &params);

    bool is_interval() const { return set == "interval"; }
};

/// Whether `name` is one of the `n` names in `list`.
bool known(const char *const *list, std::size_t n, const std::string &name);

} // namespace cora
