// unregistered_set - a set of a class that no binary operation knows, for the tests of the registry
//
// Syntax:   const UnregisteredSet U;   check(raises([&] { intersect(Z, U); }, Error::noops), ...);
// Purpose:  a pair with it has no entry in the table, whatever classes the library implements
//           later; tests of "no implementation" and of run-time registration use it
// See also: cora/cont_set/registry.h
//
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// A two-dimensional set of the class "UnregisteredSet" (ClassId::custom) that is the origin.
class UnregisteredSet : public cora::ContSet {
  public:
    UnregisteredSet() : ContSet(cora::ClassId::custom) {}

  protected:
    std::string custom_class_name() const override { return "UnregisteredSet"; }
    int64_t dim_impl() const override { return 2; }
    cora::Tensor center_impl() const override { return cora::Tensor({0.0, 0.0}); }
    cora::Tensor support_function_impl(const cora::Tensor &d) const override {
        return d.zeros_like({1, 1});
    }
    cora::Interval interval_impl() const override {
        return {cora::Tensor({0.0, 0.0}), cora::Tensor({0.0, 0.0})};
    }
    cora::Tensor rand_point_impl(int64_t, cora::Rng &) const override {
        return cora::Tensor({0.0, 0.0});
    }
    cora::Set project_impl(const std::vector<int64_t> &) const override { return *this; }
    std::vector<cora::Polygon> vertices_impl() const override { return {{{0.0, 0.0}}}; }
    std::vector<std::vector<cora::Point3>> vertices_3d_impl() const override { return {}; }
    std::string display_impl() const override { return "UnregisteredSet"; }
};

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
