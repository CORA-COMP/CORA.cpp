// cont_set - the abstract set, as CORA's contSet
//
// Every set class derives from it; code that needs only what it offers (a specification, a
// plot, a simulation) takes any set. An operation has a public, non-virtual entry point here
// that checks its input once (detail::Entry, CORA_CHECK) and calls the class's protected
// `<operation>_impl`, which checks nothing: CORA's supportFunc and supportFunc_. Sets are
// immutable values: an operation returns a new set. They run on every Tensor backend and
// batch wherever the backend does; a vector is a column (..., n, 1).
//
// Syntax:     Tensor v = S.support_function(d);   Interval I = S.interval();   Set P =
// S.project(d); Operations: support_function, interval, rand_point, project, vertices, display (one
// file each
//             in src/cont_set/cont_set/, the checks; the classes implement the _impl forms)
// See also:   cont_set/set.h, cont_set/registry.h, cont_set/zonotope/zonotope.h
//
// CORA: contSet/@contSet/contSet.m

#pragma once

#include "cora/global/settings.h"
#include "cora/global/tensor/tensor.h"

#include <array>
#include <ostream>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Interval;
class Rng;
class Set;

/// A point of the plane, and a polygon as its vertices in counter-clockwise order.
using Point = std::array<double, 2>;
using Polygon = std::vector<Point>;

// Classes -----------------------------------------------------------------------------------------

/// The set classes of CORA, in CORA's order of precedence (the lowest first); the binary
/// registry is indexed by them. Not every class is implemented yet.
enum class ClassId : unsigned char {
    empty_set,
    fullspace,
    polygon,
    level_set,
    con_poly_zono,
    spectra_shadow,
    ellipsoid,
    capsule,
    poly_zonotope,
    polytope,
    con_zonotope,
    zono_bundle,
    zonotope,
    interval,
    custom, ///< a class defined outside the library (a Python subclass of ContSet)
    count,
};

/// The name of the class (C++ and Python, e.g. "Zonotope").
constexpr const char *class_name(ClassId id) noexcept {
    constexpr const char *names[] = {"EmptySet",     "Fullspace",     "Polygon",     "LevelSet",
                                     "ConPolyZono",  "SpectraShadow", "Ellipsoid",   "Capsule",
                                     "PolyZonotope", "Polytope",      "ConZonotope", "ZonoBundle",
                                     "Zonotope",     "Interval",      "ContSet"};
    return names[static_cast<int>(id)];
}

/// CORA's precedence of the class: of two operands, the lower one implements the operation.
constexpr int precedence(ClassId id) noexcept {
    constexpr int values[] = {0, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 1000};
    return values[static_cast<int>(id)];
}

// The Abstract Set --------------------------------------------------------------------------------

class ContSet {
  public:
    virtual ~ContSet() = default;

    ClassId class_id() const noexcept { return id_; }

    /// The name of the class: "Zonotope", ..., or the name a custom class gives itself.
    std::string class_name() const {
        return id_ == ClassId::custom ? custom_class_name() : cora::class_name(id_);
    }

    /// The dimension n.
    int64_t dim() const { return dim_impl(); }

    /// The center, a column (..., n, 1).
    Tensor center() const { return center_impl(); }

    /// max_{x in S} d'x for a direction d (..., n, 1); returns (..., 1, 1).
    Tensor support_function(const Tensor &d) const;

    /// The smallest axis-aligned box containing the set.
    Interval interval() const;

    /// N random points of the set as columns (..., n, N), drawn on the host with rng.
    Tensor rand_point(int64_t N, Rng &rng) const;

    /// The projection onto the dimensions `dims` (0-based), in that order.
    Set project(const std::vector<int64_t> &dims) const;

    /// The vertices of a two-dimensional set, one polygon per batch member.
    std::vector<Polygon> vertices() const;

    /// A description of the set as text, as CORA displays it: what `std::cout << S` prints.
    std::string display() const;

  protected:
    explicit ContSet(ClassId id) noexcept : id_(id) {}
    ContSet(const ContSet &) = default;
    ContSet(ContSet &&) noexcept = default;
    ContSet &operator=(const ContSet &) = default;
    ContSet &operator=(ContSet &&) noexcept = default;

    /// The name of a class defined outside the library (ClassId::custom).
    virtual std::string custom_class_name() const { return "ContSet"; }

    // The unchecked implementations (CORA's supportFunc_, ...): the inputs are valid.
    virtual int64_t dim_impl() const = 0;
    virtual Tensor center_impl() const = 0;
    virtual Tensor support_function_impl(const Tensor &d) const = 0;
    virtual Interval interval_impl() const = 0;
    virtual Tensor rand_point_impl(int64_t N, Rng &rng) const = 0;
    virtual Set project_impl(const std::vector<int64_t> &dims) const = 0;
    virtual std::vector<Polygon> vertices_impl() const = 0;
    virtual std::string display_impl() const = 0;

  private:
    ClassId id_;
};

inline std::ostream &operator<<(std::ostream &out, const ContSet &S) { return out << S.display(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
