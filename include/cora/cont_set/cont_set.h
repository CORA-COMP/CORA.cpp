// cont_set - the abstract set, as CORA's contSet
//
// Every set class derives from it; code that needs only what it offers (a specification, a
// plot, a simulation) takes any set. An operation has a public, non-virtual entry point here
// that checks its input once (detail::Entry, CORA_CHECK) and calls the class's protected
// `<operation>_impl`, which checks nothing: CORA's supportFunc and supportFunc_. Sets are
// immutable values: an operation returns a new set. They run on every Tensor backend and
// batch wherever the backend does; a vector is a column (..., n, 1).
//
// Syntax:     Tensor v = S.support_function(d);   Interval I = S.interval();
//             Set P = S.project(d);
// Operations: support_function, interval, rand_point, project, vertices, vertices_3d, display,
//             center, is_bounded, is_full_dim, reduce, linear_map, enlarge, minus,
//             minkowski_difference, decompose, project_high_dim, project_on_hyperplane,
//             uminus, uplus, times, compact, generate_random, lift, norm, enclose (one file each in
//             src/cont_set/cont_set/, the checks; the classes implement the _impl forms, and a
//             class that does not gets CORA's noops)
// See also:   cont_set/set.h, cont_set/registry.h, cont_set/zonotope/zonotope.h
//
// CORA: contSet/@contSet/contSet.m
// CORA: contSet/@contSet/dim.m
// CORA: contSet/@contSet/lift.m
// CORA: contSet/@contSet/lift_.m
// CORA: contSet/@contSet/norm.m
// CORA: contSet/@contSet/norm_.m
// CORA: contSet/@contSet/enclose.m

#pragma once

#include "cora/global/settings.h"
#include "cora/global/tensor/tensor.h"
#include "cora/global/tensor/tolerance.h"

#include <array>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Interval;
class Rng;
class Set;

/// A point of the plane, and a polygon as its vertices in counter-clockwise order.
using Point = std::array<double, 2>;
using Polygon = std::vector<Point>;

/// A point of space; the vertices of a three-dimensional set are listed as such.
using Point3 = std::array<double, 3>;

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
    Tensor center() const;

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

    /// The vertices of a three-dimensional set, one list per batch member, in no particular order.
    std::vector<std::vector<Point3>> vertices_3d() const;

    /// A description of the set as text, as CORA displays it: what `std::cout << S` prints.
    std::string display() const;

    // Queries and maps ----------------------------------------------------------------------------

    /// Whether the set is bounded; for a batch, whether every member is.
    bool is_bounded() const;

    /// Whether the set has a nonempty interior (is not in a lower-dimensional subspace). tol
    /// relaxes the rank or width test; default: the repository tolerance (docs/architecture.md).
    bool is_full_dim(Tol tol = std::nullopt) const;

    /// A set of at most `order` times the dimension generators that encloses the set (Girard's
    /// method, CORA's default); order >= 1.
    Set reduce(int order = 1) const;

    /// The image under the matrix M (..., m, n) (CORA's mtimes(M, S)), in the class of the set.
    Set linear_map(const Tensor &M) const;

    /// The set scaled about its center by factor, a number or a column (..., n, 1): the center
    /// is subtracted, the set scaled coordinate by coordinate and the center added back.
    Set enlarge(const Tensor &factor) const;
    Set enlarge(double factor) const;

    /// The set shifted by -p for a column p (..., n, 1), and for the static form -S + p (CORA's
    /// minus with a vector); the Minkowski difference is minkowski_difference.
    Set minus(const Tensor &p) const;
    static Set minus(const Tensor &p, const ContSet &S);

    /// -S, in the class of S (CORA's uminus: -1 * S).
    Set uminus() const;

    /// S itself (CORA's uplus).
    Set uplus() const;

    /// The set scaled coordinate by coordinate, diag(factor) * S (CORA's times, S .* factor):
    /// factor is a number or a column (..., n, 1).
    Set times(const Tensor &factor) const;
    Set times(double factor) const;

    /// The set without redundancy (CORA's compact): method names a variant of the class (default:
    /// the class's own; a zonotope: "zeros", "all", "aligned"), tol relaxes "is zero" and "is
    /// parallel" (default: the repository tolerance). A box, the empty set and the whole space
    /// are returned as they are; Error::noops for a class without it.
    Set compact(const std::optional<std::string> &method = std::nullopt,
                Tol tol = std::nullopt) const;

    /// The Minkowski difference {x | x + S2 in this set} (CORA's minkDiff); type picks the
    /// algorithm of the class (default: the class's own), Error::noops for a pair without one.
    Set minkowski_difference(const ContSet &S2,
                             const std::optional<std::string> &type = std::nullopt) const;

    /// The projections onto the blocks of coordinates [first, last] (0-based, both inclusive),
    /// one set per block, as CORA's decompose.
    std::vector<Set> decompose(const std::vector<std::array<int64_t, 2>> &blocks) const;

    /// The set in N dimensions with coordinate proj[j] (0-based) the j-th coordinate of this set
    /// and 0 elsewhere (CORA's projectHighDim_: I(:, proj) * S).
    Set project_high_dim(int64_t N, const std::vector<int64_t> &proj) const;

    /// The orthogonal projection onto the hyperplane {x | c'x = d} for a normal c (..., n, 1) of
    /// any length and an offset d (..., 1, 1): (I - c c') S + d c with c normalized.
    Set project_on_hyperplane(const Tensor &c, const Tensor &d) const;

    /// The set in N >= n dimensions with its coordinate j at position proj[j] (0-based, default
    /// 0..n-1): CORA's lift. A class defines what the other coordinates are: unbounded for a
    /// box, absent for a zonotope, which therefore needs N = n and a permutation.
    Set lift(int64_t N, const std::optional<std::vector<int64_t>> &proj = std::nullopt) const;

    /// An upper bound (..., 1, 1) of the norm of the points (CORA's norm): type one, two, inf or
    /// fro; mode "ub" (default), "exact" or "ub_convex", of which a class offers what its norm_
    /// has. The empty set has norm -Inf.
    Tensor norm(Tensor::Norm type = Tensor::Norm::two, const std::string &mode = "ub") const;

    /// A set that encloses this one and S2 (CORA's enclose): the class of this set, which takes
    /// the operand it can (a box and a box, a zonotope and a zonotope); Error::noops otherwise.
    Set enclose(const ContSet &S2) const;

    /// A random set of one of the admissible classes (CORA names: "interval", "zonotope"), of
    /// dimension n (default: uniform in 1..10), drawn with rng.
    static Set generate_random(Rng &rng, std::optional<int64_t> dimension = std::nullopt,
                               const std::vector<std::string> &admissible = {"interval",
                                                                             "zonotope"});

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
    virtual std::vector<std::vector<Point3>> vertices_3d_impl() const = 0;
    virtual std::string display_impl() const = 0;

    // The unchecked forms that a class may implement; CORA's contSet raises noops for the rest.
    virtual bool is_bounded_impl() const;
    virtual bool is_full_dim_impl(Tol tol) const;
    virtual Set reduce_impl(int order) const;
    virtual Set linear_map_impl(const Tensor &M) const;
    /// The set translated by v (..., n, 1): the plus of a set and a vector.
    virtual Set translate_impl(const Tensor &v) const;
    /// CORA's uminus: -1 * S, which a class with its own negation overrides.
    virtual Set uminus_impl() const;
    /// CORA's uplus: S translated by -0, which leaves every number as it is.
    virtual Set uplus_impl() const;
    /// CORA's times: diag(factor) * S for factor (..., n, 1), which a class with its own
    /// product overrides.
    virtual Set times_impl(const Tensor &factor) const;
    /// The methods of the class's compact, which the entry checks together with tol; none: the
    /// class returns itself whatever the arguments (CORA returns before reading them).
    virtual std::optional<std::vector<std::string>> compact_methods_impl() const;
    /// CORA's compact_ of the class, with the method and tolerance as given by the caller.
    virtual Set compact_impl(const std::optional<std::string> &method, Tol tol) const;
    /// CORA's enlarge: the generic formula, which a class with a cheaper one overrides.
    virtual Set enlarge_impl(const Tensor &factor) const;
    virtual Set minkowski_difference_impl(const ContSet &S2,
                                          const std::optional<std::string> &type) const;
    /// CORA's projectHighDim_: I(:, proj) * S, which a class with a cheaper one overrides.
    virtual Set project_high_dim_impl(int64_t N, const std::vector<int64_t> &proj) const;
    /// CORA's lift_ (proj has one valid position per coordinate); noops for a class without it.
    virtual Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const;
    /// CORA's norm_ (type and mode are valid); noops for a class without it.
    virtual Tensor norm_impl(Tensor::Norm type, const std::string &mode) const;
    /// CORA's enclose of two sets; noops for a class that cannot take the class of S2.
    virtual Set enclose_impl(const ContSet &S2) const;

  private:
    ClassId id_;
};

inline std::ostream &operator<<(std::ostream &out, const ContSet &S) { return out << S.display(); }

namespace detail {

/// What an entry refuses: the identifier and the message, or nothing.
using Refusal = std::optional<std::pair<Error::Id, std::string>>;

/// The refusal of project_high_dim(N, proj) for a set of dimension n; the entries of every set
/// class that has the operation raise it with CORA_CHECK. `operation` names it in the message
/// (lift has the same checks).
Refusal project_high_dim_refusal(int64_t n, int64_t N, const std::vector<int64_t> &proj,
                                 const char *operation = "project_high_dim");

// Details -----------------------------------------------------------------------------------------

/// The lift of a set that cannot grow (a zonotope has no unbounded new dimensions) is a
/// permutation: the state of position j of the lifted set is state source[j]. The plan holds what
/// refuses the call, if anything (project_high_dim_refusal, positions that are not a permutation of
/// all N), or else the source.
struct LiftPlan {
    Refusal refusal;
    std::vector<int64_t> source;
};
LiftPlan lift_plan(int64_t n, int64_t N, const std::vector<int64_t> &dims);

} // namespace detail

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
