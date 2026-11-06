// test_tolerance - the tolerance policy: the order of resolution, the scopes, the dtype, within_tol
//
// CORA.cpp only: the policy of the repository
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"
#include "testing.h"

#include <limits>
#include <thread>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr DType f64 = DType::float64;
constexpr DType f32 = DType::float32;

void defaults_follow_the_dtype() {
    check(machine_eps(f64) == std::numeric_limits<double>::epsilon(), "machine eps of float64");
    check(machine_eps(f32) == double(std::numeric_limits<float>::epsilon()),
          "machine eps of float32");
    check(default_tol(f64) == 100 * machine_eps(f64), "float64: 100 machine epsilons");
    check(default_tol(f32) == 100 * machine_eps(f32), "float32: 100 machine epsilons of float32");
    check(default_tol(f32) > 1e6 * default_tol(f64), "float32 has the larger default");
    check(!current_tolerance(), "no setting at the start");
    check(resolve_tol(std::nullopt, f64) == default_tol(f64), "unset resolves to the dtype's");
}

void resolution_order() {
    check(resolve_tol(1e-3, f64) == 1e-3, "an argument wins");
    {
        ToleranceScope scope(1e-6);
        check(resolve_tol(std::nullopt, f32) == 1e-6, "a scope beats the dtype");
        check(resolve_tol(1e-3, f64) == 1e-3, "an argument beats a scope");
        check(resolve_tol(0.0, f64) == 0.0, "an argument of 0 is an argument");
        {
            ToleranceScope inner(1e-9);
            check(resolve_tol(std::nullopt, f64) == 1e-9, "the innermost scope wins");
        }
        check(resolve_tol(std::nullopt, f64) == 1e-6, "a scope ends with its block");
        set_default_tolerance(1e-4);
        check(resolve_tol(std::nullopt, f64) == 1e-6, "a scope beats the process default");
        set_default_tolerance(std::nullopt);
    }
    set_default_tolerance(1e-4);
    check(resolve_tol(std::nullopt, f64) == 1e-4, "the process default beats the dtype");
    check(resolve_tol(std::nullopt, f32) == 1e-4, "... for every dtype");
    set_default_tolerance(std::nullopt);
    check(resolve_tol(std::nullopt, f64) == default_tol(f64), "unsetting restores the dtype's");
}

void scopes_do_not_leak_into_threads() {
    ToleranceScope scope(1e-6);
    double seen = 0, seen_default = 0;
    std::thread other([&] { seen = resolve_tol(std::nullopt, f64); });
    other.join();
    check(seen == default_tol(f64), "another thread does not see the scope");
    set_default_tolerance(1e-4);
    std::thread third([&] {
        ToleranceScope mine(1e-12);
        seen = resolve_tol(std::nullopt, f64);
    });
    std::thread fourth([&] { seen_default = resolve_tol(std::nullopt, f64); });
    third.join();
    fourth.join();
    set_default_tolerance(std::nullopt);
    check(seen == 1e-12, "a thread has its own scope");
    check(seen_default == 1e-4, "a thread without a scope has the process default");
    check(resolve_tol(std::nullopt, f64) == 1e-6, "this thread keeps its scope");
}

void within_tol_of_numbers_and_tensors() {
    const double inf = std::numeric_limits<double>::infinity();
    check(within_tol(inf, inf), "+Inf and +Inf agree");
    check(within_tol(-inf, -inf), "-Inf and -Inf agree");
    check(!within_tol(inf, -inf), "+Inf and -Inf do not agree");
    check(!within_tol(-inf, inf, 1e300), "... whatever the tolerance");
    check(!within_tol(inf, 1.0, 1.0), "an infinity and a number do not agree");
    check(!within_tol(std::nan(""), 1.0), "NaN agrees with nothing");
    check(within_tol(1.0, 1.0 + 1e-9, 1e-8), "absolute");
    check(!within_tol(1.0, 1.0 + 1e-9, 1e-10), "absolute: too far");
    check(within_tol(1e10, 1e10 + 1.0, 1e-9), "relative to the smaller magnitude");
    check(within_tol(0.0, 0.0, 0.0), "equal numbers agree at tol 0");
    {
        ToleranceScope scope(1e-8);
        check(within_tol(1.0, 1.0 + 1e-9), "the scope sets the default of the scalar form");
    }
    check(!within_tol(1.0, 1.0 + 1e-9), "... and only in its block");

    for (const std::string &b : test::backends()) {
        BackendScope scope(b);
        const Tensor x = Tensor({{1.0, inf, -inf, 5.0}}), y = Tensor({{1.0 + 1e-9, inf, inf, 6.0}});
        check(test::close(within_tol(x, y, 1e-8), std::vector<double>{1, 1, 0, 0}), b + ": mask");
        check(test::close(within_tol(x, y, 1e-10), std::vector<double>{0, 1, 0, 0}), b + ": tight");
        ToleranceScope loose(1e-8);
        check(test::close(within_tol(x, y), std::vector<double>{1, 1, 0, 0}), b + ": scope");
    }
}

void operations_resolve_at_the_entry() {
    const Interval I(Tensor({{0.0}, {0.0}}), Tensor({{1.0}, {1.0}}));
    const Tensor p = Tensor({{1.0 + 1e-10}, {0.5}});
    check(!I.contains(p), "float64: 1e-10 outside is outside by default");
    check(I.contains(p, 1e-9), "an argument relaxes it");
    {
        ToleranceScope scope(1e-9);
        check(I.contains(p), "a scope relaxes it");
        check(!I.contains(p, 0.0), "an argument of 0 beats the scope");
    }
    set_default_tolerance(1e-9);
    check(I.contains(p), "the process default relaxes it");
    set_default_tolerance(std::nullopt);

    const Interval J(Tensor({{0.0}, {0.0}}).to(f32), Tensor({{1.0}, {1.0}}).to(f32));
    const Tensor q = Tensor({{1.0 + 1e-6}, {0.5}}).to(f32);
    check(J.contains(q) && !I.contains(Tensor({{1.0 + 1e-6}, {0.5}})),
          "1e-6 outside: inside for float32, outside for float64, both by default");
    check(I.isequal(I), "isequal with the default");
    const Interval K(Tensor({{0.0}, {0.0}}), Tensor({{1.0 + 1e-10}, {1.0}}));
    check(!I.isequal(K) && I.isequal(K, 1e-9), "isequal: the argument relaxes it");
    check(!I.is_intersecting(Interval(Tensor({{1.0 + 1e-10}, {0.0}}), Tensor({{2.0}, {1.0}}))),
          "is_intersecting: boxes 1e-10 apart do not meet by default");
    {
        ToleranceScope scope(1e-9);
        check(I.is_intersecting(Interval(Tensor({{1.0 + 1e-10}, {0.0}}), Tensor({{2.0}, {1.0}}))),
              "... and do within a scope");
    }
    const Zonotope Z(Tensor({{0.0}, {0.0}}), Tensor({{1.0, 0.2, 1e-12}, {0.5, 1.0, 0.0}}));
    check(Z.contains(Z) && Z.contains(Zonotope(Z.center(), Z.generators() * 1.0)),
          "zonotope: contains itself and its copy");
    check(Z.compact().generators().shape().back() == 3,
          "a generator of norm 1e-12 stays by default");
    {
        ToleranceScope scope(1e-10);
        check(Z.compact().generators().shape().back() == 2, "... and goes within a scope");
    }
    test::raises([&] { I.contains(p, -1.0); }, Error::wrong_value);
    ChecksScope off(false);
    check(I.contains(p, 1e-9), "with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    defaults_follow_the_dtype();
    resolution_order();
    scopes_do_not_leak_into_threads();
    within_tol_of_numbers_and_tensors();
    operations_resolve_at_the_entry();
    return test::finish("tolerance");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
