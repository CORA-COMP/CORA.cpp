// benchmark_hybrid_reach_arch23_rendezvous - spacecraft rendezvous benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_hybrid_reach_ARCH23_rendezvous_<instance>.m with the
// automata of models/Cora/spacecraftRendezvous/rendezvous_<instance>.m, numbers copied. The state
// is (x, y, vx, vy, t); the chaser approaches (location 0), docks (1) or aborts into the passive
// mode (2). The instances differ by the abort times, the guard intersection and the specs.
//
// Syntax:   build/core/benchmarks/benchmark_hybrid_reach_arch23_rendezvous [instance ...]
// Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
// See also: benchmark_hybrid.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"
#include "benchmark_rendezvous_model.h"

#include <cmath>
#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;
using cora::bench::matrix;
namespace rendezvous = cora::bench::rendezvous;

/// ~any(supremum(interval(C x) - d) > 0): CORA's custom check by the box of C x.
Specification aux_box_spec(const Tensor &C, const Tensor &d) {
    return Specification::custom([C, d](const ContSet &S) {
        const Interval I = S.linear_map(C)->interval();
        const std::vector<double> sup = I.sup().data(), b = d.data();
        for (std::size_t i = 0; i < b.size(); ++i)
            if (sup[i] - b[i] > 0) return false;
        return true;
    });
}

/// The safe set {x | C x <= d}, a polytope.
Specification aux_safe(const Tensor &C, const std::vector<double> &d) {
    return Specification::safe_set(Polytope(C, column(d)));
}

/// CORA's aux_checkSpaceStationHit: the projection onto (x, y) must miss the box [-0.1, 0.1]^2,
/// first by its box, then reduced to order 3.
Specification aux_station() {
    return Specification::custom([](const ContSet &S) {
        const Interval station(column({-0.1, -0.1}), column({0.1, 0.1}));
        // the box of the projection decides most sets, the order-3 zonotope the rest
        const Set proj = S.project({0, 1});
        if (!is_intersecting(station, proj->interval())) return true;
        return !is_intersecting(station, *proj->reduce(3));
    });
}

/// The specifications of the instance, each with the locations it applies to.
std::vector<HybridSpecification> aux_specs(const std::string &name) {
    // the velocity in the docking mode, and the line of sight
    const Tensor Cv =
        matrix(8, 5, {0, 0, 1, 0, 0, 0, 0, -1, 0,  0, 0, 0, 0,  1, 0, 0, 0, 0,  -1, 0,
                      0, 0, 1, 1, 0, 0, 0, 1,  -1, 0, 0, 0, -1, 1, 0, 0, 0, -1, -1, 0});
    const std::vector<double> dv = {3, 3, 3, 3, 4.2, 4.2, 4.2, 4.2};
    const double pi = 4 * std::atan(1.0);
    const double s = std::tan(pi / 6);
    const Tensor Cl = matrix(2, 5, {s, -1, 0, 0, 0, s, 1, 0, 0, 0});
    const std::vector<double> dl = {0, 0};
    // the scripts of SRA07/08 and SRU01/02 check the box of C x, the others the halfspaces
    const bool custom = name == "SRA07" || name == "SRA08" || name == "SRU01" || name == "SRU02";
    std::vector<HybridSpecification> specs;
    if (custom) {
        specs.push_back({aux_box_spec(Cv, column(dv)), {1}});
        specs.push_back({aux_box_spec(Cl, column(dl)), {1}});
    } else {
        specs.push_back({aux_safe(Cv, dv), {1}});
        specs.push_back({aux_safe(Cl, dl), {1}});
    }
    if (name != "SRNA01") specs.push_back({aux_station(), {2}});
    return specs;
}

cora::bench::HybridInstance aux_instance(const std::string &name) {
    // the box of the initial positions, and the options of the scripts
    const auto [problem, a] = rendezvous::options(name);
    // the unsafe instances SRU01 and SRU02 are not verified
    const bool expected = name.rfind("SRU", 0) != 0;
    return {"Rendezvous", name,    rendezvous::automaton(name, rendezvous::variant(name)),
            problem,      a,       aux_specs(name),
            nullptr,      expected};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    const std::vector<std::string> family = {"SRA01", "SRA02", "SRA03",  "SRA04", "SRA05", "SRA06",
                                             "SRA07", "SRA08", "SRNA01", "SRU01", "SRU02"};
    return cora::bench::run_hybrid_family(argc, argv, family, aux_instance);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
