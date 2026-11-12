// neur_net_contr_sys - a plant controlled by a neural network that is sampled every dt
//
// The plant x' = f(x, u) takes the first m inputs from the network, u_nn = N(x(k dt)), held
// constant over a control period, and the rest from an external input set. The closed loop is the
// plant extended by the held outputs as states with zero derivative, x_ext = (x, u_nn), whose
// inputs are the external ones (CORA's aux_computeProperties); it is built from the plant's own
// expressions, no new dynamics. reach is CORA's loop: per control period the network is enclosed
// over the current set together with the input (evaluate_with_input), the extended system is
// reached from that set with its own init_reach and post, and the set at the end of the period
// projected to x starts the next, on zonotopes or polynomial zonotopes. simulate holds the output
// over the period; verify simulates, then reaches, and splits the initial set recursively when the
// reach does not verify (CORA's verify with splitR0, the pieces on a pool of threads).
//
// Syntax:     NeurNetContrSys sys(plant, nn, dt);   ClosedLoopReach<PolyZonotope> R =
//             sys.reach(PolyZonotope(X0), std::nullopt, 5.0, {.time_step = 0.1}, {}, spec);
// Operations: reach, simulate, simulate_random, verify (one file each); the constructor
//             (neur_net_contr_sys.cpp)
// See also:   nonlinear_sys.h, neural_network.h
//
// CORA: contDynamics/@neurNetContrSys/neurNetContrSys.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "cora/specification/specification.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The problem and the results ---------------------------------------------------------------------

/// A specification that holds on a time window (CORA's specification(S, type, time)); without a
/// time it holds on the whole horizon, with t0 = t1 at one instant.
struct TimedSpecification {
    Specification spec;
    std::optional<std::pair<double, double>> time = std::nullopt;
};

/// The specifications of a problem: all of them must hold (CORA's add(spec1, spec2)).
using TimedSpecifications = std::vector<TimedSpecification>;

/// The reachable sets of the closed loop in the states of the plant, one entry per step of the
/// plant's reach (CORA's reachSet): time_interval[k] over [time[k], time[k+1]], time_point[k] at
/// time[k]. A run stops when a set violates the specification or the sets explode.
template <class S>
struct ClosedLoopReach {
    std::vector<S> time_interval, time_point;
    std::vector<double> time;
    bool completed = true; ///< the horizon was reached
    bool violated = false; ///< a set over a step violated the specification
    bool exploded = false; ///< the sets of a step grew without bound (CORA's setexplosion)
};

/// The options of reach: the algorithm of the plant (CORA's alg "lin") and those of the network.
struct ClosedLoopOptions {
    alg::Linearization plant;
    NnOptions nn;
};

/// The trajectories of simulate: x[k] holds the states (n, N) at time[k].
struct ClosedLoopTrajectories {
    std::vector<Tensor> x;
    std::vector<double> time;
};

/// The verdict of verify (CORA's res): a violating simulation that the reach from its start
/// confirms is FALSIFIED.
enum class Verdict { verified, falsified, unknown };

struct ClosedLoopVerdict {
    Verdict verdict = Verdict::unknown;
    int pieces_reached = 0;  ///< the pieces of the split initial set that were reached
    int pieces_verified = 0; ///< the leaves of the split that verified
    int max_splits = 0;      ///< the most splits along the way to a leaf, or to the failed piece
    std::optional<Tensor> violating_start; ///< the start (n, 1) of the simulation that violates
    double violation_time = 0;             ///< when it first violates the specification
    std::optional<Tensor> violating_state; ///< its state (n, 1) then
    int simulations = 0;                   ///< the number of simulated runs
    double time_reached = 0;               ///< the time up to which the reach was computed
    bool exploded = false;                 ///< the reach stopped because its sets exploded
    bool violated = false;                 ///< the reach stopped at a step that violates
};

/// The recursive splitting of verify (CORA's splitR0): the box of the initial set is cut where
/// radius times network sensitivity is largest, every piece reached on its own, split at most
/// depth - 1 times (the whole set counts as one level); a piece that still fails gives UNKNOWN.
struct VerifySplit {
    /// Where a piece is cut: CORA's rule (largest radius times network sensitivity) or the
    /// dimension whose radius is largest relative to that of the initial set (CORA.cpp only: it
    /// shrinks every dimension, which a constraint on the velocities of Docking needs).
    enum class Rule { sensitivity, widest };
    int depth = 0;       ///< splitR0; 0 and 1 reach the whole set only
    int max_threads = 8; ///< the pieces run on at most this many threads
    Rule rule = Rule::sensitivity;
};

// The system --------------------------------------------------------------------------------------

class NeurNetContrSys {
  public:
    /// The plant, its first nn.output_dim() inputs controlled by the network, sampled every dt.
    NeurNetContrSys(NonlinearSys plant, NeuralNetwork nn, double dt);

    int64_t dim() const { return plant_.dim(); }
    int64_t num_controls() const { return nn_.output_dim(); }
    int64_t num_external_inputs() const { return plant_.input_dim() - nn_.output_dim(); }
    double dt() const { return dt_; }
    const NonlinearSys &plant() const { return plant_; }
    const NeuralNetwork &network() const { return nn_; }
    /// The extended system (x, u_nn) with the external inputs (CORA's obj.sys).
    const NonlinearSys &closed_loop() const { return closed_loop_; }

    /// The reachable sets from X0 with the external input set U (none: no external input) up to
    /// t_final, on zonotopes or on polynomial zonotopes.
    ClosedLoopReach<Zonotope>
    reach(const Zonotope &X0, const std::optional<Zonotope> &U, double t_final,
          const ClosedLoopOptions &options,
          const std::optional<TimedSpecification> &spec = std::nullopt) const;
    ClosedLoopReach<PolyZonotope>
    reach(const PolyZonotope &X0, const std::optional<Zonotope> &U, double t_final,
          const ClosedLoopOptions &options,
          const std::optional<TimedSpecification> &spec = std::nullopt) const;

    /// The same with several specifications, each on its own window; the run stops at the first
    /// step that violates one.
    ClosedLoopReach<Zonotope> reach(const Zonotope &X0, const std::optional<Zonotope> &U,
                                    double t_final, const ClosedLoopOptions &options,
                                    const TimedSpecifications &specs) const;
    ClosedLoopReach<PolyZonotope> reach(const PolyZonotope &X0, const std::optional<Zonotope> &U,
                                        double t_final, const ClosedLoopOptions &options,
                                        const TimedSpecifications &specs) const;

    /// The runs from the points x0 (n, N) with a constant external input u (n_ext, N) or (n_ext, 1)
    /// per run, sampled every time_step (at most dt) with the network output held over a period.
    ClosedLoopTrajectories simulate(const Tensor &x0, const std::optional<Tensor> &u,
                                    double t_final, double time_step) const;

    /// simulate from `points` random points of X0, the first fraction_vertices of them corners,
    /// each with a constant random external input of U (CORA's simulateRandom, type standard).
    ClosedLoopTrajectories simulate_random(const Zonotope &X0, const std::optional<Zonotope> &U,
                                           double t_final, double time_step, int64_t points,
                                           double fraction_vertices, Rng &rng) const;

    /// CORA's verify: random simulations, a violating one is reached from its start (FALSIFIED if
    /// that reach does not verify), else the whole set is reached with the specifications, and
    /// split as `split` says if that does not verify (VERIFIED if every piece does).
    ClosedLoopVerdict verify(const PolyZonotope &X0, const std::optional<Zonotope> &U,
                             double t_final, const ClosedLoopOptions &options,
                             const TimedSpecifications &specs, int64_t points, Rng &rng,
                             const VerifySplit &split = {}) const;
    ClosedLoopVerdict verify(const Zonotope &X0, const std::optional<Zonotope> &U, double t_final,
                             const ClosedLoopOptions &options, const TimedSpecifications &specs,
                             int64_t points, Rng &rng, const VerifySplit &split = {}) const;
    ClosedLoopVerdict verify(const PolyZonotope &X0, const std::optional<Zonotope> &U,
                             double t_final, const ClosedLoopOptions &options,
                             const TimedSpecification &spec, int64_t points, Rng &rng,
                             const VerifySplit &split = {}) const;
    ClosedLoopVerdict verify(const Zonotope &X0, const std::optional<Zonotope> &U, double t_final,
                             const ClosedLoopOptions &options, const TimedSpecification &spec,
                             int64_t points, Rng &rng, const VerifySplit &split = {}) const;

    /// The pieces of the split of X0 (CORA's aux_VerifyRecursively, the first `levels` levels
    /// without reaching): the 2^levels boxes that cover X0, each cut at the dimension that
    /// split_dimension picks for it. Exposed for the coverage test.
    std::vector<Interval> split_pieces(const Interval &box, int levels) const;
    /// The dimension verify cuts a box at: the largest radius times mean(abs(sensitivity), 1) at
    /// the center, the first on ties.
    int64_t split_dimension(const Interval &box) const;

  private:
    NonlinearSys plant_;
    NeuralNetwork nn_;
    double dt_;
    NonlinearSys closed_loop_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
