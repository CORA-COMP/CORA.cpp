// test_linear_sys_verify_zonotope - linear_sys verify with the zonotope algorithm against MATLAB
//
// CORA
//
// Reference: MATLAB R2025b, CORA (feature/coracpp-python),
// models/Cora/ARCH/AFF/rand0{1,2}_test.json loaded by json2cora_linearSys, options.verifyAlg =
// 'reachavoid:zonotope'. The initial error bound (emax) and the steps, errors and interval hulls of
// the output sets of the first run of the adaptive reachability algorithm are pasted from a MATLAB
// run of verify for RAND01.
//
// CORA: contDynamics/@linearSys/verify.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_reach_adaptive.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

Tensor mat(int rows, int cols, const std::vector<double> &row_major) {
    return Tensor::from_data(row_major, {rows, cols});
}

Zonotope zono(const std::vector<double> &c, const std::vector<double> &G, int m) {
    const int n = static_cast<int>(c.size());
    return {Tensor::from_data(c, {n, 1}), Tensor::from_data(G, {n, m})};
}

struct Rand {
    LinearSys sys;
    ReachProblem params;
};

/// RAND01 (the two-dimensional system of models/Cora/ARCH/AFF/rand01_test.json).
Rand rand01() {
    return {
        LinearSys(
            mat(2, 2,
                {-4.762817785940338, 0.0548736150189329, -1.245752339867371, -3.8473970014254495}),
            mat(2, 1, {-0.3506721283791961, -0.4375638799088004}),
            mat(2, 2,
                {0.4224086254951373, 0.5545747548982426, -0.4016921186719066, 0.8764091437460997})),
        {zono({5.0, 5.0}, {0.1419799422010266, 0.0, 0.0, 0.043810202342537075}, 2),
         zono({-0.010285680851837772}, {2.212097470876556e-06}, 1), 1.7579573043734289}};
}

/// RAND02 (rand02_test.json).
Rand rand02() {
    return {LinearSys(mat(2, 2,
                          {-3.23572482820897, 0.16706659053778172, -0.31559399997253723,
                           -3.316537344319652}),
                      mat(2, 1, {0.4124488705282599, 0.5332712124992052}),
                      mat(2, 2,
                          {0.9886808902887005, 0.040840377520821224, -0.37781870322037275,
                           -0.4387508200979919})),
            {zono({5.0, 5.0}, {0.06616211418891371, 0.0, 0.0, 0.001398185674650776}, 2),
             zono({0.016688895501438653}, {6.865518650766428e-06}, 1), 2.1085100373254217}};
}

/// The four halfspaces of a box as RAND's JSON gives them: rows (1,0), (0,1), (-1,-0), (-0,-1)
/// with the bounds b of the file.
std::vector<Specification> rand_specs(const std::vector<double> &b1,
                                      const std::vector<double> &b2) {
    auto box = [](const std::vector<double> &b) {
        return Specification::unsafe_set({{test::column({1, 0}), b[0]},
                                          {test::column({0, 1}), b[1]},
                                          {test::column({-1, 0}), b[2]},
                                          {test::column({0, -1}), b[3]}});
    };
    return {box(b1), box(b2)};
}

/// The first run of the adaptive algorithm of RAND01 against the MATLAB sets.
void adaptive_reach_matches_matlab() {
    Rand r = rand01();
    TaylorLinSys taylor(r.sys.A());
    AdaptiveSaveData savedata;
    const Tensor GU = r.sys.B()->matmul(r.params.input_set->generators());
    const Tensor u_trans = r.sys.B()->matmul(r.params.input_set->center());
    const AdaptiveParams ap{*r.params.initial_set, GU, u_trans, r.sys.C(), r.params.t_final};
    const double emax = 0.13282709397856149;
    const AdaptiveResult R = priv_reach_adaptive(taylor, ap, {VerifyTime({{0, r.params.t_final}})},
                                                 emax, true, savedata);
    check(R.internal_checks_ok, "the error bookkeeping is consistent");

    // MATLAB: the 13 steps, their end times, the errors and the interval hulls of timeInterval.
    const double end_times[13] = {0.0412021243213, 0.0858377590026, 0.133906904044, 0.188843069806,
                                  0.243779235567,  0.32618348421,   0.408587732852, 0.518460064376,
                                  0.683268561661,  0.903013224707,  1.23263021928,  1.67211954537,
                                  1.75795730437};
    const double err[13] = {0.116418803485,   0.10873047866,  0.0990897786072, 0.0985450544468,
                            0.0755378774986,  0.119107255538, 0.0801394565426, 0.0924253121711,
                            0.122136740808,   0.102749781952, 0.0950750733264, 0.0463439758994,
                            0.000292096853698};
    const double inf[13][2] = {
        {3.87668939861, 1.77576501704},        {3.10609338248, 1.37475314797},
        {2.44522668478, 1.0388264173},         {1.85503707235, 0.745388316868},
        {1.41209034138, 0.535927147479},       {0.913129610811, 0.29665920028},
        {0.604062005048, 0.170265506481},      {0.332052066879, 0.0583273742066},
        {0.115295162289, -0.0237493166737},    {0.0151031640027, -0.0425066660497},
        {-0.0229359564228, -0.0419955611265},  {-0.013836463833, -0.0216903397491},
        {0.000769450602434, -0.00121676992421}};
    const double sup[13][2] = {
        // the supremum of the expected set
        {4.97253744676, 2.48190073449},        {4.05691829545, 1.9944844212},
        {3.2526034388, 1.57112067213},         {2.56287256743, 1.21417721633},
        {1.94906605725, 0.898694532973},       {1.4859286406, 0.674454157148},
        {0.982147671056, 0.423831204665},      {0.651896278511, 0.27190797808},
        {0.384977114785, 0.159665583526},      {0.17630849466, 0.0724059562383},
        {0.072513576713, 0.0386041593814},     {0.021431073673, 0.0176421651728},
        {0.00136393850468, -0.000244395548046}};
    check(R.time_interval.size() == 13,
          "the number of steps is " + std::to_string(R.time_interval.size()));
    if (R.time_interval.size() != 13) return;
    for (int k = 0; k < 13; ++k) {
        const std::string w = "step " + std::to_string(k + 1) + ": ";
        check(test::close(R.time[k + 1], end_times[k], 1e-9), w + "end time");
        check(test::close(R.time_interval_error[k], err[k], 1e-9), w + "error");
        const Interval I = R.time_interval[k]->interval();
        check(test::close(I.inf(), std::vector<double>{inf[k][0], inf[k][1]}, 1e-8),
              w + "hull lower");
        // the hull and the supremum of the reference
        check(test::close(I.sup(), std::vector<double>{sup[k][0], sup[k][1]}, 1e-8),
              w + "hull upper");
    }
}

/// The first run of RAND02 (emax from MATLAB): the steps up to the first unsafe one, MATLAB's
/// reachSet holds 66 of them; steps 1, 2, 10, 20, 33, 50, 65 and 66 are compared.
void adaptive_reach_rand02_matches_matlab() {
    Rand r = rand02();
    TaylorLinSys taylor(r.sys.A());
    AdaptiveSaveData savedata;
    const AdaptiveParams ap{
        *r.params.initial_set, r.sys.B()->matmul(r.params.input_set->generators()),
        r.sys.B()->matmul(r.params.input_set->center()), r.sys.C(), r.params.t_final};
    const AdaptiveResult R = priv_reach_adaptive(taylor, ap, {VerifyTime({{0, r.params.t_final}})},
                                                 0.0021200701252661563, true, savedata);
    check(R.internal_checks_ok, "RAND02: the error bookkeeping is consistent");
    const int step[8] = {1, 2, 10, 20, 33, 50, 65, 66};
    const double t_start[8] = {0,
                               0.00463295662498,
                               0.0416966096248,
                               0.0906000406663,
                               0.160609162999,
                               0.267681938332,
                               0.393286540165,
                               0.403581999332};
    // end times and errors of the reference
    const double t_end[8] = {0.00463295662498, 0.00926591324997, 0.0463295662498, 0.0957477702496,
                             0.166786438499,   0.275918305666,   0.403581999332,  0.413877458498};
    const double err[8] = {0.00199542719836, 0.00196540474507, 0.00174100142285, 0.00169191394379,
                           0.00169712922754, 0.00174754257855, 0.00157413342407, 0.00152197280866};
    check(R.time_interval.size() >= 66, "RAND02: at least 66 steps");
    if (R.time_interval.size() < 66) return;
    for (int i = 0; i < 8; ++i) {
        const int k = step[i] - 1;
        const std::string w = "RAND02 step " + std::to_string(step[i]) + ": ";
        check(test::close(R.time[k], t_start[i], 1e-9), w + "start time");
        check(test::close(R.time[k + 1], t_end[i], 1e-9), w + "end time");
        check(test::close(R.time_interval_error[k], err[i], 1e-9), w + "error");
    }
}

/// A safe set: one halfspace, or the unsafe box [lo1, hi1] x [lo2, hi2].
Specification safe_halfspace(double a1, double a2, double b) {
    return Specification::safe_set(test::column({a1, a2}), b);
}

Specification unsafe_box(double lo1, double hi1, double lo2, double hi2) {
    return Specification::unsafe_set({{test::column({1, 0}), hi1},
                                      {test::column({0, 1}), hi2},
                                      {test::column({-1, 0}), -lo1},
                                      {test::column({0, -1}), -lo2}});
}

/// Cases of the RAND01 system with MATLAB's verdict and number of refinements (the lines
/// 'Iteration i, current error' of verify with options.verbose). Specifications apply at all times
/// (the unsafe ones with time interval(0, t_final); the safe ones are halfspaces).
void refinement_cases_match_matlab() {
    struct Case {
        const char *name;
        std::vector<Specification> specs;
        bool verified;
        int iterations;
    };
    const Rand r = rand01();
    const std::vector<Case> cases = {
        {"safe y1 <= 4.98", {safe_halfspace(1, 0, 4.98)}, true, 2},
        {"safe y1 <= 4.96", {safe_halfspace(1, 0, 4.96)}, false, 3},
        {"unsafe box far away", {unsafe_box(10, 11, 0, 1)}, true, 1},
        {"unsafe box close", {unsafe_box(4.6, 5.5, 2.4695, 3)}, true, 2},
        {"unsafe box hit", {unsafe_box(4.6, 5.5, 2.40, 3)}, false, 1},
        {"safe y2 <= 2.475", {safe_halfspace(0, 1, 2.475)}, true, 2},
        {"unsafe box late", {unsafe_box(-0.05, 0.0005, -0.05, 0.0005)}, true, 1},
        {"unsafe box late, far", {unsafe_box(0.002, 0.05, 0.05, 0.1)}, true, 1},
    };
    for (const Case &c : cases) {
        const VerifyResult res = r.sys.verify(r.params, c.specs, alg::ReachAvoidZonotope{});
        // verdicts and iterations of the cases
        check(res.verified == c.verified, std::string(c.name) + ": verdict");
        check(res.iterations == c.iterations,
              std::string(c.name) + ": iterations " + std::to_string(res.iterations));
    }

    // Without an input set, and with a larger uncertain input.
    const LinearSys no_input(r.sys.A(), mat(2, 1, {0, 0}), r.sys.C());
    const ReachProblem p0{r.params.initial_set, zono({0}, {}, 0), r.params.t_final};
    VerifyResult n1 = no_input.verify(p0, {safe_halfspace(1, 0, 4.98)}, alg::ReachAvoidZonotope{});
    check(n1.verified && n1.iterations == 2, "no input, safe y1 <= 4.98");
    n1 = no_input.verify(p0, {unsafe_box(4.6, 5.5, 2.4695, 3)}, alg::ReachAvoidZonotope{});
    check(n1.verified && n1.iterations == 2, "no input, unsafe box close");
    const ReachProblem p2{r.params.initial_set, zono({0}, {0.3}, 1), r.params.t_final};
    n1 = r.sys.verify(p2, {safe_halfspace(1, 0, 5.3)}, alg::ReachAvoidZonotope{});
    check(n1.verified && n1.iterations == 2, "large input set, safe y1 <= 5.3");
    n1 = r.sys.verify(p2, {unsafe_box(4.6, 5.6, 2.6, 3)}, alg::ReachAvoidZonotope{});
    check(n1.verified && n1.iterations == 1, "large input set, unsafe box");
}

/// A three-dimensional system with three uncertain inputs and two outputs (the reduction of the
/// particular solution of U takes place), with a reference from MATLAB.
void reduction_case_matches_matlab() {
    const LinearSys sys(mat(3, 3, {-1, -4, 0, 4, -1, 0, 0, 0, -3}),
                        mat(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1}), mat(2, 3, {1, 1, 0, 0, 0, 1}));
    const ReachProblem params{zono({1, 1, 1}, {0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1}, 3),
                              zono({0.1, 0, 0.2}, {0.05, 0.02, 0, 0, 0.05, 0.01, 0.01, 0, 0.05}, 3),
                              2.0};

    // The adaptive reachable sets of the first run (emax 4.940648 as MATLAB printed it).
    TaylorLinSys taylor(sys.A());
    AdaptiveSaveData savedata;
    const AdaptiveParams ap{*params.initial_set, params.input_set->generators(),
                            params.input_set->center(), sys.C(), params.t_final};
    const AdaptiveResult R =
        priv_reach_adaptive(taylor, ap, {VerifyTime({{0, 2.0}})}, 4.940648, true, savedata);
    check(R.time_interval.size() == 5, "3-dim: five steps");
    if (R.time_interval.size() == 5) {
        const int step[4] = {0, 1, 2, 4};
        const double t_end[4] = {0.4375, 0.875, 1.375, 2.0};
        const double err[4] = {4.70253455848, 3.2106884522, 2.97737397691, 0.575899210811};
        const double inf[4][2] = {{-2.21424850989, -0.153157462114},
                                  {-2.46217013572, -0.0683082121964},
                                  {-2.20625153052, -0.0715683737546},
                                  {-0.5029888848, -0.0080018621172}};
        const double sup[4][2] = {{4.16445622063, 1.3639013812},
                                  {1.1505687946, 0.491622670767},
                                  {1.91667396042, 0.278266894323},
                                  {0.629190269945, 0.146965833889}};
        for (int i = 0; i < 4; ++i) {
            // each step of the three-dimensional reference
            const int k = step[i];
            const std::string w = "3-dim step " + std::to_string(k + 1) + ": ";
            check(test::close(R.time[k + 1], t_end[i], 1e-9), w + "end time");
            check(test::close(R.time_interval_error[k], err[i], 1e-5), w + "error");
            const Interval I = R.time_interval[k]->interval();
            check(test::close(I.inf(), std::vector<double>{inf[i][0], inf[i][1]}, 1e-5),
                  w + "lower");
            check(test::close(I.sup(), std::vector<double>{sup[i][0], sup[i][1]}, 1e-5),
                  w + "upper");
            check(R.time_interval[k]->generators().shape()[1] == 10, w + "generators");
        }
    }

    // Verdicts and refinements: a far box, a close one (y1 >= 2.35), boxes needing 2 and 14, a hit.
    VerifyResult r = sys.verify(params, {unsafe_box(5, 6, 5, 6)}, alg::ReachAvoidZonotope{});
    check(r.verified && r.iterations == 1, "3-dim: far box");
    r = sys.verify(params, {unsafe_box(2.35, 4, -1, 3)}, alg::ReachAvoidZonotope{});
    check(r.verified && r.iterations == 1, "3-dim: close box");
    r = sys.verify(params, {unsafe_box(2.25, 4, -1, 3)}, alg::ReachAvoidZonotope{});
    check(r.verified && r.iterations == 2, "3-dim: box at y1 >= 2.25");
    r = sys.verify(params, {unsafe_box(2.21, 4, -1, 3)}, alg::ReachAvoidZonotope{});
    check(r.verified && r.iterations == 14, "3-dim: box at y1 >= 2.21 takes 14 refinements");
    r = sys.verify(params, {unsafe_box(2.01, 4, -1, 3)}, alg::ReachAvoidZonotope{});
    check(!r.verified && r.iterations == 1, "3-dim: box that is hit");
}

/// The verdicts on the two benchmark instances.
void rand_instances() {
    // RAND01 is verified, RAND02 is falsified (specifications of the JSON files).
    const Rand a = rand01();
    const std::vector<Specification> sa = rand_specs(
        {-0.13277811847534904, 1.1563179372961301, 1.291179775894248, 0.081110909094638917},
        {-0.13364078500531718, -0.10831831265012903, 3.4980176475574227, 2.9104983884835862});
    const VerifyResult ra = a.sys.verify(a.params, sa, alg::ReachAvoidZonotope{});
    check(ra.verified, "RAND01 is verified");
    check(ra.iterations == 1 && ra.num_steps == 13, "RAND01: one run of 13 steps");

    const Rand b = rand02();
    const std::vector<Specification> sb = rand_specs(
        {0.9169328405934547, 1.2718590603285376, 3.4231642934567788, 0.93525050715654323},
        {1.3912151812820812, 1.1973152756325354, 0.67874974623608941, 1.0925786681947902});
    const VerifyResult rb = b.sys.verify(b.params, sb, alg::ReachAvoidZonotope{});
    check(!rb.verified, "RAND02 is falsified");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    adaptive_reach_matches_matlab();
    adaptive_reach_rand02_matches_matlab();
    rand_instances();
    refinement_cases_match_matlab();
    reduction_case_matches_matlab();
    return test::finish("linear_sys verify (zonotope)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
