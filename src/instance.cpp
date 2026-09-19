#include "instance.h"

#include "json.h"
#include "lp.h"
#include "sets.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cora {
namespace {

using Index = Eigen::Index;
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Keeps a result from being optimized away: the operation is what is measured, and
/// nothing downstream looks at what it produced.
void sink(double v) {
    static volatile double kept = 0.0;
    kept += v;
}

bool known(const char *const *list, std::size_t n, const std::string &name) {
    for (std::size_t i = 0; i < n; ++i)
        if (name == list[i]) return true;
    return false;
}

/// The fields of `params` an operation needs, resolved once.
struct Instance {
    std::string set, operation, kind, device;
    Index n = 0, m = 0, points = 0, batch = 1;
    long repetition = 1;

    explicit Instance(const std::string &params) {
        const auto text = [&](const char *key, const char *fallback) {
            return json_string(params, key).value_or(fallback);
        };
        const auto num = [&](const char *key, double fallback) {
            return static_cast<Index>(json_number(params, key).value_or(fallback));
        };
        set = text("set", "");
        operation = text("operation", "");
        kind = text("type", "standard");
        device = text("device", "");
        n = num("dim", 0);
        m = num("generators", static_cast<double>(2 * n));
        points = num("points", 0);
        batch = num("batch_size", 1);
        repetition = static_cast<long>(num("repetition", 1));
    }
};

/// Why this tool does not run the instance, or an empty string.
std::string unsupported_reason(const Instance &in) {
    if (!known(kRepresentations, 2, in.set)) return "unknown set '" + in.set + "'";
    if (!known(kOperations, 7, in.operation)) return "unknown operation '" + in.operation + "'";
    if (in.device == "cpu") return "";
    if (in.device == "gpu") return "this tool runs on the CPU only";
    return "unknown device '" + in.device + "'";
}

/// A set of either representation, so an operation is written once.
struct Set {
    Interval<double> box;
    Zonotope<double> zono;
    bool is_box = false;

    static Set random(const Instance &in, Rng &rng) {
        Set s;
        s.is_box = in.set == "interval";
        if (s.is_box) {
            s.box = random_interval(rng, in.n, in.batch);
        } else {
            s.zono = random_zonotope(rng, in.n, in.m, in.batch);
        }
        return s;
    }

    Mat<double> rand_point(Index points, Rng &rng) const {
        return is_box ? cora::rand_point(box, points, rng) : cora::rand_point(zono, points, rng);
    }
};

/// The repeated call, with its inputs generated and bound. `run` returns the containment
/// answers when there are any, so they can be checked once the measurement is over.
struct Prepared {
    const Instance &in;
    Rng &rng;
    Set s, other;
    Mat<double> matrix, direction, points_in;

    Prepared(const Instance &i, Rng &r) : in(i), rng(r) {
        if (in.operation == "startup" || in.operation == "generateRandom") return;
        s = Set::random(in, rng);
        if (in.operation == "minkSum") {
            other = Set::random(in, rng);
        } else if (in.operation == "randPoint") {
            if (in.kind != "standard") throw std::runtime_error("randPoint type " + in.kind);
        } else if (in.operation == "supportFunc") {
            if (in.kind != "upper") throw std::runtime_error("supportFunc type " + in.kind);
            direction = Mat<double>(in.n, in.batch);
            rng.normal(direction.data(), static_cast<std::size_t>(direction.size()), 1.0);
            for (Index b = 0; b < in.batch; ++b) direction.col(b).normalize();
        } else if (in.operation == "matMul") {
            matrix = Mat<double>(in.n, in.n);
            rng.normal(matrix.data(), static_cast<std::size_t>(matrix.size()), 1.0);
        } else if (in.operation == "contains") {
            points_in = s.rand_point(in.points, rng);
        }
    }

    Mask run() {
        const std::string &op = in.operation;
        if (op == "startup") {
            sink(origin<double>(in.n).g.sum());
        } else if (op == "generateRandom") {
            const Set made = Set::random(in, rng);
            sink(made.is_box ? made.box.lo(0, 0) : made.zono.c(0, 0));
        } else if (op == "randPoint") {
            sink(s.rand_point(in.points, rng)(0, 0));
        } else if (op == "supportFunc") {
            sink(s.is_box ? support_func(s.box, direction)(0)
                          : support_func(s.zono, direction)(0));
        } else if (op == "matMul") {
            sink(s.is_box ? mat_mul(s.box, matrix).lo(0, 0) : mat_mul(s.zono, matrix).c(0, 0));
        } else if (op == "minkSum") {
            sink(s.is_box ? mink_sum(s.box, other.box).lo(0, 0)
                          : mink_sum(s.zono, other.zono).c(0, 0));
        } else if (op == "contains") {
            return s.is_box ? contains(s.box, points_in, in.points)
                            : contains(s.zono, points_in, in.points);
        }
        return {};
    }
};

/// The header+row CSV the harness reads; the extra columns are kept per instance.
void write_result(const std::string &path, const std::string &verdict,
                  const std::vector<std::pair<std::string, std::string>> &extra) {
    std::ofstream file(path, std::ios::trunc);
    file << "result";
    for (const auto &[key, _] : extra) file << "," << key;
    file << "\n" << verdict;
    for (const auto &[_, value] : extra) file << "," << value;
    file << "\n";
}

std::string fixed(double v, int digits) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.*f", digits, v);
    return buffer;
}

} // namespace

std::string run_instance(const std::string &params, const std::string &results_file,
                         std::ostream &log) {
    const Instance in(params);
    if (const std::string reason = unsupported_reason(in); !reason.empty()) {
        log << "[coracpp] not run: " << reason << "\n";
        write_result(results_file, "unsupported", {});
        return "unsupported";
    }

    Rng rng(kSeed);
    const auto start_generate = Clock::now();
    Prepared prepared(in, rng);
    const double time_generate = seconds_since(start_generate);

    const auto start_operation = Clock::now();
    Mask answers;
    for (long i = 0; i < in.repetition; ++i) answers = prepared.run();
    const double time_operation = seconds_since(start_operation);

    // The points were drawn from their sets, so anything but true is a wrong answer.
    if (in.operation == "contains" && std::find(answers.begin(), answers.end(), 0) != answers.end())
        throw std::runtime_error("contains: a point drawn from a set was reported outside it");

    log << "[coracpp] " << in.operation << " x" << in.repetition << " on " << in.device << ": "
        << fixed(time_operation, 4) << "s (+" << fixed(time_generate, 4)
        << "s generating inputs)\n";
    write_result(results_file, "finished",
                 {{"time_generate", fixed(time_generate, 6)},
                  {"time_operation", fixed(time_operation, 6)},
                  {"dtype", "double"}});
    return "finished";
}

void warm_up() {
    for (const char *set : kRepresentations) {
        for (const char *op : kOperations) {
            for (int batch : {1, 2}) {
                for (int n : {1, 3, 10}) {
                    const std::string params =
                        std::string("{\"set\": \"") + set + "\", \"operation\": \"" + op
                        + "\", \"dim\": " + std::to_string(n)
                        + ", \"generators\": " + std::to_string(2 * n)
                        + ", \"device\": \"cpu\", \"repetition\": 1, \"points\": 2"
                        + ", \"batch_size\": " + std::to_string(batch) + ", \"type\": \""
                        + (std::string(op) == "supportFunc" ? "upper" : "standard") + "\"}";
                    const Instance in(params);
                    Rng rng(kSeed);
                    Prepared prepared(in, rng);
                    prepared.run();
                }
            }
        }
    }
    // The LP decides what the projections leave open, which small instances rarely do.
    Eigen::MatrixXd square(2, 2);
    square << 1.0, 0.0, 0.0, 1.0;
    Eigen::Vector2d in_set(0.5, 0.0), out_of_set(0.0, 2.0);
    if (!contains_lp(square, in_set, 0.0) || contains_lp(square, out_of_set, 0.0))
        throw std::runtime_error("warm-up: the containment LP answered wrongly");
}

void print_env(std::ostream &out) {
#ifdef _OPENMP
    out << "openmp: " << omp_get_max_threads() << " threads\n";
#else
    out << "openmp: not compiled in\n";
#endif
    out << "eigen " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
        << EIGEN_MINOR_VERSION << ", element type: double\n";
    out << "cuda: not supported — gpu instances report unsupported\n";
}

void write_error(const std::string &results_file) { write_result(results_file, "error", {}); }

} // namespace cora
