// benchmark_data - loads and runs an ARCH-COMP AFF instance from examples/benchmarks/data
//
// The instance files (<instance>.json and <instance>.bin) are written by
// benchmarks/matlab/exportBenchmarkData.m from the original CORA scripts; their format is
// described in benchmarks/README.md. run_instance verifies one instance and prints the line
// 'benchmark,instance,result,time' of the original scripts (details go to stderr).
//
// Syntax:     cora::bench::run_instance("ISSF01_ISS01");
// See also:   contDynamics/linear_sys/linear_sys.h, cora/specification/specification.h
//
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#pragma once

#include "benchmark_run.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/sparse_matrix.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::bench {

/// A JSON value; only what the instance files use (strings without escapes other than \" and \\).
struct Json {
    enum class Kind { null, number, string, array, object } kind = Kind::null;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    const Json &operator[](const std::string &key) const {
        const auto it = object.find(key);
        if (it == object.end()) throw std::runtime_error("benchmark json: missing key " + key);
        return it->second;
    }
    const Json &operator[](std::size_t i) const { return array.at(i); }
    bool is_null() const { return kind == Kind::null; }
    int64_t integer() const { return static_cast<int64_t>(number); }
};

/// A recursive-descent parser for the JSON of the instance files.
class JsonParser {
  public:
    explicit JsonParser(const std::string &text) : s_(text) {}

    Json parse() {
        Json value = value_();
        skip_();
        if (i_ != s_.size()) fail_("trailing characters");
        return value;
    }

  private:
    const std::string &s_;
    std::size_t i_ = 0;

    [[noreturn]] void fail_(const std::string &what) const {
        throw std::runtime_error("benchmark json: " + what + " at offset " + std::to_string(i_));
    }
    // whitespace is skipped before every token
    void skip_() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    void expect_(char c) {
        skip_();
        if (i_ >= s_.size() || s_[i_] != c) fail_(std::string("expected '") + c + "'");
        ++i_;
    }
    std::string string_() {
        expect_('"');
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\') ++i_;
            out += s_.at(i_++);
        }
        expect_('"');
        return out;
    }
    Json value_() {
        skip_();
        // one value, by its first character
        Json v;
        if (i_ >= s_.size()) fail_("unexpected end");
        const char c = s_[i_];
        // objects and arrays take their members until the closing bracket
        if (c == '{') {
            v.kind = Json::Kind::object;
            expect_('{');
            skip_();
            while (s_.at(i_) != '}') {
                const std::string key = string_();
                expect_(':');
                v.object[key] = value_();
                skip_();
                if (s_.at(i_) == ',') ++i_;
                skip_();
            }
            expect_('}');
        } else if (c == '[') {
            v.kind = Json::Kind::array;
            expect_('[');
            skip_();
            while (s_.at(i_) != ']') {
                v.array.push_back(value_());
                skip_();
                // members are separated by commas
                if (s_.at(i_) == ',') ++i_;
                skip_();
            }
            expect_(']');
        } else if (c == '"') {
            // a string value
            v.kind = Json::Kind::string;
            v.string = string_();
        } else if (s_.compare(i_, 4, "null") == 0) {
            i_ += 4;
        } else {
            // anything else is a number, read with strtod for exact doubles
            char *end = nullptr;
            v.kind = Json::Kind::number;
            v.number = std::strtod(s_.c_str() + i_, &end);
            if (end == s_.c_str() + i_) fail_("unexpected character");
            i_ = static_cast<std::size_t>(end - s_.c_str());
        }
        return v;
    }
};

/// An instance: the system, what verify is given, and what MATLAB CORA returned.
struct Instance {
    std::string benchmark, label;
    alg::LinearVerify alg;
    std::unique_ptr<LinearSys> sys;
    std::unique_ptr<ReachProblem> params;
    std::vector<Specification> specs;
    Json expected;
};

// Loading ---------------------------------------------------------------------------------------

inline std::string read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("benchmark data: cannot read " + path);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

/// The matrix of a "matrices" entry as dense row-major values.
inline std::vector<double> read_matrix(const Json &entry, const std::string &bin) {
    const int64_t rows = entry["shape"][0].integer(), cols = entry["shape"][1].integer();
    const int64_t nnz = entry["nnz"].integer(), offset = entry["offset"].integer();
    std::vector<double> dense(static_cast<std::size_t>(rows * cols), 0.0);
    const char *p = bin.data() + offset;
    if (entry["storage"].string == "dense") {
        std::copy_n(reinterpret_cast<const double *>(p), nnz, dense.begin());
        return dense;
    }
    // coordinates: int32 rows, int32 columns, float64 values (0-based)
    const auto *ri = reinterpret_cast<const int32_t *>(p);
    const auto *ci = ri + nnz;
    const auto *vals = reinterpret_cast<const double *>(p + 8 * nnz);
    for (int64_t k = 0; k < nnz; ++k) dense[ri[k] * cols + ci[k]] = vals[k];
    return dense;
}

/// The matrix of a "coo" entry as a sparse matrix (the large state matrices).
inline SparseMatrix read_sparse(const Json &entry, const std::string &bin) {
    const int64_t nnz = entry["nnz"].integer(), offset = entry["offset"].integer();
    if (entry["storage"].string != "coo")
        throw std::runtime_error("benchmark data: a sparse matrix must be stored as coo");
    const auto *ri = reinterpret_cast<const int32_t *>(bin.data() + offset);
    const auto *vals = reinterpret_cast<const double *>(bin.data() + offset + 8 * nnz);
    return {entry["shape"][0].integer(), entry["shape"][1].integer(),
            std::vector<int64_t>(ri, ri + nnz), std::vector<int64_t>(ri + nnz, ri + 2 * nnz),
            std::vector<double>(vals, vals + nnz)};
}

/// A zonotope from its center (n, 1) and generators (n, m); no generators become one zero one.
inline Zonotope read_zonotope(const Json &matrices, const std::string &bin, const std::string &c,
                              const std::string &G) {
    const int64_t n = matrices[c]["shape"][0].integer(), m = matrices[G]["shape"][1].integer();
    const Tensor center = Tensor::from_data(read_matrix(matrices[c], bin), {n, 1});
    if (m == 0) return Zonotope(center, Tensor::zeros({n, 1}));
    return Zonotope(center, Tensor::from_data(read_matrix(matrices[G], bin), {n, m}));
}

inline Tensor read_tensor(const Json &matrices, const std::string &bin, const std::string &name) {
    const Json &entry = matrices[name];
    return Tensor::from_data(read_matrix(entry, bin),
                             {entry["shape"][0].integer(), entry["shape"][1].integer()});
}

// Specifications --------------------------------------------------------------------------------

/// One specification: a polytope {y | A y <= b} that a safe set holds the reachable sets in, and
/// an unsafe set keeps them out of.
inline Specification read_spec(const Json &spec) {
    std::vector<double> rows;
    const int64_t m = static_cast<int64_t>(spec["A"].array.size());
    int64_t n = 0;
    std::vector<double> offsets;
    for (std::size_t k = 0; k < spec["A"].array.size(); ++k) {
        n = static_cast<int64_t>(spec["A"][k].array.size());
        for (const Json &x : spec["A"][k].array) rows.push_back(x.number);
        offsets.push_back(spec["b"][k].number);
    }
    const Polytope P(Tensor::from_data(rows, {m, n}), Tensor::from_data(offsets, {m, 1}));
    if (spec["type"].string == "safeSet") return Specification::safe_set(P);
    if (spec["type"].string != "unsafeSet")
        throw std::runtime_error("benchmark data: unknown specification type " +
                                 spec["type"].string);
    return Specification::unsafe_set(P);
}

/// The metadata <name>.json and its binary file from examples/benchmarks/data (or
/// $CORACPP_BENCHMARK_DATA).
inline std::pair<Json, std::string> read_instance_files(const std::string &name) {
    const std::string folder = std::getenv("CORACPP_BENCHMARK_DATA")
                                   ? std::getenv("CORACPP_BENCHMARK_DATA")
                                   : CORACPP_BENCHMARK_DATA;
    Json meta = JsonParser(read_file(folder + "/" + name + ".json")).parse();
    std::string bin = read_file(folder + "/" + meta["data"].string);
    return {std::move(meta), std::move(bin)};
}

/// The instance <name> for verify.
inline Instance load_instance(const std::string &name) {
    const auto [meta, bin] = read_instance_files(name);
    const Json &mat = meta["matrices"];

    // the algorithm name is the one of CORA's options.verifyAlg
    Instance inst;
    inst.benchmark = meta["benchmark"].string;
    inst.label = meta["label"].string;
    const std::string verify_alg = meta["verifyAlg"].string;
    if (verify_alg == "reachavoid:supportFunc")
        inst.alg = alg::ReachAvoidSupportFunction{};
    else if (verify_alg == "reachavoid:zonotope")
        inst.alg = alg::ReachAvoidZonotope{};
    // the Krylov support function algorithm of the scripts of RAND04-06 of the ARCH26 package
    else if (verify_alg == "reachavoid:krylovSuppFunc")
        inst.alg = alg::ReachAvoidKrylov{};
    else
        throw std::runtime_error("benchmark data: unknown verifyAlg " + verify_alg);
    inst.sys = std::make_unique<LinearSys>(read_tensor(mat, bin, "A"), read_tensor(mat, bin, "B"),
                                           read_tensor(mat, bin, "C"));
    inst.params = std::make_unique<ReachProblem>(ReachProblem{read_zonotope(mat, bin, "R0c", "R0G"),
                                                              read_zonotope(mat, bin, "Uc", "UG"),
                                                              meta["tFinal"].number});
    for (const Json &spec : meta["specs"].array) inst.specs.push_back(read_spec(spec));
    inst.expected = meta["expected"];
    return inst;
}

// Running ---------------------------------------------------------------------------------------

/// Verifies the instance, prints 'benchmark,instance,result,time' and, on stderr, the iterations,
/// steps and step size next to those of MATLAB. The time is computation_time, else the wall clock.
/// Returns whether the verdict is MATLAB's (true without a recorded one).
inline bool run_instance(const std::string &name) {
    const Instance inst = load_instance(name);
    // what will run, before it does
    std::cerr << name << ": " << inst.params->initial_set->dim() << " states, " << inst.specs.size()
              << " specifications, " << alg::name_of(inst.alg) << " algorithm, t_final "
              << inst.params->t_final << std::endl;
    const auto start = std::chrono::steady_clock::now();
    const VerifyResult res = inst.sys->verify(*inst.params, inst.specs, inst.alg);
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;

    print_line(inst.benchmark, inst.label, res.verified,
               res.computation_time > 0 ? res.computation_time : wall.count());
    std::cerr << "  iterations " << res.iterations << ", nrSteps " << res.num_steps << ", timeStep "
              << res.time_step;
    if (!inst.expected.is_null() && inst.expected.object.count("nrSteps"))
        std::cerr << "  (MATLAB: iterations " << inst.expected["iterations"].integer()
                  << ", nrSteps " << inst.expected["nrSteps"].integer() << ", timeStep "
                  << inst.expected["timeStep"].number << ")";
    std::cerr << std::endl;
    if (inst.expected.is_null() || !inst.expected.object.count("verified")) return true;
    return res.verified == (inst.expected["verified"].number != 0);
}

} // namespace cora::bench

// ---------------------------------------  END OF CODE  ---------------------------------------- //
