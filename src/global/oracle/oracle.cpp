// oracle - the dtype of new lean objects and the process that computes them (CORALean)
//
// See also: lean/tensor.h

#include "cora/global/oracle/oracle.h"
#include "cora/global/error.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Dtypes ------------------------------------------------------------------------------------------

/// The process default dtype (guarded by its mutex) and this thread's scope (null: the default).
std::mutex default_mutex;
std::string default_dtype = "binary64";
thread_local const std::string *scoped_dtype = nullptr;

/// Whether spec names a dtype of the oracle.
bool aux_known(const std::string &spec) {
    const std::string kind = spec.substr(0, spec.find(':'));
    const bool has_arg = spec.find(':') != std::string::npos;
    return spec == "binary64" || spec == "float" || spec == "nearest" ||
           (kind == "ieee" && has_arg) || (kind == "dyadic" && has_arg) ||
           (kind == "fixedpoint" && has_arg);
}

/// The message of an unknown dtype.
std::string aux_unknown(const std::string &spec) {
    return "lean: unknown dtype '" + spec +
           "'; use 'binary64', 'ieee:<format>', 'dyadic:<p>', "
           "'fixedpoint:<f>', 'float' or 'nearest'";
}

struct Process {
    bool running = false;
    FILE *in = nullptr;  // the oracle's stdout
    FILE *out = nullptr; // the oracle's stdin
#ifdef _WIN32
    HANDLE handle = nullptr;
#else
    pid_t pid = -1;
#endif
};

// [0] the sound oracle (CORACPP_ORACLE), [1] the unsound oracle (CORACPP_ORACLE_UNSOUND)
Process oracles[2];

const char *aux_command(int which) {
    const char *name = which ? "CORACPP_ORACLE_UNSOUND" : "CORACPP_ORACLE";
    const char *command = std::getenv(name);
    if (!command || !*command)
        CORA_FAIL(Error::special_error, std::string("lean: set ") + name +
                                            " to the command that starts "
                                            "the CORALean " +
                                            (which ? "unsound " : "") +
                                            "oracle, e.g. "
                                            "'cd CORALean && lake exe oracle'");
    return command;
}

// The unsound oracle serves the dtype without a soundness guarantee.
int aux_which(const std::string &dtype) { return dtype == "nearest"; }

#ifdef _WIN32

// Starts `cmd /c <command>` with its stdin and stdout on two pipes.
void aux_start(Process &oracle, int which) {
    const std::string command = std::string("cmd /s /c \"") + aux_command(which) + "\"";
    SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE child_in, parent_out, parent_in, child_out;
    if (!CreatePipe(&child_in, &parent_out, &inheritable, 0) ||
        !CreatePipe(&parent_in, &child_out, &inheritable, 0))
        CORA_FAIL(Error::special_error, "lean: cannot create pipes");
    // the parent's ends must not be inherited by the child
    SetHandleInformation(parent_out, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(parent_in, HANDLE_FLAG_INHERIT, 0);
    // the child reads and writes the pipes instead of a console
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = child_in;
    startup.hStdOutput = child_out;
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    // CreateProcess edits the command line in place, so it gets a mutable copy
    PROCESS_INFORMATION info{};
    std::string mutable_command = command;
    if (!CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, TRUE, 0, nullptr,
                        nullptr, &startup, &info))
        CORA_FAIL(Error::special_error, "lean: cannot start the oracle process: " + command);
    CloseHandle(child_in);
    CloseHandle(child_out);
    CloseHandle(info.hThread);
    // the parent talks to the child through C streams on its ends of the pipes
    oracle.handle = info.hProcess;
    oracle.out = _fdopen(_open_osfhandle(reinterpret_cast<intptr_t>(parent_out), _O_BINARY), "wb");
    oracle.in = _fdopen(
        _open_osfhandle(reinterpret_cast<intptr_t>(parent_in), _O_RDONLY | _O_BINARY), "rb");
    oracle.running = true;
}

void aux_wait(Process &oracle) {
    WaitForSingleObject(oracle.handle, INFINITE);
    CloseHandle(oracle.handle);
}

#else

// Starts `sh -c <command>` with its stdin and stdout on two pipes.
void aux_start(Process &oracle, int which) {
    const char *command = aux_command(which);
    int to_child[2], from_child[2];
    if (pipe(to_child) || pipe(from_child))
        CORA_FAIL(Error::special_error, "lean: cannot create pipes");
    oracle.pid = fork();
    if (oracle.pid < 0) CORA_FAIL(Error::special_error, "lean: cannot start the oracle process");
    if (oracle.pid == 0) {
        dup2(to_child[0], 0);
        dup2(from_child[1], 1);
        for (const int fd : {to_child[0], to_child[1], from_child[0], from_child[1]}) close(fd);
        execl("/bin/sh", "sh", "-c", command, static_cast<char *>(nullptr));
        _exit(127);
    }
    // the parent keeps the other ends
    close(to_child[0]);
    close(from_child[1]);
    oracle.out = fdopen(to_child[1], "w");
    oracle.in = fdopen(from_child[0], "r");
    oracle.running = true;
}

void aux_wait(Process &oracle) { waitpid(oracle.pid, nullptr, 0); }

#endif

std::string aux_read_line(Process &oracle) {
    std::string line;
    int ch;
    // a response is one line
    while ((ch = fgetc(oracle.in)) != EOF && ch != '\n') line += static_cast<char>(ch);
    if (ch == EOF && line.empty())
        CORA_FAIL(Error::special_error, "lean: the oracle process ended without answering");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Dtype -----------------------------------------------------------------------------------

void set_default_dtype(const std::string &spec) {
    detail::Entry entry;
    CORA_CHECK(aux_known(spec), Error::wrong_value, [&] { return aux_unknown(spec); });
    std::lock_guard<std::mutex> lock(default_mutex);
    default_dtype = spec;
}

std::string dtype() {
    if (scoped_dtype) return *scoped_dtype;
    std::lock_guard<std::mutex> lock(default_mutex);
    return default_dtype;
}

DTypeScope::DTypeScope(const std::string &spec) : dtype_(spec), previous_(scoped_dtype) {
    detail::Entry entry;
    CORA_CHECK(aux_known(spec), Error::wrong_value, [&] { return aux_unknown(spec); });
    if (spec == "nearest")
        warn("CORA:lean", "the dtype 'nearest' rounds to nearest without any guarantee; use it "
                          "only to compare with the lean backend");
    scoped_dtype = &dtype_;
}

DTypeScope::~DTypeScope() { scoped_dtype = previous_; }

// Calls ------------------------------------------------------------------------------------

// Failures of the oracle surface as exceptions with its message.
Json call(const Json &request) {
    const Json *dtype = request.find("dtype");
    const int which = dtype ? aux_which(dtype->text) : 0;
    Process &oracle = oracles[which];
    if (!oracle.running) aux_start(oracle, which);
    fputs((request.dump() + "\n").c_str(), oracle.out);
    fflush(oracle.out);
    // one line per request, one per response
    const Json response = Json::parse(aux_read_line(oracle));
    const Json *ok = response.find("ok");
    if (!ok || !ok->boolean) {
        const Json *error = response.find("error");
        const std::string message = error ? error->text : "no message";
        CORA_FAIL(Error::special_error, "lean: the oracle failed: " + message);
    }
    return response;
}

// The op list of `ping`, read at the first question.
bool serves(const std::string &op) {
    static std::mutex mutex;
    static std::set<std::string> ops;
    std::lock_guard<std::mutex> lock(mutex);
    if (ops.empty()) {
        const Json ping = call(Json::object().set("op", "ping"));
        for (const Json &name : ping.at("ops").items) ops.insert(name.text);
    }
    return ops.count(op) > 0;
}

void shutdown() {
    for (Process &oracle : oracles) {
        if (!oracle.running) continue;
        fclose(oracle.out);
        fclose(oracle.in);
        aux_wait(oracle);
        oracle = Process{};
    }
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
