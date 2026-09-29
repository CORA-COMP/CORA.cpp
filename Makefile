# CORA.cpp - no build system beyond this file.
#
#   scripts/with_env.sh make test example   the tests and the C++ examples, in the conda environment
#   make test example                       the same, inside an activated environment or a container
#   make TORCH= test                        without libtorch (Eigen only)
#   make python                             the Python package, into $(BUILD)/cora
#   make run-<name>                         builds and runs examples/cpp/<name>.cpp
#   make debug-<name>                       builds it with -O0 -g into $(BUILD)-debug
#   make DEBUG=1 <target>                   -O0 -g, for the debugger
#   make all                                the CORA-COMP binary (see competition/README.md)
#   make test-competition                   the competition's tests
#
# libtorch is taken from the torch package of $(PYTHON), if there is one; TORCH=/path overrides
# it and TORCH= turns it off. The library needs only Eigen and the compiler; the competition
# binary also needs GLPK. Anything in local.mk (written by scripts/setup_local.sh) is included.

-include local.mk

BUILD     ?= build
CXX       ?= g++
PYTHON    ?= python3
# Eigen's AVX512 triangular-solve kernel makes GCC 16 report uninitialized values and out-of-bounds
# accesses that are not there; those two are off so that a real warning is visible.
WARN       = -Wall -Wextra -Wno-maybe-uninitialized -Wno-array-bounds
PIC        = -fPIC  # the Python module is a shared object, so every object must be relocatable
LDLIBS     =
LDFLAGS    = -fopenmp

# Own variable, not CXXFLAGS: an activated conda environment sets that one to its own -O2 flags.
ifdef DEBUG
OPT       ?= -O0 -g -std=c++20 -fopenmp
else
OPT       ?= -O3 -march=native -std=c++20 -fopenmp -DNDEBUG -DEIGEN_NO_DEBUG
endif

EIGEN     ?= $(firstword $(wildcard $(CONDA_PREFIX)/include/eigen3 /usr/include/eigen3 /usr/local/include/eigen3))
INCLUDES   = -Isrc -isystem $(EIGEN)
ifdef CONDA_PREFIX
INCLUDES  += -isystem $(CONDA_PREFIX)/include
LDFLAGS   += -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
endif

# ---------------------------------- libtorch -------------------------------------------
# The directory of the torch package and the C++ ABI it was built with, asked of Python once.

ifeq ($(origin TORCH),undefined)
TORCH_INFO := $(shell $(PYTHON) -c "import os, torch; print(os.path.dirname(torch.__file__), int(torch._C._GLIBCXX_USE_CXX11_ABI))" 2>/dev/null)
TORCH      := $(word 1,$(TORCH_INFO))
endif
ifeq ($(origin TORCH_ABI),undefined)
TORCH_ABI  := $(or $(word 2,$(TORCH_INFO)),1)
endif

# ---------------------------------- the library ---------------------------------------
# src/ mirrors MATLAB CORA's folders, one operation per file; sources are found, so a new
# operation needs no edit here.

LIB_SRC = $(filter-out src/python/% src/tensor/torch.cpp,$(shell find src -name '*.cpp' | sort))

ifneq ($(TORCH),)
LIB_SRC  += src/tensor/torch.cpp
DEFS      = -DCORACPP_TORCH -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
INCLUDES += -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include
LDFLAGS  += -L$(TORCH)/lib -Wl,-rpath,$(TORCH)/lib
LDLIBS   += -ltorch -ltorch_cpu -lc10
# --no-as-needed: nothing references libtorch_cuda directly, but without it linked the CUDA
# kernels are not registered and a GPU is never used.
ifneq ($(wildcard $(TORCH)/lib/libtorch_cuda.so),)
LDLIBS   += -Wl,--no-as-needed -ltorch_cuda -lc10_cuda -Wl,--as-needed
endif
endif

# An external BLAS is opt-in, since it changes which library does the arithmetic.
ifneq ($(BLAS),)
DEFS     += -DEIGEN_USE_BLAS
LDLIBS   += -lopenblas
endif

LIB_OBJ = $(LIB_SRC:%.cpp=$(BUILD)/%.o)

# rng.cpp is the one file compiled with -ffast-math, so the logarithm and the sine of
# Box-Muller vectorize; everything that results depend on keeps strict IEEE arithmetic.
$(BUILD)/src/global/rng.o: src/global/rng.cpp
	@mkdir -p $(@D)
	$(CXX) $(OPT) $(PIC) $(DEFS) -ffast-math $(WARN) $(INCLUDES) -c -o $@ $<

$(BUILD)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(OPT) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -c -o $@ $<

# ---------------------------------- tests ----------------------------------------------
# tests/ mirrors src/; a test ending in _torch needs libtorch. test_codingConventions.py checks
# the conventions of the sources.

TEST_SRC = $(shell find tests -name 'test_*.cpp' -not -path 'tests/competition/*' | sort)
TESTS    = $(patsubst %.cpp,$(BUILD)/%,$(filter-out %_torch.cpp,$(TEST_SRC)))
ifneq ($(TORCH),)
TESTS   += $(patsubst %.cpp,$(BUILD)/%,$(filter %_torch.cpp,$(TEST_SRC)))
endif

$(TESTS): $(BUILD)/tests/%: tests/%.cpp $(LIB_OBJ) tests/testing.h
	@mkdir -p $(@D)
	$(CXX) $(OPT) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -Itests $(LDFLAGS) -o $@ $(filter %.cpp %.o,$^) $(LDLIBS)

test: $(TESTS)
	@$(PYTHON) tests/global/test_codingConventions.py
	@for t in $(TESTS); do echo "== $$t"; $$t || exit 1; done

# --------------------------------- examples --------------------------------------------
# One small file per topic in examples/cpp; the ones that name libtorch types need it.

EXAMPLE_SRC = $(sort $(wildcard examples/cpp/example_*.cpp))
EXAMPLES    = $(patsubst %.cpp,$(BUILD)/%,$(filter-out %_gpu.cpp %_batch.cpp %_gradient.cpp,$(EXAMPLE_SRC)))
ifneq ($(TORCH),)
EXAMPLES    = $(patsubst %.cpp,$(BUILD)/%,$(EXAMPLE_SRC))
endif

# Every example starts with the CORA START block: it is injected here, not written in each file.
$(BUILD)/examples/cpp/%: examples/cpp/%.cpp $(LIB_OBJ) src/global/banner.h
	@mkdir -p $(@D)
	$(CXX) $(OPT) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -include global/banner.h -DCORACPP_PROGRAM='"$*"' $(LDFLAGS) -o $@ examples/cpp/$*.cpp $(LIB_OBJ) $(LDLIBS)

example: $(EXAMPLES)
	@for e in $(EXAMPLES); do echo "== $$e"; $$e || exit 1; done

# One example: run-<name> builds and runs it, debug-<name> builds it for the debugger.
DEBUG_BUILD ?= $(BUILD)-debug

run-%: $(BUILD)/examples/cpp/%
	@$<

debug-%:
	@$(MAKE) --no-print-directory DEBUG=1 BUILD=$(DEBUG_BUILD) $(DEBUG_BUILD)/examples/cpp/$*

# ---------------------------------- Python ---------------------------------------------
# The cora package: the compiled module and the Python files beside it, in $(BUILD)/cora.
# Put $(BUILD) on PYTHONPATH. It needs libtorch, since torch tensors are its inputs.

PYEXT   = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYINC   = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYFILES = $(wildcard src/python/cora/*.py)

python: $(BUILD)/cora/_cora$(PYEXT) $(PYFILES:src/python/cora/%=$(BUILD)/cora/%)

$(BUILD)/cora/%.py: src/python/cora/%.py
	@mkdir -p $(@D)
	cp $< $@

$(BUILD)/cora/_cora$(PYEXT): src/python/bindings.cpp $(LIB_OBJ)
	@test -n "$(TORCH)" || { echo "make python needs libtorch: a torch package for $(PYTHON), or TORCH=/path"; exit 1; }
	@mkdir -p $(@D)
	$(CXX) -shared $(OPT) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -I$(PYINC) $(LDFLAGS) \
	    -o $@ $^ $(LDLIBS) -L$(TORCH)/lib -ltorch_python

# ------------------------------ the competition ----------------------------------------
# competition/ is the CORA-COMP harness and its own tuned sets; it builds on the library and
# needs GLPK. See competition/README.md.

COMP_SRC = $(filter-out competition/main.cpp competition/torch_backend.cpp \
                        competition/torch_none.cpp competition/sets/torch_contains.cpp, \
                        $(shell find competition -name '*.cpp' | sort))
ifeq ($(TORCH),)
COMP_SRC += competition/torch_none.cpp
else
COMP_SRC += competition/torch_backend.cpp competition/sets/torch_contains.cpp
endif
COMP_OBJ  = $(COMP_SRC:%.cpp=$(BUILD)/%.o)
COMP_LIBS = -lglpk

COMP_TEST_SRC = $(shell find tests/competition -name 'test_*.cpp' | sort)
COMP_TESTS    = $(patsubst %.cpp,$(BUILD)/%,$(filter-out %_torch.cpp,$(COMP_TEST_SRC)))
ifneq ($(TORCH),)
COMP_TESTS   += $(patsubst %.cpp,$(BUILD)/%,$(filter %_torch.cpp,$(COMP_TEST_SRC)))
endif

all: $(BUILD)/coracpp

$(BUILD)/coracpp: $(LIB_OBJ) $(COMP_OBJ) $(BUILD)/competition/main.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS) $(COMP_LIBS)

test-competition: $(COMP_TESTS)
	@for t in $(COMP_TESTS); do echo "== $$t"; $$t || exit 1; done

$(COMP_TESTS): $(BUILD)/tests/%: tests/%.cpp $(LIB_OBJ) $(COMP_OBJ) tests/testing.h
	@mkdir -p $(@D)
	$(CXX) $(OPT) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -Icompetition -Itests $(LDFLAGS) -o $@ $(filter %.cpp %.o,$^) $(LDLIBS) $(COMP_LIBS)

clean:
	rm -rf $(BUILD)

.PHONY: all test test-competition clean python example
