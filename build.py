import logging
import os
import shutil
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import toml
import yaml
from xvfbwrapper import Xvfb

BIN_DIR = Path("artefacts/bin").resolve()
LIB_DIR = Path("artefacts/lib").resolve()
BUILD_DIR = Path("artefacts/build").resolve()
SRC_DIR = Path("src").resolve()
PATCH_DIR = Path("patch").resolve()


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    version: str
    recursive: bool = False
    shallow_clone: bool = False
    post_checkout: Optional[Tuple[str]] = None

    @property
    def src(self):
        return SRC_DIR / self.name

    def _clone(self, c):
        clone = ["git", "clone", "--progress"]
        if self.recursive:
            clone.append("--recursive")

        if self.shallow_clone:
            clone.extend(["--depth", "1"])

        clone.extend([self.url, str(self.src)])
        c.run(" ".join(clone))

        return clone

    def _checkout(self, c):
        c.run(" ".join(["git", "-C", str(self.src), "checkout", self.version]))

    def fetch(self, c):
        if self.src.exists():
            return

        self._clone(c)
        self._checkout(c)

    def remove(self):
        shutil.rmtree(self.src)
        logging.info(f"-> removed: {self.src}")


@dataclass(frozen=True)
class Artefact:
    repo: Optional[Repo] = None

    @property
    def src(self) -> Path:
        return SRC_DIR / self.repo.name

    @property
    def install_prefix(self) -> Path:
        return BIN_DIR / self.repo.name

    @property
    def path(self) -> Path:
        return self.install_prefix / self.name

    @property
    def installed(self) -> Path:
        return os.path.lexists(self.path)

    @property
    def build_dir(self) -> Path:
        return BUILD_DIR / self.repo.name

    def remove(self, including_src=False):
        logging.info(f"Removing {self.name}:")
        self.path.unlink(missing_ok=True)
        logging.info(f"-> removed: {self.path}")
        shutil.rmtree(self.build_dir)
        logging.info(f"-> removed: {self.build_dir}")
        if including_src:
            self.repo.remove()


class Heaptrack(Artefact):
    def build(self, c):
        self.repo.fetch(c)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.install_prefix.mkdir(parents=True, exist_ok=True)

        cmake = [
            "cmake",
            "-S",
            str(self.repo.src),
            "-B",
            str(self.build_dir),
            f"-DCMAKE_INSTALL_PREFIX={self.install_prefix}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        c.run(" ".join(cmake))
        with c.cd(self.build_dir):
            c.run("make -j install")

    @property
    def path(self) -> Path:
        return self.install_prefix / "bin" / "heaptrack"


class Alloy(Artefact):
    DEFAULT_FLAGS: ClassVar[Dict[str, bool]] = {
        "finalizer-safety-analysis": False,
        "finalizer-elision": True,
        "premature-finalizer-prevention": True,
        "premature-finalizer-prevention-optimize": True,
        "gc-default-allocator": True,
    }

    def __init__(self, profile: "ExperimentProfile", metrics=False):
        self.__dict__.update(ALLOY.__dict__)
        self.profile = profile
        self.metrics = metrics
        self._config = None
        self.flags = self.DEFAULT_FLAGS.copy() | (self.profile.alloy_flags or {})

    @property
    def config(self) -> Path:
        if self._config:
            return self._config

        flags = self.flags.copy()
        if self.metrics:
            flags.update({"gc-metrics": True})

        config = {
            "alloy": flags,
            "rust": {
                "codegen-units": 0,
                "optimize": True,
                "debug": False,
            },
            "build": {
                "install-stage": 2,
            },
            "llvm": {
                "download-ci-llvm": True,
            },
            "install": {
                "sysconfdir": "etc",
            },
        }
        file = self.src / f"{self.name.replace('-','.')}.config.toml"
        with open(file, "w") as f:
            toml.dump(config, f)
        return file

    @property
    def name(self) -> str:
        if not self.flags["gc-default-allocator"]:
            base = "rustc-upstream"
        elif self.flags == self.DEFAULT_FLAGS:
            base = "default"
        else:
            base = self.profile.full
        return f"{base}-metrics" if self.metrics else base

    @property
    def install_prefix(self) -> Path:
        return BIN_DIR / self.repo.name / self.name

    @property
    def build_dir(self) -> Path:
        return BUILD_DIR / self.repo.name / self.name

    @property
    def path(self) -> Path:
        return self.install_prefix / "bin" / "rustc"

    @property
    def installed(self) -> bool:
        return self.path.exists()

    def build_alloy(self, c):
        if self.installed:
            logging.info(f"{self.path} already exists. Skipping...")
            return

        self.repo.fetch(c)
        self.install_prefix.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Starting build: {self.name}")
        build_cmd = [
            f"{str(self.src)}/x.py",
            "install",
            "--config",
            str(self.config),
            "--stage",
            "2",
            "--build-dir",
            str(self.build_dir),
            "--set",
            "build.docs=false",
            "--set",
            f"install.prefix={str(self.install_prefix)}",
            "--set",
            "install.sysconfdir=etc",
        ]

        try:
            c.run(" ".join(build_cmd))
        except Exception as e:
            logging.error(f"Failed to build {self.src}: {e}")
            raise

        logging.info(f"Successfully installed {self.install_prefix}")


REBENCH_DATA = "results.data"

HS_MAP = {
    "ripgrep": {
        "heapsize-s": "32M",
        "heapsize-l": "64M",
        "heapsize-xl": "128M",
    },
    "binary_trees": {
        "heapsize-s": "4M",
        "heapsize-l": "8M",
        "heapsize-xl": "16M",
    },
    "fd": {
        "heapsize-s": "16M",
        "heapsize-l": "32M",
        "heapsize-xl": "64M",
    },
    "alacritty": {
        "heapsize-s": "16M",
        "heapsize-l": "32M",
        "heapsize-xl": "64M",
    },
    "regex_redux": {
        "heapsize-s": "256M",
        "heapsize-l": "512M",
        "heapsize-xl": "1024M",
    },
    "som-rs-bc": {
        "heapsize-s": "32M",
        "heapsize-l": "64M",
        "heapsize-xl": "128M",
    },
    "som-rs-ast": {
        "heapsize-s": "64M",
        "heapsize-l": "96M",
        "heapsize-xl": "128M",
    },
    "grmtools": {
        "heapsize-s": "1024M",
        "heapsize-l": "2048M",
        "heapsize-xl": "4096M",
    },
}


class Aggregation(Enum):
    INDIVIDUAL = "individual"
    SUITE = "suite"
    OVERALL = "overall"
    SUITE_ARITH = "suite_arith"
    SUITE_GEO = "suite_geo"
    GLOBAL_ARITH = "global_arith"
    GLOBAL_GEO = "global_geo"
    SNAPSHOT = "snapshot"


class Agg(Enum):
    ARITH = 1
    GEO = 2


class Metric(Enum):
    def __init__(self, value, latex=None):
        self._value_ = value
        self.latex = latex

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return None

    COLLECTION_NUMBER = "collection_number"
    TOTAL_COLLECTIONS = (
        "total_collections",
        r"\# collections",
    )
    MINOR_COLLECTIONS = "minor_collections"
    MAJOR_COLLECTIONS = "major_collections"
    PHASE = "kind"
    MEM_HSIZE_ENTRY = "mem_hsize_entry"
    MEM_HSIZE_EXIT = "mem_hsize_exit"
    MEM_ALLOCD_ENTRY = "mem_allocd_entry"
    MEM_ALLOCD_EXIT = "mem_allocd_exit"
    MEM_ALLOCD_FLZQ = "mem_allocd_flzq"
    MEM_FREED_EXPLICIT_ENTRY = "mem_freed_explicit_entry"
    MEM_FREED_EXPLICIT_EXIT = "mem_freed_explicit_exit"
    MEM_FREED_SWEPT = "mem_freed_swept"
    MEM_FREED_FLZ = "mem_freed_flz"
    TIME_MARKING = "time_marking"
    TIME_FIN_Q = "time_fin_q"
    TIME_SWEEPING = "time_sweeping"
    TIME_TOTAL = "time_total"
    FLZ_REGISTERED = "flz_registered"
    FLZ_ELIDED = "flz_elided"
    FLZ_RUN = "flz_run"
    OBJ_ALLOCD_ARC = "obj_allocd_arc"
    OBJ_ALLOCD_BOX = "obj_allocd_box"
    OBJ_ALLOCD_FLZQ = "obj_allocd_flzq"
    OBJ_ALLOCD_GC = "obj_allocd_gc"
    OBJ_ALLOCD_RC = "obj_allocd_rc"
    OBJ_FREED_SWEPT = "obj_freed_swept"
    OBJ_FREED_EXPLICIT = "obj_freed_explicit"

    MEM_HSIZE_AVG = (
        "mem_hsize_avg",
        "Avg. heap size",
    )
    MAX_MEMORY = "max_heap_size"
    PCT_ELIDED = (
        "pct_elided",
        r"\% Fin. elided",
    )
    GC_TIME = (
        "gc_time",
        "Total GC pause time",
    )

    WALLCLOCK = (
        "total",
        "Wall-clock time",
    )
    USER = (
        "usr",
        "User time",
    )

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return self.value

    @property
    def pathname(self):
        if self == Metric.WALLCLOCK:
            return "wallclock"

        if self == Metric.USER:
            return "user"

        return self.value


class DoubleQuotedStr(str):
    pass


def double_quoted_str_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


yaml.add_representer(DoubleQuotedStr, double_quoted_str_representer)


REBENCH_EXEC = Path(".venv/bin/rebench")
REBENCH_CONF = Path("rebench.conf").resolve()
RESULTS_DIR = Path("results").resolve()
PATCH_DIR = Path("patch").resolve()
DEFAULT_PEXECS = 30
PEXECS = 30
DEFAULT_MEASUREMENTS = ["perf"]


class Measurement(Enum):
    PERF = "perf"
    METRICS = "metrics"
    HEAPTRACK = "heaptrack"


class ExperimentProfile(Enum):
    def __init__(self, value, latex, alloy_flags=None):
        self._value_ = value
        self.latex = latex
        self.alloy_flags = alloy_flags

    @property
    def experiment(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def full(self) -> str:
        return f"{self.experiment}-{self.value}"

    @property
    def path(self) -> Path:
        exp = self.name.lower()
        return Path(self.experiment) / Path(exp)

    @classmethod
    def experiments(cls, pexecs=DEFAULT_PEXECS) -> List["Experiment"]:
        return [
            Experiment(tuple(set(cls) & set(suite.profiles)), m, suite, pexecs)
            for suite in BenchmarkSuite.all()
            for m in cls.measurements
            if not set(cls).isdisjoint(set(suite.profiles))
        ]

    def __lt__(self, other):
        return self.value < other.value

    @classmethod
    def has_value(cls, value):
        if "default" in value:
            return True
        return value in cls._value2member_map_


class GCVS(ExperimentProfile):
    GC = ("gc", r"\texttt{Gc<T>} (Alloy)")
    RC = ("rc", r"\texttt{Arc<T>}/\texttt{Rc<T>}")
    ARC = ("arc", r"\texttt{Arc<T>}/\texttt{Rc<T>}")
    BASELINE = (
        "baseline",
        r"\texttt{Arc<T>}/\texttt{Rc<T>} (jemalloc)",
        {"gc-default-allocator": False},
    )
    TYPED_ARENA = ("typed-arena", r"\texttt{Arena<T>")
    RUST_GC = ("rust-gc", r"\texttt{Gc<T>} (Rust-GC)")

    @classmethod
    @property
    def measurements(cls):
        return [Measurement.PERF, Measurement.HEAPTRACK]

    @classmethod
    def _missing_(cls, value):
        exp = cls.__name__.lower()

        for member in cls:
            if f"rc" in value:
                return cls.RC
            elif f"arc" in value:
                return cls.RC
            elif f"baseline" in value:
                return cls.BASELINE
            elif f"typed-arena" in value:
                return cls.TYPED_ARENA
            elif f"rust-gc" in value:
                return cls.RUST_GC
            elif "default" in value:
                return cls.GC
        return None

    @property
    def baseline(cls):
        return cls.BASELINE


class PremOpt(ExperimentProfile):
    NAIVE = (
        "naive",
        "All Barriers",
        {"premature-finalizer-prevention-optimize": False},
    )
    OPT = ("opt", "Optimized Barriers")
    NONE = (
        "none",
        "No Barriers ",
        {
            "premature-finalizer-prevention": False,
            "premature-finalizer-prevention-optimize": False,
        },
    )

    @classmethod
    @property
    def measurements(cls):
        return [Measurement.PERF, Measurement.METRICS]

    @classmethod
    def _missing_(cls, value):
        if value == "default":
            return cls.OPT

        profile = value.rsplit("-", 1)[-1]

        for member in cls:
            if member.value == profile:
                return member
        return None

    @property
    def baseline(cls):
        return cls.NONE


class Elision(ExperimentProfile):
    NAIVE = ("naive", "Before", {"finalizer-elision": False})
    OPT = ("opt", "After")

    @classmethod
    @property
    def measurements(cls):
        return [Measurement.PERF, Measurement.METRICS]

    @classmethod
    def _missing_(cls, value):
        exp = cls.__name__.lower()

        for member in cls:
            if f"{exp}-{member.value}" in value:
                return member
            elif "default" in value:
                return cls.OPT
        return None

    @property
    def baseline(cls):
        return cls.NAIVE

    def __repr__(self):
        return self.value


class HeapSize(ExperimentProfile):
    DEFAULT = ("default", r"Default")
    S = ("s", "1")
    L = ("l", "2")
    XL = ("xl", "3")

    @classmethod
    @property
    def measurements(cls):
        return [Measurement.PERF]

    @classmethod
    def _missing_(cls, value):
        exp = cls.__name__.lower()

        for member in cls:
            if f"{exp}-{member.value}" in value:
                return member
            elif "default" in value:
                return cls.DEFAULT
        return None

    @property
    def baseline(cls):
        return cls.DEFAULT

    def __repr__(self):
        return self.value


def cargo_build(c, src, outdir, install_dir, build_artefact, bin, env=None):
    src = Path(src)
    outdir = Path(outdir)
    install_dir = Path(install_dir)

    install_dir.mkdir(parents=True, exist_ok=True)
    target_bin_dir = outdir / "release"

    install_path = install_dir / bin
    if install_path.exists():
        logging.info(f"{install_path} already exists. Skipping...")
        return

    logging.info(f"Building {install_path} from {src})")

    build_cmd = [
        "cargo",
        "build",
        "--release",
        "--target-dir",
        str(outdir),
        "--bin",
        build_artefact,
    ]

    with c.cd(str(src)):
        try:
            c.run(" ".join(build_cmd), env=env)
        except Exception as e:
            logging.error(f"Failed to build {src}: {e}")
            raise

    # Create symlinks for newly built binaries
    source_path = target_bin_dir / build_artefact
    target_path = install_dir / bin

    if not source_path.exists():
        logging.warning(f"Built binary {source_path} not found after build")
        return

    try:
        shutil.copy2(source_path, target_path)
        logging.info(f"Copied {source_path} -> {target_path}")
    except OSError as e:
        logging.error(f"Failed to copy {bin}: {e}")
        raise

    logging.info(f"Successfully installed {source_path} --> {target_path}")


def has_unstaged_changes(c, directory):
    with c.cd(directory):
        result = c.run("git diff --quiet", warn=True)
        return result.exited != 0


def patch_repo(c, repo_path, patch_file):
    repo_path = Path(repo_path)
    patch_file = Path(patch_file)
    logging.info(f"Attempting to patch: {repo_path}")

    with c.cd(str(repo_path)):
        # Reset to clean state
        logging.info("Resetting repository to HEAD...")
        c.run("git reset --hard HEAD")

        if not patch_file.exists():
            logging.info(f"No patch file exists: {patch_file}. Skipping..")
        else:
            # Apply the patch
            logging.info(f"Applying patch: {patch_file}")
            c.run(f"git apply {patch_file.absolute()}")

            logging.info("Patch applied successfully")


@dataclass(frozen=True)
class Benchmark:
    name: str
    extra_args: Any = None
    latex_name: str = None

    @property
    def latex(self):
        return self.latex_name or self.name

    def __repr__(self):
        return self.latex

    def __hash__(self) -> int:
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    benchmarks: Tuple["Benchmark"]
    gcvs: Optional[Tuple[ExperimentProfile]] = None

    @classmethod
    def prerequisites(self, c):
        install_linux(c)
        SOMRS.fetch(c)
        cargo_build(
            c,
            ERRORGEN,
            BUILD_DIR / "extra/errorgen",
            BIN_DIR / "extra/errorgen",
            "errorgen",
            "errorgen",
        )
        for repo in JAVA_SRCS:
            repo.fetch(c)

        for repo in JAVA_SRCS:
            if has_unstaged_changes(c, repo.src):
                logging.info(
                    f"{repo.name} sources have already had parser-errors introduced. Skipping..."
                )
                continue
            with c.cd(repo.src):
                c.run(
                    str(BIN_DIR / "extra" / "errorgen" / "errorgen") + " " + repo.name
                )

    @property
    def path(self) -> Path:
        pass

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    @property
    def latex(self) -> str:
        return f"{self.name.replace('_',' ')}"

    def raw_data(self, measurement) -> Path:
        return RESULTS_DIR / self.name / measurement.value / "results.data"

    @property
    def args(self) -> str:
        return str(self.cmd_args)

    @property
    def cmd_args(self) -> str:
        return ""

    @property
    def profiles(self) -> tuple[ExperimentProfile]:
        p = tuple(PremOpt) + tuple(Elision) + tuple(HeapSize)
        if self.gcvs:
            p = p + self.gcvs + (GCVS.BASELINE, GCVS.GC)
        return p

    @classmethod
    def all(cls) -> Set["BenchmarkSuite"]:
        return {
            Grmtools(
                "grmtools",
                (
                    Benchmark(
                        "eclipse",
                        extra_args=str(SRC_DIR / "eclipse"),
                        latex_name="Eclipse",
                    ),
                    Benchmark(
                        "hadoop",
                        extra_args=str(SRC_DIR / "hadoop"),
                        latex_name="Hadoop",
                    ),
                    Benchmark(
                        "spring",
                        extra_args=str(SRC_DIR / "spring"),
                        latex_name="Spring",
                    ),
                    Benchmark(
                        "jenkins",
                        extra_args=str(SRC_DIR / "jenkins"),
                        latex_name="Jenkins",
                    ),
                ),
                gcvs=(GCVS.RC,),
            ),
            BinaryTrees(
                "binary_trees",
                (Benchmark("binary_trees", extra_args=14, latex_name="Binary Trees"),),
                (GCVS.ARC, GCVS.TYPED_ARENA, GCVS.BASELINE, GCVS.RUST_GC),
            ),
            RegexRedux(
                "regex_redux",
                (
                    Benchmark(
                        "regex_redux",
                        extra_args="0 < redux_input.txt",
                        latex_name="Regex Redux",
                    ),
                ),
                (GCVS.ARC,),
            ),
            RipGrep(
                "ripgrep",
                (
                    Benchmark("linux_alternates", latex_name="Alternates"),
                    Benchmark("linux_alternates_casei", latex_name="Alternates (-i)"),
                    Benchmark("linux_literal", latex_name="Literal"),
                    Benchmark("linux_literal_casei", latex_name="Literal (-i)"),
                    Benchmark("linux_literal_default", latex_name="Literal (default)"),
                    Benchmark("linux_literal_mmap", latex_name="Literal (mmap)"),
                    Benchmark(
                        "linux_literal_casei_mmap", latex_name="Literal (mmap, -i)"
                    ),
                    Benchmark("linux_word", latex_name="Word"),
                    Benchmark("linux_unicode_greek", latex_name="UTF Greek"),
                    Benchmark("linux_unicode_greek_casei", latex_name="UTF Greek (-i)"),
                    Benchmark("linux_unicode_word_1", latex_name="UTF Word"),
                    Benchmark("linux_unicode_word_2", latex_name="UTF Word (alt.)"),
                    Benchmark("linux_re_literal_suffix", latex_name="Literal (regex)"),
                ),
                (GCVS.ARC,),
            ),
            Alacritty(
                "alacritty",
                (
                    Benchmark("cursor_motion", latex_name="Cur. Motion"),
                    Benchmark("dense_cells", latex_name="Dense Cells"),
                    Benchmark("light_cells", latex_name="Light Cells"),
                    Benchmark("scrolling", latex_name="Scroll"),
                    Benchmark("scrolling_bottom_region", latex_name="Scroll Btm"),
                    Benchmark(
                        "scrolling_bottom_small_region", latex_name="Scroll Btm (small)"
                    ),
                    Benchmark("scrolling_fullscreen", latex_name="Scroll (fullscreen)"),
                    Benchmark("scrolling_top_region", latex_name="Scroll Top"),
                    Benchmark(
                        "scrolling_top_small_region", latex_name="Scroll Top (small)"
                    ),
                    Benchmark("unicode", latex_name="Unicode"),
                ),
                (GCVS.ARC,),
            ),
            FdFind(
                "fd",
                (
                    Benchmark(
                        name="no-pattern",
                        extra_args=f"--hidden --no-ignore . '{LINUX.src}'",
                        latex_name="No Pattern",
                    ),
                    Benchmark(
                        name="simple-pattern",
                        extra_args=f"'.*[0-9]\\.jpg$' . '{LINUX.src}'",
                        latex_name="Simple",
                    ),
                    Benchmark(
                        name="simple-pattern-HI",
                        extra_args=f"-HI '.*[0-9]\\.jpg$' . '{LINUX.src}'",
                        latex_name="Simple (-HI)",
                    ),
                    Benchmark(
                        name="file-extension",
                        extra_args=f"-HI --extension jpg . '{LINUX.src}'",
                        latex_name="File Extension",
                    ),
                    Benchmark(
                        name="file-type",
                        extra_args=f"-HI --type l . '{LINUX.src}'",
                        latex_name="File Type",
                    ),
                    Benchmark(
                        name="command-execution",
                        extra_args=f"'ab' . '{LINUX.src}' --exec echo",
                        latex_name="Cmd Exec.",
                    ),
                    Benchmark(
                        name="command-execution-large-output",
                        extra_args=f"-tf 'ab' . '{LINUX.src}' --exec echo",
                        latex_name="Cmd Exec. (large)",
                    ),
                ),
                (GCVS.ARC,),
            ),
            SomrsAST(
                "som-rs-ast",
                SOM_BENCHMARKS,
                (GCVS.RC,),
            ),
            SomrsBC("som-rs-bc", SOM_BENCHMARKS, (GCVS.RC,)),
        }


JAVA_SRCS = (
    Repo(
        name="jenkins",
        shallow_clone=True,
        url="https://github.com/jenkinsci/jenkins",
        version="master",
    ),
    Repo(
        name="spring",
        shallow_clone=True,
        url="https://github.com/spring-projects/spring-framework",
        version="main",
    ),
    Repo(
        name="hadoop",
        shallow_clone=True,
        url="https://github.com/apache/hadoop",
        version="trunk",
    ),
    Repo(
        name="eclipse",
        shallow_clone=True,
        url="https://github.com/eclipse-platform/eclipse.platform",
        version="master",
    ),
)
ERRORGEN = Path(SRC_DIR / "errorgen")


class Grmtools(BenchmarkSuite):
    GRMTOOLS = Repo(
        name="grmtools",
        url="https://github.com/softdevteam/grmtools",
        version="a0972be0777e599a3dbca710fb0a595c39560b69",
    )

    PARSERBENCH = Path(SRC_DIR / "parserbench")

    REGEX = Repo(
        name="regex",
        url="https://github.com/rust-lang/regex",
        version="bcbe40342628b15ab2543d386c745f7f0811b791",
    )

    CACTUS = Repo(
        name="cactus",
        url="https://github.com/softdevteam/cactus",
        version="8d34c207e1479cecf0b9b2f7beb1a0c22c8949ad",
    )

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        for repo in JAVA_SRCS:
            repo.fetch(c)

        self.GRMTOOLS.fetch(c)
        self.REGEX.fetch(c)
        self.CACTUS.fetch(c)

        cargo_build(
            c,
            ERRORGEN,
            BUILD_DIR / "extra/errorgen",
            BIN_DIR / "extra/errorgen",
            "errorgen",
            "errorgen",
        )

        for repo in JAVA_SRCS:
            if has_unstaged_changes(c, repo.src):
                logging.info(
                    f"{repo.name} sources have already had parser-errors introduced. Skipping..."
                )
                continue
            with c.cd(repo.src):
                c.run(
                    str(BIN_DIR / "extra" / "errorgen" / "errorgen") + " " + repo.name
                )

        patch_repo(c, self.GRMTOOLS.src, PATCH_DIR / "grmtools.diff")

        gc_cfgs = [
            "default",
            "premopt-opt",
            "premopt-none",
            "premopt-naive",
            "elision-naive",
            "elision-none",
            "default-metrics",
            "premopt-opt-metrics",
            "premopt-none-metrics",
            "premopt-naive-metrics",
            "elision-naive-metrics",
            "elision-none-metrics",
        ]
        if any(profile in s for s in gc_cfgs):
            patch_repo(c, self.REGEX.src, PATCH_DIR / "regex.gc.diff")
            patch_repo(c, self.CACTUS.src, PATCH_DIR / "cactus.gc.diff")
        else:
            patch = profile.split("-")[-1]
            patch_repo(c, self.REGEX.src, PATCH_DIR / f"regex.{patch}.diff")
            patch_repo(c, self.CACTUS.src, PATCH_DIR / f"cactus.{patch}.diff")

        cargo_build(
            c,
            self.PARSERBENCH,
            target_dir,
            install_dir,
            build_artefact="parserbench",
            bin=bench_cfg_bin,
            env=env,
        )


class BinaryTrees(BenchmarkSuite):
    BINARY_TREES = Path(SRC_DIR / "binary_trees")

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        if profile == "baseline":
            profile = "arc"
        elif profile not in {"gc", "typed-arena", "rust-gc", "arc"}:
            profile = "gc"
        cargo_build(
            c,
            self.BINARY_TREES,
            target_dir,
            install_dir,
            profile,
            bench_cfg_bin,
            env,
        )


class RegexRedux(BenchmarkSuite):
    REGEX_REDUX = Path(SRC_DIR / "regex_redux")
    FASTA = Path(SRC_DIR / "fasta")

    REGEX = Repo(
        name="regex",
        url="https://github.com/rust-lang/regex",
        version="bcbe40342628b15ab2543d386c745f7f0811b791",
    )

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        self.REGEX.fetch(c)

        cargo_build(
            c,
            self.FASTA,
            BUILD_DIR / "extra/fasta",
            BIN_DIR / "extra/fasta",
            "fasta",
            "fasta",
        )

        redux_input = BIN_DIR / "extra" / "fasta" / "redux_input.txt"
        if not redux_input.exists():
            with c.cd(str(redux_input.parent)):
                c.run(f"{str(redux_input.parent / 'fasta')} 2500000 > redux_input.txt")

        patch_repo(c, self.REGEX.src, PATCH_DIR / f"regex.{profile}.diff")

        cargo_build(
            c,
            self.REGEX_REDUX,
            target_dir,
            install_dir,
            "regex_redux",
            bench_cfg_bin,
            env,
        )

        target_path = install_dir / "redux_input.txt"
        if not target_path.exists():
            shutil.copy2(redux_input.absolute(), target_path)
            logging.info(f"Copied {redux_input} -> {target_path}")
        else:
            logging.warning(f"{target_path} already exists. Skipping...")


def install_linux(c):
    LINUX.fetch(c)
    if not (LINUX.src / ".config").exists():
        with c.cd(str(LINUX.src)):
            c.run("make defconfig")
            c.run("make -j100")


class RipGrep(BenchmarkSuite):
    RIPGREP = Repo(
        name="ripgrep",
        url="https://github.com/burntsushi/ripgrep",
        version="de4baa10024f2cb62d438596274b9b710e01c59b",
    )

    @property
    def cmd_args(self):
        return f"-j1 $(cat {str(Path('extra/ripgrep_args').resolve())}/%(benchmark)s) {str(LINUX.src)}"

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        self.RIPGREP.fetch(c)
        install_linux(c)

        patch_repo(c, self.RIPGREP.src, PATCH_DIR / f"ripgrep.{profile}.diff")

        cargo_build(
            c,
            self.RIPGREP.src,
            target_dir,
            install_dir,
            "rg",
            bench_cfg_bin,
            env,
        )


SOMRS = Repo(
    name="som-rs",
    url="https://github.com/Hirevo/som-rs",
    version="35b780cbee765cca24201fe063d3f1055ec7f608",
    recursive=True,
)


class SomrsAST(BenchmarkSuite):
    BINARY_NAME = "som-interpreter-ast"

    @property
    def cmd_args(self):
        return (
            f"-c {SOMRS.src}/core-lib/Smalltalk "
            f"{SOMRS.src}/core-lib/Examples/Benchmarks "
            f"{SOMRS.src}/core-lib/Examples/Benchmarks/Richards "
            f"{SOMRS.src}/core-lib/Examples/Benchmarks/DeltaBlue "
            f"{SOMRS.src}/core-lib/Examples/Benchmarks/NBody "
            f"{SOMRS.src}/core-lib/Examples/Benchmarks/Json "
            f"{SOMRS.src}/core-lib/Examples/Benchmarks/GraphSearch "
            f"{SOMRS.src}/core-lib/Examples/Benchmarks/LanguageFeatures "
            f"-- BenchmarkHarness %(benchmark)s %(iterations)s"
        )

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        SOMRS.fetch(c)

        patch_repo(c, SOMRS.src, PATCH_DIR / f"som-rs.{profile}.diff")

        cargo_build(
            c,
            SOMRS.src,
            target_dir,
            install_dir,
            self.BINARY_NAME,
            bench_cfg_bin,
            env,
        )


class SomrsBC(SomrsAST):
    BINARY_NAME = "som-interpreter-bc"


class FdFind(BenchmarkSuite):
    FD = Repo(
        name="fd",
        url="https://github.com/sharkdp/fd",
        version="a4fdad6ff781b5b496c837fde24001b0e46973d6",
    )
    REGEX = Repo(
        name="regex",
        url="https://github.com/rust-lang/regex",
        version="bcbe40342628b15ab2543d386c745f7f0811b791",
    )

    @property
    def cmd_args(self):
        return "-j1"

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        LINUX.fetch(c)
        self.FD.fetch(c)
        self.REGEX.fetch(c)

        patch_repo(c, self.FD.src, PATCH_DIR / f"fd.{profile}.diff")
        patch_repo(c, self.REGEX.src, PATCH_DIR / f"regex.{profile}.diff")

        cargo_build(
            c,
            self.FD.src,
            target_dir,
            install_dir,
            "fd",
            bench_cfg_bin,
            env,
        )


class Alacritty(BenchmarkSuite):
    ALACRITTY = Repo(
        name="alacritty",
        url="https://github.com/alacritty/alacritty.git",
        version="1063706f8e8a84139e5d2b464a4978e9d840ea17",
    )

    VTE_BENCH = Repo(
        name="vtebench",
        url="https://github.com/alacritty/vtebench.git",
        version="c75155bfc252227c0efc101c1971df3e327c71c4",
    )

    @property
    def cmd_args(self):
        return (
            f"-e bash -c \"[ ! -f {self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'setup'} ] || "
            f"{self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'setup'} && "
            f"{self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'benchmark'}\""
        )

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        self.VTE_BENCH.fetch(c)
        self.ALACRITTY.fetch(c)

        patch_repo(c, self.ALACRITTY.src, PATCH_DIR / f"alacritty.{profile}.diff")

        cargo_build(
            c,
            self.ALACRITTY.src,
            target_dir,
            install_dir,
            "alacritty",
            bench_cfg_bin,
            env,
        )


SOM_BENCHMARKS = (
    Benchmark("Richards", extra_args=1),
    Benchmark("DeltaBlue", extra_args=400),
    Benchmark("NBody", extra_args=1000),
    Benchmark("JsonSmall", extra_args=7),
    Benchmark("GraphSearch", extra_args=7),
    Benchmark("PageRank", extra_args=50),
    Benchmark("Fannkuch", extra_args=7),
    Benchmark("Fibonacci", extra_args="10"),
    Benchmark("Dispatch", extra_args=10),
    Benchmark("Bounce", extra_args="10"),
    Benchmark("Loop", extra_args=10),
    Benchmark("Permute", extra_args="10"),
    Benchmark("Queens", extra_args="10"),
    Benchmark("List", extra_args="5"),
    Benchmark("Recurse", extra_args="10"),
    Benchmark("Storage", extra_args=10),
    Benchmark("Sieve", extra_args=10),
    Benchmark("BubbleSort", extra_args="10"),
    Benchmark("QuickSort", extra_args=20),
    Benchmark("Sum", extra_args=10),
    Benchmark("Towers", extra_args="3"),
    Benchmark("TreeSort", extra_args="3"),
    Benchmark("IntegerLoop", extra_args=5),
    Benchmark("FieldLoop", extra_args=5),
    Benchmark("WhileLoop", extra_args=20),
    Benchmark("Mandelbrot", extra_args=50),
)


@dataclass(frozen=True)
class Experiment:
    profiles: Tuple[ExperimentProfile]
    measurement: Measurement
    suite: "BenchmarkSuite"
    pexecs: int = 0

    @property
    def gauge_adapter(self) -> str:
        return {"AlloyAdapter": "alloy_adapter.py"}

    def run(self, c, pexecs, config):
        self.results_dir.mkdir(parents=True, exist_ok=True)

        rebench_cmd = [
            str(REBENCH_EXEC),
            "-R",
            "-D",
            "--invocations",
            str(pexecs),
            "--iterations",
            "1",
            str(config),
            self.name,
        ]

        if self.suite.name == "alacritty":
            with Xvfb(display="99"):
                c.run(" ".join(rebench_cmd), warn=True, pty=True)
        else:
            c.run(" ".join(rebench_cmd), warn=True, pty=True)

    def configurations(self, only_installed=False, only_missing=False):
        if only_installed and only_missing:
            raise ValueError("Can't select both only_installed and only_missing")

        identical = [
            GCVS.GC,
            PremOpt.OPT,
            Elision.OPT,
            HeapSize.DEFAULT,
        ]
        executors = set()

        for p in self.profiles:
            name = "default" if p in identical else p.full
            is_metrics = self.measurement == Measurement.METRICS
            alloy = Alloy(p, metrics=is_metrics)
            executors.add(Executor(self, self.suite, self.measurement, name, alloy))

        if only_installed:
            executors = [self for self in executors if self.installed]
        elif only_missing:
            executors = [self for self in executors if not self.installed]
        return executors

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.suite.name / self.measurement.value

    @property
    def results(self) -> [Path]:
        if self.measurement == Measurement.PERF:
            return {self.results_dir.parent / Measurement.PERF.value / REBENCH_DATA}
        else:
            return set(
                f
                for f in self.results_dir.rglob("*")
                if f.name != REBENCH_DATA
                and not f.is_dir()
                and (f.parent.name == "default" or self.experiment in f.parent.name)
            )

    @property
    def experiment(self):
        expected = self.profiles[0].experiment

        assert all([x.experiment == expected for x in self.profiles])
        return self.profiles[0].experiment

    @property
    def profile_class(self):
        return self.profiles[0].__class__

    @property
    def name(self) -> str:
        return f"{self.suite.name}-{self.experiment}-{self.measurement.value}"

    @property
    def build_steps(self) -> int:
        return sum([cfg.steps for cfg in self.configurations(only_missing=True)])


@dataclass(frozen=True)
class Executor:
    experiment: "Experiment"
    suite: "BenchmarkSuite"
    metric: "Measurement"
    id: str
    alloy: Alloy

    @property
    def name(self):
        is_metrics = self.metric == Measurement.METRICS
        base = f"{self.suite.name}-{self.id}"
        return f"{base}-metrics" if is_metrics else base

    @property
    def rebench_name(self):
        is_perf = self.metric == Measurement.PERF
        base = f"{self.suite.name}-{self.id}"
        return base if is_perf else f"{base}-{self.metric.value}"

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    @property
    def install_prefix(self) -> Path:
        return BIN_DIR / "benchmarks" / self.suite.name

    @property
    def path(self) -> Path:
        return self.install_prefix / self.name

    @property
    def bin(self) -> str:
        return str(self.name)

    @property
    def libgc(self) -> Path:
        return self.alloy.install_prefix / "lib" / "libgc.so"

    @property
    def run_env(self):
        env = {
            "DISPLAY": ":99",
            "LD_PRELOAD": str(self.libgc),
        }
        env["RESULTS_DIR"] = str(self.results_dir)
        env["HT_PATH"] = str(HEAPTRACK.path)
        if self.metric == Measurement.HEAPTRACK:
            env["USE_HT"] = "true"
        if self.metric == Measurement.METRICS:
            env["USE_MT"] = "true"

        if self.id == "heapsize-s":
            env["GC_INITIAL_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-s"]
            env["GC_MAXIMUM_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-s"]
        elif self.id == "heapsize-l":
            env["GC_INITIAL_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-l"]
            env["GC_MAXIMUM_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-l"]
        elif self.id == "heapsize-xl":
            env["GC_INITIAL_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-xl"]
            env["GC_MAXIMUM_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-xl"]
        elif self.id == "heapsize-xxl":
            env["GC_INITIAL_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-xxl"]
            env["GC_MAXIMUM_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-xxl"]

        return env

    @property
    def results_dir(self) -> Path:
        dir = self.experiment.results_dir / self.id
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    @property
    def build_dir(self) -> Path:
        return BUILD_DIR / "benchmarks" / self.suite.name / self.name

    @property
    def installed(self) -> bool:
        return self.path.exists()

    @property
    def patch_suffix(self) -> Optional[str]:
        if self.alloy.profile in [
            GCVS.GC,
            PremOpt.NAIVE,
            PremOpt.NONE,
            PremOpt.OPT,
            Elision.OPT,
            Elision.NAIVE,
            HeapSize.DEFAULT,
            HeapSize.XL,
            HeapSize.L,
            HeapSize.S,
        ]:
            return "gc"
        else:
            return self.alloy.profile.value

    @property
    def env(self):
        return {
            "RUSTC": str(self.alloy.path),
            "RUSTFLAGS": f"-L {(self.libgc.parent)}",
            "LD_LIBRARY_PATH": str(self.libgc.parent),
        }

    def build(self, c=None):
        env = self.env.copy()
        env.update({"GC_DONT_GC": "true"})
        self.suite.build(
            c,
            self.build_dir,
            self.install_prefix,
            self.name,
            self.patch_suffix,
            env=env,
        )


@dataclass
class Experiments:
    pexecs: int
    experiments: List[Experiment]
    _config: Optional[Path] = None

    def filter(self, *, suites=None, experiments=None, measurements=None):
        if suites is not None:
            self.experiments = [e for e in self.experiments if e.suite.name in suites]
        if experiments is not None:
            self.experiments = [
                e for e in self.experiments if e.experiment in experiments
            ]
        if measurements is not None:
            self.experiments = [
                e for e in self.experiments if e.measurement.value in measurements
            ]
        return self

    def run(self, c, pexecs):
        for e in self.experiments:
            e.run(c, pexecs, self.config)

    def remove(self):
        for e in self.experiments:
            e.results_dir.unlink(missing_ok=True)

    def alloy_variants(self, only_installed=False, only_missing=False):
        l = [cfg.alloy for cfg in self.configurations()]
        if only_installed:
            l = [a for a in l if a.installed]
        elif only_missing:
            l = [a for a in l if not a.installed]
        return list({a.name: a for a in l}.values())

    @property
    def build_steps(self):
        return sum(cfg.build_steps for cfg in self.configurations(only_missing=True))

    @property
    def total_iters(self):
        unique = sum(len(cfg.suite.benchmarks) for cfg in self.configurations())
        print(self.pexecs)
        return unique * self.pexecs

    @property
    def config(self) -> Path:
        if self._config and self._config.exists():
            return self._config

        exp_part = {}
        exec_part = {}
        bm_part = {}

        executors = self.configurations()
        for cfg in executors:
            exec_part[cfg.rebench_name] = {
                "path": str(cfg.install_prefix),
                "executable": cfg.bin,
                "env": cfg.run_env,
            }

        for e in self.experiments:
            exp_part[e.name] = {
                "suites": [e.suite.name],
                "executions": [
                    cfg.rebench_name for cfg in executors if cfg.experiment == e
                ],
                "data_file": str(e.results_dir / REBENCH_DATA),
            }

            bm_part[e.suite.name] = {
                "gauge_adapter": e.gauge_adapter,
                "command": e.suite.args,
                "benchmarks": [
                    (
                        bench.name
                        if bench.extra_args is None
                        else {
                            bench.name: {
                                "extra_args": DoubleQuotedStr(bench.extra_args)
                            }
                        }
                    )
                    for bench in e.suite.benchmarks
                ],
            }

        config = RebenchConfig(
            experiments=exp_part,
            executors=exec_part,
            benchmark_suites=bm_part,
        )

        config.write_to_file(REBENCH_CONF)
        self._config = REBENCH_CONF
        return self._config

    def configurations(
        self, only_installed=False, only_missing=False
    ) -> List["Executor"]:
        if only_installed and only_missing:
            raise ValueError("Can't select both only_installed and only_missing")

        executors = set()
        for e in self.experiments:
            executors.update(e.configurations(only_installed, only_missing))
        return executors

    @classmethod
    def all(cls, pexecs: int = 30) -> "Experiments":
        profiles = [GCVS, PremOpt, Elision, HeapSize]
        return cls(pexecs, [e for p in profiles for e in p.experiments(pexecs)])

    def results(self, measurement=None):
        results = set()
        for e in self.experiments:
            if e.measurement == measurement:
                results.update(e.results)
        return sorted(results)


@dataclass
class RebenchConfig:
    experiments: Dict[str, Any]
    executors: Dict[str, Any]
    benchmark_suites: Dict[str, Any]

    @property
    def runs(self):
        return {"max_invocation_time": 360}

    @property
    def gauge_adapter(self) -> str:
        return {"AlloyAdapter": "alloy_adapter.py"}

    def write_to_file(self, file_path: Path):
        """Write the configuration to a YAML file."""
        header = "# -*- mode: yaml -*-\n"
        yaml_content = yaml.dump(
            asdict(self),
            explicit_start=False,
            sort_keys=False,
            default_flow_style=False,
            width=1000,
            allow_unicode=True,
        )

        file_path.write_text(header + yaml_content)


ALLOY = Artefact(
    repo=Repo(
        name="alloy",
        url="https://github.com/softdevteam/alloy",
        version="master",
    ),
)

HEAPTRACK = Heaptrack(
    repo=Repo(
        name="heaptrack",
        url="https://github.com/kde/heaptrack",
        version="master",
    ),
)

LINUX = Repo(
    name="linux",
    url="https://github.com/BurntSushi/linux",
    version="master",
    shallow_clone=True,
    post_checkout=(("make", "defconfig"), ("make", "-j100")),
)
