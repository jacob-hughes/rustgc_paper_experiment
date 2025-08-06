import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple

import toml

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
            c.run(f"make -j install")

    @property
    def path(self) -> Path:
        return self.install_prefix / "bin" / "heaptrack"


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
