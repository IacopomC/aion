"""Setup for aion_python — auto-builds the C++ pybind11 extension via CMake.

Mirrors the pattern used by Hydra-based catkin packages so a single
`pip install -e .` in this directory (a) installs build deps from
pyproject.toml, (b) runs cmake on CMakeLists.txt, (c) drops the
resulting _aion_standalone.so into the wheel staging dir (or, for
editable installs, into aion_python/ inside the source tree).

Honours DEBUG=1, CMAKE_ARGS=..., and CMAKE_BUILD_PARALLEL_LEVEL=... env
vars — same as a standard CMake-extension setup.py.
"""

import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "RelWithDebInfo"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [a for a in os.environ["CMAKE_ARGS"].split() if a]

        build_args = []
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{multiprocessing.cpu_count()}"]

        if os.path.exists(self.build_temp):
            shutil.rmtree(self.build_temp)
        os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


setup(
    name="aion_python",
    version="0.1.0",
    packages=find_packages(),
    package_data={"aion_python": ["*.so", "*.pyd"]},
    python_requires=">=3.8",
    ext_modules=[CMakeExtension("aion_python._aion_standalone", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
)
