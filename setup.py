import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_TESTS=OFF"
        ]
        
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Workaround for Anaconda LD_PRELOAD issue if needed
        # subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        # subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        # More robust call
        subprocess.run(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, check=True)
        subprocess.run(["cmake", "--build", "."] + build_args, cwd=self.build_temp, check=True)

setup(
    name="pybattle",
    version="0.1.0",
    packages=["pybattle"],
    package_dir={"": "."},
    ext_modules=[CMakeExtension("pybattle.pybattle_native", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
