import sys
from cx_Freeze import setup, Executable
# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["FileDialog","os","nltk","matplotlib","scipy","numpy","sklearn","htmlentitydefs"], "excludes": ["mpl_toolkits","tkinter"]}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None

setup(  name = "guifoo",
        version = "0.1",
        description = "My GUI application!",
        options = {"build_exe": build_exe_options},
        executables = [Executable("main.py", base=base)])