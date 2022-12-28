MNN for Java
-------------------

This directory contains a compiled Java jar file with SWIG-generated
interface classes for using the MNN C++ library from Java or from any
other language that runs on the JVM such as Jython or JRuby.

*** Linux Compilation

  First of all, clone the repository and checkout to feature/swig branch.
$ git clone https://github.com/xbwee1024/MNN.git
$ git checkout -b feat/swig origin/feat/swig

  Next, compile the Java bindings:
$ cd MNN
$ cmake -Bbuild -H. -G Ninja -DRUN_SWIG=ON -DJAVA_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
$ cmake --build build --config Release

  Compile and run the test program:
$ cd swig/java
$ export CLASSPATH=.:mnn.jar
$ $JAVA_HOME/bin/javac InterpreterTest.java
$ export LD_LIBRARY_PATH=../../build
$ $JAVA_HOME/bin/java InterpreterTest

*** Mac OS X Compilation

As for Linux, but use the following instructions to compile the bindings. Also, replace LD_LIBRARY_PATH with DYLD_LIBRARY_PATH.

$ cd swig/java
$ export CLASSPATH=.:mnn.jar
$ $JAVA_HOME/bin/javac InterpreterTest.java
$ export DYLD_LIBRARY_PATH=../../build
$ $JAVA_HOME/bin/java InterpreterTest
