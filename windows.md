# Building on Windows
*All of these instructions assume you are on a 64-bit system*

libnd4j depends on some Unix utilities for compilation. So in order to compile it you will need to install  [Msys2](https://msys2.github.io/).

After you have setup Msys2 by following [their instructions](https://msys2.github.io/), you will have to install some additional development packages. Start the msys2 shell and setup the dev environment with:

    pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-extra-cmake-modules make pkg-config grep sed gzip tar mingw64/mingw-w64-x86_64-openblas

This will install the needed dependencies for use in the msys2 shell.

You will also need to setup your PATH environment variable to include `C:\msys64\mingw64\bin` (or where ever you have decided to install msys2). If you have IntelliJ (or another IDE) open, you will have to restart it before this change takes effect for applications started through them. If you don't, you probably will see a "Can't find dependent libraries" error.

For cpu, we recommend openblas. We will be adding instructions for mkl and other cpu implementations later.

Send us a pull request or [file an issue](https://github.com/deeplearning4j/libnd4j/issues) if you have something in particular you are looking for.

## Building libnd4j

libnd4j and nd4j go hand in hand, and libnd4j is required for two out of the three currently supported backends (nd4j-native and nd4j-cuda). For this reason they should always be rebuild together.

### Building the CPU Backend

Now clone this repository, and in that directory run the following to build the dll for the cpu backend:

    bash ./buildnativeoperations.sh blas cpu

### Building the CUDA Backend

The CUDA Backend has some additional requirements before it can be built:

* [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
* [Visual Studio 2012 or 2013](https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx) (Please note: Visual Studio 2015 is *NOT SUPPORTED* by CUDA 7.5 and below)

In order to build the CUDA backend you will have to setup some more environment variables first, by calling `vcvars64.bat`.
But first, set the system environment variable `SET_FULL_PATH` to `true`, so all of the variables that `vcvars64.bat` sets up, are passed to the mingw shell.

1. Inside a normal cmd.exe command prompt, run `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\vcvars64.bat`
2. Run `c:\msys64\mingw64_shell.bat` inside that
3. Change to your libnd4j folder
4. `bash buildnativeoperations.sh blas cuda` (note the `bash` at the beginning, for some reason it doesn't work to directly start the script)

This builds the CUDA nd4j.dll.


## Building nd4j

While still in the `libnd4j` folder, run:

    export LIBND4J_HOME=`pwd`

    or

    If you want to use Control Panel for that: if you have libnd4j path looking like 'c:\Users\username\libnd4j' set LIBND4J_HOME to '/Users/username/libnd4j'

Now leave the libnd4j directory and clone the [nd4j repository](https://github.com/deeplearning4j/nd4j). Run the following to compile nd4j with support for both the native cpu backend as well as the cuda backend:

    mvn clean install -DskipTests -Dmaven.javadoc.skip=true

If you don't want the cuda backend, e.g. because you didn't or can't build it, you can skip it:

    mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!org.nd4j:nd4j-cuda-7.5'

Please notice the single quotes around the last parameter, if you leave them out or use double quotes you will get an error about `event not found` from your shell.


## Using the Native Backend

In order to use your new shiny backends you will have to switch your application to use the version of ND4J that you just compiled and to use the native backend instead of x86.

For this you change the version of all your ND4J dependencies to "0.4-rc3.9-SNAPSHOT".


### CPU Backend

Exchange nd4j-x86 for nd4j-native like that:

    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native</artifactId>
        <version>0.4-rc3.9-SNAPSHOT</version>
    </dependency>

### CUDA Backend

Exchange nd4j-x86 for nd4j-cuda-7.5 like that:

    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-cuda-7.5</artifactId>
        <version>0.4-rc3.9-SNAPSHOT</version>
    </dependency>

    
## Troubleshooting

### When I start my application, I still see a "Can't find dependent libraries" error
If your application continues to run, then you are just seeing an artefact of one way we try to load the native library, but your application should run just fine.

If your application crashes (and you see that error more than once) then you probably have a problem with your PATH environment variable. Please make sure that you have your msys2 bin directory on the PATH and that you restarted your IDE. Maybe even try to restart the system.

### I'm having trouble downloading or updating packages using pacman
There are a number of things that can potentially go wrong.
First, try updating packman using the following commands:

	pacman -Syy
	pacman -Syu
	pacman -S pacman-mirrors

Note that you might need to restart the msys2 shell between/after these steps.

One user has reported issues downloading packages using the default downloader (timeouts and "error: failed retrieving file" messages). If you are experiencing these issues, it may help to switch to using the wget downloader. To do this, install wget using

	pacman -S wget

then uncomment (remove the # symbol) the following line in the /etc/pacman.conf configuration file:

	XferCommand = /usr/bin/wget --passive-ftp -c -O %o %u

### "buildnativeoperations.sh blas cpu" can't find BLAS libraries

First, make sure you have BLAS libraries intalled. Typically, this involves building OpenBLAS by downloading OpenBLAS and running the commands 'make', 'make install' in msys2.

Running the buildnativeoperations.sh script in the MinGW-w64 Win64 Shell instead of the standard msys2 shell may resolve this issue.

### I'm getting other errors not listed here

Depending on how your build environment and PATH environment variable is set up, you might experience some other issues.
Some situations that may be problematic include:

- Having older (or multiple) MinGW installs on your PATH (check: type "where c++" or "where gcc" into msys2)
- Having older (or multiple) cmake installs on your PATH (check: "where cmake" and "cmake --version")
- Having multiple BLAS libraries on your PATH (check: "where libopenblas.dll", "where libblas.dll" and "where liblapack.dll")

### I'm getting `jniNativeOps.dll: Can't find dependent libraries` errors

This is usually due to an incorrectly setup PATH (see "I'm getting other errors not listed here"). As the PATH using the msys2 shell is a little bit different then for other applications, you can check that the PATH is really the problem by running the following test program:

    public class App {
        public static void main(String[] args){
        	System.loadLibrary("libopenblas.dll");
        }
    }
    
If this also crashes with the `Can't find dependent libraries` error, then you have to setup your PATH correctly (see the introduction to this document).


**Note**: Another possible cause of "...jniNativeOps.dll: Can't find dependent libraries" seems to be having an old or incompatible version of libstc++-6.dll on your PATH. You want this file to be pulled in from mingw via you PATH environment variable.
To check your PATH/environment, run "where libstdc++-6.dll" and "where libgcc_s_seh-1.dll"; these should list the msys/mingw directories (and/or list them first, if there are other copies on the PATH).


Finally, using dumpbin (from Visual Studio) can help to show required dependencies for jniNativeOps.dll:

	dumpbin /dependents [path to jniNativeOps.dll]
