# LibND4J

This is not the [orignal](https://github.com/deeplearning4j/libnd4j) repository.

This repo is forked for raspberry pi support. Here is the method to follow :

		1. {In build machine machine}compile libnd4j
				git clone https://github.com/dschowta/libnd4j.git
			- For cross compilation use this link:http://stackoverflow.com/questions/19162072/installing-raspberry-pi-cross-compiler
			- Use the 4.9 version of gcc (raspberrypi/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-g++)
		3.{In build machine}  build using "./buildnativeoperations.sh -o rp2"
		4. {In build machine} Install maven 3.3.9 (3.0.X does not work)
		5. {In build machine} followed instructions mentioned in https://deeplearning4j.org/buildinglocally :
				export LIBND4J_HOME=<pathTond4JNI>
				
				# build and install nd4j to maven locally (using the forked nd4j specifically changed for raspberry pi)
				git clone https://github.com/dschowta/nd4j.git
				cd nd4j
				mvn clean  install -DskipTests -Dplatform=linux-arm -Dmaven.javadoc.skip=true -P linux,arm -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests
		6. {In build machine } Copy following jars from the build machine :
			mkdir myjarFolder &&  \
			cp nd4j-backends/nd4j-backend-impls/nd4j-native/target/nd4j-native-0.6.1-pi.jar\
			nd4j-backends/nd4j-backend-impls/nd4j-native/target/nd4j-native-0.6.1-pi-linux-arm.jar \
			nd4j-backends/nd4j-backend-impls/nd4j-native-platform/target/nd4j-native-platform-0.6.1-pi.jar\
			nd4j-backends/nd4j-api-parent/nd4j-api/target/nd4j-api-0.6.1-pi.jar\
			nd4j-backends/nd4j-api-parent/nd4j-native-api/target/nd4j-native-api-0.6.1-pi.jar \
			myjarFolder/
		7. {In build machine }install these jars in maven using :
			mvn install:install-file -Dfile=<path to myjarFolder>\nd4j-native-0.6.1-pi.jar -DgroupId=org.nd4j -DartifactId=nd4j-native -Dversion=0.6.1-pi -Dpackaging=jar -DgeneratePom=true
			mvn install:install-file -Dfile=<path to myjarFolder>\nd4j-native-0.6.1-pi-linux-arm.jar -DgroupId=org.nd4j -DartifactId=nd4j-native -Dversion=0.6.1-pi -Dpackaging=jar -DgeneratePom=true -Dclassifier=linux-arm
			mvn install:install-file -Dfile=<path to myjarFolder>\nd4j-api-0.6.1-pi.jar -DgroupId=org.nd4j -DartifactId=nd4j-api -Dversion=0.6.1-pi -Dpackaging=jar -DgeneratePom=true
			mvn install:install-file -Dfile=<path to myjarFolder>\nd4j-native-platform-0.6.1-pi.jar -DgroupId=org.nd4j -DartifactId=nd4j-native-platform -Dversion=0.6.1-pi -Dpackaging=jar -DgeneratePom=true
			mvn install:install-file -Dfile=<path to myjarFolder>\nd4j-native-api-0.6.1-pi.jar -DgroupId=org.nd4j -DartifactId=nd4j-native-api -Dversion=0.6.1-pi -Dpackaging=jar -DgeneratePom=true
			
		8. {In build machine } build the source of dependant appllication with above (step 7) dependencies.
		9. {In raspbian }copy the generated jar of dependant application to raspberry
		10. {In raspbian }download the libraries inside the folders of build machine nd4j-backends/nd4j-backend-impls/nd4j-native/target/classes/org/nd4j/nativeblas/linux-arm/  to a permanent folder (if possible to a system folder)
		11.{In raspbian }export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<the path to libraryy>
		12. {In raspbian }java -jar myjar.jar