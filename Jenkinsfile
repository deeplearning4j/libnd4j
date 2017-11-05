#!groovy

env.PLATFORM_NAME = 'android-x86'
node("${PLATFORM_NAME}") {
    stage ('My new stage') {
        currentBuild.displayName = "#${currentBuild.number} ${PLATFORM_NAME}"
        ws(WORKSPACE + "_" + PLATFORM_NAME) {
            step([$class: 'WsCleanup'])

            checkout scm
        }
    }
}
