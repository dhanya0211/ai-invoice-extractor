pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                // Tells Jenkins to build the Docker image using your docker-compose file
                sh 'docker-compose build'
            }
        }
        stage('Deploy') {
            steps {
                // Tells Jenkins to restart the container
                sh 'docker-compose up -d'
            }
        }
    }
}