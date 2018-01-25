set -e
mkdir relational_neural_gas
cp -r gpl-3.0.md gpl_license_header.txt javadoc pom.xml README.md rng-1.0.0.jar relational_neural_gas/.
zip -r rng-1.0.0.zip relational_neural_gas/*
rm -rf relational_neural_gas
