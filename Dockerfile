# Use an official OpenJDK runtime as a parent image
FROM eclipse-temurin:17-jdk-jammy

# Set the working directory inside the container
WORKDIR /app

# Copy the dataset and the source code into the container's /app directory
COPY spam_ham_dataset.csv .
COPY SpamServer.java .

# Compile the Java code inside the container
RUN javac SpamServer.java

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define the command to run your application
CMD ["java", "SpamServer"]
