FROM eclipse-temurin:19-jdk

RUN apt-get update && apt-get install -y libstdc++6 && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY target/bitus-0.0.1-SNAPSHOT.jar /app/app.jar
COPY models/behavior_rnn.onnx /app/models/behavior_rnn.onnx

EXPOSE 443

CMD ["java", "-jar", "/app/app.jar"]
