package com.yunlong.bitus.service;

import ai.onnxruntime.*;

import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;
import com.yunlong.bitus.model.AddInput;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

@Service
public class AddService {
    private OrtEnvironment env;
    private OrtSession session;

    public AddService() throws OrtException, IOException {
        env = OrtEnvironment.getEnvironment();

        // Load ONNX model from classpath as InputStream
        try (var modelStream = new ClassPathResource("models/add_model.onnx").getInputStream()) {
            byte[] modelBytes = modelStream.readAllBytes();
            session = env.createSession(modelBytes, new OrtSession.SessionOptions());
        }
        // Path modelPath = new
        // ClassPathResource("models/add_model.onnx").getFile().toPath();
        // session = env.createSession(Files.readAllBytes(modelPath), new
        // OrtSession.SessionOptions());
    }

    public float predict(AddInput input) throws OrtException {
        float[][] inputData = { { input.getA(), input.getB() } };
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);

        Map<String, OnnxTensor> inputs = Collections.singletonMap("input", inputTensor);
        OrtSession.Result result = session.run(inputs);

        float[][] output = (float[][]) result.get(0).getValue();
        return output[0][0];
    }
}
