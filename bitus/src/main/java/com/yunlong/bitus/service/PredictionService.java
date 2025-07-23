package com.yunlong.bitus.service;

import ai.onnxruntime.*;
import com.yunlong.bitus.model.InputSequence;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.LongBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

@Service
public class PredictionService {

    private final OrtEnvironment env;
    private final OrtSession session;

    public PredictionService() throws OrtException, IOException {
        env = OrtEnvironment.getEnvironment();

        // Load ONNX model from classpath as InputStream
        try (var modelStream = new ClassPathResource("models/behavior_rnn.onnx").getInputStream()) {
            byte[] modelBytes = modelStream.readAllBytes();
            session = env.createSession(modelBytes, new OrtSession.SessionOptions());
        }

        // Path modelPath = new
        // ClassPathResource("models/behavior_rnn.onnx").getFile().toPath();
        // session = env.createSession(Files.readAllBytes(modelPath), new
        // OrtSession.SessionOptions());
    }

    public int predictNextEvent(InputSequence input) throws OrtException {
        List<InputSequence.EventStep> seq = input.getSequence();
        int seqLen = seq.size();

        // Prepare individual arrays
        long[] eventTypes = new long[seqLen];
        float[] timeDiffs = new float[seqLen];
        long[] itemIds = new long[seqLen];

        for (int i = 0; i < seqLen; i++) {
            InputSequence.EventStep step = seq.get(i);
            eventTypes[i] = step.getEventType();
            timeDiffs[i] = step.getTimeDelta();
            itemIds[i] = step.getItemId();
        }

        // Create 2D tensors with shape [1, seq_len]
        OnnxTensor eventTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(eventTypes), new long[] { 1, seqLen });
        OnnxTensor timeTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(timeDiffs), new long[] { 1, seqLen });
        OnnxTensor itemTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(itemIds), new long[] { 1, seqLen });

        // Prepare input map
        Map<String, OnnxTensor> inputMap = new HashMap<>();
        inputMap.put("event_types", eventTensor);
        inputMap.put("time_diffs", timeTensor);
        inputMap.put("item_ids", itemTensor);

        // Run inference
        OrtSession.Result result = session.run(inputMap);
        float[][] output = (float[][]) result.get(0).getValue();

        return argmax(output[0]);
    }

    private int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best])
                best = i;
        }
        return best;
    }
}
