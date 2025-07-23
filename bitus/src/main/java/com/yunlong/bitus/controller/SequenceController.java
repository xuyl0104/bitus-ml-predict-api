package com.yunlong.bitus.controller;

import com.yunlong.bitus.model.InputSequence;
import com.yunlong.bitus.service.PredictionService;

import ai.onnxruntime.OrtException;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class SequenceController {

    private final PredictionService predictionService;

    public SequenceController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @PostMapping("/predict_behavior")
    public int predict(@RequestBody InputSequence inputSequence) {
        try {
            return predictionService.predictNextEvent(inputSequence);
        } catch (OrtException e) {
            // Handle the exception
            throw new RuntimeException("Error predicting next event", e);
        }
    }
}
