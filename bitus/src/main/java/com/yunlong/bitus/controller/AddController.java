package com.yunlong.bitus.controller;

import com.yunlong.bitus.model.AddInput;
import com.yunlong.bitus.service.AddService;

import ai.onnxruntime.OrtException;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class AddController {

    private final AddService addService;

    public AddController(AddService addService) {
        this.addService = addService;
    }

    @PostMapping("/add")
    public float add(@RequestBody AddInput input) throws OrtException {
        return addService.predict(input);
    }
}
