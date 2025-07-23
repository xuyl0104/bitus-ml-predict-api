# Bitus Labs Machine Learning Engineer Project -- User Behavior Prediction

### Brief Introduction to Yunlong Xu
My name is Yunlong Xu. I'm a PhD candidate and earned my Master's Degree at Binghamton University (State University of New York). My interests lie in the field of Computer Vision, Natural Language Processing, Machine Learning. I had experience working with machine learning platform and toolkits, such as PyTorch, scikit-learn, pandas, matplotlib, etc.

My Email: yxu133@binghamton.edu or xuyl0104@gmail.com \
My Github: https://github.com/xuyl0104 \
My LinkedIn: https://www.linkedin.com/in/xuyunlong/


I was on a vacation when I was working on the project. I had limited time working on it and there were many places that could be further polished. This is a very interesting project with a dataset that I have never worked on before. Also, transformming the trained model to ONNX and build API opon it using Java is also a new experience for me. The complexity of API development is very different from using Flask or FastAPI in Python.


## 1. Machine Learning Model Training
### 1.1 Dataset
* The dataset we are using is events.csv data in the Retailrocket Recommender System Dataset.
* Features/columns in the events.csv: visitorid, timestamp, event, and itemid.
* Preprocessing
  * grouped by visitorid and sorted by timestamp, for time-series correctness;
  * transform and map event (view/addtocard/purchase) to (0/1/2); 
  * transform timestamp to time_diffs: I take into consideration the time between two consuctive events, because that is a feature that can characterize the user (no ablation experiments yet).

### 1.2 Model Selection and Architecture
* I selected RNN as the model (Reason: this dataset and the features are time-series, so RNN or GRN or even more complicated Transformer models are naturally more appropriate for this dataset and prediction task.)
* Embedding: for the three features (item_id, time_diff, event_type), I do embeddings on them, separately;
* Padding or chuncate the users' events to the length of seq_len (number of steps of RNN, set to be 50 here);
* Architecture: 
  ```
    BehaviorRNN(
      (event_embedding): Embedding(3, 32)
      (item_embedding): Embedding(150789, 32)
      (time_embedding): Linear(in_features=1, out_features=32, bias=True)
      (rnn): GRU(96, 64, batch_first=True)
      (fc): Linear(in_features=64, out_features=3, bias=True)
    )
  ```

### 1.3 Model Training Details and Settings: \
seq_len: 50 \
num_epochs: 5 (set to 5 because of very limited tranining resources, especially when I'm on a trip) \
optimizer: optim.Adam(learning_rate=0.001) \
loss function: nn.CrossEntropyLoss
early_stopping: yes
imbalance class handling: class weight when calculating loss function


* Training Result:
  ```
    Epoch 1: Train Loss=1.0608, Val Loss=1.0552, Val Acc=0.8491
    Epoch 2: Train Loss=1.0531, Val Loss=1.0697, Val Acc=0.7526
    Epoch 3: Train Loss=1.0499, Val Loss=1.0947, Val Acc=0.8412
    Epoch 4: Train Loss=1.0462, Val Loss=1.0735, Val Acc=0.8569
    Early stopping triggered after 4 epochs.
    Test Loss=1.0558, Test Acc=0.8598
  ```

* Test with Data
    * the format of the test data can be in the following format:
    ```
      new_user_seq = [
        (600000, "view", 601),               # Initial product view
        (600030, "view", 602),               # Related product
        (600060, "view", 601),               # Back to original product
        (600090, "addtocart", 601),          # Adds it to cart
        (600120, "view", 603),               # Still exploring alternatives
        (600150, "view", 601), 
        (600180, "addtocart", 603),  
      ]


      Predicted next event: view
      Probabilities: [0.44539365 0.3790203  0.17558601]
    ```

### 1.4 Model Save and Transform to ONNX
* Model was saved as PyTorch checkpoint file (behavior_rnn_full.pth) and transformed to ONNX (behavior_rnn.onnx) 
* Please place the transformed .onnx model in Spring Boot folder `bitus/src/main/resources/models`
  
  ```
  # !NOTE eval mode
  model.eval()

  # Create dummy inputs with the correct shape (batch_size=1, seq_len=50)
  dummy_event_types = torch.zeros((1, 50), dtype=torch.long).to(device)
  dummy_time_diffs = torch.zeros((1, 50), dtype=torch.float32).to(device)
  dummy_item_ids = torch.zeros((1, 50), dtype=torch.long).to(device)

  # Export to ONNX
  torch.onnx.export(
      model,
      (dummy_event_types, dummy_time_diffs, dummy_item_ids),
      "behavior_rnn.onnx",
      input_names=["event_types", "time_diffs", "item_ids"],
      output_names=["logits"],
      dynamic_axes={
          "event_types": {0: "batch_size", 1: "seq_len"},
          "time_diffs": {0: "batch_size", 1: "seq_len"},
          "item_ids": {0: "batch_size", 1: "seq_len"},
          "logits": {0: "batch_size"}
      },
      opset_version=14,
  )
  ```

## 2. API Development Using Java Spring Boot
### 2.1 Language and Framework Selection
* Language: Java
* API development framework: Spring Boot
* Build: Maven
* HTTPS: Yes
* Certificate: self-certificate (certificate in `./bitus/src/main/resources/keystore.p12`)
* API call method:  `POST /predict_behavior`
* API input: (JSON) please transfer the raw events data to the form of the following with simple Python code
  ```
  {
    "sequence": [
      {
        "eventType": 0,
        "timeDelta": 0.1,
        "itemId": 5
      },
      {
        "eventType": 0,
        "timeDelta": 0.3,
        "itemId": 7
      },
      {
        "eventType": 0,
        "timeDelta": 0.2,
        "itemId": 5
      },
      {
        "eventType": 0,
        "timeDelta": 0.5,
        "itemId": 10
      }
    ]
  }

  ```

* Nginx: not supported yet
  

## 3. Dockerize and Deployment
* Preparation
    * Build the API Java code to `bitus-0.0.1-SNAPSHOT.jar` (into `/bitus/target` folder)
      ```
        mvn clean package
      ```
    * Put the `jar` file in the folder `/bitus-docker/target`

* Dockerfile
  ```
  FROM eclipse-temurin:19-jdk

  RUN apt-get update && apt-get install -y libstdc++6 && rm -rf /var/lib/apt/lists/*


  WORKDIR /app

  COPY target/bitus-0.0.1-SNAPSHOT.jar /app/app.jar
  COPY models/behavior_rnn.onnx /app/models/behavior_rnn.onnx

  EXPOSE 443

  CMD ["java", "-jar", "/app/app.jar"]
  ```

* docker image build
  ```
  docker build -t behavior-api .
  ```

* docker run
  ```
  docker run -p 443:8443 behavior-api
  ```


## 4. Docker Hub
* The API docker image has been uploaded to Docker Hub
* You can pull and build and run the image using
  ```
  docker image pull xuyl0104/ml-predict-api
  ```

  ```
  docker run -p 443:8443 ml-predict-api
  ```

* In Postman or similar API test tools, call the API using:
  ```
  https://localhost:443/predict_behavior
  ```


## 5. GitHub Repo
4 branches
* main: README file
* model-dev: contains Jupyter Notebook and the checkpoint pth file
* api-java: all the Spring Boot API code
* docker: Dockerfile and necessary files for dockerization


