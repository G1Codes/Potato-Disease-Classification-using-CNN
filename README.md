# Potato Disease Classification using CNN  

## 🖼️ Sample Images
| Early Blight | Late Blight | Healthy |
|--------------|-------------|---------|
| <img src="https://cropserve.co.zw/wp-content/uploads/2019/03/early-blight-potato_open-source.jpg" width="150"> | <img src="https://spudsmart.com/wp-content/uploads/2017/05/potato_late-blight_08_zoom-Photo-OMAFRA-900x580.jpeg" width="150"> | <img src="https://img.freepik.com/premium-photo/fresh-green-leaf-potato-plant-isolated_696657-22650.jpg" width="150"> |

## **📌 Overview**  
This project uses a **Convolutional Neural Network (CNN)** to classify potato leaf diseases (**Early Blight**, **Late Blight**, **Healthy**) with **99.5% test accuracy**. The model helps farmers detect diseases early by simply uploading an image of a potato leaf.  

🔗 **Dataset:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)  

---

## **🚀 Features**  
✔ **High Accuracy (99.5%)** - CNN model trained on 2,152 images  
✔ **Data Augmentation** - Random flips & rotations for better generalization  
✔ **Optimized Training** - ModelCheckpoint & EarlyStopping  
✔ **Easy Inference** - Predict on new images with confidence scores  

---

## **🛠️ Tech Stack**  
- **Python** (TensorFlow, Keras, OpenCV, Matplotlib)  
- **CNN Architecture** (6 Conv2D + MaxPooling Layers)  
- **Data Preprocessing** (Resizing, Normalization, Augmentation)  

---

## **📂 Project Structure**  
```
Potato-Disease-Classification/  
├── Dataset/  
│   ├── Potato___Early_blight/  
│   ├── Potato___Late_blight/  
│   └── Potato___Healthy/  
├── Models/  
│   └── best_model.h5  
├── Notebooks/  
│   └── Potato_Disease_Classification.ipynb  
├── README.md  
└── requirements.txt  
```

---

## **🔧 Installation**  
1. **Clone the repo**  
```bash
git clone [https://github.com/yourusername/Potato-Disease-Classification.git]
cd Potato-Disease-Classification
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **Download Dataset**  
- Get the dataset from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)  
- Extract into `Dataset/` folder  

---

## **📊 Model Training**  
Run the Jupyter notebook:  
```bash
jupyter notebook Notebooks/Potato_Disease_Classification.ipynb
```

### **Key Steps:**  
1. **Data Loading**  
   - Images resized to `(256, 256)`  
   - Split into Train (80%), Val (10%), Test (10%)  

2. **Model Architecture**  
   ```python
   model = Sequential([
       Rescaling(1./255),  # Normalization
       RandomFlip("horizontal"),  # Augmentation
       Conv2D(32, (3,3), activation='relu'),
       MaxPooling2D(),
       # ... (6 Conv Layers)
       Dense(3, activation='softmax')  # 3 Classes
   ])
   ```

3. **Training**  
   - Optimizer: `Adam`  
   - Loss: `Sparse Categorical Crossentropy`  
   - Callbacks: `ModelCheckpoint`, `EarlyStopping`  

---

## **🔮 Inference on New Images**  
Load the trained model and predict:  
```python
model = tf.keras.models.load_model('Models/best_model.h5')

def predict(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    pred = model.predict(img_array)
    class_names = ['Early Blight', 'Late Blight', 'Healthy']
    return class_names[np.argmax(pred[0])], 100 * np.max(pred[0])

pred_class, confidence = predict("unseen_potato_leaf.jpg")
print(f"Predicted: {pred_class} | Confidence: {confidence:.2f}%")
```

---

## **📈 Results**  
| Metric          | Value  |
|-----------------|--------|
| **Test Accuracy** | 99.5%  |
| **Validation Accuracy** | 99.1% |
| **Training Time** (20 epochs) | ~11 mins (CPU) |

**Confusion Matrix:**  
![Confusion Matrix](https://miro.medium.com/max/1400/1*Z54JgbS4DUwWSknhDCvNTQ.png)  

---

## **📜 License**  
MIT License - Feel free to use and modify!  
