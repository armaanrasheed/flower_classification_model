# 🌼 Flower Image Classification Project

## 📝 Project Overview
This project demonstrates image classification using TensorFlow and Keras to categorize flowers into five classes: `daisy`, `dandelion`, `roses`, `sunflowers`, and `tulips`. It includes dataset preparation, model training, performance evaluation, and predictions using a trained convolutional neural network (CNN).

---

## 📂 Project Structure
```
/project-folder
|-- main.ipynb   # Jupyter notebook containing the project code
```

---

## ⚙️ Dependencies
The project uses the following libraries:
- `TensorFlow`
- `NumPy`
- `Matplotlib`
- `PIL` (Python Imaging Library)

Install the dependencies via:
```bash
pip install tensorflow numpy matplotlib pillow
```

---

## 📊 Dataset
The dataset used is the **Flower Photos Dataset** from TensorFlow's example datasets:
- **Source URL:** `https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz`
- The dataset contains **3,670 images** across five categories of flowers.

---

## 🏗️ Model Architecture
The CNN model is implemented using Keras `Sequential` API:
1. **Input Layer:**
   - Image rescaling layer to normalize pixel values between 0 and 1.
2. **Convolutional Layers:**
   - Three convolutional layers with ReLU activation and `MaxPooling` for down-sampling.
3. **Dropout Layer:** Prevents overfitting.
4. **Fully Connected Layers:** Dense layers for output predictions.

---

## 🚀 How to Run
1. **Download Dataset:**
   - The dataset is automatically downloaded and extracted via TensorFlow's `keras.utils.get_file()`.
2. **Train-Test Split:**
   - 80% of images are used for training, and 20% for validation.
3. **Training:**
   - Run the training loop for the model by calling `model.fit()`.

---

## 🧪 Key Sections in the Code

### 1. **Dataset Preparation**
   The images are loaded and resized to a uniform size (180x180 pixels) and grouped into batches:
   ```python
   train_ds = tf.keras.utils.image_dataset_from_directory(
       data_dir,
       validation_split=0.2,
       subset="training",
       seed=123,
       image_size=(img_height, img_width),
       batch_size=batch_size
   )
   ```

### 2. **Data Augmentation**
   To enhance model generalization, random image transformations (flip, rotation, zoom) are applied:
   ```python
   data_augmentation = keras.Sequential([
       layers.RandomFlip("horizontal"),
       layers.RandomRotation(0.1),
       layers.RandomZoom(0.1)
   ])
   ```

### 3. **Model Definition**
   The CNN model is defined as:
   ```python
   model = Sequential([
       data_augmentation,
       layers.Rescaling(1./255),
       layers.Conv2D(16, 3, padding='same', activation='relu'),
       layers.MaxPooling2D(),
       layers.Conv2D(32, 3, padding='same', activation='relu'),
       layers.MaxPooling2D(),
       layers.Conv2D(64, 3, padding='same', activation='relu'),
       layers.MaxPooling2D(),
       layers.Dropout(0.2),
       layers.Flatten(),
       layers.Dense(128, activation='relu'),
       layers.Dense(num_classes)
   ])
   ```

### 4. **Model Training**
   The model is trained for multiple epochs using:
   ```python
   model.fit(
       train_ds,
       validation_data=val_ds,
       epochs=15
   )
   ```

### 5. **Evaluation and Visualization**
   Training and validation accuracy/loss are plotted using `Matplotlib`:
   ```python
   plt.plot(epochs_range, acc, label='Training Accuracy')
   plt.plot(epochs_range, val_acc, label='Validation Accuracy')
   ```

---

## ✅ Results
- **Initial Training Accuracy:** ~43% in the first epoch.
- **Final Validation Accuracy:** ~72% after 15 epochs.

---

## 🔮 Sample Prediction
A sample prediction is made for a new sunflower image:
```python
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(f'This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f}% confidence.')
```
**Example Output:**
```
This image most likely belongs to sunflowers with a 99.52% confidence.
```

---

## 📈 Improvements
- Implement early stopping to avoid overfitting.
- Tune hyperparameters (learning rate, filter size, batch size).
- Experiment with transfer learning using pre-trained models.

---

