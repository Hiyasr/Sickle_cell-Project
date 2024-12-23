import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Paths to training, validation, and test directories
train_dir = r"C:\Users\hiyas\OneDrive\Desktop\Project Repo\Sickle-cell-anemia-bone-scan-images\train"
val_dir = r"C:\Users\hiyas\OneDrive\Desktop\Project Repo\Sickle-cell-anemia-bone-scan-images\validation"
test_dir = r"C:\Users\hiyas\OneDrive\Desktop\Project Repo\Sickle-cell-anemia-bone-scan-images\test"

# Function to check if directories exist
def check_directory_exists(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist. Please check the path.")
    else:
        print(f"Directory {directory} exists: True")

# Check if the directories exist
try:
    check_directory_exists(train_dir)
    check_directory_exists(val_dir)
    check_directory_exists(test_dir)
except FileNotFoundError as e:
    print(e)
    # Exit the script if directories are missing
    exit()

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Data Preparation for training, validation, and test datasets
train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_dataset = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None  # No labels in test data
)

# Data Augmentation for Training Set
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

# Caching and Prefetching for Efficiency
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Load Pre-trained Model for CNN
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model for transfer learning

# Build CNN Model
cnn_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile CNN Model
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_cb = ModelCheckpoint(
    'sickle_cell_best_cnn_model.keras', save_best_only=True, monitor='val_loss'
)
early_stopping_cb = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Train CNN Model
history = cnn_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# Fine-tuning (optional)
base_model.trainable = True  # Unfreeze some layers
for layer in base_model.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False

# Recompile CNN Model with Lower Learning Rate
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Continue Training (Fine-tuning)
history_fine_tune = cnn_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# Save Final CNN Model
cnn_model.save('sickle_cell_final_cnn_model.keras')

# Extract Features for Classical Models
train_features, train_labels = [], []
for images, labels in train_dataset:
    features = base_model(images, training=False)
    pooled_features = tf.keras.layers.GlobalAveragePooling2D()(features)
    train_features.append(pooled_features.numpy())
    train_labels.append(labels.numpy())

train_features = np.vstack(train_features)
train_labels = np.concatenate(train_labels).ravel()  # Fix for warning

val_features, val_labels = [], []
for images, labels in val_dataset:
    features = base_model(images, training=False)
    pooled_features = tf.keras.layers.GlobalAveragePooling2D()(features)
    val_features.append(pooled_features.numpy())
    val_labels.append(labels.numpy())

val_features = np.vstack(val_features)
val_labels = np.concatenate(val_labels).ravel()  # Fix for warning

# Train and Evaluate Classical Models
classical_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(kernel='linear', probability=True),
    'LDA': LDA(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

accuracies = {}
for name, model in classical_models.items():
    model.fit(train_features, train_labels)
    predictions = model.predict(val_features)
    accuracy = accuracy_score(val_labels, predictions)
    accuracies[name] = accuracy

    # Print accuracy as a percentage
    print(f'{name} Accuracy: {accuracy * 100:.2f}%')

# Plot Algorithm Accuracy Comparison
all_accuracies = {'CNN': history.history['val_accuracy'][-1]}
all_accuracies.update(accuracies)

plt.figure(figsize=(10, 6))
plt.bar(all_accuracies.keys(), all_accuracies.values(), color='skyblue')
plt.title('Algorithm Accuracy Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.show()
