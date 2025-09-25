import tensorflow as tf
from tensorflow.keras import layers, Model

def build_unet(input_shape=(224, 224, 1)):
    inputs = layers.Input(input_shape)
    
    # -------- Encoder --------
    # Block 1
    conv1 = layers.Conv2D(64, 3, padding='same')(inputs)
    conv1 = layers.ReLU()(conv1)
    conv1 = layers.Conv2D(64, 3, padding='same')(conv1)
    conv1 = layers.ReLU()(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(0.25)(pool1)
    
    # Block 2
    conv2 = layers.Conv2D(128, 3, padding='same')(pool1)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Conv2D(128, 3, padding='same')(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(0.25)(pool2)
    
    # Block 3
    conv3 = layers.Conv2D(256, 3, padding='same')(pool2)
    conv3 = layers.ReLU()(conv3)
    conv3 = layers.Conv2D(256, 3, padding='same')(conv3)
    conv3 = layers.ReLU()(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(0.25)(pool3)
    
    # Block 4
    conv4 = layers.Conv2D(512, 3, padding='same')(pool3)
    conv4 = layers.ReLU()(conv4)
    conv4 = layers.Conv2D(512, 3, padding='same')(conv4)
    conv4 = layers.ReLU()(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(0.25)(pool4)
    
    # -------- Bottleneck --------
    conv5 = layers.Conv2D(1024, 3, padding='same')(pool4)
    conv5 = layers.ReLU()(conv5)
    conv5 = layers.Conv2D(1024, 3, padding='same')(conv5)
    conv5 = layers.ReLU()(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    
    # -------- Decoder --------
    # Block 1
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(512, 3, padding='same')(up6)
    conv6 = layers.ReLU()(conv6)
    conv6 = layers.Conv2D(512, 3, padding='same')(conv6)
    conv6 = layers.ReLU()(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Dropout(0.25)(conv6)
    
    # Block 2
    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(256, 3, padding='same')(up7)
    conv7 = layers.ReLU()(conv7)
    conv7 = layers.Conv2D(256, 3, padding='same')(conv7)
    conv7 = layers.ReLU()(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Dropout(0.25)(conv7)
    
    # Block 3
    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(128, 3, padding='same')(up8)
    conv8 = layers.ReLU()(conv8)
    conv8 = layers.Conv2D(128, 3, padding='same')(conv8)
    conv8 = layers.ReLU()(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Dropout(0.25)(conv8)
    
    # Block 4
    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(64, 3, padding='same')(up9)
    conv9 = layers.ReLU()(conv9)
    conv9 = layers.Conv2D(64, 3, padding='same')(conv9)
    conv9 = layers.ReLU()(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Dropout(0.25)(conv9)
    
    # Output Layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build and compile model
unet_model = build_unet()
unet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("U-Net model built and compiled successfully.")

from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dice Coefficient Metric
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1e-7)

# Data augmentation for segmentation
datagenSeg = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # Set to True if horizontal flipping is acceptable
    fill_mode='nearest'
)
datagenSeg.fit(images_preprocessed)

# 3-Fold Cross-Validation
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
fold_results = []
best_val_acc = -1
best_fold = -1
all_histories = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(images_preprocessed, masks_preprocessed)):
    print(f'\nFold {fold + 1}/3')

    # Split train/validation for this fold
    X_train, X_val = images_preprocessed[train_idx], images_preprocessed[val_idx]
    y_train, y_val = masks_preprocessed[train_idx], masks_preprocessed[val_idx]
    
    # Build a fresh U-Net model for each fold
    model = build_unet(input_shape=(256, 256, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', dice_coefficient]
    )
    
    # Train model
    history = model.fit(
        datagenSeg.flow(X_train, y_train, batch_size=4),
        validation_data=(X_val, y_val),
        epochs=25,
        verbose=1
    )
    
    # Evaluate on validation data
    val_loss, val_acc, val_dice = model.evaluate(X_val, y_val, verbose=0)
    fold_results.append({'fold': fold + 1, 'val_loss': val_loss, 'val_acc': val_acc, 'val_dice': val_dice})
    
    # Save model for this fold
    model.save(f'unet_model_fold_{fold + 1}.h5')
    
    # Track best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_fold = fold + 1
    
    all_histories.append(history.history)

# Final test split (15% of data)
_, X_test, _, y_test = train_test_split(
    images_preprocessed, masks_preprocessed, test_size=0.15, random_state=42
)

# Print results per fold
print("\n3-Fold cross-validation results:")
for result in fold_results:
    print(f"Fold {result['fold']}: Val Loss = {result['val_loss']:.4f}, Val Accuracy = {result['val_acc']:.4f}, Val Dice = {result['val_dice']:.4f}")

print(f"\nBest Fold: Fold {best_fold} with Val Accuracy = {best_val_acc:.4f}")

# Load best model
best_model = tf.keras.models.load_model(
    f'unet_model_fold_{best_fold}.h5',
    custom_objects={'dice_coefficient': dice_coefficient}
)
print(f"Loaded best model from unet_model_fold_{best_fold}.h5")

# Prepare average metrics across folds
min_epochs = min(len(h['accuracy']) for h in all_histories)
avg_accuracy = np.mean([h['accuracy'][:min_epochs] for h in all_histories], axis=0)
avg_val_accuracy = np.mean([h['val_accuracy'][:min_epochs] for h in all_histories], axis=0)
avg_loss = np.mean([h['loss'][:min_epochs] for h in all_histories], axis=0)
avg_val_loss = np.mean([h['val_loss'][:min_epochs] for h in all_histories], axis=0)
avg_dice = np.mean([h['dice_coefficient'][:min_epochs] for h in all_histories], axis=0)
avg_val_dice = np.mean([h['val_dice_coefficient'][:min_epochs] for h in all_histories], axis=0)

# Plot averaged metrics
plt.figure(figsize=(18, 4))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(avg_accuracy, label='Avg Train Accuracy', color='blue')
plt.plot(avg_val_accuracy, label='Avg Val Accuracy', color='orange', linestyle='--')
plt.title('Average Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(avg_loss, label='Avg Train Loss', color='blue')
plt.plot(avg_val_loss, label='Avg Val Loss', color='orange', linestyle='--')
plt.title('Average Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Dice plot
plt.subplot(1, 3, 3)
plt.plot(avg_dice, label='Avg Train Dice', color='blue')
plt.plot(avg_val_dice, label='Avg Val Dice', color='orange', linestyle='--')
plt.title('Average Dice Coefficient Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
