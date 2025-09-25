# Function to build DenseNet121 model
def build_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

import cv2

# Initialize parameters
n_splits = 10  # Change to 5 or 3 for further analysis
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_results = []  # Store loss and accuracy per fold
fold_times = []
best_val_acc = -1
best_fold = -1
all_histories = []
total_start_time = time.time()

# K-Fold loop
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):
    print(f'\nFold {fold}/{n_splits}')
    start_time = time.time()
    
    # Resize images for training and validation in this fold
    X_train_fold = np.array([cv2.resize(img, (224, 224)) for img in X_train[train_idx]])
    y_train_fold = y_train[train_idx]
    X_val_fold = np.array([cv2.resize(img, (224, 224)) for img in X_train[val_idx]])
    y_val_fold = y_train[val_idx]
    
    # Build a new model for each fold
    model = build_model()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Fit generator for augmentation
    train_generator = datagenKlas.flow(X_train_fold, y_train_fold, batch_size=8)
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train_fold) // 8,
        validation_data=(X_val_fold, y_val_fold),
        epochs=50,  # Increase number of epochs
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on validation data for this fold
    val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_results.append({'fold': fold, 'val_loss': val_loss, 'val_acc': val_acc})
    
    # Record fold processing time
    fold_time = time.time() - start_time
    fold_times.append(fold_time)
    
    # Save model for this fold
    model.save(f'densenet_model_fold_{fold}.h5')
    
    # Update best fold based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_fold = fold
    
    # Save training history
    all_histories.append(history.history)
    
    print(f'Accuracy Fold {fold}: {val_acc:.4f}, Loss Fold {fold}: {val_loss:.4f}, Time: {fold_time:.2f} seconds')

# Calculate total and average time per fold
total_time = time.time() - total_start_time
avg_fold_time = np.mean(fold_times)

# Handle different epoch lengths to calculate averages
min_epochs = min(len(h['accuracy']) for h in all_histories)
avg_accuracy = np.mean([h['accuracy'][:min_epochs] for h in all_histories], axis=0)
avg_val_accuracy = np.mean([h['val_accuracy'][:min_epochs] for h in all_histories], axis=0)
avg_loss = np.mean([h['loss'][:min_epochs] for h in all_histories], axis=0)
avg_val_loss = np.mean([h['val_loss'][:min_epochs] for h in all_histories], axis=0)

# Print results
print(f"\nTraining with {n_splits}-fold cross-validation completed.")
print(f"Average Accuracy Across Folds: {np.mean([r['val_acc'] for r in fold_results]):.4f} (+/- {np.std([r['val_acc'] for r in fold_results]):.4f})")
print(f"Average Loss Across Folds: {np.mean([r['val_loss'] for r in fold_results]):.4f} (+/- {np.std([r['val_loss'] for r in fold_results]):.4f})")
print(f"Best Fold: Fold {best_fold} with Val Accuracy = {best_val_acc:.4f}")
print(f"Total Processing Time: {total_time:.2f} seconds")
print(f"Average Processing Time per Fold: {avg_fold_time:.2f} seconds")
for i, (result, t) in enumerate(zip(fold_results, fold_times), 1):
    print(f"Fold {i}: Accuracy = {result['val_acc']:.4f}, Loss = {result['val_loss']:.4f}, Time = {t:.2f} seconds")

# Load best model
best_model = tf.keras.models.load_model(f'densenet_model_fold_{best_fold}.h5')
print(f"Successfully loaded best model from densenet_model_fold_{best_fold}.h5")

# Visualize average accuracy and loss
plt.figure(figsize=(12, 4))

# Subplot 1: Average Accuracy
plt.subplot(1, 2, 1)
plt.plot(avg_accuracy, label='Average Train Accuracy', color='blue', linewidth=2)
plt.plot(avg_val_accuracy, label='Average Validation Accuracy', color='orange', linestyle='--', linewidth=2)
plt.title(f'Average Model Accuracy Across {n_splits} Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Subplot 2: Average Loss
plt.subplot(1, 2, 2)
plt.plot(avg_loss, label='Average Train Loss', color='blue', linewidth=2)
plt.plot(avg_val_loss, label='Average Validation Loss', color='orange', linestyle='--', linewidth=2)
plt.title(f'Average Model Loss Across {n_splits} Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
