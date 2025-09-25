import time
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

# For segmenation training with U-Net (example)
def train_unet_kfold(images, masks, n_splits=3, epochs=25, batch_size=4):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagenSeg = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Set True if appropriate
        fill_mode='nearest'
    )
    datagenSeg.fit(images)
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    best_val_acc = -1
    best_fold = -1
    all_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(images, masks), 1):
        X_train, X_val = images[train_idx], images[val_idx]
        y_train, y_val = masks[train_idx], masks[val_idx]
        
        model = build_unet(input_shape=images.shape[1:])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', dice_coefficient]
        )
        
        history = model.fit(
            datagenSeg.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            verbose=1
        )
        
        val_loss, val_acc, val_dice = model.evaluate(X_val, y_val, verbose=0)
        fold_results.append({'fold': fold, 'val_loss': val_loss, 'val_acc': val_acc, 'val_dice': val_dice})
        model.save(f'unet_model_fold_{fold}.h5')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_fold = fold
        
        all_histories.append(history.history)
    
    return fold_results, best_fold, best_val_acc, all_histories

# For classification training with DenseNet121 (example)
def train_densenet_kfold(X_train, y_train, n_splits=10, epochs=50, batch_size=8):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagenKlas = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    fold_times = []
    best_val_acc = -1
    best_fold = -1
    all_histories = []
    total_start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):
        start_time = time.time()
        
        X_train_fold = np.array([cv2.resize(img, (224, 224)) for img in X_train[train_idx]])
        y_train_fold = y_train[train_idx]
        X_val_fold = np.array([cv2.resize(img, (224, 224)) for img in X_train[val_idx]])
        y_val_fold = y_train[val_idx]
        
        model = build_model()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        train_generator = datagenKlas.flow(X_train_fold, y_train_fold, batch_size=batch_size)
        
        history = model.fit(
            train_generator,
            steps_per_epoch=len(X_train_fold) // batch_size,
            validation_data=(X_val_fold, y_val_fold),
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=1
        )
        
        val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        fold_results.append({'fold': fold, 'val_loss': val_loss, 'val_acc': val_acc})
        
        fold_times.append(time.time() - start_time)
        
        model.save(f'densenet_model_fold_{fold}.h5')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_fold = fold
        
        all_histories.append(history.history)
        
        print(f'Fold {fold}: Accuracy = {val_acc:.4f}, Loss = {val_loss:.4f}, Time = {fold_times[-1]:.2f}s')
    
    total_time = time.time() - total_start_time
    avg_fold_time = np.mean(fold_times)
    
    print(f'Total Training Time: {total_time:.2f}s, Average Time per Fold: {avg_fold_time:.2f}s')
    
    return fold_results, best_fold, best_val_acc, all_histories
