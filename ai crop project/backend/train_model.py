"""
PlantCare AI - Model Training Pipeline
Transfer Learning with MobileNetV2 on PlantVillage Dataset.
Includes data splitting, augmentation, transfer learning, fine-tuning,
early stopping, model checkpointing, evaluation metrics, and artifact generation.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress excessive TensorFlow debug logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError as e:
    print("\n[!] Error: TensorFlow or scikit-learn is not installed.")
    print("    Please run: pip install tensorflow scikit-learn matplotlib seaborn\n")
    sys.exit(1)


# Default Configurations
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 15
DEFAULT_FINE_TUNE_EPOCHS = 10
RANDOM_SEED = 42


def parse_args():
    """Parses command-line arguments for training customization."""
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 Transfer Learning Model on Plant Disease Dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "dataset", "PlantVillage"),
        help="Path to dataset directory containing class subfolders."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="Directory to save trained model and class indices."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of epochs for initial transfer learning training."
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=DEFAULT_FINE_TUNE_EPOCHS,
        help="Number of epochs for fine-tuning top layers (0 to disable)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size (reduce to 16 if laptop runs out of RAM/GPU memory)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for classification head."
    )
    return parser.parse_args()


def prepare_datasets(data_dir: str, batch_size: int):
    """
    Loads images from class subdirectories and partitions them cleanly
    into train (70%), validation (15%), and held-out test (15%) sets.
    Applies data augmentation to the training split only to prevent data leakage.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found. "
            "Please download the PlantVillage dataset and place folders in dataset/PlantVillage."
        )

    print(f"\n[*] Scanning dataset from: {os.path.abspath(data_dir)}")
    
    # 1. Load initial 85% train+val and 15% test
    # Seed ensures deterministic split without data leakage
    train_val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.15,
        subset="training",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.15,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_val_ds.class_names
    num_classes = len(class_names)
    print(f"[+] Detected {num_classes} plant/disease classes:")
    for idx, name in enumerate(class_names):
        print(f"    [{idx:2d}] {name}")

    # Further partition train_val into train (70% total) and val (15% total)
    # Total train_val batches = total * 0.85; val needs 0.15 / 0.85 = ~17.6% of train_val
    train_val_batches = tf.data.experimental.cardinality(train_val_ds).numpy()
    val_batches = max(1, int(train_val_batches * 0.176))

    val_ds = train_val_ds.take(val_batches)
    train_ds = train_val_ds.skip(val_batches)

    # Data Augmentation pipeline (active only during training)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # Map preprocessing: augmentation on train only, MobileNetV2 normalizer on all
    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int, learning_rate: float) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs transfer learning model based on MobileNetV2.
    Pretrained ImageNet weights are frozen initially.
    """
    print("\n[*] Initializing pretrained MobileNetV2 backbone (ImageNet weights)...")
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    base_model.trainable = False  # Freeze all base layers initially

    # Classification Head
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_leaf_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dropout(0.3, name="head_dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_256")(x)
    x = layers.Dropout(0.2, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_prediction")(x)

    model = tf.keras.Model(inputs, outputs, name="PlantCare_MobileNetV2")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("[+] Model architecture assembled successfully.")
    return model, base_model


def plot_training_curves(history, output_path: str):
    """Plots and saves loss and accuracy learning curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2e7d32", lw=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#388e3c", linestyle="--", lw=2)
    plt.title("Model Accuracy Across Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#c62828", lw=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#d32f2f", linestyle="--", lw=2)
    plt.title("Cross-Entropy Loss Across Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved training history curves to: {output_path}")


def evaluate_and_report(model, test_ds, class_names, output_dir: str):
    """
    Evaluates model on held-out test set and prints precision, recall,
    F1 score, and confusion matrix.
    """
    print("\n" + "=" * 60)
    print("   HELD-OUT TEST SET EVALUATION (Zero Data Leakage)")
    print("=" * 60)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification Metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix - Plant Disease Classification", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Ground Truth Class", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix heatmap saved to: {cm_path}")

    # Interpretation Guide
    print("\n[i] HOW TO INTERPRET THESE METRICS:")
    print("  - Precision: When the model predicts 'Tomato Late Blight', how often is it truly Late Blight?")
    print("  - Recall (Sensitivity): Out of all true 'Late Blight' leaves, how many did the model detect?")
    print("  - F1-Score: Harmonic balance between precision and recall; critical in medical/agricultural diagnostics.")
    print("  - Note on Softmax: A 95% softmax confidence is a relative score among supported classes,")
    print("    not an absolute probability. Out-of-distribution leaves may still produce high scores.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_save_path = os.path.join(args.output_dir, "plant_disease_mobilenetv2.keras")
    classes_save_path = os.path.join(args.output_dir, "class_indices.json")

    print("\n" + "=" * 60)
    print("   PlantCare AI - Model Training Engine (MobileNetV2)")
    print("=" * 60)
    print(f"Dataset Directory : {os.path.abspath(args.data_dir)}")
    print(f"Output Directory  : {os.path.abspath(args.output_dir)}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Initial Epochs    : {args.epochs}")
    print(f"Fine-Tune Epochs  : {args.fine_tune_epochs}")

    # Step 1: Load and partition dataset
    train_ds, val_ds, test_ds, class_names = prepare_datasets(args.data_dir, args.batch_size)
    num_classes = len(class_names)

    # Save class indices mapping
    class_indices_dict = {str(i): name for i, name in enumerate(class_names)}
    with open(classes_save_path, "w", encoding="utf-8") as f:
        json.dump(class_indices_dict, f, indent=2)
    print(f"[+] Class indices saved to: {classes_save_path}")

    # Step 2: Build model
    model, base_model = build_model(num_classes, args.lr)

    # Step 3: Setup Callbacks
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # Step 4: Phase 1 Training (Frozen Base)
    print("\n[*] Phase 1: Training Classification Head (Backbone Frozen)...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )

    # Step 5: Phase 2 Fine-Tuning (Optional unfreezing of top layers)
    if args.fine_tune_epochs > 0:
        print("\n[*] Phase 2: Fine-Tuning top MobileNetV2 layers...")
        base_model.trainable = True
        # Freeze all layers except the last 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Recompile with very small learning rate for gentle fine-tuning
        model.compile(
            optimizer=optimizers.Adam(learning_rate=args.lr * 0.1),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=len(history_phase1.epoch),
            callbacks=[model_checkpoint, early_stopping, reduce_lr]
        )
        combined_history = history_phase2
    else:
        combined_history = history_phase1

    # Step 6: Plot Learning Curves
    curves_path = os.path.join(args.output_dir, "training_history.png")
    plot_training_curves(combined_history, curves_path)

    # Step 7: Load Best Weights and Evaluate on Held-Out Test Set
    if os.path.exists(model_save_path):
        print(f"\n[*] Loading best checkpoint weights from: {model_save_path}")
        best_model = tf.keras.models.load_model(model_save_path)
    else:
        best_model = model

    evaluate_and_report(best_model, test_ds, class_names, args.output_dir)

    print("\n" + "=" * 60)
    print("   [+] TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"   Model saved: {os.path.abspath(model_save_path)}")
    print(f"   Classes file: {os.path.abspath(classes_save_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
