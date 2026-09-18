import csv
import numpy as np
import tensorflow as tf

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from dataset import (
    test_paths,
    test_labels,
    class_names
)

from tensorflow_dataset import (
    create_dataset
)


def evaluate_experiment(
    percentage,
    run
):

    experiment_name = (
        f"cnn_"
        f"{int(percentage * 100)}"
        f"_run{run}"
    )


    model_path = (
        f"models/"
        f"{experiment_name}.keras"
    )


    # DATASET DE TESTE

    test_dataset = create_dataset(
        test_paths,
        test_labels,
        training=False
    )


    # CARREGAR MODELO

    model = (
        tf.keras.models.load_model(
            model_path
        )
    )


    # AVALIAÇÃO KERAS

    test_loss, test_accuracy = (
        model.evaluate(
            test_dataset,
            verbose=1
        )
    )


    # PREVISÕES

    predictions = model.predict(
        test_dataset,
        verbose=1
    )


    predicted_labels = np.argmax(
        predictions,
        axis=1
    )


    # MÉTRICAS

    accuracy = accuracy_score(
        test_labels,
        predicted_labels
    )


    precision = precision_score(
        test_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )


    recall = recall_score(
        test_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )


    f1 = f1_score(
        test_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )


    print()
    print("========================================")
    print(
        f"RESULTADO: "
        f"{experiment_name}"
    )
    print("========================================")


    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall: "
        f"{recall:.4f}"
    )

    print(
        f"F1: "
        f"{f1:.4f}"
    )


    # RELATÓRIO POR CLASSE

    report = classification_report(
        test_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print()
    print(report)


    # SALVAR MÉTRICAS

    metrics_dir = Path(
        "results/metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    metrics_path = (
        metrics_dir /
        f"{experiment_name}_metrics.csv"
    )


    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "percentage",
            "run",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "test_loss"
        ])


        writer.writerow([
            percentage,
            run,
            accuracy,
            precision,
            recall,
            f1,
            test_loss
        ])


    # MATRIZ DE CONFUSÃO

    matrix = confusion_matrix(
        test_labels,
        predicted_labels
    )


    confusion_dir = Path(
        "results/confusion_matrices"
    )

    confusion_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    confusion_path = (
        confusion_dir /
        f"{experiment_name}"
        "_confusion_matrix.csv"
    )


    with open(
        confusion_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "classe_real",
            *class_names
        ])


        for class_name, row in zip(
            class_names,
            matrix
        ):

            writer.writerow([
                class_name,
                *row
            ])


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "test_loss": test_loss
    }