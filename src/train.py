import csv
import time
import tensorflow as tf

from pathlib import Path

from dataset import (
    training_subsets,
    validation_paths,
    validation_labels
)

from tensorflow_dataset import (
    create_dataset
)

from model import create_cnn


def train_experiment(
    percentage,
    run,
    epochs
):

    # Limpa estado de modelos anteriores
    tf.keras.backend.clear_session()


    experiment_name = (
        f"cnn_"
        f"{int(percentage * 100)}"
        f"_run{run}"
    )


    # DADOS

    train_paths, train_labels = (
        training_subsets[
            percentage
        ]
    )


    train_dataset = (
        create_dataset(
            train_paths,
            train_labels,
            training=True
        )
    )


    validation_dataset = (
        create_dataset(
            validation_paths,
            validation_labels,
            training=False
        )
    )


    print()
    print("========================================")
    print("TREINAMENTO")
    print("========================================")

    print(
        f"Experimento: "
        f"{experiment_name}"
    )

    print(
        f"Treino: "
        f"{len(train_paths)} imagens"
    )

    print(
        f"Validação: "
        f"{len(validation_paths)} imagens"
    )

    print(
        f"Épocas: {epochs}"
    )


    # MODELO

    model = create_cnn()


    # TREINAMENTO

    start_time = (
        time.perf_counter()
    )


    history = model.fit(
        train_dataset,
        validation_data=(
            validation_dataset
        ),
        epochs=epochs,

        # O dataset já foi embaralhado
        # pelo tf.data
        shuffle=False
    )


    end_time = (
        time.perf_counter()
    )


    training_time = (
        end_time - start_time
    )


    # RESULTADOS

    print()
    print("========================================")
    print("TREINAMENTO FINALIZADO")
    print("========================================")


    print(
        f"Tempo: "
        f"{training_time:.2f}s"
    )


    print(
        f"Accuracy treino: "
        f"{history.history['accuracy'][-1]:.4f}"
    )


    print(
        f"Accuracy validação: "
        f"{history.history['val_accuracy'][-1]:.4f}"
    )


    # SALVAR MODELO

    models_dir = Path(
        "models"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    model_path = (
        models_dir /
        f"{experiment_name}.keras"
    )


    model.save(
        model_path
    )


    # SALVAR HISTÓRICO

    history_dir = Path(
        "results/histories"
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    history_path = (
        history_dir /
        f"{experiment_name}_history.csv"
    )


    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "epoch",
            "accuracy",
            "loss",
            "val_accuracy",
            "val_loss"
        ])


        for epoch in range(
            len(
                history.history[
                    "accuracy"
                ]
            )
        ):

            writer.writerow([
                epoch + 1,

                history.history[
                    "accuracy"
                ][epoch],

                history.history[
                    "loss"
                ][epoch],

                history.history[
                    "val_accuracy"
                ][epoch],

                history.history[
                    "val_loss"
                ][epoch]
            ])


    # SALVAR DADOS DO TREINAMENTO

    training_dir = Path(
        "results/training"
    )

    training_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    training_path = (
        training_dir /
        f"{experiment_name}_training.csv"
    )


    with open(
        training_path,
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
            "training_images",
            "validation_images",
            "epochs",
            "training_time",
            "final_accuracy",
            "final_val_accuracy"
        ])


        writer.writerow([
            percentage,
            run,
            len(train_paths),
            len(validation_paths),
            epochs,
            training_time,

            history.history[
                "accuracy"
            ][-1],

            history.history[
                "val_accuracy"
            ][-1]
        ])


    return model_path