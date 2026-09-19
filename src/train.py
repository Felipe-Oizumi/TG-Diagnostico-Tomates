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

from resource_monitor import (
    ResourceMonitor
)


def train_experiment(percentage, run, epochs):

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

    monitor = ResourceMonitor(interval=1.0)


    start_time = (time.perf_counter())

    monitor.start()


    history = model.fit(
        train_dataset,
        validation_data=(validation_dataset),
        epochs=epochs,
        shuffle=False
    )


    resources = monitor.stop()

    end_time = (time.perf_counter())


    training_time = (end_time - start_time)


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

    models_dir = Path("models")

    models_dir.mkdir(parents=True, exist_ok=True)


    model_path = (models_dir / f"{experiment_name}.keras")


    model.save(model_path)


    # SALVAR HISTÓRICO

    history_dir = Path("results/histories")

    history_dir.mkdir(parents=True, exist_ok=True)


    history_path = (history_dir / f"{experiment_name}_history.csv")


    with open(history_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "accuracy",
            "loss",
            "val_accuracy",
            "val_loss"
        ])


        for epoch in range(
            len(history.history["accuracy"])
        ):

            writer.writerow([
                epoch + 1,
                history.history["accuracy"][epoch],
                history.history["loss"][epoch],
                history.history["val_accuracy"][epoch],
                history.history["val_loss"][epoch]
            ])


    # SALVAR DADOS DO TREINAMENTO

    training_dir = Path("results/training")

    training_dir.mkdir(parents=True, exist_ok=True)


    training_path = (training_dir / f"{experiment_name}_training.csv" )


    with open(training_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "percentage",
            "run",
            "training_images",
            "validation_images",
            "epochs",
            "training_time",
            "cpu_avg",
            "cpu_peak",
            "cpu_time",
            "ram_avg_mb",
            "ram_peak_mb",
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
            resources["cpu_avg" ],
            resources["cpu_peak"],
            resources["cpu_time"],
            resources["ram_avg_mb"],
            resources["ram_peak_mb" ],
            history.history["accuracy"][-1],
            history.history["val_accuracy"][-1]
        ])

    resources_dir = Path("results/resources")

    resources_dir.mkdir(parents=True, exist_ok=True)


    resources_path = (resources_dir / f"{experiment_name}_resources.csv")


    with open(resources_path, "w", newline="", encoding="utf-8" ) as file:

        writer = csv.writer(file)

        writer.writerow(["time_seconds", "cpu_percent", "ram_mb"])

        for sample in monitor.samples:

            writer.writerow([
                sample["time_seconds"],
                sample["cpu_percent"],
                sample["ram_mb"]
            ])

    return model_path