from train import (
    train_experiment
)

from evaluate import (
    evaluate_experiment
)


TRAINING_PERCENTAGES = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00
]


RUNS = 5

EPOCHS = 10


for percentage in (
    TRAINING_PERCENTAGES
):

    for run in range(
        1,
        RUNS + 1
    ):

        experiment_name = (
            f"cnn_"
            f"{int(percentage * 100)}"
            f"_run{run}"
        )


        print()
        print("########################################")
        print(f"EXPERIMENTO: " f"{experiment_name}")
        print("########################################")

        print(
            f"Dados: "
            f"{percentage * 100:.0f}%"
        )

        print(
            f"Run: {run}/{RUNS}"
        )

        print(
            f"Épocas: {EPOCHS}"
        )


        # TREINAMENTO

        train_experiment(
            percentage=percentage,
            run=run,
            epochs=EPOCHS
        )


        # TESTE

        evaluate_experiment(
            percentage=percentage,
            run=run
        )


print()
print("========================================")
print("TODOS OS EXPERIMENTOS FINALIZADOS")
print("========================================")