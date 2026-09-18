from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import random


# CONFIGURAÇÕES

DATASET_DIR = Path("dataset/Tomato")

SEED = 42

TRAIN_PERCENTAGE = 0.70

TRAINING_PERCENTAGES = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00
]


# IDENTIFICAR CLASSES

class_names = sorted([
    folder.name
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
])


class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}


# ENCONTRAR IMAGENS

image_paths = []
labels = []


for class_name in class_names:

    class_dir = DATASET_DIR / class_name

    for image_path in class_dir.glob("*"):

        if image_path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:

            image_paths.append(
                str(image_path)
            )

            labels.append(
                class_to_index[class_name]
            )


# TREINO / VALIDAÇÃO / TESTE

train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        image_paths,
        labels,
        train_size=TRAIN_PERCENTAGE,
        random_state=SEED,
        stratify=labels
    )
)


validation_paths, test_paths, validation_labels, test_labels = (
    train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels
    )
)


# ORGANIZAR TREINO POR CLASSE

class_data = {}


for class_index in range(len(class_names)):

    paths = [
        path
        for path, label
        in zip(train_paths, train_labels)
        if label == class_index
    ]

    random.Random(
        SEED + class_index
    ).shuffle(paths)

    class_data[class_index] = paths


# SUBCONJUNTOS DE TREINAMENTO

training_subsets = {}


for percentage in TRAINING_PERCENTAGES:

    subset_paths = []
    subset_labels = []

    for class_index in range(
        len(class_names)
    ):

        paths = class_data[class_index]

        quantity = int(
            len(paths) * percentage
        )

        selected_paths = (
            paths[:quantity]
        )

        subset_paths.extend(
            selected_paths
        )

        subset_labels.extend(
            [class_index]
            * len(selected_paths)
        )

    training_subsets[percentage] = (
        subset_paths,
        subset_labels
    )


# FUNÇÃO APENAS PARA INSPEÇÃO

def show_class_distribution(
    name,
    dataset_labels
):

    counter = Counter(
        dataset_labels
    )

    print()
    print(name)

    for class_index, class_name in enumerate(
        class_names
    ):

        print(
            f"{class_name}: "
            f"{counter[class_index]}"
        )


# EXECUTAR DIRETAMENTE

if __name__ == "__main__":

    print("Classes:")

    for class_name, index in (
        class_to_index.items()
    ):
        print(
            f"{index}: {class_name}"
        )

    print()
    print(
        f"Total: {len(image_paths)}"
    )

    print(
        f"Treinamento: "
        f"{len(train_paths)}"
    )

    print(
        f"Validação: "
        f"{len(validation_paths)}"
    )

    print(
        f"Teste: "
        f"{len(test_paths)}"
    )

    for percentage, (
        paths,
        subset_labels
    ) in training_subsets.items():

        print()
        print(
            f"{percentage * 100:.0f}%: "
            f"{len(paths)} imagens"
        )

        show_class_distribution(
            "Distribuição:",
            subset_labels
        )