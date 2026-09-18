from dataset import (
    train_paths,
    train_labels,
    validation_paths,
    validation_labels
)

from tensorflow_dataset import (
    create_dataset
)


train_dataset = create_dataset(
    train_paths,
    train_labels,
    training=True
)


validation_dataset = (
    create_dataset(
        validation_paths,
        validation_labels,
        training=False
    )
)


print("Treinamento:", len(train_paths))

print("Validação:", len(validation_paths))


for images, labels in (
    train_dataset.take(1)
):

    print()
    print("Formato das imagens:")

    print( images.shape)

    print()
    print("Formato dos labels:")

    print(labels.shape)

    print()
    print("Labels:")
    print(labels.numpy())