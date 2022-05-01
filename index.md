Welcome to Team 16 
### Features
Implementing the Vision Transformer (ViT) model for American Sign Language (ASL) image classification.

Dataset: Sign Language MNIST.

The code are partially based on the ViT model on keras. ViT are proposed by Alexey Dosovitskiy et al. for image classification, which applies the Transformer architecture with self-attention to sequences of image patches, without using convolution layers.

```training
Syntax highlighted code block
basePath = ''
train = pd.read_csv(basePath + "sign_mnist_train.csv")
test = pd.read_csv(basePath + "sign_mnist_test.csv")

# generate pictures stored path
if not os.path.exists(basePath + "train_pic"):
    os.mkdir(basePath + "train_pic")
if not os.path.exists(basePath + "test_pic"):
    os.mkdir(basePath + "test_pic")

train_pic_path = basePath + "train_pic/"
test_pic_path = basePath + "test_pic/"
```

### Data augmentation
Apply data augmentation on images
```
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)


# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)
```
### Patch creation layer
Let's display patches for a sample image

```
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")
```

### Data Loading

![data loading](https://user-images.githubusercontent.com/100370619/166123725-10a7becc-e12e-42cd-aaac-1245dd0ccdc7.PNG)

### Construct the model

The ViT model that consists of multiple Transformer blocks

```
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
```
### Model Training
![epoch 8](https://user-images.githubusercontent.com/100370619/166130516-f13d34a8-a32e-477c-b314-0212dc6c8bc8.PNG)

### Evaluation results
After 8 epochs, the ViT model achieves around 99.2% accuracy and 100% top-5 accuracy on the ASL test data.
![accuracy plot](https://user-images.githubusercontent.com/100370619/166130530-df374224-4bcd-4cc6-a3c2-2c4e87f9d4f6.PNG)

### Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/uAUrPkzlEgs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

