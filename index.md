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
### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/giannog/team16/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
