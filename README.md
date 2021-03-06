# Team16 - American Sign Language Recognition
## Web Site
https://giannog.github.io/team16/  
## Poster
https://github.com/giannog/team16/blob/main/poster.pdf  
## Presentation Video Link
https://youtu.be/uAUrPkzlEgs  

## Dataset Information
The dataset format is patterned to match closely with the classic MNIST. Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions). The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255.  
### Dataset Download
Dataset is original from: [Kaggle Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)  
The processed one can be downloaded from here: [Google Drive](https://drive.google.com/drive/folders/1qqG8eZ96EcRoO-1jaVLY1na6iS3iUi2T?usp=sharing)
### Dataset Usage
**Choose only one method below:**  
#### A. Original one  
1. You may need to register for one account to download the kaggle dataset
2. When downloading, name the zip file into "Sign Language MNIST.zip"
3. Uncompress the file:  
    `Windows`: Right click the zip file and select `Extract All...`, and click `Extract`  
    `Linux` ( To present the whitespace, you need to use escape character `\`, see below ):
    ```
    mkdir Sign\ Language\ MNIST
    unzip Sign\ Language\ MNIST.zip -d ./Sign\ Language\ MNIST
    ```
    `MAC OS`: Directly double-click the zip file
4. Once uncompressed, you will have a folder named "Sign Language MNIST" which contains the following contents:  
![Original Dataset Content](original%20dataset%20content.png)
5. Copy the whole folder to your Google Drive under the following path (create the path if you do not have):  
`Colab Notebooks/Artificial Intelligence/Data/`
6. Once done, you can try the notebook file without any issue.
#### B. Processed one  
1. Copy the whole folder to your Google Drive under the following path (create the path if you do not have):  
`Colab Notebooks/Artificial Intelligence/Data/`
2. Once done, you can try the notebook file without any issue.
