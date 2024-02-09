# Web Phishing Detection with Deep Learning
The main goal of this repository is to identify and classify phishing websites using deep learning techniques. Phishing websites are malicious sites that try to trick users into providing sensitive information, such as usernames, passwords, and credit card details, by pretending to be trustworthy entities. The raw data being analyzed here are URLs and HTML content of websites. URL analysis involves examining the structure and contents of the URL for signs of phishing, such as misspelled domain names or suspicious subdomains. HTML content analysis involves examining the actual contents of the webpage, such as forms that ask for sensitive information, the presence of suspicious scripts, or the overall structure and design of the page.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Room For Improvement](#room-for-improvement)
- [References](#references)
- [Attachment](#attachment)

# Introduction
Phishing attacks are increasingly prevalent in today's digital age. These malicious acts aim to deceive users into revealing sensitive information such as usernames, passwords, and credit card details by posing as trustworthy entities. The repercussions of these attacks can be serious, leading to identity theft, financial loss, and significant harm to an individual's or an organization’s reputation. 

Traditionally, the detection of phishing has heavily relied on analyzing URLs. These systems scrutinize suspicious patterns in URLs, such as misspelled domain names or unusual subdomains. However, this approach is not without its flaws. Skilled phishers have devised numerous tactics to evade URL-based detection. For instance, they might employ methods like URL redirection, where the URL initially appears legitimate but redirects the user to a malicious site. Other tactics include using URL shorteners or concealing the malicious URL within seemingly harmless hyperlinks.

In response to this, I have developed a more robust approach to phishing detection in this repository. This approach analyzes not just URLs, but also the HTML content of web pages. By utilizing deep learning techniques, the model can extract and learn from complex features and patterns within the raw data. Deep learning, a branch of machine learning that uses multi-layered neural networks, is particularly effective for this task. It enables the model to delve into the actual contents of web pages, such as forms that ask for sensitive information, the presence of suspicious scripts, or the overall structure and design of the page, going beyond merely examining the structure of URLs.

After processing both the URLs and HTML content, the model predicts whether the webpage is legitimate or a potential phishing threat. By combining URL and HTML content analysis, this approach offers a more comprehensive and reliable phishing detection system. Through this project, I hope to contribute to making the internet a safer place for users.

# Dataset
The 'look-before-you-leap' dataset, accessible on Kaggle, is employed in this project. It's a balanced dataset consisting of 45,373 instances, equally representing both benign and phishing web pages. Each instance encompasses a variety of HTML document elements such as texts, hyperlinks, images, tables, lists, and diverse URL components from subdomains to queries.

The dataset's creator has already prudently removed URL prefixes like HTTP:// and HTTPS:// from the dataset. This essential modification allows the model to focus on the more critical parts of the URL. It also guarantees the model's consistent performance across different URL datasets, enhancing its generality and avoiding skewed results.

The dataset comprises real-world data collected from Alexa.com for legitimate web pages and phishtank.com for phishing web pages. The use of these trusted sources guarantees a realistic data mix, creating a robust and authentic training environment for the deep learning model.

---

# Methodology
## Exploratory Data Analysis
### Overview of Data
This part begins by printing the shape of both URL and HTML datasets to understand the size of the data in terms of the number of samples.

```
URL Dataset Shape: (45373, 4)
HTML Dataset Shape: (45373, 4)
```

### Content Examination
Random samples from both datasets are displayed to get a sense of the actual content being worked with, including the format and variability of URLs and HTML content.

```
Sample URLs:
25715                       www.isoc.org/internet/history/
18810                                       www.aeon-jp.cc
40421                         miolkoijhjhjhb.gq/CC_POSTALE
18744    www.theatlantic.com/unbound/digitalreader/dr20...
28869                 dimacs.rutgers.edu/Workshops/Faster/
Name: Data, dtype: object

Sample HTML Contents:
34050    <!DOCTYPE html><!--[if IE 8]><html class="ie8 ...
28401    <!DOCTYPE html><html class="no-js" lang="ja" p...
30395    <!DOCTYPE html><!--[if IE 8]><html class="no-j...
16411    <!DOCTYPE html>', '<html itemscope="" itemtype...
17942    <!DOCTYPE html>', '', '<html dir="rtl" prefix=...
Name: Data, dtype: object
```

### Data Types
The data types of columns in both datasets are printed to ensure that the data is in the expected format, which is crucial for preprocessing and modeling.
```
URL Dataset Data Types:
Category            object
Data                object
Cleaned_Data        object
Category_Encoded     int64
dtype: object

HTML Dataset Data Types:
Category            object
Data                object
Cleaned_Data        object
Category_Encoded     int64
dtype: object
```

### Missing Values
A check for missing values in both datasets is performed, as missing data can affect model training and might require imputation or removal.
```
Missing Values in URL Dataset:
Category            0
Data                0
Cleaned_Data        0
Category_Encoded    0
dtype: int64

Missing Values in HTML Dataset:
Category            0
Data                0
Cleaned_Data        0
Category_Encoded    0
dtype: int64
```

### Value Counts
The distribution of categories (e.g., phishing/spam vs. legitimate/ham) in both datasets is examined to understand the balance or imbalance between classes, which can impact model performance.
```
Category Distribution in URL Dataset:
ham     22687
spam    22686
Name: Category, dtype: int64

Category Distribution in HTML Dataset:
ham     22687
spam    22686
Name: Category, dtype: int64
```

### Unique Values
The number of unique URLs and HTML contents is calculated and printed to assess data diversity and redundancy.
```
Unique URLs: 44078
Unique HTML Contents: 32707
```

### URL and HTML Distribution
The distribution of HTML and URL is visualized using donut charts.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/d485510d-6a03-4e18-b9ec-282a0d557b3e)

### Distribution of Categories
The category distribution in both datasets is visualized using bar charts, which helps in understanding class balance and may inform the need for stratification or rebalancing techniques.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/b34eb679-830b-4b70-b608-7c9770c9d79c)

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/5e4ac6e6-c94c-4796-814f-3a7dbfb9ae09)

### Distribution Percentages
The percentage distribution of categories in both datasets is calculated and printed to provide a clearer view of class imbalance in percentage terms.

### Visualizing Category Distribution Percentages
Donut charts are created to visually represent the percentage distribution of categories in both datasets, offering an intuitive understanding of class proportions.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/1fbbebed-56dc-4feb-b22a-486e505b0e0e)

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/636dd305-9d46-4511-808d-4488098b4065)

### URL and HTML Length Analysis
The length of URLs and HTML content with respect to their categories is analyzed and visualized using box plots. This can reveal patterns such as longer or shorter lengths being associated with phishing/spam or legitimate content, which might be useful features for modeling.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/7deb2ef0-7c54-4cee-a1d7-367de4fc056d)

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/538050c4-e68a-4122-9c07-4eec102e0c1c)

---

## Preprocessing

### 1. **Download NLTK Resources**
The Natural Language Toolkit (NLTK) resources 'punkt', 'stopwords', and 'wordnet' are downloaded. These are used for tokenizing, removing stopwords, and lemmatizing words respectively.

### 2. **Load Stopwords**
The stopwords from the 'english' language are loaded into a set for efficient removal later.

### 3. **Initialize Lemmatizer**
The WordNetLemmatizer from NLTK is initialized. This is used to convert words to their base form.

### 4. **Clean and Preprocess URL Data**
The URLs are cleaned and preprocessed by converting to lowercase, removing 'http' or 'https', removing 'www', removing special characters, and removing extra spaces. The URLs are then tokenized, stopwords are removed, and words are lemmatized. Though there are no 'http' or 'https' in the dataset, I just wanted to make sure.

```
# Function to clean and preprocess URL data
def preprocess_url(url):
    url = url.lower()  # Convert to lowercase
    url = re.sub(r'https?://', '', url)  # Remove http or https
    url = re.sub(r'www\.', '', url)  # Remove www
    url = re.sub(r'[^a-zA-Z0-9]', ' ', url)  # Remove special characters
    url = re.sub(r'\s+', ' ', url).strip()  # Remove extra spaces
    tokens = word_tokenize(url)  # Tokenize
    tokens = [word for word in tokens if word not in STOPWORDS]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)
```

### 5. **Clean and Preprocess HTML Data**
The HTML data is cleaned and preprocessed by removing HTML tags, converting to lowercase, removing 'http' or 'https', removing special characters, and removing extra spaces. The HTML data is then tokenized, stopwords are removed, and words are lemmatized.

```
# Function to clean and preprocess HTML data
def preprocess_html(html):
    html = re.sub(r'<[^>]+>', ' ', html)  # Remove HTML tags
    html = html.lower()  # Convert to lowercase
    html = re.sub(r'https?://', '', html)  # Remove http or https
    html = re.sub(r'[^a-zA-Z0-9]', ' ', html)  # Remove special characters
    html = re.sub(r'\s+', ' ', html).strip()  # Remove extra spaces
    tokens = word_tokenize(html)  # Tokenize
    tokens = [word for word in tokens if word not in STOPWORDS]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)
```

### 6. **Clean URL and HTML 'Data' Columns**
The 'Data' columns in the URL and HTML dataframes are cleaned using the above preprocessing functions.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/97d730cf-9a00-44ab-aa36-34e0bc7eb308)

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/81119ab3-ee0d-4ecc-9895-f178b4a876a6)


### 7. **Define Maximum Length of Sequences**
The maximum length of sequences for the URL and HTML data is defined.

```
max_url_length = 180
max_html_length = 2000
```

### 8. **Define Maximum Number of Words**
The maximum number of words/tokens is defined.

```
# Define the maximum number of words/tokens
max_words = 10000
```

### 9. **URL and HTML Tokenization and Padding**
The cleaned URL and HTML data are tokenized and padded to the maximum length of sequences. For URLs, character-level tokenization is used. For HTML data, word-level tokenization is used.

### 10. **Encode 'Category' Column**
The 'Category' column in the URL and HTML dataframes is encoded into numerical values using Label Encoding.

### 11. **Split Datasets Into Training and Testing Sets**
The URL and HTML datasets are split into training and testing sets, with 80% of the data used for training and 20% used for testing.

---

## Model Building
The architecture of the model is inspired by this [paper](https://www.sciencedirect.com/science/article/pii/S0957417423016858), but with several modifications to fine-tune the model's performance. These alterations include the addition of up to three Convolutional Layers, an increase in the number of filters within these layers from 32 to 128, and the introduction of Fully Connected Layers. These modifications serve as hyperparameters adjustments to enhance the model's learning capacity and accuracy. Below are modified hyperparameters:

```
def create_model():
    # Adjusted hyperparameters
    embedding_dim = 32  # Increased embedding dimension
    conv_filters = 128  # Increased number of filters in convolutional layers
    kernel_size = 10 # Increased kernel size
    dense_units_1 = 128 
    dense_units_2 = 64  
    learning_rate = 0.0005  # Adjusted learning rate
```

### Model Architecture

```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 url_input (InputLayer)      [(None, 180)]                0         []                            
                                                                                                  
 html_input (InputLayer)     [(None, 2000)]               0         []                            
                                                                                                  
 url_embedding (Embedding)   (None, 180, 32)              1216      ['url_input[0][0]']           
                                                                                                  
 html_embedding (Embedding)  (None, 2000, 32)             320000    ['html_input[0][0]']          
                                                                                                  
 conv1d (Conv1D)             (None, 171, 128)             41088     ['url_embedding[0][0]']       
                                                                                                  
 conv1d_1 (Conv1D)           (None, 1991, 128)            41088     ['html_embedding[0][0]']      
                                                                                                  
 max_pooling1d (MaxPooling1  (None, 85, 128)              0         ['conv1d[0][0]']              
 D)                                                                                               
                                                                                                  
 max_pooling1d_1 (MaxPoolin  (None, 995, 128)             0         ['conv1d_1[0][0]']            
 g1D)                                                                                             
                                                                                                  
 flatten (Flatten)           (None, 10880)                0         ['max_pooling1d[0][0]']       
                                                                                                  
 flatten_1 (Flatten)         (None, 127360)               0         ['max_pooling1d_1[0][0]']     
                                                                                                  
 concatenate_layer (Concate  (None, 138240)               0         ['flatten[0][0]',             
 nate)                                                               'flatten_1[0][0]']           
                                                                                                  
 dense1 (Dense)              (None, 128)                  1769484   ['concatenate_layer[0][0]']   
                                                          8                                       
                                                                                                  
 dense2 (Dense)              (None, 64)                   8256      ['dense1[0][0]']              
                                                                                                  
 output_layer (Dense)        (None, 1)                    65        ['dense2[0][0]']              
                                                                                                  
==================================================================================================
Total params: 18106561 (69.07 MB)
Trainable params: 18106561 (69.07 MB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
```

---

## Training

### 1. **Early Stopping**
I use early stopping to prevent overfitting. Training stops if there's no improvement in validation loss for 3 consecutive epochs.

### 2. **Training Data**
The model is trained on URL and HTML content.

### 3. **Target Data**
The model tries to predict whether each URL is spam or legitimate.

### 4. **Validation Data**
We use a separate validation set to monitor the model's performance during training.

### 5. **Epochs and Batch Size**
The model is trained for 25 epochs with a batch size of 8.

### 6. **Callbacks**
Early stopping is implemented as a callback function during training.

---

## Evaluation

### 1. **Model Evaluation**
The model's predictions on the test data are generated and converted to binary form (spam or legitimate).

### 2. **Performance Metrics Calculation**
Accuracy, precision, recall, and F1 score of the model are computed on the test data.

### 3. **Metrics Display**
The calculated metrics are printed for review.

```
Accuracy: 0.9794
Precision: 0.9804
Recall: 0.9781
F1 Score: 0.9792
```

### 4. **Metrics Visualization**
A bar chart is created to visualize the model's performance metrics.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/920edab6-908e-4f24-a095-1a5ce201f109)

### 5. **Confusion Matrix**
A confusion matrix is created to see the model's performance in more detail.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/624fd0ef-06d4-4df3-bd6c-67ad521d3ff8)

### 6. **Plot Training and Validation Loss**
A line graph is created to visualize the model's loss on the training and validation data across the epochs.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/25a48273-0e02-426c-a2be-7fab9777226d)

### 7. **Plot Training and Validation Accuracy**
A line graph is created to visualize the model's accuracy on the training and validation data across the epochs.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/fc62ddbd-f71c-4c1a-916a-e919963e19f0)

---

# Results
Following the evaluation phase, the model was stored securely in Google Drive. Subsequently, this saved model was loaded for testing on a fresh benchmark dataset. The results from this test are presented below.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/1ffba4cd-111e-4ed0-8c7b-ff6a5746b209)

The model demonstrated perfect performance with high accuracy on the benchmark dataset, successfully categorizing websites as either legitimate or spam based on their URLs and HTML content. Despite its complexity as a deep learning model, it achieved a notably low inference time, making predictions for the entire benchmark dataset in just 0.5 seconds. This efficiency makes it an excellent candidate for deployment and practical use.

---

# Room For Improvement
Despite the innovative approach of this model, which leverages both HTML and URL data to predict phishing websites, it isn't without limitations. As the landscape of social engineering, particularly phishing, continues to evolve, cybercriminals are persistently devising new strategies for exploitation. Consequently, this model might fall short when encountering unfamiliar phishing techniques. Moving forward, the goal is to develop a model capable of continuous learning, to better adapt to the ever-changing nature of phishing threats.

---

# References
- Opara, C., Chen, Y., & Wei, B. (2023). Look before you leap: Detecting phishing web pages by exploiting raw URL and HTML characteristics. Expert Systems with Applications, 236, 121183.
- Benavides-Astudillo, E., Fuertes, W., Sanchez-Gordon, S., Nuñez-Agurto, D., & Rodríguez-Galán, G. (2023). A phishing-attack-detection model using natural language processing and deep learning. Applied Sciences, 13(9), 5275. https://doi.org/10.3390/app13095275

---

# Attachment
- [Google Colab](https://colab.research.google.com/drive/1DgbA4Pkel0GFhC84LhMYe1A2hrxnYlL-)
- [Model](https://drive.google.com/file/d/11RDifuDQhTepg7lWhjAWrYOLiIKsc6Pw/view?usp=sharing)
