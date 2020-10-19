# Sentiment Analysis
Sentiment analysis (also known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information.

#### Data Source
The repository contains corpus.csv which is being used by sentiment_analyzer.py file.

The data in corpus file has two columns: Text and Category

The repo is using small dataset for training by ML algorithms. But it should be as best as much we have data. Larger data to train, better results we would have.

#### Explanation
sentiment_analyzer.py is using corpus.csv as data source to train data and perform sentiment analysis on it.

It has following main modules:

**pandas:** a software library for data manipulation and analysis

**numpy:** a software library that has support for large, multi-dimentional arrays and matrics, along with a large collection of high-level mathematical functions to operate on these arrays.

**nltk:** Natural Language Toolkit, is a library for symbolic and statistical natural language processing for English written in Python programming language.

**scikit-learn:** a software machine learning library for Python. It features various algorithms like support vector machine, random forests, and k-neighbours, and it also supports Python numerical and scientific libraries like Numpy and SciPy


#### Usage
The sentiment_analyzer.py takes corpus.csv as input using following line of code:
`file_to_read = Path('corpus.csv')`

It reads file using `read_dataset` method where **pandas** is being used to read .csv file with following line of code:

`dataset = pandas.read_csv(file_path, encoding='ISO-8859-1')`

Code removes some invalid characters from datasource using **remove_invalid_characters** method. It is highly recommended to use cleaned data for better results.

It then performs lemmatization on cleaned data using **WordNetLemmatizer** class of nltk module.
After performing lemmatization, code tokenizes text using **CountVectorizer** class of scikit-learn module and converting data to term frequency.

At the end, code applies two ML algorithms to predict result:

1- **MultinomialNB:** The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).

2- **SGDClassifier:** SGDClassifier is a linear classifier which implements regularized linear models with stochastic gradient descent (SGD) learning
 


Overall, code is taking input text from user and after performing sentiment analysis, tells you the category of entered text or mood of user based on entered text.
