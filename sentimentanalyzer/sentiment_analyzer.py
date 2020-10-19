from pathlib import Path

import numpy as np
import pandas
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB


class SentimentAnalyzer:

    def __init__(self):
        self.stop_words = [
            'am', 'few', 'all', 'those', 'been', 'y', 'this', 'hers', 'will', 'doing', 'his', 'has', 'both',
            'isn', 'once', 'between', 'our', 'ours', 'didn', 'from', 'wouldn', 'being', 'against', 'you',
            'him', 'sometime', 'much', 'll', 're', 'd', 'due', 'own', 'now', 'shan', 'very', 'having', 'a',
            'always', 'for', 'we', 'couldn', 'shouldn', 'herself', 's', 'here', 'then', 'as', 'with', 'she',
            'after', 'something', 'should', 'hadn', 'my', 'more', 'is', 'mightn', 'towards', 'have', 'had',
            'are', 've', 'them', 'through', 'further', 'about', 'm', 'aren', 'her', 'haven', 'it', 'below',
            'don', 'or', 'me', 'off', 'so', 'wasn', 'its', 'ma', 'ourselves', 'yourselves', 'was', 'above',
            'because', 'during', 'out', 'again', 'themselves', 'beyond', 'into', 'nor', 'that', 'just',
            'weren', 'same', 'everything', 'too', 'ain', 'and', 'did', 'itself', 'theirs', 'does', 'but',
            'such', 'to', 'mustn', 'already', 'yourself', 'under', 'while', 'were', 'the', 'until', 'some',
            'i', 'down', 'if', 'any', 'yours', 'these', 'by', 't', 'only', 'each', 'hasn', 'there', 'most',
            'in', 'be', 'than', 'of', 'myself', 'soon', 'o', 'over', 'up', 'anymore', 'doesn', 'on', 'needn',
            'before', 'an', 'he', 'they', 'himself', 'your', 'anything', 'other', 'their', 'sometimes', 'at',
        ]

    def read_dataset(self, file_path):
        dataset = pandas.read_csv(file_path, encoding='ISO-8859-1')
        return dataset

    def remove_invalid_characters(self, dataset):
        in_valid_characters = ['#', '$', '>', '<', '+', '-', '/', '*', '(', ')', '%', '&', '|', '\', ''/']
        modified_dataset = pandas.DataFrame(columns=['text', 'category'])
        for text, label in zip(dataset['text'].tolist(), dataset['category'].tolist()):
            for in_valid_character in in_valid_characters:
                text = text.replace(in_valid_character, '')
                label = label.replace(in_valid_character, '')
            modified_dataset = modified_dataset.append({'text': text, 'category': label}, ignore_index=True)
        return modified_dataset

    def perform_lemmatization(self, dataset):
        index = 0
        lemmatizer = WordNetLemmatizer()
        for sentence in dataset['text']:
            sentence_tokens = sentence.split(' ')
            lemmatize_sentence = []
            for token in sentence_tokens:
                lemmatize_token = lemmatizer.lemmatize(token)
                lemmatize_sentence.append(lemmatize_token)

            lemmatize_sentence = ' '.join(lemmatize_sentence)
            dataset['text'][index] = lemmatize_sentence.lower() if lemmatize_sentence else sentence.lower()
            index = index + 1

        return dataset

    def tokenizing_text(self, xtrain):
        count_vect = CountVectorizer(analyzer='word', stop_words=self.stop_words, min_df=1, max_df=.5,
                                     ngram_range=(1, 3))
        X_train_counts = count_vect.fit_transform(xtrain)
        return count_vect, X_train_counts

    def data_to_term_frequency(self, X_train_counts):
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        return tfidf_transformer, X_train_tfidf

    # setting the multinomialNB algorithm on data (train the data)
    def apply_multinomialNB_algo(self, X_train_tfidf, ytrain):
        clf = MultinomialNB().fit(X_train_tfidf, ytrain)
        return clf

    # setting the SGD algorithm on data (train the data)
    def apply_sgd_algorithm(self, X_train_tfidf, ytrain):
        clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42).fit(X_train_tfidf, ytrain)
        return clf

    def predict_result_multinomial(self, clf, count_vect, tfidf_transformer, input_value):
        X_new_counts = count_vect.transform([input_value])
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted_result = clf.predict(X_new_tfidf)
        return predicted_result

    def predict_result_sgd(self, clf, count_vect, tfidf_transformer, input_value):
        X_new_counts = count_vect.transform([input_value])
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted_result = clf.predict(X_new_tfidf)
        return predicted_result

    def result_accuracy(self, predicted_result, ytrain):
        # accuracy = (np.mean(predicted_result == list(set(ytrain)), dtype=np.float64)) * 1000
        # accuracy = (np.mean(predicted_result == ytrain, dtype=np.float64)) * 1000
        # accuracy = 100 if accuracy >= 100 else accuracy  # in case accuracy is 0.1 * 1000
        accuracy = np.mean(predicted_result == ytrain, dtype=np.float64)
        return accuracy


def lemmatize_input(input_value):
    lemmatizer = WordNetLemmatizer()
    sentence_tokens = input_value.lower().split(' ')
    lemmatize_sentence = []
    for token in sentence_tokens:
        lemmatize_token = lemmatizer.lemmatize(token)
        lemmatize_sentence.append(lemmatize_token)
    lemmatize_sentence = ' '.join(lemmatize_sentence)
    return lemmatize_sentence


file_to_read = Path('corpus.csv')
if not file_to_read.exists():
    raise Exception('File not found')

sentiment_analyzer = SentimentAnalyzer()
dataset = sentiment_analyzer.read_dataset(file_to_read)
dataset = sentiment_analyzer.remove_invalid_characters(dataset)
dataset = sentiment_analyzer.perform_lemmatization(dataset)

xtrain = dataset['text'].tolist()
ytrain = dataset['category'].tolist()

count_vect, X_train_counts = sentiment_analyzer.tokenizing_text(xtrain)
tfidf_transformer, X_train_tfidf = sentiment_analyzer.data_to_term_frequency(X_train_counts)
multinomial_algo = sentiment_analyzer.apply_multinomialNB_algo(X_train_tfidf, ytrain)
sgd_algo = sentiment_analyzer.apply_sgd_algorithm(X_train_tfidf, ytrain)

input_value = input('Enter your feeling: ')

if not input_value:
    raise Exception('Input is not provided')

input_text = input_value
input_value = lemmatize_input(input_value)

# main execution
mul_predicted_result = sentiment_analyzer.predict_result_multinomial(multinomial_algo, count_vect, tfidf_transformer,
                                                                     input_value)
sgd_predicted_result = sentiment_analyzer.predict_result_sgd(sgd_algo, count_vect, tfidf_transformer, input_value)
predicted_result_accuracy_mul = sentiment_analyzer.result_accuracy(mul_predicted_result, ytrain)
predicted_result_accuracy_sgd = sentiment_analyzer.result_accuracy(sgd_predicted_result, ytrain)

if predicted_result_accuracy_mul > predicted_result_accuracy_sgd:
    print('\n********** Multinomial Algorithm **********')
    print(f'\nPredicted result of "{input_text}" is "{mul_predicted_result[0]}"')
    print(f'\nAccuracy of predictation: {round(predicted_result_accuracy_mul, 1)}%')
elif predicted_result_accuracy_mul < predicted_result_accuracy_sgd:
    print('\n************** SGD Algorithm **************')
    print(f'\nPredicted result of "{input_text}" is "{sgd_predicted_result[0]}"')
    print(f'\nAccuracy of predictation: {round(predicted_result_accuracy_sgd, 1)}%')
else:
    print('\n********** Multinomial Algorithm **********')
    print(f'\nPredicted result of "{input_text}" is "{mul_predicted_result[0]}"')
    print(f'\nAccuracy of predictation: {round(predicted_result_accuracy_mul, 1)}%')
    print('\n************** SGD Algorithm **************')
    print(f'\nPredicted result of "{input_text}" is "{sgd_predicted_result[0]}"')
    print(f'\nAccuracy of predictation: {round(predicted_result_accuracy_sgd, 1)}%')
