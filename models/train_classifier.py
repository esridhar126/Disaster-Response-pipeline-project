import sys
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
nltk.download(['punkt', 'wordnet','stopwords'])


def load_data(database_filepath):
    '''
    Load data from database as dataframe
    Input:
        database_filepath: Location of sql database
    Output:
        X: Message data (features)
        y: Categories (target)
 
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('my_message_cat', engine)
    #df.drop('original', axis=1, inplace=True) # A lot of NaN values for this column do dropping it
    #df.dropna(how='any', inplace=True)
    X = df['message']
    y = df.iloc[:,4:]
    
    return X, y


def tokenize(text):
    '''
    input:
        text: Text data for tokenization.
    output:
        tokenized_clean: List of results after tokenization.
    '''
    # remove useless characters 
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    #make all the characters to lower_case
    text = text.lower()

    # Tokenization of words
    words = word_tokenize(text)

    # Renoving of stop words such as the, in, is .....
    stop_word = list(set(stopwords.words('english')))
    words = [x for x in words if x not in stop_word]

    # Lemmatization can be done through both noun and verb
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(x, pos='n').strip() for x in words]
    tokenized_data = [lemmatizer.lemmatize(x, pos='v').strip() for x in lemmatized]
    
    return tokenized_data


def build_model():
    '''
    Run Machine Learning pipeline using ifidf, random forest, and gridsearch
    Input: None
    Output:
        cv: GridSearchCV output
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {'clf__estimator__min_samples_split': [2, 4],
              #'clf__estimator__max_features': ['log2', 'auto', 'sqrt', None],
              #'clf__estimator__criterion': ['gini', 'entropy'],
              #'clf__estimator__max_depth': [None, 25, 50, 100, 150, 200],
             }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, y):
    '''
    Evaluate model performance using test data
    Input: 
        model: Model to be evaluated
        X_test: Test data
        Y_test: Labels for Test data
    Output:
        Report of accuracy and classfication for each category
    '''
    Y_pred = model.predict(X_test)
    
    for o in range(len(y.columns)):
        print(y.columns[o], classification_report(Y_test.iloc[:, x].values, Y_pred[:, x]))
        print(y.columns[o], accuracy_score(Y_test.iloc[:, x].values, Y_pred[:,x]))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: Model to save as pickle file
        model_filepath: path of the output pickle file
    Output:
        Pickle file of model
    '''
    pickle.dump(model, open(model_filepath, "wb"))
    
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
        
        print('Model Buildup')
        model = build_model()
        
        print('Train')
        model.fit(X_train, Y_train)
        
        print('Evaluate')
       # evaluate_model(model, X_test, Y_test, y)

        print('Save\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)


    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()