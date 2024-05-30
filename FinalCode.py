import praw
import pandas as pd
import datetime
import time
from transformers import pipeline
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg') # how I made this work in PyCharm
import matplotlib.pyplot as plt
import numpy as np

# # Gensim
import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models
from gensim.models.phrases import Phrases, Phraser
from gensim.models.coherencemodel import CoherenceModel

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#spacy
import spacy
nlp = spacy.load("en_core_web_sm")

#sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# below is data for 2021
post21 = "https://www.reddit.com/r/eurovision/comments/nilq57/live_thread_eurovision_2021_grand_final_2100_cest/"
date21 = [2021, 5, 22]
hours21 = {"Cyprus": [21, 11], "Albania": [21, 15], "Israel": [21, 20], "Belgium": [21, 23],
           "Russia": [21, 27], "Malta": [21, 31], "Portugal": [21, 36], "Serbia": [21, 39],
           "United_Kingdom": [21, 46], "Greece": [21, 51], "Switzerland": [21, 54], "Iceland": [21, 58],
           "Spain": [22, 4], "Moldova": [22, 7], "Germany": [22, 12], "Finland": [22, 14],
           "Bulgaria": [22, 23], "Lithuania": [22, 27], "Ukraine": [22, 30], "France": [22, 35],
           "Azerbaijan": [22, 38], "Norway": [22, 42], "Netherlands": [22, 47], "Italy": [22, 52],
           "Sweden": [22, 57], "San_Marino": [23, 0]}
votes21 = {"Italy": 'top5', "Ukraine": "top5", "France": "top5", "Finland": "top5", "Iceland": "top5",
           "Switzerland": "topmid", "Lithuania": "topmid", "Russia": "topmid", "Serbia": "topmid",
           "Greece": "topmid", "Sweden": "topmid", "Moldova": "topmid", "Norway": "topmid",
           "Malta": "lowmid", "Cyprus": "lowmid", "Albania": "lowmid", "Azerbaijan": "lowmid",
           "Bulgaria": "lowmid", "Portugal": "lowmid", "Israel": "lowmid", "San_Marino": "lowmid",
           "Belgium": "bot5", "Netherlands": "bot5", "Spain": "bot5", "Germany": "bot5", "United_Kingdom": "bot5"}
overall21 = {"Italy": 'top5', "France": "top5", "Switzerland": "top5", "Ukraine": "top5", "Iceland": "top5",
           "Finland": "topmid", "Malta": "topmid", "Lithuania": "topmid", "Russia": "topmid",
           "Greece": "topmid", "Bulgaria": "topmid", "Portugal": "topmid", "Moldova": "topmid",
           "Sweden": "lowmid", "Serbia": "lowmid", "Cyprus": "lowmid", "Israel": "lowmid",
           "Norway": "lowmid", "Belgium": "lowmid", "Azerbaijan": "lowmid", "Albania": "lowmid",
           "San_Marino": "bot5", "Netherlands": "bot5", "Spain": "bot5", "Germany": "bot5", "United_Kingdom": "bot5"}

# below is data for 2022
post22 = "https://www.reddit.com/r/eurovision/comments/umdmqi/eurovision_song_contest_2022_grand_final_2100_cest/"
date22 = [2022, 5, 14]
hours22 = {'Czechia': [21, 18], 'Romania': [21, 22], 'Portugal': [21, 26], 'Finland': [21, 30],
           'Switzerland': [21, 35], 'France': [21, 40], 'Norway': [21, 43], 'Armenia': [21, 47],
           'Italy': [21, 57], 'Spain': [22, 1], 'Netherlands': [22, 5], 'Ukraine': [22, 9],
           'Germany': [22, 15], 'Lithuania': [22, 18], 'Azerbaijan': [22, 23], 'Belgium': [22, 27],
           'Greece': [22, 34], 'Iceland': [22, 37], 'Moldova': [22, 42], 'Sweden': [22, 46],
           'Australia': [22, 50], 'United_Kingdom': [22, 58], 'Poland': [23, 4], 'Serbia': [23, 8],
           'Estonia': [23, 13]}
votes22 = {'Ukraine': 'top5', 'Moldova': 'top5', 'Spain': 'top5', 'Serbia': 'top5', 'United_Kingdom': 'top5',
           'Sweden': 'topmid', 'Norway': 'topmid', 'Italy': 'topmid', 'Poland': 'topmid',
           'Estonia': 'topmid', 'Lithuania': 'topmid', 'Greece': 'topmid', 'Romania': 'topmid',
           'Netherlands': 'lowmid', 'Portugal': 'lowmid', 'Finland': 'lowmid', 'Armenia': 'lowmid',
           'Iceland': 'lowmid', 'France': 'lowmid', 'Germany': 'lowmid',
           'Belgium': 'bot5', 'Czechia': 'bot5', 'Azerbaijan': 'bot5', 'Australia': 'bot5', 'Switzerland': 'bot5'}
overall22 = {'Ukraine': 'top5', 'United_kingdom': 'top5', 'Spain': 'top5', 'Sweden': 'top5', 'Serbia': 'top5',
           'Italy': 'topmid', 'Moldova': 'topmid', 'Greece': 'topmid', 'Portugal': 'topmid',
           'Norway': 'topmid', 'Netherlands': 'topmid', 'Poland': 'topmid', 'Estonia': 'topmid',
           'Lithuania': 'lowmid', 'Australia': 'lowmid', 'Azerbaijan': 'lowmid', 'Switzerland': 'lowmid',
           'Romania': 'lowmid', 'Belgium': 'lowmid', 'Armenia': 'lowmid',
           'Finalnd': 'bot5', 'Czechia': 'bot5', 'Iceland': 'bot5', 'France': 'bot5', 'Germany': 'bot5'}

# below is data for 2023
post23 = "https://www.reddit.com/r/eurovision/comments/13gld0d/live_thread_eurovision_song_contest_2023_grand/"
date23 = [2023, 5, 13]
hours23 = {'Austria': [21, 20], 'Portugal': [21, 24], 'Switzerland': [21, 30],
           'Poland': [21, 33], 'Serbia': [21, 38], 'France': [21, 42],
           'Cyprus': [21, 48], 'Spain': [21, 52], 'Sweden': [21, 59],
           'Albania': [22, 3], 'Italy': [22, 8], 'Estonia': [22, 12],
           'Finland': [22, 17], 'Czechia': [22, 22], 'Australia': [22, 26],
           'Belgium': [22, 30], 'Armenia': [22, 37], 'Moldova': [22, 42],
           'Ukraine': [22, 46], 'Norway': [22, 50], 'Germany': [22, 55],
           'Lithuania': [22, 59], 'Israel': [23, 4], 'Slovenia': [23, 7],
           'Croatia': [23, 12], 'United_Kingdom': [23, 15]}
votes23 = {'Finland': 'top5', 'Sweden': 'top5', 'Norway': 'top5', 'Ukraine': 'top5',
           'Israel': 'top5',
           'Italy': 'topmid', 'Croatia': 'topmid', 'Poland': 'topmid', 'Moldova': 'topmid',
           'Albania': 'topmid', 'Cyprus': 'topmid', 'Belgium': 'topmid', 'Armenia': 'topmid',
           'France': 'lowmid', 'Lithuania': 'lowmid', 'Slovenia': 'lowmid', 'Czechia': 'lowmid',
           'Switzerland': 'lowmid', 'Estonia': 'lowmid', 'Australia': 'lowmid', 'Austria': 'lowmid',
           'Portugal': 'bot5', 'Serbia': 'bot5', 'Germany': 'bot5', 'United_Kingdom': 'bot5', 'Spain': 'bot5'}
overall23 = {'Sweden': 'top5', 'Finland': 'top5', 'Israel': 'top5', 'Italy': 'top5',
           'Norway': 'top5',
           'Ukraine': 'topmid', 'Belgium': 'topmid', 'Estonia': 'topmid', 'Australia': 'topmid',
           'Czechia': 'topmid', 'Lithuania': 'topmid', 'Cyprus': 'topmid', 'Croatia': 'topmid',
           'Armenia': 'lowmid', 'Austria': 'lowmid', 'France': 'lowmid', 'Spain': 'lowmid',
           'Moldova': 'lowmid', 'Poland': 'lowmid', 'Switzerland': 'lowmid', 'Slovenia': 'lowmid',
           'Albania': 'bot5', 'Portugal': 'bot5', 'Serbia': 'bot5', 'United_Kingdom': 'bot5', 'Germany': 'bot5'}

# below is data for 2024
post24 = "https://www.reddit.com/r/eurovision/comments/1cpm08a/live_thread_eurovision_song_contest_2024_grand/"
date24 = [2024, 5, 11]
hours24 = {"Sweden": [21, 19], "Ukraine": [21, 23], "Germany": [21, 27], "Luxembourg": [21, 31],
           "Israel": [21, 35], "Lithuania": [21, 39], "Spain": [21, 47], "Estonia": [21, 53],
           "Ireland": [21, 56], "Latvia": [22, 0], "Greece": [22, 4], "United_Kingdom": [22, 8],
           "Norway": [22, 17], "Italy": [22, 21], "Serbia": [22, 26], "Finland": [22, 30],
           "Portugal": [22, 37], "Armenia": [22, 40], "Cyprus": [22, 45], "Switzerland": [22, 52],
           "Slovenia": [22, 56], "Croatia": [23, 4], "Georgia": [23, 8], "France": [23, 11],
           "Austria": [23, 15]}

votes24 = {"Croatia": "top5", "Israel": "top5", "Ukraine": "top5", "France": "top5",
           "Switzerland": "top5",
           "Ireland": "topmid", "Italy": "topmid", "Greece": "topmid", "Armenia": "topmid",
           "Lithuania": "topmid", "Sweden": "topmid", "Cyprus": "topmid", "Estonia": "topmid",
           "Serbia": "lowmid", "Finland": "lowmid", "Latvia": "lowmid", "Luxembourg": "lowmid",
           "Georgia": "lowmid", "Germany": "lowmid", "Portugal": "lowmid",
           "Slovenia": "bot5", "Spain": "bot5", "Austria": "bot5", "Norway": "bot5",
           "United_Kingdom": "bot5"}
overall24 = {"Switzerland": "top5", "Croatia": "top5", "Ukraine": "top5", "France": "top5",
           "Israel": "top5",
           "Ireland": "topmid", "Italy": "topmid", "Armenia": "topmid", "Sweden": "topmid",
           "Portugal": "topmid", "Greece": "topmid", "Germany": "topmid", "Luxembourg": "topmid",
           "Lithuania": "lowmid", "Cyprus": "lowmid", "Latvia": "lowmid", "Serbia": "lowmid",
           "United_Kingdom": "lowmid", "Finland": "lowmid", "Estonia": "lowmid",
           "Georgia": "bot5", "Spain": "bot5", "Slovenia": "bot5", "Austria": "bot5",
           "Norway": "bot5"}


def get_data(post, date, hours, year):
    """
    This function accesses the reddit API
    :param post: variable with link to post
    :param date: date the post is taken from
    :param hours: dictionary with timestamps in GMT+2
    :param year: year for the given data
    :return: dataframe with ID, comments, timestamps in UNIX, and roughly matched countries
    """

    # setting up my ID for the API requests
    reddit = praw.Reddit(
        client_id="Xvp2Om9ZdHPmL6rXqyacwQ",
        client_secret="uQxjMdQJBFxqJ8AFtNKGZE4WMCR42w",
        user_agent="webscrapping app by /u/TortelliniFussili")
    submission = reddit.submission(url=post)

    # get all top comments and open up 32 API requests
    submission.comment_sort = "top"
    submission.comments.replace_more()
    all_comments = submission.comments.list()

    # finding the raw comment text and the timestamps
    comment_body = []
    comment_unix = []
    for comment in all_comments:
        comment_body.append(comment.body)
        unix_time = comment.created_utc
        comment_unix.append(unix_time)

    # process the time and date for each entry, converting it to unix time to match them with the timestamps
    min_max = []
    timestamps = {}
    for timestamp in hours:
        date_hour = date + hours[timestamp]
        d = datetime.datetime(date_hour[0], date_hour[1], date_hour[2], date_hour[3], date_hour[4])
        unix = time.mktime(d.timetuple())
        timestamps[timestamp] = int(unix)
        min_max.append(int(unix))

    # matching the comments to a country
    country = {}
    for value in sorted(comment_unix, reverse=True):
        for times in timestamps:
            if timestamps[times] < value < timestamps[times] + 240:
                country[int(value)] = times

    # putting the dataframe together
    df = pd.DataFrame({"ID": all_comments, "Body": comment_body, "Timestamp": comment_unix})
    df.sort_values(by=["Timestamp"], ascending=False, inplace=True)
    df['country'] = df['Timestamp'].map(country, na_action='ignore')

    # cutting off the dataset to only the relevant parts
    df = df[df.Timestamp > min_max[0] - 200]
    df = df[df.Timestamp < min_max[-1] + 400]

    # saving the dataset
    df.to_csv("raw_" + year + ".csv", encoding="utf-8-sig")


# example call: get_data(post24, date24, hours24, "24")

# at this point we do some manual corrections of our dataset before continuing!

def make_dataset(dataset, votes, year):
    """
    :param dataset: raw dataset created above and manually corrected
    :param votes: dictionary with countries matched to their public vote ranking
    :param year: year for the dataset
    :return: dataframe before with added sentiment scores, length of comments, and public voting ranking
    """

    df = pd.read_csv(dataset)

    # remove the irrelevant comments, ie. those not matched to a country
    df = df.dropna(subset=["country"])

    # loading the sentiment analysis model
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True,
                    max_length=512)
    comments = df['Body'].tolist()

    # doing the sentiment analysis and converting strings to number value (-1, 0, 1)
    SH = []
    for label in pipe(comments):
        if label["label"] == "positive":
            SH.append(1)
        elif label["label"] == "negative":
            SH.append(-1)
        elif label["label"] == "neutral":
            SH.append(0)

    # add sentiment to dataframe
    df['SentimentsH'] = SH

    # restructure dictionary above so it can easily be made into a dataframe
    pub_vot = {'country': [], 'Placement': []}
    for result in votes:
        pub_vot['country'].append(result)
        pub_vot['Placement'].append(votes[result])

    # dataframe with conuntries matched to placements
    df_feature = pd.DataFrame(pub_vot)

    #counting the ammount of comments for each country
    comment_amount = []
    for country in pub_vot['country']:
        comment_amount.append(len(df[df['country'] == country]))

    # adding it to dataframe and making as percentage column
    df_feature['amount'] = comment_amount
    df_feature['per_amount'] = df_feature['amount'] / len(df) * 100

    # merge on country
    df = pd.merge(df, df_feature, on="country")

    # save dataframe
    df.to_csv("moredone_" + year + ".csv", encoding="utf-8-sig")


# below is the code I used to create and combine my three dataframes:

"""
year21 = make_dataset("raw_21.csv", votes21, "21")
year22 = make_dataset("raw_22.csv", votes22, "22")
year23 = make_dataset("raw_23.csv", votes23, "23")

df21 = pd.read_csv("moredone_21.csv")
df22 = pd.read_csv("moredone_22.csv")
df23 = pd.read_csv("moredone_23.csv")
df2122 = pd.read_csv("together.csv")

together = pd.concat([df21, df22])
together2 = pd.concat([together, df23])

together2.to_csv("togetherall.csv", encoding="utf-8-sig")

year24 = make_dataset("raw_24.csv", votes24, "24")
"""

def add_overall(dataset, votes, year):
    """
    This function was to add the absolute placements of countries - added this later, hence the seperate function
    :param dataset: dataset from make_dataset()
    :param votes: dictionary with absolute placements (jury + public)
    :param year: year for data
    :return: a new dataframe with absolute placements added
    """
    df = pd.read_csv(dataset)

    # as above; restructure dictionary to easily create a dataframe
    pub_vot = {'country': [], 'Placement': []}
    for result in votes:
        pub_vot['country'].append(result)
        pub_vot['Placement'].append(votes[result])

    # make and merge dataframes on country
    df_feature = pd.DataFrame(pub_vot)
    df = pd.merge(df, df_feature, on="country")

    #save dataframe
    df.to_csv("bothplacements_" + year + ".csv", encoding="utf-8-sig")

# code below I used to create and merge datasets
"""
#year21 = add_overall("moredone_21.csv", overall21, "21")
#year22 = add_overall("moredone_22.csv", overall22, "22")
#year23 = add_overall("moredone_23.csv", overall23, "23")
#year24 = add_overall("moredone_24.csv", overall24, "24")

#df21 = pd.read_csv("bothplacements_21.csv")
#df22 = pd.read_csv("bothplacements_22.csv")
#df23 = pd.read_csv("bothplacements_23.csv")

#together = pd.concat([df21, df22])
#together2 = pd.concat([together, df23])

#together2.to_csv("togetherallplac.csv", encoding="utf-8-sig")
"""

def prepreprocess(dataset):
    """
    This is the first of two preprocessing function
    :param dataset: dataset output from add_overall()
    :return: dataframe with preprocessed data 1.0
    """
    df = pd.read_csv(dataset)

    # make comments into a list to more easily parse them
    comments = df['Body'].tolist()

    # i've lemmatised and gotten rid of pos seperately because otherwise "don't" etc is not lemmatized properly and "not" is lost

    # lemmatisation
    data_ready = []
    for sent in comments:
        obj = nlp(sent)
        data_ready.append([token.lemma_ for token in obj])

    # make lemmatised objects into one text so I didn't have to change syntax
    data_ready2 = []
    for entry in data_ready:
        doc = nlp(" ".join(entry))
        data_ready2.append(str(doc))

    # only keep object with allowed POS tags or "not"
    allowed = ['NOUN', 'ADJ', 'VERB', "ADV"]
    data_ready3 = []
    for sent in data_ready2:
        obj = nlp(sent)
        data_ready3.append([token.text for token in obj if token.pos_ in allowed or token.text == "not"])

    # collect tokens into string again
    data_ready4 = []
    for entry in data_ready3:
        doc = nlp(" ".join(entry))
        data_ready4.append(str(doc))

    # make dataframe with preprocessed comments. Delete empty comments
    df['text'] = data_ready4
    df = df.dropna(subset=["text"])
    df = df.drop(df[df['text'] == ' '].index)

    return df

#example: prepreprocess("moredone_24.csv")

def preprocessing(dataset):
    """
    Second preprocessing function
    :param dataset:
    :return: list where each comment is a list with tokens in seperate strings
    """
    df = prepreprocess(dataset)
    data = df['text'].tolist()

    # simple preprocessing based on gensim's own preprocessing tool
    data_ready = []
    for comment in data:
        sent = gensim.utils.simple_preprocess(str(comment), deacc=True)
        data_ready.append(sent)

    # adding words to stopwords expected to not be important. Adding "delete" to make sure deleted comments are removed
    stop_words.extend(["eurovision", "song", "delete"])

    # exclude stoplist words
    data_ready2 = []
    for sent in data_ready:
        sents = []
        for word in sent:
            if word not in stop_words or word == "not":
                sents.append(word)
        data_ready2.append(sents)

    #create bigrams
    phrases = Phrases(data_ready2, min_count=1, threshold=5)
    bigrams = Phraser(phrases)

    #bigram data
    bigram = []
    for sent in data_ready2:
        sents = []
        for word in bigrams[sent]:
            sents.append(word)
        bigram.append(sents)

    return bigram

# below is code I used to check statistics on how long comments were
"""
#loading data
testing = preprocessing("moredone_24.csv")
train = preprocessing("togetherallplac.csv")
data_ready = train + testing

len_com = 0
one_counts = 0
two_to_seven = 0
eight_above = 0
for comment in data_ready:
    len_com += len(comment)
    if len(comment) == 1:
        one_counts += 1
    if 2 >= len(comment) and len(comment) <= 7:
        two_to_seven += 1
    if len(comment) >= 8:
        eight_above += 1

#overall average length        
print(len_com/len(data_ready))

#number of comments with certain counts
print(one_counts)
print(two_to_seven)
print(eight_above)
"""

def vectors(dataset, test):
    """
    This function creates the Topic Modelling vectors
    :param dataset: training data (2021, 2022, 2022 concatenated)
    :param test: testing data (2024)
    :return: vectors
    """

    # load data
    train = preprocessing(dataset)
    testing = preprocessing(test)

    #combine data to make consistent topics
    data_ready = train + testing


    # training the lda. update_every = 0, meaning batch learning
    id2word = corpora.Dictionary(data_ready)
    corpus = [id2word.doc2bow(text) for text in data_ready]
    lda_train = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                alpha='auto',
                                                update_every= 0)

    #Load the datasets
    train = pd.read_csv(dataset)
    test = pd.read_csv(test)
    rev_train = pd.concat([train, test])

    # add sentiment scores and comment length to each vector
    train_vecs = []
    for i in range(len(rev_train)):
        top_topics = lda_train.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(10)]
        topic_vec.extend([rev_train.iloc[i].SentimentsH])
        topic_vec.extend([rev_train.iloc[i].per_amount])
        train_vecs.append(topic_vec)


    # checking and visualizing topics
    #print(lda_train.print_topics(num_topics=10, num_words=10))
    #vis = pyLDAvis.gensim_models.prepare(lda_train, corpus, id2word)
    #pyLDAvis.save_html(vis, 'vis4.html')

    # return training vectors
    return train_vecs


def classifier_tm(dataset, test):
    """
    Classifier using the topic modelling vectors
    :param dataset: training data (2021, 2022, 2022 concatenated)
    :param test: testing data (2024)
    :return: list of predictions
    """

    # load data
    dataset_read = pd.read_csv(dataset)
    testing_read = pd.read_csv(test)

    # load the vectors
    data_tm = np.array(vectors(dataset, test))

    # seperate the vectors into training and test vectors
    len_test = len(testing_read)
    len_train = len(dataset_read)
    testing = data_tm[-int(len_test):]
    data = data_tm[:int(len_train)]

    # get true labels (change Placement_x = public ranking, Placement_y = absolute ranking)
    target = dataset_read['Placement_y'].to_numpy()
    target_test = testing_read['Placement_y'].to_numpy()

    # train classifier using training vectors and labels
    model = GaussianNB()
    model.fit(data, target)

    # make predictions based on test data
    expected = target_test
    predicted = model.predict(testing)

    #summarize results
    print(classification_report(expected, predicted))
    print(confusion_matrix(expected, predicted))

    # get list of predictions
    predictions = model.predict(testing)

    return predictions

def classifier_tfidf(dataset, test):
    """
    Makes tfidf vectors, trains and tests the classfier for tfidf vectors
    :param dataset: training data (2021, 2022, 2022 concatenated)
    :param test: testing data (2024)
    :return: list of predictions
    """

    # load preprocessed data for training data
    train_data = preprocessing(dataset)
    # join tokens together to make full strings for each comment
    train_data2 = []
    for entry in train_data:
        doc = nlp(" ".join(entry))
        train_data2.append(str(doc))

    # repeat above for test data
    test_data = preprocessing(test)
    test_data2 = []
    for entry in test_data:
        doc = nlp(" ".join(entry))
        test_data2.append(str(doc))

    # train tfidf vectorizer
    vectorizer = TfidfVectorizer()
    data = vectorizer.fit_transform(train_data2).toarray()
    # fit testing data to fitted vectorizer
    testing = vectorizer.transform(test_data2).toarray()

    dataset_read = pd.read_csv(dataset)
    testing_read = pd.read_csv(test)

    # add sentiments and length of comment to all vectors for training data
    data_list = data.tolist()
    data_addon = []
    for i in range(len(dataset_read)):
        vectors = data_list[i]
        vectors.extend([dataset_read.iloc[i].SentimentsH])
        vectors.extend([dataset_read.iloc[i].per_amount])
        data_addon.append(vectors)
    data_addon = np.array(data_addon)

    # the same as above for test data
    test_list = testing.tolist()
    testing_addon = []
    for i in range(len(testing_read)):
        vectors = test_list[i]
        vectors.extend([testing_read.iloc[i].SentimentsH])
        vectors.extend([testing_read.iloc[i].per_amount])
        testing_addon.append(vectors)

    testing_addon = np.array(testing_addon)

    # get true labels (change Placement_x = public ranking, Placement_y = absolute ranking)
    target = dataset_read['Placement_y'].to_numpy()
    target_test = testing_read['Placement_y'].to_numpy()

    # get true labels (change Placement_x = public ranking, Placement_y = absolute ranking)
    target = dataset_read['Placement_y'].to_numpy()
    target_test = testing_read['Placement_y'].to_numpy()

    # train classifier using training vectors and labels
    model = GaussianNB()
    model.fit(data, target)

    # make predictions based on test data
    expected = target_test
    predicted = model.predict(testing)

    # summarize results
    print(classification_report(expected, predicted))
    print(confusion_matrix(expected, predicted))

    # get list of predictions
    predictions = model.predict(testing)

    return predictions

def results(dataset, test):
    """
    Decides on how ccountries will be placed based on classifier results
    :param dataset: training data (2021, 2022, 2022 concatenated)
    :param test: testing data (2024)
    :return: Predictions
    """

    # load data and predictions
    df = pd.read_csv(test)
    predictions = classifier_tm(dataset, test)

    # add a predictions column to dataframe
    df["Predictions"] = predictions

    # make a list with only countries and the predictions, fill in with the number of countries in each placement category
    df_select = df[["Predictions", "country"]]
    df_pivot = pd.pivot_table(df_select, index='country',
                              columns=['Predictions'], aggfunc=len, fill_value=0)
    df_pivot.reset_index(inplace=True)

    # Converting the raw counts to percentages of whole year
    unique_predictions = df_select['Predictions'].unique()
    df_sums = df_pivot[unique_predictions]
    sums = []
    for i in range(len(df_sums)):
        sum = df_sums.iloc[i].sum(axis=0)
        sums.append(sum)
    result_df = df_sums.div(sums, axis=0)
    result_df = result_df.mul(100, axis=0)

    country = df_pivot["country"].tolist()
    result_df["country"] = country

    # print out table showing percentage of comments in each category
    print(result_df)

    # selecting top comments from categories and deleting them from dataset as they are deleted
    dict_results = {}

    top_5 = result_df.nlargest(5, "top5")
    top = top_5["country"].tolist()
    for country in top:
        result_df = result_df.drop(result_df[result_df["country"] == country].index)
    dict_results["top5"] = top

    topmid = result_df.nlargest(8, "topmid")
    topmid = topmid["country"].tolist()
    for country in topmid:
        result_df = result_df.drop(result_df[result_df["country"] == country].index)
    dict_results["topmid"] = topmid

    lowmid = result_df.nlargest(7, "lowmid")
    lowmid = lowmid["country"].tolist()
    for country in lowmid:
        result_df = result_df.drop(result_df[result_df["country"] == country].index)
    dict_results["botmid"] = lowmid

    # bot5 category is just leftover countries
    bot5 = result_df["country"].tolist()
    dict_results["bot5"] = bot5

    # print our predicted rankings
    print(dict_results)

# example to get results: results("togetherallplac.csv", "bothplacements_24.csv")


# below are functions not in the main pipeline but used to improve or check model

def grid_search(dataset, test):
    """
    Function used to check for ideal number of topics by looking at coherence scores
    :param dataset: training data (2021, 2022, 2022 concatenated)
    :param test: testing data (2024)
    :return: graph showing coherence scores by topics and time taken to calculate them
    """
    # loading and combining data
    train = preprocessing(dataset)
    testing = preprocessing(test)
    data_ready = train + testing

    # create dictionary and corpus
    id2word = corpora.Dictionary(data_ready)
    corpus = [id2word.doc2bow(text) for text in data_ready]

    #dictionary used to make graph
    perplexities = {'topics': [], 'coherence': [], 'runtime': []}

    #train lda models. Use topics between 3 and 50, skipping every second to speed up
    for i in range(3,50, 2):
        # check time when starting
        start = datetime.datetime.now()
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=i,
                                                alpha='auto',
                                                update_every= 0)

        # check time when ending to see how long it took
        run_time = (datetime.datetime.now() - start).seconds

        # add time taken, cohernece score, and number of topics to dictionary above
        perplexities['topics'].append(i)
        coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
        perplexities['coherence'].append(coherence_model_lda.get_coherence())
        perplexities['runtime'].append(run_time)

    # make df of dictionary
    df = pd.DataFrame(perplexities)
    print(df)

    # plot graph
    sns.lineplot(data = df, y='coherence', x = 'topics')
    ax2 = plt.twinx()
    sns.lineplot(data=df, y='runtime', ax=ax2, x = 'topics', color = 'r')
    plt.show()

def finished_data(dataset, votes, year):
    """
    This is an older function used to create a singular dataset made for the sentiment x length graph. Very similar to make_dataset()
    :param dataset: raw dataset created above and manually corrected
    :param votes: dictionary with countries matched to their public vote ranking
    :param year: year for the dataset
    :return: dataframe which has each performance as a single row with sentiment score and length of comment
    """

    # loading dataset with sentiment scores
    raw_df = pd.read_csv(dataset)

    # restructuring dictionary
    pub_vot = {'country': [], 'Placement': []}
    for result in votes:
        pub_vot['country'].append(result)
        pub_vot['Placement'].append(votes[result])

    #dictionary into dataframe
    processed_df = pd.DataFrame(pub_vot)

    # averaging sentiment scores
    processed_df2 = raw_df.groupby('country')['SentimentsH'].mean().reset_index()

    # merging sentiment scores into previous dataframe
    processed_df = pd.merge(processed_df, processed_df2, on="country")

    # getting numbers of comments for each country
    comment_amount = []
    for country in pub_vot['country']:
        comment_amount.append(len(raw_df[raw_df['country'] == country]))

    # add comment amoutn raw and as a percentage to dataframe
    processed_df['amount'] = comment_amount
    processed_df['per_amount'] = processed_df['amount'] / len(raw_df) * 100

    # save dataframe
    processed_df.to_csv("final_" + year + ".csv")

# below code was used to create figure with avergae sentiment scores and percentage of comment amounts
"""
df21 = pd.read_csv("final_21.csv")
df22 = pd.read_csv("final_22.csv")
df23 = pd.read_csv("final_23.csv")
df24 = pd.read_csv("final_24.csv")
together = pd.concat([df21, df22])
together = pd.concat([together, df23])
together = pd.concat([together, df24])

sns.scatterplot(data = together, x = 'SentimentsH', y = 'per_amount', hue = 'Placement')
plt.show()
"""