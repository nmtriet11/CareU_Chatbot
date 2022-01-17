import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import time
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
from numpy.lib.arraysetops import isin
from tensorflow import keras
from datetime import date, datetime
from threading import Thread
from tkinter import *
import helpers as helpers
import pyaudio
import wave
import speech_recognition as sr

class Chatbot():
    def __init__(self, intents_path = 'intents.json', 
                       words_path = 'words.pkl', 
                       classes_path = 'classes.pkl', 
                       model_path = 'chatbot_model.h5'
                ):
        self.intents_path = intents_path
        self.words_path = words_path
        self.classes_path = classes_path
        self.model_path = model_path
        
        self.intents = json.loads(open(intents_path).read())
        self.words = pickle.load(open(words_path, 'rb'))
        self.classes = pickle.load(open(classes_path, 'rb'))
        self.model = keras.models.load_model(model_path)
        

    def reload(self):
        self.words = pickle.load(open(self.words_path, 'rb'))
        self.classes = pickle.load(open(self.classes_path, 'rb'))
        self.model = keras.models.load_model(self.model_path)


    def train(self, ):
        words = []
        classes = []
        documents = []
        ignore_words = ['?', '!']
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:

                # take each word and tokenize it
                w = nltk.word_tokenize(pattern)
                words.extend(w)
                # adding documents
                documents.append((w, intent['tag']))

                # adding classes to our class list
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
        words = sorted(list(set(words)))

        classes = sorted(list(set(classes)))
        pickle.dump(words, open(self.words_path, 'wb'))
        pickle.dump(classes, open(self.classes_path, 'wb'))
        # initializing training data
        training = []
        output_empty = [0] * len(classes)
        for doc in documents:
            # initializing bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            # create our bag of words array with 1, if word match found in current pattern
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])
        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)
        # create train and test lists. X - patterns, Y - intents
        train_x = list(training[:,0])
        train_y = list(training[:,1])
        print("Training data created")
        # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
        # equal to number of intents to predict output intent with softmax
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        #fitting and saving the model
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        model.save(self.model_path, hist)

        print("model created")
        # Reload the files
        self.reload()


    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words


    def bow(self, sentence, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):  
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return(np.array(bag))


    def predict_class(self, sentence):
        # filter out predictions below a threshold
        p = self.bow(sentence)
        if not np.any(p):
            return [{"intent": 'noanswer', "probability": 1.0}]
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list


    def get_response_from_top_predict(self, ints):
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        result = ''
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result


    def get_response_from_tag(self, tag):
        list_of_intents = self.intents['intents']
        result = ''
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result


    def get_response_from_msg(self, msg):
        ints = self.predict_class(msg)
        res = self.get_response_from_top_predict(ints)
        return res


class CareuBot():
    def __init__(self, events_path='./events.json', user_path='./user.json'):
        self.events_path = events_path
        self.user_path = user_path

        self.user = json.loads(open(user_path).read())
        # Events dictionary: {month: {day: [tag1, tag2,...]}}
        self.events = json.loads(open(events_path).read()) 

        self.chatbot = Chatbot()
        self.response_history = {}
        self.action_tags = []
        self.personel_tags = []

        # There are 3 different status: Sad, normal, and happy
        self.change_status_dict = {
            'how_today_sad': 'sad',
            'how_today_normal': 'normal',
            'how_today_happy': 'happy'
        }

        # Status based suggestions
        self.suggest_status = {
            'sad': ['sad_eating', 'sad_drink', 'sad_hangout', 'sad_back_family', 'sad_sharebot_story'],
            'happy': ['happy_song', 'happy_other']
        }

        self.add_unlike_list = ['sad_sharebot_story_no', 'sad_back_family_no']
        self.remove_unlike_list = []

    def change_status(self, new_status, user_id):
        if self.user[user_id]['status'] == new_status:
            return True
        try:
            self.user[user_id]['status'] = new_status
            with open(self.user_path, 'w', encoding='utf-8') as f:
                json.dump(self.user, f, ensure_ascii=False, indent=4)
        except:
            return False

    def change_unlikes(self, action='add', new=None, user_id=None):
        if action == 'add':
            try:
                self.user[user_id]['unlikes'].append(new)
                with open(self.user_path, 'w', encoding='utf-8') as f:
                    json.dump(self.user, f, ensure_ascii=False, indent=4)
                return True
            except:
                return False
        elif action == 'remove':
            try:
                self.user[user_id]['unlikes'] = [x for x in self.user[user_id]['unlikes'] if x != new]
                with open(self.user_path, 'w', encoding='utf-8') as f:
                    json.dump(self.user, f, ensure_ascii=False, indent=4)
                return True
            except:
                return False

    def response_personel_questions(self, tag):
        pass
    
    def respond(self, msg=None, user_id='000001'):
        response_list = []
        if msg:
            ints = self.chatbot.predict_class(msg)
            tag = ints[0]['intent']
            if tag in self.personel_tags:
                res = self.response_personel_questions(tag)
            else:
                res = self.chatbot.get_response_from_tag(tag)
            response = {
                'msg': msg,
                'tag': tag,
                'response': res,
                'action': None
            }
            response_list.append(response)
            if tag in self.action_tags:
                response['action'] = tag

            # Get user status
            if tag in list(self.change_status_dict.keys()):
                self.change_status(self.change_status_dict[tag], user_id)
                temp = self.suggest_status[self.change_status_dict[tag]]
                temp = [x for x in temp if x not in self.user[user_id]['unlikes']]
                if len(temp) == 0:
                    tag = 'inspiration'
                else:
                    tag = random.choice(temp)
                res = self.chatbot.get_response_from_tag(tag)
                response = {
                    'msg': msg,
                    'tag': tag,
                    'response': res,
                    'action': None
                }
                response_list.append(response)
                
            if tag in self.add_unlike_list:
                self.change_unlikes('add', tag[:-3], user_id)
                temp = self.suggest_status[self.user[user_id]['status']]
                temp = [x for x in temp if x not in self.user[user_id]['unlikes']]
                if len(temp) == 0:
                    tag = 'inspiration'
                else:
                    tag = random.choice(temp)
                res = self.chatbot.get_response_from_tag(tag)
                response = {
                    'msg': msg,
                    'tag': tag,
                    'response': res,
                    'action': None
                }
                response_list.append(response)
            elif tag in self.remove_unlike_list:
                self.change_unlikes('remove', tag[-3], user_id)

            try:
                self.response_history[user_id].extend(response_list)
            except:
                self.response_history[user_id] = response_list
            return response_list
        
        # If there's no input msg at all
        # Check birthday / work anniversary
        month = date.today().month
        day = date.today().day
        for tag in ['birthday', 'work_aniversary']:
            if self.user[user_id][tag]['day'] == day and self.user[user_id][tag]['month'] == month:
                response = {
                    'msg': msg,
                    'tag': tag,
                    'response': self.chatbot.get_response_from_tag(tag),
                    'action': None
                }
                response_list.append(response)
        # Check events
        try:
            for event in self.events[month][day]:
                res = self.chatbot.get_response_from_tag(event)
                response = {
                    'msg': '',
                    'tag': event,
                    'response': res,
                    'action': None
                }
                response_list.append(response)
        except:
            pass
        # Greeting, how are you today
        temp_list = ['greeting', 'how_today']
        if len(response_list) != 0:
            temp_list = ['how_today']
        for tag in temp_list:
            res = self.chatbot.get_response_from_tag(tag)
            response = {
                'msg': '',
                'tag': tag,
                'response': res,
                'action': None
            }
            response_list.append(response)
        
        try:
            self.response_history[user_id].extend(response_list)
        except:
            self.response_history[user_id] = response_list
        return response_list

    def run(self, msg=None, id="000001"):
        if msg:
            if msg == 'init':   # What the bot will do when init for the 1st time
                response_list = []
                if self.user[id]['status'] != 'sad':
                    # Check events, first quote of the day,...                
                    month = date.today().month
                    day = date.today().day
                    try:
                        for event in self.events[month][day]:
                            res = self.chatbot.get_response_from_tag(event)
                            response = {
                                'msg': '',
                                'tag': event,
                                'response': res,
                                'action': None
                            }
                            response_list.append(response)
                    except:
                        res = self.chatbot.get_response_from_tag('inspiration')
                        response = {
                            'msg': '',
                            'tag': 'inspiration',
                            'response': res,
                            'action': None
                        }
                        response_list.append(response)
                    self.response_history.extend(response_list)
                    return response_list
                else: # What to do when user are sad?
                    res = self.chatbot.get_response_from_tag('inspiration')
                    response = {
                        'msg': '',
                        'tag': 'inspiration',
                        'response': res,
                        'action': None
                    }
                    response_list.append(response)
                    self.response_history.extend(response_list)
                    return response_list
            else:
                ints = self.chatbot.predict_class(msg)
                tag = ints[0]['intent']
                if tag in self.personel_tags:
                    res = self.response_personel_questions(tag)
                else:
                    res = self.chatbot.get_response_from_tag(tag)
                response = {
                    'msg': msg,
                    'tag': tag,
                    'response': res,
                    'action': None
                }
                if tag in self.action_tags:
                    response['action'] = tag
                self.response_history.append(response)
                return [response]
        

    def save_history(self):
        y = json.dump(self.response_history, indent = 4)
        with open("history.json", "w") as fp:
            fp.write(y)

    def train(self):
        self.chatbot.train()

def test():
    chatbot = CareuBot()
    res = chatbot.respond()
    for r in res:
        print('bot:', r['response'])
    while 1:
        msg = input('>>> ')
        if msg == 'abc':
            break
        res = chatbot.respond(msg)
        for r in res:
            print('bot:', r['response'])