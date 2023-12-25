import csv
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


class Chatbot:
    def __init__(self, data_file, application_file):
        self.data = self.load_data(data_file)
        self.applications = []
        self.load_applications(application_file)
        self.topic = None
        self.vectorizer = TfidfVectorizer()
        self.model = None

    def load_data(self, data_file):
        with open(data_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            data = {}
            for row in reader:
                topic = row['Konu']
                if topic not in data:
                    data[topic] = []
                data[topic].append(row)
            return data

    def load_applications(self, application_file):
        try:
            with open(application_file, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.applications.append(row)
        except FileNotFoundError:
            # Dosya bulunamazsa veya ilk defa oluşturuluyorsa hata almayı önlemek için boş bir liste oluşturulur.
            self.applications = []

    def save_applications(self, application_file):
        with open(application_file, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Konu', 'Soru', 'Cevap']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for application in self.applications:
                writer.writerow(application)

    def train_model(self):
        X = []
        y = []
        for topic, questions in self.data.items():
            for question in questions:
                X.append(question['Soru'])
                y.append(topic)

        self.model = make_pipeline(self.vectorizer, MultinomialNB())
        self.model.fit(X, y)

    def start_conversation(self):
        print("Merhaba! Ben bir sanal çalışanım. Yazılım Şirketi iş başvurunuz için size bazı sorular soracağım.")

        # İlk başta isminizi ve bitirdiğiniz bölümü sor
        name = input("İsminiz nedir? ")
        major = input("Bitirdiğiniz bölüm nedir? ")

        # İsim ve bölüm bilgilerini applications listesine ekleyelim
        self.applications.append({'Konu': 'İsim ve Bölüm', 'Soru': 'İsminiz nedir?', 'Cevap': name})
        self.applications.append({'Konu': 'İsim ve Bölüm', 'Soru': 'Bitirdiğiniz bölüm nedir?', 'Cevap': major})

        # Tecrübe, Programlama Dili ve Yabancı Dil bilgilerini sor
        experience = input("Kaç yıl tecrübeniz var? ")
        programming_language = input("Hangi programlama diline hakimsiniz? ")
        language_level = input("Yabancı dil seviyeniz nedir? ")

        # Bilgileri applications listesine ekleyelim
        self.applications.append({'Konu': 'Tecrübe', 'Soru': 'Kaç yıl tecrübeniz var?', 'Cevap': experience})
        self.applications.append({'Konu': 'Programlama Dili', 'Soru': 'Hangi programlama diline hakimsiniz?', 'Cevap': programming_language})
        self.applications.append({'Konu': 'Yabancı Dil', 'Soru': 'Yabancı dil seviyeniz nedir?', 'Cevap': language_level})

        print(f"Merhaba {name}! Şimdi yazılım sektöründe hangi alanda çalışmak istediğinizi soracağım.")
        self.ask_topic()

    def ask_topic(self):
        for index, key in enumerate(self.data, start=1):
            print(f"{index}. {key}")

        choice = input("Hangi alanda iş başvurusu yaptınız? ")

        if choice.isdigit() and 1 <= int(choice) <= len(self.data):
            self.topic = list(self.data.keys())[int(choice) - 1]
            self.ask_questions()
        else:
            print("Geçersiz seçim. Lütfen geçerli bir konu seçin.")
            self.ask_topic()

    def ask_questions(self):
        if self.topic in self.data and len(self.data[self.topic]) >= 4:
            asked_questions = []
            application_data = {'Konu': self.topic, 'Soru': '', 'Cevap': ''}
            for _ in range(4):
                available_questions = [q for q in self.data[self.topic] if q['Soru'] not in asked_questions]
                if not available_questions:
                    print("Bu konuda başka soru bulunmuyor.")
                    break

                question_data = random.choice(available_questions)
                asked_questions.append(question_data['Soru'])

                print(f"{question_data['Soru']}")
                for j in range(1, 5):
                    option = f"Secenek{j}"
                    print(f"{j}. {question_data[option]}")

                choice = input(f"{question_data['Soru']} için bir seçenek girin: ")

                if choice.isdigit() and 1 <= int(choice) <= 4:
                    application_data['Soru'] += f"{question_data['Soru']}:{question_data[option]} "
                    application_data['Cevap'] += f"{question_data[f'Secenek{choice}']} "
                else:
                    print("Geçersiz seçim. Lütfen geçerli bir seçenek girin.")
                    self.ask_questions()

            self.applications.append(application_data)
            self.save_applications("isbasvurulari.csv")
            print("Teşekkürler, iş başvurunuz alınmıştır. İyi günler dileriz.")
        else:
            print("Teşekkürler, iş başvurunuz alınmıştır. İyi günler dileriz.")

    def predict_topic(self, text):
        return self.model.predict([text])[0]

# Chatbot'u başlat
chatbot = Chatbot("veri_seti.csv", "isbasvurulari.csv")
chatbot.train_model()
chatbot.start_conversation()
