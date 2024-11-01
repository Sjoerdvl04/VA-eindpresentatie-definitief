import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
import seaborn as sns
import folium
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.svm import SVC
from streamlit_option_menu import option_menu

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Introductie", "Versie 1", "Versie 2", "Conclusie", "Bronvermelding"]
    )

if selected == "Introductie":
    st.header('Titanic Kaggle Case', divider='grey')
    st.write("""
De Kaggle Titanic-opdracht is een klassieke beginnersuitdaging binnen de data science wereld. Het doel van deze competitie is om een voorspellend model te bouwen dat kan bepalen of een passagier de scheepsramp van de Titanic in 1912 zou hebben overleefd. Gebaseerd op gegevens zoals leeftijd, geslacht, ticketklasse en andere kenmerken, moeten deelnemers patronen herkennen en analyseren om de overlevingskansen van passagiers te voorspellen.
             """)

    st.image(r"C:\Users\sjoer\OneDrive\Bureaublad\DS Blok1\Introduction to Data Science\Nieuwe map\Titanic_foto.jpg", use_column_width=True)

    st.write("""
Deze opdracht hebben wij al uitgevoerd voor case 1. Dit dashboard vergelijkt de Titanic Case van week 1/2 en een verbeterd model hiervan. Er is geprobeerd om de score van de eerste versie te verbeteren middels geavanceerde technieken in tegenstelling tot de technieken van de eerste twee weken.

### Gebruik van het Dashboard

Selecteer in de sidebar welke blogcategorie je wilt bekijken. Kies, indien mogelijk, uit een van de  secties. Binnen de secties is gebruik gemaakt van een aantal dropdown-menu's om interactieviteit te bieden binnen het dashboard
             """)

# ======================================================================================================================================================================

if selected == "Versie 1":
    def df_clean(df):
        df['Age'].fillna(df['Age'].median(), inplace=True)
            
        df.drop(['Cabin'], axis=1, inplace=True)
            
        df.drop(['Name'], axis=1, inplace=True)
            
        df.drop(['Ticket'], axis=1, inplace=True)

        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
            
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

        leeftijdsklassen = []
        for age in df['Age']:
            if age <= 4:
                leeftijdsklassen.append("Baby")
            elif age > 4 and age <= 12:
                leeftijdsklassen.append("Kind")
            elif age > 12 and age <= 18:
                leeftijdsklassen.append("Tiener")
            elif age > 18 and age <= 35:
                leeftijdsklassen.append("Jong volwassene")
            elif age > 35 and age <= 50:
                leeftijdsklassen.append("Volwassene")
            elif age > 50 and age <= 65:
                leeftijdsklassen.append("Middelbare leeftijd")
            else:
                leeftijdsklassen.append("Senior")
            
        df['Leeftijdsklasse'] = leeftijdsklassen
            
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
                
        return df

    df_train = df_clean(train_data)
    df_test = df_clean(test_data)

    with st.sidebar:
        blog_post_sub_1 = option_menu(
            menu_title=None,
            options=["Exploratory Data Analysis", "Visualisatie", "Model"]
            )

    if blog_post_sub_1 == 'Exploratory Data Analysis':
        st.header('Exploratory Data Analysis', divider='grey')
        st.write("""
Een Exploratory Data Analysis is een essentieel onderdeel van het toepassen van regressie. Hierdoor krijg je een eerste indruk van de variabelen in de dataset. In deze case maken we gebruik van 2 methods om de dataset te bekijken. We kijken hoeveel 'NaN' waarden elke kolom bevat en we kijken naar de verdeling van alle numerieke variabelen                 
                 """)
        st.subheader('Aantal NaN-waarden', divider='grey')
        st.write("""
Het aantal NaN-waarden in een kolom is een belangrijke statisiek. Opties om hier vanaf te komen zijn: NaN-waarden verwijderen, invullen met gemiddelde/mediaan/modus en eventueel per groep. De onderstaande tabel presenteert het aantal NaN-waarden per kolom.
                 """)
        df_train = pd.read_csv('train.csv')
        df_test = pd.read_csv('test.csv')

        nan_waarden_df_1 = pd.DataFrame(df_train.isnull().sum(), columns=['Aantal NaN'])
        st.table(nan_waarden_df_1)

        st.subheader('Verdeling van numerieke variabelen', divider='grey')
        st.write("""
De verdelingen alle numerieke variabelen, kan aangeven of een variabele uitschieters heeft of dat deze op een aparte manier verdeeld is. In de onderstaande tabel zien we de numerieke variabelen en de verdeling hiervan.
                 """)
        verdeling_df_1 = pd.DataFrame(df_train.describe())
        st.table(verdeling_df_1)

        st.subheader('Omgaan met NaN-waarden en het creëren van nieuwe variabelen', divider='grey')
        st.write("""
Hoe er omgegaan wordt met missende waarden kan een hoop invloed hebben op je uiteindelijke model. In deze case hebben wij het volgende gedaan met betrekking op missende waarden en het aanmaken van nieuwe variabelen.
                 """)
        st.markdown("""
- **Verwijderen van kolommen:** De kolommen 'Name', 'Ticket' en 'Cabin' zijn verwijderd vanwege onvoldoende of overbodige informatie. In de kolom 'Fare' zijn de missende waarden ingevuld met de mediaan.
- **NaN-waarden invullen:** In de kolom 'Age' zijn de missende waarden ingevuld met de mediaan vanwege een redelijk normale verdeling. In de kolom 'Embarked' zijn de missende waarden ingevuld met de modus.
- **Nieuwe variabele 'Leeftijdsklasse':** Er wordt een nieuwe variabele aangemaakt genaamd 'Leeftijdsklasse'.
- **Nieuwe variabele 'Familiegrootte':** Er wordt een nieuwe variabele aangemaakt genaamd 'Familysize'.
- **Verwijderen variabele 'SibSp' en 'Parch':** De variabelen 'SibSp' en 'Parch' worden verwijderd uit de dataset, nadat de variabele 'Familysize' is aangemaakt.
                    """)
        st.write("""
Al deze aanpassingen zijn verwerkt in één grootte functie. In het onderstaande blokje code, wordt deze functie gepresenteerd.
                 """)
        st.code("""
def df_clean(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
            
    df.drop(['Cabin'], axis=1, inplace=True)
            
    df.drop(['Name'], axis=1, inplace=True)
            
    df.drop(['Ticket'], axis=1, inplace=True)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
            
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    leeftijdsklassen = []
    for age in df['Age']:
        if age <= 4:
            leeftijdsklassen.append("Baby")
        elif age > 4 and age <= 12:
            leeftijdsklassen.append("Kind")
        elif age > 12 and age <= 18:
            leeftijdsklassen.append("Tiener")
        elif age > 18 and age <= 35:
            leeftijdsklassen.append("Jong volwassene")
        elif age > 35 and age <= 50:
            leeftijdsklassen.append("Volwassene")
        elif age > 50 and age <= 65:
            leeftijdsklassen.append("Middelbare leeftijd")
        else:
            leeftijdsklassen.append("Senior")
            
    df['Leeftijdsklasse'] = leeftijdsklassen
            
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
                
    return df

df_train = df_clean(train_data)
df_test = df_clean(test_data)
                """)

    elif blog_post_sub_1 == 'Visualisatie':
        st.header('Visualisatie', divider='grey')
        st.write("""
Vervolgens moeten er worden gekeken welke variabelen opmerkelijke statistieken vertonen als het gaat om de overlevingskansen. Selecteer een variabelen in het dropdownmenu en analyseer de grafiek.
                 """)

        variabelen_versie_1 = ['Leeftijdsklasse', 'Geslacht', 'Klasse']

        selected_variabele_versie_1 = st.selectbox('Selecteer een variabele', variabelen_versie_1)

        if selected_variabele_versie_1 == 'Leeftijdsklasse':
            st.subheader('Leeftijdsklasse', divider='grey')
            st.write("""
In de onderstaande grafiek zijn de verschillende leeftijdsklassen uitgezet tegen het aantal passagiers. Binnen de leeftijdsklassen zijn er twee staven zichtbaar. Een voor het aantal passagiers die de ramp wel hebben overleefd en een voor het aantal passagiers die de ramp niet hebben overleefd.
                     """)

            order = ["Baby", "Kind", "Tiener", "Jong volwassene", "Volwassene", "Middelbare leeftijd", "Senior"]

            plt.figure(figsize=(10,6))
            sns.countplot(data=df_train, x = "Leeftijdsklasse", hue = "Survived", palette = 'Set2', order = order)

            plt.title("Overleving per leeftijdsklasse")
            plt.xlabel("Leeftijdsklassen")
            plt.ylabel("Aantal passagiers")
            plt.xticks(rotation = 30)
            plt.legend(title="Overleving", labels = ["Niet overleefd", "Overleefd"])

            st.pyplot(plt)
        
        elif selected_variabele_versie_1 == 'Geslacht':
            st.subheader('Geslacht', divider='grey')
            st.write("""
In de onderstaande grafiek zijn de twee geslachten uitgezet tegen het aantal passagiers. Voor beide geslachten is er onderscheid gemaakt tussen het aantal passagiers die de ramp wel hebben overleefd en het aantal passagiers die de ramp niet hebben overleefd.
                     """)

            plt.figure(figsize=(10,6))
            sns.countplot(data=df_train, x = "Sex", hue = "Survived", palette = "Set2")

            plt.title("Overleving per geslacht")
            plt.xlabel("Geslacht")
            plt.ylabel("Aantal passagiers")
            plt.legend(title="Overleving", labels = ["Niet overleefd", "Overleefd"])

            st.pyplot(plt)
        
        elif selected_variabele_versie_1 == 'Klasse':
            st.subheader('Ticketklasse', divider='grey')
            st.write("""
In de onderstaande grafiek zijn de drie ticketklassen uitgezet tegen het aantal passagiers in de vorm van een stapeldiagram. Voor alle drie de klassen wordt het aantal passagiers die de ramp wel en niet hebben overleefd gepresenteerd
                     """)

            kruistabel_klassen = pd.crosstab(df_train["Pclass"], df_train["Survived"])
            kruistabel_klassen.plot(kind='bar', stacked=True, figsize=(10,6), colormap='Set2')

            plt.title("Gestapeld staafdiagram van het aantal overlevende per klasse")
            plt.xlabel("De klassen")
            plt.ylabel("Aantal passagiers")
            plt.xticks(ticks=[0, 1, 2], labels=["Eerste klas", "Tweede klas", "Derde klas"])
            plt.xticks(rotation = 30)
            plt.legend(title="Overleving", labels = ["Niet overleefd", "Overleefd"])

            st.pyplot(plt)

    elif blog_post_sub_1 == 'Model':
        st.header('Model', divider='grey')
        st.write("""
Een bijpassende methode is essentieel om een goed beeld te krijgen van hoe goed jouw model nou eigenlijk is. In deze case hebben wij twee modellen gemaakt. Selecteer in het dropdownmenu een model. Vervolgens wordt de bijbehorende confusion matrix gevisualiseerd.
                 """)
        
        model_versie_1 = ['Np.where', 'Logistieke Regressie']

        selected_model_versie_1 = st.selectbox('Selecteer een model', model_versie_1)

        if selected_model_versie_1 == 'Np.where':
            st.subheader('Np.where', divider='grey')
            st.write("""
Het np.where model is een hand geprogrammeerd model die is opgebouwd aan de hand van een aantal beslissingen op basis van de eerder gemaakte visualisaties. Dit model is wat rechttoe rechttaan, maar wel makkelijk om te begrijpen. De volgorde van de beslissingen die zijn genomen is:
                     """)
            st.markdown("""
- Op het moment dat een passagier zich in de leeftijdsklasse 'Senior' bevindt, dan overleeft hij de ramp niet.
- Op het moment dat een passagier van het vrouwelijke geslacht is, overleeft zij de ramp wel.
- Op het moment dat een passagier jonger is dan 10 en een ticket heeft met klasse 2 of 3 dan overleeft hij de ramp wel.
- Op het moment dat een passagier een ticket heeft in klasse 1, dan overleeft hij de ramp wel.
- Anders overleeft de passagier de ramp niet.
                        """)

            test_data = pd.read_csv('test.csv')
            train_data = pd.read_csv('train.csv')

            df_train = df_clean(train_data)
            df_test = df_clean(test_data)

            df_train['predicted_survival'] = np.where(df_train["Leeftijdsklasse"] == "Senior", 
                                          0, 
                                          np.where(df_train['Sex'] == 'female', 
                                                   1,
                                                   np.where((df_train["Age"] <= 10) & ((df_train["Pclass"] == 2) | (df_train["Pclass"] == 3)), 
                                                            1,
                                                            np.where(df_train['Pclass'] == 1, 
                                                                     1, 
                                                                     0))))
            y = df_train["Survived"]
            acc_np_1 = accuracy_score(y, df_train['predicted_survival'])
            prec_np_1 = precision_score(y, df_train['predicted_survival'])
            rec_np_1 = recall_score(y, df_train['predicted_survival'])
            f1_np_1 = f1_score(y, df_train['predicted_survival'])

            cm_train_np = confusion_matrix(y, df_train['predicted_survival'])

            plt.figure(figsize=(8,6))
            sns.heatmap(cm_train_np, annot=True, fmt='g', cmap='Blues')
            plt.title('Confusion Matrix Np.where model')
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            st.pyplot(plt)

            st.subheader('Kwaliteitsmaat', divider='grey')
            st.write("""
Een passende kwaliteitsmaat is essentieel voor het begrijpen van je resultaat. Zo is niet elke kwaliteitsmaat even representatief voor elk model. Kies een kwaliteitsmaat in het dropdownmenu en zie de score.
                     """)

            kwaliteitsmaat_np_versie_1 = ['Accuracy', 'Precision', 'Recall', 'F1-score']
            selected_kwaliteitsmaat_np_versie_1 = st.selectbox('Selecteer een kwaliteitsmaat', kwaliteitsmaat_np_versie_1)

            if selected_kwaliteitsmaat_np_versie_1 == 'Accuracy':
                st.write("De accuracy score op de train set is: ", acc_np_1, ". Dit betekent ongeveer 75% van de voorspellingen juist was.")
            elif selected_kwaliteitsmaat_np_versie_1 == 'Precision':
                st.write("De precision score op de train set is: ", prec_np_1, ". Dit betekent dat ongeveer 63% van de positieve voorspelling juist was.")
            elif selected_kwaliteitsmaat_np_versie_1 == 'Recall':
                st.write("De recall score op de train set is: ", rec_np_1, ". Dit betekent dat ongeveer 86% van de werkelijke positieven als positief werd geclassificeerd.")
            elif selected_kwaliteitsmaat_np_versie_1 == 'F1-score':
                st.write("De F1-score op de train set is: ", f1_np_1, ". Dit is het gemiddelde van de precision en de recall waarbij de relatieve bijdrage van beide metrieken gelijk is.")

            df_test['predicted_survival'] = np.where(df_test["Leeftijdsklasse"] == "Senior", 
                                          0, 
                                          np.where(df_test['Sex'] == 'female', 
                                                   1,
                                                   np.where((df_test["Age"] <= 10) & ((df_test["Pclass"] == 2) | (df_test["Pclass"] == 3)), 
                                                            1,
                                                            np.where(df_test['Pclass'] == 1, 
                                                                     1, 
                                                                     0))))
            df3 = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": df_test['predicted_survival']})
            #df3.to_csv("Opdracht (2.2) Titanic Case.csv", index = False)
        

        elif selected_model_versie_1 == 'Logistieke Regressie':
            st.subheader('Logistieke Regressie', divider='grey')
            st.write("""
Het logistieke regressie model is gekopiëerd van Youtube (Aladdin Persson). Hierin wordt een basis logistiek regressie model opgezet. Wij hebben onze eigen variabelen aangemaakt en gebruikt en vervolgens dit toegepast op het basis logistieke regressie model van Persson.
                     """)
            le = preprocessing.LabelEncoder()

            cols = ["Sex", "Leeftijdsklasse", "Embarked"]

            for col in cols:
                df_train[col] = le.fit_transform(df_train[col])
                df_test[col] = le.transform(df_test[col])
                print(le.classes_)
            
            y = df_train["Survived"]
            X = df_train.drop("Survived", axis=1)

            clf = LogisticRegression(random_state = 0, max_iter = 1000).fit(X, y)

            predictions = clf.predict(X)
            acc_lr_1 = accuracy_score(y, predictions)
            prec_lr_1 = precision_score(y, predictions)
            rec_lr_1 = recall_score(y, predictions)
            f1_lr_1 = f1_score(y, predictions)

            cm_train_lg = confusion_matrix(y, predictions)

            plt.figure(figsize=(8,6))
            sns.heatmap(cm_train_lg, annot=True, fmt='g', cmap='Blues')

            plt.title('Confusion Matrix Logistieke Regressie model')
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            st.pyplot(plt)

            st.subheader('Kwaliteitsmaat', divider='grey')
            st.write("""
Een passende kwaliteitsmaat is essentieel voor het begrijpen van je resultaat. Zo is niet elke kwaliteitsmaat even representatief voor elk model. Kies een kwaliteitsmaat in het dropdownmenu en zie de score.
                     """)

            kwaliteitsmaat_lg_versie_1 = ['Accuracy', 'Precision', 'Recall', 'F1-score']
            selected_kwaliteitsmaat_lg_versie_1 = st.selectbox('Selecteer een kwaliteitsmaat', kwaliteitsmaat_lg_versie_1)

            if selected_kwaliteitsmaat_lg_versie_1 == 'Accuracy':
                st.write("De accuracy score op de train set is: ", acc_lr_1, ". Dit betekent dat ongeveer 80% van de voorspellingen juist was.")
            elif selected_kwaliteitsmaat_lg_versie_1 == 'Precision':
                st.write("De precision score op de train set is: ", prec_lr_1, ". Dit betekent dat ongeveer 76% van de positieve voorspelling juist was.")
            elif selected_kwaliteitsmaat_lg_versie_1 == 'Recall':
                st.write("De recall score op de train set is: ", rec_lr_1, ". Dit betekent dat ongeveer 70% van de werkelijke positieven als positief werd geclassificeerd.")
            elif selected_kwaliteitsmaat_lg_versie_1 == 'F1-score':
                st.write("De F1-score op de train set is: ", f1_lr_1, ". Dit is het gemiddelde van de precision en de recall waarbij de relatieve bijdrage van beide metrieken gelijk is.")

            submission_preds = clf.predict(df_test)
            df2 = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": submission_preds})
            #df2.to_csv("Opdracht (2.1) Titanic Case.csv", index = False)

if selected == "Versie 2":
    def df_clean(df):
        df['Age'].fillna(df['Age'].median(), inplace=True)
            
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

        embarked_mapping = {
                'S': 'Southampton',
                'C': 'Cherbourg',
                'Q': 'Queenstown'}

        df['Embarked'] = df['Embarked'].map(embarked_mapping)
        
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

        leeftijdsklassen = []
        for age in df['Age']:
            if age <= 4:
                leeftijdsklassen.append("Baby")
            elif age > 4 and age <= 12:
                leeftijdsklassen.append("Kind")
            elif age > 12 and age <= 18:
                leeftijdsklassen.append("Tiener")
            elif age > 18 and age <= 35:
                leeftijdsklassen.append("Jong volwassene")
            elif age > 35 and age <= 50:
                leeftijdsklassen.append("Volwassene")
            elif age > 50 and age <= 65:
                leeftijdsklassen.append("Middelbare leeftijd")
            else:
                leeftijdsklassen.append("Senior")
        
        df['Leeftijdsklasse'] = leeftijdsklassen
        
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df.drop(['SibSp', 'Parch'], axis=1, inplace=True, errors='ignore')

        df['is_alleen'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)


        df["Deck"] = df["Cabin"].str.slice(0,1)
        df["Room"] = df["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
        
        df["Deck"] = df["Deck"].fillna("N")
        df["Room"] = df["Room"].fillna(df["Room"].mean())
        df["Room"] = df["Room"].astype(int)

        df['Numeric_ticket'] = df.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
        df['Ticket_letters'] = df.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() if len(x.split(' ')[:-1]) > 0 else 0)
        
        df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme','Mrs')

        df['Getrouwd'] = df['Title'].apply(lambda x: 1 if x == 'Mrs' else 0)
            
        return df

    df_train = df_clean(train_data)
    df_test = df_clean(test_data)

    with st.sidebar:
        blog_post_sub_2 = option_menu(
            menu_title=None,
            options=["Exploratory Data Analysis", "Route", "Visualisatie", "Model"]
            )

    if blog_post_sub_2 == 'Exploratory Data Analysis':
        st.header('Exploratory Data Analysis', divider='grey')
        st.write("""
Het Exploratory Data Analysis gedeelte van de verbeterde versie zal niet veel verschillen met de eerste versie. Het aantal NaN-waarden in de dataset blijft gelijk en de verdelingen van de numerieke variabelen verandert ook niet, daarom zullen de tabellen hier niet nogmaals zichtbaar zijn. Het onderdeel waar beter naar gekeken wordt is het omgaan met NaN-waarden en het creëeren van nieuwe variabelen. Daarnaast is er ook voor elke nieuwe variabele een tabel met het aantal mensen die het per categorie wel en niet overleeft.
                 """)
        st.subheader('Omgaan met NaN-waarden en het creëeren van nieuwe variabelen', divider='grey')
        st.write("""
Zoals eerder vermeld kan de manier van omgaan met missende waarden een hoop invloed hebben op je uitendelijke model. In deze verbeterde versie van de case wordt er beter gekeken naar kolommen die eerder verwijderd werden. Wij hebben het volgende gedaan met betrekking op missende waarden en het aanmaken van nieuwe variabelen:
                 """)
        st.markdown("""
- **NaN-waarden invullen:** In de kolom 'Age' zijn de missende waarden ingevuld met de mediaan vanwege een redelijk normale verdeling. In de kolom 'Embarked' zijn de missende waarden ingevuld met de modus. In de kolom 'Fare' zijn de missende waarden ingevuld met de mediaan.
- **Nieuwe variabele 'Leeftijdsklasse':** Er wordt een nieuwe variabele aangemaakt genaamd 'Leeftijdsklasse'.
- **Nieuwe variabele 'Familiegrootte':** Er wordt een nieuwe variabele aangemaakt genaamd 'Familysize'.
- **Verwijderen variabele 'SibSp' en 'Parch':** De variabelen 'SibSp' en 'Parch' worden verwijderd uit de dataset, nadat de variabele 'Familysize' is aangemaakt.
- **Nieuwe variabele 'Single':** Er wordt een nieuwe variabele aangemaakt op basis van de variabele 'Familiegrootte'. Als de familiegrootte '1' is dan betekent dat dat deze passagier geen familieleden op de cruise mee nam. We beschouden de passagier dan als 'single'.
- **Nieuwe variabele 'Deck':** Er wordt een nieuwe variabele aangemaakt op basis van de variabele 'Cabin'. Hier wordt de letter uit het cabinnummer gehaald om te achterhalen op welk deck de kamer van de passagier was.
- **Nieuwe variabele 'Room':** Er wordt een nieuwe variabele aangemaakt op basis van de variabele 'Cabin'. Hier worden de cijfers uit het cabinnummer gehaald om te achterhalen wat het kamernummer van de passagier was.
- **Nieuwe variabele 'Numeric_ticket:** Er wordt een nieuwe binaire variabele aangemaakt op basis van de variabele 'Ticket'. Deze is '1' als het ticketnummer alleen bestaat uit cijfers. Als het ticketnummer bestaat uit cijfers en letters wordt '0' toegewezen aan passagier in de kolom 'Numeric_ticket'.
- **Nieuwe variabele 'Ticket_letters:** Er wordt een nieuwe variabele aangemaakt op basis van de variabele 'Ticket'. Hier worden de letters uit de variabele 'Ticket' ticket gehaald en toegewezen aan de kolom 'Ticket_letters'.
- **Nieuwe variabele 'Title':** Er wordt een nieuwe variabele aangemaakt op basis van de variabele 'Name' middels een regular expression. De titel van de passagier wordt toegewezen aan deze kolom.
- **Vervangen Titel:** Er worden 3 verschillende titels vervangen door andere titels. Dit komt omdat 'Mlle', 'Ms' en 'Mme' eigenlijk 'Miss', 'Miss' en 'Mrs' betekenen, maar in een andere taal.
- **Nieuwe variabele 'Getrouwd':** Er wordt een nieuwe variabele aangemaakt op basis van de variabele 'Title'. Als de passagier de titel 'Mrs' heeft dan is deze persoon getrouwd.
                    """)
        st.write("""
Al deze aanpassingen zijn verwerkt in één grootte functie. In het onderstaande blokje code, wordt deze functie gepresenteerd.
                 """)
        st.code("""
def df_clean(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
        
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    embarked_mapping = {
        'S': 'Southampton',
        'C': 'Cherbourg',
        'Q': 'Queenstown'}

    df['Embarked'] = df['Embarked'].map(embarked_mapping)
    
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    leeftijdsklassen = []
    for age in df['Age']:
        if age <= 4:
            leeftijdsklassen.append("Baby")
        elif age > 4 and age <= 12:
            leeftijdsklassen.append("Kind")
        elif age > 12 and age <= 18:
            leeftijdsklassen.append("Tiener")
        elif age > 18 and age <= 35:
            leeftijdsklassen.append("Jong volwassene")
        elif age > 35 and age <= 50:
            leeftijdsklassen.append("Volwassene")
        elif age > 50 and age <= 65:
            leeftijdsklassen.append("Middelbare leeftijd")
        else:
            leeftijdsklassen.append("Senior")
    
    df['Leeftijdsklasse'] = leeftijdsklassen
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True, errors='ignore')

    df['is_alleen'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

    df["Deck"] = df["Cabin"].str.slice(0,1)
    df["Room"] = df["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
    
    df["Deck"] = df["Deck"].fillna("N")
    df["Room"] = df["Room"].fillna(df["Room"].mean())
    df["Room"] = df["Room"].astype(int)

    df['Numeric_ticket'] = df.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    df['Ticket_letters'] = df.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() if len(x.split(' ')[:-1]) > 0 else 0)
    
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme','Mrs')

    df_train['Getrouwd'] = df_train['Title'].apply(lambda x: 1 if x == 'Mrs' else 0)
        
    return df

df_train = df_clean(train_data)
df_test = df_clean(test_data)
                """)
        st.subheader('Inzicht overlevende nieuwe variabelen', divider='grey')
        st.write("""
In de onderstaande tabel kan worden gekeken of de nieuwe variabelen daadwerkelijk andere inzichten opleveren. Zien we opmerkelijke statistieken in een bepaalde categorie of was dit te verwachten? Selecteer een variabele in het dropdownmenu en bekijk de tabel. De '0' betekent dat de passagier de ramp niet heeft overleeft, een '1' betekent dat de passagier de ramp wel heeft overleeft.
        """)

        variabelen_EDA_2 = ['Deck', 'Room', 'Numeric ticket', 'Ticket letters', 'Title']
        selected_variabelen_EDA_2 = st.selectbox('Selecteer een variabele', variabelen_EDA_2)

        if selected_variabelen_EDA_2 == 'Deck':
            pivot_tabel_deck = pd.pivot_table(df_train, index='Survived', columns='Deck', values='Name', aggfunc='count')
            st.table(pivot_tabel_deck)
        elif selected_variabelen_EDA_2 == 'Room':
            pivot_tabel_room = pd.pivot_table(df_train, index='Survived', columns='Room', values='Name', aggfunc='count')
            st.table(pivot_tabel_room)
        elif selected_variabelen_EDA_2 == 'Numeric ticket':
            pivot_tabel_numeric_ticket = pd.pivot_table(df_train, index='Survived', columns='Numeric_ticket', values='Name', aggfunc='count')
            st.table(pivot_tabel_numeric_ticket)
        elif selected_variabelen_EDA_2 == 'Ticket letters':
            pivot_tabel_ticket_letter = pd.pivot_table(df_train, index='Survived', columns='Ticket_letters', values='Name', aggfunc='count')
            st.table(pivot_tabel_ticket_letter)
        elif selected_variabelen_EDA_2 == 'Title':
            pivot_tabel_title = pd.pivot_table(df_train, index='Survived', columns='Title', values='Name', aggfunc='count')  
            st.table(pivot_tabel_title)

    elif blog_post_sub_2 == 'Route':
        st.header('Route Titanic', divider='grey')
        st.write("""
Om een beter beeld te krijgen van de route van de Titanic, is deze in kaart gebracht. De markers in Europa zijn de plaatsen waar passagiers zijn opgestapt en de rode marker is de plek waar de Titanic gezonken is. Door op de markers te klikken kom je meer te weten over de gemiddelde overlevingskans of de meest voorkomende ticketklasse. Gebruik de zoom-knop linksboven om in te zoomen of uit te zoomen.
                 """)

        # Coördinaten voor de locaties
        coordinates = {
            'Southampton': (50.9097, -1.4044),
            'Cherbourg': (49.6333, -1.6167),
            'Queenstown': (51.8492, -8.2946)
        }

        # Voorbeeld DataFrame met de gegevens (plaatsvervanger voor jouw df_train)
        data = {
            'LAT/LNG': [(50.9097, -1.4044), (49.6333, -1.6167), (51.8492, -8.2946)],
            'Embarked': ['Southampton', 'Cherbourg', 'Queenstown'],
            'Pclass': [1, 2, 3],
            'Survived': [0.38, 0.55, 0.68],
            'Fare': [30.0, 45.5, 23.0],
            'FamilySize': [1.2, 2.1, 3.5]
        }
        info_map = pd.DataFrame(data)

        # Coördinaten voor de route
        southampton = [50.9097, -1.4044]
        cherbourg = [49.6333, -1.6167]
        route = [49.346043, -6.970773]
        queenstown = [51.8492, -8.2946]
        sinking_site = [41.7325, -49.9469]
        new_york = [40.7128, -74.0060]

        # Folium kaart
        m = folium.Map(location=[45, -35], zoom_start=3)

        # Voeg route toe: Southampton -> Cherbourg -> Queenstown -> Zinklocatie
        folium.PolyLine(locations=[southampton, cherbourg, route, queenstown, sinking_site], 
                        color="blue", 
                        weight=2.5, 
                        smooth_factor=10).add_to(m)

        # Voeg stippellijn toe vanaf de zinklocatie naar New York City
        folium.PolyLine(locations=[sinking_site, new_york], color="red", weight=2.5, dash_array='10').add_to(m)

        # Marker voor de zinklocatie met een kruis
        folium.Marker(
            location=sinking_site,
            icon=folium.Icon(icon='glyphicon-remove', color='red'),
            popup="Zinklocatie Titanic"
        ).add_to(m)

        # Voeg markers toe voor elke locatie in info_map
        for index, row in info_map.iterrows():
            lat, lng = row['LAT/LNG']  # Haal de latitude en longitude uit de tuple
            # HTML layout voor de popup
            html = f"""
                <div style="width:300px;">
                    <table style="width:100%;">
                        <tr><th>Vertrekplaats:</th><td>{row['Embarked']}</td></tr>
                        <tr><th>Klassen:</th><td>{row['Pclass']}</td></tr>
                        <tr><th>Overleefd:</th><td>{row['Survived']:.2f}</td></tr>
                        <tr><th>Ticket prijs:</th><td>{row['Fare']:.2f}</td></tr>
                        <tr><th>Familie grootte:</th><td>{row['FamilySize']:.2f}</td></tr>
                    </table>
                </div>
            """
            popup = folium.Popup(html, max_width=300)
            
            # Marker toevoegen met aangepaste popup
            folium.Marker(location=[lat, lng],  # gebruik afzonderlijke latitude en longitude
                        popup=popup).add_to(m)

        # Weergave van de kaart in Streamlit
        folium_static(m)

    elif blog_post_sub_2 == 'Visualisatie':
        st.header('Visualisatie', divider='grey')
        st.write("""
De volgende stap is kijken of er variabelen zijn met opvallende statistieken. Als eerste kijken we naar een distributieplot van de ticketprijs per klasse. Door een gebied te selecteren in de plot kan je in en uit zoomen. Daarnaast kan je ook op de klassen klikken om deze te verwijderen uit de plot. Door nog een keer te klikken verschijnt de klasse weer in de plot
                 """)
        group_1 = df_train[df_train['Pclass'] == 1]['Fare']
        group_2 = df_train[df_train['Pclass'] == 2]['Fare']
        group_3 = df_train[df_train['Pclass'] == 3]['Fare']

        x = [group_1, group_2, group_3]

        group_labels = ['Ticketklasse 1', 'Ticketklasse 2', 'Ticketklasse 3']

        fig_fare = ff.create_distplot(x, group_labels)

        fig_fare.update_xaxes(title_text='Ticketprijs (Euro)')
        fig_fare.update_yaxes(title_text='Kans')
        fig_fare.update_layout(title='Distributieplot van de ticketprijs per klasse')

        st.plotly_chart(fig_fare)

        sns.set_theme(style='whitegrid')
        custom_palette = sns.color_palette("Set2", 3)

        st.subheader('Verkenning van de variabelen', divider='grey')
        st.write("""
In de eerste versie zijn er maar drie variabelen verkend. In de verbeterede versie worden alle variabelen verkend doormiddel van verschillende plots. Voor de categorische variabelen is een barplot gemaakt. Per categorie is te zien hoeveel passagiers de ramp wel hebben overleeft en hoeveeel passagiers de ramp niet hebben overleeft. Om dit in een ander perspectief te zetten, is er ook een tabel gemaakt met het overlevingspercentage van de desbetreffende categorie. Kies in het dropdownmenu een variabele en ontdek!
        """)

        variabelen_visualisatie_2 = ['Ticketklasse', 'Geslacht', 'Startlocatie', 'Leeftijdsklasse', 'Familygrootte', 'Dek', 'Numeriek ticket', 'Ticket letters', 'Titel', 'Single', 'Getrouwd']
        selected_variabelen_visualisatie_2 = st.selectbox('Selecteer een variabele', variabelen_visualisatie_2)

        if selected_variabelen_visualisatie_2 == 'Dek':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='Deck',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_rate_per_deck = df_train.groupby('Deck')['Survived'].mean()

            st.write(f"Overlevingspercentage per {selected_variabelen_visualisatie_2}:")
            st.table(survival_rate_per_deck)
        
        elif selected_variabelen_visualisatie_2 == 'Ticketklasse':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='Pclass',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_rate_per_Pclass = df_train.groupby('Pclass')['Survived'].mean()

            st.write(f'Overlevingspercentage per {selected_variabelen_visualisatie_2}:')
            st.table(survival_rate_per_Pclass)

        elif selected_variabelen_visualisatie_2 == 'Geslacht':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='Sex',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_Rate_per_Sex = df_train.groupby('Sex')['Survived'].mean()

            st.write(f'Overlevingspercentage per {selected_variabelen_visualisatie_2}:')
            st.table(survival_Rate_per_Sex)

        elif selected_variabelen_visualisatie_2 == 'Startlocatie':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='Embarked',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_rate_per_Embarked = df_train.groupby('Embarked')['Survived'].mean()

            st.write(f'Overlevingspercentage per {selected_variabelen_visualisatie_2}:')
            st.table(survival_rate_per_Embarked)

        elif selected_variabelen_visualisatie_2 == 'Leeftijdsklasse':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='Leeftijdsklasse',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_rate_per_leeftijdsklasse = df_train.groupby('Leeftijdsklasse')['Survived'].mean()

            st.write(f'Overlevingspercentage per {selected_variabelen_visualisatie_2}:')
            st.table(survival_rate_per_leeftijdsklasse)

        elif selected_variabelen_visualisatie_2 == 'Familygrootte':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='FamilySize',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_rate_per_Familysize = df_train.groupby('FamilySize')['Survived'].mean()

            st.write(f'Overlevingspercentage per {selected_variabelen_visualisatie_2}:')
            st.table(survival_rate_per_Familysize)

        elif selected_variabelen_visualisatie_2 == 'Numeriek ticket':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='Numeric_ticket',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_rate_per_Numeric_ticket = df_train.groupby('Numeric_ticket')['Survived'].mean()

            st.write(f'Overlevingspercentage per {selected_variabelen_visualisatie_2}:')
            st.table(survival_rate_per_Numeric_ticket)

        elif selected_variabelen_visualisatie_2 == 'Ticket letters':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='Ticket_letters',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_rate_per_Ticket_letters = df_train('Ticket_letters')['Survived'].mean()

            st.write(f'Overlevingspercentage per {selected_variabelen_visualisatie_2}:')
            st.table(survival_rate_per_Ticket_letters)

        elif selected_variabelen_visualisatie_2 == 'Titel':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='Title',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_rate_per_Title = df_train.groupby('Title')['Survived'].mean()

            st.write(f'Overlevingspercentage per {selected_variabelen_visualisatie_2}:')
            st.table(survival_rate_per_Title)

        elif selected_variabelen_visualisatie_2 == 'Single':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='is_alleen',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_rate_per_single = df_train.groupby('is_alleen')['Survived'].mean()

            st.write(f'Overlevingspercentage per {selected_variabelen_visualisatie_2}:')
            st.table(survival_rate_per_single)

        elif selected_variabelen_visualisatie_2 == 'Getrouwd':
            plt.figure(figsize=(10,6))
            ax = sns.countplot(data=df_train,
            x='Getrouwd',
            hue='Survived',
            palette=custom_palette,
            edgecolor='black'
            )

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0,10),
                textcoords='offset points',
                fontsize=10)
            
            ax.set_xlabel(f'{selected_variabelen_visualisatie_2}', fontsize=12)
            ax.set_ylabel('Aantal passagiers', fontsize = 12)
            ax.set_title(f'Aantal passagiers per {selected_variabelen_visualisatie_2} en Overleving', fontsize=16)

            plt.legend(title='Survived', loc='upper right')
            plt.xticks(rotation = 30)
            plt.tight_layout()
            st.pyplot(plt)

            survival_rate_per_getrouwd = df_train.groupby('Getrouwd')['Survived'].mean()

            st.write(f'Overlevingspercentage per {selected_variabelen_visualisatie_2}:')
            st.table(survival_rate_per_getrouwd)

    elif blog_post_sub_2 == 'Model':
        st.header('Model', divider='grey')
        st.write("""
De laatste stap is het implementeren van het model en de accuracy score analyseren. Hierbij kun je in het dropdownmenu een keuze maken tussen drie verschillende modellen; 'Logistieke Regressie', 'XGBoost model' en 'SVM model'. Als je een model hebt geselecteerd volgt er een korte uitleg met wat voordelen en nadelen. Vervolgens verschijnt de accuracy score van het model op de test set.
                 """)
        st.write("""
De variabelen die uiteindelijk zijn meegenomen zijn: Ticketklasse, Geslacht, Leeftijdsklasse, Familiegrootte, Startlocatie, Dek, Kamer, Titel, Single en Getrouwd.
        """)
        model_versie_2 = ['Logistieke Regressie', 'XGBoost model', 'SVM model']
        selected_model_versie_2 = st.selectbox('Selecteer een model', model_versie_2)

        y = df_train["Survived"]
        X_train = df_train.drop("Survived", axis=1)
        X_test = df_test
        X_train.drop(["LAT/LNG", "Name", "Ticket", "SibSp", "Parch", "Cabin","Ticket_letters", "Numeric_ticket"], 
                    axis=1, inplace=True, errors='ignore')

        X_test.drop(["Name", "Ticket", "SibSp", "Parch", "Cabin", "Ticket_letters", "Numeric_ticket"], 
                    axis=1, inplace=True, errors='ignore')

        # LabelEncoder
        le = preprocessing.LabelEncoder()

        # Combine X_train and X_test to ensure consistent encoding
        combined = pd.concat([X_train, X_test])

        cols = ["Sex", "Embarked", "Leeftijdsklasse", "Deck", "Title"]

        for col in cols:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)

            le.fit(combined[col])  # Fit on the combined data
            X_train[col] = le.transform(X_train[col])  # Transform train set
            X_test[col] = le.transform(X_test[col])  # Transform test set

        if selected_model_versie_2 == 'Logistieke Regressie':

            st.markdown("""
*Logistieke regressie* is een statistische methode die wordt gebruikt voor binaire classificatieproblemen, waarbij de uitkomst slechts twee mogelijke waarden heeft (bijvoorbeeld ja/nee, overleefd/niet overleefd). Het doel is om de kans te voorspellen dat een bepaalde gebeurtenis zich voordoet, op basis van een of meer onafhankelijke variabelen.

Voordelen:
 - **Eenvoudig:** Logistieke regressie is relatief eenvoudig te implementeren en te begrijpen.
 - **Interpretatie:** De coëfficiënten zijn gemakkelijk te interpreteren; ze geven aan hoe veranderingen in de onafhankelijke variabelen de kans op de gebeurtenis beïnvloeden.
 - **Efficiënt**: Het kan goed presteren met kleinere datasets en is minder gevoelig voor overfitting dan complexere modellen.
             
Nadelen:
 - **Lineairerelatie:** Logistieke regressie veronderstelt een lineaire relatie tussen de onafhankelijke variabelen en de log-odds van de afhankelijke variabele.
 - **Beperkingen:** Het kan minder effectief zijn voor zeer complexe relaties of wanneer de klassen niet-lineair gescheiden zijn.
            """)
            accuracy = 0.77751
            st.markdown(
f"""
<div style="border: 2px solid green; background-color: rgba(0, 255, 0, 0.2); padding: 10px; border-radius: 5px; text-align: center;"><h2>Accuracy van het model op test data: {accuracy:.5f}</h2> </div>
""",unsafe_allow_html=True
            )

            
            # Fit the Logistic Regression model
            clf = LogisticRegression(random_state=0, max_iter=2000).fit(X_train, y)

            # Predictions and evaluation
            predictions = clf.predict(X_train)
            accuracy = accuracy_score(y, predictions)
            print(f"Accuracy: {accuracy}")

            # Predictions on test set
            submission_preds = clf.predict(X_test)

            # Prepare submission file
            df4 = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": submission_preds})
            #df4.to_csv("Herkansing_Score_Titanic_Case.csv", index=False)
        
        elif selected_model_versie_2 == 'XGBoost model':
            st.write("""
*XGBoost (Extreme Gradient Boosting)* is een krachtige implementatie van het gradient boosting-framework, dat speciaal is ontworpen voor efficiëntie, snelheid en prestaties. Het wordt vaak gebruikt in machine learning-competities en praktische toepassingen vanwege zijn vermogen om complexe gegevensstructuren goed te modelleren.
             
Voordelen:
 - **Snelheid:** XGBoost is geoptimaliseerd voor snelheid en efficiëntie, waardoor het snel kan worden getraind op grote datasets.
 - **Flexibiliteit:** Het ondersteunt verschillende soorten loss-functies en kan worden aangepast voor specifieke toepassingen.
 - **Prestaties:** Het biedt vaak betere voorspellingen dan traditionele modellen door de geavanceerde boosting-technieken.
             
Nadelen:
 - **Complexiteit:** De configuratie van hyperparameters kan complex zijn, en het kost soms tijd om de beste instellingen te vinden.
 - **Overfitting:** Ondanks de ingebouwde regularisatie kan XGBoost nog steeds overfitten, vooral bij kleine datasets of met te veel complexe bomen.
 - **Gevoeligheid voor outliers:** Het kan minder robuust zijn tegen outliers in de gegevens, wat invloed kan hebben op de prestaties.
            """)
            accuracy = 0.77990
            st.markdown(
f"""
<div style="border: 2px solid green; background-color: rgba(0, 255, 0, 0.2); padding: 10px; border-radius: 5px; text-align: center;"><h2>Accuracy van het model op test data: {accuracy:.5f}</h2> </div>
""",unsafe_allow_html=True
            )

            # Fit the XGBoost model
            clf = XGBClassifier(random_state=0, n_estimators=2000, learning_rate=0.0004)
            clf.fit(X_train, y)

            # Predictions and evaluation
            predictions = clf.predict(X_train)
            accuracy = accuracy_score(y, predictions)
            print(f"Accuracy: {accuracy}")

            # Predictions on test set
            submission_preds = clf.predict(X_test)

            # Prepare submission file
            df5 = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": submission_preds})
            #df5.to_csv("Herkansing_Score_Titanic_Case_XGBoost.csv", index=False)
        
        elif selected_model_versie_2 == 'SVM model':
            st.write("""
*Support Vector Machine (SVM)* is een type machine learning-algoritme dat helpt om dingen in verschillende groepen te verdelen. Het kan gebruikt worden voor zowel het voorspellen van categorieën (classificatie) als voor het schatten van getallen (regressie).

Voordelen:
 - **Effectief bij hoge dimensies:** SVM presteert goed in situaties met een groot aantal kenmerken (features) in vergelijking met het aantal datapunten.
 - **Robuust tegen overfitting:** Met de juiste keuze van 𝐶 en kernel kan SVM effectief omgaan met overfitting, vooral in situaties waar de datapunten goed gescheiden zijn.
 - **Flexibel:** Het kan zowel voor lineaire als niet-lineaire classificatie worden gebruikt door verschillende kernels toe te passen.

Nadelen:
 - **Computationally Intensive:** SVM kan traag zijn bij het trainen op zeer grote datasets, vooral als er veel datapunten zijn.
 - **Kiezen van de juiste kernel:** Het kiezen van de juiste kernel en het afstemmen van hyperparameters (zoals 𝐶 en de kernelparameters) kan tijdrovend zijn en vereist vaak ervaring of experimentatie.
 - **Gevoelig voor outliers:** De prestaties van SVM kunnen worden beïnvloed door outliers, omdat ze de positie van het hypervlak kunnen verstoren.
            """)
            accuracy = 0.77751
            st.markdown(
f"""
<div style="border: 2px solid green; background-color: rgba(0, 255, 0, 0.2); padding: 10px; border-radius: 5px; text-align: center;"><h2>Accuracy van het model op test data: {accuracy:.5f}</h2></div>
""",unsafe_allow_html=True
           )

            svm_model = SVC(kernel='linear', C=1, random_state=0)
            svm_model.fit(X_train, y)

            y_pred = svm_model.predict(X_train)

            accuracy = accuracy_score(y, y_pred)
            print(f"Accuracy van SVM model: {accuracy * 100:.2f}%")

            submission_preds = svm_model.predict(X_test)

            # Prepare submission file
            df6 = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": submission_preds})
            #df6.to_csv("Herkansing_Score_Titanic_Case_SVM.csv", index=False)

# =====================================================================================================================================================================

if selected == "Conclusie":
    st.header('Conclusie van de Titanic Analyse', divider='grey')
    st.markdown("""
In deze analyse van de Titanic-gegevens hebben we verschillende modellen ontwikkeld om de overlevingskansen van passagiers te voorspellen. Om de nauwkeurigheid van onze voorspellingen te verbeteren, hebben we extra variabelen toegevoegd, waaronder:
                
 - **Leeftijdsklasse:** Passagiers werden ingedeeld in leeftijdsgroepen (zoals Baby, Kind, Volwassene), wat inzicht biedt in de demografische samenstelling.
 - **Familiegrootte:** Dit werd berekend op basis van het aantal broers/zus en ouders/voogden, wat kan helpen bij het begrijpen van sociale netwerken aan boord.
 - **Titel:** De titels van passagiers (zoals Mr., Mrs., Miss) bieden context over hun sociale status en huwelijkse staat, wat mogelijk invloed heeft op hun overlevingskansen.
 - **Is alleen:** Was de persoon alleen of niet.
 - **Getrouwd:** Is deze persoon getrouwd aan de hand van de titel. 
 - **Deck:** Aan de hand Cabin een deck bepaald.
 - **Kamernummer:** Aan de hand van Cabin het kamer nummer bepaald. Onbekende Cabin hebben we met mediaan ingevuld. 
                
Deze toevoegingen hebben ons geholpen om drie verschillende modellen te evalueren:

1. **Logistische Regressie**
  - *Train Set Accuracy:* 0,81032
  - *Test Set Accuracy:* 0,77751
                
2. **XGBoost**
  - *Train Set Accuracy:* 0,88552
  - *Test Set Accuracy:* 0,77990
                
3. **Support Vector Machine (SVM)**
  - *Train Set Accuracy:* 0,8070
  - *Test Set Accuracy:* 0,77751
                
**Logistische Regressie** toonde een redelijke prestatie met een testsetaccuracy van 0,77751. Dit model is eenvoudig te interpreteren en biedt waardevolle inzichten in de impact van verschillende variabelen op de overlevingskansen.

**XGBoost** overtrof de andere modellen met de hoogste train accuracy van 0,88552 en een testsetaccuracy van 0,77990. Dit algoritme is krachtiger in het vastleggen van niet-lineaire relaties en interacties tussen variabelen, wat waarschijnlijk bijdraagt aan de betere prestaties.

**SVM** had een testsetaccuracy van 0,77751, gelijk aan de logistische regressie, maar met een iets lagere train accuracy van 0,8070. Dit kan erop wijzen dat het SVM-model mogelijk meer complexiteit mist in de training, of dat het gevoeliger is voor de schaling van de variabelen.

Al met al laat deze analyse zien dat het toevoegen van extra variabelen nuttig is voor het verbeteren van de voorspellingskracht van de modellen. Ondanks dat XGBoost de beste prestaties vertoonde, was de verbetering in testaccuracy relatief klein in vergelijking met de andere modellen. Dit suggereert dat er nog ruimte is voor verdere optimalisatie, bijvoorbeeld door hyperparameter-tuning, het stacken van verschillende modellen of het verkennen van andere machine learning-technieken. De resultaten benadrukken ook het belang van een goede dataverwerking en het selecteren van relevante kenmerken voor het bouwen van effectieve voorspellingsmodellen.
    """)

    st.subheader('Aanbevelingen',divider='grey')
    st.markdown("""
Voor extra verberingen van de modellen zullen er of extra feature variabelen moeten worden toegevoegd of zal er een beter passend model gemaakt moeten worden. Bij extra feature variabelen kan je denken aan:
  - Socio economische status: Dit kan worden afgeleid door de combinatie van passagiersklasse en het tarief, wat een indicatie geeft van de rijkdom van de passagier.
  - Groepering van familiegrootte: Grote familie 6 of meer, gemiddelde familie 3-6 en kleine familie 1-3
  - Andere groeperingen van leeftijd
                
Aan andere mogeljk beter passende modellen kan je denken aan:
  - Random forest algoritme
  - Cross-validation: Het opsplitsen van je train set in kleinere train set om zo een betere schatting van de modelprestaties te krijgen
  - Model stacking: Als je meerdere modellen heb kan je proberen deze te combineren voor een betere output. Nadeel is alleen dat de rekentijd erg groot wordt
    """)

# ======================================================================================================================================================================

if selected == "Bronvermelding":
    st.header('Bronvermelding', divider='grey')
    st.write("""
Tijdens het maken van dit het algoritme hebben wij ons laten inspireren door wat videos en hebben we gebruik gemaakt van openbare AI. Hierbij een link naar de bronnen die zijn gebruikt:
    """)
    st.markdown("""
-[Klik hier voor de inspiratie voor de verbeterde case](https://www.youtube.com/watch?v=I3FBJdiExcg&t=1244s)

-[Klik hier voor meer inspiratie voor de verbeterde case](https://www.kaggle.com/code/nivedambadipudi/titanic-dataset-simple-way-to-get-an-80)

-[Klik hier voor de inspiratie voor het logistieke regressie model](https://www.youtube.com/watch?v=pUSi5xexT4Q&t=824s)

-[Klik hier om naar de openbare AI te gaan](https://chatgpt.com/)
    """)

# =====================================================================================================================================================================
