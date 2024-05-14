import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder


# base de treino
treino = pd.read_csv('train.csv')
print(treino.head(3))

profile = ProfileReport(treino, title = "treino_titanic") # descricao rapida dos dados
#profile.to_file("treino_titanic.html")

print(treino.info())
print(treino.dtypes.count())
print(treino.isnull().sum().sort_values(ascending=False).head(5))

print()

# base de teste
teste = pd.read_csv('test.csv')
teste.head(3)

print(teste.info())
print(teste.dtypes.count())
print(teste.isnull().sum().sort_values(ascending=False).head(5))

print()

# tratamento dos dados

print(treino.shape)
print(treino.nunique().sort_values(ascending=False))

# remocao de colunas de texto que impedem a generalizacao para uma boa previsao

treino = treino.drop(['Name','Ticket','Cabin'],axis=1)
teste = teste.drop(['Name','Ticket','Cabin'],axis=1)

# tratamento de idades vazias pela media

treino.loc[treino.Age.isnull(),'Age'] = treino.Age.mean()
teste.loc[teste.Age.isnull(),'Age'] = teste.Age.mean()

# tratamento de embarques vazios pela moda

treino.loc[treino.Embarked.isnull(),'Embarked'] = treino.Embarked.mode()[0]
teste.loc[teste.Embarked.isnull(),'Embarked'] = teste.Embarked.mode()[0]

print()

print(treino.isnull().sum().sort_values(ascending=False).head(5))
print(teste.isnull().sum().sort_values(ascending=False).head(5))

print()

# tratamento da coluna fare dos dados de teste, pois sobrou 1 valor nulo

teste.loc[teste.Fare.isnull(),'Fare'] = teste.Fare.mean()

# tratamento das colunas de texto

print(treino.dtypes == 'object')
print(treino.Sex.value_counts())

# transforma a coluna male/female para 0/1
treino['MaleCheck'] = treino.Sex.apply(lambda x: 1 if x == 'male' else 0)
print(treino[['MaleCheck', 'Sex']].value_counts())

teste['MaleCheck'] = teste.Sex.apply(lambda x: 1 if x == 'male' else 0)

# transforma a coluna embarked em 3 novas colunas com valores 0/1
ohe = OneHotEncoder(handle_unknown='ignore', dtype='int32')
ohe = ohe.fit(treino[['Embarked']])
ohe.transform(treino[['Embarked']]).toarray()
ohe_df = pd.DataFrame(ohe.transform(treino[['Embarked']]).toarray(), columns=ohe.get_feature_names_out())

treino = pd.concat([treino, ohe_df], axis=1)
print(treino[['Embarked', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].value_counts())

ohe_df = pd.DataFrame(ohe.transform(teste[['Embarked']]).toarray(), columns=ohe.get_feature_names_out())
teste = pd.concat([teste, ohe_df], axis=1)

treino = treino.drop(['Sex', 'Embarked'],axis=1)
teste = teste.drop(['Sex', 'Embarked'],axis=1)


# separar em treino e teste

x = treino.drop(['PassengerId','Survived'],axis=1)
y = treino.Survived

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# arvore de classificacao (algoritmo de previsao)

clf_arvore = tree.DecisionTreeClassifier(random_state=42)
clf_arvore = clf_arvore.fit(x_train, y_train)
y_pred_arvore = clf_arvore.predict(x_test)

# k neighbors

clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn = clf_knn.fit(x_train, y_train)
y_pred_knn = clf_knn.predict(x_test)

# regressao logistica

clf_rl = LogisticRegression(random_state=42)
clf_rl = clf_rl.fit(x_train, y_train)
y_pred_rl = clf_rl.predict(x_test)

# avaliando os modelos

print(accuracy_score(y_test, y_pred_arvore))
print(accuracy_score(y_test, y_pred_knn))
print(accuracy_score(y_test, y_pred_rl))

# 0.7491525423728813
# 0.7152542372881356
# 0.8101694915254237 -> melhor valor, logo sera o modelo escolhido

print()

# matriz de confusao

print(confusion_matrix(y_test, y_pred_arvore))
print(confusion_matrix(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_rl))

# previsao dos dados de teste

print(x_train.head(3))
print(teste.head(3))

x_teste = teste.drop("PassengerId", axis=1) #remocao para igualar as bases
y_pred = clf_rl.predict(x_teste)

teste['Survived'] = y_pred
base_envio = teste[['PassengerId','Survived']]
base_envio.to_csv('base_envio2_titanic.csv', index=False) # pontuacao kaggle ->