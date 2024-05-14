import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


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

# checagem das colunas e selecao das numericas

col_treino_nr = treino.columns[treino.dtypes != 'object']
print(col_treino_nr)
treino_nr = treino.loc[:,col_treino_nr] # apenas valores numericos

col_teste_nr = teste.columns[teste.dtypes != 'object']
print(col_teste_nr)
teste_nr = teste.loc[:,col_teste_nr]

print()

# a base de teste nao possui a coluna "survived", pois é exatamente o buscado

x = treino_nr.drop(['PassengerId','Survived'], axis=1) # removida a coluna passengerID para facilitar a previsao
y = treino.Survived

# separar em treino e teste
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

# 0.6169491525423729
# 0.6542372881355932
# 0.7254237288135593 -> melhor valor, logo sera o modelo escolhido

print()

# matriz de confusao

print(confusion_matrix(y_test, y_pred_arvore))
print(confusion_matrix(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_rl))

# previsao dos dados de teste

print(x_train.head(3))
print(teste_nr.head(3))

x_teste = teste_nr.drop("PassengerId", axis=1) #remocao para igualar as bases
y_pred = clf_rl.predict(x_teste)

teste['Survived'] = y_pred
base_envio = teste[['PassengerId','Survived']]
base_envio.to_csv('base_envio_titanic.csv', index=False) # pontuacao kaggle -> 0.66746


