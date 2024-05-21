import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Visualizando a base de treino

treino = pd.read_csv('train.csv')

print(treino.head(3))

# Visualizando a base de teste

teste = pd.read_csv('test.csv')

print(teste.head(3))

# ---------------------------------------------- #

# Tratamento
# Eliminando as colunas com elevada cardinalidade

treino = treino.drop(['Name','Ticket','Cabin'],axis=1)

teste = teste.drop(['Name','Ticket','Cabin'],axis=1)

# Usando a média para substituir valores nulos na coluna de idade

treino.loc[treino.Age.isnull(),'Age'] = treino.Age.mean()

teste.loc[teste.Age.isnull(),'Age'] = teste.Age.mean()

# Tratando a coluna Embarked da base de treino usando a moda

treino.loc[treino.Embarked.isnull(),'Embarked'] = treino.Embarked.mode()[0]

# E também a coluna Fare da base de teste usando a média

teste.loc[teste.Fare.isnull(),'Fare'] = teste.Fare.mean()

# ---------------------------------------------- #

# Engenharia de recursos
# Usando uma lambda function para tratar a coluna "Sex"

treino['MaleCheck'] = treino.Sex.apply(lambda x: 1 if x == 'male' else 0)

teste['MaleCheck'] = teste.Sex.apply(lambda x: 1 if x == 'male' else 0)

# Fazendo o RobustScaler das colunas Age e Fare

transformer = RobustScaler().fit(treino[['Age','Fare']])

treino[['Age','Fare']] = transformer.transform(treino[['Age','Fare']])

# e para a base de teste

transformer = RobustScaler().fit(teste[['Age','Fare']])

teste[['Age','Fare']] = transformer.transform(teste[['Age','Fare']])

# Adicionando a coluna sozinho

def sozinho(a,b):

    if (a == 0 and b == 0):

        return 1

    else:

        return 0

treino['Sozinho'] = treino.apply(lambda x: sozinho(x.SibSp,x.Parch),axis=1)

teste['Sozinho'] = teste.apply(lambda x: sozinho(x.SibSp,x.Parch),axis=1)

# E criando a coluna de familiares

treino['Familiares'] = treino.SibSp + treino.Parch

teste['Familiares'] = treino.SibSp + treino.Parch

# Fazendo o OrdinalEncoder para a coluna Embarked

categorias = ['S','C','Q']

enc = OrdinalEncoder(categories=[categorias],dtype='int32')

enc = enc.fit(treino[['Embarked']])

treino['Embarked'] = enc.transform(treino[['Embarked']])

teste['Embarked'] = enc.transform(teste[['Embarked']])

# Apagando as colunas de texto

treino = treino.drop('Sex',axis=1)

teste = teste.drop('Sex',axis=1)

# Visualizando a base resultante

# Visualizando a base de treino

treino.head(3)

# ---------------------------------------------- #
# Previsao
# Separando a base de treino em X e y

X = treino.drop(['PassengerId','Survived'],axis=1)

y = treino.Survived

# Separando em treino e validação

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Regressão Logística
# Criando o classificador

clf_rl = LogisticRegression(random_state=42)

# Definindo os parâmetros

parametros_rl = {

    'penalty': ['l1','l2'],

    'C': [0.01,0.1,1,10],

    'solver': ['lbfgs','liblinear','saga'],

    'max_iter': [100,1000,5000,10000]

}
# ---------------------------------------------- #

# Random Forest
# Criando o classificador

clf_rf = RandomForestClassifier(random_state=42)

parametros_rf = {

    'n_estimators': [100,200,500,1000],

    'criterion': ['gini','entropy','log_loss'],

    'max_depth': [2,4,6,8,None],

    'max_features': ['sqrt','log2',None]

}

# MLPClassifier (Redes Neurais)
# Criando o classificador

clf_mlp = MLPClassifier(random_state=42)

parametros_mlp = {

    'solver':  ['lbfgs','sgd','adam'],

    'alpha': [10.0**(-1),10.0**(-5),10.0**(-7),10.0**(-10)],

    'max_iter': [200,500,1000,5000]

}

# Fazendo o grid_search


warnings.filterwarnings('ignore')

def hora_atual():

    agora = datetime.now()

    print(str(agora.hour)+':'+str(agora.minute)+":"+str(agora.second))


# Para a Regressão Logística

hora_atual()

kfold_rl = KFold(shuffle=True,random_state=42,n_splits=8)

grid_search_rl = GridSearchCV(clf_rl, parametros_rl,scoring='accuracy',cv=kfold_rl)

grid_search_rl = grid_search_rl.fit(X_train,y_train)

hora_atual()

# Para o RandomForest

hora_atual()

kfold_rf = KFold(shuffle=True,random_state=42,n_splits=8)

grid_search_rf = GridSearchCV(clf_rf, parametros_rf,scoring='accuracy',cv=kfold_rf)

grid_search_rf = grid_search_rf.fit(X_train,y_train)

hora_atual()

# Para o MLPClassifier

hora_atual()

kfold_mlp = KFold(shuffle=True,random_state=42,n_splits=8)

grid_search_mlp = GridSearchCV(clf_mlp, parametros_mlp,scoring='accuracy',cv=kfold_mlp)

grid_search_mlp = grid_search_mlp.fit(X_train,y_train)

hora_atual()

#Verificando os melhores scores

# Verificando o melhor score da regressão logística

print(grid_search_rl.best_score_)

#0.8089887640449438

# Para o RandomForest

print(grid_search_rf.best_score_)

#0.8314606741573034

# e para o MLPClassifier

print(grid_search_mlp.best_score_)

#0.8174157303370786

# Verificando os melhores parâmetros da regressão logística

grid_search_rl.best_params_

{'C': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'lbfgs'}

# Para o RandomForest

grid_search_rf.best_params_
{
    'criterion': 'entropy',
    'max_depth': 6,
    'max_features': 'sqrt',
    'n_estimators': 100
 }

# e para o MLPClassifier

grid_search_mlp.best_params_
{
    'alpha': 0.1,
    'max_iter': 200,
    'solver': 'adam'
}


# Avaliando os modelos
# Para a regressão logística

clf_best_rl = grid_search_rl.best_estimator_

y_pred_rl = clf_best_rl.predict(X_val)

# Para o RandomForest

clf_best_rf = grid_search_rf.best_estimator_

y_pred_rf = clf_best_rf.predict(X_val)

# e para o MLPClassifier

clf_best_mlp = grid_search_mlp.best_estimator_

y_pred_mlp = clf_best_mlp.predict(X_val)

# Para a Regressão Logística

print(accuracy_score(y_val, y_pred_rl))

#0.8044692737430168

# Para o Random Forest

print(accuracy_score(y_val, y_pred_rf))

#0.8100558659217877

# Para o MLPClassifier (Redes Neurais)

print(accuracy_score(y_val, y_pred_mlp))


# Avaliando a matriz de confusão
# Para a Regressão Logística

print(confusion_matrix(y_val, y_pred_rl))

# Para o Random Forest

print(confusion_matrix(y_val, y_pred_rf))

# Para o MLPClassifier (Redes Neurais)

print(confusion_matrix(y_val, y_pred_mlp))


# Fazendo a previsão
# Visualizando o X_train

print(X_train.head(3))

# Visualizando a base de teste

print(teste.head(3))

# Para a base de teste ser igual à base de treino, precisamos eliminar a coluna de id

X_teste = teste.drop('PassengerId',axis=1)

# Utilizando o melhor modelo na base de teste

y_pred = clf_best_rf.predict(X_teste)

# Criando uma nova coluna com a previsão na base de teste

teste['Survived'] = y_pred

# Selecionando apenas a coluna de Id e Survived para fazer o envio

base_envio = teste[['PassengerId','Survived']]

# Exportando para um csv

base_envio.to_csv('resultados5 .csv',index=False)

#Resultado = 0,7823