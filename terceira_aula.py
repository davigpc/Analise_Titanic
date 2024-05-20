import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder


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

# ---------------------------------------------- #
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

# ---------------------------------------------- #
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

as_arvore = accuracy_score(y_test, y_pred_arvore)
as_knn = accuracy_score(y_test, y_pred_knn)
as_rl = accuracy_score(y_test, y_pred_rl)

print()

as_df = pd.DataFrame({
    'modelos': ['arvore', 'knn', 'rl'],
    'inicial': [as_arvore, as_knn, as_rl]
})

# matriz de confusao

print(confusion_matrix(y_test, y_pred_arvore))
print(confusion_matrix(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_rl))

# previsao dos dados de teste

print(x_train.head(3))
print(teste.head(3))

# ---------------------------------------------- #
# engenharia de recursos

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15,5))

ax.boxplot(treino.iloc[:,1:11])

ax.set_xticks(range(1,treino.iloc[:,1:11].shape[1]+1),treino.iloc[:,1:11].columns)

plt.show()

# como a coluna age e fare estão em uma escala muito diferente, é necessario trata-las

# Criando o scaler

transformer = RobustScaler().fit(treino[['Age','Fare']])

# Fazendo o transformação dos dados

treino[['Age','Fare']] = transformer.transform(treino[['Age','Fare']])

# Fazendo o mesmo para a base de teste

transformer = RobustScaler().fit(teste[['Age','Fare']])

teste[['Age','Fare']] = transformer.transform(teste[['Age','Fare']])

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

as_arvore = accuracy_score(y_test, y_pred_arvore)
as_knn = accuracy_score(y_test, y_pred_knn)
as_rl = accuracy_score(y_test, y_pred_rl)

as_df['escala'] = [as_arvore, as_knn, as_rl]
print(as_df)

# Visualizando novamente os dados

fig, ax = plt.subplots(figsize=(15,5))

ax.boxplot(treino.iloc[:,1:11])

ax.set_xticks(range(1,treino.iloc[:,1:11].shape[1]+1),treino.iloc[:,1:11].columns)

plt.show()

# ---------------------------------------------- #
# Parch -> numero de pais/filhos a bordo
# SibSp -> numero de conjuges/irmaos a bordo

# sera que uma familia grande tem maior de sobreviver?
# sera que um casal tem mais chance?

# sobrevivencia para sibsp

SibSp_df = treino.groupby('SibSp')['Survived'].agg(['sum','count','mean']).reset_index()

SibSp_df.columns = ['SibSp','sobrev','total','tx_sobrev']

print(SibSp_df)

# Verificando a sobrevivência para Parch

Parch_df = treino.groupby('Parch')['Survived'].agg(['sum','count','mean']).reset_index()

Parch_df.columns = ['Parch','sobrev','total','tx_sobrev']

print()
print(Parch_df)

# ---------------------------------------------- #
# Visualizando essas informações graficamente

fig, ax = plt.subplots(ncols=2,nrows=2,figsize=(10,6))

ax[0,0].plot(SibSp_df.SibSp, SibSp_df.sobrev)

ax[0,0].plot(SibSp_df.SibSp, SibSp_df.total)

ax[0,0].set_title('sobreviventes por nº de irmãos/cônjuges', fontsize=10)

labels1 = ax[0,1].bar(SibSp_df.SibSp, SibSp_df.tx_sobrev)

ax[0,1].bar_label(labels1,fmt="%.02f")

ax[0,1].set(ylim=(0,0.6))

ax[0,1].set_title('taxa de sobrevivência por nº de irmãos/cônjuges', fontsize=10)

ax[1,0].plot(Parch_df.Parch, Parch_df.sobrev)

ax[1,0].plot(Parch_df.Parch, Parch_df.total)

ax[1,0].set_title('sobreviventes por nº de pais/filhos', fontsize=10)

labels2 = ax[1,1].bar(Parch_df.Parch, Parch_df.tx_sobrev)

ax[1,1].bar_label(labels2,fmt="%.02f")

ax[1,1].set(ylim=(0,0.7))

ax[1,1].set_title('taxa de sobrevivência por nº de pais/filhos', fontsize=10)

plt.subplots_adjust(hspace=0.5)

plt.show()

# ---------------------------------------------- #
# Criando uma função para verificar se os dois valores são vazios

def sozinho(a,b):

    if (a == 0 and b == 0):

        return 1

    else:

        return 0

# Aplicando essa função na base de treino

treino['Sozinho'] = treino.apply(lambda x: sozinho(x.SibSp,x.Parch),axis=1)

# Verificando os valores nessa coluna

print(treino.groupby('Sozinho')[['SibSp','Parch']].mean())

# Fazendo o mesmo para a base de teste

teste['Sozinho'] = teste.apply(lambda x: sozinho(x.SibSp,x.Parch),axis=1)

# Criando para a base de treino

treino['Familiares'] = treino.SibSp + treino.Parch

# E para a base de teste

teste['Familiares'] = treino.SibSp + treino.Parch

# Verificando a sobrevivência para Familiares

Familiares_df = treino.groupby('Familiares')['Survived'].agg(['sum','count','mean']).reset_index()

Familiares_df.columns = ['Familiares','sobrev','total','tx_sobrev']

print(Familiares_df)

# ---------------------------------------------- #

# Visualizando a informação de familiares

fig, ax = plt.subplots(ncols=2,figsize=(10,3))

ax[0].plot(Familiares_df.Familiares, Familiares_df.sobrev)

ax[0].plot(Familiares_df.Familiares, Familiares_df.total)

ax[0].set_title('sobreviventes por nº de familiares', fontsize=10)

labels1 = ax[1].bar(Familiares_df.Familiares, Familiares_df.tx_sobrev)

ax[1].bar_label(labels1,fmt="%.02f")

ax[1].set(ylim=(0,0.8))

ax[1].set_title('taxa de sobrevivência por nº de familiares', fontsize=10)

plt.show()

# ---------------------------------------------- #

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

as_arvore = accuracy_score(y_test, y_pred_arvore)
as_knn = accuracy_score(y_test, y_pred_knn)
as_rl = accuracy_score(y_test, y_pred_rl)

as_df['pessoas'] = [as_arvore, as_knn, as_rl]
print(as_df)

# ao adicionar essa nova coluna, houve piora no desempenho dos modelos
# nesse sentido, é necessario selecionar melhor as variaveis

# ---------------------------------------------- #

# Analisando a correlação entre as variáveis

print(treino.corr())

# Tornando a correlação mais visual

fig, ax = plt.subplots(figsize=(10,5))

sns.heatmap(treino.corr(), annot=True, fmt=".2f")

plt.show()

# Visualizando a relação entre essas variáveis

fig, ax = plt.subplots(figsize=(10,3))

sns.boxplot(data=treino,x='Pclass',y='Fare',hue='Survived')

ax.set_title('Pclass x Fare', fontsize=10)

plt.show()

# ---------------------------------------------- #

# Verificando a taxa de sobrevivência em cada uma das classes

print(treino.groupby('Pclass')['Survived'].mean())

# Entendendo a relação entre Pclass x Fare

print(treino.groupby(['Pclass','Survived'])['Fare'].agg(['min','mean','max']))

# existe uma correlacao muito forte entre a coluna Pclass e Pfare
# analisando o grafico, é possivel perceber que pessoas que pagaram mais caro
# obtiveram mais taxa de sobrevivencia

# Podemos importar novamente as bases para "recuperar" a coluna de embarque

treino2 = pd.read_csv('train.csv')

teste2 = pd.read_csv('test.csv')

print(treino2.head(3))

# Como temos valores vazios, podemos novamente fazer o tratamento dos dados

treino2['Embarked'] = treino2['Embarked'].fillna('S')

# ---------------------------------------------- #

# Criando o encoder
# Faz sentido criar um encoder cardinal pois a ordem de embarque importa sim
# isso se dá pelo fato de que a passagem de quem embarca no ultimo porto
# é mais barata, já que há menos tempo de viagem

categorias = ['S','C','Q']

enc = OrdinalEncoder(categories=[categorias],dtype='int32')

# Fazendo o fit com os dados

enc = enc.fit(treino2[['Embarked']])

# Podemos então adicionar essa coluna na base de treino original

treino['Embarked'] = enc.transform(treino2[['Embarked']])

# E fazer o fit com os dados de teste

enc = enc.fit(teste2[['Embarked']])

# E adicionar na base de teste original

teste['Embarked'] = enc.transform(teste2[['Embarked']])

# Agora podemos eliminar as colunas desnecessárias

treino = treino.drop(['Embarked_C','Embarked_Q','Embarked_S'],axis=1)

teste = teste.drop(['Embarked_C','Embarked_Q','Embarked_S'],axis=1)

# Visualizando novamente a correlação

fig, ax = plt.subplots(figsize=(10,5))

sns.heatmap(treino.corr(), annot=True, fmt=".2f")

plt.show()

# ---------------------------------------------- #

# Separando X e y

X = treino.drop(['PassengerId','Survived'],axis=1)

y = treino.Survived

# Usando a regressão logística nos dados

clf_rl = LogisticRegression(random_state=42,max_iter=1000).fit(X,y)

# Verificando a importância

print(clf_rl.coef_[0])

# Agora usando a árvore de classificação

clf_ac = tree.DecisionTreeClassifier(random_state=42).fit(X,y)

# Verificando a importância

print(clf_ac.feature_importances_)

# Criando um DataFrame

imp = pd.DataFrame({

    'colunas': X.columns,

    'reg. log.': clf_rl.coef_[0],

    'arvore': clf_ac.feature_importances_

})

print(imp)

# ---------------------------------------------- #

# Podemos apenas manter as colunas mais relevantes

treino = treino.drop(['SibSp','Parch'],axis=1)

teste = teste.drop(['SibSp','Parch'],axis=1)

# ---------------------------------------------- #

# Separando a base de treino em X e y

X = treino.drop(['PassengerId','Survived'],axis=1)

y = treino.Survived

# Separando em treino e validação

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Criando o classificador (arvore)

clf_ac = tree.DecisionTreeClassifier(random_state=42)

# Fazendo o fit com os dados

clf_ac = clf_ac.fit(X_train,y_train)

# Fazendo a previsão

y_pred_ac = clf_ac.predict(X_val)

# knn

clf_knn = KNeighborsClassifier(n_neighbors=3)

# Fazendo o fit com os dados

clf_knn = clf_knn.fit(X_train,y_train)

# Fazendo a previsão

y_pred_knn = clf_knn.predict(X_val)

# regressao logistica

clf_rl = LogisticRegression(random_state=42,max_iter=1000)

# Fazendo o fit com os dados

clf_rl = clf_rl.fit(X_train,y_train)

# Fazendo a previsão

y_pred_rl = clf_rl.predict(X_val)


# Avaliando a acuracia dos modelos
# Para a árvore

as_ac = accuracy_score(y_val, y_pred_ac)

print(as_ac)

# Para o knn

as_knn = accuracy_score(y_val, y_pred_knn)

print(as_knn)

# Para a regressão logística

as_rl = accuracy_score(y_val, y_pred_rl)

print(as_rl)

# Criando primeiramente o DataFrame

# as_df = pd.DataFrame({

#     'modelos': ['arvore','knn','reg. log.'],

#     'inicial': [as_ac,as_knn,as_rl]

# })

#

# as_df

# Adicionando novas colunas no DafaFrame

# as_df['escala'] = [as_ac,as_knn,as_rl]

# as_df['pessoas'] = [as_ac,as_knn,as_rl]

# as_df['colunas'] = [as_ac,as_knn,as_rl]

# Visualizando

print(as_df)

# Para a base de teste ser igual à base de treino, precisamos eliminar a coluna de id

X_teste = teste.drop('PassengerId',axis=1)

# Utilizando a regressão logística na base de teste

y_pred = clf_rl.predict(X_teste)

# Criando uma nova coluna com a previsão na base de teste

teste['Survived'] = y_pred

# Selecionando apenas a coluna de Id e Survived para fazer o envio

base_envio = teste[['PassengerId','Survived']]

# Exportando para um csv

base_envio.to_csv('resultados_escala.csv',index=False)

# Exportando para um csv

base_envio.to_csv('resultados_escala.csv',index=False)