### Sumário

- Representação das soluções e problemas exemplos
- Partição igual com mínima diferença na soma (Problema da herança)
- Otimização da função de duas variáveis reais f(x, y)
- Problema do caixeiro viajante
- Representações genotípicas e espaço de busca
- Conclusão

### Representação das soluções e problemas exemplos

Como discutimos anteriormente para aplicarmos os algorítmos evolutivos precisamos de uma representação genotípica para nossas soluções, isto é, primeiro temos que definir como representar a solução do problema de otimização e nosso espaço de busca.

Ao definirmos um problema de otimização costumamos definir também como essa solução é representada e também as restrições do problema, juntos esses dois fatores determinam qual será o nosso espaço de busca. Normalmente um mesmo problema de otimização pode adotar diferentes representações sendo uma ou outra mais conveniente dependendo das características do problema a ser resolvido e do algorítmo usado.

Para entender melhor o que é esse espaço de busca e quais representações podem ser usadas, vamos analisar três diferentes problemas de otimização com diferentes representações de suas soluções, adotaremos estes problemas nos seguintes tópicos para discutir os algorítmos evolutivos e operadores genético.

### Partição igual com mínima diferença na soma (Problema da herança)

Neste problema nosso objetivo a dividir um número finitos de items em dois conjuntos cuja soma do valor dos items em cada conjunto sejam a mais proxima possível onde cada item a ser dividido possui um valor proprio. Apesar de simples de difinir achar a solução ótima não é nada trivial, na verdade esse problema é NP-Completo.

Vamos ver uma instância particular desse problema:

Dividir um os 10 itens com os seguintes valores: 

Valores: [1, 2, 3, 5 ,5, 10 ,5, 5, 8, 4]

Um possível solução seria:

Conjunto A = [8, 4, 5, 3, 2]
Conjunt B = [5, 5, 10, 4, 1]

Onde a diferença seria: |22-25| = 3.

Sabemos que está não é a solução otima pois a diferença não é menor que o menor valor dos itens, sempre que isso acontece podemos pegar um item de menor valor e trocar de conjunto para melhorar o resultado. Porém repetir esse procedimento de maneira ganaciosa não nos levará a solução otíma para todos os casos.

### Otimização da função de duas variáveis reais f(x, y)

Para analisarmos problemas com vetores de valores reais vamos tentar maximizar a seguinte função:

$f(x, y) = xsin(4\pi x) - ysin(4\pi y + \pi) + 1 $

No intervalo [-1. 2]

Vamos ver como essa função é:


```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```


```python
x_range = np.linspace(-1, 2, 100)
y_range = np.linspace(-1, 2, 100)
```


```python
def func(x, y):
    return x*np.sin(4*np.pi*x) - y*np.sin(4*np.pi*y + np.pi) + 1
```


```python
grid_x, grid_y = np.meshgrid(x_range, y_range)
```


```python
grid_x_f = grid_x.reshape(-1, 1)
grid_y_f = grid_y.reshape(-1, 1)
```


```python
z = func(grid_x_f, grid_y_f)
z = z.reshape(grid_x.shape)
```


```python
fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(grid_x, grid_y, z, cmap='inferno')
ax.view_init(elev=40, azim=-135)
ax.set_title("f(x, y)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
ax = fig.add_subplot(122)
countour = ax.contour(grid_x, grid_y, z, cmap='inferno', levels=6)
ax.set_title("f(x, y) - Curvas de nível")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
```




    (-1, 2)




![png](Algor%C3%ADtmos_Evolutivos_-_Representa%C3%A7%C3%A3o_das_solu%C3%A7%C3%B5es_e_problemas_exemplos_files/Algor%C3%ADtmos_Evolutivos_-_Representa%C3%A7%C3%A3o_das_solu%C3%A7%C3%B5es_e_problemas_exemplos_13_1.png)


Como podemos ver no gráfico a função apresenta diversos pontos de máximo e mínimo, porém apresenta um máximo global próximo as coordenadas coordenadas [1.63, 1.63]. Justamente por apresentar diversos pontos de máximo e mínimo esta função apresenta um desafio para algorítmos de otimização que progridem a partir de um único ponto ou usam informações referentes a derivada da função para identificar pontos estacionários onde o gradiente é nulo.

### Problema do caixeiro viajante

O problema do caixeiro viajante consiste de um viajante que precisa visitar N cidades com a liberdade de ir de um cidade para qualquer outra mas com a restrição de não revisitar as cidades já vistas pelo caminho. Objetivo é visitar todas as cidades percorrendo a menor distância possível. Esse clássico problema da computação é NP-Completo.

Como exemplo vamos supor que temos 7 cidades com as seguintes coordenadas.


```python
cities = np.array([[0, 2], [1, 3], [1.3, 0.2], [2, 2.3], [2.7, 1.5], [3, 3.1], [2.8, 1]])
```


```python
route = np.arange(len(cities))
```

Vamos agora gerar duas rotas aleatórias:


```python
np.random.seed(32)
```


```python
route_1 = np.copy(route)
np.random.shuffle(route)
route_2 = np.copy(route)
```


```python
route_1 = np.r_[route_1, route_1[0]]
route_2 = np.r_[route_2, route_2[0]]
```


```python
fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(121)
ax.plot(cities[:, 0], cities[:, 1], 'o')
ax.plot(cities[route_1, 0], cities[route_1, 1], '--')
ax.set_title("Cities - Route 1")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid()
ax = fig.add_subplot(122)
ax.plot(cities[:, 0], cities[:, 1], 'o')
ax.plot(cities[route_2, 0], cities[route_2, 1], '--')
ax.set_title("Cities - Route 2")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid()
```


![png](Algor%C3%ADtmos_Evolutivos_-_Representa%C3%A7%C3%A3o_das_solu%C3%A7%C3%B5es_e_problemas_exemplos_files/Algor%C3%ADtmos_Evolutivos_-_Representa%C3%A7%C3%A3o_das_solu%C3%A7%C3%B5es_e_problemas_exemplos_23_0.png)


E calcular a distância de cada rota:


```python
from scipy.spatial import distance_matrix
```


```python
dmatrix = distance_matrix(cities, cities)
```


```python
def route_distance(route, dmatrix):
    dist = 0
    for i in range(0, len(cities)):
        dist += dmatrix[route[i], route[i+1]]
    return dist
```


```python
route_distance(route_1, dmatrix),  route_distance(route_2, dmatrix)
```




    (14.21744619387674, 12.670736679703483)



Portanto segundo nosso critério de distancia mínima a rota número 2 é melhor que a 1

### Representações genotípicas e espaço de busca

Agora que apresentamos os três exemplos de probremas que iremos discutir quando ao falarmos dos operadores genêticos vamos olhar como cada um deles será representado.

- Representação Problema da Herença:

Neste problema cada solução pode ser representada com um par de listas onde cada item está presente em uma das dias listas. Porém, outra forma de representar e que será mais adequada aos nossos algorítmos é termos um vetor binário com o comprimento igual ao número de itens. Ex: [0, 1, 0, 1, 1, 0].

No qual os itens com 1 pertencem a um grupo e com 0 a outro, veja que com essa representação faz com que nosso espaço de busca será $F_{2}^6$, ou seja temos $2^6$ soluções possíveisl. Veja que esse número cresce absurdamente com o aumento do número de itens.

- Representação Função Real:

No caso de otimização uma função de variáveis reais nossa represetação é mais real e cada possível solução será um vetor de números reais de tamanho 2, $ x \subset R^2$.

- Representação Problema Caixeiro Viajante:

Neste caso a presentação consiste cada possivel solução consiste numa lista de cidades contendo a ordem em que elas são visitadas. Assim para 7 cidades teremos como possível solução todas as permutaçoes dessa list, no caso 7!.

Desse modo iremos trabalhar com até 3 representações diferentes:

- Vetores de números reais
- Vetores binários
- Listas de tamanho fixo

Podemos ver que em todos os casos o espaço de busca é muito grande, para o problema da herença e do caixeiro viajante temos outro agravante, estes problemas necessiam de soluções inteiras e assim não conseguimos obter informações de derivadas para guiar o processo de otimização. Embora existam algorímos muito bons que podem ser usados, normalmente relaxando as restrições do problema, vamos restringir a nossa discussão somente aos algorítmos evolutivos.

### Conclusão

Neste tópico discutimos:
- Mostramos como diferentes problemas de otimização adotam diferentes tipos de representação de suas soluções
- Apresentamos três problemas diferentes, o problema de herança, otimização de uma função de variáveis reais e o problema do caixeiro viajante.
- Discutimos as representações usadas em cada problema e o tamanho do espaço de busca ao adotar essas representações.

