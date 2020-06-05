### Sumário

- Introdução a otimização
- Método do gradiente descendente
- Relação convergência e o parâmetro $\alpha$
- Relação convergência e o valor inicial
- Conclusão

### Introdução a otimização 

Antes de falar do assunto propriamente tratado neste tópico (Método do gradiente descendente), vamos primeiro fazer um breve introdução sobre otimização.

De modo geral, otimizar uma função significa achar quais valores suas variáveis devem assumir de modo que a função atinja o máximo ou minimo valor possível, como definido de forma concisa na experssão abaixo:

$\underset{\overline{x}}{\operatorname{argmax}} f(\overline{x})$  ou $\underset{\overline{x}}{\operatorname{argmin}} f(\overline{x})$ onde $\overline{x} \in S$

Com S representando um conjunto finito ou não.

Este problema, definido de forma aparentente simples tem na verdade um enorme numero de aplicações nos mais variados campos da ciência desde de mercados financeiros, energia, telecomunições, aprendizado de máquina, para citar alguns. Devido ao grande variedade de problemas que podem ser expressos como um problema de otimização, diversos algorimos foram desenvolvidos para atacar o problema e são normalmente agrupados dependendo das características da função f(x) e do conjunto S, algums exemplos são otimização Programação Linear onde a f(x) é uma função linear, Programação Inteira onde x pertence ao conjunto dos inteiros, Programação Não Linear, etc.

Neste tópico vamos falar de um dos mais simples algorítmos de otimização, porém ainda poderoso, o Método do Gradiente Descendente, mais particularmente sua versão para funções de 1 dimençao.

### Método do Gradiente Descendente

O método gradiente descente, nos permite achar o ponto de máximo ou mínimo de uma função, de modo iterativo, ou seja, diferentes pontos candidatos a mínimo são encontrados ao longo da execução do algorítimo até que este convirja para uma solução e um ponto estável. O algorítimo pode ser descrito através dos seguintes passos:

- Inicia-se um ponto inicial x0, um limite para declarar a convegência ou um máximo número de iterações
- Equanto não convergir e/ou o número de iterações for menos que o máximo fazemos:
  - $d_k = \frac{\partial{f}}{{dx}}$ em $x_k$
  - $x_{k+1} = x_{k} - \alpha*d_{k}$
  
Como pode ser visto, basta conhecermos o gradiente da função a ser otimizada, escolher o ponto inicial e o valor de $\alpha$ que irá determinar o quanto iremos nos mover na direção oposta ao gradiente (no caso de minimização). Veremos como as escolhas do valor inicial do algorítmo e do parâmetro $\alpha$ afetam o desempenho do algorítmo.

Vejamos um exemplo, minimizar uma função do segundo grau como mínimo local igual a $f(x_{min})$ = 1 em $x_{min}$ = 2


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
def f(x):
    return 4*(x-2)**2 + 1

def grad_f(x):
    return 8*(x-2)
```


```python
x = np.linspace(0, 4, 1000)

fig, axs = plt.subplots(1, 2, figsize=(16,4))
axs[0].plot(x, f(x))
axs[0].set_ylim([0, 20])
axs[0].set_title("f(x)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].grid()

axs[1].plot(x, grad_f(x))
axs[1].set_xlim([0, 4])
axs[1].set_title("f(x) gradient")
axs[1].set_xlabel("x")
axs[1].set_ylabel("f(x)")
axs[1].grid()
```


![png](M%C3%A9todo_do_gradiente_descendente_1D_files/M%C3%A9todo_do_gradiente_descendente_1D_8_0.png)



```python
# Implementação método gradiente descendente
def gradient_descent(fx, gradx, x0, delta_lim, alpha, max_it = 300):
    x_steps = [x0]
    fx_steps = [f(x0)]
    
    delta = f(x0)
    x = x0
    it = 0
    while np.abs(delta) > delta_lim and it < max_it:
        x_prev = x
        x -= alpha*gradx(x)
        delta = x_prev - x
        it += 1
        x_steps.append(x)
        fx_steps.append(f(x))
    
    return x_steps, fx_steps
    
```

### Relação convergência e o parâmetro $\alpha$

Vamos agora realizar alguns experimentos para observar o efeito do parâmetro $\alpha$ na convergência do algorítmo. Para isso, fixamos o ponto inicial do algorítmo e testamos três diferetes valores de $\alpha$.


```python
alphas = [5e-3, 1e-2, 2.3e-1]
experiments = []
for alpha in alphas:
    experiments.append(gradient_descent(f, grad_f, 0, 1e-5, alpha))
```


```python
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
for experiment in experiments:
    axs[0].plot(experiment[0])
    axs[0].set_xlim([0, 70])
    axs[0].set_ylim((0, 3))
    axs[0].set_title("Convergência do valor de x para o mínimo = 2")
    axs[0].set_ylabel("x")
    axs[0].set_xlabel("iteration")
    axs[0].grid()
    axs[1].plot(experiment[1])
    axs[1].set_xlim([0, 70])
    axs[1].set_ylim((0, 18))
    axs[1].set_title("Convergência do valor de f(x) para o mínimo = 1")
    axs[1].set_ylabel("f(x)")
    axs[1].set_xlabel("iteration")
    axs[1].grid()
axs[0].legend(alphas)
axs[1].legend(alphas)
fig.tight_layout()
```


![png](M%C3%A9todo_do_gradiente_descendente_1D_files/M%C3%A9todo_do_gradiente_descendente_1D_13_0.png)


Podemos ver o comportamento para os três diferentes valores de $\alpha$, vemos que para $\alpha$ muito pequeno o algoritmo precisa de mais iterações para convergir para o mínimo da função, já para valores maiores, a convergência é mais rápida e mais instável. Caso esse valor seja muito alto, o algorítmo pode até divergir.

Vamos visualizar a mesma informação, agora vendo como as soluções intermediárias do algorítmo se movimentam sobre a curva da função f(x):


```python
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(x, f(x))
axs[0].plot(experiments[0][0][0:-1:8], experiments[0][1][0:-1:8], '*--')
axs[0].set_ylim([0, 20])
axs[0].set_title("Evolução dos valores de x para diferentes iterações")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].legend([str(alphas[0])])
axs[0].grid()

axs[1].plot(x, f(x))
axs[1].plot(experiments[2][0][0:-1:3], experiments[2][1][0:-1:3], '*--')
axs[1].set_ylim([0, 20])
axs[1].set_title("Evolução dos valores de x para diferentes iterações")
axs[1].set_xlabel("x")
axs[1].set_ylabel("f(x)")
axs[1].grid()
axs[1].legend([str(alphas[2])])
fig.tight_layout()
```


![png](M%C3%A9todo_do_gradiente_descendente_1D_files/M%C3%A9todo_do_gradiente_descendente_1D_15_0.png)


Aqui podemos ver como a solução se aproxima do ponto ótimo, para $\alpha$ menor temos uma aproximação suave da esquerda para direita, já no segundo caso, a trajetória tem formato em zig-zag até a convergência.

Como pode-se ver a escolha de $\alpha$ é fundamental para determinar o sucesso do algorítmo, há diversas heurísticas para melhor ajustar esse parâmetro inclusive de forma dinâmica ao longo das iterações, não iremos discutir nesse tópico tais opções, mas mais informações podem ser vistas por exemplo em: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1

### Relação convergência e o valor inicial

Para vermos os efeitos da escolha do ponto inicial no algorítmo vamos usar como exemplo, outra função:

$f(x) = \exp^{-0.1(x-\frac{2\pi}{3})}\sin{x}$

e sua derivada que pela regra de derivada do produto é:

$\frac{\partial{f(x)}}{\partial{x}} = -0.1\exp^{-0.1(x-\frac{2\pi}{3})}\sin{x} + \exp^{-0.1(x-\frac{2\pi}{3})}\cos{x}$


```python
def f(x):
    return np.exp(-0.1*(x-2*np.pi/3))*np.sin(x)

def grad_f(x):
    return np.exp(-0.1*(x-2*np.pi/3))*(-0.1)*np.sin(x) + np.exp(-0.1*(x-2*np.pi/3))*np.cos(x)
```


```python
x = np.linspace(2*np.pi/3, 4*np.pi, 1000)

fig, axs = plt.subplots(1, 2, figsize=(16,4))
axs[0].plot(x, f(x))
axs[0].set_ylim([-1, 1])
axs[0].set_title("f(x)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].grid()

axs[1].plot(x, grad_f(x))
axs[1].set_ylim([-1, 1])
axs[1].set_title("f(x) gradient")
axs[1].set_xlabel("x")
axs[1].set_ylabel("f(x)")
axs[1].grid()
```


![png](M%C3%A9todo_do_gradiente_descendente_1D_files/M%C3%A9todo_do_gradiente_descendente_1D_20_0.png)


Vamos avaliar o algorítmo para algumas combinações de pontos iniciais a $\alpha$


```python
x_steps_p1, fx_steps_p1 = gradient_descent(f, grad_f, 2.5, 1e-5, 1e-2)
x_steps_p2, fx_steps_p2 = gradient_descent(f, grad_f, 8, 1e-5, 3e-2)
x_steps_p3, fx_steps_p3 = gradient_descent(f, grad_f, 2.5, 1e-5, 4)
x_steps_p4, fx_steps_p4 = gradient_descent(f, grad_f, 2.5, 1e-5, 8)
```


```python
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(x, f(x))
axs[0, 0].plot(x_steps_p1[0:-1:12], fx_steps_p1[0:-1:12], '*--')
axs[0, 0].set_ylim([-1, 1])
axs[0, 0].set_title("f(x)")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("f(x)")
axs[0, 0].legend(["x0 = 2.5"])
axs[0, 0].grid()

axs[0, 1].plot(x, f(x))
axs[0, 1].plot(x_steps_p2[0:-1:12], fx_steps_p2[0:-1:12], '*--')
axs[0, 1].set_ylim([-1, 1])
axs[0, 1].set_title("f(x)")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("f(x)")
axs[0, 1].legend(["x0 = 8"])
axs[0, 1].grid()

axs[1, 0].plot(x, f(x))
axs[1, 0].plot(x_steps_p3[0:-1:13], fx_steps_p3[0:-1:13], '*--')
axs[1, 0].set_ylim([-1, 1])
axs[1, 0].set_title("f(x)")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("f(x)")
axs[1, 0].legend(["x0 = 2.5"])
axs[1, 0].grid()

axs[1, 1].plot(x, f(x))
axs[1, 1].plot(x_steps_p4[0:-1:12], fx_steps_p4[0:-1:12], '*--')
axs[1, 1].set_ylim([-1, 1])
axs[1, 1].set_title("f(x)")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("f(x)")
axs[1, 1].legend(["x0 = 2.5"])
axs[1, 1].grid()

fig.tight_layout()
```


![png](M%C3%A9todo_do_gradiente_descendente_1D_files/M%C3%A9todo_do_gradiente_descendente_1D_23_0.png)


Primeiro, notemos que há dois pontos de mínimo na região mostrada de f(x) e que dependo da escolha inicial de x, o algorítmo pode convergir tanto para o mínimo em 4.35 ou em 10.68, sendo o segundo o mínimo com menor f(x). 

Vamos nas primeiras duas figuras, que o algorítmo pode convergir para diferentes mínimos dependendo do ponto inicial. Além disso, também podemos ter a convergência para diferentes mínimos alterando o parâmetro $\alpha$ como mostram as duas útilmas figuras.

Ou seja, para funções mais complexas, com mais de um ponto de mínimo, o algorítmo não garante a convergência para um mínimo global, pondendo convergir para diferentes mínimos locais dependendo do ponto inicial e $\alpha$ escolhido. 

### Conclusão

Neste tópicos discutimos:

- Uma breve introdução sobre otímização
- O algorítimo do método do gradiente descendente e sua implementação
- A influência do parâmetro \alpha na velocidade de convergência
- A influência do ponto inicial na solução final encontrada pelo algorítimo
- Possibilidade da convergência para um mínimo local
