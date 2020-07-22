### Sumário

- Introdução
- Etapas algorítmos evolutivos
- Etapas algorítmos evolutivos - Otimização
- Classes de algorítmos evolutivos
- Conclusão

### Introdução

Hoje discutiremos as ideias básicas usadas pelos algorítmos evolutivos.

Como o nome sugere esta classe de algorítmos se basea nas ideias da teoria de evolução Neo-Darwiniana e tentam reproduzir o processo de evolução para resolver problemas de otimização, encontrando soluções melhor adaptadas (mais próximo do ótimo) a cada iteração.

Embora tenhamos um grande número de algorítmos de otimização, os algorítmos evolutivos podem em alguns contextos atacar de maneira melhor problemas de otimização que algorítmos tradicionais de otimização tem dificuldades.

Para começarmos a discussão vamos neste tópico somente apresentar as idéias básicas dos algorítmos evolutivos. 


### Etapas algorítmos evolutivos

Tendo por base a teoria neo-Darwiniana da evolução, é possível propor um
algoritmo evolutivo básico ou padrão com as seguintes características:

- Uma população de candidatos à solução (denominados indivíduos ou cromossomos) que se reproduzem com herança genética: cada indivíduo corresponde a uma estrutura de dados que representa ou codifica um ponto em um espaço de busca.
- Esses indivíduos devem se reproduzir (de forma sexuada  ou assexuada), gerando filhos que herdam algumas das características de seu(s) pai(s). No caso de reprodução sexuada, há troca de material genético entre dois ou mais indivíduos pais;
- Variação genética: durante o processo reprodutivo, os filhos não apenas herdam características de seu(s) pai(s), como eles também podem sofrer mutações genéticas;
- Seleção natural: a avaliação dos indivíduos em seu ambiente – através de uma função de avaliação ou fitness – resulta em um valor correspondente à adaptação (qualidade) ou ao fitness deste indivíduo. A comparação dos valores individuais de fitness resultará em uma competição pela sobrevivência e reprodução no ambiente, havendo uma vantagem seletiva daqueles indivíduos com valores elevados de fitness.
- Quando todos os passos acima (reprodução, variação genética e seleção) tiverem sido executados, diz-se que ocorreu uma geração.

### Etapas algorítmos evolutivos - Otimização

Formalização matemática (notação voltada para problemas de otimização):

- Representação genotípica: codificação das soluções candidatas (um subconjunto delas comporá a população a cada geração);
- Representação fenotípica: interpretação do código.
- Espaço de busca: tem como elementos todos os candidatos à solução do problema;
- Função de adaptação ou de fitness: atribui a cada elemento do espaço de busca um valor de adaptação, que será usado como medida relativa de desempenho. Representa a pressão do ambiente sobre o fenótipo dos indivíduos.
- Operadores de inicialização: produzem a primeira geração de indivíduos (população inicial), tomando elementos do espaço de busca.
- Operadores genéticos: implementam o mecanismo de introdução de variabilidade aleatória no genótipo da população.
- Operadores de seleção: implementam o mecanismo de “seleção natural”.

Como podemos ver, os passos definidos anteriormente são de fato implementados pelos diferentes operadores genéticos, estes operadores de fato alteram os indíviduos e são responsável pela seleção dos indivíduos mais adaptados.

### Classes de algorítmos evolutivos

Nos seguintes tópicos do assunto iremos ver exemplos e entrar detalhes desses operadores montando nossas primeiras versões de algorítmos evolutivos. Na literature há diferentes algorítmos inspirados na teoria Neo-Darwiana onde cada um deles da maior ênfase a um etapa do processo de evolução, normalmente depende de onda essa ênfase esta constuma-se classificar os da aos algorítmos evolutivos diferentes subclasses. Sendos estas:  

- Algoritmos Genéticos (AG) – Genetic Algorithms (GA)
- Estratégias Evolutivas (EE) – Evolution Strategies (ES)
- Programação Evolutiva (PE) – Evolutionary Programming (EP)
- Programação Genética (PG) – Genetic Programming (GP)

É importante lembrar que a fronteira entre esses classes não algo rígido sendo comum encontrarmos algorítmos e heurísricas que possa combinar diferentes ideias dessas classes.

### Conclusão

Neste tópico discutimos:
- As principais idéias usadas nos algorítmos evolutivos
- As principais etapas do algorítmo e os termos usados quando aplicados a problemas de otimização
- A subdivisão desses algorítmos em diferentes classes
