<div align='center'>
  <h3>
    Q-learning 🆚️ Sarsa Algorithm 
  </h3>
</div>

O objetivo do projeto é a comparação entre o desempenho de dois algorítimos muito utilizados em Reinforcement Learning, o [QLearning](https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-q-learning) e o [Sarsa](https://towardsdatascience.com/reinforcement-learning-with-sarsa-a-good-alternative-to-q-learning-algorithm-bf35b209e1c). Para demonstrar os diferentes desempenhos, utilizou-se dois ambientes implementados pela biblioteca `gym`, o [TaxiDriver](https://www.gymlibrary.dev/environments/toy_text/taxi/) e o [Cliff Walking](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/)

### Configurações ⚙️

Instale as bibliotecas necessárias utilizando o comando:

```bash

pip install -r requirements.txt

```

Para rodar o projeto siga o padrão:

```bash

python main.py <ambiente> <algoritimo> <reuse_data>

```

Onde:

- **ambiente** : 'taxi' ou 'cliff'. 

Ambiente onde o algorítimo ocorrerá.

- **algoritimo**: 'q' ou 'sarsa' ou 'both' . 

Algorítimo que será utilizado para resolver o problema. 

Caso a opção 'both' seja escolhida, ambos os algorítimos rodaram e um gráfico de sem desempenho comaprativo será criado, caso utilizado essa opção, **reuse_data = 0** necessariamente.

- **reuse_data** : '0' ou '1'. 

Utilizar ou não um csv existente como valor da **q_table**. 

### TaxiDriver - Desempenho 🚕️

### Cliff Walking - Desempenho 🧙‍♂️️

### Cliff Walking vs  TaxiDriver - Vantagens e Desvantagens 📌️ 
