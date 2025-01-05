# Trading agents team




  ![alt text](./readme_assets/image.png)



- **Classe Principal: `Backtester`**
  - **Atributos:**
    - `agent`: Função responsável por executar os agentes de trading.
    - `ticker`: Código do ativo financeiro a ser analisado.
    - `start_date`: Data de início para o backtest.
    - `end_date`: Data de término para o backtest.
    - `initial_capital`: Capital inicial disponível para o investimento.
    - `portfolio`: Estrutura que armazena informações sobre o portfólio, incluindo caixa e ações.
    - `portfolio_values`: Histórico dos valores do portfólio ao longo do tempo.
  - **Métodos:**
    - `parse_action(agent_output)`: Analisa a ação de trading proposta pelo agente.
    - `execute_trade(action, quantity, current_price)`: Executa a transação de compra ou venda.
    - `run_backtest()`: Executa todo o processo de backtesting.
    - `analyze_performance()`: Avalia o desempenho do portfólio após o backtesting.

- **Funções Globais:**
  - `run_agent()`: Orquestra a execução sequencial dos agentes.
  - `calculate_trading_signals(historical_data)`: Calcula sinais de trading usando médias móveis.
  - `get_price_data(ticker, start_date, end_date)`: Busca dados de preço histórico para o ticker especificado.

### 2. Diagrama de Sequência

- **Objetivo:** Demonstrar a interação sequencial entre os agentes e o fluxo de mensagens dentro do sistema.
  
- **Fluxo de Processo:**
  1. **Usuário Inicia o Processo**: O usuário chama `run_agent`.
  2. **Agentes de Mercado** (`market_data_agent`): Coleta e processa dados de mercado.
  3. **Agentes Quantitativos** (`quant_agent`): Analisa os dados de mercado e gera sinais de trading.
  4. **Gerenciamento de Risco** (`risk_management_agent`): Avalia o risco do portfólio com base nos sinais.
  5. **Gerenciamento de Portfólio** (`portfolio_management_agent`): Toma decisões finais de compra/venda com base na análise de risco.

### 3. Diagrama de Atividades

- **Objetivo:** Visualizar o fluxo completo do processo de backtesting, desde a coleta de dados até a execução de transações.
  
- **Fluxo do Backtest:**
  1. **Início**: Inicialização do backtest com parâmetros como `ticker`, `start_date`, `end_date`, e `initial_capital`.
  2. **Coleta de Dados**: `get_price_data` é utilizado para buscar dados históricos.
  3. **Análise de Mercado**: `market_data_agent` processa os dados e calcula sinais de trading.
  4. **Análise Quantitativa**: `quant_agent` interpreta os sinais.
  5. **Avaliação de Risco**: `risk_management_agent` determina limites e scores de risco.
  6. **Execução de Transações**: Decisões de trading são executadas por `portfolio_management_agent`.
  7. **Registro de Resultados**: Atualização e registro dos valores do portfólio.
  8. **Análise de Desempenho**: `analyze_performance` avalia o retorno total, volatilidade, e outras métricas de performance.





# Bara executar os agentes, em ambiente windows.

1- baixar esse repositório
2- dentro da pasta do projeto, criar um arquivo com o nome .env
3- no arquivo .env, adicionar as configurações abaixo, substituindo as chaves pelas chaves corretas

```

OPENAI_API_KEY=sk-pr....gA
FINANCIAL_DATASETS_API_KEY=9a245e...766255
```

4- Executar o programa pelo atalho ExcecutarAgentes