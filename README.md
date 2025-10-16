extract_atributos.py serve para extrair os atributos de um CSV contendo sites de phishing e confiáveis.
processo.py executa o treinamento dos modelos e a exibição dos resultados.

Para replicar o artigo:
1. Em extract_atributos.py, na parte do PageRank, colocar sua chave da API https://www.domcop.com/openpagerank/.

2. Escolher em processo.py quantos componentes serão usados no PCA do Naive Bayes.

3. Em extract_atributos.py, comente ou descomente a linha features.update(extract_external_service_features(url_com_protocolo)) para utilizar ou não os atributos dependentes de serviços de terceiros.

4. Mover um dos arquivos CSV para fora da pasta CSVs e renomeá-lo para apenas features.csv.

CSVs:
   - MOZ_urls.csv (lista com os sites confiaveis)
   - phishing_urls.csv (lista com os sites phishing)
   - featuressemfiltro.csv (todas as URLs de phishing e MOZ, features com valores padrões para quando não é possível obtê-las)
   - featurescortadas.csv (featuressemfiltro com a eliminação de URLs com dados padrões)
   - featuresfiltradas.csv (CSV com URLs filtradas para serem aceitas conforme os critérios explícitos no artigo)
   - featuresreduzidas.csv (CSV sem a utilização dos atributos dependentes de serviços de terceiros)
   - urls.csv (csv que contem a união dos sites confiaveis de phishing, usado em extract_atributs.py)
