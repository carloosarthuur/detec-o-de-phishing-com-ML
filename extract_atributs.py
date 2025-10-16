import pandas as pd
import requests
from bs4 import BeautifulSoup
import whois
from dns import resolver
from urllib.parse import urlparse, urljoin
from collections import Counter
from datetime import datetime, UTC

def _get_most_frequent_domain(base_url, tags, attribute='href'):
    """
    Analisa uma lista de tags, extrai domínios dos links e verifica
    se o mais frequente corresponde ao domínio base do site.
    """
    base_domain = urlparse(base_url).netloc
    domains = []
    for tag in tags:
        link = tag.get(attribute)
        if link and not link.startswith('#') and not 'javascript:void' in link:
            absolute_link = urljoin(base_url, link)
            try:
                domain = urlparse(absolute_link).netloc
                if domain:
                    domains.append(domain)
            except Exception:
                continue

    if not domains:
        return 0

    most_common_domain = Counter(domains).most_common(1)[0][0]
    return 1 if most_common_domain == base_domain else 0

def extract_url_string_features(url):
    """Extrai os atributos da string da URL."""
    features = {}
    hostname = urlparse(url).hostname
    if not hostname:
        raise ValueError("Hostname inválido ou não encontrado na URL.")

    features['A1'] = len(url)
    features['A2'] = 1 if all(part.isdigit() for part in hostname.split('.')) and hostname.count('.') == 3 else 0
    features['A3'] = 1 if "@" in url else 0
    features['A4'] = url.count('.')
    features['A5'] = 1 if "-" in hostname else 0
    features['A6'] = 1 if "https" in urlparse(url).path else 0
    features['A7'] = 1 if "//" in url[8:] else 0
    features['A8'] = 1 if url.startswith("https://") else 0
    return features

def extract_external_service_features(url):
    """Extrai atributos de WHOIS, DNS e Open Page Rank."""
    features = {}
    domain = urlparse(url).hostname

    # Open Page Rank API
    try:
        # IMPORTANTE: Insira sua chave de API aqui
        api_key = "SUA_CHAVE_DE_API_AQUI"
        api_url = f"https://openpagerank.com/api/v1.0/getPageRank?domains[]={domain}"
        headers = {'API-OPR': api_key}
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('status_code') == 200 and data['response']:
            features['A9'] = data['response'][0].get('page_rank_integer', 0)
        else:
            features['A9'] = 0
    except Exception:
        features['A9'] = 0

    # WHOIS
    domain_info = whois.whois(domain)
    creation_date = domain_info.creation_date
    if not creation_date:
        raise ValueError("Data de criação não encontrada no WHOIS.")
        
    creation_date_single = creation_date[0] if isinstance(creation_date, list) else creation_date
    age = (datetime.now(UTC) - creation_date_single.replace(tzinfo=UTC)).total_seconds() * 1000
    features['A10'] = age

    # DNS
    records = resolver.resolve(domain, 'A')
    features['A12'] = len(records)
    features['A14'] = records.ttl
    
    try:
        records_aaaa = resolver.resolve(domain, 'AAAA')
        features['A13'] = len(records_aaaa)
        features['A15'] = records_aaaa.ttl
    except (resolver.NoAnswer, resolver.NXDOMAIN):
        features['A13'], features['A15'] = 0, 0
    
    return features

def extract_html_features(url, soup):
    """Extrai todos os atributos do código-fonte HTML."""
    features = {}
    
    features['A17'] = 1 if soup.find('iframe') else 0
    features['A25'] = 1 if soup.find('a') else 0

    all_links_tags = soup.find_all('a')
    all_links_hrefs = [tag.get('href', '') for tag in all_links_tags]
    
    features['A18'] = _get_most_frequent_domain(url, all_links_tags)
    
    all_link_tags_head = soup.find_all('link')
    features['A19'] = _get_most_frequent_domain(url, all_link_tags_head)
    
    if not all_links_hrefs:
        features['A21'] = 0
    else:
        most_common_count = Counter(all_links_hrefs).most_common(1)[0][1]
        features['A21'] = most_common_count / len(all_links_hrefs)

    null_links_total = sum(1 for href in all_links_hrefs if href.startswith('#') or 'javascript:void(0);' in href)
    features['A23'] = null_links_total / len(all_links_hrefs) if all_links_hrefs else 0

    footer = soup.find('footer')
    if footer:
        footer_links_tags = footer.find_all('a')
        footer_links_hrefs = [tag.get('href', '') for tag in footer_links_tags]
        
        if not footer_links_hrefs:
            features['A22'] = 0
        else:
            most_common_footer_count = Counter(footer_links_hrefs).most_common(1)[0][1]
            features['A22'] = most_common_footer_count / len(footer_links_hrefs)
        
        null_links_footer = sum(1 for href in footer_links_hrefs if href.startswith('#') or 'javascript:void(0);' in href)
        features['A24'] = null_links_footer / len(footer_links_hrefs) if footer_links_hrefs else 0
    else:
        features['A22'] = 0
        features['A24'] = 0

    return features

# --- SCRIPT PRINCIPAL ---
try:
    df_urls = pd.read_csv('urls.csv')
    print(f"Arquivo 'urls.csv' carregado com {len(df_urls)} URLs.")
    df_urls.drop_duplicates(subset=['url'], keep='first', inplace=True)
    print(f"Após remover duplicatas, restam {len(df_urls)} URLs únicas para processar.")
except FileNotFoundError:
    print("Erro: Arquivo 'urls.csv' não encontrado.")
    exit()

results = []

for index, row in df_urls.iterrows():
    url_original = row['url']
    label = row['label']
    
    url_com_protocolo = url_original
    if not url_com_protocolo.startswith('http'):
        url_com_protocolo = 'https://' + url_com_protocolo
    
    print(f"Processando URL ({index + 1}/{len(df_urls)}): {url_original}")
    
    try:
        features = {'url': url_original, 'label': label}
        
        response = requests.get(url_com_protocolo, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        features.update(extract_url_string_features(url_com_protocolo))
        features.update(extract_external_service_features(url_com_protocolo))
        features.update(extract_html_features(url_com_protocolo, soup))
        
        results.append(features)
        print(" -> Sucesso.")
        
    except Exception as e:
        print(f" -> ERRO: URL descartada. Motivo: {type(e).__name__}")
        continue

if results:
    df_features = pd.DataFrame(results)
    df_features.to_csv('features.csv', index=False)
    print(f"\nExtração concluída. {len(df_features)} URLs válidas foram salvas em 'features.csv'.")
else:
    print("\nNenhuma URL pôde ser processada com sucesso.")