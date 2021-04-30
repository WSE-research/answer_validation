# Sqlite3 table was created with statements:
# CREATE TABLE main (key TEXT NOT NULL PRIMARY KEY, value TEXT)
# CREATE INDEX index_value ON main(value)

import sqlite3

cache3 = sqlite3.connect('filename of labels cache here')

MONTHS = (
    'Zero',
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
)

ALLOWED_URL = (
    'http://www.wikidata.org/',
    'http://www.w3.org/',
    'http://schema.org',
    'http://wikiba.se/ontology#'
)

DISALLOWED_URL = (
    'http://www.wikidata.org/entity/statement/',
    'http://www.wikidata.org/reference/',
    'http://www.wikidata.org/value/'
)

last_request = None

def wait_wikidata(wait):
    """
    Wikidata tends to reject frequent requests
    1 second delay between requests fix unwanted rejects
    """
    global last_request
    
    if last_request and time.time() - last_request < wait:
        time.sleep(wait - (time.time() - last_request))
       
    last_request = time.time()
    
def prepare_URI(URI, add_brackets=True):
    """
    removes extra brackets and spaces from link
    """
    res = URI.lstrip(' {<').rstrip(' }>')

    if res.startswith('?') or not add_brackets:
        return res.strip()
    else:
        return '<' + res.strip() +'>'
      
def get_label_by_URI(URI):
    """
    get Wikidata label by URI
    returns label or None
    """
    parts = URI.split('/')
    
    query = f"""
SELECT ?label WHERE {{
  wd:{parts[-1]} rdfs:label ?label . 
  FILTER (lang(?label) = 'en')
}}
"""

    wait_wikidata(1.0)
    
    r = requests.get(endpoint, params={'format': 'json', 'query': query})
    
    if r.status_code == 200:
        data = r.json()
        
        if data and data['results'] and data['results']['bindings'] and data['results']['bindings'][0]['label']:
            return data['results']['bindings'][0]['label']['value']

    return None
      
def get_wikidata_label(URI):
    """
    transforms value, returned from Wikidata, into label
    URI - URI or literal
    returns label or None
    """
    res = re.match('^(\-?\d\d\d\d)-(\d\d)-(\d\d)T\d\d:\d\d:\d\dZ$', URI)
    if res:
        return f'{res.group(3)} {MONTHS[int(res.group(2))]} {res.group(1)}'
    
    res = re.match("^[’'–,\w\s\.\-\(\)]+$", URI)
    if res:
        return URI
        
    if not any([i in URI for i in ALLOWED_URL]):
        return None
    
    if any([i in URI for i in DISALLOWED_URL]):
        return None
    
    URI = prepare_URI(URI)
    cursor = cache3.execute('SELECT value FROM main WHERE key=(?)', (URI, ))
    labels = cursor.fetchone()
    
    if labels:
        return labels[0]
    else:
        label = get_label_by_URI(URI)
        cache3.execute('INSERT OR IGNORE INTO main VALUES (?,?)', (URI, label))
        cache3.commit()
