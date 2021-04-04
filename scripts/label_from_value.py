import re

MONTHS = (
    'Zero',
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
)

ALLOWED_URL = (
    'http://www.wikidata.org/',
    'http://www.w3.org/',
)

DISALLOWED_URL = (
    'http://www.wikidata.org/entity/statement/',
    'http://www.wikidata.org/reference/',
    'http://www.wikidata.org/value/'
)

def label_from_value(value):
    """
    Translate value returned from Wikidata to label:
      xxxx-xx-xxTxx:xx:xxZ to Day Month Year
      Alphanumeric value directly to label
      or call URI_to_label
    """
    res = re.match('^(\-?\d\d\d\d)-(\d\d)-(\d\d)T\d\d:\d\d:\d\dZ$', value)
    if res:
        return f'{res.group(3)} {MONTHS[int(res.group(2))]} {res.group(1)}'
    
    res = re.match("^[’'–,\w\s\.\-\(\)]+$", value)
    if res:
        return value
      
    if not any([i in URI for i in ALLOWED_URL]):
        return None
    
    if any([i in URI for i in DISALLOWED_URL]):
        return None
    
    return URI_to_label(value) # check the cache, request Wikidata if miss
