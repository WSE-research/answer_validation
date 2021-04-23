def check_triple(triple, labels, variable, entity, predicate):
    return (
        (entity == labels[0] and predicate == labels[1] and variable == triple[2]) or 
        (variable == triple[0] and predicate == labels[1] and entity == labels[2])
    )
  
def prepare_URI(URI, add_brackets=True):
    URI = URI.lstrip(' {').rstrip(' }')
    URI = URI.lstrip(' <').rstrip(' >')

    if URI.startswith('?') or not add_brackets:
        return URI.strip()
    else:
        return '<' + URI.strip() +'>'
        
def get_request_info(request):
    res = re.search('.* (\?.*|\*) WHERE { (.*) }(.*)', request)

    if res:
        variable = res.group(1)
        triples = res.group(2)
        limit = res.group(3)
        triples = re.findall('(\?[^\s]*|<[^\s]*>|VALUES|{.*})', triples)
        triples = [prepare_URI(i) for i in triples]
        return variable, triples, limit
    else:
        return None, None, None

def fill_triple(triple, values):
    result = []
    for i in triple:
        if i.startswith('?'):
            result.append(get_wikidata_label(values[i[1:]]))
        else:
            result.append(get_wikidata_label(i))
    return result

def evaluate_request(request, response, entity, predicate):
    """
    request: SPARQL sended to Wikidata
    response: dictionary of returned variables, eg: { '?s1': 'URI', '?o1': 'literal' }
    entity: entity's label as in VANiLLa
    predicate: predicate's label (get it from question relation)
    returns number of correct triples in request and total number of triples
    correctness of request can be calculated as triples_count/triples_total if triples_total else 0
    """
    variable, triples, _ = get_request_info(request)
    if not triples or not variable or len(triples) % 3 != 0:
        return 0, 0
    chunks = [triples[x:x+3] for x in range(0, len(triples), 3)]
        
    triples_total = 0
    triples_count = 0

    for k in chunks:
        triples_total += 1
        labels = fill_triple(k, response)
        triples_count += int(check_triple(k, labels, variable, entity, predicate))
        
    return triples_count, triples_total
