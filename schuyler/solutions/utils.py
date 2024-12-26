import re

def tokenize(text):
        tokens = re.split(r'[\s_.,;:\-]+', text)
        camel_case_split = []
        for token in tokens:
            camel_case_split.extend(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', token))
        return [token.lower() for token in camel_case_split if token]