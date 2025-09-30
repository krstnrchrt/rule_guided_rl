import pandas as pd
import numpy as np
import json, ast, re

ALLOWED_RULES = {
    "normalize_verb_tense",
    "convert_passive_to_active",
    "split_compound",
    "convert_word_to_number",
    "rewrite_apposition",
    "simplify_subordinate",
    "clean_punctuation",  # to be dropped later
}

def dedup_preserve_order(lst):
    seen = set()
    out = []
    for x in lst:
        if x not in seen and x:
            out.append(x)
            seen.add(x)
    return out

def normalize_rule_string(rule):
    """
    Ensure a rule value like '["normalize_verb_tense"]' becomes 'normalize_verb_tense'.
    """
    if pd.isna(rule):
        return None
    s = str(rule).strip()

    # Matches ["rule"] or ['rule']
    m = re.match(r'^\[\s*[\'"]?([A-Za-z0-9_]+)[\'"]?\s*\]$', s)
    if m:
        return m.group(1)

    # Otherwise return cleaned string
    return s.strip("'\"")