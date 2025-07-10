import re
import json
# import os
# import spacy

def load_abbreviation_map(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def remove_parentheses(text):
    """Remove parentheses and their contents from the text including surrounding spaces."""
    return re.sub(r"\s*\([^)]*\)\s*", " ", text).strip()


def expand_abbreviations(text, abbr_dict):
    for abbr in sorted(abbr_dict.keys(), key=len, reverse=True):
        pattern = r'\b' + re.escape(abbr) + r'\b'
        expansion = abbr_dict[abbr].split(";")[0].strip()
        text = re.sub(pattern, expansion, text)
    return text

SUBSTITUTION_DICT = {
    "$": "Dollar",
    "¢": "Cent",
    "§": "Paragraph",
}

#allowing characters including COMMA, DASH and semicolon 
# so they can be used for sentence splitting
ALLOWED_CHARACTERS = r"[^a-zA-ZäöüÄÖÜß0-9 .?!:;„“\"',\-–\n]"

def normalize_characters(text: str) -> str:
    """
    Removes disallowed characters from the text.
   """
    return re.sub(ALLOWED_CHARACTERS, '', text)

def character_substitution(text: str) -> str:
    """
    Replace specified disallowed characters with simple words (e.g., "$" → "Dollar").
    """
    for char, replacement in SUBSTITUTION_DICT.items():
        text = text.replace(char, replacement)
    return text