import os
import spacy
import regex as re

from typing import List, Set, Generator, Dict, Any
from spacy.tokens import Doc, Token
from word2num_de import word_to_number
from sentence_transformers import SentenceTransformer, util

from helper import SUBORDINATE_MARKERS, COORD_CONJ

nlp = spacy.load("de_core_news_lg")
from german_compound_splitter import comp_split

utf8_file = os.path.join("german_dict", "german_utf8.dic")
ahoc = comp_split.read_dictionary_from_file(utf8_file) #activate the compound_spliter


# ---------- General Checker Functions for Rules ----------

def ok_numbers_converted(doc: Doc) -> float:
    """
    A single, self-contained function to check for unconverted numbers.
    This has NO external dependencies to ensure the correct logic is always executed.
    """
    # All necessary constants are defined INSIDE this function
    NUMBER_DICT = {
    # Ordinals
    "erster": "1.", "zweiter": "2.", "dritter": "3.", "vierter": "4.", "fünfter": "5.", "sechster": "6.", "siebter": "7.",
    "achter": "8.", "neunter": "9.", "zehnter": "10.", "elfter": "11.", "zwölfter": "12.",
    # Fractions
    "halb": "0.5", "eineinhalb": "1.5", "zweieinhalb": "2.5", "dreieinhalb": "3.5", "viereinhalb": "4.5",
    "fünfeinhalb": "5.5", "sechseinhalb": "6.5", "siebeneinhalb": "7.5", "achteinhalb": "8.5", "neuneinhalb": "9.5", "zehneinhalb": "10.5",
}
    RE_NUMERIC = re.compile(r"^\d+([.,]\d+)?$")
    RE_ORDINAL = re.compile(r"^\d+\.$")

    # --- Internal helper to check each token ---

    def _is_unconverted_internal(token: Token) -> bool:
        """Internal helper to check a single token."""
        # This is the 'like_num' logic, using token attributes correctly
        text, lemma = token.text, token.lemma_
        text_lower, lemma_lower = text.lower(), lemma.lower()
        is_like_num = False
        if lemma_lower in NUMBER_DICT or text_lower in NUMBER_DICT: is_like_num = True
        elif RE_NUMERIC.match(text) or RE_ORDINAL.match(text): is_like_num = True
        elif text.isdigit(): is_like_num = True
        else:
            try:
                word_to_number(lemma_lower)
                is_like_num = True
            except Exception:
                try:
                    word_to_number(text_lower)
                    is_like_num = True
                except Exception:
                    is_like_num = False
        
        # 'is_number' logic
        is_a_number = False
        if token.text.lower() == "ein" and token.pos_ != "NUM":
            is_a_number = False
        else:
            is_a_number = is_like_num or token.pos_ == "NUM"

        # 'is_number_word_that_should_be_converted' logic
        if not is_a_number:
            return False
        return not token.text.isdigit()

    # --- Main calculation ---
    violating_tokens = [token for token in doc if _is_unconverted_internal(token)]
    violation_count = len(violating_tokens)
    total_tokens = len(doc)
    
    score = 1.0 # Default score is perfect (1.0)
    if total_tokens > 0: # Calculate penalty based on violations
        penalty = min(1.0, violation_count / total_tokens) # Normalize the penalty
        score = 1.0 - penalty

    return score # if I want to return all for tracking/debugging violation_count, total_tokens, violating_tokens

def has_unsplit_compound(doc: spacy.tokens.Doc, ahoc: set) -> bool:
    """
    Checks if a document contains any unsplit compound nouns
    that should be simplified according to the given rules.

    This function iterates through each token in a spaCy Doc object and
    applies a set of heuristics to determine if it is a compound that
    should have been split but wasn't.

    Args:
        doc (spacy.tokens.Doc): The spaCy document object to check.
        ahoc (set): A lexicon or set of valid German words for checking
                    the validity of split parts.

    Returns:
        bool: True if at least one unsplit compound is found, False otherwise.
    """
    for token in doc:
        # Step 1: Preliminary checks on the token based on your logic.
        # This combines the logic from your `check_compound_split` and
        # `should_split` functions.
        if token.pos_ != "NOUN" or token.ent_type_ in {"PER", "LOC", "ORG"}:
            continue

        # Step 2: Attempt to split the compound using your splitter.
        parts =  comp_split.dissect(token.text, ahoc)
        
        # Step 3: Check if the token is a compound that can be split.
        if len(parts) < 2:
            continue
        
        # Step 4: Apply your "Konvens" rule check.
        # This rule suggests that if both the first and last parts of a
        # compound are short (<= 4 characters), it's not considered
        # "hard to read" and shouldn't be flagged as a violation.
        if len(parts[0]) <= 4 and len(parts[-1]) <= 4:
            continue

        # Step 5: Check if the split parts are valid words in the lexicon.
        # This ensures we don't try to split non-compounds or proper nouns
        # that aren't marked as named entities.
        valid_parts_count = sum(p.lower() in ahoc for p in parts)
        
        # Step 6: If a majority of the parts are valid, it's a compound that
        # should have been split. We've found a violation.
        if valid_parts_count / len(parts) >= 0.8:
            print(f"Violation detected: Found unsplit compound '{token.text}'")
            return True
            
    # If the loop completes without finding any violations, the rule is followed.
    return False

def has_apposition(doc: spacy.tokens.Doc) -> bool: #regex finds likely comma apposition
    if any(tok.dep_ == "app" for tok in doc):
        return True
    # Fallback: regex check for ', ... ,'
    # Only trigger if pattern matches (not followed by "die", "der", etc.)
    match = re.search(r', (?!die |der |das |und |aber |weil |obwohl )[^,]+,', doc.text)
    return bool(match)

def has_subordinate_clause(doc: spacy.tokens.Doc) -> bool:
    for tok in doc:
        if (tok.text.lower() in SUBORDINATE_MARKERS and 
            (tok.dep_ in "cp" or tok.pos_ == "SCONJ")):
            return True
    return False

def has_coordinate_clause(doc: spacy.tokens.Doc) -> bool:
    """Check if the document contains a coordinate clause."""
    return any(tok.dep_ == "cd" and tok.text.lower() in COORD_CONJ for tok in doc)

def has_disallowed_tense(doc: spacy.tokens.Doc) -> bool:
    for tok in doc:
        if tok.pos_ in ("VERB", "AUX"):
            tense = tok.morph.get("Tense")
            form = tok.morph.get("VerbForm")
            mood = tok.morph.get("Mood")
            if ("Pres" not in tense and "Part" not in form) or ("Sub" in mood):
                return True
    return False

def is_passive(doc: spacy.tokens.Doc) -> bool:
    # Werden + participle: Vorgangspassiv (event passive)
    has_werden = any(tok.lemma_ == "werden" and tok.pos_ == "AUX" for tok in doc)
    has_participle = any(tok.pos_ == "VERB" and "Part" in tok.morph.get("VerbForm", []) for tok in doc)
    if has_werden and has_participle:
        return True
    # Sein + participle: Zustandspassiv (state passive), only for transitive verbs!
    has_sein = any(tok.lemma_ == "sein" and tok.pos_ == "AUX" for tok in doc)
    if has_sein and has_participle:
        # Check: is the main verb transitive (does it take an object)?
        if any(tok.dep_ in {"oa", "obj"} for tok in doc):  # object present
            return True
    return False

