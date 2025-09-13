import spacy
import os
import pandas as pd
import json
import re
import importlib
import text_simpl_utils
import datetime
importlib.reload(text_simpl_utils)
from text_simpl_utils import *
#from compounds import analyze_compound

from word2num_de import word_to_number
#from compound_split import doc_split

from german_compound_splitter import comp_split

import logging

from spacy.tokens import Doc
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

from pprint import pprint
from pprint import pformat


nlp = spacy.load("de_core_news_lg")

#---- Load German dict for compound splitter 
# Returns the script's  directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build your path *relative* to the script's location
utf8_file = os.path.join(SCRIPT_DIR, "resources", "german_dict", "german_utf8.dic")
ahoc = comp_split.read_dictionary_from_file(utf8_file) #activate the compound_spliter

# --- Number Transformation ---

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

def like_num(text):
    text_lower = text.lower()

    if text_lower in NUMBER_DICT:
        return True
    if RE_NUMERIC.match(text) or RE_ORDINAL.match(text):
        return True
    try:
        num =  word_to_number(text_lower) # word2num_de
        return isinstance(num, (int, float)) # only accept if parsing gives an int/float
    except Exception:
        try:
            word_to_number(text_lower) # word2num_de
            return True
        except Exception:
            return False

def is_number(token):
    #token.text and token.pos_
    token_text = token['text']
    token_pos = token['pos']
    lemma = token['lemma'].lower()
    if token_text.lower() == "ein" and token_pos != "NUM":
        return False
    
    if like_num(token_text): #explicit matches
        return True
    
    if token_pos == "ADJ" and lemma in NUMBER_DICT: #ordinals (ADJ) that are in NUMBER_DICT
        return True
    
    return token_pos == "NUM"
    
def convert_word_to_number(text):
    doc = nlp(text)
    lemma = doc[0].lemma_.lower()
    #print("Lemma:", lemma)
    #print("Text:", text)

    if lemma in NUMBER_DICT:
        return NUMBER_DICT[lemma]
    if RE_NUMERIC.match(text):
        return text.replace(",", ".")
    if RE_ORDINAL.match(text):
        return text[:-1]
    try:
        return str(word_to_number(text.lower()))  # word2num_de
    except Exception:
        return text #if nothing applies, always return original text

# ----

# def check_compound_split(token):
#     return (
#         token['pos'] in {"NOUN", "PROPN"} and
#         len(token['text']) > 5 and
#         token['ent_type'] not in {"PER", "LOC", "ORG"}
#     )
def check_compound_split(token):
    if token["text"] is None:
        print("Token text is None, cannot check compound split.")
    return (
        token["pos"] == "NOUN"
            and len(token["text"]) >= 5  
            and token["text"][0].isupper()       
            and token["ent_type"] not in {"PER", "LOC", "ORG"}
            )  

MIDDLE_DOT = "·"
HYPHEN_RX = re.compile(r"[-–—]") #recognise hyphen-minus, en-dash, em-dash

# def check_compound_split(token):
#     return (
#         token.get("pos") == "NOUN"
#         and len(token.get("text", "")) >= 5
#         and token["text"][0].isupper()          # safe because len already checked
#         and token.get("ent_type", "") not in {"PER", "LOC", "ORG"}
#     )

def split_compound_word(text: str, ahoc):

    # if already hyphenated, replace with middle dot
    if HYPHEN_RX.search(text):
        # remove optional spaces around the dash, then swap dash for mediopunkt
        # cleaned = HYPHEN_RX.sub(MIDDLE_DOT, text.replace(" ", ""))
        # return cleaned

        # If the word is already hyphenated, split by the hyphen, capitalize each part, then join with middle dot
        # Use re.split to handle different dash types
        parts = [p.capitalize() for p in HYPHEN_RX.split(text) if p]
        return MIDDLE_DOT.join(parts)
    
     # apply dictionary splitter  (guarded against IndexError)
    try:
        raw_parts = comp_split.dissect(text, ahoc)
        if raw_parts is None:
            logger.warning("dissect returned None for text=%s", text)
            return text
    except Exception as e:
        logger.error("dissect failed for text=%s, error=%s", text, e)
        return text
    # except IndexError:
    #     return text
    
    if not raw_parts or len(raw_parts) < 2:
        return text

    # Otherwise continue as before
    parts = comp_split.merge_fractions(raw_parts)
    if not parts or len(parts) < 2: #make sure at least two segments are returned
        return text
    # if any(len(part) < 3 for part in parts): # # if any part is too short, return original text
    #     return text
    if len(parts[0]) <= 4 and len(parts[-1]) <= 4: # following konvens, if both elements are <=4 char, not hard to read
        return text
    
    # post-process each part: keep first letter, lowercase the rest
    #EXPERIMENTALparts = [p[0] + p[1:].lower() for p in parts]
    #Capitalize EACH part of the compound correctly
    parts = [p.capitalize() for p in parts]
    return MIDDLE_DOT.join(parts)




#outdated compound splitter

# def split_compound_word(text):
#     parts = doc_split.maximal_split(text)
#     if not parts or len(parts) < 2:
#         return text
#     if any(len(part) < 3 for part in parts):
#         return text
#     if len(parts[0]) <= 4 and len(parts[-1]) <= 4:
#         return text
#     return doc_split.MIDDLE_DOT.join(parts)

# ============== Casing helper function
# def casing_fix(doc: spacy.tokens.Doc) -> str:
#     """
#     Corrects sentence casing using linguistic information.
#     - Capitalizes the first token.
#     - Lowercases other tokens unless they are proper nouns or acronyms.
#     """
#     final_tokens = []
#     for i, token in enumerate(doc):
#         token_text = token.text
#         if i == 0:
#             # Always capitalize the first letter of the first token
#             final_tokens.append(token_text.capitalize())
#         # Preserve proper nouns (PROPN) and entities like persons/locations
#         elif token.pos_ == 'PROPN' or token.ent_type_ in {"PER", "LOC", "ORG"}:
#             final_tokens.append(token_text)
#         # Preserve acronyms (heuristic: all-caps and more than 1 letter)
#         elif token_text.isupper() and len(token_text) > 1:
#             final_tokens.append(token_text)
#         else:
#             # Lowercase all other tokens
#             final_tokens.append(token_text.lower())
    
#     #Join tokens with correct spacing based on the original Doc
#     #return "".join([tok.text_with_ws for tok in doc]).strip() 
#     return "".join([final_tokens[i] + doc[i].whitespace_ for i in range(len(final_tokens))]).strip()

def casing_fix(doc: spacy.tokens.Doc) -> str:
    """
    A token-based function to correct sentence casing for German.
    - Capitalizes the first word of every sentence.
    - Capitalizes all words tagged as NOUN or PROPN.
    - Lowercases other words (unless they are all-caps like acronyms).
    - Correctly returns the modified string.
    """
    final_words = []
    is_first_token = True
    # Iterate through each sentence detected by spaCy
    for sent in doc.sents:
        # Enumerate the tokens within each sentence
        for i, token in enumerate(sent):
            word = token.text

            # Rule 0: If a token is a compound split with a middle dot, or hyphen, fix its casing first.
            if '·' in word or '-' in word:
                # Split by either character, capitalize each part, and rejoin with a middle dot.
                parts = [p.capitalize() for p in re.split('[·-]', word)]
                word = '·'.join(parts)

            # Rule 1: Capitalize the first alphabetic token of the sentence.
            if i == 0 and token.is_alpha:
                word = word.capitalize()
            
            # Rule 2: In German, ALL nouns must be capitalized.
            elif token.pos_ in {"NOUN", "PROPN"}:
                word = word.capitalize()

            # Rule 3: Preserve acronyms that are fully uppercase.
            elif token.is_upper and len(word) > 1:
                pass  # Keep the word as is
            
            # Rule 4: Default to lowercasing for all other words.
            else:
                word = word.lower()

            # Append the corrected word AND its original trailing whitespace.
            # This correctly reconstructs the sentence.
            final_words.append(word)
            final_words.append(token.whitespace_)

            # FIX 2: Correctly update the flag for the NEXT token.
            # If the current token is sentence-ending punctuation, the next one is a new start.
            if token.is_sent_end:
                is_first_token = True

    return "".join(final_words).strip()
    

# == Text Replacement Functions ===
# NOT IN USE CURRENTLY
# Load the mapping from JSON
with open("resources/replace_words/alternative_woerter.json", encoding="utf-8") as f:
    mapped_dict = json.load(f)

# Build a regex that matches any key as a whole word
# Escape keys in case they contain regex metacharacters.
pattern = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in mapped_dict) + r")\b"
)
# Replacement callback
def _replacer(m):
    return mapped_dict[m.group(1)]
    
# One-pass replace function
def replace_easy_german(text: str) -> str:
    return pattern.sub(_replacer, text)
# =================================

# -- Simplifier Pipeline
class SimplifierPipeline:
    def __init__(self):
        self.uid = None # unique id for every sentence in a given file
        self.simplification_log = []  # Each entry will be a dict with orig, rule, simplified
        self.current_doc_name = None
        # Initialize rules
        self.rules = [
            {"condition": detect_punctuation, "action": clean_syntactic_punctuation},
            {"condition": has_apposition, "action": rewrite_apposition},
            {"condition": has_subordinate_clause, "action": simplify_subordinate},
            {"condition": has_coordinate_clause, "action": simplify_coordinate},
            {"condition": is_passive, "action": convert_passive_to_active},
            {"condition": has_disallowed_tense, "action": normalize_verb_tense}
        ]

    # def log_compound(self, uid, token, new_text):
    #     """Custom logging for compound splitting step."""
    #     applied = token["text"] != new_text
    #     self.log_step(uid,
    #                   token["text"],
    #                   "split_compound",
    #                   applied,
    #                   new_text)    

    def simplify_tokens(self, tokens):
        simplified = []
        #logger.debug("-------- \n tokens \n -------- \n%s", pformat(tokens))
        for token in tokens:
            text = token["text"]
            #logger.debug("-------- \n text \n -------- \n%s", pformat(text))

            # 1) Convert word to number if it's numeric
            applied_num = False
            if is_number(token):
                new_text = convert_word_to_number(text)
                applied_num = token["text"] != new_text

                self.log_step(
                    self.uid,
                    token["text"],           # original
                    "convert_word_to_number",
                    applied_num,             # True or False
                    new_text                 # final form
                )
                text = new_text
            else:
                text = token["text"]  # keep original text if not a number

            # 2) Split compound if applicable
            applied_compound_split = False
            if check_compound_split(token):
                new_text = split_compound_word(text, ahoc)  # Use the initialised compound splitter
                applied_compound_split = token["text"] != new_text
                text = new_text
                # Log the compound splitting step
                #self.log_compound(self.uid, token, text)

                self.log_step(
                self.uid,
                token["text"],           # original
                "split_compound",
                applied_compound_split,  # True or False
                text                     # final form
                )

            # Step 3: Substitute complex word with simpler synonym using predefined dict
            #text = replace_easy_german(text)

            # Copy and update token
            updated_token = token.copy()
            updated_token["text"] = text
            simplified.append(updated_token)
            #logger.debug("-------- \n simplified \n -------- \n%s", pformat(simplified))
        return simplified

    def log_step(self, uid, original, rule, applied, simplified, doc_name=None):
        # If 'simplified' is a list, log each separately for full traceability
        if doc_name is None:
            doc_name = self.current_doc_name # Use current doc name if not provided
            
        if isinstance(simplified, list):
            for item in simplified:
                entry = {
                    "uid": uid,
                    "original": str(original),
                    "rule": rule,
                    "applied": applied,
                    "simplified": str(item)
                }
                if doc_name:
                    entry["doc_name"] = doc_name
                self.simplification_log.append(entry)
        else:
            entry = {
                "uid": uid,
                "original": str(original),
                "rule": rule,
                "applied": applied,
                "simplified": str(simplified)
            }
            if doc_name:
                entry["doc_name"] = doc_name
            self.simplification_log.append(entry)
        
    def append_log_to_csv(self, csv_path="simplification_log.csv"):
        if not self.simplification_log:
            return  # nothing to write
        df = pd.DataFrame(self.simplification_log)
        file_exists = os.path.isfile(csv_path)
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
        self.simplification_log = []  # Optional: clear after writing

    def simplify_parsed_sentence(self, tokens):
        text = " ".join(token["text"] for token in tokens)
        text_sentences = [text] 
        simplified_sentences = []
        
        for text_sentence in text_sentences:
        #for token in tokens:
            
            doc_text_sentence = nlp(text_sentence)
            logger.debug("-------- \n INPUT: -------- %s\n", pformat(doc_text_sentence))
            #logger.debug("-------- \n token \n -------- \n%s", pformat(token))
            #logger.debug("-------- \n tokens \n -------- \n%s", pformat(tokens))

            logger.info("--------------------------")
            if detect_punctuation(doc_text_sentence):
                logger.info("Applying Now !!!: %s\n", doc_text_sentence)
                punc_applied = clean_syntactic_punctuation(doc_text_sentence)
                logger.info("clean_punctuation Applied: %s\n", punc_applied)
                if isinstance(punc_applied, str):
                    punc_applied = [punc_applied] # list of strings (sentences) or Doc
                logger.info("*** clean_punctuation Applied ***")

                #once rule was applied, save it to the log
                self.log_step(self.uid, doc_text_sentence, "clean_punctuation", True, punc_applied)
                
            else:
                punc_applied = [doc_text_sentence]
                logger.info("*** clean_punctuation Not Applied ***")
                self.log_step(self.uid, doc_text_sentence, "clean_punctuation", False, doc_text_sentence)

            logger.info("--------------------------")
            

            app_applied = []

            for sub_part in punc_applied:
                doc_sub_part = nlp(sub_part) #if isinstance(sub_part, str) else sub_part

                if has_apposition(doc_sub_part):
                    logger.info("Applying Now !!!: %s\n", doc_sub_part)
                    app_applied_subpart = rewrite_apposition(doc_sub_part)
                    logger.info("rewrite_apposition Applied: %s\n", app_applied_subpart)
                    if isinstance(app_applied_subpart, list):
                        app_applied.extend(app_applied_subpart)
                    else:
                        app_applied.append(app_applied_subpart)
                    self.log_step(self.uid, doc_sub_part, "rewrite_apposition", True, app_applied_subpart)
                else:
                    app_applied.append(doc_sub_part if hasattr(doc_sub_part, "text") else str(doc_sub_part))
                    logger.info("*** rewrite_apposition Not Applied ***")
                    self.log_step(self.uid, doc_sub_part, "rewrite_apposition", False, doc_sub_part if hasattr(doc_sub_part, "text") else str(doc_sub_part))
                    logger.info("--------------------------")

            # {"condition": has_apposition, "action": split_apposition}
            # app_applied = []
            # is_app_applied = False
            # for sub_part in punc_applied:
            #     doc_sub_part = nlp(sub_part) if isinstance(sub_part, str) else sub_part
            #     if has_apposition(doc_sub_part):
            #         logger.info("Applying Now !!! for -- %s\n", doc_sub_part)
            #         app_applied_sub_part = split_apposition(doc_sub_part)
            #         logger.info("split_apposition Applied: %s\n", app_applied_sub_part)
            #         app_applied.extend(app_applied_sub_part)
            #         is_app_applied = True
            #         logger.info("*** split_apposition Applied ***")
            #     else:
            #         app_applied.extend(doc_sub_part)
            # if not is_app_applied:
            #     app_applied = punc_applied
            #     logger.info("*** split_apposition Not Applied ***")
                
            logger.info("--------------------------")

            # {"condition": has_subordinate_clause, "action": simplify_subordinate}
            sub_applied = []
            is_sub_applied = False
            
            for sub_part in app_applied:
                doc_sub_part = nlp(sub_part) if isinstance(sub_part, str) else sub_part
                
                if has_subordinate_clause(doc_sub_part):
                    logger.info("Applying Now !!! for -- %s", doc_sub_part)
                    sub_applied_sub_part = simplify_subordinate(doc_sub_part)
                    logger.info("simplify_subordinate Applied: %s", sub_applied_sub_part)
                    sub_applied.extend(sub_applied_sub_part) # BECAUSER STRINGS
                    is_sub_applied = True
                    logger.info("*** simplify_subordinate Applied ***")
                    
                    # once rule was applied, save it to the log
                    self.log_step(self.uid, doc_text_sentence, "simplify_subordinate", True, sub_applied_sub_part)

                else:
                    #sub_applied.extend(doc_sub_part)
                    sub_applied.append(doc_sub_part) # KEEP OTHERWIESE, either string or doc #TODO
                    self.log_step(self.uid, doc_text_sentence, "simplify_subordinate", False, doc_sub_part)

            if not is_sub_applied:
                sub_applied = app_applied
                #sub_applied = punc_applied
                logger.info("*** simplify_subordinate Not Applied ***")
                
            logger.info("--------------------------")

            

            # {"condition": has_subordinate_clause, "action": simplify_subordinate}
            # coor_applied = []
            # is_coor_applied = False
            
            # for sub_part in sub_applied:
            #     doc_sub_part = nlp(sub_part) if isinstance(sub_part, str) else sub_part
                
            #     if has_coordinate_clause(doc_sub_part):
            #         logger.info("Applying Now !!! for -- %s", doc_sub_part)
            #         coor_applied_sub_part = simplify_coordinate(doc_sub_part)
            #         logger.info("simplify_coordinate Applied: %s", coor_applied_sub_part)
            #         #log
            #         self.log_step(self.uid, doc_sub_part, "simplify_coordinate", True, coor_applied_sub_part)
                    
            #         if isinstance(coor_applied_sub_part, list):
            #             coor_applied.extend(coor_applied_sub_part)
            #         else:
            #             coor_applied.append(coor_applied_sub_part)
            #         is_coor_applied = True
            #         logger.info("*** simplify_coordinate Applied ***")
                
            #     else:
            #         #coor_applied.extend(doc_sub_part)
            #         coor_applied.append(doc_sub_part) # KEEP OTHERWIESE, either string or doc #TODO
            #         self.log_step(self.uid, doc_sub_part, "simplify_coordinate", False, doc_sub_part)
                    
            # if not is_coor_applied:
            #     coor_applied = sub_applied
            #     logger.info("*** simplify_coordinate Not Applied ***")
                
            logger.info("--------------------------")

            
            
            #TODO
            """
            # Apply SVO reordering NEEDS WORK
            svo_applied = []

            for sub_part in coor_applied:
                doc_sub_part = nlp(sub_part) if isinstance(sub_part, str) else sub_part
                logger.info("Applying Now reorder_SVO for -- %s", doc_sub_part)
                svo_sentence = reorder_SVO(doc_sub_part)
                logger.info("reorder_SVO Applied: %s", svo_sentence)
                svo_applied.append(svo_sentence)
            
            logger.info("--------------------------")
            """
            passive_active_applied = []
            is_passive_to_active = False

            #for sub_part in svo_applied: #TODO
            for sub_part in sub_applied:
                doc_sub_part = nlp(sub_part) if isinstance(sub_part, str) else sub_part
                

                if is_passive(doc_sub_part):
                    logger.info("Applying Now !!! for -- %s", doc_sub_part)
                    pass_act_applied_sub_part = convert_passive_to_active(doc_sub_part)
                    logger.info("convert_passive_to_active Applied: %s", pass_act_applied_sub_part)
                    self.log_step(self.uid, doc_sub_part, "convert_passive_to_active", True, pass_act_applied_sub_part)

                    if isinstance(pass_act_applied_sub_part, list):
                        passive_active_applied.extend(pass_act_applied_sub_part)
                    else:
                        passive_active_applied.append(pass_act_applied_sub_part)
                    is_passive_to_active = True
                    logger.info("*** convert_passive_to_active Applied ***")
                    
                else:
                    passive_active_applied.append(doc_sub_part) #TODO
                    self.log_step(self.uid, doc_sub_part, "convert_passive_to_active", False, doc_sub_part)
            if not is_passive_to_active:
                passive_active_applied = sub_applied
                logger.info("*** convert_passive_to_active Not Applied ***")
                
           
            logger.info("--------------------------")
            
            # {"condition": has_disallowed_tense, "action": normalize_verb_tense}
            normalized_tense_applied = []
            is_normalized_tense = False

            for sub_part in passive_active_applied:
                doc_sub_part = nlp(sub_part) if isinstance(sub_part, str) else sub_part

                if has_disallowed_tense(doc_sub_part):
                    logger.info("Applying Now !!! for -- %s", doc_sub_part)
                    tense_applied_sub_part = normalize_verb_tense(doc_sub_part)
                    logger.info("normalize_verb_tense Applied: %s", tense_applied_sub_part)
                    self.log_step(self.uid, doc_sub_part, "normalize_verb_tense", True, tense_applied_sub_part)
                    # if it returns a list, extend, if it returns a string, append
                    if isinstance(tense_applied_sub_part, list):
                        normalized_tense_applied.extend(tense_applied_sub_part)
                    else:
                        normalized_tense_applied.append(tense_applied_sub_part)
                    is_normalized_tense = True
                    logger.info("*** normalized_tense Applied ***")

                else:
                    normalized_tense_applied.append(doc_sub_part) #TODO
                    self.log_step(self.uid, doc_sub_part, "normalize_verb_tense", False, doc_sub_part)


            if not is_normalized_tense:
                normalized_tense_applied = passive_active_applied
                logger.info("*** normalized_tense Not Applied ***")

            simplified_sentences.append(normalized_tense_applied)
            logger.debug("-------- \n simplified_sentence \n -------- \n%s", pformat(normalized_tense_applied))
        
        df = pd.DataFrame(self.simplification_log)

        #print(type(simplified_sentences))
        return simplified_sentences
        
    def simplify_from_lines(self, conll_lines, doc_name=None):
        sentences = parse_blocks(conll_lines)
        all_simplified_conll = []
        all_simplified_plain = []
        self.simplification_log = []
        self.current_doc_name = doc_name
        self.uid = 0

        # --- MAIN LOOP: Process each original sentence ---
        for tokens in sentences:
            self.uid += 1
            if (self.uid % 1000) == 0: #implement a progress tracker
                print("Progress:", self.uid)

            # 1. TOKEN-LEVEL SIMPLIFICATION (Numbers, Compounds)
            # Input: list of token dicts -> Output: list of token dicts
            simplified_tokens = self.simplify_tokens(tokens)

            # 2. SENTENCE-LEVEL REWRITING (Clauses, Passive Voice, etc.)
            # Input: list of token dicts -> Output: A messy list of strings, Docs, etc.
            rewritten_items = self.simplify_parsed_sentence(simplified_tokens)

            # 3. CLEANUP & FLATTEN
            # Take the messy list and produce a clean, flat list of final sentence strings.
            final_sentence_strings = []
            for sent in flatten_to_sentences(rewritten_items):
                text = sent.strip()
                if text and len(text.split()) >= 2:
                    final_sentence_strings.append(text)

            if not final_sentence_strings:
                continue

            # 4. Perform FINAL PROCESSING for all sentences
            # This list will hold the definitive, fully cased and processed strings.
            fully_processed_strings = []
            for text_sentence in final_sentence_strings:
                # Parse the clean sentence string ONCE.
                doc = nlp(text_sentence)

                # Further split if necessary (e.g., on semicolons).
                # This function should return a list of spaCy Doc/Span objects.
                segments = split_on_syntactic_punctuation(doc)

                for seg in segments:
                    #seg is currently a list of tokens, convert to Doc
                    if not seg:
                        continue
                    #convert list of tokens back to Doc
                    text_seg = "".join([token.text_with_ws for token in seg])
                    doc_seg = nlp(text_seg)
                    
                    
                    # Apply the final casing fix. This returns a string.
                    cased_text = casing_fix(doc_seg)
                    fully_processed_strings.append(cased_text)
                    
            # 5a. Collect plain text output
            plain_text_output = " ".join(fully_processed_strings)
            all_simplified_plain.append(final_cleanup(plain_text_output))

            # 5b. Collect CoNLL output
            for final_text in fully_processed_strings:
                final_doc = nlp(final_text)
                formatted = format_doc_to_conll(final_doc)
                all_simplified_conll.append(formatted)

            # 6. After all sentences are processed, append the log once
        if doc_name is None:
            doc_name = "unknown_doc"
        base_name = os.path.splitext(os.path.basename(doc_name))[0]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = f"simplification_logs/{base_name}_log_{timestamp}.csv"
        self.append_log_to_csv(csv_path)  # Save log after each document

        #return "\n".join(all_simplified)
        # Return both outputs (optionally, as a tuple or dict)
        return {
        "conll": "\n".join(all_simplified_conll),
        "plain": "\n".join(all_simplified_plain)
        }
    # def simplify_from_lines(self, conll_lines, doc_name=None):
    #     #logger.debug("-------- \n conll_lines \n -------- \n%s", pformat(conll_lines))
    #     sentences = parse_blocks(conll_lines)
    #     #logger.debug("-------- \n sentences \n -------- \n%s", pformat(sentences))
    #     all_simplified = []
    #     all_simplified_text = []  # Collect normal text output

    #     # Reset log at start of a batch/document
    #     self.simplification_log = []
    #     # Update doc_name within function
    #     self.current_doc_name = doc_name

        
        
    #     self.uid = 0
    #     for tokens in sentences:
    #         self.uid = self.uid + 1
    #         if (self.uid % 1000) == 0: #implement a progress tracker
    #             print ("Progress:", self.uid)
    #         #print ("********")
    #         #print (tokens)
    #         #print ("********")
            
              
    #         # 1)Token-level simplification -- Numbers, Compounds
    #         # Input: list of token dicts -> Output: list of token dicts
    #         simplified_tokens = self.simplify_tokens(tokens)
    #         #logger.debug("-------- \n simplified_tokens \n -------- \n%s", pformat(simplified_tokens))

    #         # 2)Sentence-level rewriting
    #         rewritten_sentences = self.simplify_parsed_sentence(simplified_tokens)
    #         logger.debug("-------- \n rewritten_sentences \n -------- \n%s", pformat(rewritten_sentences))

    #         # Extra step: flatten any nested lists
    #         # flat_rewritten_sentences = []
    #         # for x in rewritten_sentences:
    #         #     if isinstance(x, list):
    #         #         flat_rewritten_sentences.extend(x)
    #         #     else:
    #         #         flat_rewritten_sentences.append(x)

    #         # # Extra step: remove empty strings and short sentences
    #         # for rewritten_sent in flat_rewritten_sentences:
    #         #         text = rewritten_sent.strip()
    #         #         if not text or len(text.split()) < 2:
    #         #             continue 

    #         # 2.5) Cleanup and flatten.
    #         simplified_texts = [] #for all output sentences from this input
    #         for rewritten_sent in flatten_to_sentences(rewritten_sentences):
    #             text = rewritten_sent.strip()
    #             if not text or len(text.split()) < 2:
    #                 continue
    #             simplified_texts.append(text)

    #                 # NEW: Collect normal text output
    #                 #all_simplified_text.append(text)  # One line per simplified sentence

    #             # 3. Parse new rewritten sentence
    #             doc = nlp(text)

    #             # 4. further punctuation-based segmentation
    #             segments = split_on_syntactic_punctuation(doc)

    #             # 5. Apply casing fix and format each segment into CoNLL-style output
    #             for seg in segments:
    #                 fixed_casing = casing_fix(seg) # Apply casing fix
    #                 corrected_seg = nlp(fixed_casing) # Re-parse after casing fix
    #                 # Format to CoNLL
    #                 formatted = format_doc_to_conll(corrected_seg)
    #                 all_simplified.append(formatted)

    #             # 6.After processing all splits for this input,
    #             #join and save ONE output line
    #         joined = " ".join(simplified_texts)
    #         all_simplified_text.append(final_cleanup(joined))  # One line per simplified sentence
    
        #         # 6. After all sentences are processed, append the log once
        # if doc_name is None:
        #     doc_name = "unknown_doc"
        # base_name = os.path.splitext(os.path.basename(doc_name))[0]
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # csv_path = f"simplification_logs/{base_name}_log_{timestamp}.csv"
        # self.append_log_to_csv(csv_path)  # Save log after each document

        # #return "\n".join(all_simplified)
        # # Return both outputs (optionally, as a tuple or dict)
        # return {
        # #"conll": "\n".join(all_simplified),
        # #"plain": "\n".join(all_simplified_text)
        # }
    

    #If I wish to implement that all logs are to be saved in ONE CSV file at the end
    # comment out the log resetter in simplfy_from_lines beginning
    # only to reset the log variable when i want to start a new logging session
    
    # def save_all_logs(self, csv_path="simplification_log.csv"):
    #     if not self.simplification_log:
    #         return
    #     df = pd.DataFrame(self.simplification_log)
    #     file_exists = os.path.isfile(csv_path)
    #     df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

#     at the end of the text simplification pipeline add:

# pipeline = SimplifierPipeline()
# for file in all_my_input_files:
#     conll_lines = load_your_conll(file)
#     pipeline.simplify_from_lines(conll_lines)
# pipeline.save_all_logs("simplification_log.csv")