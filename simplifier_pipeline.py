import spacy
import os
import pandas as pd
import json
import re
import importlib
import text_simpl_utils
importlib.reload(text_simpl_utils)
from text_simpl_utils import *
#from compounds import analyze_compound

from word2num_de import word_to_number
from compound_split import doc_split

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
ahocs = comp_split.read_dictionary_from_file(utf8_file) #activate the compound_spliter

# ---

NUMBER_DICT = {
    "erster": "1", "zweiter": "2", "dritter": "3", "vierter": "4", "fünfter": "5",
    "sechster": "6", "siebter": "7", "achter": "8", "neunter": "9", "zehnter": "10",
    "eineinhalb": "1.5", "zweieinhalb": "2.5", "dreieinhalb": "3.5",
    "viereinhalb": "4.5", "fünfeinhalb": "5.5", "sechseinhalb": "6.5",
    "siebeneinhalb": "7.5", "achteinhalb": "8.5", "neuneinhalb": "9.5", "zehneinhalb": "10.5"
}

# -- Utlity Functions
def is_number(token):
    text_lower = token['text'].lower()
    return text_lower != "ein" and (text_lower in NUMBER_DICT or token['pos'] == "NUM")

def convert_word_to_number(text):
    text_lower = text.lower()
    if text_lower in NUMBER_DICT:
        return NUMBER_DICT[text_lower]
    if text.isdigit():
        return text
    try:
        return str(word_to_number(text_lower))
    except Exception:
        return text #fallback, return as it was

def check_compound_split(token):
    return (
        token['pos'] in {"NOUN", "PROPN"} and
        len(token['text']) > 5 and
        token['ent_type'] not in {"PER", "LOC", "ORG"}
    )

MIDDLE_DOT = "·"
def split_compound_word(text, ahocs):
    parts = comp_split.merge_fractions(comp_split.dissect(text, ahocs))
    if not parts or len(parts) < 2:
        return text
    if any(len(part) < 3 for part in parts):
        return text
    if len(parts[0]) <= 4 and len(parts[-1]) <= 4:
        return text
    return MIDDLE_DOT.join(parts)

def split_compound_word(text):
    parts = doc_split.maximal_split(text)
    if not parts or len(parts) < 2:
        return text
    if any(len(part) < 3 for part in parts):
        return text
    if len(parts[0]) <= 4 and len(parts[-1]) <= 4:
        return text
    return doc_split.MIDDLE_DOT.join(parts)

# == Text Replacement Functions ===
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
# ==

# -- Simplifier Pipeline
class SimplifierPipeline:
    def __init__(self):
        self.simplification_log = []  # Each entry will be a dict with orig, rule, simplified
        self.current_doc_name = None
        # Initialize rules
        self.rules = [
            {"condition": should_split_on_punctuation, "action": split_on_punctuation},
            {"condition": has_apposition, "action": split_apposition},
            {"condition": has_subordinate_clause, "action": simplify_subordinate},
            {"condition": has_coordinate_clause, "action": simplify_coordinate},
            {"condition": is_passive, "action": convert_passive_to_active},
            {"condition": has_disallowed_tense, "action": normalize_verb_tense}
        ]

    def simplify_tokens(self, tokens):
        simplified = []
        #logger.debug("-------- \n tokens \n -------- \n%s", pformat(tokens))
        for token in tokens:
            text = token["text"]
            #logger.debug("-------- \n text \n -------- \n%s", pformat(text))

            # 1) Convert word to number if it's numeric
            if is_number(token):
                text = convert_word_to_number(text)

            # 2) Split compound if applicable
            if check_compound_split(token):
                text = split_compound_word(text)

            # Step 3: Substitute complex word with simpler synonym using predefined dict
            text = replace_easy_german(text)

            # Copy and update token
            updated_token = token.copy()
            updated_token["text"] = text
            simplified.append(updated_token)
            #logger.debug("-------- \n simplified \n -------- \n%s", pformat(simplified))
        return simplified

    def log_step(self, original, rule, applied, simplified, doc_name=None):
        # If 'simplified' is a list, log each separately for full traceability
        if doc_name is None:
            doc_name = self.current_doc_name # Use current doc name if not provided
            
        if isinstance(simplified, list):
            for item in simplified:
                entry = {
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
            
            #logger.debug("-------- \n text_sentence \n -------- \n%s", pformat(text_sentence))
            doc_text_sentence = nlp(text_sentence)
            logger.debug("-------- \n INPUT: -------- %s\n", pformat(doc_text_sentence))
            #logger.debug("-------- \n token \n -------- \n%s", pformat(token))
            #logger.debug("-------- \n tokens \n -------- \n%s", pformat(tokens))

            logger.info("--------------------------")

            # {"condition": should_split_on_punctuation, "action": split_on_punctuation}
            if should_split_on_punctuation(doc_text_sentence):
                logger.info("Applying Now !!!: %s\n", doc_text_sentence)
                punc_applied = split_on_punctuation(doc_text_sentence)
                logger.info("split_on_punctuation Applied: %s\n", punc_applied)
                logger.info("*** split_on_punctuation Applied ***")

                #once rule was applied, save it to the log
                self.log_step(doc_text_sentence, "split_on_punctuation", True, punc_applied)
                
            else:
                punc_applied = [doc_text_sentence]
                logger.info("*** split_on_punctuation Not Applied ***")
                self.log_step(doc_text_sentence, "split_on_punctuation", False, doc_text_sentence)

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
            
            #for sub_part in app_applied:
            for sub_part in punc_applied:
                doc_sub_part = nlp(sub_part) if isinstance(sub_part, str) else sub_part
                
                if has_subordinate_clause(doc_sub_part):
                    logger.info("Applying Now !!! for -- %s", doc_sub_part)
                    sub_applied_sub_part = simplify_subordinate(doc_sub_part)
                    logger.info("simplify_subordinate Applied: %s", sub_applied_sub_part)
                    sub_applied.extend(sub_applied_sub_part) # BECAUSER STRINGS
                    is_sub_applied = True
                    logger.info("*** simplify_subordinate Applied ***")
                    
                    # once rule was applied, save it to the log
                    self.log_step(doc_text_sentence, "simplify_subordinate", True, sub_applied_sub_part)

                else:
                    #sub_applied.extend(doc_sub_part)
                    sub_applied.append(doc_sub_part) # KEEP OTHERWIESE, either string or doc #TODO
                    self.log_step(doc_text_sentence, "simplify_subordinate", False, doc_sub_part)

            if not is_sub_applied:
                #sub_applied = app_applied
                sub_applied = punc_applied
                logger.info("*** simplify_subordinate Not Applied ***")
                
            logger.info("--------------------------")

            

            # {"condition": has_subordinate_clause, "action": simplify_subordinate}
            coor_applied = []
            is_coor_applied = False
            
            for sub_part in sub_applied:
                doc_sub_part = nlp(sub_part) if isinstance(sub_part, str) else sub_part
                
                if has_coordinate_clause(doc_sub_part):
                    logger.info("Applying Now !!! for -- %s", doc_sub_part)
                    coor_applied_sub_part = simplify_coordinate(doc_sub_part)
                    logger.info("simplify_coordinate Applied: %s", coor_applied_sub_part)
                    #log
                    self.log_step(doc_sub_part, "simplify_coordinate", True, coor_applied_sub_part)
                    
                    if isinstance(coor_applied_sub_part, list):
                        coor_applied.extend(coor_applied_sub_part)
                    else:
                        coor_applied.append(coor_applied_sub_part)
                    is_coor_applied = True
                    logger.info("*** simplify_coordinate Applied ***")
                
                else:
                    #coor_applied.extend(doc_sub_part)
                    coor_applied.append(doc_sub_part) # KEEP OTHERWIESE, either string or doc #TODO
                    self.log_step(doc_sub_part, "simplify_coordinate", False, doc_sub_part)
                    
            if not is_coor_applied:
                coor_applied = sub_applied
                logger.info("*** simplify_coordinate Not Applied ***")
                
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
            for sub_part in coor_applied:
                doc_sub_part = nlp(sub_part) if isinstance(sub_part, str) else sub_part
                

                if is_passive(doc_sub_part):
                    logger.info("Applying Now !!! for -- %s", doc_sub_part)
                    pass_act_applied_sub_part = convert_passive_to_active(doc_sub_part)
                    logger.info("convert_passive_to_active Applied: %s", pass_act_applied_sub_part)
                    self.log_step(doc_sub_part, "convert_passive_to_active", True, pass_act_applied_sub_part)

                    if isinstance(pass_act_applied_sub_part, list):
                        passive_active_applied.extend(pass_act_applied_sub_part)
                    else:
                        passive_active_applied.append(pass_act_applied_sub_part)
                    is_passive_to_active = True
                    logger.info("*** convert_passive_to_active Applied ***")
                    
                else:
                    passive_active_applied.append(doc_sub_part) #TODO
                    self.log_step(doc_sub_part, "convert_passive_to_active", False, doc_sub_part)
            if not is_passive_to_active:
                passive_active_applied = coor_applied
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
                    self.log_step(doc_sub_part, "normalize_verb_tense", True, tense_applied_sub_part)
                    # if it returns a list, extend, if it returns a string, append
                    if isinstance(tense_applied_sub_part, list):
                        normalized_tense_applied.extend(tense_applied_sub_part)
                    else:
                        normalized_tense_applied.append(tense_applied_sub_part)
                    is_normalized_tense = True
                    logger.info("*** normalized_tense Applied ***")

                else:
                    normalized_tense_applied.append(doc_sub_part) #TODO
                    self.log_step(doc_sub_part, "normalize_verb_tense", False, doc_sub_part)


            if not is_normalized_tense:
                normalized_tense_applied = passive_active_applied
                logger.info("*** normalized_tense Not Applied ***")

            simplified_sentences.append(normalized_tense_applied)
            logger.debug("-------- \n simplified_sentence \n -------- \n%s", pformat(normalized_tense_applied))
        
        df = pd.DataFrame(self.simplification_log)
        df.to_csv("simplification_log.csv", index=False)

        #print(type(simplified_sentences))
        return simplified_sentences
        
    def simplify_from_lines(self, conll_lines, doc_name=None):
        #logger.debug("-------- \n conll_lines \n -------- \n%s", pformat(conll_lines))
        sentences = parse_blocks(conll_lines)
        #logger.debug("-------- \n sentences \n -------- \n%s", pformat(sentences))
        all_simplified = []
        all_simplified_text = []  # Collect normal text output

        # Reset log at start of a batch/document
        self.simplification_log = []
        # Update doc_name within function
        self.current_doc_name = doc_name

        for tokens in sentences:
              
            # 1)Token-level simplification
            simplified_tokens = self.simplify_tokens(tokens)
            #logger.debug("-------- \n simplified_tokens \n -------- \n%s", pformat(simplified_tokens))

            # 2)Sentence-level rewriting
            rewritten_sentences = self.simplify_parsed_sentence(simplified_tokens)
            logger.debug("-------- \n rewritten_sentences \n -------- \n%s", pformat(rewritten_sentences))

            # Extra step: flatten any nested lists
            # flat_rewritten_sentences = []
            # for x in rewritten_sentences:
            #     if isinstance(x, list):
            #         flat_rewritten_sentences.extend(x)
            #     else:
            #         flat_rewritten_sentences.append(x)

            # # Extra step: remove empty strings and short sentences
            # for rewritten_sent in flat_rewritten_sentences:
            #         text = rewritten_sent.strip()
            #         if not text or len(text.split()) < 2:
            #             continue 

            simplified_texts = [] #for all output sentences from this input
            for rewritten_sent in flatten_to_sentences(rewritten_sentences):
                text = rewritten_sent.strip()
                if not text or len(text.split()) < 2:
                    continue
                simplified_texts.append(text)

                    # NEW: Collect normal text output
                    #all_simplified_text.append(text)  # One line per simplified sentence

                # 3. Parse new rewritten sentence
                doc = nlp(text)

                # 4. further punctuation-based segmentation
                segments = split_on_syntactic_punctuation(doc)

                # 5. Format each segment into CoNLL-style output
                for seg in segments:
                    formatted = format_doc_to_conll(seg)
                    all_simplified.append(formatted)

                # 6.After processing all splits for this input,
                #join and save ONE output line
            joined = " ".join(simplified_texts)
            all_simplified_text.append(joined)  # One line per simplified sentence

        # 6. After all sentences are processed, append the log once
        if doc_name is None:
            doc_name = "unknown_doc"
        base_name = os.path.splitext(os.path.basename(doc_name))[0]
        csv_path = f"simplification_logs/{base_name}_log.csv"
        self.append_log_to_csv(csv_path)  # Save log after each document

        #return "\n".join(all_simplified)
        # Return both outputs (optionally, as a tuple or dict)
        return {
        "conll": "\n".join(all_simplified),
        "plain": "\n".join(all_simplified_text)
        }
    

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