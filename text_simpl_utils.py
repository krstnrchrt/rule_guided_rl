import os
import re
import csv
import spacy
from spacy.tokens import Span
from pattern.de import conjugate
from pattern.de import INFINITIVE, PRESENT, SG, SUBJUNCTIVE, PAST, PARTICIPLE, SG, PL

import logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# -- Utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_ist_verbs_from_csv(csv_path):
    hilfsverb_dict = {}
    with open(csv_path, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = row["Infinitive"].strip().lower()
            hilfsverb = row["Hilfsverb"].strip().lower()
            hilfsverb_dict[lemma] = hilfsverb
    return hilfsverb_dict

# Load once at start
AUX_DICT = os.path.join(SCRIPT_DIR, "resources", "verbs.csv")
AUX_VERB_DICT = load_ist_verbs_from_csv(AUX_DICT)

# -- CoNLL format Sentence Processing

def parse_blocks(lines):
    sentences = []
    current = []

    for line in lines:
        line = line.strip()
        if line == "<s>":
            current = []
        elif line == "</s>":
            if current:
                sentences.append(current)
        elif line:
            parts = line.split("\t")
            if len(parts) >= 8:
                token = {
                    "text": parts[0],
                    "lemma": parts[1],
                    "pos": parts[2],
                    "dep": parts[3],
                    "head": parts[4],
                    "morph": parts[5],
                    "is_stop": parts[6],
                    "ent_type": parts[7],
                    "raw": line
                }
                current.append(token)
    return sentences

def split_on_syntactic_punctuation(doc):
    result = []
    buffer = []
    i = 0

    while i < len(doc):
        tok = doc[i]
        buffer.append(tok)

        if tok.text == "–" and not is_likely_compound(doc, i):
            result.append(buffer[:-1])
            buffer = []

        elif tok.text == ";":
            result.append(buffer)
            buffer = []

        elif tok.text == ":":
            rest = doc[i + 1:]
            if any(t.dep_ == "ROOT" for t in rest):
                result.append(buffer)
                buffer = []

        i += 1

    if buffer:
        result.append(buffer)

    return result

# def format_parsed_segments(segments):
#     out = []
#     for seg in segments:
#         out.append("<s>")
#         out.extend(tok["raw"] for tok in seg)
#         out.append("</s>\n")
#     return "\n".join(out)

# -- Utility Functions
def format_doc_to_conll(doc):
    lines = ["<s>"]
    for tok in doc:
        lines.append("\t".join([
            tok.text, tok.lemma_, tok.pos_, tok.dep_, tok.head.text,
            tok.morph.to_json(), str(tok.is_stop), tok.ent_type_ or "-"
        ]))
    lines.append("</s>\n")
    return "\n".join(lines)

def flatten_to_sentences(lst):
    """Flatten a nested list of strings or objects with 'text' attribute to a flat list of sentences."""
    out = []
    for item in lst:
        if isinstance(item, list):
            out.extend(flatten_to_sentences(item))
        elif isinstance(item, str):
            out.append(item)
        elif hasattr(item, "text"):
            out.append(item.text)
        else:
            out.append(str(item))
    return out

# -- Utility Functiona

def is_likely_compound(doc, idx):
    if idx == 0 or idx >= len(doc) - 1:
            return False
    left = doc[idx - 1].pos_
    right = doc[idx + 1].pos_
    return left in {"PROPN", "NOUN"} and right in {"PROPN", "NOUN"}

def get_subject(doc):
    # Returns the subject string of the sentence (or empty)
    for tok in doc:
        if tok.dep_ in ("nsubj", "sb"):
            return tok.text
    return ""

# -- Sentence-Level Simplification Rules

nlp = spacy.load("de_core_news_lg")

SUBORDINATE_MARKERS = {
    "weil": "Deshalb", "da": "Denn", #causal
    "obwohl": "trotzdem", #concessive
    "sodass": "deshalb", #consecutive
    "damit": "dazu", "um":"dazu", #final
    "nachdem": "dann", "bevor": "dann", #temporal
    "seit": "dann", "während": "dann",
    "sobald": "dann", "als": "dann"
}
COORD_CONJ = {"oder", "aber", "dennoch"} #took "und" out


def reorder_SVO(doc):
    import re

    subordinators = set([
        "weil", "da", "obwohl", "sodass", "damit", "um", "nachdem", "bevor",
        "seit", "während", "sobald", "als"
    ])
    coordinators = set(["und", "oder", "aber", "dennoch", "allerdings", "doch"])

    # Try to detect fragment: no verb or subject, short clause, etc.
    has_subj = any(tok.dep_ == "sb" for tok in doc)
    finite_verbs = [tok for tok in doc if (tok.pos_ in {"VERB", "AUX"} and "Fin" in tok.morph.get("VerbForm"))]
    has_verb = bool(finite_verbs)

    if not (has_subj and has_verb):
        # Just return the clause as text (to avoid scrambling fragments)
        return doc.text.strip()

    # Clause type detection
    first_token = doc[0].text.lower()
    if any(tok.dep_ == "cp" and tok.text.lower() in subordinators for tok in doc):
        clause_type = "subordinate"
    elif first_token in coordinators:
        clause_type = "coordinate"
    else:
        clause_type = "main"

    # Roles
    subj = [t.text for t in doc if t.dep_ == "sb"]
    finite_verbs = [t.text for t in doc if (t.pos_ in {"VERB", "AUX"} and "Fin" in t.morph.get("VerbForm"))]
    auxiliaries = [t.text for t in doc if t.pos_ == "AUX" and "Fin" not in t.morph.get("VerbForm")]
    objects = [t.text for t in doc if t.dep_ in {"oa", "da", "op"}]
    predcomps = [t.text for t in doc if t.dep_ in {"pd", "oc"}]
    adverbials = [t.text for t in doc if t.dep_ in {"mo", "advmod"}]
    punctuation = [t.text for t in doc if t.is_punct]

    core = set(subj + finite_verbs + auxiliaries + objects + predcomps + adverbials + punctuation)
    rest = [t.text for t in doc if t.text not in core]

    # Reordering logic
    if clause_type == "subordinate":
        # Subordinate: subord (if present) + subject + objects/adverbs + all verbs (aux + finite) at end
        subord = [t.text for t in doc if t.dep_ == "cp" and t.text.lower() in subordinators]
        verbs = auxiliaries + finite_verbs
        parts = subord + subj + objects + adverbials + predcomps + rest + verbs
    elif clause_type == "coordinate":
        # Coordinator + SVO as for main
        coord = [doc[0].text] if first_token in coordinators else []
        parts = coord + subj + finite_verbs + auxiliaries + objects + predcomps + adverbials + rest
    else:  # main
        parts = subj + finite_verbs + auxiliaries + objects + predcomps + adverbials + rest

    # Remove duplicates (keep order)
    seen = set()
    ordered = []
    for w in parts:
        if w not in seen:
            ordered.append(w)
            seen.add(w)

    # Add punctuation (single at end)
    if punctuation:
        if ordered and ordered[-1] not in punctuation:
            ordered += punctuation
    sent = " ".join(ordered)
    sent = re.sub(r'\s+([.?!,])', r'\1', sent).strip()
    return sent



# Split on punctuation
def should_split_on_punctuation(doc):
    return any(tok.text in {"–", ";", ":"} for tok in doc)

def split_on_punctuation(doc):
    text = doc.text
    parts = re.split(r"\s*–\s*", text)
    return [part.strip() for part in parts if part.strip()]

# Split comma separated appositions
def has_apposition(doc):
    return any(tok.dep_ == 'app' for tok in doc)

def split_apposition(doc):
    for tok in doc:
        if tok.dep_ == "app":
            head = tok.head
            appo = list(tok.subtree)
            appo_phrase = " ".join([t.text for t in appo])
            main = doc.text.replace("," + appo_phrase, "").strip()
            appo_sent = f"{head.text} ist {appo_phrase}."
            return [main, appo_sent]
    # If no apposition found, just return original sentence
    return [doc.text]

# -- Subordinate Clause Detection and Simplification
def has_subordinate_clause(doc):
    for tok in doc:
        if (tok.text.lower() in SUBORDINATE_MARKERS and 
            (tok.dep_ in "cp" or tok.pos_ == "SCONJ")):
            return tok
    return None

def extract_clause_spans(doc, marker_token):
    # Subordinate clause = subtree of marker's HEAD (the main verb of subclause)
    sub_head = marker_token.head
    sub_clause_tokens = list(sub_head.subtree)
    sub_clause_span = doc[sub_clause_tokens[0].i:sub_clause_tokens[-1].i+1]
    # Main clause: all tokens not in subordinate clause span
    main_clause_tokens = [tok for tok in doc if tok not in sub_clause_span]
    return sub_clause_span, main_clause_tokens

def clean_subordinate_text(sub_clause_span, marker_token):
    return " ".join(
        tok.text for tok in sub_clause_span 
        if tok.i != marker_token.i and tok.text != ","
    )

def build_main_clause(main_clause_tokens, marker_token):
    """Build main clause and add connective"""
    connective = SUBORDINATE_MARKERS[marker_token.text.lower()]
    main_text = " ".join(tok.text for tok in main_clause_tokens if tok.text != ",")
    return f"{connective.capitalize()} {main_text}"

def add_period_and_strip(text):
    text = text.strip().rstrip(',')
    if not text.endswith('.'):
        text += '.'
    return text

def handle_um_zu(sub_span, main_subject):
    # For "um ... zu" construction, add subject if missing
    sub_text = " ".join(tok.text for tok in sub_span if tok.text != ",")
    if "um" in [tok.text.lower() for tok in sub_span] and "zu" in [tok.text.lower() for tok in sub_span]:
        # Check if subject is missing (very basic: look for nsubj)
        if not any(tok.dep_ in ("nsubj", "sb") for tok in sub_span):
            # Insert subject right after "um"
            parts = sub_text.split()
            try:
                um_idx = parts.index("um")
                parts.insert(um_idx+1, main_subject)
                sub_text = " ".join(parts)
            except ValueError:
                pass
    return sub_text

def simplify_subordinate(doc):
    marker_token = has_subordinate_clause(doc)
    if not marker_token:
        return [doc.text]
    
    sub_span, main_clause_tokens = extract_clause_spans(doc, marker_token)
    
    #Special case for "um..zu"
    main_subject = get_subject(doc)
    if marker_token.text.lower() == "um":
        sub_text = handle_um_zu(sub_span, main_subject)
    else:
        # Clean subordinate clause text, remove marker and commas
        sub_text = clean_subordinate_text(sub_span, marker_token)
    
    # Build main clause text with connective
    main_text = build_main_clause(main_clause_tokens, marker_token)
    
    # Ensure periods, no double dots
    sub_text = add_period_and_strip(sub_text)
    main_text = add_period_and_strip(main_text)
    
    return [sub_text, main_text]

# -- Coordinate Clause Detection and Simplification

def has_coordinate_clause(doc): #
    """Check if the document contains a coordinate clause."""
    return any(tok.dep_ == "cd" and tok.text.lower() in COORD_CONJ for tok in doc)

def get_subject_phrase(token):
    """Return a simple subject phrase for a verb token (remove adjectives, preps)."""
    # Look for the subject
    for child in token.children:
        if child.dep_ == "sb":
            # Only keep the head word (e.g. 'Mitarbeiter') or its noun chunk
            if child.n_lefts > 0 or child.n_rights > 0:
                # Reconstruct a minimal noun phrase (ignoring preps, adjectives, genitives, etc.)
                chunk_tokens = [t for t in child.subtree if t.pos_ in ("NOUN", "PROPN", "PRON")]
                return " ".join([t.text for t in chunk_tokens])
            return child.text
    return None

def extract_subtree_span(token):
    subtree = list(token.subtree)
    if not subtree:
        return ""
    start = min([t.i for t in subtree])
    end = max([t.i for t in subtree]) + 1
    return token.doc[start:end]

def simplify_coordinate(doc):
    clauses = []
    conj_indices = [i for i, t in enumerate(doc) if t.text.lower() in COORD_CONJ and t.dep_ == "cd"] #check fot coordinating conjunction
    
    if not conj_indices:
        return [doc]
        
    start = 0
    for idx, conj_idx in enumerate(conj_indices):
        # Everything before this conjunction (since start)
        first_clause = doc[start:conj_idx]
        # The conjunction itself
        conj = doc[conj_idx].text
        # Everything after this conjunction, up to next conjunction or end
        next_conj_idx = conj_indices[idx + 1] if idx + 1 < len(conj_indices) else len(doc)
        second_clause = doc[conj_idx + 1:next_conj_idx]

        # --- Clean up the first clause ---
        first_text = " ".join([t.text for t in first_clause]).strip().rstrip(",")
        # Only add period if not already present and not empty
        if first_text and not first_text.endswith('.'):
            first_text += "."
        if first_text:
            clauses.append(first_text)

        # --- Clean up the second clause ---
        second_text = " ".join([t.text for t in second_clause]).strip().lstrip(",").rstrip(",")
        # Only add period if not already present and not empty
        if second_text:
            if not second_text.endswith('.'):
                second_text += "."
            # Add conjunction at start (capitalize for Leichte Sprache)
            second_sentence = f"{conj.capitalize()} {second_text}"
            clauses.append(second_sentence)

        start = next_conj_idx

    return clauses


# -- Passive Voice Detection and Conversion

def get_perfekt_aux_for_verb(lemma):
    aux = AUX_VERB_DICT.get(lemma, "haben")
    if "/" in aux:
        aux = aux.split("/")[0]
    return aux

# List of verbs that take 'ist' as auxiliary in Perfekt
#IST_VERBS = {"gehen", "kommen", "bleiben", "fahren", "fallen", "sterben", "sein", "werden", "aufstehen", "aufwachen"}
# def select_auxiliary(lemma):
#     """Include subject's number/number/gender/preson"""
#     if lemma in IST_VERBS:
#         return "ist"
#     return "hat"
#OUTDATED perfect verb lookup

def is_passive(doc):
    # Passive: Aux ("werden"/"sein") + participle
    has_passive_aux = any(tok.lemma_ in {"werden", "sein"} for tok in doc if tok.pos_ == "AUX")
    has_participle = any(tok.pos_ == "VERB" and "Part" in tok.morph.get("VerbForm", []) for tok in doc)
    return has_passive_aux and has_participle

def convert_passive_to_active(doc):
    # 1) Get the patient as the full phrase (objec in avtive)
    patient = None
    patient_tokens = []
    for tok in doc:
        if tok.dep_ == "sb":
            # get the whole noun phrase for the patient
            # (tokens in subtree of the subject)
            patient_tokens = [t.text for t in tok.subtree]
            patient = " ".join(patient_tokens)
            break

    # 2) Find main verb in participle form adn lemma
    participle = None
    verb_lemma = None
    for tok in doc:
        if tok.pos_ == "VERB" and "Part" in tok.morph.get("VerbForm", []):
            participle = tok.text
            verb_lemma = tok.lemma_
            break
        if tok.pos_ == "VERB":
            verb_lemma = tok.lemma_

    # 3)Find agent introduced by 'von'
    agent = None
    for tok in doc:
        if tok.text.lower() == "von":
            # get the token after "von" (usually the agent)
            # agent_tokens = [t for t in tok.subtree if t.text.lower() != "von"]
            # agent = " ".join(t.text for t in agent_tokens)
            agent_tokens = [t.text for t in tok.subtree if t.text.lower() != "von"]
            agent = " ".join(agent_tokens)
            break

    # 4)Choose auxiliary
    aux_lemma = get_perfekt_aux_for_verb(verb_lemma)
    aux = get_aux_form(aux_lemma, agent)

    # 5)Build simplified active sentence
    subject = agent if agent else "Man"
    if not patient:
        patient = "" #fallback

    # If there is a participle, use Perfekt
    if participle:
        return f"{subject} {aux} {patient} {participle}."
    #Fallback - if no participla, aplly präteritum
    if verb_lemma:
        prat_apply = conjugate(verb_lemma, PAST, person=3, number=SG)
        return f"{subject} {prat_apply} {patient}."
    
    # If no participle or verb, return original text
    return doc.text

def has_disallowed_tense(doc):
    for tok in doc:
        if tok.pos_ in ("VERB", "AUX"):
            tense = tok.morph.get("Tense")
            form = tok.morph.get("VerbForm")
            mood = tok.morph.get("Mood")
            if ("Pres" not in tense and "Part" not in form) or ("Sub" in mood):
                return True
    return False

def find_subject(doc):
    for tok in doc:
        if tok.dep_ == "sb":  # parsed subject
            return tok
    return None

def get_aux_form(aux_lemma, subj):
    """
    Determines the correct auxiliary verb form for 'haben' or 'sein'
    based on subject person & number.
    FIRST LEVEL PATCH - 
    """
    if subj is None or isinstance(subj, str):
        # fallback to 3rd person singular
        return "ist" if aux_lemma == "sein" else "hat"
    else:
        number = subj.morph.get("Number")
        person = subj.morph.get("Person")

    if aux_lemma == "sein":
        if person == "1" and number == "Sing":
            return "bin"
        if person == "2" and number == "Sing":
            return "bist"
        if person == "3" and number == "Sing":
            return "ist"
        if person == "1" and number == "Plur":
            return "sind"
        if person == "2" and number == "Plur":
            return "seid"
        if person == "3" and number == "Plur":
            return "sind"
        return "ist"
    else:  # aux_lemma == "haben"
        if person == "1" and number == "Sing":
            return "habe"
        if person == "2" and number == "Sing":
            return "hast"
        if person == "3" and number == "Sing":
            return "hat"
        if person == "1" and number == "Plur":
            return "haben"
        if person == "2" and number == "Plur":
            return "habt"
        if person == "3" and number == "Plur":
            return "haben"
        return "hat"


def normalize_verb_tense(doc):
    """
    Converts:
    - Präteritum → Perfekt (hat/ist + Partizip II)
    - Futur → Präsens
    - Konjunktiv  → Perfekt
    Leaves verbs already in Präsens or Partizip unchanged.
    Returns simplified sentence as string.
    """
    subj = find_subject(doc)
    subject_token = subj.text if subj else ""
    person = int(subj.morph.get("Person")[0]) if subj and subj.morph.get("Person") else 3
    number = subj.morph.get("Number")[0] if subj and subj.morph.get("Number") else "Sing"

    new_tokens = []
    # Initialize variables for participle and auxiliary
    participle = None
    aux = None
    applied_perfekt = False

    for tok in doc:
        # if tok.pos_ != "VERB":
        #     new_tokens.append(tok.text)
        #     continue

        pos = tok.pos_
        lemma = tok.lemma_
        tense = tok.morph.get("Tense", [])
        mood = tok.morph.get("Mood", [])
        verbform = tok.morph.get("VerbForm", [])
        # logger.info(f"Token: {tok.text}, POS: {tok.pos_}, Lemma: {lemma}, Tense: {tense}, Mood: {mood}, VerbForm: {verbform}")
        # logger.info(f"Should be AUX{tok.pos_})")
        # print(f"Should be AUX {tok.pos_}, mood {mood})")

        # --- CASE 4.1: Convert auxiliary in Konjunktiv to indicativ
        if pos == "AUX" and "Sub" in mood:
            # Get indicative form (present)
            indicative = conjugate(lemma, PRESENT, person=person, number=SG if number == "Sing" else PL)
            new_tokens.append(indicative if indicative else lemma)
            continue   
        
        # --- CASE 1: Already Präsens or Partizip II → skip
        if "Pres" in tense or "Part" in verbform:
            new_tokens.append(tok.text)
            continue

        # --- CASE 2: Präteritum → Perfekt
        if "Past" in tense:
            participle = conjugate(lemma, PAST+PARTICIPLE)
            aux_lemma = get_perfekt_aux_for_verb(lemma)
            aux = get_aux_form(aux_lemma, subj)
            applied_perfekt = True
            continue

        # --- CASE 3: Futur I or II → Präsens
        if "Fut" in tense:
            present = conjugate(lemma, PRESENT, person=person, number=SG if number == "Sing" else PL)
            new_tokens.append(present if present else lemma)
            continue

        # --- CASE 4: Konjunktiv → Perfekt
        if pos == "VERB" and "Sub" in mood:
            participle = conjugate(lemma, PAST+PARTICIPLE)
            aux_lemma = get_perfekt_aux_for_verb(lemma)
            aux = get_aux_form(aux_lemma, subj)
            applied_perfekt = True
            continue
         


        # --- Fallback: keep original
        new_tokens.append(tok.text)
    # If we applied Perfekt, append auxiliary and participle
        # If Perfekt was constructed, put aux after subject, participle at end
    if applied_perfekt and aux and participle:
        # Move punctuation to the end if present
        punct = ""
        if new_tokens and new_tokens[-1] in {".", "!", "?"}:
            punct = new_tokens.pop()
        # Output: subj + aux + [rest] + participle + punct
        # If subject is first in tokens, don't duplicate it
        # Remove subject from tokens_out if present (to avoid duplication)
        tokens_out = [t for t in new_tokens if t != subject_token]
        result = " ".join([subject_token, aux] + tokens_out + [participle])
        if punct:
            result += punct
        return result.strip()
    else:
        return " ".join(new_tokens)

    

# -- Recursive simplification function
def simplify_sentence(doc):
    """ Input: spacy parsed doc
    Output: list of simplified sentences"""

    if should_split_on_punctuation(doc):
        return split_on_syntactic_punctuation(doc)
    elif has_apposition(doc):
        return split_apposition(doc)
    elif has_subordinate_clause(doc):
        return simplify_subordinate(doc)
    elif has_coordinate_clause(doc):
        return simplify_coordinate(doc)
    else:
        return [doc.text] #default, return as it is
