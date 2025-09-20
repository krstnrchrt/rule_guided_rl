import regex as re

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

RE_NUMERIC = re.compile(r"^\d+([.,]\d+)?$")
RE_ORDINAL = re.compile(r"^\d+\.$")

