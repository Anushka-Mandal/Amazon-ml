"""Text normalisation for business names and addresses.

All rules here are generic string canonicalisation (abbreviation folding, accent stripping,
state-name folding) plus the dictionaries learned from training data in learn_translit.py.
Nothing is looked up externally.  Rules are applied to every record whatever its country label,
so an unseen country (e.g. France in test) goes through the same code path.
"""
import json
import re
import unicodedata

from unidecode import unidecode

# ----------------------------------------------------------------------------- names
NAME_CANON = {
    # legal forms
    "private": "pvt", "pvt": "pvt", "prvt": "pvt", "pvtltd": "pvt ltd",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co", "comp": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc",
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "snc": "snc",
    "etablissements": "ets", "ets": "ets", "gmbh": "gmbh", "sci": "sci", "scp": "scp",
    "selarl": "selarl", "scop": "scop", "sca": "sca", "gie": "gie", "selas": "selas",
    # frequent words
    "and": "and", "et": "and", "intl": "international", "int'l": "international",
    "svcs": "services", "svc": "services", "servs": "services", "service": "services",
    "assoc": "associates", "assocs": "associates", "associate": "associates",
    "bros": "brothers", "mfg": "manufacturing", "dept": "department", "ctr": "center",
    "cntr": "center", "centre": "center", "grp": "group", "hosp": "hospital",
    "natl": "national", "univ": "university", "mgmt": "management", "mgt": "management",
    "tech": "tech", "technologies": "technologies", "sys": "systems", "ent": "enterprises",
    "entp": "enterprises", "ind": "industries", "inds": "industries", "hldgs": "holdings",
    "hldg": "holdings", "sr": "shree", "shri": "shree", "sri": "shree", "sree": "shree",
    "st": "saint",
}
LEGAL = {"pvt", "ltd", "inc", "corp", "co", "llc", "llp", "lp", "pllc", "pc", "plc", "sa",
         "sas", "sasu", "sarl", "eurl", "snc", "ets", "gmbh", "sci", "scp", "selarl", "scop",
         "sca", "gie", "selas"}
NAME_STOP = LEGAL | {"and", "of", "the", "mr", "mrs", "ms", "smt", "m/s", "ms", "india",
                     "france", "usa", "us", "dba", "le", "la", "les", "de", "du", "des", "l",
                     "d", "www", "com", "in", "net", "org", "fr"}

# ----------------------------------------------------------------------------- addresses
ADDR_CANON = {
    "street": "st", "str": "st", "st": "st", "saint": "st", "ste": "suite", "sainte": "st",
    "road": "rd", "rd": "rd", "raod": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave", "avn": "ave",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "lane": "ln", "ln": "ln",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd", "bld": "blvd",
    "court": "ct", "ct": "ct", "crt": "ct",
    "place": "pl", "pl": "pl", "plc": "pl",
    "circle": "cir", "cir": "cir", "crcl": "cir",
    "highway": "hwy", "hwy": "hwy", "hiway": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "terrace": "ter", "ter": "ter", "terr": "ter",
    "trail": "trl", "trl": "trl",
    "square": "sq", "sq": "sq",
    "route": "rte", "rte": "rte", "rt": "rte",
    "chemin": "chemin", "che": "chemin", "ch": "chemin",
    "impasse": "imp", "imp": "imp",
    "allee": "allee", "all": "allee",
    "faubourg": "fbg", "fbg": "fbg",
    "rue": "rue", "r": "rue",
    "quai": "quai", "qu": "quai",
    "north": "n", "n": "n", "south": "s", "s": "s", "east": "e", "e": "e", "west": "w", "w": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "floor": "fl", "flr": "fl", "fl": "fl",
    "apartment": "apt", "apt": "apt", "appt": "apt", "suite": "suite", "unit": "unit",
    "building": "bldg", "bldg": "bldg", "bldng": "bldg",
    "plot": "plot", "plt": "plot",
    "room": "room", "rm": "room",
    "opposite": "opp", "opp": "opp", "near": "near", "nr": "near", "behind": "behind",
    "beh": "behind", "sector": "sector", "sec": "sector", "sect": "sector",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "ground": "gf",
    "centre": "center", "ctr": "center", "cntr": "center",
    "complex": "complex", "cmplx": "complex", "cplx": "complex",
    "colony": "colony", "col": "colony", "industrial": "indl", "indl": "indl", "ind": "indl",
    "estate": "estate", "est": "estate", "market": "market", "mkt": "market",
    "chowk": "chowk", "marg": "marg",
    # historic / alternate city names (spelling variants of the same place)
    "bombay": "mumbai", "madras": "chennai", "bengaluru": "bangalore", "calcutta": "kolkata",
    "cochin": "kochi", "gurugram": "gurgaon", "poona": "pune", "baroda": "vadodara",
    "trivandrum": "thiruvananthapuram", "mysuru": "mysore", "vizag": "visakhapatnam",
    "odisha": "orissa", "pondicherry": "puducherry", "rangareddi": "rangareddy",
}
ADDR_DROP = {"no", "nos", "number", "h", "hno", "door", "null", "na", "n/a", "the", "of",
             "de", "du", "la", "le", "les", "des", "d", "l", "et", "and", "city", "po",
             "box", "pobox", "pmb", "p", "o", "at", "post", "dist", "district", "tal",
             "taluka", "via", "house", "flat", "ndeg"}
UNIT_WORDS = {"apt", "suite", "unit", "fl", "pmb", "box", "room", "#"}

# region/state names folded to a short code; matched on a whole comma-separated segment
STATES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
    # India (prefixed so codes cannot collide with US ones)
    "uttar pradesh": "up", "up": "up", "maharashtra": "mh", "mh": "mh",
    "telangana": "tg", "tg": "tg", "ts": "tg", "andhra pradesh": "ap", "ap": "ap",
    "karnataka": "ka", "ka": "ka", "kerala": "kl", "kl": "kl",
    "tamil nadu": "tn", "tamilnadu": "tn", "gujarat": "gj", "gj": "gj",
    "west bengal": "wb", "wb": "wb", "delhi": "dl", "dl": "dl",
    "orissa": "od", "odisha": "od", "od": "od",
    "haryana": "hr", "hr": "hr", "rajasthan": "rj", "rj": "rj",
    "punjab": "pb", "pb": "pb", "madhya pradesh": "mp", "mp": "mp",
    "bihar": "br", "br": "br", "jharkhand": "jh", "jh": "jh",
    "chhattisgarh": "cg", "cg": "cg", "uttarakhand": "uk", "uk": "uk",
    "goa": "ga", "assam": "as", "himachal pradesh": "hp", "hp": "hp",
    "jammu and kashmir": "jk", "jammu & kashmir": "jk", "jk": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    # France: regions and their departments
    "hauts-de-france": "fr-hdf", "hauts de france": "fr-hdf", "nord": "fr-hdf",
    "pas-de-calais": "fr-hdf", "somme": "fr-hdf", "oise": "fr-hdf", "aisne": "fr-hdf",
    "nouvelle-aquitaine": "fr-naq", "nouvelle aquitaine": "fr-naq", "gironde": "fr-naq",
    "charente": "fr-naq", "charente-maritime": "fr-naq", "correze": "fr-naq",
    "creuse": "fr-naq", "dordogne": "fr-naq", "landes": "fr-naq", "lot-et-garonne": "fr-naq",
    "pyrenees-atlantiques": "fr-naq", "deux-sevres": "fr-naq", "vienne": "fr-naq",
    "haute-vienne": "fr-naq",
    "pays de la loire": "fr-pdl", "pays-de-la-loire": "fr-pdl", "loire-atlantique": "fr-pdl",
    "maine-et-loire": "fr-pdl", "mayenne": "fr-pdl", "sarthe": "fr-pdl", "vendee": "fr-pdl",
    "ile-de-france": "fr-idf", "ile de france": "fr-idf", "paris": "fr-idf",
    "auvergne-rhone-alpes": "fr-ara", "occitanie": "fr-occ", "grand est": "fr-ges",
    "bretagne": "fr-bre", "normandie": "fr-nor", "bourgogne-franche-comte": "fr-bfc",
    "centre-val de loire": "fr-cvl", "provence-alpes-cote d'azur": "fr-pac", "corse": "fr-cor",
}

_ORD = re.compile(r"\b(\d+)(st|nd|rd|th|er|eme|e|ieme)\b")
_NUM = re.compile(r"\d+")
_WEB = re.compile(r"(www\.|https?://)|(\.(com|in|net|org|co\.in|fr|biz|info|us))\b")
_ACRONYM = re.compile(r"\b(?:[a-z]\.){2,}")


class Normalizer:
    def __init__(self, translit_path=None):
        self.tok_map, self.seg_map = {}, {}
        if translit_path:
            with open(translit_path) as f:
                d = json.load(f)
            self.tok_map, self.seg_map = d["tokens"], d["segments"]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def ascii_fold(s):
        if s.isascii():
            return s
        return unidecode(unicodedata.normalize("NFKC", s))

    @staticmethod
    def skeleton(tok):
        """Crude phonetic key: folds spelling / transliteration variants (laxmi~lakshmi)."""
        t = tok.replace("ph", "f").replace("x", "ks").replace("q", "k").replace("c", "k")
        t = t.replace("z", "s").replace("w", "v").replace("y", "i")
        t = t[:1] + t[1:].replace("h", "")
        if not t:
            return tok
        out = [t[0]]
        for ch in t[1:]:
            if ch in "aeiou":
                continue
            if ch != out[-1]:
                out.append(ch)
        sk = "".join(out)
        return sk if len(sk) >= 2 else tok

    # ------------------------------------------------------------------ names
    def name(self, raw):
        """Return (norm, core, concat, skel, flags)."""
        s = raw or ""
        flags = 0
        if re.search(r"[ऀ-෿]", s):
            flags |= 1  # native-script name
            s = " ".join(self.tok_map.get(t, t) for t in s.split())
        s = self.ascii_fold(s).lower()
        if _WEB.search(s) or "@" in s:
            flags |= 2  # web / handle style
            s = _WEB.sub(" ", s)
        s = s.replace("&", " and ").replace("+", " and ").replace("@", " ")
        s = _ACRONYM.sub(lambda m: m.group(0).replace(".", ""), s)
        s = s.replace("'", "").replace("’", "")
        s = re.sub(r"[^a-z0-9]+", " ", s)
        toks = []
        for t in s.split():
            c = NAME_CANON.get(t, t)
            toks.extend(c.split())
        # merge runs of single letters ("l l c" -> "llc")
        merged, buf = [], []
        for t in toks + [None]:
            if t is not None and len(t) == 1 and t.isalpha():
                buf.append(t)
                continue
            if buf:
                w = "".join(buf)
                merged.append(NAME_CANON.get(w, w))
                buf = []
            if t is not None:
                merged.append(t)
        toks = merged
        core = [t for t in toks if t not in NAME_STOP]
        if not core:
            core = [t for t in toks if t not in LEGAL] or toks
        concat = "".join(core)
        if flags & 2 and concat.endswith("com") and len(concat) > 6:
            concat = concat[:-3]
        skel = [self.skeleton(t) for t in core]
        return " ".join(toks), " ".join(core), concat, " ".join(skel), flags

    # ------------------------------------------------------------------ addresses
    def address(self, raw):
        """Return (norm, alpha, nums, unitnums, state, flags)."""
        s = raw or ""
        flags = 0
        if not s.strip() or s.strip().lower() in ("null", "n/a", "na", "none"):
            return "", "", "", "", "", 4
        segs = [x.strip() for x in s.split(",")]
        out_segs, state = [], ""
        for seg in segs:
            if not seg:
                continue
            if seg in self.seg_map:
                seg = self.seg_map[seg]
                flags |= 1
            seg = self.ascii_fold(seg).lower().strip()
            key = re.sub(r"\s+", " ", seg.replace(".", "")).strip()
            if key in STATES:
                state = STATES[key]
                continue
            if key in ("null", "n/a", "na", ""):
                continue
            out_segs.append(seg)
        text = " , ".join(out_segs)
        text = text.replace("n°", " no ").replace("#", " # ")
        text = _ORD.sub(r"\1", text)
        text = re.sub(r"[^a-z0-9#,]+", " ", text)
        toks, alpha, nums, unitnums = [], [], [], []
        prev = ""
        for t in text.split():
            if t == ",":
                prev = ""
                continue
            # split alnum tokens such as 5cnew / g57 / 1002a
            parts = re.findall(r"\d+|[a-z]+|#", t)
            for p in parts:
                if p.isdigit():
                    q = p.lstrip("0") or "0"
                    toks.append(q)
                    if prev in UNIT_WORDS:
                        unitnums.append(q)
                    else:
                        nums.append(q)
                    prev = "num"
                elif p == "#":
                    prev = "#"
                else:
                    c = ADDR_CANON.get(p, p)
                    prev = c
                    if c in ADDR_DROP or c in ("#",):
                        if c in ("box", "pmb", "po", "pobox"):
                            prev = "box"
                        continue
                    toks.append(c)
                    alpha.append(c)
        return (" ".join(toks), " ".join(dict.fromkeys(alpha)), " ".join(dict.fromkeys(nums)),
                " ".join(dict.fromkeys(unitnums)), state, flags)
