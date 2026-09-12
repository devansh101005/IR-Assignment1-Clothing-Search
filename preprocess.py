# Part A: tokenisation, stop word removal and stemming.
# The indexer and the query side both call analyse(), so the document
# pipeline and the query pipeline stay identical.

import re


# Standard NLTK English stop list, typed in here so nltk is not needed.
NLTK_ENGLISH_STOPWORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now",
]

# Entries we take back out of the standard list. "s" and "t" are clothing
# sizes and the T of T-shirt; the prepositions are part of garment names
# (down jacket, zip-up, oversized, off white). Reasons are in REPORT.md.
KEEP_THESE = ["s", "t", "down", "up", "over", "out", "off", "no", "not"]

STOPWORDS = set(NLTK_ENGLISH_STOPWORDS)
for w in KEEP_THESE:
    if w in STOPWORDS:
        STOPWORDS.remove(w)


def tokenize(text):
    """Lower case, strip punctuation, return the token list.
    Stop words are NOT removed here - see analyse()."""

    text = text.lower()
    text = text.replace(u"’", "'")

    # "men's" -> "men". Doing this before the punctuation rule avoids an
    # extra "s" token, which would collide with the size S.
    text = re.sub(r"'s\b", "", text)

    # t-shirt -> tshirt, v-neck -> vneck. Any other hyphen just splits,
    # so skin-friendly -> skin friendly.
    text = re.sub(r"\b([a-z])-([a-z])", r"\1\2", text)

    # digits are kept, "100% cotton" is worth searching for
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return text.split()


def is_stopword(token):
    return token in STOPWORDS


# --- stemmer ---------------------------------------------------------
# Our own suffix stripper. Full Porter was too aggressive on a 127 term
# dictionary (durable -> dur, essentials -> essenti), so the conditions
# below use "stem must keep a vowel and stay long enough" instead of
# Porter's measure m, and we stop after a few steps.

VOWELS = "aeiou"

IRREGULAR = {
    "leggings": "legging",
    "legging": "legging",
    "jeans": "jean",
    "dresses": "dress",
    "dress": "dress",
    "trousers": "trouser",
    "clothes": "cloth",
    "clothing": "cloth",
    "shoes": "shoe",
    "men": "men",
    "women": "women",
}


def _has_vowel(word):
    for ch in word:
        if ch in VOWELS:
            return True
    return False


def _ends_with_double_consonant(word):
    if len(word) < 2:
        return False
    if word[-1] != word[-2]:
        return False
    return word[-1] not in VOWELS


def _is_cvc(word):
    """Ends consonant-vowel-consonant, last letter not w/x/y.
    Used to put back a silent e: strip -> stripe."""
    if len(word) < 3:
        return False
    a = word[-3]
    b = word[-2]
    c = word[-1]
    if a in VOWELS or b not in VOWELS or c in VOWELS:
        return False
    if c in "wxy":
        return False
    return True


def _fix_after_removal(stem_so_far):
    """Clean up after chopping -ed or -ing."""
    if (stem_so_far.endswith("at") or stem_so_far.endswith("bl")
            or stem_so_far.endswith("iz")):
        return stem_so_far + "e"
    if _ends_with_double_consonant(stem_so_far) and stem_so_far[-1] not in "lsz":
        return stem_so_far[:-1]
    if len(stem_so_far) <= 5 and _is_cvc(stem_so_far):
        return stem_so_far + "e"
    return stem_so_far


def stem(word):
    """Reduce one token to its stem."""

    if word in IRREGULAR:
        return IRREGULAR[word]

    # sizes (s, m, l, xl) are left alone
    if len(word) <= 3:
        return word

    # plurals
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith("ss") and not word.endswith("us"):
        word = word[:-1]

    # -ed / -ing
    if word.endswith("eed"):
        if len(word) > 4:
            word = word[:-1]
    elif word.endswith("ed") and _has_vowel(word[:-2]) and len(word) > 4:
        word = _fix_after_removal(word[:-2])
    elif word.endswith("ing") and _has_vowel(word[:-3]) and len(word) > 5:
        word = _fix_after_removal(word[:-3])

    # -ly, only after a consonant: friendly -> friend, but daily stays
    if word.endswith("ly") and len(word) > 4 and word[-3] not in VOWELS:
        word = word[:-2]

    # -able / -ible. The length guard keeps durable intact.
    if word.endswith("able") and len(word) - 4 >= 4:
        word = word[:-4]
    elif word.endswith("ible") and len(word) - 4 >= 4:
        word = word[:-4]

    if word.endswith("ness") and len(word) - 4 >= 4:
        word = word[:-4]
    elif word.endswith("ment") and len(word) - 4 >= 4:
        word = word[:-4]
    elif word.endswith("ion") and len(word) > 5 and word[-4] in "st":
        word = word[:-3]

    # no final-e stripping, so size/sizes both give "size"
    return word


def analyse(text):
    """Full Part A pipeline. Returns a list of (position, term).

    Positions are indices in the full token stream, i.e. counted before
    stop words are dropped. Numbering after removal would make
    "cotton of the shirt" look like the phrase "cotton shirt".
    """
    out = []
    position = 0
    for token in tokenize(text):
        if not is_stopword(token):
            out.append((position, stem(token)))
        position = position + 1
    return out


def analyse_query(text):
    """Alias used on the query side. Must stay identical to analyse()."""
    return analyse(text)


if __name__ == "__main__":
    demo = "Men's Cotton Crew Neck T-Shirt - Black. It features breathable fabric, machine washable."
    print("raw      :", demo)
    print("tokens   :", tokenize(demo))
    print("analysed :", analyse(demo))
    print()
    print("stop words in use :", len(STOPWORDS))
    for w in ["shirts", "washable", "breathable", "striped", "construction",
              "friendly", "sizes", "leggings", "durable", "stitching",
              "comfortable", "oversized", "daily", "essentials"]:
        print("   %-14s -> %s" % (w, stem(w)))
