# Part C: phrase and proximity search over the positional index.
# The VSM only knows which documents hold a term and how often. Here we
# also know where, so "cotton" in line 1 and "shirt" in the last line is
# no longer treated as the phrase "cotton shirt".

import preprocess


def _positions(index, term, docid):
    if term not in index.positional:
        return []
    if docid not in index.positional[term]:
        return []
    return index.positional[term][docid]


def _docs_containing_all(index, terms):
    """Intersect the postings lists, rarest term first to keep the
    intermediate set small."""

    for t in terms:
        if t not in index.positional:
            return []

    ordered = sorted(terms, key=lambda t: index.df[t])

    result = set(index.positional[ordered[0]].keys())
    for t in ordered[1:]:
        result = result & set(index.positional[t].keys())
        if len(result) == 0:
            break

    return sorted(result)


def phrase_search(index, phrase):
    """Documents where the query words appear consecutively.
    Returns ([(docID, [start positions])], stems).

    Positional intersection: take each position of the first term and walk
    forward requiring term i+1 at exactly the next position. Checking only
    that the words co-occur would be wrong.
    """

    terms = []
    for position, term in preprocess.analyse(phrase):
        terms.append(term)

    if len(terms) == 0:
        return [], terms

    if len(terms) == 1:
        out = []
        for docid in sorted(index.positional.get(terms[0], {})):
            out.append((docid, list(index.positional[terms[0]][docid])))
        return out, terms

    results = []
    for docid in _docs_containing_all(index, terms):

        starts = []
        for start in _positions(index, terms[0], docid):
            ok = True
            expected = start + 1
            i = 1
            while i < len(terms):
                if expected not in _positions(index, terms[i], docid):
                    ok = False
                    break
                expected = expected + 1
                i = i + 1
            if ok:
                starts.append(start)

        if len(starts) > 0:
            results.append((docid, starts))

    return results, terms


def proximity_search(index, term1, term2, k, ordered=True):
    """term1 WITHIN/k term2.

        ordered   ->  0 < p2 - p1 <= k
        unordered ->  0 < |p2 - p1| <= k

    Returns ([(docID, [(p1, p2), ...])], stems). k counts positions on the
    full token stream, so WITHIN/1 is the same as a two word phrase.
    """

    t1 = _single_term(term1)
    t2 = _single_term(term2)

    if t1 is None or t2 is None:
        return [], (t1, t2)
    if t1 not in index.positional or t2 not in index.positional:
        return [], (t1, t2)

    results = []
    common = set(index.positional[t1].keys()) & set(index.positional[t2].keys())

    for docid in sorted(common):
        list1 = _positions(index, t1, docid)
        list2 = _positions(index, t2, docid)

        hits = []
        # the lists are short here so a double loop is fine; on a big
        # collection we would walk the two sorted lists together instead
        for p1 in list1:
            for p2 in list2:
                gap = p2 - p1
                if ordered:
                    if gap > 0 and gap <= k:
                        hits.append((p1, p2))
                else:
                    if gap != 0 and abs(gap) <= k:
                        hits.append((p1, p2))

        if len(hits) > 0:
            results.append((docid, hits))

    return results, (t1, t2)


def _single_term(word):
    """Stem one query word. None if it is a stop word and disappears."""
    pairs = preprocess.analyse(word)
    if len(pairs) == 0:
        return None
    return pairs[0][1]


def snippet_around(index, docid, position, window=6):
    """Rebuild the words around a stored position so the match can be seen.
    Re-tokenising without stop word removal reproduces exactly the stream
    the positions were numbered on."""
    d = index.docs[docid]
    full_text = d["category"] + " " + d["title"] + " " + d["text"]
    tokens = preprocess.tokenize(full_text)

    lo = position - window
    if lo < 0:
        lo = 0
    hi = position + window + 1
    if hi > len(tokens):
        hi = len(tokens)

    parts = []
    i = lo
    while i < hi:
        if i == position:
            parts.append("[" + tokens[i] + "]")
        else:
            parts.append(tokens[i])
        i = i + 1

    text = " ".join(parts)
    if lo > 0:
        text = "... " + text
    if hi < len(tokens):
        text = text + " ..."
    return text


def phrase_snippet(index, docid, start, length, window=5):
    """Same thing but brackets a whole phrase."""
    d = index.docs[docid]
    full_text = d["category"] + " " + d["title"] + " " + d["text"]
    tokens = preprocess.tokenize(full_text)

    lo = start - window
    if lo < 0:
        lo = 0
    hi = start + length + window
    if hi > len(tokens):
        hi = len(tokens)

    parts = []
    i = lo
    while i < hi:
        if i == start:
            parts.append("[")
        parts.append(tokens[i])
        if i == start + length - 1:
            parts.append("]")
        i = i + 1

    text = " ".join(parts).replace("[ ", "[").replace(" ]", "]")
    if lo > 0:
        text = "... " + text
    if hi < len(tokens):
        text = text + " ..."
    return text


if __name__ == "__main__":
    import indexer
    idx = indexer.build_index("data/corpus_100.txt")

    hits, terms = phrase_search(idx, "cotton shirt")
    print("phrase 'cotton shirt' -> stems", terms, "->", len(hits), "documents")
    for docid, starts in hits[:3]:
        print("  ", docid, "at positions", starts)
        print("      ", phrase_snippet(idx, docid, starts[0], len(terms)))

    print()
    hits, pair = proximity_search(idx, "winter", "wear", 3)
    print("winter WITHIN/3 wear -> stems", pair, "->", len(hits), "documents")
    for docid, pairs in hits[:3]:
        print("  ", docid, pairs)
