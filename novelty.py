# Our extension, not required by parts A-E. Two things, both built on the
# positional index:
#
#   1. PB-VSM, a proximity boosted ranking. lnc.ltc throws positions away,
#      so a document with "cotton shirt" as a phrase and one with cotton in
#      the first line and shirt in the last can score the same. Worse on
#      this corpus, 27 terms have df = 100 and idf = 0, so the cosine score
#      cannot see them at all - but they still have positions.
#
#   2. Edit distance spelling correction for query terms that are not in
#      the dictionary (cotten, tshrit, hoodei).

import math

import preprocess


def smallest_window(position_lists):
    """Length of the shortest span of the document containing at least one
    occurrence of every term. Term A at [2, 40] and B at [3, 55] gives 2.

    Merge the positions into one sorted stream tagged with the term they
    came from, slide a window, shrink from the left whenever all terms are
    covered. Linear in the number of postings, which matters because the
    boilerplate terms have very long position lists.
    """

    n_terms = len(position_lists)
    if n_terms == 0:
        return 0
    if n_terms == 1:
        return 1

    events = []
    i = 0
    while i < n_terms:
        for p in position_lists[i]:
            events.append((p, i))
        i = i + 1
    events.sort()

    if len(events) == 0:
        return 0

    counts = {}
    inside = 0
    best = None
    left = 0

    right = 0
    while right < len(events):
        pos_r, term_r = events[right]
        if counts.get(term_r, 0) == 0:
            inside = inside + 1
        counts[term_r] = counts.get(term_r, 0) + 1

        while inside == n_terms:
            pos_l, term_l = events[left]
            width = pos_r - pos_l + 1
            if best is None or width < best:
                best = width
            counts[term_l] = counts[term_l] - 1
            if counts[term_l] == 0:
                inside = inside - 1
            left = left + 1

        right = right + 1

    if best is None:
        return 0
    return best


def proximity_factor(index, docid, query_terms):
    """How tightly the query terms sit together in one document.

        coverage  = matched / asked      how much of the query is there
        tightness = matched / span       1.0 when the terms are adjacent
        factor    = coverage * tightness

    A term counts as matched even when its idf is 0 - those are invisible
    to the cosine score but their positions are exactly what we want.
    """

    present = []
    for term in query_terms:
        plist = index.positional.get(term, {}).get(docid)
        if plist is not None and len(plist) > 0:
            present.append(plist)

    matched = len(present)
    asked = len(query_terms)

    result = {
        "matched": matched,
        "asked": asked,
        "span": 0,
        "coverage": 0.0,
        "tightness": 0.0,
        "factor": 0.0,
    }

    # one term has no proximity, leave the score alone
    if matched < 2 or asked < 2:
        return result

    span = smallest_window(present)
    result["span"] = span
    result["coverage"] = float(matched) / float(asked)
    if span > 0:
        result["tightness"] = float(matched) / float(span)
    result["factor"] = result["coverage"] * result["tightness"]
    return result


def rank_pbvsm(index, query_string, top_k=10, alpha=0.5):
    """final = cosine * (1 + alpha * coverage * tightness)

    alpha = 0 collapses back to the plain Part B ranking, so the same code
    produces the before and the after. 0.5 gives an adjacent phrase a 50%
    bonus, enough to beat a document that only won on tf without wrecking
    the cosine ordering.

    This re-ranks, it does not retrieve - the candidate set is still what
    the VSM matched. Since only 10 results are shown, reordering still
    changes which documents the user sees.
    """

    import vsm

    scores, info = vsm.cosine_scores(index, query_string)

    # every query term in the dictionary, including the idf = 0 ones
    query_terms = []
    for term in info["terms"]:
        if term in index.positional and term not in query_terms:
            query_terms.append(term)

    results = []
    for docid in scores:
        base = scores[docid]
        prox = proximity_factor(index, docid, query_terms)
        boost = 1.0 + alpha * prox["factor"]
        results.append({
            "docid": docid,
            "base": base,
            "boost": boost,
            "score": base * boost,
            "span": prox["span"],
            "matched": prox["matched"],
            "asked": prox["asked"],
            "factor": prox["factor"],
        })

    results.sort(key=lambda r: (-r["score"], r["docid"]))
    return results[:top_k], info


def compare_rankings(index, query_string, top_k=10, alpha=0.5):
    """Plain VSM and PB-VSM on the same query, side by side."""

    import vsm

    plain, info = vsm.rank(index, query_string, top_k)
    boosted, _ = rank_pbvsm(index, query_string, top_k, alpha)

    plain_ids = []
    for docid, score in plain:
        plain_ids.append(docid)

    boosted_ids = []
    for r in boosted:
        boosted_ids.append(r["docid"])

    moved_in = []
    for d in boosted_ids:
        if d not in plain_ids:
            moved_in.append(d)
    moved_out = []
    for d in plain_ids:
        if d not in boosted_ids:
            moved_out.append(d)

    return {
        "plain": plain,
        "boosted": boosted,
        "entered": moved_in,
        "left": moved_out,
        "order_changed": plain_ids != boosted_ids,
        "info": info,
    }


def edit_distance(a, b, cutoff=3):
    """Levenshtein, two row version. Bails out once the whole row is past
    the cutoff, which also stops silly suggestions."""

    if abs(len(a) - len(b)) > cutoff:
        return cutoff + 1

    previous = list(range(len(b) + 1))

    i = 1
    while i <= len(a):
        current = [i]
        j = 1
        while j <= len(b):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1
            current.append(min(previous[j] + 1,          # delete
                               current[j - 1] + 1,       # insert
                               previous[j - 1] + cost))  # substitute
            j = j + 1
        if min(current) > cutoff:
            return cutoff + 1
        previous = current
        i = i + 1

    return previous[len(b)]


def suggest_term(index, term):
    """Closest dictionary term, or (None, None) if nothing is near enough.

    The limit depends on length - 2 edits on a 4 letter word gives a
    different word. Ties go to the higher df, the usual "commoner word is
    more likely" rule.
    """

    if len(term) <= 3:
        limit = 1
    elif len(term) <= 6:
        limit = 2
    else:
        limit = 2

    best = None
    best_distance = limit + 1
    best_df = -1

    for candidate in index.df:
        d = edit_distance(term, candidate, limit)
        if d > limit:
            continue
        if d < best_distance or (d == best_distance and index.df[candidate] > best_df):
            best = candidate
            best_distance = d
            best_df = index.df[candidate]

    if best is None:
        return None, None
    return best, best_distance


def correct_query(index, query_string):
    """Replace unknown content words with the closest known term.
    Returns (new_query, [(typed, suggested)], [words we could not fix])."""

    tokens = preprocess.tokenize(query_string)

    rebuilt = []
    changes = []
    dropped = []

    for token in tokens:
        if preprocess.is_stopword(token):
            rebuilt.append(token)
            continue

        stemmed = preprocess.stem(token)
        if stemmed in index.df:
            rebuilt.append(token)
            continue

        suggestion, distance = suggest_term(index, stemmed)
        if suggestion is None:
            dropped.append(token)
            rebuilt.append(token)      # kept, the engine will ignore it
        else:
            changes.append((token, suggestion))
            rebuilt.append(suggestion)

    return " ".join(rebuilt), changes, dropped


if __name__ == "__main__":
    import indexer
    idx = indexer.build_index("data/corpus_100.txt")

    print("--- smallest window sanity check ---")
    print("adjacent  [4] [5]      ->", smallest_window([[4], [5]]))
    print("far apart [2] [40]     ->", smallest_window([[2], [40]]))
    print("two options            ->", smallest_window([[2, 40], [3, 55]]))

    print()
    print("--- spelling correction ---")
    for bad in ["cotten", "shrt", "hoodei", "jaket", "umbrella", "laptop"]:
        s, d = suggest_term(idx, preprocess.stem(bad))
        print("   %-10s -> %s (distance %s)" % (bad, s, d))

    print()
    print("--- plain VSM vs PB-VSM for 'winter wear' ---")
    cmp = compare_rankings(idx, "winter wear", 5)
    for i in range(5):
        left = cmp["plain"][i]
        right = cmp["boosted"][i]
        print("  %d. %s %.4f      |  %s %.4f (span %d)"
              % (i + 1, left[0], left[1], right["docid"], right["score"],
                 right["span"]))
