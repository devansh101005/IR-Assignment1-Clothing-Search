# Building a Clothing Search Engine

**CSD358 - Information Retrieval, Assignment 1**

| | |
|---|---|
| **Devansh Pandey** | 2310110461 |
| **Abhinav Bachchas** | 2310110383 |

Corpus: 100 clothing product descriptions
Language / libraries: Python 3, standard library only (no NLTK, no scikit-learn, no numpy)

---

## Contents

1. [How to run it](#1-how-to-run-it)
2. [Part A - Corpus and pre-processing](#2-part-a---corpus-and-pre-processing)
3. [Part B - Vector Space Model (lnc.ltc)](#3-part-b---vector-space-model-lncltc)
4. [Part C - Positional index](#4-part-c---positional-index)
5. [Part D - The application](#5-part-d---the-application)
6. [Part E - Testing](#6-part-e---testing)
7. [Novelty - Proximity-Boosted VSM and query correction](#7-novelty---what-we-added-beyond-the-handout)
8. [What we learned, and what is still wrong](#8-what-we-learned-and-what-is-still-wrong)
9. [File list](#9-file-list)

---

## 1. How to run it

```
python build_index.py      # builds both indexes, writes the dumps in output/
python run_tests.py        # runs the whole Part E test plan
python search_cli.py       # the terminal interface
python search_gui.py       # the window interface
```

Nothing needs to be installed. We used only the Python standard library on
purpose - we wanted the marker to be able to run it without a `pip install`,
and we also wanted to actually implement the stemmer and the cosine ranking
instead of calling a library that does it for us.

---

## 2. Part A - Corpus and pre-processing

### 2.1 Reading the corpus

`corpus_100.txt` is an SGML-style file. Each record gives a `DOCID`, a
`CATEGORY`, a `TITLE` and a `TEXT`. We pull the four fields out with regular
expressions (`indexer.py`). We did not use an XML parser because the file is
not well-formed XML - the `%` in "100% cotton" and the apostrophes are not
escaped.

**We index `CATEGORY + TITLE + TEXT` together.** The title repeats the
important words (garment type, cut, colour), so a word in the title
automatically gets a higher term frequency and therefore a higher document
weight. This gives us crude field weighting for free. We first tried indexing
only `TEXT` and the ranking was visibly worse - a search for *black t-shirt*
did not put the black T-shirts on top.

### 2.2 The pipeline

Everything lives in `preprocess.py`, and **the same function is used for
documents and for queries**. This is deliberate: if the two pipelines drift
apart even slightly, the query terms stop matching the indexed terms and
phrase search silently breaks.

```
raw text
   -> lower case
   -> "men's" / "women's"  ->  "men" / "women"
   -> t-shirt -> tshirt,  v-neck -> vneck   (single letter + hyphen + word)
   -> skin-friendly -> skin friendly        (every other hyphen splits)
   -> drop all remaining punctuation, keep digits
   -> number the tokens          <-- positions are assigned HERE
   -> remove stop words
   -> stem
```

Two of these steps are decisions we had to argue about, so they are worth
explaining.

**Why `'s` is stripped before the punctuation rule.** If we let the generic
"punctuation becomes a space" rule handle `men's`, we get two tokens, `men`
and `s`. That stray `s` then collides with the clothing size **S** ("Available
in size S"), which every one of the 100 products mentions. In our first
version the term `s` had a document frequency of 90 instead of 20 and size
queries were meaningless. Handling `'s` first fixes it.

**Why `t-shirt` becomes one token.** If we split on the hyphen we get `t` and
`shirt`. That has two bad effects: `t` on its own is a useless term, and a
search for **shirt** would drag in all ten T-shirts, which are a different
product. Gluing a single letter to the following word (`t-shirt` → `tshirt`,
`v-neck` → `vneck`) keeps the product name intact and keeps *shirt* meaning
*shirt*. Every other hyphen is treated as a word separator, so
`skin-friendly` → `skin`, `friendly`, which is what we want.

### 2.3 Stop-word policy (and its justification)

We use **the standard 179-word NLTK English stop list**, typed into
`preprocess.py` by hand so that the program does not need NLTK installed.
We then deliberately **take nine words back out of it**:

| Word removed from the stop list | Why |
|---|---|
| `s`, `t` | The standard list contains these only as leftovers of splitting `it's` and `don't`. In a clothing corpus `s` is a **size** and `t` is half of **T-shirt**. Removing them made size queries impossible. |
| `down` | A *down jacket* is a garment, not a preposition. |
| `up` | *zip-up hoodie*. |
| `over` | *overcoat*, *oversized*. |
| `out`, `off` | *cover-up*, *off shoulder*, *off white*. |
| `no`, `not` | A shopper may type *no iron shirt*; we do not want the negation silently deleted. |

That leaves **118 stop words**, applied identically to documents and queries.

The effect on this corpus: **7,303 raw tokens → 5,203 indexed terms**, i.e.
**28.8 % of the token stream is removed**, leaving a dictionary of **127
distinct terms**.

Removing them is clearly the right call here, because every one of the 100
products ends with the *same* boilerplate sentence:

> "The garment is suitable for comfortable regular use and can be paired with
> common wardrobe essentials."

so words like *the*, *for*, *and*, *is*, *with* contribute nothing at all.
Section 3.3 shows that this boilerplate causes a much more interesting problem
that stop-word removal alone cannot fix.

### 2.4 Stemming

We wrote **our own conservative suffix stripper** rather than importing a
ready-made Porter stemmer.

We started with full Porter and did not like what it did to this vocabulary:
`durable` became `dur` and `essentials` became `essenti`. Those are not wrong
as far as Porter is concerned, but on a 127-word dictionary an over-stemmed
term can merge two unrelated products, and that costs more precision than the
recall it buys. So our rules use a simpler test than Porter's *measure m* -
"the remaining stem must still contain a vowel and must be long enough" - and
we stop earlier. **We would rather under-stem than over-stem.**

The steps are: plurals → `-ed`/`-ing` (with the consonant-doubling and silent-
`e` repairs) → `-ly` → `-able`/`-ible` → `-ness`/`-ment`/`-ion`. We do **not**
strip the final silent `e` the way Porter step 5 does, so `size` and `sizes`
both land on `size` and the printed dictionary stays readable.

There is also a **small** exception table (`leggings` → `legging`,
`jeans` → `jean`, …) for clothing nouns where the general rules give an ugly
answer. We kept it deliberately short - past a certain size it stops being a
stemmer and becomes a lookup table.

| word | our stem | | word | our stem |
|---|---|---|---|---|
| shirts | shirt | | comfortable | comfort |
| washable | wash | | stitching | stitch |
| breathable | breath | | construction | construct |
| striped | stripe | | oversized | oversize |
| friendly | friend | | durable | durable *(left alone on purpose)* |
| sizes | size | | daily | daily *(dropping `-ly` gave `dai`)* |

### 2.5 The dictionary

`output/inverted_index.txt` - term → df → `[(docID, tf), …]`
`output/dictionary.txt` - term, df, idf, collection frequency, sorted by df
`output/corpus_stats.txt` - the summary numbers quoted above

| | |
|---|---|
| Documents, N | 100 |
| Distinct terms | 127 |
| Postings | 3,934 |
| Indexed terms (after stop words) | 5,203 |
| Average terms per document | 52.0 |

---

## 3. Part B - Vector Space Model (lnc.ltc)

### 3.1 The weighting

| | document side (`lnc`) | query side (`ltc`) |
|---|---|---|
| **l** - log tf | `1 + log10(tf)` | `1 + log10(tf)` |
| **n / t** - idf | none | `× log10(N/df)` |
| **c** - normalise | `÷ ‖d‖` | `÷ ‖q‖` |

Because the document weights are divided by the document length that we stored
at index time, the cosine similarity is simply the dot product of the two
normalised vectors. Document lengths are computed **once** in
`indexer._compute_doc_lengths()` and reused for every query - that is the
entire point of storing them, and they are dumped to `output/doc_lengths.txt`.

Scoring walks only the postings lists of the query terms (term-at-a-time with
an accumulator dictionary). Documents containing no query term are never
touched, because their score is zero by construction.

Results are sorted by **decreasing score, ties broken by increasing docID**,
and cut at 10.

### 3.2 A worked example

Query: **`black cotton t-shirt`** → stems `black`, `cotton`, `tshirt`

| term | df | idf = log10(100/df) | raw w<sub>q</sub> | normalised w<sub>q</sub> |
|---|---|---|---|---|
| black | 19 | 0.7212 | 0.7212 | 0.5745 |
| cotton | 58 | 0.2366 | 0.2366 | 0.1884 |
| tshirt | 10 | 1.0000 | 1.0000 | 0.7965 |

‖q‖ = √(0.7212² + 0.2366² + 1.0000²) = 1.2555

Scoring **D001** (*Men's Cotton Crew Neck T-Shirt - Black*):

| term | tf in D001 | w<sub>d</sub> = 1+log10(tf) | w<sub>q</sub> × w<sub>d</sub> |
|---|---|---|---|
| black | 3 | 1.4771 | 0.8486 |
| cotton | 3 | 1.4771 | 0.2783 |
| tshirt | 4 | 1.6021 | 1.2761 |
| | | **dot product** | **2.4030** |

‖D001‖ = 6.9566  →  **cosine = 2.4030 / 6.9566 = 0.3454**, which is rank 1.

The CLI reproduces this table with `explain D001`, and the GUI shows it when
you double-click a result row - we added that because we wanted to be able to
prove the numbers rather than assert them.

### 3.3 A problem this corpus creates: **27 terms have idf = 0**

Every document ends with the same boilerplate sentence, so **27 of the 127
dictionary terms occur in all 100 documents**:

```
avail casual colour comfort common depend design essential everyday feature
festive garment indian made office pair regular size standard styling suit
travel use wardrobe wear well work
```

For each of them `idf = log10(100/100) = 0`, so **their query weight is exactly
zero and they cannot influence the cosine ranking at all**. Note that this list
includes words a shopper would really type: *wear*, *festive*, *regular*,
*comfort*, *casual*, *size*.

In the extreme case - a query like `comfortable regular wear` where *every*
term has idf 0 - the whole query vector is zero and plain lnc.ltc cannot rank
anything. We handle this explicitly rather than dividing by zero: `vsm.py`
falls back to query weights without the idf factor and sets a flag so the
interface can tell the user what happened.

This is exactly the gap that the positional index fills, and it is what
motivated our novelty in Section 7.

---

## 4. Part C - Positional index

### 4.1 Structure

```
term  ->  df  ->  [(docID, tf, [p1, p2, ...]), ...]
```

written to `output/positional_index.txt`. Example:

```
cotton  ->  df=58
        (D001, tf=3, [2, 8, 16])
        (D002, tf=2, [3, 8])
        ...
```

We build the plain inverted index **and** the positional index side by side.
Strictly the first is redundant (tf is just `len(positions)`), but the
assignment asks us to compare retrieval before and after adding positions, and
keeping them separate makes both dumps readable.

### 4.2 The design decision that matters: **where positions are counted**

Positions are offsets in the **full token stream, numbered before stop words
are removed**.

This is not the obvious choice - it would be simpler to number the terms after
filtering - but numbering after filtering is **wrong**. Consider a document
containing *"cotton of the shirt"*. With positions assigned after stop-word
removal, `cotton` and `shirt` would be at consecutive positions and the phrase
query `"cotton shirt"` would report a match that does not exist. Counting on
the full stream costs nothing and keeps every phrase and proximity answer
honest. It also means `k` in a proximity query counts real tokens, which is
what a user expects.

### 4.3 Phrase search

Textbook positional intersection (`positional.phrase_search`):

1. Intersect the postings lists of all phrase terms, **starting from the
   rarest term** so the intermediate set stays small.
2. For each surviving document, take each position of the first term and walk
   forward, requiring term *i+1* at exactly position *p+i*.
3. Return the start positions of every occurrence.

We do **not** merely check that the words co-occur in the document - that is
the mistake the handout warns about, and Section 6.4 shows a query where the
difference is 20 documents versus 0.

### 4.4 Proximity search

`term1 WITHIN/k term2` matches a document when there are positions with

```
0 < p2 - p1 <= k          (ordered: term1 must come first)
```

`positional.proximity_search` also supports the unordered variant
(`0 < |p2-p1| <= k`), which we added while testing because we wanted to see
how much word order actually matters. It matters: *high waist* and *waist
high* are not the same query.

Note that `k = 1` is by definition the same thing as a two-word exact phrase.
`run_tests.py` checks this at run time as a consistency test between the two
independent code paths - they agree (5 documents each for `cotton shirt`).

---

## 5. Part D - The application

Two interfaces, both driven by the same `engine.ClothingSearchEngine`, so they
can never disagree about what a query means.

### `search_cli.py` - terminal

Mode `1` free text · mode `2` phrase/proximity · mode `3` side-by-side
comparison. Extra commands: `stats`, `term <word>` (dumps the postings *and*
the stored positions for one term), `explain <docID>` (the lnc.ltc arithmetic),
`boost on|off`, `alpha <n>`.

### `search_gui.py` - window (tkinter, ships with Python)

Four radio buttons that map one-to-one onto the assignment:

| Mode | Part |
|---|---|
| Free text (ranked) | B |
| Exact phrase | C |
| Proximity WITHIN/k | C |
| Compare with / without positions | our extension |

The results table shows **docID, category, product title and cosine score** as
required, plus the boosted score and the span when the boost is on. The panel
underneath always shows the **evidence**: the stored positions, a snippet of
the document with the matched phrase bracketed, the df/idf of every query term,
and the boost arithmetic. Double-clicking a row prints the full lnc.ltc
calculation and the positions of each query term inside that document.

Screenshots are in `screenshots/`:

| File | What it shows |
|---|---|
| `01_freetext_black_cotton_tshirt.png` | ranked retrieval, top 10 with scores |
| `02_freetext_plain_vsm_partB.png` | the same engine with the boost switched off (pure Part B) |
| `03_phrase_cotton_shirt.png` | **phrase match with the term positions displayed** |
| `04_phrase_zip_closure_nohits.png` | a phrase whose words both exist but never adjacently |
| `05_proximity_cotton_within3_shirt.png` | `cotton WITHIN/3 shirt` with the (p1,p2) pairs |
| `06_proximity_high_within1_waist.png` | proximity at k = 1 |
| `07_compare_regular_fit.png` | positional information changing the result set |
| `08_compare_short_sleeves_cotton.png` | positional information changing the order |
| `09_spelling_correction.png` | `blak hoodei` repaired to `black hoodie` |
| `10_zero_idf_query.png` | a query where every term has idf = 0 |
| `11_score_breakdown_doubleclick.png` | the lnc.ltc arithmetic for one document |

---

## 6. Part E - Testing

`python run_tests.py` runs everything and writes `output/test_results.txt`.

**No expected document IDs are hard-coded anywhere.** The query lists are
fixed; the answers are whatever the engine returns. Where the script makes a
claim about a result ("the phrase query returns fewer documents than the VSM",
"k = 1 equals the phrase query") it *checks that claim at run time* from the
numbers it just computed.

| Requirement | Asked for | We ran |
|---|---|---|
| Free-text queries | ≥ 10 | **12** |
| Exact phrase queries | ≥ 5 | **8** |
| Proximity queries | ≥ 3, different k | **6**, with k ∈ {1, 3, 4, 5} |
| Query with a term not in the corpus | ≥ 1 | **4** |
| Cases where positions change the result | ≥ 2 | **3, explained below** |

### 6.1 Case 1 - `regular fit` : the result **set** changes (5 of 10 documents)

`regular` has df = 100, so **idf = 0 and the cosine score cannot see it at
all**. Plain lnc.ltc therefore ranks this query purely on `fit`, and the
boilerplate phrase *"comfortable fit"* hands that term to products which are
not regular fit at all:

| rank | plain VSM (Part B) | | with positions |
|---|---|---|---|
| 1 | D076 Women's **Casual Fit** Dress - Green | → | D004 Men's **Regular Fit** Kurta - Pink |
| 2 | D036 Women's **Casual Fit** Dress - Black | → | D064 Men's **Regular Fit** Kurta - White |
| 3 | D004 Men's Regular Fit Kurta - Pink | → | D024 Men's **Regular Fit** Kurta - Teal |
| … | | | |

Five documents are pushed out of the top 10 and five enter it. Every document
in the positional ranking has **span = 2**, i.e. the two query terms are
literally adjacent - these are the products whose title really says *Regular
Fit*. This is the clearest demonstration in the whole assignment that a term
with zero idf can still carry information, provided you know where it is.

### 6.2 Case 2 - `short sleeves cotton` : the **order** changes completely

Here both rankings contain the *same* ten documents, but in nearly reversed
order.

| | plain VSM | span | with positions |
|---|---|---|---|
| 1-5 | Women's Solid Midi Dress (×5) | 16 | Women's Casual Fit Dress (×5) - **span 2** |
| 6-10 | Women's Casual Fit Dress (×5) | 2 | Women's Solid Midi Dress (×5) - **span 16** |

The Casual Fit dresses actually contain the phrase *"short sleeves"*
(span 2). In the Midi dresses the three query words are scattered across
about 16 tokens of description. The cosine score cannot tell these two
situations apart - the tf values are almost identical - but the positions can.

### 6.3 Case 3 - `comfortable regular wear` : the VSM **cannot rank at all**

All three terms have df = 100. The query vector is entirely zero, plain
lnc.ltc has nothing to work with, and the engine has to fall back to weights
without idf. The positional information is the only real signal left, and it
still separates the documents sensibly (two documents change places in the top
10). This is the pathological case that the boilerplate in this corpus creates.

### 6.4 Phrase search versus the VSM on the same query

| query | documents the VSM scores | documents containing the phrase |
|---|---|---|
| `zip closure` | 20 | **0** |
| `cotton shirt` | 63 | 5 |
| `stretch denim` | 20 | 7 |

`zip closure` is the interesting one. **Both words are in the corpus** - `zip`
appears in *"zip fly"* (jeans) and `closure` in *"button closure"* (shirts) -
so a free-text search happily returns 20 documents. But no document ever puts
the two words next to each other, so the exact-phrase answer is correctly
**zero**. Without a positional index there is no way to know that; the VSM
would confidently return 20 wrong answers.

### 6.5 The effect of k

`black WITHIN/k cotton`:

| k | 1 | 2 | 3 | 4 | 6 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| documents | 0 | 1 | 6 | 10 | 10 | 10 | 10 |

As the window loosens more documents qualify, until it saturates at the number
of documents containing both terms at all. **That saturation point is where a
proximity query stops being different from a boolean AND** - which is a neat
way of seeing what the parameter is actually buying you.

### 6.6 Terms that are not in the corpus

| query | behaviour |
|---|---|
| `cotton umbrella` | `umbrella` is unknown and has no close neighbour; it is dropped, and the query still ranks on `cotton`. |
| `leather laptop backpack` | every term unknown, no corrections possible → **0 results**, which is the honest answer. |
| `cotten shrt` | 0 results raw → corrected to `cotton shirt` → the five Checked Cotton Shirts. |
| `blak hoodei` | 0 results raw → corrected to `black hoodie` → the four black hoodies. |

---

## 7. Novelty - what we added beyond the handout

### 7.1 PB-VSM: Proximity-Boosted Vector Space Model

**The problem.** Plain lnc.ltc throws word positions away. Two documents can
get an identical cosine score even though one contains *"cotton shirt"* as a
phrase and the other mentions *cotton* in the first line and *shirt* in the
last. For a shopping search those are not the same product. And on this
particular corpus the problem is worse than usual, because 27 terms have
idf = 0 and are invisible to the cosine score entirely (Section 3.3).

**The idea.** We already built a positional index for Part C. Instead of using
it only for a separate yes/no phrase mode, we use it to *re-score* the ranked
list:

```
final_score = cosine_score × (1 + α × coverage × tightness)

    matched   = how many distinct query terms the document actually contains
    asked     = how many query terms are in the dictionary
    span      = width of the shortest window of the document containing
                all the matched terms
    coverage  = matched / asked        (did we find most of the query?)
    tightness = matched / span         (1.0 when the terms are adjacent)
```

`span` is computed by `novelty.smallest_window()`, a **linear-time sliding
window** over the merged position lists (this matters - the boilerplate terms
have very long position lists, so the naive all-pairs version was noticeably
slow).

**Why this shape.** The ideal document is one where all the query terms appear
as a phrase: then `span = matched`, so `tightness = 1`, `coverage = 1`, and the
factor is exactly 1 - the maximum. A document where the terms are 30 tokens
apart gets a factor near 0 and is left essentially unchanged. The product
`coverage × tightness` is the trade-off between *finding more of the query* and
*finding it closer together*, which is why in Case 2 a document matching 2 of 3
terms adjacently can legitimately beat one matching 3 of 3 terms 16 tokens
apart.

**Crucially, `matched` counts terms whose idf is 0.** Those terms are invisible
to the cosine score but they still have positions - and on this corpus they are
precisely the interesting ones (*wear*, *regular*, *festive*, *fit*).

**Choosing α.** α = 0 collapses PB-VSM back to the plain Part B ranking
exactly, which is a useful property: the same code produces the "before" and
the "after", so the comparison in Section 6 is honest. We settled on **α = 0.5**
by hand: a perfectly adjacent phrase then gets a 50 % bonus, which is enough to
lift it above a document that only beat it on tf, but not so much that the
cosine ordering is destroyed. The GUI exposes α as a slider so this can be
demonstrated live.

**Honest limitation:** PB-VSM is a **re-ranker**, not a retriever. The
candidate set is still whatever the VSM matched, so a document containing
*only* zero-idf query terms is never rescued. Because we report just the top
10, re-ranking does change which documents the user sees (Case 1: five
documents swapped), but a full solution would score positions during retrieval.

### 7.2 Query spelling correction

A clothing shop gets *cotten*, *tshrit*, *hoodei*. Plain lnc.ltc returns
nothing for those - the term is not in the dictionary, so it contributes no
weight, and if every term is unknown the result list is empty.

`novelty.correct_query()` runs each unknown query term against the 127
dictionary terms using **Levenshtein distance** (our own implementation, two-row
version with an early exit). The allowed distance depends on word length -
allowing 2 edits on a 4-letter word turns it into a different word - and ties
are broken by **higher document frequency**, the usual "the more common word is
more likely" rule.

| typed | corrected to | distance |
|---|---|---|
| cotten | cotton | 1 |
| shrt | shirt | 1 |
| jaket | jacket | 1 |
| hoodei | hoodie | 2 |
| blak | black | 1 |
| umbrella, laptop, backpack | *(no suggestion)* | - |

The last row matters as much as the others: for a word that genuinely has
nothing to do with clothing the corrector **declines to guess**. Inventing a
correction there would be worse than returning nothing.

---

## 8. What we learned, and what is still wrong

**What the corpus taught us.** We expected stop-word removal to be the
interesting pre-processing decision. It was not - the interesting one was
discovering that 28 % of every document is *identical boilerplate*, which
silently pushes 27 content-bearing terms to idf = 0. Stop-word lists cannot fix
that, because these are not stop words in English; they are stop words *in this
collection*. It is a concrete example of why idf is defined relative to a
collection and not to a language.

**Known limitations, honestly:**

1. **The spelling corrector can make a plausible-but-wrong substitution.**
   `case` → `care` at edit distance 1. Edit distance has no idea what the words
   *mean*. A better version would check whether the correction actually
   improves the result set, or use character bigram overlap as a second filter.
2. **PB-VSM re-ranks rather than retrieves** - see Section 7.1.
3. **α = 0.5 was tuned by eye**, not by a metric. With relevance judgements we
   could sweep α and pick it by MAP or nDCG. We do not have judgements for this
   corpus, and inventing them ourselves would have been circular.
4. **The stemmer is tuned to this vocabulary.** It is conservative by design,
   but the small exception table and length thresholds were chosen by looking
   at these 127 terms. On a bigger corpus we would use real Porter.
5. **Phrase search does not handle a stop word inside the phrase.** `"out of
   stock"` would lose `of`. Because we number positions on the unfiltered
   stream the machinery is already there - a positional gap could be allowed -
   but we did not implement it.

**If we had more time:** proper BM25 as a third ranker to compare against
lnc.ltc; a biword index to make two-word phrase lookup O(1); and a small set of
relevance judgements so that all of the above could be argued with numbers
rather than with examples.

---

## 9. File list

| File | What it is |
|---|---|
| `preprocess.py` | Part A - tokenizer, stop-word policy, our stemmer |
| `indexer.py` | corpus reader, inverted index, positional index, doc lengths |
| `vsm.py` | Part B - lnc.ltc weighting, cosine ranking, score explanation |
| `positional.py` | Part C - phrase search, proximity search, snippets |
| `novelty.py` | **our extension** - PB-VSM re-ranking and spelling correction |
| `engine.py` | the layer all three interfaces share, plus the `WITHIN/k` parser |
| `build_index.py` | writes the index dumps into `output/` |
| `run_tests.py` | Part E - the whole test plan |
| `search_cli.py` | Part D - terminal interface |
| `search_gui.py` | Part D - tkinter window interface |
| `data/corpus_100.txt` | the supplied corpus |
| `output/inverted_index.txt` | **deliverable** - dictionary / inverted index |
| `output/positional_index.txt` | **deliverable** - positional index |
| `output/dictionary.txt` | term / df / idf / cf table, sorted by df |
| `output/doc_lengths.txt` | cosine normalisation lengths |
| `output/corpus_stats.txt` | collection statistics, incl. the 27 idf = 0 terms |
| `output/test_results.txt` | **deliverable** - full Part E output |
| `screenshots/` | **deliverable** - 11 screenshots of the application |

Every module also runs standalone (`python preprocess.py`, `python vsm.py`, …)
and prints a small self-check of just that stage. We used those constantly
while building it.
