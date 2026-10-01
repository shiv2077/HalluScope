"""Independent answer-equivalence judgements made during the audit (2026-09-13).

Rule: "C" if the model answer commits to the reference answer or an obvious alias
(yes/no questions are judged on the yes/no), "I" if it commits to a different
answer, gives no answer, or never states the reference. "U" means the committed
CSV holds the intervention template instead of the model's answer, so no
judgement is possible.

Limitation: the judge (the auditing model) saw the SBERT cosine and label next to
each answer while judging, so these labels are not blind. AMBIGUOUS lists the
borderline calls so agreement can be reported with and without them.
"""

API_C = """0005 0006 0012 0014 0015 0016 0018 0020 0022 0023 0024 0027 0028 0029 0033
0036 0038 0041 0042 0043 0047 0050 0051 0052 0055 0056 0057 0062 0066 0067 0070 0071
0072 0075 0079 0080 0081 0084 0085 0086 0089 0090 0091 0093 0094 0095 0097 0098""".split()
API_I = """0001 0002 0003 0007 0008 0009 0010 0011 0013 0019 0031 0034 0035 0037 0039
0040 0044 0045 0046 0053 0061 0064 0068 0069 0074 0076 0078 0082 0083 0087 0088 0092
0096 0099""".split()

LOCAL_C = "0016 0023 0069 0098".split()
LOCAL_I = """0008 0029 0031 0032 0044 0051 0057 0065 0067 0079 0081 0083 0085 0090
0093 0094""".split()

GOLD_C = [1, 3, 6, 8, 9, 10, 11, 15, 16, 17, 18, 20, 21, 22, 27, 29, 31, 36, 38, 40,
          41, 42, 43, 45, 46, 47, 49, 51, 52, 53, 54, 55, 56, 59, 60, 61, 62, 65, 67,
          68, 69, 70, 71, 73, 76, 77, 78, 79]
GOLD_I = [2, 4, 5, 7, 12, 13, 14, 19, 23, 24, 25, 26, 28, 30, 32, 33, 34, 35, 37, 39,
          44, 48, 50, 57, 58, 63, 64, 66, 72, 74, 75, 80]


def _ids(prefix, width, nums):
    return [f"{prefix}{int(n):0{width}d}" for n in nums]


def _block(prefix, width, c, i, n_total):
    c, i = _ids(prefix, width, c), _ids(prefix, width, i)
    assert not set(c) & set(i), "an id is judged both correct and incorrect"
    out = {k: "U" for k in _ids(prefix, width, range(1, n_total + 1))}
    out.update({k: "C" for k in c})
    out.update({k: "I" for k in i})
    return out


JUDGE = {
    "api": _block("halu_q_", 4, API_C, API_I, 100),
    "local": _block("halu_q_", 4, LOCAL_C, LOCAL_I, 100),
    "gold": _block("gold_", 5, GOLD_C, GOLD_I, 80),
}
assert len(API_C) + len(API_I) == 82 and len(LOCAL_C) + len(LOCAL_I) == 20
assert len(GOLD_C) + len(GOLD_I) == 80 and "U" not in JUDGE["gold"].values()

NOTES = {
    "api": {
        "halu_q_0001": "names Mason Wildlife Sanctuary, not Talbot-Taylor",
        "halu_q_0003": "Best Actress for Piku, not Best Female Debut",
        "halu_q_0040": "names Ali Zafar, not Nadia Ali",
        "halu_q_0044": "gives 'Robert Ellsworth Wills', not James Robert Wills",
        "halu_q_0014": "answers No = reference",
        "halu_q_0028": "states Galveston, Texas",
        "halu_q_0043": "states October 22, 1992",
        "halu_q_0047": "states 2006",
        "halu_q_0050": "answers Yes = reference",
        "halu_q_0062": "states Park Dietz",
        "halu_q_0070": "states Shame",
        "halu_q_0071": "Hredmonath, the Old English month named for Rheda",
        "halu_q_0072": "Kumbaya = Kum ba yah",
        "halu_q_0075": "concludes Dildarian is not an art dealer = no",
        "halu_q_0080": "answers Yes = reference",
        "halu_q_0085": "states Doug TenNapel",
        "halu_q_0091": "answers No = reference",
        "halu_q_0097": "states 1985",
        "halu_q_0018": "right choice, wrong composer named in passing",
        "halu_q_0057": "names Downton Abbey but hedges on the premise",
        "halu_q_0088": "mentions Australia but gives youth culture as the link",
    },
    "local": {
        "halu_q_0069": "answers No = reference",
        "halu_q_0098": "states Channel 4",
        "halu_q_0093": "never names Square One Mall (truncated restatement)",
    },
    "gold": {
        "gold_00018": "about 15,000 vs 15,023",
        "gold_00032": "describes the hijacking, never names Operation Entebbe",
        "gold_00037": "basketball only; reference implies soccer and basketball",
        "gold_00041": "position right (defensive coordinator), team wrong",
        "gold_00068": "the Long War is one of the Ottoman-Habsburg wars",
    },
}

AMBIGUOUS = {
    "halu_q_0018", "halu_q_0057", "halu_q_0071", "halu_q_0088",
    "gold_00018", "gold_00032", "gold_00037", "gold_00041", "gold_00068",
}
