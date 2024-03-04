from typing import List, Tuple, Dict
from itertools import chain, permutations

import scipy.stats as ss
import pandas as pd

import weight_parser
import utils as u


logger = u.get_logger(name=__file__)


def spearmanr_combination(weight_avgs: List[weight_parser.WeightAvg], naming: str = "seed") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Calculates spearmanr across all possible combinations"""
    corrs, pvals = [], []
    names = chain(*[[(f"{wa.model_name}_{seed}", wa) for seed in wa.seeds] for wa in weight_avgs])
    for (c1, wa1), (c2, wa2) in permutations(iterable=names, r=2):
        m1, s1 = c1.split("_")
        m2, s2 = c2.split("_")
    
        lst1, lst2 = u.to_list(wa1.xai_dicts[int(s1)]), u.to_list(wa2.xai_dicts[int(s2)])
        if u.check_nan(lst1) or u.check_nan(lst2):
            logger.warn("There is nan value in one of seeds %s or %s", c1, c2)
        if naming == "xai":
            c1, c2 = f"{wa1.xai_method} {s1}", f"{wa2.xai_method} {s2}"
        else:
            c1, c2 = c1.replace("_", " "), c2.replace("_", " ")
        corr, pval = ss.spearmanr(lst1, lst2)
        corrs.append({"compare1": c1, "compare2": c2, "corr": corr})
        pvals.append({"compare1": c1, "compare2": c2, "pval": pval})
    
    corrs, pvals = pd.DataFrame(corrs), pd.DataFrame(pvals)
    corrs = corrs.pivot(columns="compare2", index="compare1", values="corr")
    pvals = pvals.pivot(columns="compare2", index="compare1", values="pval")
    return corrs, pvals


def spearmanr_vs(weight_avgs: List[weight_parser.WeightAvg],
                 meta_dicts: Dict[str, float],
                 naming: str = "xai"):
    key = {"xai": "Deep Learning", "model": "XAI Method"}[naming]
    corrs, pvals = [], []
    for meta in meta_dicts:
        meta_stats = u.to_list(meta_dicts[meta])
        for wa in weight_avgs:
            for seed in wa.seeds:
                dl_stats = u.to_list(wa.xai_dicts[seed])
                corr, pval = ss.spearmanr(meta_stats, dl_stats)
                corrs.append({"Conventional": meta, "corr": corr,
                              key: f"{wa.model_name} {seed}" if naming == "xai" else f"{wa.xai_method} {seed}"})
                pvals.append({"Conventional": meta, "pval": pval,
                              key: f"{wa.model_name} {seed}" if naming == "xai" else f"{wa.xai_method} {seed}"})
    corrs, pvals = pd.DataFrame(corrs), pd.DataFrame(pvals)
    corrs = corrs.pivot(columns="Conventional", index=key, values="corr")
    pvals = pvals.pivot(columns="Conventional", index=key, values="pval")
    return corrs, pvals
