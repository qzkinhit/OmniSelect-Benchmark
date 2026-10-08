"""Selection methods, one directory each. Importing this package registers every strategy.

Each <Name>/method.py holds the pure selection function(s) and a strategy adapter registered in
omniselect.core.portfolio.registry. The names re-exported below are the pure functions used by
the tests and the tracks.
"""
from benchmark.Methods.AlignmentOnly.method import alignment_only  # noqa: F401
from benchmark.Methods.AuthenticityOnly.method import auth2_only, auth3_only, auth_bottom, auth_only  # noqa: F401
from benchmark.Methods.CCS.method import ccs  # noqa: F401
from benchmark.Methods.CleanOracle.method import clean_oracle  # noqa: F401
from benchmark.Methods.Cooperative.method import clean_top, coop_herding  # noqa: F401
from benchmark.Methods.Coverage.method import coreset, kmeans_coverage  # noqa: F401
from benchmark.Methods.D4.method import d4  # noqa: F401
from benchmark.Methods.Density.method import density_select  # noqa: F401
from benchmark.Methods.DMF.method import dmf_dynamic, dmf_published_update  # noqa: F401
from benchmark.Methods.DsDm.method import dsdm_scores  # noqa: F401
from benchmark.Methods.DSIR.method import dsir_select  # noqa: F401
from benchmark.Methods.EL2N.method import el2n  # noqa: F401
from benchmark.Methods.FixedFusion.method import fixed_fusion  # noqa: F401
from benchmark.Methods.GLISTER.method import glister  # noqa: F401
from benchmark.Methods.GRADMATCH.method import gradmatch  # noqa: F401
from benchmark.Methods.GraNd.method import grand, grand_expected  # noqa: F401
from benchmark.Methods.Herding.method import herding  # noqa: F401
from benchmark.Methods.IFMates import method as _if_mates  # noqa: F401
from benchmark.Methods.InfluenceOnly.method import influence_only  # noqa: F401
from benchmark.Methods.InfoMax.method import infomax  # noqa: F401
from benchmark.Methods.KCenter.method import kcenter_greedy  # noqa: F401
from benchmark.Methods.LESS import method as _less  # noqa: F401
from benchmark.Methods.QuaDMix.method import quadmix, quadmix_expected_counts, quadmix_published_core  # noqa: F401
from benchmark.Methods.Random.method import random_strategy  # noqa: F401
from benchmark.Methods.RegMix.method import perpcorr_select, regmix_mixture  # noqa: F401
from benchmark.Methods.SemDeDup.method import semdedup  # noqa: F401
from benchmark.Methods.ScoreAdapt import method as _score_adapt  # noqa: F401
from benchmark.Methods.TabAICL import method as _tab_aicl  # noqa: F401
from benchmark.Methods.ZIP import method as _zip  # noqa: F401

__all__ = [
    "herding", "kcenter_greedy", "el2n", "grand", "grand_expected", "ccs", "dmf_dynamic",
    "dmf_published_update", "quadmix", "quadmix_expected_counts", "quadmix_published_core",
    "dsir_select", "semdedup", "density_select", "d4", "dsdm_scores", "regmix_mixture",
    "perpcorr_select", "kmeans_coverage", "fixed_fusion", "glister", "gradmatch", "infomax", "clean_oracle",
]
