"""Required COT predictors survive sparse training and constrained selection."""
import unittest

import numpy as np
import pandas as pd

from MonthFu.src.modeling import Candidate, candidates, fit_candidate, predict_member, select_recipe


class AllCOTPolicyTests(unittest.TestCase):
    def test_sparse_constant_and_empty_cot_columns_are_retained(self):
        frame = pd.DataFrame({"target_return": np.linspace(-.1,.1,20),
            "price_signal": np.arange(20.), "cot_sparse": [np.nan]*19+[7.],
            "cot_constant": 3., "cot_unavailable": np.nan, "other_empty": np.nan})
        columns = [c for c in frame if c != "target_return"]
        member = fit_candidate(Candidate("all","price_cot","ridge",100,require_all_cot=True),frame,columns)
        self.assertEqual(set(member["required_cot_features"]), {"cot_sparse","cot_constant","cot_unavailable"})
        self.assertTrue(set(member["required_cot_features"]).issubset(member["features"]))
        self.assertNotIn("other_empty",member["features"])
        self.assertTrue(np.isfinite(predict_member(member,frame)).all())
        with self.assertRaisesRegex(ValueError,"schema"):
            predict_member(member,frame.drop(columns="cot_constant"))

    def test_required_schema_does_not_fabricate_past_values(self):
        frame = pd.DataFrame({"target_return":[-.1,.1,-.05,.05],"price_signal":[0.,1.,2.,3.],
                              "cot_archive":[np.nan]*4})
        member = fit_candidate(Candidate("all","price_cot","ridge",100,require_all_cot=True),
                               frame,["price_signal","cot_archive"])
        self.assertTrue(frame.cot_archive.isna().all())
        self.assertEqual(member["cot_feature_audit"][0]["observed_rows"],0)
        self.assertIn("cot_archive",member["features"])

    def test_price_only_winner_is_ineligible_in_required_cot_selection(self):
        specs = [Candidate(n,"price","ridge") for n in ["previous_rf","previous_ar","previous_hw","price","cot"]]
        actual = np.array([-.1,.1,-.05,.05])
        oof = pd.DataFrame({"target_return":actual,**{s.name: actual if s.name=="price" else actual*.8 for s in specs}})
        recipe,ranking = select_recipe(oof,specs,eligible_names={"cot"})
        self.assertEqual(set(recipe["weights"]),{"cot"})
        self.assertFalse(ranking.loc[ranking.recipe.eq("price"),"eligible_for_selection"].iloc[0])

    def test_required_candidates_include_all_cot_groups_only(self):
        required = [s for s in candidates(require_all_cot=True) if s.require_all_cot]
        self.assertTrue(required)
        self.assertTrue({s.group for s in required} <= {"price_cot","price_cot_weather","monthly_core_all_cot","engineered"})
        self.assertTrue(any(s.group=="monthly_core_all_cot" for s in required))


if __name__ == "__main__":
    unittest.main()
