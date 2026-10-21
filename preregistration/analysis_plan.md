# Pre-registration: quick-commerce ads geo holdout

**Registered:** 21 Oct 2026, before stage-2 data (weeks 88–104) exists.
**Commit rule:** this file and `design.json` are committed to GitHub before stage 2 is run. Any later change is a new, dated commit with the reason stated.
**Design notebook:** `notebooks/03A_experiment_design.ipynb`. All numbers below come from it.

## 1. Question

What incremental revenue do quick-commerce (QC) ads create per ₹1 of spend?

MMM v1 estimates ₹0.71 per ₹1, with a 90% interval of 0.14–3.49. That uncertainty puts ₹29.7 Cr of revenue at risk, the most of any channel, which is why this channel is tested (rule set in notebook 02B).

## 2. Hypotheses

- **H0:** pausing QC ads does not change quick-commerce revenue in the paused cities.
- **H1:** quick-commerce revenue in the paused cities changes. The test is two-sided, α = 0.05.

## 3. Design

| Item | Specification |
|---|---|
| Treatment | QC ad spend set to ₹0 |
| Treated cities | Bengaluru, Hyderabad (chosen for the smallest detectable effect among all 28 city pairs) |
| Control cities | Delhi NCR, Mumbai, Chennai, Pune, Kolkata, Ahmedabad: no change to any channel |
| Test window | Weeks 88–95 (8 weeks) |
| Follow-up | Weeks 96–104, to check for carry-over and recovery |
| Other channels | Unchanged in all cities |

## 4. Metrics

- **Primary:** weekly quick-commerce revenue (`rev_quick_commerce_lakh`), summed over the treated cities.
- **Secondary:** weekly total revenue (`revenue_lakh`) in the treated cities. This checks whether paused QC sales moved to other sales channels.

## 5. Analysis

1. **Primary estimator: synthetic control.**
   - A twin is fitted to the treated cities' QC revenue in weeks 62–87 (26 weeks), using non-negative least squares over the 6 control cities. No intercept.
   - Effect = mean(treated − twin) over weeks 88–95, minus mean(treated − twin) over weeks 62–87.
2. **Inference: placebo in time.**
   - The same estimator is applied to every 8-week window in weeks 27–87 (54 windows) with no treatment.
   - The 90% and 95% intervals of the effect use the standard deviation of those placebo effects (pre-test estimate: ₹0.142 lakh per week).
3. **Robustness 1: placebo in space.** Rerun the estimator treating each pair of control cities as "treated". The real effect should sit outside that distribution.
4. **Robustness 2: difference-in-differences.** Compare log QC revenue, treated vs all 6 controls, for weeks 62–87 vs 88–95.
5. **Converting to ROI.**
   - Spend withheld = treated cities' mean weekly QC spend in weeks 80–87 × (control cities' QC spend in weeks 88–95 ÷ their spend in weeks 80–87) × 8. Pre-test estimate: ₹42 lakh.
   - Test ROI = −(effect per week × 8) ÷ spend withheld. Revenue lost in weeks 96–97 is added if the follow-up shows carry-over.

## 6. Validity checks (reported whatever the result)

- Twin fit error before the test is below 3% of weekly QC revenue. Pre-test value: 1.3%.
- The twin's gap is about 0 in the check weeks 80–87. Pre-test value: −₹0.04 lakh per week.
- Treated and control cities had no other changes in weeks 88–95: spend on the other five channels is checked.

## 7. Decision rule

Contribution margin on QC sales after platform commission is about 30%. So QC ads pay for themselves only above an ROI of about 3.3 (₹1 ÷ 0.30).

| Test result (90% interval for ROI) | Decision |
|---|---|
| Entirely below 1.0 | QC ads return less revenue than they cost. Move QC ad spend to the −50% bound in the optimiser and renegotiate placement fees with the platforms. |
| Overlaps 1.0, entirely below 3.3 | QC ads add sales but lose contribution at today's spend. Reduce toward the point where marginal ROI reaches 3.3. |
| Entirely above 3.3 | QC ads are profitable at the margin. Protect or grow them. |

Whatever the result, the lift (spend withheld, revenue lost, its standard error) is entered into MMM v2 as a lift-test calibration (Stage E). The MMM v1 vs test comparison is reported even if they disagree.

## 8. What would make the result invalid

- Twin fit error above 3% before the test.
- A shock in a treated city during weeks 88–95 that is unrelated to the test.
- The continuity check failing in stage 2 (weeks 1–87 changed).

If any of these happens, the test is reported as inconclusive. It is not re-analysed with a different method chosen after seeing the data.
