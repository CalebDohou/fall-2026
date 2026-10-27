********************************************************************************
* In-Class Activity: Residential Mobility Preferences
* Simplified Replication Exercise
********************************************************************************
version 17
clear all
set more off

capture log close
log using mobility_activity.log, replace

* Set seed for reproducibility
set seed 12345

********************************************************************************
* 1. LOAD DATA
********************************************************************************

* Import data from GitHub
import delimited "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/LectureNotes/11-SubjExp/SCEmobilityExample.csv", clear

* Examine the data
describe
summarize

********************************************************************************
* 2. DATA EXPLORATION
********************************************************************************

* Count individuals and scenarios
* -distinct- is user-written; install it from SSC if missing
capture which distinct
if _rc ssc install distinct
distinct scuid
distinct scuid scennum
distinct scuid blkscen
distinct scuid blkscen altnum

* Summary statistics for key variables
summarize ratio income homecost crime dist family moved

* Histograms of choice probabilities
histogram ratio, bin(30) title("Distribution of Choice Ratios") ///
    xtitle("Log Odds Ratio") graphregion(color(white))
graph export ratio_hist.png, replace

********************************************************************************
* 3. BASIC ESTIMATION: MEDIAN REGRESSION
********************************************************************************

* Declare panel structure
xtset scuid

* Basic quantile regression (median)
qreg ratio income homecost crime dist family size mvcost taxes ///
    norms schqual withincitymove copyhome moved

* Store results
estimates store basic_qreg

********************************************************************************
* 3B. OLS FOR COMPARISON (INCONSISTENT)
********************************************************************************

* OLS regression
regress ratio income homecost crime dist family size mvcost taxes ///
    norms schqual withincitymove copyhome moved

estimates store ols_reg

* Clustered standard errors
regress ratio income homecost crime dist family size mvcost taxes ///
    norms schqual withincitymove copyhome moved, vce(cluster scuid)

estimates store ols_cluster

* Compare OLS vs Median Regression
estimates table ols_reg ols_cluster basic_qreg, ///
    b(%9.3f) se stats(N r2) ///
    title("OLS vs Quantile Regression")

********************************************************************************
* 4. BOOTSTRAPPED STANDARD ERRORS
********************************************************************************

// * Cluster-bootstrapped quantile regression
// bootstrap, cluster(scuid) reps(100): ///
//     qreg ratio income homecost crime dist family size mvcost taxes ///
//         norms schqual withincitymove copyhome moved
// 
// estimates store bs_qreg

********************************************************************************
* 5. COMPUTE WILLINGNESS-TO-PAY
********************************************************************************

* Manual WTP calculation (after basic qreg)
quietly qreg ratio income homecost crime dist family size mvcost taxes ///
    norms schqual withincitymove copyhome moved

* Extract coefficients
scalar b_inc = _b[income]
scalar b_crime = _b[crime]
scalar b_dist = _b[dist]
scalar b_moved = _b[moved]

* WTP to avoid an increase dx in attribute x (utility is linear in log income):
*   b_inc*ln(Y - WTP) = b_inc*ln(Y) + b_x*dx  =>  WTP = (1 - exp(b_x*dx/b_inc)) * Y
* Using median income = $65,000
scalar medinc = 65000

* Crime WTP (for doubling crime rate: ln(2))
scalar wtp_crime = (1 - exp(b_crime/b_inc * ln(2))) * medinc
display "WTP to avoid doubling crime: $" wtp_crime

* Distance WTP (dist is in hundreds of miles)
scalar wtp_dist = (1 - exp(b_dist/b_inc)) * medinc
display "WTP to avoid living 100 miles farther away: $" wtp_dist

* Moving cost WTP
scalar wtp_moved = (1 - exp(b_moved/b_inc)) * medinc
display "Non-pecuniary moving cost (WTP to avoid moving): $" wtp_moved

********************************************************************************
* 6. BOOTSTRAPPED WTP WITH UNCERTAINTY
********************************************************************************

* Bootstrap WTP estimates (fewer reps for speed)
bootstrap crime_wtp = ((1-exp(_b[crime]/_b[income]*ln(2)))*65000) ///
         dist_wtp = ((1-exp(_b[dist]/_b[income]))*65000) ///
         moved_wtp = ((1-exp(_b[moved]/_b[income]))*65000), ///
         cluster(scuid) reps(20): ///
    qreg ratio income homecost crime dist family size mvcost taxes ///
        norms schqual withincitymove copyhome moved

log close
exit

********************************************************************************
* 6B. BOOTSTRAPPED WTP WITH UNCERTAINTY AND HETEROGENEITY
********************************************************************************

* Bootstrap WTP estimates (fewer reps for speed)
bootstrap crime_wtp = ((1-exp(_b[crime]/_b[income]*ln(2)))*65000) ///
         dist_wtp = ((1-exp(_b[dist]/_b[income]))*65000) ///
         moved_wtp = ((1-exp(_b[moved]/_b[income]))*65000), ///
         cluster(scuid) reps(20): ///
    qreg ratio income homecost crime dist family size mvcost taxes ///
        norms schqual withincitymove copyhome moved if age<40

* Bootstrap WTP estimates (fewer reps for speed)
bootstrap crime_wtp = ((1-exp(_b[crime]/_b[income]*ln(2)))*65000) ///
         dist_wtp = ((1-exp(_b[dist]/_b[income]))*65000) ///
         moved_wtp = ((1-exp(_b[moved]/_b[income]))*65000), ///
         cluster(scuid) reps(20): ///
    qreg ratio income homecost crime dist family size mvcost taxes ///
        norms schqual withincitymove copyhome moved if age>=40


********************************************************************************
* 7. COMPARE RESULTS
********************************************************************************

estimates table basic_qreg bs_qreg, ///
    b(%9.3f) star stats(N) ///
    title("Comparison: Standard vs. Bootstrapped SE")

********************************************************************************
* END
********************************************************************************

log close
exit

