# zfit_3D_unbinned_angular_fit_test

Test code to reproduce error in zfit when using weighted data. Required packages:
 - zfit
 - numpy
 - root-pandas
 - json
 - matplotlib
 - tensorflow

Code reads a toy dataset `test.root` with the angular fit variables `(costheta_D, costheta_X_VV, chi_VV)` and the sWeight branch `n_sig_sw` used to subtract background by assigning weights to each event. Two `JSON` files are loaded to access fixed parameters used in the fit PDF. The code is run by doing:
```
python unbinned_angular_fit.py
```
