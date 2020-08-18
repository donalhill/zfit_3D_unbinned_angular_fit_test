import tensorflow as tf
import zfit
import numpy as np
from root_pandas import read_root, to_root
import json
import matplotlib.pyplot as plt
from zfit.minimize import DefaultToyStrategy

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)

#3D decay rate PDF with acceptance corrections
class CustomPDF(zfit.pdf.ZPDF):
    _PARAMS = ['H0_amp',
    		   'Hm_amp',
    		   'Hp_amp',
    		   'H0_phi',
    		   'Hm_phi',
    		   'Hp_phi',
    		   'costheta_D_p0',
			   'costheta_D_p1',
			   'costheta_D_p2',
			   'costheta_D_p3',
			   'costheta_D_p4',
			   'costheta_D_p5',
			   'costheta_D_p6',
			   'costheta_X_p0',
			   'costheta_X_p1',
			   'costheta_X_p2',
			   'costheta_X_p3',
			   'costheta_X_p4',
			   'costheta_X_p5',
			   'costheta_X_p6',
			   'chi_p0',
			   'chi_p1',
			   'chi_p2',
			   'chi_p3',
			   'chi_p4'
			   ]
    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        costheta_D, costheta_X, chi = x.unstack_x()

        H0_amp = self.params['H0_amp']
        H0_phi = self.params['H0_phi']
        Hm_amp = self.params['Hm_amp']
        Hp_amp = self.params['Hp_amp']
        Hm_phi = self.params['Hm_phi']
        Hp_phi = self.params['Hp_phi']
        costheta_D_p0 = self.params['costheta_D_p0']
        costheta_D_p1 = self.params['costheta_D_p1']
        costheta_D_p2 = self.params['costheta_D_p2']
        costheta_D_p3 = self.params['costheta_D_p3']
        costheta_D_p4 = self.params['costheta_D_p4']
        costheta_D_p5 = self.params['costheta_D_p5']
        costheta_D_p6 = self.params['costheta_D_p6']
        costheta_X_p0 = self.params['costheta_X_p0']
        costheta_X_p1 = self.params['costheta_X_p1']
        costheta_X_p2 = self.params['costheta_X_p2']
        costheta_X_p3 = self.params['costheta_X_p3']
        costheta_X_p4 = self.params['costheta_X_p4']
        costheta_X_p5 = self.params['costheta_X_p5']
        costheta_X_p6 = self.params['costheta_X_p6']
        chi_p0 = self.params['chi_p0']
        chi_p1 = self.params['chi_p1']
        chi_p2 = self.params['chi_p2']
        chi_p3 = self.params['chi_p3']
        chi_p4 = self.params['chi_p4']

        sintheta_D = tf.math.sqrt( 1.0 - costheta_D * costheta_D )
        sintheta_X = tf.math.sqrt( 1.0 - costheta_X * costheta_X )
        sintheta_D2 = (2.0 * sintheta_D * costheta_D)
        sintheta_X2 = (2.0 * sintheta_X * costheta_X)
        coschi = tf.math.cos(chi)
        sinchi = tf.math.sin(chi)
        cos2chi = 2*coschi*coschi-1
        sin2chi = 2*sinchi*coschi

        h_0 = tf.complex(H0_amp*tf.cos(H0_phi),H0_amp*tf.sin(H0_phi))
        h_p = tf.complex(Hp_amp*tf.cos(Hp_phi),Hp_amp*tf.sin(Hp_phi))
        h_m = tf.complex(Hm_amp*tf.cos(Hm_phi),Hm_amp*tf.sin(Hm_phi))

        h_0st = tf.math.conj(h_0)
        h_pst = tf.math.conj(h_p)
        h_mst = tf.math.conj(h_m)

        HpHmst = h_p*h_mst
        HpH0st = h_p*h_0st
        HmH0st = h_m*h_0st

        #True decay rate
        pdf = H0_amp**2 * costheta_D**2 * sintheta_X**2
        pdf += 0.25 * (Hp_amp**2 + Hm_amp**2) * sintheta_D**2 * (1.0 + costheta_X**2)
        pdf += -0.5 * tf.math.real(HpHmst) * sintheta_D**2 * sintheta_X**2 * cos2chi
        pdf += 0.5 * tf.math.imag(HpHmst) * sintheta_D**2 * sintheta_X**2 * sin2chi
        pdf += -0.25 * tf.math.real(HpH0st + HmH0st) * sintheta_D2 * sintheta_X2 * coschi
        pdf += 0.25 * tf.math.imag(HpH0st - HmH0st) * sintheta_D2 * sintheta_X2 * sinchi
        pdf = pdf * 9./8.

        #Normalise to 1
        #pdf = pdf * (1./tf.reduce_sum(pdf))

        #Acceptance corrections
        acc_pdf_costheta_D = costheta_D_p0 + costheta_D_p1 * costheta_D + costheta_D_p2 * costheta_D**2 + costheta_D_p3 * costheta_D**3 + costheta_D_p4 * costheta_D**4 + costheta_D_p5 * costheta_D**5 + costheta_D_p6 * costheta_D**6
        acc_pdf_costheta_X = costheta_X_p0 + costheta_X_p1 * costheta_X + costheta_X_p2 * costheta_X**2 + costheta_X_p3 * costheta_X**3 + costheta_X_p4 * costheta_X**4 + costheta_X_p5 * costheta_X**5 + costheta_X_p6 * costheta_X**6
        acc_pdf_chi = chi_p0 + chi_p1 * chi + chi_p2 * chi**2 + chi_p3 * chi**3 + chi_p4 * chi**4
        acc_pdf_chi = tf.abs(acc_pdf_chi)

        acc_pdf = tf.multiply(acc_pdf_costheta_D,acc_pdf_costheta_X)
        acc_pdf = tf.multiply(acc_pdf,acc_pdf_chi)

        pdf = tf.multiply(pdf,acc_pdf)

        return pdf

def main():

    #Angular phase space
    xobs = zfit.Space('costheta_D_reco', (-1, 1))
    yobs = zfit.Space('costheta_X_VV_reco', (-1, 1))
    zobs = zfit.Space('chi_VV_reco', (-np.pi, np.pi))
    obs = xobs * yobs * zobs

    obs_dict = {"costheta_D_reco": [0,"$\\cos(\\theta_D)$",-1,1,xobs],
                "costheta_X_VV_reco": [1,"$\\cos(\\theta_X)$",-1,1,yobs],
                "chi_VV_reco": [2,"$\\chi$ [rad]",-np.pi,np.pi,zobs]
                }

    #Helicity amplitude parameters
    fL_val = 0.579
    H0_amp = zfit.Parameter("H0_amp", np.sqrt(fL_val), floating=False)
    Hm_amp = zfit.Parameter("Hm_amp", 0.2, 0., 1.)
    def Hp_amp_func(H0_amp, Hm_amp):
	       return tf.sqrt(1. - H0_amp**2 - Hm_amp**2)
    Hp_amp = zfit.ComposedParameter("Hp_amp", Hp_amp_func, dependents=[H0_amp, Hm_amp])
    H0_phi =  zfit.Parameter("H0_phi", 0., floating=False)
    Hp_phi =  zfit.Parameter("Hp_phi", 0., -np.pi, np.pi)
    Hm_phi =  zfit.Parameter("Hm_phi", 0.,-np.pi, np.pi)

    #Acceptance corrections

    #Default costheta_D and costheta_X corrections
    with open("acc_pars_costheta_X_D.json",'r') as f:
        acc_dict = json.load(f)

    costheta_D_p0 = zfit.Parameter("costheta_D_p0" , acc_dict[f"costheta_D_a0"][0], floating=False)
    costheta_D_p1 = zfit.Parameter("costheta_D_p1" , acc_dict[f"costheta_D_a1"][0], floating=False)
    costheta_D_p2 = zfit.Parameter("costheta_D_p2" , acc_dict[f"costheta_D_a2"][0], floating=False)
    costheta_D_p3 = zfit.Parameter("costheta_D_p3" , acc_dict[f"costheta_D_a3"][0], floating=False)
    costheta_D_p4 = zfit.Parameter("costheta_D_p4" , acc_dict[f"costheta_D_a4"][0], floating=False)
    costheta_D_p5 = zfit.Parameter("costheta_D_p5" , acc_dict[f"costheta_D_a5"][0], floating=False)
    costheta_D_p6 = zfit.Parameter("costheta_D_p6" , acc_dict[f"costheta_D_a6"][0], floating=False)

    costheta_X_p0 = zfit.Parameter("costheta_X_p0" , acc_dict[f"costheta_X_VV_a0"][0], floating=False)
    costheta_X_p1 = zfit.Parameter("costheta_X_p1" , acc_dict[f"costheta_X_VV_a1"][0], floating=False)
    costheta_X_p2 = zfit.Parameter("costheta_X_p2" , acc_dict[f"costheta_X_VV_a2"][0], floating=False)
    costheta_X_p3 = zfit.Parameter("costheta_X_p3" , acc_dict[f"costheta_X_VV_a3"][0], floating=False)
    costheta_X_p4 = zfit.Parameter("costheta_X_p4" , acc_dict[f"costheta_X_VV_a4"][0], floating=False)
    costheta_X_p5 = zfit.Parameter("costheta_X_p5" , acc_dict[f"costheta_X_VV_a5"][0], floating=False)
    costheta_X_p6 = zfit.Parameter("costheta_X_p6" , acc_dict[f"costheta_X_VV_a6"][0], floating=False)

    with open("acc_pars_chi.json",'r') as f:
        acc_dict_chi = json.load(f)

    chi_p0 = zfit.Parameter("chi_p0" , acc_dict_chi["chi_VV_a0"][0], floating=False)
    chi_p1 = zfit.Parameter("chi_p1" , acc_dict_chi["chi_VV_a1"][0], floating=False)
    chi_p2 = zfit.Parameter("chi_p2" , acc_dict_chi["chi_VV_a2"][0], floating=False)
    chi_p3 = zfit.Parameter("chi_p3" , acc_dict_chi["chi_VV_a3"][0], floating=False)
    chi_p4 = zfit.Parameter("chi_p4" , acc_dict_chi["chi_VV_a4"][0], floating=False)

    custom_pdf = CustomPDF(obs=obs,
                           H0_amp=H0_amp,
                           Hm_amp=Hm_amp,
                           Hp_amp=Hp_amp,
                           H0_phi=H0_phi,
                           Hm_phi=Hm_phi,
                           Hp_phi=Hp_phi,
                           costheta_D_p0 = costheta_D_p0,
                           costheta_D_p1 = costheta_D_p1,
                           costheta_D_p2 = costheta_D_p2,
                           costheta_D_p3 = costheta_D_p3,
                           costheta_D_p4 = costheta_D_p4,
                           costheta_D_p5 = costheta_D_p5,
                           costheta_D_p6 = costheta_D_p6,
                           costheta_X_p0 = costheta_X_p0,
                           costheta_X_p1 = costheta_X_p1,
                           costheta_X_p2 = costheta_X_p2,
                           costheta_X_p3 = costheta_X_p3,
                           costheta_X_p4 = costheta_X_p4,
                           costheta_X_p5 = costheta_X_p5,
                           costheta_X_p6 = costheta_X_p6,
                           chi_p0 = chi_p0,
                           chi_p1 = chi_p1,
                           chi_p2 = chi_p2,
                           chi_p3 = chi_p3,
                           chi_p4 = chi_p4)

    integral = custom_pdf.integrate(limits=obs)  # = 1 since normalized

    #Load data
    path = "test.root"
    df = read_root(path, "RooTreeDataStore_data_data", columns = ["costheta_D_reco","costheta_X_VV_reco","chi_VV_reco","n_sig_sw"])
    df_data = df[["costheta_D_reco","costheta_X_VV_reco","chi_VV_reco"]]
    df_w = df["n_sig_sw"]

    data = zfit.Data.from_pandas(df_data, obs=obs, weights=df_w)

    # Stage 1: create an unbinned likelihood with the given PDF and dataset
    nll = zfit.loss.UnbinnedNLL(model=custom_pdf, data=data)

    # Stage 2: instantiate a minimiser (in this case a basic minuit
    minimizer = zfit.minimize.Minuit(tolerance=1e-5)

    #Stage 3: minimise the given negative likelihood
    result = minimizer.minimize(nll)

    param_errors, _ = result.errors(method="minuit_minos")

    print("Function minimum:", result.fmin)
    print("Converged:", result.converged)
    print("Full minimizer information:", result.info)

    params = result.params
    print(params)

    #Plot results as 1D projections in the angles

    bins = 30
    data_np = zfit.run(data)
    print(data_np)

    for v in obs_dict:

        fig, ax = plt.subplots(figsize=(7,7))
        lower = obs_dict[v][2]
        upper = obs_dict[v][3]
        counts, bin_edges = np.histogram(df_data[v], bins=bins, range=(lower, upper), weights=df["n_sig_sw"])
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
        err = np.sqrt(counts)
        #Normalise
        err = err / np.sum(counts)
        counts = counts / np.sum(counts)

        #Plot the data
        plt.errorbar(bin_centres, counts, yerr=err, fmt='o', color='xkcd:black')

        #Plot the PDF
        n_p = 1000
        x_plot = np.linspace(lower, upper, num=n_p)
        x_plot_data = zfit.Data.from_numpy(array=x_plot, obs=obs_dict[v][4])

        #Other dims
        other_obs = []
        for k in obs_dict:
            if(k!=v):
                other_obs.append(obs_dict[k][4])

        #Observables to integrate over
        obs_int = other_obs[0] * other_obs[1]

        #Integrating over the other dimensions
        pdf_x = custom_pdf.partial_integrate(x=x_plot_data, limits=obs_int)
        y_plot = zfit.run(pdf_x)
        #Normalise
        y_plot = y_plot / np.sum(y_plot) * (n_p/bins)

        plt.plot(x_plot, y_plot, color='crimson')

        #Labels and axis ranges
        plt.xlabel(obs_dict[v][1],fontsize=25)
        bin_width = float(upper - lower)/bins
        y_min, y_max = ax.get_ylim()
        plt.ylim(0.0,y_max*1.4)
        plt.xlim(lower, upper)
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.tight_layout()
        fig.savefig(f"{v}_unbinned_fit.pdf")

if __name__ == '__main__':
    main()
