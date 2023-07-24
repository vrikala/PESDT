# Import external modules
from PESDT.plot import Plot
import matplotlib
# matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.io as io
# from pyJETPPF.JETPPF import JETPPF
from PESDT.analyse import AnalyseSynthDiag
from collections import OrderedDict

font = {'family': 'normal',
        'weight': 'normal',
        'size': 14}
matplotlib.rc('font', **font)
import matplotlib.font_manager as font_manager
# path = '/usr/share/fonts/msttcore/arial.ttf'
path = '/usr/share/fonts/gnu-free/FreeSans.ttf'
# path = '/home/bloman/fonts/msttcore/arial.ttf'
prop = font_manager.FontProperties(fname=path)
matplotlib.rcParams['font.family'] = prop.get_name()
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = prop.get_name()
matplotlib.rc('lines', linewidth=1.2)
matplotlib.rc('axes', linewidth=1.2)
matplotlib.rc('xtick.major', width=1.2, pad=7)
matplotlib.rc('ytick.major', width=1.2, pad=7)
matplotlib.rc('xtick.minor', width=1.2)
matplotlib.rc('ytick.minor', width=1.2)

# The directory where the PESDT 2ddata.pkl, synt_diag.json, and proc_synth_diag.json
# are located in
workdir = 'PESDT_cases/'

# Dictionary of the edge2d-eirene cases
# write your case under key 'case'
cases = {'1': {
            'sim_color': 'b', 'sim_zo': 1,
            'case': 'home_jhl7340_edge2d_tran_nhorst_81472_nov_1419_seq#2_tran',
			},
		}

# setup plot figures
left  = 0.2  # the left side of the subplots of the figure
right = 0.85    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.93      # the top of the subplots of the figure
wspace = 0.25   # the amount of width reserved for blank space between subplots
hspace = 0.15  # the amount of height reserved for white space between subplots

# Hydrogenic line emissions of format wl: [n, m], where n is the excited state
# and m is the (exicitated) state after emission
Hlines_dict = OrderedDict([
#	('1215.2', ['2', '1']), # Ly_a
#	('1025.3', ['3', '1']), # Ly_b
#	('972.1', ['4', '1']), # Ly_gamma
	('6561.9', ['3', '2']), # Bal_a
#	('4339.9', ['5', '2']), # Bal_b
	('4101.2', ['6', '2']), # Bal_gamma
#	('3969.5', ['7', '2'])  # Bal_delta
])

spec_line_dict = {
	'1':  # HYDROGEN
		{'1': Hlines_dict}
}
diag = 'KS3O'

fig0, ax0 = plt.subplots(nrows=3, ncols=1, figsize=(6, 12), sharex=True)
fig1, ax1 = plt.subplots(nrows=len(Hlines_dict), ncols=1, figsize=(6, 12), sharex=True)
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
# fig2, ax2 = plt.subplots(nrows=len(carbon_lines_dict), ncols=1, figsize=(6, 12), sharex=True)
# plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
#fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), sharex=True)
#plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

for casekey, case in cases.items():

	plot_dict = {
		'spec_line_dict':spec_line_dict, 
		#'prof_param_defs':{'diag': diag, 'axs': ax0,
		#				   'include_pars_at_max_ne_along_LOS': False,
		#				   'include_sum_Sion_Srec': False,
		#			   	   'include_target_vals': False,
		#				   'Sion_H_transition':[[2,1], [3,2]],
		#				   'Srec_H_transition':[[7,2]],
		#				   'xlim': [2.15, 2.90],
		#				   'coord': 'R', # 'angle' 'R' 'Z'
		#				   'color': case['sim_color'], 'zorder': 10},
		#'prof_Hemiss_defs': {'diag': diag,
		#					 'lines': spec_line_dict['1']['1'],
		#					 'excrec': True,
		#					 'axs': ax1,
		#					 'coord': 'R',# 'angle' 'R' 'Z'
		#					 'color': case['sim_color'],
		#					 'zorder': 10,
		#					 'write_csv': True},
		'2d_defs': {'lines': spec_line_dict,
					'diagLOS': [diag],
					#'Rrng': [2.36, 2.96],
					#'Zrng': [-1.73, -1.29],
					'Rrng': [1.5, 4],
					'Zrng':[-1.73, 2],
					'max_emiss':1.e22, # adjust color scale
					'save': False},
	}

	o = Plot(workdir, case['case'], plot_dict=plot_dict)

# Print out results dictionary tree
# Plot.pprint_json(o.res_dict['mds1_hr']['1']['los_int'])

plt.show()
