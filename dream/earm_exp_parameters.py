# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:35:09 2015

@author: Erin
"""

#Import whole cell volume and scaling factor for mitchondrial membrane volume
from earm.shared import mito_fractional_volume
from scipy.constants import N_A
import numpy as np

cell_vol = 1.661e-12 #1.661 picoL -- volume of HeLa cytoplasm
mito_vol = mito_fractional_volume*cell_vol #approximate volume of HeLa mitochondria (7% of cytoplasmic volume)

cyto_bi_mol_rxn_scale = 1./(N_A*cell_vol)
mito_bi_mol_rxn_scale = 1./(N_A*mito_vol)

cyto_kd_mol_rxn_scale = N_A*cell_vol
mito_kd_mol_rxn_scale = N_A*mito_vol

kf_lower_cyto = np.log10(1e-4*cyto_bi_mol_rxn_scale)
kf_upper_cyto = np.log10(1e11*cyto_bi_mol_rxn_scale)
kf_lower_mito = np.log10(1e-4*mito_bi_mol_rxn_scale)
kf_upper_mito = np.log10(1e11*mito_bi_mol_rxn_scale)
kr_lower_cyto = np.log10(1e-15*cyto_kd_mol_rxn_scale)
kr_upper_cyto = np.log10(1e3*cyto_kd_mol_rxn_scale)
kr_lower_mito = np.log10(1e-15*mito_kd_mol_rxn_scale)
kr_upper_mito = np.log10(1e3*mito_kd_mol_rxn_scale)
kc_lower = -6
kc_upper = 3

print 'kf mito: ',kf_lower_mito,' - ',kf_upper_mito
print 'kr mito: ',kr_lower_mito,' - ',kr_upper_mito
print 'kf cyto: ',kf_lower_cyto,' - ',kf_upper_cyto
print 'kr cyto: ',kr_lower_cyto,' - ',kr_upper_cyto
#All kr values are actually KDs
#sd and uniform upper/lower limits are all in log10 space


earm_rates = {'bind_L_R_to_LR_kf':{'type': 'normal', 'mean': 1e4*cyto_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB': ['1D4V', '1D0G', '1DU3']},
              'bind_L_R_to_LR_kr':{'type': 'normal', 'mean': 10*cyto_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Truneh 2000']},
              'convert_LR_to_DISC_kc':{'type': 'uniform', 'lower': kc_lower, 'upper': kc_upper},
              'bind_DISC_C8pro_to_DISCC8pro_kf':{'type':'uniform', 'lower':kf_lower_cyto, 'upper':kf_upper_cyto},
              'bind_DISC_C8pro_to_DISCC8pro_kr':{'type':'uniform', 'lower':kr_lower_cyto, 'upper':kr_upper_cyto},
              'catalyze_DISCC8pro_to_DISC_C8A_kc':{'type':'uniform', 'lower':kc_lower, 'upper':kc_upper},
              'bind_C8A_BidU_to_C8ABidU_kf': {'type': 'uniform', 'lower':kf_lower_cyto, 'upper':kf_upper_cyto},
              'bind_C8A_BidU_to_C8ABidU_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'catalyze_C8ABidU_to_C8A_BidT_kc': {'type': 'uniform', 'lower': kc_lower, 'upper': kc_upper},
              'bind_DISC_flip_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'bind_DISC_flip_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'bind_BAR_C8A_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'bind_BAR_C8A_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'equilibrate_SmacC_to_SmacA_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'equilibrate_SmacC_to_SmacA_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'equilibrate_CytoCC_to_CytoCA_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'equilibrate_CytoCC_to_CytoCA_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'bind_CytoCA_ApafI_to_CytoCAApafI_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kr_upper_cyto},
              'bind_CytoCA_ApafI_to_CytoCAApafI_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'catalyze_CytoCAApafI_to_CytoCA_ApafA_kc': {'type': 'uniform', 'lower':kc_lower, 'upper': kc_upper},
              'convert_ApafA_C9_to_Apop_kf': {'type': 'normal', 'mean': 3.49e1*cyto_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB': ['4RHW']},
              'convert_ApafA_C9_to_Apop_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'bind_Apop_C3pro_to_ApopC3pro_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'bind_Apop_C3pro_to_ApopC3pro_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'catalyze_ApopC3pro_to_Apop_C3A_kc': {'type': 'normal', 'mean': 1e-1, 'sd': 1./3, 'source': 'experimental', 'ref': ['Garcia-Calvo 1999', 'Zou 2003']},
              'bind_Apop_XIAP_kf': {'type': 'normal', 'mean':2.52e3*cyto_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB': ['1NW9']},
              'bind_Apop_XIAP_kr': {'type': 'normal', 'mean':11e-9*cyto_kd_mol_rxn_scale, 'sd': 1./3, 'source': 'experimental', 'ref': ['Boatright 2003', 'Sun 2000']},
              'bind_SmacA_XIAP_kf': {'type': 'normal', 'mean':8.17e5*cyto_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB': ['1G73']},
              'bind_SmacA_XIAP_kr': {'type': 'normal', 'mean':1e-6*cyto_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Liu 2000']},
              'bind_C8A_C3pro_to_C8AC3pro_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'bind_C8A_C3pro_to_C8AC3pro_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'catalyze_C8AC3pro_to_C8A_C3A_kc': {'type': 'normal', 'mean':.37, 'sd':1./3, 'source': 'experimental', 'ref': ['Garcia-Calvo 1999']},
              'bind_XIAP_C3A_to_XIAPC3A_kf': {'type': 'normal', 'mean':2.5e6*cyto_bi_mol_rxn_scale, 'sd':1., 'source': 'experimental', 'ref': ['Riedl 2001']},
              'bind_XIAP_C3A_to_XIAPC3A_kr': {'type': 'normal', 'mean':.5e-9*cyto_kd_mol_rxn_scale, 'sd':1./3, 'source': 'experimental', 'ref': ['Deveraux 1997', 'Suzuki 2001', 'Scott 2001', 'Silke 2001']},
              'catalyze_XIAPC3A_to_XIAP_C3ub_kc': {'type': 'uniform', 'lower': kc_lower, 'upper': kc_upper},
              'bind_C3A_PARPU_to_C3APARPU_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'bind_C3A_PARPU_to_C3APARPU_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'catalyze_C3APARPU_to_C3A_PARPC_kc': {'type': 'normal', 'mean': 1., 'sd': 2./3, 'source': 'experimental', 'ref': ['Garcia-Calvo 1999', 'Pop 2003']},
              'bind_C3A_C6pro_to_C3AC6pro_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'bind_C3A_C6pro_to_C3AC6pro_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'catalyze_C3AC6pro_to_C3A_C6A_kc': {'type': 'normal', 'mean': 9.1, 'sd': 2./3, 'source': 'experimental', 'ref': ['Garcia-Calvo 1999']},
              'bind_C6A_C8pro_to_C6AC8pro_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'bind_C6A_C8pro_to_C6AC8pro_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'catalyze_C6AC8pro_to_C6A_C8A_kc': {'type': 'uniform', 'lower': kc_lower, 'upper': kc_upper},
              'equilibrate_BidT_to_BidM_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'equilibrate_BidT_to_BidM_kr': {'type': 'normal', 'mean': 100e-6*cyto_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Shivakumar 2014', 'Shamas-Din 2013']},
              'equilibrate_BaxC_to_BaxM_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'equilibrate_BaxC_to_BaxM_kr': {'type': 'normal', 'mean': 50e-9*cyto_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Billen 2008']},
              'equilibrate_BclxLC_to_BclxLM_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'equilibrate_BclxLC_to_BclxLM_kr': {'type': 'normal', 'mean': 50e-9*cyto_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Billen 2008']},
              'equilibrate_BadC_to_BadM_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'equilibrate_BadC_to_BadM_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'equilibrate_NoxaC_to_NoxaM_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kf_upper_cyto},
              'equilibrate_NoxaC_to_NoxaM_kr': {'type': 'uniform', 'lower': kr_lower_cyto, 'upper': kr_upper_cyto},
              'bind_BidM_BaxC_to_BidMBaxC_kf': {'type': 'uniform', 'lower': kf_lower_cyto, 'upper': kr_upper_cyto},
              'bind_BidM_BaxC_to_BidMBaxC_kr': {'type': 'normal', 'mean': 600e-9*cyto_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Walensky 2006']},
              'catalyze_BidMBaxC_to_BidM_BaxM_kc': {'type': 'uniform', 'lower': kc_lower, 'upper': kc_upper},
              'bind_BidM_BaxM_to_BidMBaxM_kf': {'type': 'normal', 'mean': 1.75e4*mito_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB': ['4ZIG', '4BD2']},
              'bind_BidM_BaxM_to_BidMBaxM_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'catalyze_BidMBaxM_to_BidM_BaxA_kc': {'type': 'uniform', 'lower': kc_lower, 'upper': kc_upper},
              'bind_BidM_BakM_to_BidMBakM_kf': {'type': 'normal', 'mean': 7.31e5*mito_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB': ['2M5B']},
              'bind_BidM_BakM_to_BidMBakM_kr': {'type': 'normal', 'mean': 5e-3*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Moldoveanu 2013']},
              'catalyze_BidMBakM_to_BidM_BakA_kc': {'type': 'uniform', 'lower': kc_lower, 'upper': kc_upper},
              'bind_BaxA_BaxM_to_BaxABaxM_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'bind_BaxA_BaxM_to_BaxABaxM_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'catalyze_BaxABaxM_to_BaxA_BaxA_kc': {'type': 'uniform', 'lower':kc_lower, 'upper': kc_upper},
              'bind_BakA_BakM_to_BakABakM_kf': {'type': 'uniform', 'lower':kf_lower_mito, 'upper': kf_upper_mito},
              'bind_BakA_BakM_to_BakABakM_kr': {'type': 'uniform', 'lower':kr_lower_mito, 'upper': kr_upper_mito},
              'catalyze_BakABakM_to_BakA_BakA_kc': {'type': 'uniform', 'lower':kc_lower, 'upper': kc_upper},
              'bind_BidM_Bcl2_kf': {'type': 'uniform', 'lower':kf_lower_mito, 'upper': kf_upper_mito},
              'bind_BidM_Bcl2_kr': {'type': 'normal', 'mean': 100e-9*mito_kd_mol_rxn_scale, 'sd': 1., 'source': 'experimental', 'ref': ['Certo 2006', 'Ku 2011', 'Letai 2002']},
              'bind_BidM_BclxLM_kf': {'type': 'normal', 'mean': 2.37e4*mito_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB':['4QVE']},
              'bind_BidM_BclxLM_kr': {'type': 'normal', 'mean': 200e-9*mito_kd_mol_rxn_scale, 'sd': 1., 'source': 'experimental', 'ref': ['Certo 2006', 'Follis 2014', 'Ku 2011', 'Walensky 2006', 'Kuwana 2005']},
              'bind_BidM_Mcl1M_kf': {'type': 'normal', 'mean': 1.58e4*mito_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB':['2KBW']},
              'bind_BidM_Mcl1M_kr': {'type': 'normal', 'mean': 1e-9*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Certo 2006']},
              'bind_BaxA_Bcl2_kf': {'type': 'normal', 'mean': 5.44e3*mito_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB': ['2XA0']},
              'bind_BaxA_Bcl2_kr': {'type': 'normal', 'mean': 100e-9*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Zhai 2008', 'Ku 2011']},
              'bind_BaxA_BclxLM_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'bind_BaxA_BclxLM_kr': {'type': 'normal', 'mean': 150e-9*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Zhai 2008', 'Ku 2011']},
              'bind_BaxA_Mcl1_kf': {'type': 'normal', 'mean': 1.63e4*mito_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB': ['3PK1']},
              'bind_BaxA_Mcl1_kr': {'type': 'normal', 'mean': 39.5e-9*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Ku 2011']},
              'bind_BakA_Bcl2_kf': {'type': 'uniform', 'lower':kf_lower_mito, 'upper': kf_upper_mito},
              'bind BakA_Bcl2_kr': {'type': 'normal', 'mean': 95e-9*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Zhai 2008', 'Ku 2011']},
              'bind_BakA_BclxLM_kf': {'type': 'normal', 'mean': 1.1e5*mito_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB': ['1BXL', '2LP8']},
              'bind_BakA_BclxLM_kr': {'type': 'normal', 'mean': 15e-9*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Zhai 2008', 'Ku 2011']},
              'bind_BakA_Mcl1_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'bind_BakA_Mcl1_kr': {'type': 'normal', 'mean': 10e-9*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Zhai 2008', 'Ku 2011']},
              'bind_BadM_Bcl2_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'bind_BadM_Bcl2_kr': {'type': 'normal', 'mean': 14e-9*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Certo 2006', 'Ku 2011']},
              'bind_BadM_BclxLM_kf': {'type': 'uniform', 'lower':kf_lower_mito, 'upper': kf_upper_mito},
              'bind_BadM_BclxLM_kr': {'type': 'normal', 'mean': 10e-9*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Certo 2006', 'Ku 2011', 'Walensky 2006', 'Campbell 2015', 'Chen 2013', 'Kuwana 2005']},
              'bind_NoxaM_Mcl1M_kf': {'type': 'normal', 'mean': 2.31e5*mito_bi_mol_rxn_scale, 'sd': 1., 'source': 'TransComp', 'PDB':['2ROD', '2JM6']},
              'bind_NoxaM_Mcl1M_kr': {'type': 'normal', 'mean': 24e-9*mito_kd_mol_rxn_scale, 'sd': 2./3, 'source': 'experimental', 'ref': ['Certo 2006', 'Ku 2011']},
              'assemble_pore_sequential_Bax_2_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'assemble_pore_sequential_Bax_2_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'assemble_pore_sequential_Bax_3_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'assemble_pore_sequential_Bax_3_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'assemble_pore_sequential_Bax_4_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'assemble_pore_sequential_Bax_4_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'assemble_pore_sequential_Bak_2_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'assemble_pore_sequential_Bak_2_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'assemble_pore_sequential_Bak_3_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'assemble_pore_sequential_Bak_3_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'assemble_pore_sequential_Bak_4_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'assemble_pore_sequential_Bak_4_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'pore_transport_complex_BaxA_4_CytoCM_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'pore_transport_complex_BaxA_4_CytoCM_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'pore_transport_dissociate_BaxA_4_CytoCC_kc': {'type': 'uniform', 'lower': kc_lower, 'upper': kc_upper},
              'pore_transport_complex_BaxA_4_SmacM_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'pore_transport_complex_BaxA_4_SmacM_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'pore_transport_dissociate_BaxA_4_SmacC_kc': {'type': 'uniform', 'lower': kc_lower, 'upper': kc_upper},
              'pore_transport_complex_BakA_4_CytoCM_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'pore_transport_complex_BakA_4_CytoCM_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'pore_transport_dissociate_BakA_4_CytoCC_kc': {'type': 'uniform', 'lower': kc_lower, 'upper': kc_upper},
              'pore_transport_complex_BakA_4_SmacM_kf': {'type': 'uniform', 'lower': kf_lower_mito, 'upper': kf_upper_mito},
              'pore_transport_complex_BakA_4_SmacM_kr': {'type': 'uniform', 'lower': kr_lower_mito, 'upper': kr_upper_mito},
              'pore_transport_dissociate_BakA_4_SmacC_kc': {'type': 'uniform', 'lower':kc_lower, 'upper': kc_upper}}
            
                              
              
