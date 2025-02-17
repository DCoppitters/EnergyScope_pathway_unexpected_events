import os, sys
from pathlib import Path
import time
import pandas as pd
import csv

curr_dir = Path(os.path.dirname(__file__))

pymodPath = os.path.abspath(os.path.join(curr_dir.parent, 'pylib'))
sys.path.insert(0, pymodPath)

from ampl_object import AmplObject
from ampl_preprocessor import AmplPreProcessor
from ampl_collector import AmplCollector
from ampl_uq import AmplUQ
import pickle

#########################
### inputs start here ###
#########################
'''
choose type of model: monthly (MO) or typical days (TD).
NOTE that the specific early stage decisions available only work under TD evaluations (as presented in the paper).
This because the specific starting conditions were simulated considering TD evaluations.

NOTE the MO model can be used for quick evaluations. Only works without an early-stage decision specified (line 58)
'''

type_of_model = 'TD' #MO or TD
nbr_tds = 12  # number of Typical Days (only relevant for TD evaluation)

# file where results (costs and if transition succeeded) are stored
results_file = 'scenarios_pf.csv'

# base name for file where the technologies and resources used at each phase of the transition are stored
results_tech_file_base = 'scenarios_pf_technologies'

# number of unexpected events evaluated
n_unexpected_events = 1000

# starting point in the unexpected event scenarios file
start_point = 0

'''
determine the optimization approach.
EITHER perfect foresight settings
# N_year_opti = [30] #30 means perfect foresight - optimize over entire energy transition
# N_year_overlap = [0] #0 means no overlap needed, i.e., perfect foresight

OR myopic foresight settings
N_year_opti = [10] #10 means myopic foresight - optimize using rolling horizon
N_year_overlap = [5] #5 means overlap between phases to modify decisions when future becomes more clear, i.e., myopic foresight
'''
N_year_opti = [30]
N_year_overlap = [0]

# start over and wipe the results file
start_over = True

'''
# the input data file with costs, availabilities for the technologies and resources: PICK ONE
input_data_file = 'PES_data_year_related.dat' #no specific early-stage decision (works for TD and MO)
input_data_file = 'PES_data_year_related_delay_ren.dat' #delay renewables early-stage decision (works only for TD) 
input_data_file = 'PES_data_year_related_acc_ren.dat' #accelerate renewables early-stage decision (works only for TD)
input_data_file = 'PES_data_year_related_acc_h2.dat' #accelerate hydrogen early-stage decision (works only for TD)
input_data_file = 'PES_data_year_related_no_nuclear.dat' #no nuclear early-stage decision (works only for TD)
input_data_file = 'PES_data_year_related_myopic_baseline.dat' #myopic baseline early-stage decision (works only for TD)
'''
input_data_file = 'PES_data_year_related.dat' #no specific early-stage decision (works for TD and MO)

#######################
### inputs end here ###
#######################

pth_esmy = os.path.join(curr_dir.parent, 'ESMY')
pth_model = os.path.join(pth_esmy, 'STEP_2_Pathway_Model')

## Options for ampl and gurobi
gurobi_options = ['predual=-1',
                  'method = 2',
                  'crossover=0',
                  'prepasses = 3',
                  'barconvtol=1e-6',
                  'presolve=-1']

gurobi_options_str = ' '.join(gurobi_options)

ampl_options = {'show_stats': 1,
                'log_file': os.path.join(pth_model, 'log.txt'),
                'presolve': 10,
                'presolve_eps': 1e-6,
                'presolve_fixeps': 1e-6,
                'show_boundtol': 0,
                'gurobi_options': gurobi_options_str,
                '_log_input_only': False}

# Read the CSV file into a DataFrame
df_events = pd.read_csv('samples_unexpected_events.csv')
names_events = list(df_events.columns) + ['sum_pv', 'sum_wind_on', 'sum_wind_off', 'sum_demand']

years = [
    2020,
    2025,
    2030,
    2035,
    2040,
    2045,
    2050,
]

# all resources considered in ES Pathway
target_resources = [
    'ELECTRICITY', 'GASOLINE', 'DIESEL', 'BIOETHANOL', 'BIODIESEL', 'LFO', 'GAS', 'GAS_RE', 'WOOD', 'WET_BIOMASS',
    'COAL', 'URANIUM', 'WASTE', 'H2', 'H2_RE', 'AMMONIA', 'METHANOL', 'AMMONIA_RE', 'METHANOL_RE', 'ELEC_EXPORT',
    'CO2_EMISSIONS', 'RES_WIND', 'RES_SOLAR', 'RES_HYDRO', 'RES_GEO', 'CO2_ATM', 'CO2_INDUSTRY', 'CO2_CAPTURED'
]

# all technologies considered in ES Pathway
target_technologies = [
    'NUCLEAR', 'NUCLEAR_SMR', 'CCGT', 'CCGT_AMMONIA', 'COAL_US', 'COAL_IGCC', 'PV', 'WIND_ONSHORE', 'WIND_OFFSHORE',
    'HYDRO_RIVER', 'GEOTHERMAL', 'IND_COGEN_GAS', 'IND_COGEN_WOOD', 'IND_COGEN_WASTE', 'IND_BOILER_GAS',
    'IND_BOILER_WOOD', 'IND_BOILER_OIL', 'IND_BOILER_COAL', 'IND_BOILER_WASTE', 'IND_DIRECT_ELEC', 'DHN_HP_ELEC',
    'DHN_COGEN_GAS', 'DHN_COGEN_WOOD', 'DHN_COGEN_WASTE', 'DHN_COGEN_WET_BIOMASS', 'DHN_COGEN_BIO_HYDROLYSIS',
    'DHN_BOILER_GAS', 'DHN_BOILER_WOOD', 'DHN_BOILER_OIL', 'DHN_DEEP_GEO', 'DHN_SOLAR', 'DEC_HP_ELEC', 'DEC_THHP_GAS',
    'DEC_COGEN_GAS', 'DEC_COGEN_OIL', 'DEC_ADVCOGEN_GAS', 'DEC_ADVCOGEN_H2', 'DEC_BOILER_GAS', 'DEC_BOILER_WOOD',
    'DEC_BOILER_OIL', 'DEC_SOLAR', 'DEC_DIRECT_ELEC', 'TRAMWAY_TROLLEY', 'BUS_COACH_DIESEL', 'BUS_COACH_HYDIESEL',
    'BUS_COACH_CNG_STOICH', 'BUS_COACH_FC_HYBRIDH2', 'TRAIN_PUB', 'CAR_GASOLINE', 'CAR_DIESEL', 'CAR_NG',
    'CAR_METHANOL', 'CAR_HEV', 'CAR_PHEV', 'CAR_BEV', 'CAR_FUEL_CELL', 'TRAIN_FREIGHT', 'BOAT_FREIGHT_DIESEL',
    'BOAT_FREIGHT_NG', 'BOAT_FREIGHT_METHANOL', 'TRUCK_DIESEL', 'TRUCK_METHANOL', 'TRUCK_FUEL_CELL', 'TRUCK_ELEC',
    'TRUCK_NG', 'EFFICIENCY', 'DHN', 'GRID', 'H2_ELECTROLYSIS', 'SMR', 'H2_BIOMASS', 'GASIFICATION_SNG',
    'SYN_METHANATION', 'BIOMETHANATION', 'BIO_HYDROLYSIS', 'PYROLYSIS_TO_LFO', 'PYROLYSIS_TO_FUELS', 'ATM_CCS',
    'INDUSTRY_CCS', 'SYN_METHANOLATION', 'METHANE_TO_METHANOL', 'BIOMASS_TO_METHANOL', 'HABER_BOSCH', 'AMMONIA_TO_H2',
    'OIL_TO_HVC', 'GAS_TO_HVC', 'BIOMASS_TO_HVC', 'METHANOL_TO_HVC', 'BATT_LI', 'BEV_BATT', 'PHEV_BATT', 'PHS',
    'TS_DEC_DIRECT_ELEC', 'TS_DEC_HP_ELEC', 'TS_DEC_THHP_GAS', 'TS_DEC_COGEN_GAS', 'TS_DEC_COGEN_OIL',
    'TS_DEC_ADVCOGEN_GAS', 'TS_DEC_ADVCOGEN_H2', 'TS_DEC_BOILER_GAS', 'TS_DEC_BOILER_WOOD', 'TS_DEC_BOILER_OIL',
    'TS_DHN_DAILY', 'TS_DHN_SEASONAL', 'TS_HIGH_TEMP', 'GAS_STORAGE', 'H2_STORAGE', 'DIESEL_STORAGE',
    'GASOLINE_STORAGE', 'LFO_STORAGE', 'AMMONIA_STORAGE', 'METHANOL_STORAGE', 'CO2_STORAGE'
]

# create results csv files (unexpected events and the technology files—one per year)
column_names = target_technologies + target_resources + ["transition_cost", "fail"]
df_res = pd.DataFrame(columns=column_names)

for year in years:
    results_tech_file = '%s_%i.csv' %(results_tech_file_base,year)
    if start_over:
        df_res.to_csv(results_tech_file, index=False)

column_names = names_events + ["transition_cost", "fail"]
dfr = pd.DataFrame(columns=column_names)

# Check if the file does not already exist before saving; overwrite if desired (start_over is True)
if start_over:
    dfr.to_csv(results_file, index=False)
    n_ex_rows = 0
else:
    n_ex_rows = len(pd.read_csv(results_file))

if __name__ == '__main__':

    if type_of_model == 'MO':
        mod_1_path = [os.path.join(pth_model, 'PESMO_model.mod'),
                      os.path.join(pth_model, 'PESMO_store_variables.mod'),
                      os.path.join(pth_model, 'PESMO_RL/PESMO_RL_v7.mod'),
                      os.path.join(pth_model, 'PES_store_variables.mod')]
        mod_2_path = [os.path.join(pth_model, 'PESMO_initialise_2020.mod'),
                      os.path.join(pth_model, 'fix.mod')]
        dat_path = [os.path.join(pth_model, 'PESMO_data_all_years.dat')]
    else:
        mod_1_path = [os.path.join(pth_model, 'PESTD_model.mod'),
                      os.path.join(pth_model, 'PESTD_store_variables.mod'),
                      os.path.join(pth_model, 'PESTD_RL/PESTD_RL_v8.mod'),
                      os.path.join(pth_model, 'PES_store_variables.mod')]
        mod_2_path = [os.path.join(pth_model, 'PESTD_initialise_2020.mod'),
                      os.path.join(pth_model, 'fix.mod')]
        dat_path = [os.path.join(pth_model, 'PESTD_data_all_years.dat'),
                    os.path.join(pth_model, 'PESTD_{}TD.dat'.format(nbr_tds))]

    dat_path += [os.path.join(pth_model, 'PES_data_all_years.dat'),
                 os.path.join(pth_model, 'PES_seq_opti.dat'),
                 os.path.join(pth_model, input_data_file),
                 os.path.join(pth_model, 'PES_data_efficiencies.dat'),
                 os.path.join(pth_model, 'PES_data_set_AGE_2020.dat')]
    dat_path_0 = dat_path + [os.path.join(pth_model, 'PES_data_remaining.dat'),
                             os.path.join(pth_model, 'PES_data_decom_allowed_2020.dat')]

    dat_path += [os.path.join(pth_model, 'PES_data_remaining_wnd.dat'),
                 os.path.join(pth_model, 'PES_data_decom_allowed_2020.dat')]

    ## Paths
    pth_output_all = os.path.join(curr_dir.parent, 'out')


    # Iterate over each row in the DataFrame
    for sample_ex_event_index, sample_ex_event in df_events.iloc[n_ex_rows + start_point : n_ex_rows + n_unexpected_events + start_point].iterrows():

        for m in range(len(N_year_opti)):

            # TO DO ONCE AT INITIALISATION OF THE ENVIRONMENT
            i = 0
            n_year_opti = N_year_opti[m]
            n_year_overlap = N_year_overlap[m]

            case_study = '{}_{}_{}_{}'.format(type_of_model, n_year_opti, n_year_overlap, sample_ex_event_index)
            expl_text = 'GWP budget to reach carbon neutrality with {} years of time window and {} years of overlap'.format(
                n_year_opti, n_year_overlap)

            output_folder = os.path.join(pth_output_all, case_study)
            output_file = os.path.join(output_folder, '_Results.pkl')
            ampl_0 = AmplObject(mod_1_path, mod_2_path, dat_path_0, ampl_options, type_model=type_of_model)
            ampl_0.clean_history()
            ampl_pre = AmplPreProcessor(ampl_0, n_year_opti, n_year_overlap)
            ampl_collector = AmplCollector(ampl_pre, output_file, expl_text)

            t = time.time()

            for i in range(len(ampl_pre.years_opti)):
                # TO DO AT EVERY STEP OF THE TRANSITION
                t_i = time.time()
                curr_years_wnd = ampl_pre.write_seq_opti(i).copy()
                ampl_pre.remaining_update(i)

                ampl = AmplObject(mod_1_path, mod_2_path, dat_path, ampl_options, type_model=type_of_model)

                # Define GWP limits in a dictionary
                gwp_limits = {f'YEAR_{year}': 124000 - index * 20098.84665 for index, year in
                              enumerate(range(2020, 2055, 5))}

                # Set the parameters using the precomputed dictionary
                ampl.set_params('gwp_limit', gwp_limits)

                ampl_uq = AmplUQ(ampl)
                years_wnd = ['YEAR_2025', 'YEAR_2030', 'YEAR_2035', 'YEAR_2040', 'YEAR_2045', 'YEAR_2050']
                
                # years when unexpected events arrive
                years_ex_events = ['YEAR_2035', 'YEAR_2040', 'YEAR_2045', 'YEAR_2050']

                # overwrite starting conditions based on unexpected-event scenario considered
                ampl_uq.transcript_unexpected_events(sample_ex_event, years_ex_events, target_technologies, target_resources)

                solve_result = ampl.run_ampl()
                ampl.get_results()

                if i == 0:
                    ampl_collector.init_storage(ampl)

                if i > 0:
                    curr_years_wnd.remove(ampl_pre.year_to_rm)

                ampl_collector.update_storage(ampl, curr_years_wnd, i)

                ampl.set_init_sol()

                elapsed_i = time.time() - t_i
                print('Time to solve the window #' + str(i + 1) + ': ', elapsed_i)

                if i == len(ampl_pre.years_opti) - 1:
                    elapsed = time.time() - t
                    print('Time to solve the whole problem: ', elapsed)
                    ampl_collector.clean_collector()
                    ampl_collector.pkl()
                    break

        pkl_file = os.path.join(pth_output_all, case_study, '_Results.pkl')
        open_file = open(pkl_file, "rb")
        loaded_results = pickle.load(open_file)
        open_file.close()

        gwp_tot = 1e8
        for year in years:
            gwp_tot_new = \
            loaded_results['TotalGwp'].loc[loaded_results['TotalGwp'].index == 'YEAR_%i' % year, 'TotalGWP'].values[
                0]

            # store the lowest gwp among the years (0 means the transition failed)
            gwp_tot = min(gwp_tot, gwp_tot_new)
        fail = 0 if 0 < gwp_tot <= gwp_limits['YEAR_2050']+0.1 else 1
        transition_cost = 0 if fail == 1 else loaded_results['Transition_cost'].loc[
            loaded_results['Transition_cost'].index == 'YEAR_2050', 'Transition_cost'
        ].values[0]

        new_data = pd.Series([transition_cost, fail])
        resources = loaded_results['Resources']['Res']
        assets = loaded_results['Assets']['F']

        # Summing values for 'pv', 'wind_on', 'wind_off', and 'demand' and adding them to sample_ex_event
        for key in ['avail_pv', 'avail_wind_on', 'avail_wind_off', 'demand']:
            sample_ex_event[f'sum_{key}'] = sum(sample_ex_event[f'{key}_{year}'] for year in range(2035, 2051, 5))

        # Convert sample_ex_event to a pandas Series, concatenate with new_data, and reset the index
        sample = pd.concat([pd.Series(sample_ex_event), new_data]).reset_index(drop=True)

        # Open the existing CSV file in append mode ('a')
        with open(results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(sample)

        # if solution exists, store the technology and resources deployed for each phase
        if fail == 0:

            for year in years:
                resource_values = []
                tech_values = []
                for target_resource in target_resources:
                    if target_resource in resources['YEAR_%i' % year]:
                        resource_value = resources['YEAR_%i' % year][target_resource]
                    else:
                        resource_value = 0.
                    resource_values.append(resource_value)

                for target_tech in target_technologies:
                    if target_tech in assets['YEAR_%i' % year]:
                        tech_value = assets['YEAR_%i' % year][target_tech]
                    else:
                        tech_value = 0  # Set tech_value to 0 if target_tech is not found
                    tech_values.append(tech_value)

                sample_tech = tech_values + resource_values + [transition_cost, fail]

                results_tech_file = '%s_%i.csv' %(results_tech_file_base,year)

                # Open the existing CSV file in append mode ('a')
                with open(results_tech_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(sample_tech)
