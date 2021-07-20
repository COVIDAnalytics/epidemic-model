# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import os
import yaml
import logging
import time
import psutil
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import datetime, timedelta
from functools import partial
from tqdm import tqdm
from scipy.optimize import dual_annealing
from DELPHI_utils_V4_static import (
    DELPHIAggregations, DELPHIDataSaver, DELPHIDataCreator, get_initial_conditions,
    get_mape_data_fitting, create_fitting_data_from_validcases, get_residuals_value,get_past_parameters,get_single_case,get_start_date,
    save_replace_param_jnj
)
from DELPHI_utils_V4_dynamic import get_bounds_params_from_pastparams
from utils.DELPHI_utils_V3 import runProcessData
from DELPHI_params_V4 import (
    fitting_start_date,
    default_parameter_list,
    dict_default_reinit_parameters,
    dict_default_reinit_lower_bounds,
    dict_default_reinit_upper_bounds,
    default_upper_bound,
    default_lower_bound,
    percentage_drift_upper_bound,
    percentage_drift_lower_bound,
    percentage_drift_upper_bound_annealing,
    percentage_drift_lower_bound_annealing,
    default_upper_bound_annealing,
    default_lower_bound_annealing,
    default_lower_bound_t_jump,
    default_upper_bound_t_jump,
    default_lower_bound_std_normal,
    default_upper_bound_std_normal,
    default_bounds_params,
    validcases_threshold,
    IncubeD,
    RecoverID,
    RecoverHD,
    DetectD,
    VentilatedD,
    default_maxT,
    p_v,
    p_d,
    p_h,
    max_iter,
    VACCINE_MODEL
)
from DELPHI_model_V4_with_policies_additional_states import run_model_V4_with_policies

## Initializing Global Variables ##########################################################################
with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]
time_beginning = time.time()

yesterday_logs_filename = "".join(
    (str(datetime.now().date() - timedelta(days=1)) + f"_{datetime.now().hour}H{datetime.now().minute}M").split("-")
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--run_config', '-rc', type=str, required=True,
    help="specify relative path for the run config YAML file"
)
parser.add_argument(
    '--type', '-t', type=str, required=False,
    help="running for GLOBAL: global, or skip this, US counties : US , ex US regions: ExUS "
)
parser.add_argument(
    '--input_date', '-d', type=str, required=False,
    help="only when you want to run the US and ExUS you should have this as input for example: 112920"
)
parser.add_argument(
    '--global_JNJ', '-g', type=str, required=False,
    help="send true if you want to run 'Australia', 'China', 'Canada', 'Taiwan', 'USA' for JNJ"
)
parser.add_argument(
    '--run7days_before_JNJ', '-weekago', type=str, required=False,
    help="to run the model using the parameters of seven days ago: true as an input. false or skip means otherwise."
)

arguments = parser.parse_args()
with open(arguments.run_config, "r") as ymlfile:
    RUN_CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)

USER_RUNNING = RUN_CONFIG["arguments"]["user"]
if USER_RUNNING == "ali":
    USER = os.getenv('USER')
    USER_RUNNING = "ali" if USER == 'ali' else 'server'
INPUT_DATE = "112920"
if arguments.type is not None:
    TYPE_RUNNING = arguments.type
    if arguments.input_date is not None:
        INPUT_DATE = arguments.input_date
    else:
        INPUT_DATE = str(datetime.now().strftime("%m%d%y"))
else:
    TYPE_RUNNING = "global"
if arguments.global_JNJ is not None:
    GLOBAL_JJ = arguments.global_JNJ
else:
    GLOBAL_JJ = 'false'

if arguments.run7days_before_JNJ is not None:
    run7days_before = arguments.run7days_before_JNJ
else:
    run7days_before = 'false'

logger_filename = (
        CONFIG_FILEPATHS["logs"][USER_RUNNING] +
        f"model_fitting/delphi_model_V4_{yesterday_logs_filename}.log"
)
logging.basicConfig(
    filename=logger_filename,
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%m-%d-%Y %I:%M:%S %p",
)

# OPTIMIZER = RUN_CONFIG["arguments"]["optimizer"]
GET_CONFIDENCE_INTERVALS = bool(int(RUN_CONFIG["arguments"]["confidence_intervals"]))
SAVE_TO_WEBSITE = bool(int(RUN_CONFIG["arguments"]["website"]))
SAVE_SINCE100_CASES = bool(int(RUN_CONFIG["arguments"]["since100case"]))
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PATH_TO_DATA_SANDBOX = CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING]
PATH_TO_WEBSITE_PREDICTED = CONFIG_FILEPATHS["website"][USER_RUNNING]
past_prediction_date = "".join(str(datetime.now().date() - timedelta(days=14)).split("-"))
replace_already_existing_par = run7days_before == 'true'
#############################################################################################################

def check_cumulative_cases(input_table):
    correct = True
    count = input_table['day_since100'].iloc[0]
    for ind, row in input_table.iterrows():
        if count != row['day_since100']:
            correct = False
            break
        else:
            count += 1
    return correct


def solve_and_predict_area(
        tuple_area_state_: tuple,
        yesterday_: str,
        past_parameters_: pd.DataFrame,
        popcountries: pd.DataFrame,
        current_parameters_:pd.DataFrame,
        TYPE_RUNNING: str,
        startT: str = None, # added to change optimmization start date
):
    """
    Parallelizable version of the fitting & solving process for DELPHI V4, this function is called with multiprocessing
    :param tuple_area_: tuple corresponding to (continent, country, province)
    :param yesterday_: string corresponding to the date from which the model will read the previous parameters. The
    format has to be 'YYYYMMDD'
    :param past_parameters_: Parameters from yesterday_ used as a starting point for the fitting process
    :param popcountries: DataFrame containing population information for all countries and provinces
    :startT: string for the date from when the pandemic will be modelled (format should be 'YYYY-MM-DD')
    :return: either None if can't optimize (either less than 100 cases or less than 7 days with 100 cases) or a tuple
    with 3 dataframes related to that tuple_area_ (parameters df, predictions since yesterday_+1, predictions since
    first day with 100 cases) and a scipy.optimize object (OptimizeResult) that contains the predictions for all
    16 states of the model (and some other information that isn't used)
    """
    time_entering = time.time()
    continent, country, province, initial_state = tuple_area_state_
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    file_name = get_single_case(TYPE_RUNNING,PATH_TO_FOLDER_DANGER_MAP,PATH_TO_DATA_SANDBOX,country_sub,province_sub)
    if os.path.exists(file_name):
        if current_parameters_ is not None:
            current_parameter = current_parameters_[
                (current_parameters_.Country == country) &
                (current_parameters_.Province == province)
                ].reset_index(drop=True)
            if len(current_parameter) > 0 and replace_already_existing_par == False:
                print(
                    f"Parameters already exist on day after {yesterday_} " +
                    f"Continent={continent}, Country={country} and Province={province}"
                )
                logging.info(
                    f"Parameters already exist on day after {yesterday_} " +
                    f"Continent={continent}, Country={country} and Province={province}"
                )
                return None
        print(f"starting to predict for {continent}, {country}, {province}")
        logging.info(f"starting to predict for {continent}, {country}, {province}")
        totalcases = pd.read_csv(file_name)
        if totalcases.day_since100.max() < 0:
            print(
                f"Not enough cases (less than 100) for Continent={continent}, Country={country} and Province={province}"
            )
            logging.info(f"Not enough cases (less than 100) for Continent={continent}, Country={country} and Province={province}")
            return None

        if past_parameters_ is not None:
            parameter_list_total = past_parameters_[
                (past_parameters_.Country == country)
                & (past_parameters_.Province == province)
            ].reset_index(drop=True)
            if len(parameter_list_total) > 0:
                parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
                parameter_list = parameter_list_line[5:]
                bounds_params = get_bounds_params_from_pastparams(
                    optimizer=OPTIMIZER,
                    parameter_list=parameter_list,
                    dict_default_reinit_parameters=dict_default_reinit_parameters,
                    percentage_drift_lower_bound=percentage_drift_lower_bound,
                    default_lower_bound=default_lower_bound,
                    dict_default_reinit_lower_bounds=dict_default_reinit_lower_bounds,
                    percentage_drift_upper_bound=percentage_drift_upper_bound,
                    default_upper_bound=default_upper_bound,
                    dict_default_reinit_upper_bounds=dict_default_reinit_upper_bounds,
                    percentage_drift_lower_bound_annealing=percentage_drift_lower_bound_annealing,
                    default_lower_bound_annealing=default_lower_bound_annealing,
                    percentage_drift_upper_bound_annealing=percentage_drift_upper_bound_annealing,
                    default_upper_bound_annealing=default_upper_bound_annealing,
                    default_lower_bound_t_jump=default_lower_bound_t_jump,
                    default_upper_bound_t_jump=default_upper_bound_t_jump,
                    default_lower_bound_std_normal=default_lower_bound_std_normal,
                    default_upper_bound_std_normal=default_upper_bound_std_normal,
                )
                start_date = pd.to_datetime(parameter_list_line[3])
                bounds_params = tuple(bounds_params)
            else:
                # Otherwise use established lower/upper bounds
                parameter_list = default_parameter_list
                bounds_params = default_bounds_params
                start_date = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].iloc[-1])
        else:
            # Otherwise use established lower/upper bounds
            parameter_list = default_parameter_list
            bounds_params = default_bounds_params
            start_date = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].iloc[-1])

        if startT is not None:
            start_date = max(pd.to_datetime(startT), start_date)
            validcases = totalcases[
                (totalcases.date >= str(start_date))
                & (totalcases.date <= str((pd.to_datetime(yesterday_) + timedelta(days=1)).date()))
            ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
        else:
            validcases = totalcases[
                (totalcases.day_since100 >= 0)
                & (totalcases.date <= str((pd.to_datetime(yesterday_) + timedelta(days=1)).date()))
            ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
        # Now we start the modeling part:
        if len(validcases) <= validcases_threshold:
            print(
                f"Not enough historical data (less than a week)"
                + f"for Continent={continent}, Country={country} and Province={province}"
            )
            logging.info(f"Not enough historical data (less than a week)"
                         + f"for Continent={continent}, Country={country} and Province={province}"
                         )
            return None
        else:
            PopulationT = popcountries[
                (popcountries.Country == country) & (popcountries.Province == province)
            ].pop2016.iloc[-1]
            N = PopulationT
            PopulationI = validcases.loc[0, "case_cnt"]
            PopulationD = validcases.loc[0, "death_cnt"]
            if initial_state is not None:
                R_0 = initial_state[9]
            else:
                R_0 = validcases.loc[0, "death_cnt"] * 5 if validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]> validcases.loc[0, "death_cnt"] * 5 else 0
                bounds_params_list = list(bounds_params)
                bounds_params_list[-1] = (0.999,1)
                bounds_params = tuple(bounds_params_list)
            cases_t_14days = totalcases[totalcases.date >= str(start_date- pd.Timedelta(14, 'D'))]['case_cnt'].values[0]
            deaths_t_9days = totalcases[totalcases.date >= str(start_date - pd.Timedelta(9, 'D'))]['death_cnt'].values[0]
            R_upperbound = validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]
            R_heuristic = cases_t_14days - deaths_t_9days
            if int(R_0*p_d) >= R_upperbound and R_heuristic >= R_upperbound:
                logging.error(f"Initial conditions for PopulationR too high for {country}-{province}, on {startT}")
            """
            Fixed Parameters based on meta-analysis:
            p_h: Hospitalization Percentage
            RecoverHD: Average Days until Recovery
            VentilationD: Number of Days on Ventilation for Ventilated Patients
            maxT: Maximum # of Days Modeled
            p_d: Percentage of True Cases Detected
            p_v: Percentage of Hospitalized Patients Ventilated,
            balance: Regularization coefficient between cases and deaths
            """
            maxT = (default_maxT - start_date).days + 1
            t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
            balance, cases_data_fit, deaths_data_fit = create_fitting_data_from_validcases(validcases)
            GLOBAL_PARAMS_FIXED = (N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v)

            def model_covid(
                t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3
            ) -> list:
                """
                SEIR based model with 16 distinct states, taking into account undetected, deaths, hospitalized and
                recovered, and using an ArcTan government response curve, corrected with a Gaussian jump in case of
                a resurgence in cases
                :param t: time step
                :param x: set of all the states in the model (here, 16 of them)
                :param alpha: Infection rate
                :param days: Median day of action (used in the arctan governmental response)
                :param r_s: Median rate of action (used in the arctan governmental response)
                :param r_dth: Rate of death
                :param p_dth: Initial mortality percentage
                :param r_dthdecay: Rate of decay of mortality percentage
                :param k1: Internal parameter 1 (used for initial conditions)
                :param k2: Internal parameter 2 (used for initial conditions)
                :param jump: Amplitude of the Gaussian jump modeling the resurgence in cases
                :param t_jump: Time where the Gaussian jump will reach its maximum value
                :param std_normal: Standard Deviation of the Gaussian jump (~ time span of the resurgence in cases)
                :param k3: Internal parameter 2 (used for initial conditions)
                :return: predictions for all 16 states, which are the following
                [0 S, 1 E, 2 I, 3 UR, 4 DHR, 5 DQR, 6 UD, 7 DHD, 8 DQD, 9 R, 10 D, 11 TH, 12 DVR,13 DVD, 14 DD, 15 DT]
                """
                r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
                r_d = np.log(2) / DetectD  # Rate of detection
                r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
                r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
                r_rv = np.log(2) / VentilatedD  # Rate of recovery under ventilation
                gamma_t = (
                    (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1
                    + jump * np.exp(-(t - t_jump) ** 2 / (2 * std_normal ** 2 + 1e-5))
                )
                p_dth_mod = (2 / np.pi) * (p_dth - 0.001) * (np.arctan(-t / 20 * r_dthdecay) + np.pi / 2) + 0.001
                assert (
                    len(x) == 16
                ), f"Too many input variables, got {len(x)}, expected 16"
                S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT = x
                # Equations on main variables
                dSdt = -alpha * gamma_t * S * I / N
                dEdt = alpha * gamma_t * S * I / N - r_i * E
                dIdt = r_i * E - r_d * I
                dARdt = r_d * (1 - p_dth_mod) * (1 - p_d) * I - r_ri * AR
                dDHRdt = r_d * (1 - p_dth_mod) * p_d * p_h * I - r_rh * DHR
                dDQRdt = r_d * (1 - p_dth_mod) * p_d * (1 - p_h) * I - r_ri * DQR
                dADdt = r_d * p_dth_mod * (1 - p_d) * I - r_dth * AD
                dDHDdt = r_d * p_dth_mod * p_d * p_h * I - r_dth * DHD
                dDQDdt = r_d * p_dth_mod * p_d * (1 - p_h) * I - r_dth * DQD
                dRdt = r_ri * (AR + DQR) + r_rh * DHR
                dDdt = r_dth * (AD + DQD + DHD)
                # Helper states (usually important for some kind of output)
                dTHdt = r_d * p_d * p_h * I
                dDVRdt = r_d * (1 - p_dth_mod) * p_d * p_h * p_v * I - r_rv * DVR
                dDVDdt = r_d * p_dth_mod * p_d * p_h * p_v * I - r_dth * DVD
                dDDdt = r_dth * (DHD + DQD)
                dDTdt = r_d * p_d * I
                return [
                    dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt,
                    dDQDdt, dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt,
                ]

            def residuals_totalcases(params) -> float:
                """
                Function that makes sure the parameters are in the right range during the fitting process and computes
                the loss function depending on the optimizer that has been chosen for this run as a global variable
                :param params: currently fitted values of the parameters during the fitting process
                :return: the value of the loss function as a float that is optimized against (in our case, minimized)
                """
                # Variables Initialization for the ODE system
                alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = params
                # Force params values to stay in a certain range during the optimization process with re-initializations
                params = (
                    max(alpha, dict_default_reinit_parameters["alpha"]),
                    days,
                    max(r_s, dict_default_reinit_parameters["r_s"]),
                    max(min(r_dth, 1), dict_default_reinit_parameters["r_dth"]),
                    max(min(p_dth, 1), dict_default_reinit_parameters["p_dth"]),
                    max(r_dthdecay, dict_default_reinit_parameters["r_dthdecay"]),
                    max(k1, dict_default_reinit_parameters["k1"]),
                    max(k2, dict_default_reinit_parameters["k2"]),
                    max(jump, dict_default_reinit_parameters["jump"]),
                    max(t_jump, dict_default_reinit_parameters["t_jump"]),
                    max(std_normal, dict_default_reinit_parameters["std_normal"]),
                    max(k3, dict_default_reinit_lower_bounds["k3"]),
                )

                x_0_cases = get_initial_conditions(
                    params_fitted=params, global_params_fixed=GLOBAL_PARAMS_FIXED
                )
                x_sol_total = solve_ivp(
                    fun=model_covid,
                    y0=x_0_cases,
                    t_span=[t_cases[0], t_cases[-1]],
                    t_eval=t_cases,
                    args=tuple(params),
                )
                x_sol = x_sol_total.y
                weights = list(range(1, len(cases_data_fit) + 1))
                # weights = [(x/len(cases_data_fit))**2 for x in weights]
                if x_sol_total.status == 0:
                    residuals_value = get_residuals_value(
                        optimizer=OPTIMIZER,
                        balance=balance,
                        x_sol=x_sol,
                        cases_data_fit=cases_data_fit,
                        deaths_data_fit=deaths_data_fit,
                        weights=weights
                    )
                else:
                    residuals_value = 1e16
                return residuals_value

            if OPTIMIZER in ["tnc", "trust-constr"]:
                output = minimize(
                    residuals_totalcases,
                    parameter_list,
                    method=OPTIMIZER,
                    bounds=bounds_params,
                    options={"maxiter": max_iter},
                )
            elif OPTIMIZER == "annealing":
                output = dual_annealing(
                    residuals_totalcases, x0=parameter_list, bounds=bounds_params
                )
            else:
                logging.error("Optimizer not in 'tnc', 'trust-constr' or 'annealing' so not supported")
                raise ValueError("Optimizer not in 'tnc', 'trust-constr' or 'annealing' so not supported")

            if (OPTIMIZER in ["tnc", "trust-constr"]) or (OPTIMIZER == "annealing" and output.success):
                best_params = output.x
                t_predictions = [i for i in range(maxT)]
    
                def solve_best_params_and_predict(optimal_params):
                    # Variables Initialization for the ODE system
                    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = optimal_params
                    optimal_params = [
                        max(alpha, dict_default_reinit_parameters["alpha"]),
                        days,
                        max(r_s, dict_default_reinit_parameters["r_s"]),
                        max(min(r_dth, 1), dict_default_reinit_parameters["r_dth"]),
                        max(min(p_dth, 1), dict_default_reinit_parameters["p_dth"]),
                        max(r_dthdecay, dict_default_reinit_parameters["r_dthdecay"]),
                        max(k1, dict_default_reinit_parameters["k1"]),
                        max(k2, dict_default_reinit_parameters["k2"]),
                        max(jump, dict_default_reinit_parameters["jump"]),
                        max(t_jump, dict_default_reinit_parameters["t_jump"]),
                        max(std_normal, dict_default_reinit_parameters["std_normal"]),
                        max(k3, dict_default_reinit_lower_bounds["k3"]),
                    ]
                    x_0_cases = get_initial_conditions(
                        params_fitted=optimal_params,
                        global_params_fixed=GLOBAL_PARAMS_FIXED,
                    )
                    x_sol_best = solve_ivp(
                        fun=model_covid,
                        y0=x_0_cases,
                        t_span=[t_predictions[0], t_predictions[-1]],
                        t_eval=t_predictions,
                        args=tuple(optimal_params),
                    ).y
                    return x_sol_best

                x_sol_final = solve_best_params_and_predict(best_params)
                data_creator = DELPHIDataCreator(
                    x_sol_final=x_sol_final,
                    date_day_since100=start_date,
                    best_params=best_params,
                    continent=continent,
                    country=country,
                    province=province,
                    testing_data_included=False,
                )
                mape_data = get_mape_data_fitting(
                    cases_data_fit=cases_data_fit, deaths_data_fit=deaths_data_fit, x_sol_final=x_sol_final
                )
                
                logging.info(f"In-Sample MAPE Last 15 Days {country, province}: {round(mape_data, 3)} %")
                logging.debug(f"Best fitted parameters for {country, province}: {best_params}")
                df_parameters_area = data_creator.create_dataset_parameters(mape_data)
                # Creating the datasets for predictions of this area
                if GET_CONFIDENCE_INTERVALS:
                   df_predictions_since_today_area, df_predictions_since_100_area = (
                       data_creator.create_datasets_with_confidence_intervals(
                           cases_data_fit, deaths_data_fit,
                           past_prediction_file=PATH_TO_FOLDER_DANGER_MAP + f"predicted/Global_V2_{past_prediction_date}.csv",
                           past_prediction_date=str(pd.to_datetime(past_prediction_date).date()))
                   )
                else:
                    df_predictions_since_today_area, df_predictions_since_100_area = data_creator.create_datasets_predictions()
                inf = f"Finished predicting for Continent={continent}, Country={country} and Province={province} in " +\
                      f"{round(time.time() - time_entering, 2)} seconds\n" + \
                      "--------------------------------------------------------------------------------------------"
                print(inf)
                logging.info(inf)
                return (
                    df_parameters_area,
                    df_predictions_since_today_area,
                    df_predictions_since_100_area,
                    output,
                )
            else:
                return None
    else:  # file for that tuple (continent, country, province) doesn't exist in processed files
        print(
            f"Skipping Continent={continent}, Country={country} and Province={province} as no processed file available"
        )
        logging.info(f"Skipping Continent={continent}, Country={country} and Province={province} as no processed file available")
        return None

def run_model_eachday(current_time,OPTIMIZER, popcountries, df_initial_states):
    str_print = f"running for {current_time} optimizer {OPTIMIZER} replace_already_existing_par {replace_already_existing_par}"
    print(str_print)
    logging.info(str_print)
    current_parameters = get_past_parameters(PATH_TO_FOLDER_DANGER_MAP,PATH_TO_DATA_SANDBOX,current_time,OPTIMIZER, False,TYPE_RUNNING)
    yesterday_date = current_time - timedelta(days=1)
    yesterday = "".join(str(current_time.date() - timedelta(days=1)).split("-"))
    OPTIMIZER_tnc = "tnc"
    if "global" in TYPE_RUNNING or OPTIMIZER == "tnc":
        past_parameters = get_past_parameters(PATH_TO_FOLDER_DANGER_MAP,PATH_TO_DATA_SANDBOX,yesterday_date,OPTIMIZER_tnc, False,TYPE_RUNNING)
    else:
        past_parameters_annealing = get_past_parameters(PATH_TO_FOLDER_DANGER_MAP,PATH_TO_DATA_SANDBOX,yesterday_date,OPTIMIZER, False,TYPE_RUNNING)
        if past_parameters_annealing is None:
            past_parameters = get_past_parameters(PATH_TO_FOLDER_DANGER_MAP,PATH_TO_DATA_SANDBOX,yesterday_date,OPTIMIZER_tnc, False,TYPE_RUNNING)
        else:
            past_parameters = past_parameters_annealing

    ### Fitting the Model ###
    # Initalizing lists of the different dataframes that will be concatenated in the end
    list_df_global_predictions_since_today = []
    list_df_global_predictions_since_100_cases = []
    list_df_global_parameters = []
    obj_value = 0
    solve_and_predict_area_partial = partial(
        solve_and_predict_area,
        yesterday_=yesterday,
        past_parameters_=past_parameters,
        popcountries=popcountries,
        current_parameters_ = current_parameters,
        TYPE_RUNNING = TYPE_RUNNING,
        startT=fitting_start_date
    )
    n_cpu = 90 # psutil.cpu_count(logical = False)
    print(f"Number of CPUs found and used in this run: {n_cpu}")
    logging.info(f"Number of CPUs found and used in this run: {n_cpu}")
    list_tuples = [(
        r.continent,
        r.country,
        r.province,
        r.values[:16] if not pd.isna(r.S) else None
    ) for _, r in df_initial_states.iterrows()]
    print(f"Number of areas to be fitted in this run: {len(list_tuples)}")
    logging.info(f"Number of areas to be fitted in this run: {len(list_tuples)}")
    with mp.Pool(n_cpu) as pool:
        for result_area in tqdm(
                pool.map_async(solve_and_predict_area_partial, list_tuples).get(),
                total=len(list_tuples),
        ):
            if result_area is not None:
                (
                    df_parameters_area,
                    df_predictions_since_today_area,
                    df_predictions_since_100_area,
                    output,
                ) = result_area
                obj_value = obj_value + output.fun
                # Then we add it to the list of df to be concatenated to update the tracking df
                list_df_global_parameters.append(df_parameters_area)
                list_df_global_predictions_since_today.append(df_predictions_since_today_area)
                list_df_global_predictions_since_100_cases.append(df_predictions_since_100_area)
            else:
                continue
        print("Finished the Multiprocessing for all areas")
        logging.info("Finished the Multiprocessing for all areas")
        pool.close()
        pool.join()

    # Appending parameters, aggregations per country, per continent, and for the world
    # for predictions today & since 100
    if len(list_df_global_parameters) > 0:
        df_global_parameters = save_replace_param_jnj(list_df_global_parameters,current_parameters,replace_already_existing_par)
        df_global_predictions_since_today = pd.concat(list_df_global_predictions_since_today)
        df_global_predictions_since_today = DELPHIAggregations.append_all_aggregations(
            df_global_predictions_since_today
        )
        df_global_predictions_since_100_cases = pd.concat(list_df_global_predictions_since_100_cases)
        if GET_CONFIDENCE_INTERVALS:
            df_global_predictions_since_today, df_global_predictions_since_100_cases = DELPHIAggregations.append_all_aggregations_cf(
                df_global_predictions_since_100_cases,
                past_prediction_file=PATH_TO_FOLDER_DANGER_MAP + f"predicted/Global_V2_{past_prediction_date}.csv",
                past_prediction_date=str(pd.to_datetime(past_prediction_date).date())
            )
        else:
            df_global_predictions_since_100_cases = DELPHIAggregations.append_all_aggregations(
                df_global_predictions_since_100_cases
            )

        logger = logging.getLogger("V4Logger")
        delphi_data_saver = DELPHIDataSaver(
            path_to_folder_danger_map=PATH_TO_FOLDER_DANGER_MAP,
            path_to_website_predicted=PATH_TO_WEBSITE_PREDICTED,
            df_global_parameters=df_global_parameters,
            df_global_predictions_since_today=df_global_predictions_since_today,
            df_global_predictions_since_100_cases=df_global_predictions_since_100_cases,
            logger=logger
        )
        delphi_data_saver.save_all_datasets(optimizer=OPTIMIZER, save_since_100_cases=SAVE_SINCE100_CASES,
                                            website=SAVE_TO_WEBSITE,current_time = current_time, TYPE_RUNNING= TYPE_RUNNING,
                                            PATH_TO_DATA_SANDBOX = PATH_TO_DATA_SANDBOX)
        inf =f"Exported all 3 datasets to website & danger_map repositories, " + \
            f"total runtime was {round((time.time() - time_beginning)/60, 2)} minutes"
        print(inf)
        logging.info(inf)
    else:
        inf = f"Nothing changed for, " + \
            f"total runtime was {round((time.time() - time_beginning)/60, 2)} minutes"
        print(inf)
        logging.info(inf)


if __name__ == "__main__":
    assert USER_RUNNING in CONFIG_FILEPATHS["delphi_repo"].keys(), f"User {USER_RUNNING} not referenced in config.yml"
    if not os.path.exists(CONFIG_FILEPATHS["logs"][USER_RUNNING] + "model_fitting/"):
        os.mkdir(CONFIG_FILEPATHS["logs"][USER_RUNNING] + "model_fitting/")
    current_time_str = str(datetime.now().strftime("%Y%m%d"))
    OPTIMIZER = "tnc"
    inf = f"The user is {USER_RUNNING}, the chosen optimizer for this run is variable type {TYPE_RUNNING} and " + \
          f"generation of Confidence Intervals' flag is {GET_CONFIDENCE_INTERVALS}"
    print(inf)
    logging.info(inf)
    if "global" in TYPE_RUNNING:
        popcountries = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv"
        )
        prev_param_file = PATH_TO_DATA_SANDBOX + f"predicted/raw_predictions/Predicted_model_state_V3_{fitting_start_date}.csv"
        training_start_date = datetime.now()
        training_end_date = datetime.now()
    else:
        popcountries = pd.read_csv(
            PATH_TO_DATA_SANDBOX + f"processed/Population_Global.csv"
        )
        last_date_c = runProcessData(INPUT_DATE,logging, TYPE_RUNNING,GLOBAL_JJ)
        training_start_date = last_date_c - timedelta(days=5) # get_start_date(last_date_c,PATH_TO_FOLDER_DANGER_MAP,PATH_TO_DATA_SANDBOX,OPTIMIZER,TYPE_RUNNING)
        training_end_date = datetime.now() - timedelta(days=8) if run7days_before == 'true' else last_date_c
        prev_param_file = PATH_TO_DATA_SANDBOX + f"predicted/raw_predictions/Predicted_model_provinces_V3_{fitting_start_date}.csv"

    print(f"Start date: {training_start_date}, End date: {training_end_date}")
    logging.info(f"Start date: {training_start_date}, End date: {training_end_date}")

    if not os.path.exists(prev_param_file):
        inf = f"Initial model state {prev_param_file} file not found, can not train from {fitting_start_date}. Use model_V3 to train on entire data."
        logging.error(inf)
        print(inf)
        raise FileNotFoundError
    df_initial_states = pd.read_csv(prev_param_file)
    if TYPE_RUNNING == "ExUS":
        if GLOBAL_JJ == "true":
            df_initial_states = df_initial_states[df_initial_states.province == 'None']
        else:
            df_initial_states = df_initial_states[df_initial_states.country != 'US']
    elif TYPE_RUNNING == "US":
        df_initial_states = df_initial_states[df_initial_states.country == 'US']
        df_initial_states =  df_initial_states[~df_initial_states['province'].str.contains("NE")]
    elif TYPE_RUNNING == VACCINE_MODEL:
        df_initial_states = df_initial_states[df_initial_states['country'].isin(["US"])]
    n_days_to_train = (training_end_date - training_start_date).days
    # prov = df_initial_states.province.values
    # stat =  [x.split('_')[0] for x in prov]
    # allStats = np.unique(stat)
    # print(allStats)
    
    # for s in allStats:
    #     if s == 'NE':
    #         continue;
    #     print(f"state {s}")
    #     df_initial_states_p = df_initial_states[df_initial_states['province'].str.contains(s)]

    for n_days_after in range(min(1,n_days_to_train), max(n_days_to_train+1 ,1)):
        if n_days_after == n_days_to_train:
           OPTIMIZER = "annealing"
        current_time = training_start_date +  timedelta(days=n_days_after)
        run_model_eachday(current_time,OPTIMIZER, popcountries, df_initial_states)

    upload_to_s3 = True
    run_model_V4_with_policies(current_time_str,PATH_TO_FOLDER_DANGER_MAP, PATH_TO_DATA_SANDBOX, training_end_date,upload_to_s3,TYPE_RUNNING,OPTIMIZER,popcountries,df_initial_states,logging,GLOBAL_JJ,run7days_before)
    
