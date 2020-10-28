# PointNovo is publicly available for non-commercial uses.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import json
import logging
from itertools import combinations

logger = logging.getLogger(__name__)
# ==============================================================================
# FLAGS (options) for this app
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--beam_size", type=int, default="5")
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--search_denovo", dest="search_denovo", action="store_true")
parser.add_argument("--search_db", dest="search_db", action="store_true")
parser.add_argument("--valid", dest="valid", action="store_true")
parser.add_argument("--test", dest="test", action="store_true")
parser.add_argument("--serialize_model", dest="serialize_model", action="store_true")
parser.add_argument("--onnx", dest="onnx", action="store_true")

parser.set_defaults(train=False)
parser.set_defaults(search_denovo=False)
parser.set_defaults(search_db=False)
parser.set_defaults(valid=False)
parser.set_defaults(test=False)
parser.set_defaults(serialize_model=False)
parser.set_defaults(onnx=False)

args = parser.parse_args()
python_obj_dict = {"FLAGS": args}


class JasonConfig(object):
    def __init__(self, config_filename: str):
        with open(config_filename, 'r') as f:
            self.__dict__ = json.load(f)

    def get_param_dict(self):
        return self.__dict__


params_obj = JasonConfig("params.json")
constant_params_obj = JasonConfig("constant_params.json")

# Special vocabulary symbols - we always put them at the start.
_START_VOCAB = [constant_params_obj._PAD, constant_params_obj._GO, constant_params_obj._EOS]

assert constant_params_obj.PAD_ID == 0


# The parameters that need to be changed from time to time are stored in the json format. The relatively constant params
# are hard coded in this script.


params_obj.vocab_reverse = _START_VOCAB + params_obj.vocab_reverse
logger.info("Training vocab_reverse ", params_obj.vocab_reverse)

params_obj.vocab = dict([(x, y) for (y, x) in enumerate(params_obj.vocab_reverse)])
logger.info("Training vocab ", params_obj.vocab)

params_obj.vocab_size = len(params_obj.vocab_reverse)
logger.info("Training vocab_size ", params_obj.vocab_size)

# database search parameter
## the PTMs to be included in the database search


def _fix_transform(aa: str):
    def trans(peptide: list):
        return [x if x != aa else params_obj.fix_mod_dict[x] for x in peptide]
    return trans


def fix_mod_peptide_transform(peptide: list):
    """
    apply fix modification transform on a peptide
    :param peptide:
    :return:
    """
    for aa in params_obj.fix_mod_dict.keys():
        trans = _fix_transform(aa)
        peptide = trans(peptide)
    return peptide


def _find_all_ptm(peptide, position_list):
    if len(position_list) == 0:
        return [peptide]
    position = position_list[0]
    aa = peptide[position]
    result = []
    temp = peptide[:]
    temp[position] = params_obj.var_mod_dict[aa]
    result += _find_all_ptm(temp, position_list[1:])
    return result


def var_mod_peptide_transform(peptide: list):
    """
    apply var modification transform on a peptide, the max number of var mod is max_num_mod
    :param peptide:
    :return:
    """
    position_list = [position for position, aa in enumerate(peptide) if aa in params_obj.var_mod_dict]
    position_count = len(position_list)
    num_mod = min(position_count, params_obj.max_num_mod)
    position_combination_list = []
    for x in range(1, num_mod+1):
        position_combination_list += combinations(position_list, x)
    # find all ptm peptides
    ptm_peptide_list = []
    for position_combination in position_combination_list:
        ptm_peptide_list += _find_all_ptm(peptide, position_combination)
    return ptm_peptide_list


python_obj_dict["mass_ID"] = [constant_params_obj.mass_AA[params_obj.vocab_reverse[x]] for x in range(params_obj.vocab_size)]
python_obj_dict["mass_ID_np"] = np.array(python_obj_dict["mass_ID"], dtype=np.float32)
python_obj_dict["mass_AA_min"] = constant_params_obj.mass_AA["G"]  # 57.02146
python_obj_dict["mass_AA_min_round"] = int(round(python_obj_dict["mass_AA_min"] * constant_params_obj.KNAPSACK_AA_RESOLUTION))  # 57.02146
python_obj_dict["MAX_LEN"] = 60 if args.search_denovo or args.search_db else 30
logger.info("MAX_LEN ", python_obj_dict["MAX_LEN"])
logger.info("num_ion ", constant_params_obj.num_ion)
# ==============================================================================
# GLOBAL VARIABLES for PRECISION, RESOLUTION, temp-Limits of MASS & LEN
# ==============================================================================
logger.info("weight_decay ", constant_params_obj.weight_decay)
python_obj_dict["n_position"] = int(constant_params_obj.MZ_MAX) * constant_params_obj.spectrum_reso

# ==============================================================================
# DATASETS
# ==============================================================================


python_obj_dict["denovo_output_file"] = params_obj.denovo_input_feature_file + ".deepnovo_denovo"

python_obj_dict["db_output_file"] = params_obj.search_db_input_feature_file + '.pin'

# test accuracy
python_obj_dict["predicted_file"] = python_obj_dict["denovo_output_file"]

python_obj_dict["accuracy_file"] = python_obj_dict["predicted_file"] + ".accuracy"
python_obj_dict["denovo_only_file"] = python_obj_dict["predicted_file"] + ".denovo_only"
python_obj_dict["scan2fea_file"] = python_obj_dict["predicted_file"] + ".scan2fea"
python_obj_dict["multifea_file"] = python_obj_dict["predicted_file"] + ".multifea"


class Config(object):
    def __init__(self, param: JasonConfig, constant_param:JasonConfig, obj_dict: dict):
        self.__dict__.update(param.get_param_dict())
        self.__dict__.update(constant_param.get_param_dict())
        self.__dict__.update(obj_dict)


config = Config(params_obj, constant_params_obj, python_obj_dict)

