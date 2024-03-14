from file_utils import read_file_in_dir
from mains.healthy_plates import healthy_plates_main
from mains.lesioned_lfw_people import lesioned_lfw_people_main
from mains.lesioned_plates import lesioned_plates_main
from mains.healthy_ops import healthy_ops_main
from mains.lesioned_ops import lesioned_ops_main
from mains.healthy_lfw_people import healthy_lfw_people_main
from mains.lesioned_retrain_plates import lesioned_retrain_plates_main
from mains.healthy_lfw_pairs import healthy_lfw_pairs_main
from mains.lesioned_lfw_pairs import lesioned_lfw_pairs_main
import sys


if __name__ == '__main__':
    """
        Hyperparameter Tuning: 
        python3 main.py <config_name> [param1] [comma_sep_values]
        E.g. python3 main.py healthy_plates lr 0.0001,0.00001
        """
    config_name, param, values = sys.argv[1:]
    args = read_file_in_dir('configs/', config_name + '.json')

    if args['name'] == 'healthy_plates':
        healthy_plates_main(args, param, values)
    elif args['name'] == 'lesioned_plates':
        lesioned_plates_main(args, param, values)
    elif args['name'].startswith('lesioned_retrain_plates'):
        lesioned_retrain_plates_main(args, param, values)
    elif args['name'] == 'healthy_ops':
        healthy_ops_main(args, param, values)
    elif args['name'] == 'healthy_lfw_people':
        healthy_lfw_people_main(args, param, values)
    elif args['name'] == 'lesioned_lfw_people':
        lesioned_lfw_people_main(args, param, values)
    elif args['name'] == 'healthy_lfw_pairs':
        healthy_lfw_pairs_main(args, param, values)
    elif args['name'] == 'lesioned_ops':
        lesioned_ops_main(args, param, values)
    elif args['name'].startswith('lesioned_lfw_pairs'):
        lesioned_lfw_pairs_main(args, param, values)
    else:
        raise NotImplementedError()
