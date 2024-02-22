from file_utils import read_file_in_dir
from mains.healthy_plates import healthy_plates_main
from mains.lesioned_plates import lesioned_plates_main
from mains.healthy_faces_v_obj import healthy_faces_v_obj_main
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
    elif args['name'] == 'healthy_faces_v_obj':
        healthy_faces_v_obj_main(args, param, values)
    else:
        raise NotImplementedError()
