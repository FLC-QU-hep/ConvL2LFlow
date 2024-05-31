import os
import argparse

import yaml

from src import data_util

script_template = """\
#!/bin/bash
cd {repo_path:s}
# source ~/.bashrc
conda activate convL2LFlows
for i in {flow:s}
do
    python src/train.py -l $i {config:s}
done
"""

no_loop_template = """\
#!/bin/bash
cd {repo_path:s}
# source ~/.bashrc
conda activate convL2LFlows
python src/train.py {config:s}
"""

def get_args():
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('param_file', help='where to find the parameters')
    parser.add_argument('-f', '--flow', default='',
        help='which flows to train')
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['result_path'] = data_util.setup_result_path(params['run_name'], args.param_file)

    os.mkdir(os.path.join(params['result_path'], 'checkpoints'))
    os.mkdir(os.path.join(params['result_path'], 'data'))
    os.mkdir(os.path.join(params['result_path'], 'flows'))
    os.mkdir(os.path.join(params['result_path'], 'plots'))
    os.mkdir(os.path.join(params['result_path'], 'log'))

    conf_file = os.path.join(params['result_path'], 'conf.yaml')
    run_file = os.path.join(params['result_path'], 'run.sh')
    repo_path = os.path.dirname(os.path.abspath(__file__))

    if args.flow:
        flows = args.flow.replace('-', '..').replace(':', '..')
        if '..' in flows:
            flows = '{' + flows + '}'
        script = script_template.format(
            flow=flows,
            config=os.path.relpath(conf_file, repo_path),
            repo_path=repo_path
        )
    else:
        script = no_loop_template.format(
            config=os.path.relpath(conf_file, repo_path),
            repo_path=repo_path
        )
    with open(run_file, 'w') as file:
        file.write(script)

if __name__=='__main__':
    main()
