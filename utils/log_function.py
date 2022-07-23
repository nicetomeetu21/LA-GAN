import os

def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = 'description: '+ parser.description + '\n'
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    file_name = os.path.join(opt.result_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def print_network(net, opt):
    file_name = os.path.join(opt.result_dir, 'network.txt')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    message = str(net)
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
        opt_file.write('Total number of parameters: %d' % num_params)
        opt_file.write('\n')