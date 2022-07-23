def device0_config(parser):
    parser.add_argument('--A_dir', type=str,
                        default='../data/A')
    parser.add_argument('--B_dir', type=str,
                        default='../data/B')
    parser.add_argument('--C_dir', type=str,
                        default='../data/C')
    parser.add_argument('--D_dir', type=str,
                        default='../data/D')
    parser.add_argument('--result_root', type=str, default='../data/result')
    parser.add_argument('--num_workers', type=int, default=3)

def get_data_setting(parser, device_id = 0):
    if device_id == 0:
        return device0_config(parser)